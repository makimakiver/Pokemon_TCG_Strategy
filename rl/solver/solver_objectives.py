"""Solver RL objectives behind one interface: reinforce_half | ppo | cispo.

All three consume the same ``list[Rollout]`` (a Rollout = the micro-steps of one
game + its terminal reward + the scenario id it was sampled from) and the current
``prior_weight``, and return ``(loss, metrics)``. Identical reward + curriculum
across arms makes the PPO-vs-CISPO ablation (plan §3.3, P3) fair.

Key faithful details from the plan:
* **REINFORCE½** (reference arm): log-likelihood of *winning* rollouts, restricted
  to scenarios whose win-rate <= 0.5. Cheap, entropy-preserving.
* **CISPO degenerate-group fallback**: hard targets give all-win / all-loss groups
  (std ≈ 0) where the group-normalized advantage is undefined; those groups fall
  back to REINFORCE½ instead of producing garbage gradients (plan §3.3).
* **Length-normalize** each rollout's loss by its number of decisions.
* **Collapse diagnostics** are returned in ``metrics`` every update.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from rl.config import CONFIG


@dataclass
class Rollout:
    scenario_id: str
    steps: list[dict] = field(default_factory=list)   # each: {"obs","action","logp","value"}
    reward: float = 0.0                                # terminal ±1 / 0
    win: bool = False


# --- helpers ----------------------------------------------------------------
def _group_winrates(rollouts: list[Rollout]) -> dict[str, float]:
    wins, n = defaultdict(int), defaultdict(int)
    for r in rollouts:
        n[r.scenario_id] += 1
        wins[r.scenario_id] += int(r.win)
    return {k: wins[k] / max(1, n[k]) for k in n}


def _reeval(policy, step, prior_weight):
    return policy.evaluate(step["obs"], step["action"], prior_weight)


def _winrate_histogram(winrates: dict[str, float]) -> list[int]:
    """5-bin histogram of per-scenario win-rates; mass at the ends == collapse."""
    bins = [0, 0, 0, 0, 0]
    for w in winrates.values():
        bins[min(4, int(w * 5))] += 1
    return bins


# --- objectives -------------------------------------------------------------
class SolverObjective:
    name = "base"

    def compute_loss(self, policy, rollouts, prior_weight: float):
        raise NotImplementedError


class ReinforceHalf(SolverObjective):
    name = "reinforce_half"

    def compute_loss(self, policy, rollouts, prior_weight, only_groups=None):
        wr = _group_winrates(rollouts)
        loss = torch.zeros(())
        ent_sum, n_steps, used = torch.zeros(()), 0, 0
        for r in rollouts:
            if only_groups is not None and r.scenario_id not in only_groups:
                continue
            if wr[r.scenario_id] > 0.5 or not r.win:
                # learn only from wins on still-hard problems
                if not (only_groups is not None and r.win):
                    continue
            if not r.steps:
                continue
            logps, ents = [], []
            for s in r.steps:
                lp, ent, _ = _reeval(policy, s, prior_weight)
                logps.append(lp); ents.append(ent)
            traj_logp = torch.stack(logps).sum() / len(logps)   # length-normalized
            loss = loss - traj_logp
            ent_sum = ent_sum + torch.stack(ents).mean()
            n_steps += len(logps); used += 1
        if used == 0:
            return torch.zeros(()), {"used": 0}
        loss = loss / used
        ent = (ent_sum / used).item()
        loss = loss - CONFIG.entropy_coef * (ent_sum / used)
        return loss, {"used": used, "entropy": ent,
                      "winrate_hist": _winrate_histogram(wr)}


class PPO(SolverObjective):
    name = "ppo"

    def compute_loss(self, policy, rollouts, prior_weight):
        clip, vc, ec = CONFIG.ppo_clip, CONFIG.value_coef, CONFIG.entropy_coef
        pol_loss, val_loss, ent_sum, n = torch.zeros(()), torch.zeros(()), torch.zeros(()), 0
        # Monte-Carlo return to terminal reward (sparse) with discounting.
        for r in rollouts:
            T = len(r.steps)
            if T == 0:
                continue
            ret = torch.zeros(T)
            g = r.reward
            for t in reversed(range(T)):
                g = g * CONFIG.gamma
                ret[t] = g
            for t, s in enumerate(r.steps):
                lp, ent, value = _reeval(policy, s, prior_weight)
                adv = (ret[t] - value).detach()
                ratio = torch.exp(lp - s["logp"])
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
                pol_loss = pol_loss - torch.min(unclipped, clipped)
                val_loss = val_loss + (ret[t] - value) ** 2
                ent_sum = ent_sum + ent
                n += 1
        if n == 0:
            return torch.zeros(()), {"used": 0}
        loss = (pol_loss + vc * val_loss - ec * ent_sum) / n
        wr = _group_winrates(rollouts)
        return loss, {"used": n, "entropy": (ent_sum / n).item(),
                      "value_loss": (val_loss / n).item(),
                      "winrate_hist": _winrate_histogram(wr)}


class CISPO(SolverObjective):
    name = "cispo"

    def __init__(self):
        self._fallback = ReinforceHalf()

    def compute_loss(self, policy, rollouts, prior_weight):
        groups: dict[str, list[Rollout]] = defaultdict(list)
        for r in rollouts:
            groups[r.scenario_id].append(r)

        clip, eps = CONFIG.cispo_clip, CONFIG.cispo_std_eps
        loss, n = torch.zeros(()), 0
        degenerate_groups = []
        ent_sum = torch.zeros(())
        for sid, rs in groups.items():
            rewards = torch.tensor([r.reward for r in rs])
            std = rewards.std(unbiased=False)
            if std < eps:                       # all-win / all-loss -> undefined adv
                degenerate_groups.append(sid)
                continue
            adv_group = (rewards - rewards.mean()) / (std + 1e-8)
            for r, a in zip(rs, adv_group):
                for s in r.steps:
                    lp, ent, _ = _reeval(policy, s, prior_weight)
                    is_w = torch.clamp(torch.exp(lp - s["logp"]).detach(), max=clip)
                    loss = loss - is_w * a.detach() * lp / max(1, len(r.steps))
                    ent_sum = ent_sum + ent
                    n += 1
        metrics = {"degenerate_groups": len(degenerate_groups),
                   "total_groups": len(groups)}
        # Fallback REINFORCE½ on the degenerate groups (the paper's failure regime).
        if degenerate_groups:
            fb_loss, fb_m = self._fallback.compute_loss(
                policy, rollouts, prior_weight, only_groups=set(degenerate_groups))
            loss = loss + fb_loss * max(1, n)   # rescale to combine on equal footing
            metrics["fallback_used"] = fb_m.get("used", 0)
        if n == 0 and not degenerate_groups:
            return torch.zeros(()), {**metrics, "used": 0}
        denom = max(1, n)
        loss = loss / denom - CONFIG.entropy_coef * (ent_sum / denom)
        wr = _group_winrates(rollouts)
        metrics.update({"used": n, "entropy": (ent_sum / denom).item() if n else 0.0,
                        "winrate_hist": _winrate_histogram(wr)})
        return loss, metrics


class AlphaZero(SolverObjective):
    """AlphaZero-style distillation: a DENSE per-step signal that does not depend
    on wins (so it never starves like REINFORCE½/CISPO on low-variance groups).

      L = (1/N) Σ_steps [ CE(π_mcts, π_net)  +  value_coef·(v_net − z)²  −  ent_coef·H ]

    π_mcts is the MCTS visit distribution stored on each step (``step["pi"]``,
    length n_options+1, aligned to the policy logits incl. STOP). z is the game
    outcome (the rollout's terminal reward) applied to every step. The policy
    cross-entropy is summed ONLY over legal slots — masked logits are −inf and the
    target is 0 there, so restricting to legal avoids 0·(−inf)=NaN. Steps with no
    π target (non-MCTS) contribute only the value term."""
    name = "alphazero"

    def compute_loss(self, policy, rollouts, prior_weight):
        vc, ec = CONFIG.value_coef, CONFIG.entropy_coef
        pol_loss, val_loss, ent_sum, n = (torch.zeros(()), torch.zeros(()),
                                          torch.zeros(()), 0)
        for r in rollouts:
            z = float(r.reward)                       # outcome target for every step
            for s in r.steps:
                g = torch.as_tensor(s["obs"]["global"], dtype=torch.float32)
                o = torch.as_tensor(s["obs"]["options"], dtype=torch.float32)
                m = torch.as_tensor(s["obs"]["mask"], dtype=torch.float32)
                logits, value = policy.forward(g, o, m, prior_weight)
                legal = m > 0.5
                logp = torch.log_softmax(logits, dim=0)
                val_loss = val_loss + (value - z) ** 2
                pi = s.get("pi")
                if pi is not None:
                    t = torch.as_tensor(np.asarray(pi), dtype=torch.float32)
                    if t.shape[0] == logits.shape[0]:
                        pol_loss = pol_loss - (t[legal] * logp[legal]).sum()
                        p = torch.softmax(logits, dim=0)
                        ent_sum = ent_sum - (p[legal] * logp[legal]).sum()
                n += 1
        if n == 0:
            return torch.zeros(()), {"used": 0}
        loss = (pol_loss + vc * val_loss - ec * ent_sum) / n
        wr = _group_winrates(rollouts)
        return loss, {"used": n,
                      "policy_loss": float((pol_loss / n).detach()),
                      "value_loss": float((vc * val_loss / n).detach()),
                      "entropy": float((ent_sum / n).detach()),
                      "winrate_hist": _winrate_histogram(wr)}


_REGISTRY = {o.name: o for o in [ReinforceHalf(), PPO(), CISPO(), AlphaZero()]}


def get_objective(name: str | None = None) -> SolverObjective:
    name = name or CONFIG.objective
    if name not in _REGISTRY:
        raise ValueError(f"unknown objective '{name}'; choices: {list(_REGISTRY)}")
    return _REGISTRY[name]
