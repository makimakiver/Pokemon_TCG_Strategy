"""Inner net-RL loop: collect rollouts, apply a Solver objective, update (plan §5 P1).

Single-process collector (engine is one-battle-per-process; ``vec.py`` scales this
out when needed). Rollouts are grouped by scenario so the objectives can compute
per-group win-rates (REINFORCE½) and group-normalized advantages (CISPO).
Diagnostics (entropy, win-rate histogram, grad-norm, KL-to-reference) print every
update — collapse is visible, per the plan's mandatory instrumentation.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import CONFIG, RUNS_DIR, solver_deck_path
try:
    from .env import TCGEnv          # engine-only; used only as a type annotation
except Exception:                    # non-engine host: annotation is a string (PEP 563)
    TCGEnv = None
from .policy import PointerPolicy, prior_weight_at, save as save_policy
from .solver_objectives import get_objective, Rollout


def _load_deck(path) -> list[int]:
    return json.load(open(path))


def make_policy_actor(policy):
    """Default actor: sample directly from the net (no search)."""
    def actor(env, obs, prior_weight):
        return policy.act(obs, prior_weight)
    return actor


def make_mcts_actor(mcts, policy):
    """MCTS actor: pick via MCTS, but ALWAYS return an action that is legal in the
    env's CURRENT obs mask, recording log(pi_mcts[action]) as the behavior log-prob.

    MCTS roots at the engine search node and is unaware of the env's per-micro-step
    ``pending`` mask (which zeroes already-picked options during a multi-select), so
    its raw visit-argmax can be an illegal/duplicate option. We mask the MCTS visit
    distribution by ``obs["mask"]`` and renormalize; if no legal option carries MCTS
    mass (or in live turn-0 mode), we defer to the masked net policy, which always
    yields a legal action (incl. STOP). This keeps every recorded action legal so its
    re-eval log-prob is finite — an illegal action gives log_prob=-inf, which poisons
    the objective (CISPO -> NaN, REINFORCE -> +inf) and stalls multi-pick selects in a
    no-op loop."""
    def actor(env, obs, prior_weight):
        if getattr(env, "mode", None) != "search":
            return policy.act(obs, prior_weight)
        _, pi, value = mcts.search_policy(env.search_state())
        n = obs["n_options"]
        mask = np.asarray(obs["mask"], dtype=np.float64)
        pi = np.asarray(pi, dtype=np.float64)
        if n > 0 and pi.shape[0] == n:
            legal_pi = pi * mask[:n]            # restrict to options legal right now
            s = legal_pi.sum()
            if s > 0:
                a = int(legal_pi.argmax())
                return a, float(np.log(legal_pi[a] / s + 1e-12)), float(value)
        # No legal MCTS option (mid multi-pick its mass sat on a picked option, or
        # n == 0): the masked net policy always returns a legal action / STOP.
        return policy.act(obs, prior_weight)
    return actor


def collect_rollouts(policy, env: TCGEnv, scenarios, k: int, prior_weight: float,
                     actor=None):
    """k rollouts per scenario (None == a live turn-0 game). Returns list[Rollout].

    ``actor(env, obs, prior_weight) -> (action, logp, value)`` selects each move;
    defaults to sampling from ``policy`` directly. An MCTS actor records the
    MCTS visit-distribution log-prob as ``logp`` (CISPO's behavior policy)."""
    if actor is None:
        actor = make_policy_actor(policy)
    rollouts: list[Rollout] = []
    for sc in scenarios:
        sid = sc.target_id if sc is not None else "live"
        for _ in range(k):
            obs = env.reset(sc)
            steps = []
            guard = 0
            while not env.done and guard < CONFIG.max_steps:
                guard += 1
                action, logp, value = actor(env, obs, prior_weight)
                steps.append({"obs": obs, "action": action, "logp": float(logp),
                              "value": float(value)})
                obs, _, done, _ = env.step(action)
            reward = env._reward()
            rollouts.append(Rollout(scenario_id=sid, steps=steps,
                                    reward=reward, win=(reward > 0)))
    return rollouts


def train(scenarios=None, generations=None, run_name="p1", init_policy=None):
    """Train the Solver vs a fixed anchor (P1) or on a scenario batch (P2+)."""
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)

    solver_deck = _load_deck(solver_deck_path())
    import importlib
    opponent = importlib.import_module(CONFIG.opponent_module)
    opp_deck = _load_deck(solver_deck_path()) if not hasattr(opponent, "my_deck") \
        else list(opponent.my_deck)
    env = TCGEnv(solver_deck, opp_deck, opponent)

    policy = init_policy or PointerPolicy()
    ref_policy = PointerPolicy()
    ref_policy.load_state_dict(policy.state_dict())
    objective = get_objective()
    opt = torch.optim.Adam(policy.parameters(), lr=CONFIG.lr)

    if scenarios is None:
        scenarios = [None]      # live turn-0 self-play vs the anchor
    generations = generations or CONFIG.updates_per_gen

    out = Path(RUNS_DIR) / run_name
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for update in range(generations):
        pw = prior_weight_at(update)
        rollouts = collect_rollouts(policy, env, scenarios, CONFIG.k_rollouts, pw)
        loss, metrics = objective.compute_loss(policy, rollouts, pw)
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), CONFIG.grad_clip)
            opt.step()
        else:
            gnorm = torch.tensor(0.0)

        winrate = np.mean([r.win for r in rollouts]) if rollouts else 0.0
        kl = _kl_to_ref(policy, ref_policy, rollouts, pw)
        loss_val = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        rec = {"update": update, "winrate": float(winrate),
               "loss": loss_val, "grad_norm": float(gnorm),
               "prior_weight": pw, "kl_ref": kl, **metrics}
        history.append(rec)
        print(f"[{run_name}] gen {update:3d} | win {winrate:.2f} | loss {loss_val:+.3f} "
              f"| ent {metrics.get('entropy', 0):.3f} | hist {metrics.get('winrate_hist')} "
              f"| pw {pw:.2f} | gnorm {float(gnorm):.2f}")
        if update % 10 == 0 or update == generations - 1:
            save_policy(policy, out / f"solver_{update:04d}.pt")
            json.dump(history, open(out / "history.json", "w"), indent=2)

    env.close()
    save_policy(policy, out / "solver_final.pt")
    return policy, history


def _kl_to_ref(policy, ref, rollouts, pw) -> float:
    if not rollouts:
        return 0.0
    sample = rollouts[0].steps[:8]
    kls = []
    for s in sample:
        with torch.no_grad():
            g = torch.as_tensor(s["obs"]["global"]); o = torch.as_tensor(s["obs"]["options"])
            m = torch.as_tensor(s["obs"]["mask"])
            lp, _ = policy.forward(g, o, m, pw)
            lr, _ = ref.forward(g, o, m, pw)
            p = torch.softmax(lp, 0); q = torch.softmax(lr, 0)
            kls.append(float((p * (torch.log(p + 1e-9) - torch.log(q + 1e-9))).sum()))
    return float(np.mean(kls)) if kls else 0.0


if __name__ == "__main__":
    train(run_name="p1_anchor")
