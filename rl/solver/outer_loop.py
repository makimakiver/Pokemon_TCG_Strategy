"""SGS Algorithm 1 driver (plan §2 / §5 P2).

Per generation:
  1. Sample a batch B ⊆ D (the fixed target set).
  2. Split into solved / unsolved by the running win-rate vs τ.
  3. For each unsolved target, the Conjecturer emits an edit-script -> x̃, loaded
     via search_begin (legality-gated by env.reset + the Guide).
  4. The Solver plays k games on B ∪ B_synth; the engine verifies wins.
  5. R_solve(x̃) = 1·(1 - s);  R_guide(x̃) = ρ(x, x̃);  R_synth = R_solve · R_guide.
  6. Update the Solver (PPO|CISPO|REINFORCE½) on the rollouts.
  7. Update the Conjecturer (REINFORCE on R_synth).

The GPU is never contended: the parametric Conjecturer is CPU-only, so P0-P3 are
a single RL stack. Headline metric = held-out gauntlet win-rate (eval.py); the
secondary SGS metric is cumulative solve-rate on D.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from rl.config import CONFIG, RUNS_DIR, solver_deck_path
from rl.core.env import TCGEnv
from rl.solver.policy import PointerPolicy, prior_weight_at, save as save_policy, load as load_policy
from rl.solver.solver_objectives import get_objective, Rollout
from rl.core.targets import build_target_set
from rl.conjecturer import get_conjecturer
from rl.solver.guide import guide_score
from rl.solver.train_solver import collect_rollouts, _load_deck, make_mcts_actor
from rl.solver.mcts import MCTS
from rl.core.problem_set import load_problem_set, seed_scenarios
from rl.core.scenario import ScenarioSpec


def _winrate(policy, env, scenario, k, pw, actor=None):
    rs = collect_rollouts(policy, env, [scenario], k, pw, actor=actor)
    return np.mean([r.win for r in rs]), rs


def run_sgs(generations: int = 20, batch_size: int = 8, run_name: str = "p2_sgs",
            use_mcts: bool = False, problem_set=None, objective: str | None = None,
            conjecture_after: int = 2, init_ckpt: str | None = None,
            live_games: int = 0):
    random.seed(CONFIG.seed); np.random.seed(CONFIG.seed); torch.manual_seed(CONFIG.seed)
    rng = random.Random(CONFIG.seed)

    D = build_target_set()
    if not D:
        raise RuntimeError("empty target set D; check data/loser/*.json")
    print(f"[sgs] target set |D| = {len(D)}")

    # Warm-start each target's current lemma from the conjecturer's problem set.
    seeded: dict[str, ScenarioSpec] = {}
    if problem_set:
        ps = load_problem_set(problem_set)
        for s in seed_scenarios(D, ps):
            seeded[s.target_id] = s
        print(f"[sgs] seeded {len(ps)} lemmas from {problem_set}")

    solver_deck = _load_deck(solver_deck_path())
    import importlib
    opponent = importlib.import_module(CONFIG.opponent_module)
    opp_deck = list(getattr(opponent, "my_deck", solver_deck))
    env = TCGEnv(solver_deck, opp_deck, opponent)

    import os
    if init_ckpt and os.path.exists(init_ckpt):
        policy = load_policy(init_ckpt)
        print(f"[sgs] warm-started solver from {init_ckpt}")
    else:
        policy = PointerPolicy()
    objective = get_objective(objective)   # default (None) -> CONFIG.objective (reinforce_half)
    conjecturer = get_conjecturer()
    opt = torch.optim.Adam(policy.parameters(), lr=CONFIG.lr)

    actor = None
    if use_mcts:
        mcts = MCTS(policy, opponent, solver_seat=CONFIG.seat)
        actor = make_mcts_actor(mcts, policy)
        print(f"[sgs] MCTS actor ON ({CONFIG.mcts_simulations} sims/decision)")

    solve_rate = defaultdict(float)
    stale = defaultdict(int)        # target_id -> consecutive unsolved attempts (seed-first)
    seeding = bool(problem_set)
    out = Path(RUNS_DIR) / run_name
    out.mkdir(parents=True, exist_ok=True)
    history = []

    for gen in range(generations):
        pw = prior_weight_at(gen)
        batch = rng.sample(D, min(batch_size, len(D)))
        rollouts: list[Rollout] = []
        synth_updates = []

        for target in batch:
            tid = target.target_id
            solved = solve_rate[tid] >= CONFIG.tau
            base = seeded.get(tid, target)        # the SmolLM problem (or raw target if unseeded)
            scenario = base
            edits = None
            conj_idx = None
            # Generate a similar, easier variation only when stuck on the seed
            # (or always, when there is no seed to practice first).
            if not solved and (not seeding or stale[tid] >= conjecture_after):
                edited, edits, conj_idx = conjecturer.propose(base, rng)
                scenario = edited
            try:
                wr, rs = _winrate(policy, env, scenario, CONFIG.k_rollouts, pw, actor=actor)
                legal = True
            except Exception:
                wr, rs, legal = 0.0, [], False
            rollouts.extend(rs)
            solve_rate[tid] = float(wr)
            stale[tid] = 0 if wr >= CONFIG.tau else stale[tid] + 1

            if edits is not None:
                r_solve = 1.0 * (1.0 - wr)               # reward hard-but-solvable lemmas
                r_guide = guide_score(scenario, edits, legal=legal)
                r_synth = r_solve * r_guide
                synth_updates.append((conj_idx, r_synth))

        # --- Live turn-0 full games (teach the OPENING, which the problem set lacks) ---
        # MCTS is search-mode only, so the actor falls back to the net (no search) in
        # live mode. Win-gate the policy target: winning openings are reinforced
        # (pi=one-hot kept); losing openings update only the value head (pi=None), so we
        # never reinforce a losing line.
        live_wr = None
        if live_games > 0:
            live_rs = collect_rollouts(policy, env, [None], live_games, pw, actor=actor)
            for r in live_rs:
                if not r.win:
                    for s in r.steps:
                        s["pi"] = None
            rollouts.extend(live_rs)
            live_wr = float(np.mean([r.win for r in live_rs])) if live_rs else 0.0

        # --- Solver update ---
        loss, metrics = objective.compute_loss(policy, rollouts, pw)
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), CONFIG.grad_clip)
            opt.step()
        else:
            gnorm = torch.tensor(0.0)

        # --- Conjecturer update (REINFORCE on R_synth) ---
        for idx, r_synth in synth_updates:
            conjecturer.update(idx, r_synth)

        cum_solved = sum(1 for v in solve_rate.values() if v >= CONFIG.tau)
        loss_val = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        rec = {"gen": gen, "cum_solved": cum_solved, "n_targets": len(D),
               "solve_rate": cum_solved / len(D), "loss": loss_val,
               "grad_norm": float(gnorm), "prior_weight": pw,
               "live_winrate": live_wr,
               "conjecturer": conjecturer.snapshot()
               if hasattr(conjecturer, "snapshot") else None, **metrics}
        history.append(rec)
        live_str = f" | live_win {live_wr:.2f}" if live_wr is not None else ""
        print(f"[sgs] gen {gen:3d} | solved {cum_solved}/{len(D)} "
              f"({cum_solved/len(D):.0%}) | loss {loss_val:+.3f} "
              f"| ent {metrics.get('entropy', 0):.3f} | synth {len(synth_updates)}{live_str}")
        json.dump(history, open(out / "history.json", "w"), indent=2)
        if gen % 5 == 0 or gen == generations - 1:
            save_policy(policy, out / f"solver_{gen:04d}.pt")

    env.close()
    save_policy(policy, out / "solver_final.pt")
    return policy, history


if __name__ == "__main__":
    run_sgs()
