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

from .config import CONFIG, RUNS_DIR, solver_deck_path
from .env import TCGEnv
from .policy import PointerPolicy, prior_weight_at, save as save_policy
from .solver_objectives import get_objective, Rollout
from .targets import build_target_set
from .conjecturer import get_conjecturer
from .guide import guide_score
from .train_solver import collect_rollouts, _load_deck, make_mcts_actor
from .mcts import MCTS
from .problem_set import load_problem_set, seed_scenarios
from .scenario import ScenarioSpec


def _winrate(policy, env, scenario, k, pw, actor=None):
    rs = collect_rollouts(policy, env, [scenario], k, pw, actor=actor)
    return np.mean([r.win for r in rs]), rs


def run_sgs(generations: int = 20, batch_size: int = 8, run_name: str = "p2_sgs",
            use_mcts: bool = False, problem_set=None, objective: str | None = None):
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

    policy = PointerPolicy()
    objective = get_objective(objective or ("cispo" if use_mcts else None))
    conjecturer = get_conjecturer()
    opt = torch.optim.Adam(policy.parameters(), lr=CONFIG.lr)

    actor = None
    if use_mcts:
        mcts = MCTS(policy, opponent, solver_seat=CONFIG.seat)
        actor = make_mcts_actor(mcts, policy)
        print(f"[sgs] MCTS actor ON ({CONFIG.mcts_simulations} sims/decision)")

    solve_rate = defaultdict(float)
    out = Path(RUNS_DIR) / run_name
    out.mkdir(parents=True, exist_ok=True)
    history = []

    for gen in range(generations):
        pw = prior_weight_at(gen)
        batch = rng.sample(D, min(batch_size, len(D)))
        rollouts: list[Rollout] = []
        synth_updates = []

        for target in batch:
            solved = solve_rate[target.target_id] >= CONFIG.tau
            scenario = seeded.get(target.target_id, target)   # warm-started lemma
            edits = None
            conj_idx = None
            if not solved:
                # Conjecture an easier lemma for an unsolved target (parametric in-loop).
                edited, edits, conj_idx = conjecturer.propose(target, rng)
                scenario = edited
            try:
                wr, rs = _winrate(policy, env, scenario, CONFIG.k_rollouts, pw, actor=actor)
                legal = True
            except Exception:
                wr, rs, legal = 0.0, [], False
            rollouts.extend(rs)
            solve_rate[target.target_id] = float(wr)

            if not solved and edits is not None:
                r_solve = 1.0 * (1.0 - wr)               # reward hard-but-solvable lemmas
                r_guide = guide_score(scenario, edits, legal=legal)
                r_synth = r_solve * r_guide
                synth_updates.append((conj_idx, r_synth))

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
               "conjecturer": conjecturer.snapshot()
               if hasattr(conjecturer, "snapshot") else None, **metrics}
        history.append(rec)
        print(f"[sgs] gen {gen:3d} | solved {cum_solved}/{len(D)} "
              f"({cum_solved/len(D):.0%}) | loss {loss_val:+.3f} "
              f"| ent {metrics.get('entropy', 0):.3f} | synth {len(synth_updates)}")
        json.dump(history, open(out / "history.json", "w"), indent=2)
        if gen % 5 == 0 or gen == generations - 1:
            save_policy(policy, out / f"solver_{gen:04d}.pt")

    env.close()
    save_policy(policy, out / "solver_final.pt")
    return policy, history


if __name__ == "__main__":
    run_sgs()
