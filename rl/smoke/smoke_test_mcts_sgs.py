"""End-to-end smoke for the NN+MCTS solver env (plan Task 6). Docker only.

  docker build --platform=linux/amd64 -f rl/Dockerfile -t cabt-rl .
  docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app cabt-rl \
      python -m rl.smoke.smoke_test_mcts_sgs

Verifies: parametric problem-set export -> load/seed -> 2 SGS generations with the
MCTS actor under CISPO -> a finite loss and a written checkpoint. Tiny knobs so it
finishes fast; correctness of training quality is out of scope for a smoke.
"""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

# Tiny, fast settings BEFORE importing CONFIG-consumers.
os.environ.setdefault("RL_MCTS_SIMS", "8")
os.environ.setdefault("RL_K", "2")
os.environ.setdefault("RL_OBJECTIVE", "alphazero")

from rl.sgs.targets import build_target_set
from rl.sgs.problem_set import load_problem_set, seed_scenarios, write_problem_set
from rl.sgs.conjecturer import get_conjecturer
from rl.sgs.conjecturer.export_problems import export_problems
from rl.sgs.outer_loop import run_sgs


def main() -> int:
    D = build_target_set()
    assert D, "empty target set D; need data/loser/*.json in the image/mount"
    print(f"[smoke] |D| = {len(D)}")

    # 1) Export a parametric problem set, then load + seed it.
    rows = export_problems(get_conjecturer("parametric"), D, random.Random(0),
                           backend="parametric")
    tmp = Path(tempfile.mkdtemp()) / "problems.jsonl"
    write_problem_set(rows, tmp)
    ps = load_problem_set(tmp)
    seeded = seed_scenarios(D, ps)
    assert len(seeded) == len(D)
    assert len(ps) >= 1, "no problems exported"
    print(f"[smoke] exported+loaded {len(ps)} problems")

    # 2) Run 2 SGS generations with the MCTS actor under CISPO.
    #    conjecture_after=0 forces the parametric conjecturer to propose a variation
    #    every unsolved gen, so this smoke also exercises the propose/guide/synth path
    #    (with the default conjecture_after=2 it would never fire inside 2 gens).
    policy, history = run_sgs(generations=2, batch_size=min(4, len(D)),
                              run_name="smoke_mcts_sgs", use_mcts=True,
                              problem_set=str(tmp), objective="alphazero",
                              conjecture_after=0)
    assert len(history) == 2, history
    for rec in history:
        assert rec["loss"] == rec["loss"], "loss is NaN"   # NaN != NaN
    ckpt = Path("rl/runs/smoke_mcts_sgs/solver_final.pt")
    assert ckpt.exists(), f"missing checkpoint {ckpt}"
    print(f"[smoke] OK — 2 gens ran, checkpoint at {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
