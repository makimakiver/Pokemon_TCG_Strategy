"""Bootstrap the Solver net's INITIAL POINT from the scripted main_v5 pilot.

This is the no-API counterpart to ``rl.distill_claude``: instead of cloning Claude
(``rl.llm_agent``, which needs ANTHROPIC_API_KEY), it behavioral-clones
``agents.main_v5`` — the tuned dual-attacker Honchkrow/Porygon2 pilot bound to the
pinned solver deck — into a fresh ``PointerPolicy``. The result is the warm-start
checkpoint that ``rl.training.solver.train_solver(init_policy=...)`` continues RL from, and that
``rl.net_agent`` can ship directly.

Pipeline (all inside the Docker engine image):
  1. main_v5 plays N games as the solver vs the anchor opponent; record every
     (state -> pick) decision (``sft.collect_traces``).
  2. SFT-clone those decisions into a fresh net (``sft.train_sft``).
  3. Save -> rl/runs/solver_init.pt  (point RL_NET_CKPT here, or pass init_policy).
  4. Report win-rates so the bootstrap is not a black box:
       - untrained net   vs anchor      (floor)
       - SFT'd  net      vs anchor      (did cloning help?)
       - SFT'd  net      vs main_v5      (mirror: how faithfully did it clone?)

Run:
  docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app cabt-rl \
    -m rl.training.bootstrap.bootstrap_solver --teacher-games 40 --epochs 3 --eval-games 40
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from rl.config import CONFIG, RUNS_DIR, solver_deck_path
from rl.training.solver.policy import PointerPolicy, save as save_policy
from rl.training.bootstrap.sft import collect_traces, train_sft
from rl.training.bootstrap.distill_claude import head_to_head   # net (greedy) vs a module pilot, seats swapped

TEACHER = "agents.main_v5"


def _load(p):
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser(description="SFT-bootstrap the solver net from main_v5")
    ap.add_argument("--teacher-games", type=int, default=40,
                    help="games main_v5 plays as the solver to generate cloning data")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--opp", default=CONFIG.opponent_module,
                    help="anchor opponent module (default from RL_OPP)")
    ap.add_argument("--out", default=str(RUNS_DIR / "solver_init.pt"),
                    help="where to save the warm-start net (use via RL_NET_CKPT / init_policy)")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(CONFIG.seed); torch.manual_seed(CONFIG.seed)

    solver_deck = _load(solver_deck_path())
    teacher_mod = importlib.import_module(TEACHER)
    opp_mod = importlib.import_module(args.opp)
    # Each pilot plays its own my_deck; fall back to the solver deck if it pins none.
    opp_deck = list(getattr(opp_mod, "my_deck", solver_deck))
    teacher_deck = list(getattr(teacher_mod, "my_deck", solver_deck))

    print("=" * 70)
    print(f"BOOTSTRAP solver init  |  teacher={TEACHER}  anchor={args.opp}")
    print(f"  solver deck: {solver_deck_path().name}  ({len(solver_deck)} cards)")
    print("=" * 70)

    print(f"[1] collecting {args.teacher_games} teacher games (main_v5 as solver vs {args.opp}) ...")
    data = collect_traces(solver_deck, opp_deck, TEACHER, args.opp,
                          n_games=args.teacher_games)
    print(f"    {len(data)} (state -> pick) decisions collected")

    print("[2] floor: UNTRAINED net vs anchor ...")
    floor = PointerPolicy()
    floor_wr = head_to_head(floor, solver_deck, opp_mod, opp_deck, args.eval_games)
    print(f"    untrained net win-rate: {floor_wr:.1%}")

    print(f"[3] SFT-cloning main_v5 into a fresh net ({args.epochs} epochs) ...")
    net = PointerPolicy()
    train_sft(net, data, epochs=args.epochs)
    save_policy(net, args.out)
    print(f"    saved warm-start net -> {args.out}")

    print("[4] SFT'd net vs anchor ...")
    anchor_wr = head_to_head(net, solver_deck, opp_mod, opp_deck, args.eval_games)
    print(f"[5] SFT'd net vs main_v5 teacher (mirror, same deck) ...")
    mirror_wr = head_to_head(net, solver_deck, teacher_mod, teacher_deck, args.eval_games)

    print("\n" + "=" * 70)
    print(f"  untrained net vs anchor   : {floor_wr:.1%}   (floor)")
    print(f"  SFT'd net     vs anchor   : {anchor_wr:.1%}   (Δ {100*(anchor_wr-floor_wr):+.1f} pts)")
    print(f"  SFT'd net     vs main_v5  : {mirror_wr:.1%}   (->50% = faithful clone)")
    print(f"  init checkpoint           : {args.out}")
    print("=" * 70)
    print("Next: rl.training.solver.train_solver.train(init_policy=rl.training.solver.policy.load('%s'))" % args.out)


if __name__ == "__main__":
    main()
