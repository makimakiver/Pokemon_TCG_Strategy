"""Distill the LLM-driven policy (Claude) into the shippable pointer net, then
compare the distilled net against the bare pilot on the same deck.

Pipeline:
  1. Claude (rl.llm_agent) plays N games as the solver vs the bare pilot; we
     record every (state -> pick) decision (sft.collect_traces).
  2. Behavioral-clone those decisions into a fresh PointerPolicy (sft.train_sft).
  3. Head-to-head, seats swapped: distilled net vs bare_agent on the SAME deck
     (the Honchkrow 26267 solver deck) -> a clean "did distillation help?" number.
     We also score an UNTRAINED net as the floor.

Run the REAL distillation (needs a key):
  docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
    -e ANTHROPIC_API_KEY=sk-... -e RL_LLM_MODEL=claude-opus-4-8 cabt-rl \
    -m rl.distill_claude --teacher-games 40 --epochs 3 --eval-games 40

Without a key, rl.llm_agent falls back to the scripted prior, so this becomes a
plumbing dry-run (distills the prior, not Claude) — the printed banner says which.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from rl.config import CONFIG, RUNS_DIR, solver_deck_path
from rl.engine.env import TCGEnv
from rl.training.solver.policy import PointerPolicy, save as save_policy
from rl.training.bootstrap.sft import collect_traces, train_sft


def _load(p):
    return json.load(open(p))


def head_to_head(policy, solver_deck, opp_module, opp_deck, games: int) -> float:
    """Net (greedy) vs the bare pilot on the same deck, seats swapped each game."""
    wins = 0
    for g in range(games):
        seat = g % 2
        env = TCGEnv(solver_deck, opp_deck, opp_module, solver_seat=seat)
        obs = env.reset(None)
        guard = 0
        while not env.done and guard < CONFIG.max_steps:
            guard += 1
            action, _, _ = policy.act(obs, prior_weight=0.0, greedy=True)
            obs, _, _, _ = env.step(action)
        if env.result == seat:
            wins += 1
        env.close()
    return wins / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-games", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-games", type=int, default=6)
    ap.add_argument("--out", default=str(RUNS_DIR / "distilled_claude.pt"),
                    help="where to save the distilled net (reusable via rl.net_agent)")
    ap.add_argument("--traces", default=str(RUNS_DIR / "claude_traces.json"),
                    help="cache file for teacher decisions")
    ap.add_argument("--load-traces", action="store_true",
                    help="reuse cached teacher traces instead of calling Claude again")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(CONFIG.seed); torch.manual_seed(CONFIG.seed)
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print("=" * 70)
    print(f"TEACHER: {'Claude (' + os.environ.get('RL_LLM_MODEL', 'claude-opus-4-8') + ')' if have_key else 'SCRIPTED PRIOR (no ANTHROPIC_API_KEY -> dry run)'}")
    print("=" * 70)

    solver_deck = _load(solver_deck_path())
    bare = importlib.import_module("agents.bare_agent")
    opp_deck = list(bare.my_deck)

    if args.load_traces and Path(args.traces).exists():
        cached = json.load(open(args.traces))
        data = [(d["obs"], d["target"]) for d in cached]
        # rehydrate numpy arrays
        for obs, _ in data:
            obs["global"] = np.asarray(obs["global"], np.float32)
            obs["options"] = np.asarray(obs["options"], np.float32)
            obs["mask"] = np.asarray(obs["mask"], np.float32)
        print(f"[1] loaded {len(data)} cached teacher decisions from {args.traces}")
    else:
        print(f"[1] collecting {args.teacher_games} teacher games (rl.llm_agent as solver) ...")
        data = collect_traces(solver_deck, opp_deck, "rl.inference.agents.llm_agent", "agents.bare_agent",
                              n_games=args.teacher_games)
        print(f"    {len(data)} (state -> pick) decisions collected")
        # Cache so re-SFT never re-pays for Claude calls.
        serial = [{"obs": {"global": o["global"].tolist(),
                           "options": o["options"].tolist(),
                           "mask": o["mask"].tolist(), "n_options": o["n_options"]},
                   "target": int(t)} for o, t in data]
        json.dump(serial, open(args.traces, "w"))
        print(f"    cached teacher traces -> {args.traces}")

    print("[2] baseline: UNTRAINED net vs bare pilot (mirror) ...")
    floor = PointerPolicy()
    floor_wr = head_to_head(floor, solver_deck, bare, opp_deck, args.eval_games)
    print(f"    untrained net win-rate: {floor_wr:.1%}")

    print(f"[3] SFT-cloning the teacher into a fresh net ({args.epochs} epochs) ...")
    net = PointerPolicy()
    train_sft(net, data, epochs=args.epochs)
    save_policy(net, args.out)
    print(f"    saved distilled net -> {args.out}  (run it via rl.net_agent)")

    print("[4] distilled net vs bare pilot (mirror) ...")
    dist_wr = head_to_head(net, solver_deck, bare, opp_deck, args.eval_games)

    print("\n" + "=" * 70)
    print(f"  untrained net : {floor_wr:.1%}")
    print(f"  distilled net : {dist_wr:.1%}   (Δ {100*(dist_wr-floor_wr):+.1f} pts)")
    print(f"  teacher       : {'Claude' if have_key else 'scripted prior (dry run)'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
