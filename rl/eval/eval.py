"""Held-out gauntlet evaluation (plan §4) — the headline ship metric.

Plays the trained Solver (live, turn-0) against each ORIGINAL unmutated meta deck
piloted by ``agents.bare_agent``, seats swapped each game. Returns per-deck and
meta-share-weighted win-rate. These decks are NEVER used for conjecturer training,
and the conjecturer only *edits* target scenarios, so there is no leakage.

Ship when the weighted win-rate beats the current ``main`` agent and plateaus.
Runs in the Docker engine image.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import numpy as np

from rl.config import CONFIG, DECKS_DIR, EVAL_GAUNTLET, solver_deck_path
from rl.core.env import TCGEnv

# Meta-share weights (limitlesstcg TEF-CRI). Uniform placeholder until the exact
# shares are wired in; keys are deck json filenames.
META_SHARES = {d: 1.0 for d in EVAL_GAUNTLET}


def _load(path) -> list[int]:
    return json.load(open(path))


def _bare_for(deck_path: str):
    os.environ["BARE_DECK"] = str(deck_path)
    import agents.bare_agent as ba
    importlib.reload(ba)        # re-derive roles for this deck
    return ba


def evaluate(policy, games_each: int = 20, use_mcts: bool | None = None,
             gauntlet=None) -> dict:
    gauntlet = gauntlet or EVAL_GAUNTLET
    use_mcts = CONFIG.mcts_enabled if use_mcts is None else use_mcts
    solver_deck = _load(solver_deck_path())

    results = {}
    for deck_name in gauntlet:
        opp = _bare_for(DECKS_DIR / deck_name)
        opp_deck = list(opp.my_deck)
        wins = 0
        for g in range(games_each):
            seat = g % 2
            env = TCGEnv(solver_deck, opp_deck, opp, solver_seat=seat)
            obs = env.reset(None)
            guard = 0
            while not env.done and guard < CONFIG.max_steps:
                guard += 1
                action, _, _ = policy.act(obs, prior_weight=0.0, greedy=True)
                obs, _, done, _ = env.step(action)
            if env.result == seat:
                wins += 1
            env.close()
        results[deck_name] = wins / games_each
        print(f"[eval] {deck_name:34s} {results[deck_name]:.2%}")

    weights = np.array([META_SHARES.get(d, 1.0) for d in gauntlet])
    wr = np.array([results[d] for d in gauntlet])
    weighted = float((weights * wr).sum() / weights.sum())
    results["_weighted"] = weighted
    print(f"[eval] meta-share-weighted win-rate: {weighted:.2%}")
    return results


if __name__ == "__main__":
    from rl.solver.policy import load
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(Path(CONFIG.device))
    policy = load(ckpt)
    evaluate(policy)
