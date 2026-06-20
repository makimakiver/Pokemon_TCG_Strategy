# rl/dsl/gate.py
"""P0 gate: does the handcrafted DSL rule-set match/beat bare_agent on the gauntlet?

Run (Docker): docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
  --entrypoint python cabt-rl -m rl.shared.dsl.gate
"""
from __future__ import annotations

import importlib
import json
import os

from rl.config import DECKS_DIR, EVAL_GAUNTLET, solver_deck_path
from rl.shared.engine.env import TCGEnv, STOP   # reuse the env for play; STOP imported for parity
from rl.shared.agents import dsl_agent


def _bare_for(deck_path):
    os.environ["BARE_DECK"] = str(deck_path)
    import agents.bare_agent as ba
    importlib.reload(ba)
    return ba


def _play(solver_mod, opp_mod, solver_deck, opp_deck, games):
    """solver_mod pilots solver_deck vs opp_mod; seats swapped; returns win-rate.
    Drives both via their agent(obs_dict) interface through the engine."""
    from cg.game import battle_start, battle_select, battle_finish
    wins = 0
    for g in range(games):
        seat = g % 2
        decks = (solver_deck, opp_deck) if seat == 0 else (opp_deck, solver_deck)
        mods = (solver_mod, opp_mod) if seat == 0 else (opp_mod, solver_mod)
        obs, start = battle_start(decks[0], decks[1])
        if obs is None:
            continue
        try:
            for _ in range(20000):
                cur = obs.get("current")
                if cur is None or cur.get("result", -1) != -1 or obs.get("select") is None:
                    break
                who = cur["yourIndex"]
                action = mods[who].agent(obs)
                obs = battle_select(action if isinstance(action, list) else list(action))
            res = (obs.get("current") or {}).get("result", -1)
        finally:
            battle_finish()
        if res == seat:
            wins += 1
    return wins / games


def main():
    games = int(os.environ.get("RL_GATE_GAMES", "20"))
    solver_deck = json.load(open(solver_deck_path()))
    print(f"P0 gate: dsl_agent vs bare_agent, {games} games/deck, seats swapped\n")
    dsl_total, bare_total = [], []
    for deck_name in EVAL_GAUNTLET:
        bare = _bare_for(DECKS_DIR / deck_name)
        opp_deck = list(bare.my_deck)
        # dsl_agent (our strategy) vs bare opponent
        dsl_wr = _play(dsl_agent, bare, solver_deck, opp_deck, games)
        # baseline: bare pilot (same solver deck) vs bare opponent
        bare_solver = _bare_for(solver_deck_path())   # bare piloting OUR deck
        bare_wr = _play(bare_solver, bare, solver_deck, opp_deck, games)
        dsl_total.append(dsl_wr); bare_total.append(bare_wr)
        print(f"  {deck_name:34s} dsl {dsl_wr:5.1%} | bare {bare_wr:5.1%}")
    d, b = sum(dsl_total) / len(dsl_total), sum(bare_total) / len(bare_total)
    print(f"\n  AVG  dsl {d:.1%} | bare {b:.1%}   ->  "
          f"{'PASS' if d >= b else 'FAIL'} (gate: dsl >= bare)")


if __name__ == "__main__":
    main()
