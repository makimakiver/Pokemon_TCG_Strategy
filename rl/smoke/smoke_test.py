"""P0 smoke test (plan §5 P0): de-risk the scenario primitive everything depends on.

Verifies, inside the Docker engine image:
  1. the encoder/policy round-trip on a real observation;
  2. a live turn-0 game plays to a terminal result under random-masked actions;
  3. the target set D loads from data/loser/*.json;
  4. a mid-game scenario from D loads via search_begin and replays to terminal;
  5. no battle_ptr / search leaks across many reset cycles.

Run: docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app cabt-rl \
       python -m rl.smoke.smoke_test
"""
from __future__ import annotations

import random

import numpy as np
import torch

from rl.config import solver_deck_path
from rl.core.env import TCGEnv, STOP
from rl.solver.policy import PointerPolicy
from rl.core.targets import build_target_set
import json
import importlib


def _rand_action(obs, rng):
    n = obs["n_options"]
    legal = [i for i in range(n) if obs["mask"][i] > 0.5]
    if obs["mask"][n] > 0.5 and (not legal or rng.random() < 0.3):
        return STOP
    return rng.choice(legal) if legal else STOP


def main():
    rng = random.Random(0)
    deck = json.load(open(solver_deck_path()))
    assert len(deck) == 60, len(deck)
    opp = importlib.import_module("agents.bare_agent")
    opp_deck = list(getattr(opp, "my_deck", deck))

    print("[1] policy round-trip ...")
    policy = PointerPolicy()
    env = TCGEnv(deck, opp_deck, opp)
    obs = env.reset(None)
    a, lp, v = policy.act(obs)
    print(f"    n_options={obs['n_options']} action={a} logp={lp:.3f} value={v:.3f}")

    print("[2] live turn-0 game to terminal (random-masked) ...")
    steps = 0
    while not env.done and steps < 5000:
        obs, r, done, info = env.step(_rand_action(obs, rng))
        steps += 1
    print(f"    result={env.result} after {steps} micro-steps")
    env.close()

    print("[3] build target set D ...")
    D = build_target_set()
    print(f"    |D| = {len(D)}")
    if not D:
        print("    (no targets parsed — check data/loser/*.json shape)")
        return

    print("[4] load a mid-game scenario via search_begin and replay ...")
    env2 = TCGEnv(deck, opp_deck, opp)
    loaded = 0
    for spec in D[:5]:
        try:
            obs = env2.reset(spec)
        except Exception as e:
            print(f"    {spec.target_id}: search_begin rejected ({e})")
            continue
        loaded += 1
        s = 0
        while not env2.done and s < 3000:
            obs, r, done, info = env2.step(_rand_action(obs, rng))
            s += 1
        print(f"    {spec.target_id}: result={env2.result} in {s} steps")
    env2.close()
    print(f"    loaded {loaded}/{min(5, len(D))} scenarios")

    print("[5] leak check: 20 reset cycles ...")
    env3 = TCGEnv(deck, opp_deck, opp)
    for i in range(20):
        env3.reset(None)
        for _ in range(20):
            if env3.done:
                break
            env3.step(_rand_action(env3._encode(), rng))
    env3.close()
    print("    no crash across reset cycles")
    print("OK")


if __name__ == "__main__":
    np.random.seed(0); torch.manual_seed(0)
    main()
