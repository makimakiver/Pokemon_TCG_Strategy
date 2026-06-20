"""Run a trained/distilled PointerPolicy as a harness agent — the SHIPPABLE form.

This loads a saved net checkpoint and exposes the standard agent interface
(``agent(obs_dict) -> list[int]`` + ``my_deck``), so the distilled-from-Claude
net (or any RL-trained net) plays anywhere `llm_agent` does — runner.py, the env,
and, crucially, as the competition submission: it makes ZERO external API calls.

Checkpoint path comes from ``RL_NET_CKPT`` (default rl/runs/distilled_claude.pt).
If the checkpoint is missing it falls back to the scripted prior (with a warning)
so the module is always importable and runnable.

Greedy decode: the net is micro-stepped, but the agent interface wants the whole
selection for one observation, so we score every option once and greedily take
the highest-logit options until the engine's min/max is satisfied (STOP ranked as
an extra edge). Requires ``cg`` (Linux) -> Docker image.
"""
from __future__ import annotations

import json
import os

import numpy as np

from cg.api import to_observation_class
from rl.engine import encode
from rl.config import CONFIG, RUNS_DIR, solver_deck_path

my_deck = json.load(open(solver_deck_path()))
assert len(my_deck) == 60

_CKPT = os.environ.get("RL_NET_CKPT", str(RUNS_DIR / "distilled_claude.pt"))
_policy = None
_failed = False


def _get_policy():
    global _policy, _failed
    if _policy is not None or _failed:
        return _policy
    try:
        from rl.training.solver.policy import load
        _policy = load(_CKPT)
        print(f"[net_agent] loaded checkpoint {_CKPT}")
    except Exception as e:
        _failed = True
        print(f"[net_agent] no usable checkpoint ({e}) -> scripted-prior fallback")
    return _policy


def _prior_picks(obs):
    sel = obs.select
    scores = encode.option_prior_scores(obs)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n = max(sel.minCount, min(sel.maxCount, len(order)))
    picks = order[:n]
    return picks if len(picks) >= sel.minCount else order[: sel.minCount]


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return list(my_deck)
    sel = obs.select
    n = len(sel.option)
    if n == 0:
        return []
    policy = _get_policy()
    if policy is None:
        return _prior_picks(obs)

    import torch
    g, opts, _ = encode.featurize(obs)
    mask = np.ones(n + 1, np.float32)
    with torch.no_grad():
        logits, _ = policy.forward(torch.as_tensor(g), torch.as_tensor(opts),
                                   torch.as_tensor(mask), 0.0)
    logits = logits.cpu().numpy()
    order = sorted(range(n), key=lambda i: logits[i], reverse=True)
    stop_logit = logits[n]
    picks = []
    for i in order:
        if len(picks) >= sel.maxCount:
            break
        if len(picks) >= sel.minCount and logits[i] < stop_logit:
            break
        picks.append(i)
    if len(picks) < sel.minCount:
        picks = order[: sel.minCount]
    return picks
