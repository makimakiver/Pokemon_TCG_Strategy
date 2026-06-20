"""Offline behavioral cloning of scripted pilots into the Solver net (plan §3.2).

Bootstraps the pointer net to ~scripted strength before RL. We run a scripted
pilot (``agents.honchkrow`` / ``agents.fire`` / ``agents.main_v1``) as the solver
inside ``TCGEnv`` and record, at every micro-step, the encoded state and the
*next* pick the scripted agent makes (its full selection, replayed pick-by-pick,
terminated by STOP). Then we minimize masked cross-entropy of the policy against
those targets.

Runs in the Docker engine image.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rl.config import CONFIG
from rl.shared.engine import encode
from rl.shared.engine.env import TCGEnv, STOP, _to_dict
from rl.shared.policy import PointerPolicy


# Card-id signature of the Team Rocket's Honchkrow ("Crow") deck, used to detect
# which seat Crow occupied in a logged replay (the seat swaps between games).
CROW_DECK_SIG = frozenset({891, 463, 15, 1216, 1217, 1218, 1219, 1220})


def _deck_signature_seat(replay: dict, sig: frozenset[int]) -> int:
    """Return the seat (0/1) whose 60-card deck best matches ``sig``."""
    def deck(seat: int) -> set[int]:
        for step in replay.get("steps", [])[:3]:
            act = step[seat].get("action") or []
            if len(act) == 60:
                return set(act)
        return set()
    return 0 if len(sig & deck(0)) >= len(sig & deck(1)) else 1


def collect_offline_traces(replay_paths, deck_sig: frozenset[int] = CROW_DECK_SIG,
                           only_when_seat_won: bool = True):
    """Offline BC traces from logged kaggle-style replays (e.g. data/Crow/win_*.json).

    The live :func:`collect_traces` runs a scripted pilot inside ``TCGEnv``; this
    is its offline twin — it reads the *logged* observations directly and clones
    the decisions of the deck identified by ``deck_sig`` (Crow). The seat Crow
    sat in is auto-detected per replay because it swaps between games. By default
    only games Crow *won* are cloned (cloning a loss would teach losing play —
    feed losses to the SGS target set ``D`` via ``targets.py`` instead).

    Returns ``[(obs_dict, target_idx)]`` in the exact format :func:`train_sft`
    consumes (``obs_dict`` keys: global/options/mask/n_options).
    """
    from cg.api import to_observation_class

    data = []
    for path in replay_paths:
        replay = json.load(open(path))
        seat = _deck_signature_seat(replay, deck_sig)
        rewards = replay.get("rewards") or [0, 0]
        if only_when_seat_won and rewards[seat] != 1:
            continue
        for step in replay.get("steps", []):
            ag = step[seat]
            if ag.get("status") != "ACTIVE":
                continue
            obs = ag.get("observation")
            if not isinstance(obs, dict) or obs.get("select") is None:
                continue
            cur = obs.get("current")
            if not cur or cur.get("result", -1) != -1:
                continue
            o = to_observation_class(obs)
            g, opts, _ = encode.featurize(o)
            n = len(o.select.option)
            if n == 0:
                continue
            enc = {"global": g, "options": opts,
                   "mask": np.ones(n + 1, np.float32), "n_options": n}
            for pick in (ag.get("action") or []):
                if isinstance(pick, int) and 0 <= pick < n:
                    data.append((enc, pick))
    return data


def collect_traces(solver_deck, opponent_deck, scripted_module: str,
                   opponent_module: str, n_games: int = 20, seed: int = 0):
    """Play ``scripted_module`` as the solver; return [(obs_dict, target_idx)]."""
    scripted = importlib.import_module(scripted_module)
    opponent = importlib.import_module(opponent_module)
    env = TCGEnv(solver_deck, opponent_deck, opponent)
    data = []
    for _ in range(n_games):
        env.reset()
        guard = 0
        while not env.done and guard < CONFIG.max_steps:
            guard += 1
            sel = env._obs.select
            if sel is None:
                break
            full = scripted.agent(_to_dict(env._obs))     # absolute option indices
            full = [a for a in (full if isinstance(full, list) else list(full))
                    if 0 <= a < len(sel.option)]
            for pick in full:
                if env.done:
                    break
                data.append((env._encode(), pick))
                env.step(pick)
            # Close a variable-length select the scripted agent stopped short of.
            if not env.done and env._obs.select is sel and env._pending:
                data.append((env._encode(), env._encode()["n_options"]))  # STOP target
                env.step(STOP)
    env.close()
    return data


def train_sft(policy: PointerPolicy, data, epochs: int = 3, lr: float = 1e-3):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for ep in range(epochs):
        np.random.shuffle(data)
        total, n = 0.0, 0
        for obs, target in data:
            g = torch.as_tensor(obs["global"], dtype=torch.float32)
            o = torch.as_tensor(obs["options"], dtype=torch.float32)
            m = torch.as_tensor(obs["mask"], dtype=torch.float32)
            logits, _ = policy.forward(g, o, m, prior_weight=0.0)
            tgt = torch.tensor(min(target, logits.shape[0] - 1))
            loss = F.cross_entropy(logits.unsqueeze(0), tgt.unsqueeze(0))
            if torch.isinf(loss) or torch.isnan(loss):
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()); n += 1
        print(f"[sft] epoch {ep+1}/{epochs}  loss={total/max(1,n):.4f}  n={n}")
    return policy
