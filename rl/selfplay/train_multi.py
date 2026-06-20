"""Live multi-opponent PPO from a warm-start — "fight the other agents".

Rotates the opponent across a POOL of scripted agents each generation (memory
``kaggle-mcts-selfplay``: pure-bare COLLAPSES the net; opponent-mixing + warm-start
works). Anneals ``prior_weight`` to 0 across the run so the DEPLOYABLE raw net
(``net_agent`` runs at pw=0) is what actually gets optimized — the prior crutch
fades and the net stands on its own.

Run (Docker engine image):
  docker run --rm --platform=linux/amd64 -e PYTHONUNBUFFERED=1 -e RL_OBJECTIVE=ppo \
    -e RL_PRIOR_ANNEAL=300 -v "$PWD":/app -w /app cabt-rl \
    -m rl.train_multi --generations 300 --init rl/runs/solver_init.pt
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from rl.config import CONFIG, RUNS_DIR, solver_deck_path
from rl.engine.env import TCGEnv
from rl.training.solver.policy import PointerPolicy, prior_weight_at, load as load_policy, save as save_policy
from rl.training.solver.solver_objectives import get_objective
from rl.training.solver.train_solver import collect_rollouts, _load_deck

# Default opponent pool: weak -> strong, so there is always a gradient of
# winnable games (pure-strong starves; pure-weak collapses).
DEFAULT_POOL = [
    "agents.bare_agent",
    "agents.main",
    "agents.honchkrow",
    "agents.fire",
    "agents.main_v3_pure",
]


def train_multi(generations: int = 300, run_name: str = "train_multi",
                init: str | None = None, pool: list[str] | None = None):
    pool = pool or DEFAULT_POOL
    random.seed(CONFIG.seed); np.random.seed(CONFIG.seed); torch.manual_seed(CONFIG.seed)
    rng = random.Random(CONFIG.seed)

    solver_deck = _load_deck(solver_deck_path())
    mods = []
    for name in pool:
        m = importlib.import_module(name)
        deck = list(getattr(m, "my_deck", solver_deck))
        mods.append((name, m, deck))
    print(f"[multi] pool ({len(mods)}): {[n for n, _, _ in mods]}")

    env = TCGEnv(solver_deck, mods[0][2], mods[0][1])
    policy = (load_policy(init) if init and os.path.exists(init) else PointerPolicy())
    print(f"[multi] init = {init if init and os.path.exists(init) else 'fresh net'}")
    objective = get_objective()             # RL_OBJECTIVE=ppo
    opt = torch.optim.Adam(policy.parameters(), lr=CONFIG.lr)

    out = Path(RUNS_DIR) / run_name
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for gen in range(generations):
        pw = prior_weight_at(gen)
        name, mod, deck = rng.choice(mods)          # sample an opponent this generation
        env.opponent_agent = mod
        env.opponent_deck = deck
        rollouts = collect_rollouts(policy, env, [None], CONFIG.k_rollouts, pw)
        loss, metrics = objective.compute_loss(policy, rollouts, pw)
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), CONFIG.grad_clip)
            opt.step()
        else:
            gnorm = torch.tensor(0.0)
        winrate = float(np.mean([r.win for r in rollouts])) if rollouts else 0.0
        loss_val = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        rec = {"gen": gen, "opp": name, "winrate": winrate, "loss": loss_val,
               "prior_weight": pw, "grad_norm": float(gnorm), **metrics}
        history.append(rec)
        print(f"[multi] gen {gen:3d} | opp {name.split('.')[-1]:14s} | win {winrate:.2f} "
              f"| loss {loss_val:+.3f} | pw {pw:.2f} | gnorm {float(gnorm):.2f}")
        if gen % 20 == 0 or gen == generations - 1:
            save_policy(policy, out / f"solver_{gen:04d}.pt")
            json.dump(history, open(out / "history.json", "w"), indent=2)

    env.close()
    save_policy(policy, out / "solver_final.pt")
    return policy, history


def main():
    ap = argparse.ArgumentParser(description="Live multi-opponent PPO from a warm-start")
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--run-name", default="train_multi")
    ap.add_argument("--init", default="rl/runs/solver_init.pt")
    ap.add_argument("--pool", nargs="*", default=None, help="opponent module names")
    args = ap.parse_args()
    train_multi(args.generations, args.run_name, args.init, args.pool)


if __name__ == "__main__":
    main()
