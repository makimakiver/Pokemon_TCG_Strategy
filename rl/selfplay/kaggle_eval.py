"""Load a saved kaggle_mcts checkpoint and play with it (inference only).

`kaggle_mcts.py` runs its training driver at import time. That driver is a
`for counter in range(OUTER_ITERS)` loop reading RL_OUTER_ITERS, so setting
RL_OUTER_ITERS=0 *before* importing makes the import a no-op (it still defines
MyModel / mcts_agent / random_agent / sample_deck, but trains nothing and
touches no checkpoint). We then load the requested weights into a fresh net.

Defaults to round 2 (out/model2.pth) — the 70% net from the balanced run.
`model{n}.pth` is the net saved right before round n's eval, i.e. the exact
weights that produced metrics round n.

Run inside the cabt-rl image (cg/libcg.so is Linux x86-64 only):

    # use the round-2 checkpoint, 30 games vs random
    docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
      --entrypoint python cabt-rl -m rl.selfplay.kaggle_eval

    # any checkpoint + game count + search budget
    docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
      -e RL_EVAL_GAMES=50 -e RL_SEARCH_COUNT=10 \
      --entrypoint python cabt-rl -m rl.selfplay.kaggle_eval out/model3.pth
"""
import os
import sys

# Suppress kaggle_mcts's training driver on import (see module docstring).
os.environ["RL_OUTER_ITERS"] = "0"

import torch

from rl.selfplay.kaggle_mcts import MyModel, mcts_agent, random_agent, sample_deck
from cg.game import battle_start, battle_finish, battle_select


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "out/model2.pth"
    n_games = int(os.environ.get("RL_EVAL_GAMES", "30"))

    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt} — run rl.selfplay.kaggle_mcts first to produce out/model*.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Same architecture the trainer used: MyModel(128, 2, 256, 1, 1).
    model = MyModel(128, 2, 256, 1, 1).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"Loaded {ckpt} on {device}; playing {n_games} games vs random_agent.", flush=True)

    results = [0, 0, 0]  # [wins, losses, draws]
    with torch.inference_mode():
        for i in range(n_games):
            obs, start_data = battle_start(sample_deck, sample_deck)
            if start_data.errorPlayer >= 0:
                raise ValueError(f"deck error: errorType={start_data.errorType}")
            your_index = i % 2  # alternate seats, like the trainer's eval
            while obs["current"]["result"] < 0:
                if obs["current"]["yourIndex"] == your_index:
                    selected, _ = mcts_agent(obs, sample_deck, model)
                else:
                    selected = random_agent(obs)
                obs = battle_select(selected)
            battle_finish()

            r = obs["current"]["result"]
            if r == 2:
                results[2] += 1
            elif r == your_index:
                results[0] += 1
            else:
                results[1] += 1
            sys.stderr.write(f"\r  game {i + 1}/{n_games}  W{results[0]} L{results[1]} D{results[2]}   ")
            sys.stderr.flush()
    sys.stderr.write("\n")

    decisive = results[0] + results[1]
    win_rate = 100 * results[0] // decisive if decisive else 0
    print(f"{ckpt}: win rate {win_rate}%  ({results[0]}W / {results[1]}L / {results[2]}D over {n_games} games)",
          flush=True)


if __name__ == "__main__":
    main()
