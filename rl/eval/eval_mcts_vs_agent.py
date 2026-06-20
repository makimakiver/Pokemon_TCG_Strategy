"""Eval the kaggle_mcts NN+MCTS net vs any agents.<name> module, in full live
games (MCTS lookahead per move). Reuses kaggle_mcts.mcts_agent + sample_deck.

  docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
    -e EVAL_OPP=agents.main -e RL_EVAL_GAMES=16 -e RL_SEARCH_COUNT=10 \
    --entrypoint python cabt-rl -m rl.eval_mcts_vs_agent out/model_best.pth
"""
import os
import sys

os.environ["RL_OUTER_ITERS"] = "0"          # make kaggle_mcts import a no-op (no training)
os.environ.setdefault("RL_EVAL_OPP", "self")  # avoid module-level opponent side effects

import importlib
import torch

from rl.kaggle.kaggle_mcts import MyModel, mcts_agent, sample_deck
from cg.game import battle_start, battle_finish, battle_select


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "out/model_best.pth"
    opp_name = os.environ.get("EVAL_OPP", "agents.main")
    if "." not in opp_name:
        opp_name = f"agents.{opp_name}"
    n = int(os.environ.get("RL_EVAL_GAMES", "16"))

    # kaggle_mcts.sample_deck is hardcoded 59 cards; pad to a legal 60 with its
    # filler energy (the net is deck-conditioned on this list; +1 energy is benign).
    net_deck = list(sample_deck)
    while len(net_deck) < 60:
        net_deck.append(net_deck[-1])
    net_deck = net_deck[:60]

    opp = importlib.import_module(opp_name)
    opp_deck = list(getattr(opp, "my_deck", net_deck))
    if len(opp_deck) != 60:
        opp_deck = opp.agent({"select": None, "logs": [], "current": None})
    device = torch.device("cpu")
    model = MyModel(128, 2, 256, 1, 1).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"Loaded {ckpt}; MCTS net vs {opp_name} over {n} games (seats swapped).", flush=True)

    W = L = D = 0
    with torch.inference_mode():
        for i in range(n):
            net_seat = i % 2
            obs, sd = (battle_start(net_deck, opp_deck) if net_seat == 0
                       else battle_start(opp_deck, net_deck))
            if sd.errorPlayer >= 0:
                raise ValueError(f"deck error: errorType={sd.errorType}")
            while obs["current"]["result"] < 0:
                if obs["current"]["yourIndex"] == net_seat:
                    selected, _ = mcts_agent(obs, net_deck, model)
                else:
                    selected = opp.agent(obs)
                obs = battle_select(selected)
            battle_finish()
            r = obs["current"]["result"]
            if r == 2:
                D += 1
            elif r == net_seat:
                W += 1
            else:
                L += 1
            sys.stderr.write(f"\r  game {i+1}/{n}  W{W} L{L} D{D}   ")
            sys.stderr.flush()
    sys.stderr.write("\n")
    dec = W + L
    print(f"{ckpt} vs {opp_name}: net win {100*W//dec if dec else 0}%  "
          f"({W}W / {L}L / {D}D over {n})", flush=True)


if __name__ == "__main__":
    main()
