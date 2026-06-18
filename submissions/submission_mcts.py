"""COMPETITION SUBMISSION — round-2 MCTS net (kaggle_mcts `out/model2.pth`).

This is the `(deck + agent)` pair built from the AlphaZero-style self-play net.
Unlike the scripted submissions in this folder, it is NOT pure-`cg`: it also needs
`torch` and the trained weights file `model2.pth` (the round-2 checkpoint — the
70%-vs-random net). At each decision it runs MCTS guided by the net and returns
the chosen option list, exactly like `kaggle_mcts.mcts_agent` does during eval.

Local validation (cabt-rl image — has torch; cg is Linux x86-64 only):

    docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
      --entrypoint python cabt-rl runner.py \
      --a submissions.submission_mcts --b agents.bare_agent -n 10

Knobs (env): RL_SEARCH_COUNT (MCTS sims/move, default 10 — lower = faster,
weaker), RL_CKPT (path to the .pth; otherwise out/model2.pth, then a Kaggle glob).

CAVEATS for an actual Kaggle upload (read before submitting):
  * The net was only validated vs `random_agent` (~63-70%). It is far weaker than
    the scripted submissions here, which beat real meta decks. Expect it to lose
    to strong opponents.
  * MCTS + a transformer per node on CPU is SLOW. Confirm it fits the competition
    per-move time limit (tune RL_SEARCH_COUNT down if not).
  * Kaggle packaging: bundle model2.pth (upload as a dataset) and inline the model
    /MCTS code so it does not depend on the repo's `rl/` package being present.
"""
import glob
import os

# kaggle_mcts runs its training driver on import; RL_OUTER_ITERS=0 makes that a
# no-op so we only get the building blocks (model, MCTS, encoders, sample_deck).
os.environ.setdefault("RL_OUTER_ITERS", "0")

import torch

from rl.kaggle_mcts import MyModel, mcts_agent, sample_deck

# The deck the net was trained on (the agent must play the same list it learned).
my_deck = list(sample_deck)
assert len(my_deck) == 60, f"deck has {len(my_deck)} cards"


def _find_checkpoint():
    """Locate the net weights: explicit RL_CKPT, then the staged best net
    (out/model_best.pth = mix R3, 53% vs bare-mirror), then older checkpoints,
    then a Kaggle dataset glob."""
    for p in (os.environ.get("RL_CKPT"), "out/model_best.pth",
              "out/model2.pth", "rl/out/model_best.pth"):
        if p and os.path.exists(p):
            return p
    for pat in ("/kaggle/input/**/model_best.pth", "/kaggle/input/**/model*.pth"):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = MyModel(128, 2, 256, 1, 1).to(_device)
_ckpt = _find_checkpoint()
if _ckpt is not None:
    _model.load_state_dict(torch.load(_ckpt, map_location=_device))
    print(f"[submission_mcts] loaded {_ckpt} on {_device}", flush=True)
else:
    # No weights found — play with the (random-init) net rather than crash, so the
    # runner's deck-request path still works. Real submissions must ship the .pth.
    print("[submission_mcts] WARNING: model2.pth not found; using untrained weights", flush=True)
_model.eval()


def agent(obs_dict: dict) -> list[int]:
    """Competition agent interface: observation dict -> chosen option indices."""
    # Deck-request path (runner.py probes with a null observation).
    if obs_dict.get("current") is None or obs_dict.get("select") is None:
        return list(my_deck)
    with torch.inference_mode():
        selected, _sample = mcts_agent(obs_dict, my_deck, _model)
    return selected
