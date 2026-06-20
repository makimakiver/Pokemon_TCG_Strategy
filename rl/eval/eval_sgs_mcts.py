"""Eval the SGS PointerPolicy in FULL games via DETERMINIZED live-MCTS.

The SGS solver was trained in search-mode (mid-game positions); this harness lets
it play a whole game. At each solver decision in a live ``battle_start`` game we:
  1. determinize the hidden info from the live obs + the two known decks
     (reuse targets.py's prediction logic),
  2. ``search_begin`` that determinized position,
  3. run ``rl.mcts.MCTS`` (PointerPolicy prior+value, scripted opponent in-tree),
  4. pick the move by the root visit distribution, and apply it to the LIVE game.
This is the apples-to-apples measure of the SGS net in real games.

  docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
    -e RL_NET_CKPT=rl/runs/eval_ckpts/az_final.pt -e EVAL_OPP=main_v3_pure \
    -e RL_EVAL_GAMES=12 -e RL_MCTS_SIMS=8 \
    --entrypoint python cabt-rl -m rl.eval_sgs_mcts
"""
import importlib
import os
import sys

import numpy as np
import torch

from cg.api import to_observation_class, search_begin, search_release
from cg.game import battle_start, battle_select, battle_finish

from rl.config import CONFIG, solver_deck_path
from rl.solver.policy import load as load_policy
from rl.core import encode
from rl.core.scenario import ScenarioSpec
from rl.solver.mcts import MCTS
from rl.core.targets import _pokemon_ids, _visible_ids, _hidden_pool, _take
from rl.solver.train_solver import _load_deck


def _build_spec(obs, deck_me, deck_op, poke_ids):
    cur = obs["current"]
    mi = cur["yourIndex"]
    me, op = cur["players"][mi], cur["players"][1 - mi]
    my_pool = _hidden_pool(deck_me, _visible_ids(me, include_hand=True))
    op_pool = _hidden_pool(deck_op, _visible_ids(op, include_hand=False))
    deck_sel = (obs.get("select") or {}).get("deck")
    your_deck = [] if deck_sel is not None else _take(my_pool, me["deckCount"])
    your_prize = _take(my_pool, len(me["prize"]))
    op_active = op.get("active") or []
    need_active = len(op_active) > 0 and op_active[0] is None
    opp_active = []
    if need_active:
        basic = next((x for x in op_pool if x in poke_ids), None)
        if basic is not None:
            op_pool.remove(basic); opp_active = [basic]
        else:
            opp_active = [op_pool[0]] if op_pool else [0]
    opp_deck = _take(op_pool, op["deckCount"])
    opp_prize = _take(op_pool, len(op["prize"]))
    opp_hand = _take(op_pool, op["handCount"])
    return ScenarioSpec(obs=obs, your_deck=your_deck, your_prize=your_prize,
                        opponent_deck=opp_deck, opponent_prize=opp_prize,
                        opponent_hand=opp_hand, opponent_active=opp_active)


def _net_greedy(obs, policy):
    """Fallback: rank options by the net (no MCTS), like net_agent."""
    o = to_observation_class(obs)
    sel = o.select
    n = len(sel.option)
    g, opts, _ = encode.featurize(o)
    mask = np.ones(n + 1, np.float32)
    with torch.no_grad():
        logits, _ = policy.forward(torch.as_tensor(g), torch.as_tensor(opts),
                                   torch.as_tensor(mask), 0.0)
    logits = logits.cpu().numpy()
    return _picks_from_scores(logits[:n], logits[n], sel.minCount, sel.maxCount, n)


def _picks_from_scores(scores, stop_score, mn, mx, n):
    mn = mn or 1; mx = mx or 1
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    picks = []
    for i in order:
        if len(picks) >= mx:
            break
        if len(picks) >= mn and scores[i] < stop_score:
            break
        picks.append(i)
    if len(picks) < mn:
        picks = order[:mn]
    return picks


def sgs_mcts_move(obs, deck_me, deck_op, policy, opp_mod, poke_ids):
    sel = obs["select"]
    n = len(sel.get("option") or [])
    mn = sel.get("minCount", 1) or 1
    if n == 0:
        return []
    if n <= 1 or mn >= n:                      # forced -> no search
        return list(range(n))[:max(mn, 1)]
    try:
        spec = _build_spec(obs, deck_me, deck_op, poke_ids)
        spec.validate_shapes()
        state = search_begin(to_observation_class(obs), **spec.search_begin_kwargs())
    except Exception:
        return _net_greedy(obs, policy)        # determinization failed -> net greedy
    seat = obs["current"]["yourIndex"]
    mcts = MCTS(policy, opp_mod, solver_seat=seat)
    try:
        _, pi, _ = mcts.search_policy(state)
    except Exception:
        try:
            search_release(state.searchId)
        except Exception:
            pass
        return _net_greedy(obs, policy)
    try:
        search_release(state.searchId)
    except Exception:
        pass
    if pi is None or len(pi) == 0 or float(np.sum(pi)) <= 0:
        return _net_greedy(obs, policy)
    return _picks_from_scores(pi, 1e-9, sel.get("minCount", 1), sel.get("maxCount", 1), n)


def main():
    ckpt = os.environ.get("RL_NET_CKPT", "rl/runs/eval_ckpts/az_final.pt")
    opp_name = os.environ.get("EVAL_OPP", "main_v3_pure")
    if "." not in opp_name:
        opp_name = f"agents.{opp_name}"
    n_games = int(os.environ.get("RL_EVAL_GAMES", "12"))

    deck_me = _load_deck(solver_deck_path())
    opp = importlib.import_module(opp_name)
    deck_op = list(getattr(opp, "my_deck", deck_me))
    if len(deck_op) != 60:
        deck_op = opp.agent({"select": None, "logs": [], "current": None})
    policy = load_policy(ckpt)
    poke_ids = _pokemon_ids()
    print(f"[sgs-mcts] {ckpt} vs {opp_name} | {n_games} games | "
          f"{CONFIG.mcts_simulations} sims/decision", flush=True)

    W = L = D = 0
    for i in range(n_games):
        net_seat = i % 2
        obs, start = (battle_start(deck_me, deck_op) if net_seat == 0
                      else battle_start(deck_op, deck_me))
        if obs is None:
            raise RuntimeError(f"battle_start failed: errorType={start.errorType}")
        guard = 0
        while obs["current"]["result"] < 0 and guard < 5000:
            guard += 1
            if obs["current"]["yourIndex"] == net_seat:
                picks = sgs_mcts_move(obs, deck_me, deck_op, policy, opp, poke_ids)
            else:
                try:
                    picks = opp.agent(obs)
                except Exception:
                    sel = obs["select"]
                    picks = list(range((sel or {}).get("minCount", 1) or 1))
            obs = battle_select(picks)
        battle_finish()
        r = obs["current"]["result"]
        if r == 2:
            D += 1
        elif r == net_seat:
            W += 1
        else:
            L += 1
        sys.stderr.write(f"\r  game {i+1}/{n_games}  W{W} L{L} D{D}   "); sys.stderr.flush()
    sys.stderr.write("\n")
    dec = W + L
    print(f"{ckpt} vs {opp_name}: SGS-MCTS net win {100*W//dec if dec else 0}%  "
          f"({W}W / {L}L / {D}D over {n_games})", flush=True)


if __name__ == "__main__":
    main()
