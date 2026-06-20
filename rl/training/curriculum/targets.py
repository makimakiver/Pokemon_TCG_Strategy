"""Build the fixed target set D of hard losing positions (plan §3.1 / open item).

Source: ``data/loser/lost_*.json`` — kaggle-style replays where
``steps[i] = [agent0, agent1]`` and each agent carries an ``observation`` (with a
``search_begin_input`` blob) plus the ``action`` it took. The step-1 actions are
the 60-card deck submissions, so BOTH decks are fully recoverable, which lets us
reconstruct each player's hidden-info predictions for ``search_begin``.

A *target* = a decision point where the eventual loser (reward == -1) was to act
and the position is mid-game (result == -1) with a valid blob. We inject that
exact position via ``ScenarioSpec`` and ask the Solver to win from it.

Hidden-info prediction is APPROXIMATE: we know each deck's full multiset and
subtract the visible cards, then partition the remaining hidden pool into
deck / prize / hand / face-down-active slots to match the observed counts. The
engine only requires counts to match and ids to be valid; the exact identities
of face-down cards are a belief, which is exactly what ``search_begin`` models.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rl.config import LOSER_DIR
from rl.engine.scenario import ScenarioSpec


def _deck_from_step1(replay: dict, agent: int) -> list[int] | None:
    steps = replay.get("steps") or []
    if len(steps) < 2:
        return None
    act = steps[1][agent].get("action")
    if isinstance(act, list) and len(act) == 60:
        return list(act)
    # Some replays submit the deck at step 0.
    act0 = steps[0][agent].get("action")
    if isinstance(act0, list) and len(act0) == 60:
        return list(act0)
    return None


def _pokemon_ids() -> set[int]:
    """Lazily load the set of Pokémon card ids (for a legal face-down active)."""
    try:
        data = json.load(open(Path(LOSER_DIR).parent / "cards.json"))
        return {c["id"] for c in data["cards"] if c.get("cardType") == 0}
    except Exception:
        return set()


def _visible_ids(player_state: dict, include_hand: bool) -> Counter:
    """All known card ids for one player: board (+energy/tools/pre-evo), discard,
    stadium-agnostic, and hand if visible."""
    c = Counter()

    def add_poke(p):
        if not p:
            return
        c[p["id"]] += 1
        for grp in ("energyCards", "tools", "preEvolution"):
            for card in (p.get(grp) or []):
                c[card["id"]] += 1

    for p in (player_state.get("active") or []):
        add_poke(p)
    for p in (player_state.get("bench") or []):
        add_poke(p)
    for card in (player_state.get("discard") or []):
        c[card["id"]] += 1
    if include_hand:
        for card in (player_state.get("hand") or []):
            c[card["id"]] += 1
    # Known (face-up) prizes, if any.
    for card in (player_state.get("prize") or []):
        if card is not None:
            c[card["id"]] += 1
    return c


def _hidden_pool(full_deck: list[int], visible: Counter) -> list[int]:
    pool = Counter(full_deck)
    pool.subtract(visible)
    out = []
    for cid, n in pool.items():
        if n > 0:
            out.extend([cid] * n)
    return out


def _take(pool: list[int], n: int) -> list[int]:
    """Pop n ids off the pool (pads by repeating the last id if short)."""
    if n <= 0:
        return []
    if len(pool) >= n:
        chunk = pool[:n]
        del pool[:n]
        return chunk
    chunk = list(pool)
    pad = pool[-1] if pool else 0
    pool.clear()
    return chunk + [pad] * (n - len(chunk))


def replay_to_targets(path: str | Path) -> list[ScenarioSpec]:
    replay = json.load(open(path))
    rewards = replay.get("rewards") or [0, 0]
    loser = 1 if rewards[0] >= rewards[1] else 0     # reward == -1 side
    deck_me = _deck_from_step1(replay, loser)
    deck_op = _deck_from_step1(replay, 1 - loser)
    if deck_me is None or deck_op is None:
        return []
    poke_ids = _pokemon_ids()

    specs: list[ScenarioSpec] = []
    steps = replay.get("steps") or []
    for i, step in enumerate(steps):
        ag = step[loser]
        if ag.get("status") != "ACTIVE":
            continue
        obs = ag.get("observation")
        if not isinstance(obs, dict) or obs.get("select") is None:
            continue
        cur = obs.get("current")
        if not cur or cur.get("result", -1) != -1 or not obs.get("search_begin_input"):
            continue
        my_index = cur["yourIndex"]
        me, op = cur["players"][my_index], cur["players"][1 - my_index]

        my_pool = _hidden_pool(deck_me, _visible_ids(me, include_hand=True))
        op_pool = _hidden_pool(deck_op, _visible_ids(op, include_hand=False))

        # Partition my hidden pool -> deck then prize.
        deck_sel = (obs.get("select") or {}).get("deck")
        your_deck = [] if deck_sel is not None else _take(my_pool, me["deckCount"])
        your_prize = _take(my_pool, len(me["prize"]))

        # Opp face-down active must be a Pokémon id.
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

        spec = ScenarioSpec(
            obs=obs, your_deck=your_deck, your_prize=your_prize,
            opponent_deck=opp_deck, opponent_prize=opp_prize,
            opponent_hand=opp_hand, opponent_active=opp_active,
            source=f"{Path(path).name}#step{i}",
            target_id=f"{Path(path).stem}_{i}",
        )
        try:
            spec.validate_shapes()
            specs.append(spec)
        except ValueError:
            continue
    return specs


def build_target_set(loser_dir: str | Path = LOSER_DIR) -> list[ScenarioSpec]:
    """Parse every lost_*.json into the fixed target set D."""
    targets: list[ScenarioSpec] = []
    for p in sorted(Path(loser_dir).glob("lost_*.json")):
        targets.extend(replay_to_targets(p))
    return targets


if __name__ == "__main__":
    D = build_target_set()
    print(f"built {len(D)} targets from {LOSER_DIR}")
    for t in D[:10]:
        print(" ", t.target_id, "|", t.source)
