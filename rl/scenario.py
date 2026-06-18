"""Scenario specs + edit-scripts compiled into ``search_begin`` kwargs.

A *scenario* is a concrete mid-game position the Solver is asked to win from. It
is loaded into the engine with ``cg.api.search_begin``, which takes a real
serialized position (the ``search_begin_input`` blob carried on every
observation) plus the agent's *predictions of the hidden information*:

    your_deck, your_prize,
    opponent_deck, opponent_prize, opponent_hand, opponent_active

IMPORTANT ENGINE CONSTRAINT (verified against cg/api.py:517):
``search_begin`` reconstructs the *actual board* (active/bench/energy/damage)
from the opaque blob. The ONLY things a caller can vary are the six hidden-info
id-lists above. So an "edit-script" here mutates *predictions*, not the board:

  * reduce-threat / isolate-tactic : choose what the opponent is *predicted* to
    hold (a weak hand / no gust) or which Pokémon sits face-down in their active
    (``opponent_active`` — legal only when their active is face-down).
  * partial-credit / fix-draw      : choose the order of *our* deck/prizes so the
    next draws favour the line we want the Solver to learn.

Board-state rewrites from the plan's wish-list ("prune opponent bench to 1",
"pre-attach one extra energy") are NOT expressible through ``search_begin`` and
are intentionally out of scope for v1 — they would require editing the opaque
blob, which the engine does not support. ``EditOp`` rejects them explicitly so
the limitation is loud, not silent.

This module is pure-Python (no ``cg`` import) so it parses on any host; the
actual ``search_begin`` call lives in ``env.py``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal


# --- Scenario spec ----------------------------------------------------------
@dataclass
class ScenarioSpec:
    """A loadable mid-game position + the baseline hidden-info predictions."""
    obs: dict                       # full observation dict incl. "search_begin_input"
    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]      # only non-empty when opp active is face-down
    manual_coin: bool = False
    source: str = "unknown"         # provenance: lost_3.json#step12, selfplay, turn0, ...
    target_id: str = ""             # stable id used to track solve-rate on D

    @property
    def my_index(self) -> int:
        return self.obs["current"]["yourIndex"]

    def required_lengths(self) -> dict[str, int]:
        """Lengths the six prediction vectors must satisfy (from the obs)."""
        st = self.obs["current"]
        me = st["players"][self.my_index]
        op = st["players"][1 - self.my_index]
        deck_sel = (self.obs.get("select") or {}).get("deck")
        return {
            "your_deck": 0 if deck_sel is not None else me["deckCount"],
            "your_prize": len(me["prize"]),
            "opponent_deck": op["deckCount"],
            "opponent_prize": len(op["prize"]),
            "opponent_hand": op["handCount"],
            "opponent_active": _facedown_active_count(op),
        }

    def search_begin_kwargs(self) -> dict:
        return dict(
            your_deck=list(self.your_deck),
            your_prize=list(self.your_prize),
            opponent_deck=list(self.opponent_deck),
            opponent_prize=list(self.opponent_prize),
            opponent_hand=list(self.opponent_hand),
            opponent_active=list(self.opponent_active),
            manual_coin=self.manual_coin,
        )

    def validate_shapes(self) -> None:
        req = self.required_lengths()
        for name, n in req.items():
            got = len(getattr(self, name))
            if got < n:
                raise ValueError(
                    f"{name}: need >= {n} predicted ids, got {got} "
                    f"(scenario {self.target_id or self.source})")


def _facedown_active_count(player_state: dict) -> int:
    active = player_state.get("active") or []
    return 1 if (len(active) > 0 and active[0] is None) else 0


# --- Edit-script ------------------------------------------------------------
EditKind = Literal[
    "set_opponent_active",   # predict the face-down opp active is card_id X
    "stack_your_deck_top",   # move predicted card_id X to the top of our deck
    "set_opponent_hand",     # replace a slot of opp's predicted hand with card_id X
    "weaken_opponent_hand",  # fill opp predicted hand with a benign card_id
]


@dataclass
class EditOp:
    kind: EditKind
    card_id: int = 0
    slot: int = 0

    def __post_init__(self):
        if self.kind not in (
            "set_opponent_active", "stack_your_deck_top",
            "set_opponent_hand", "weaken_opponent_hand",
        ):
            raise ValueError(
                f"Unsupported edit '{self.kind}'. Board-state edits are not "
                f"expressible through search_begin; see scenario.py docstring.")


@dataclass
class EditScript:
    ops: list[EditOp] = field(default_factory=list)
    budget: int = 4

    def size(self) -> int:
        return len(self.ops)

    def apply(self, spec: ScenarioSpec) -> ScenarioSpec:
        """Return a NEW ScenarioSpec with the edits applied to its predictions.

        Edits beyond ``budget`` are dropped (the Guide's relevance term already
        penalises large scripts; this is a hard backstop).
        """
        s = ScenarioSpec(
            obs=spec.obs,
            your_deck=list(spec.your_deck),
            your_prize=list(spec.your_prize),
            opponent_deck=list(spec.opponent_deck),
            opponent_prize=list(spec.opponent_prize),
            opponent_hand=list(spec.opponent_hand),
            opponent_active=list(spec.opponent_active),
            manual_coin=spec.manual_coin,
            source=spec.source,
            target_id=spec.target_id,
        )
        for op in self.ops[: self.budget]:
            _apply_one(op, s)
        return s


def _apply_one(op: EditOp, s: ScenarioSpec) -> None:
    if op.kind == "set_opponent_active":
        if s.opponent_active:           # only legal when active is face-down
            s.opponent_active[0] = op.card_id
    elif op.kind == "stack_your_deck_top":
        if op.card_id in s.your_deck:
            s.your_deck.remove(op.card_id)
            s.your_deck.insert(0, op.card_id)
    elif op.kind == "set_opponent_hand":
        if 0 <= op.slot < len(s.opponent_hand):
            s.opponent_hand[op.slot] = op.card_id
    elif op.kind == "weaken_opponent_hand":
        for i in range(len(s.opponent_hand)):
            s.opponent_hand[i] = op.card_id


def turn0_scenario(your_deck: list[int], opponent_deck: list[int]) -> None:
    """Placeholder: turn-0 games don't need search_begin (use battle_start).

    Kept as an explicit no-op marker so callers branch clearly: a turn-0
    scenario is loaded by ``env.reset(scenario=None)`` via ``battle_start``,
    while a mid-game scenario is loaded via ``search_begin``.
    """
    return None
