"""
Semantic event schema for Pokémon TCG video logs, plus an adapter that maps
onto the `observation.logs` vocabulary used by the Kaggle `cabt` episode format.

Design note (read this):
  The SEMANTIC layer (EventType / Zone / SemanticEvent) is the source of truth.
  It is grounded in real TCG semantics and is what the LLM extractor produces.

  The CABT adapter (`to_cabt_log`) is a *best-effort, lossy* projection onto the
  numeric `type`/`area` codes observed in a single real replay (lost_1.json).
  Those numeric codes were reverse-engineered from one file, so treat the mapping
  as provisional and calibrate it if you get more replays. Anything without a
  clean correspondence is preserved in the semantic payload and emitted as a
  passthrough note rather than forced into a wrong code.

  Reminder: a video can NEVER reconstruct `action`, `observation.select`, or
  `search_begin_input` — those are engine-internal option indices. This module
  only fills the `logs` portion of the episode.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Zone(str, Enum):
    DECK = "deck"
    HAND = "hand"
    ACTIVE = "active"
    BENCH = "bench"
    DISCARD = "discard"
    PRIZE = "prize"
    STADIUM = "stadium"
    LOST_ZONE = "lost_zone"
    ATTACHED = "attached"     # energy/tool attached to a Pokémon
    UNKNOWN = "unknown"


class EventType(str, Enum):
    # setup / housekeeping
    SETUP = "setup"
    MULLIGAN = "mulligan"
    DRAW = "draw"
    SHUFFLE = "shuffle"
    SEARCH = "search"
    DISCARD = "discard"
    # board development
    PLAY_BASIC = "play_basic"
    EVOLVE = "evolve"
    ATTACH_ENERGY = "attach_energy"
    ATTACH_TOOL = "attach_tool"
    PLAY_SUPPORTER = "play_supporter"
    PLAY_ITEM = "play_item"
    PLAY_STADIUM = "play_stadium"
    ABILITY = "ability"
    # positioning
    RETREAT = "retreat"
    SWITCH = "switch"
    # combat
    ATTACK = "attack"
    DAMAGE = "damage"
    HEAL = "heal"
    STATUS = "status"          # asleep/confused/paralyzed/burned/poisoned
    COIN_FLIP = "coin_flip"
    KNOCKOUT = "knockout"
    TAKE_PRIZE = "take_prize"
    # control flow
    END_TURN = "end_turn"
    GAME_RESULT = "game_result"
    NOTE = "note"              # caster commentary that isn't a hard game action


class SemanticEvent(BaseModel):
    """One grounded game event extracted from the transcript."""
    turn: int = Field(..., ge=0, description="1-indexed turn; 0 = pre-game setup")
    player: int = Field(..., ge=0, le=1, description="acting player index (0 or 1)")
    type: EventType
    t_start: Optional[float] = Field(None, description="transcript timestamp (s)")

    card: Optional[str] = None
    target: Optional[str] = None
    from_zone: Optional[Zone] = None
    to_zone: Optional[Zone] = None

    amount: Optional[int] = None          # damage / heal / draw count / prizes
    energy_type: Optional[str] = None
    attack_name: Optional[str] = None
    status: Optional[str] = None
    coin: Optional[str] = None            # "heads" / "tails"
    winner: Optional[int] = None          # only for GAME_RESULT

    confidence: float = Field(0.5, ge=0.0, le=1.0)
    source_text: Optional[str] = Field(None, description="transcript snippet")

    @field_validator("card", "target", "attack_name", "energy_type", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) and v.strip() else None

    def dedupe_key(self) -> tuple:
        """Identity used to drop duplicates from overlapping transcript windows."""
        return (self.turn, self.player, self.type.value,
                self.card, self.target, self.attack_name, self.amount)


# --------------------------------------------------------------------------- #
# CABT log adapter (provisional — calibrate against more replays)
# --------------------------------------------------------------------------- #
# Numeric area codes are a guess from lost_1.json (area 2 appeared for hand-like
# selection). Kept centralized so it's a one-line fix once calibrated.
_AREA = {
    Zone.DECK: 1,
    Zone.HAND: 2,
    Zone.ACTIVE: 4,
    Zone.BENCH: 5,
    Zone.DISCARD: 6,
    Zone.PRIZE: 7,
    Zone.STADIUM: 8,
    Zone.LOST_ZONE: 9,
    Zone.ATTACHED: 10,
    Zone.UNKNOWN: 0,
}

# cabt `type` codes observed in lost_1.json, with our reading of each:
#   4 reveal-card{cardId,serial}  6 move{cardId,from,to,serial}
#   7 move-hidden{from,to}        8 swap-active/bench
#   15 attack{attackId,cardId,serial}   16 damage{cardId,putDamageCounter,value,serial}
_MOVE_LIKE = {
    EventType.DRAW: (Zone.DECK, Zone.HAND),
    EventType.PLAY_BASIC: (Zone.HAND, Zone.BENCH),
    EventType.EVOLVE: (Zone.HAND, Zone.ACTIVE),
    EventType.ATTACH_ENERGY: (Zone.HAND, Zone.ATTACHED),
    EventType.ATTACH_TOOL: (Zone.HAND, Zone.ATTACHED),
    EventType.PLAY_SUPPORTER: (Zone.HAND, Zone.DISCARD),
    EventType.PLAY_ITEM: (Zone.HAND, Zone.DISCARD),
    EventType.PLAY_STADIUM: (Zone.HAND, Zone.STADIUM),
    EventType.DISCARD: (Zone.HAND, Zone.DISCARD),
    EventType.RETREAT: (Zone.ACTIVE, Zone.BENCH),
    EventType.TAKE_PRIZE: (Zone.PRIZE, Zone.HAND),
}


def to_cabt_log(ev: SemanticEvent) -> Dict[str, Any]:
    """Project a SemanticEvent onto a cabt-style log dict.

    Always preserves the full semantic record under `_semantic` so nothing is
    lost when the numeric projection is approximate or absent.
    """
    base: Dict[str, Any] = {"playerIndex": ev.player, "_semantic": ev.model_dump(
        exclude_none=True, mode="json")}

    if ev.type == EventType.ATTACK:
        base.update(type=15, attackId=None, cardId=None, serial=None)
    elif ev.type in (EventType.DAMAGE, EventType.KNOCKOUT):
        base.update(type=16, cardId=None, serial=None,
                    putDamageCounter=True, value=ev.amount)
    elif ev.type == EventType.SWITCH:
        base.update(type=8, cardIdActive=None, cardIdBench=None,
                    serialActive=None, serialBench=None)
    elif ev.type in _MOVE_LIKE:
        frm = ev.from_zone or _MOVE_LIKE[ev.type][0]
        to = ev.to_zone or _MOVE_LIKE[ev.type][1]
        base.update(type=6, cardId=None, serial=None,
                    fromArea=_AREA[frm], toArea=_AREA[to])
    else:
        # STATUS, COIN_FLIP, ABILITY, GAME_RESULT, NOTE, etc. — no clean code.
        base.update(type=None)  # semantic-only; consumer reads `_semantic`
    return base


# --------------------------------------------------------------------------- #
# Episode assembly (mirrors lost_1.json top-level shape)
# --------------------------------------------------------------------------- #
class Episode(BaseModel):
    players: List[str]
    events: List[SemanticEvent]
    winner: Optional[int] = None
    source_url: Optional[str] = None
    emit_cabt: bool = True

    def build(self) -> Dict[str, Any]:
        """Return a dict shaped like a kaggle_environments episode replay.

        One step per turn. Both agent slots carry the turn's logs (video has no
        hidden information, so the per-player asymmetry in lost_1 collapses).
        `action` / `select` / `search_begin_input` are intentionally null —
        they are unreconstructable from video and are left as honest blanks.
        """
        by_turn: Dict[int, List[SemanticEvent]] = {}
        for ev in sorted(self.events, key=lambda e: (e.turn, e.t_start or 0.0)):
            by_turn.setdefault(ev.turn, []).append(ev)

        steps: List[List[Dict[str, Any]]] = []
        for turn in sorted(by_turn):
            sem = [e.model_dump(exclude_none=True, mode="json") for e in by_turn[turn]]
            cabt = [to_cabt_log(e) for e in by_turn[turn]] if self.emit_cabt else None
            acting = by_turn[turn][0].player if by_turn[turn] else None
            step = []
            for agent_idx in range(2):
                obs = {
                    "step": turn,
                    "actingPlayer": acting,
                    "logs": cabt if self.emit_cabt else sem,
                    "semanticLogs": sem,
                    # honest blanks — cannot come from video:
                    "select": None,
                    "search_begin_input": None,
                    "current": None,
                }
                step.append({
                    "action": [],            # no engine option indices exist
                    "observation": obs,
                    "reward": 0,
                    "status": "ACTIVE",
                })
            steps.append(step)

        rewards = [0, 0]
        statuses = ["DONE", "DONE"]
        if self.winner in (0, 1):
            rewards = [1 if i == self.winner else -1 for i in range(2)]

        return {
            "name": "ptcg-video",
            "description": "Semantic log reconstructed from match video (logs only).",
            "schema_version": 1,
            "source_url": self.source_url,
            "info": {
                "Agents": [{"Name": p, "ThumbnailUrl": None} for p in self.players],
                "TeamNames": list(self.players),
            },
            "configuration": {"episodeSteps": len(steps)},
            "rewards": rewards,
            "statuses": statuses,
            "steps": steps,
            "_caveats": [
                "logs-only: action/select/search_begin_input are not reconstructable from video",
                "cabt numeric type/area codes are provisional (calibrate vs real replays)",
                "this is Standard paper TCG, a different game from the cabt engine",
            ],
        }
