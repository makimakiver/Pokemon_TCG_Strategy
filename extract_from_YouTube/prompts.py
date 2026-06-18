"""System + user prompt construction for the LLM event extractor."""
from __future__ import annotations
from typing import List, Dict

SYSTEM = """You convert Pokémon TCG match commentary into structured game events.

You are given a timestamped transcript window from a recorded Standard-format
(paper) Pokémon TCG feature match, plus a name->player-index map. The casters
narrate most actions. Extract ONLY grounded game events that are stated or
clearly implied. Do not invent board state you cannot justify from the text.

Output a single JSON array (no prose, no markdown fences). Each element:
{
  "turn": int,            // 1-indexed; use 0 for pre-game setup/coin flip
  "player": 0 | 1,        // acting player via the name map
  "type": one of: setup, mulligan, draw, shuffle, search, discard,
                  play_basic, evolve, attach_energy, attach_tool,
                  play_supporter, play_item, play_stadium, ability,
                  retreat, switch, attack, damage, heal, status,
                  coin_flip, knockout, take_prize, end_turn, game_result, note,
  "t_start": float,       // timestamp (seconds) of the transcript line
  "card": str|null,       // Pokémon / Trainer / Energy name if named
  "target": str|null,     // target Pokémon if relevant (e.g. attack target)
  "from_zone": one of: deck,hand,active,bench,discard,prize,stadium,lost_zone,attached,unknown | null,
  "to_zone": same enum | null,
  "amount": int|null,     // damage / heal / draw count / prizes taken
  "energy_type": str|null,
  "attack_name": str|null,
  "status": str|null,     // asleep/confused/paralyzed/burned/poisoned
  "coin": "heads"|"tails"|null,
  "winner": 0|1|null,     // only for game_result
  "confidence": float,    // 0..1, how sure you are this happened
  "source_text": str      // the transcript snippet you used
}

Rules:
- One event per discrete action. An attack that KOs and lets a player take a
  prize is THREE events: attack, knockout, take_prize.
- Track the turn number across the window using the running turn given to you.
- If the speaker is ambiguous about who acted, set confidence < 0.5.
- Prefer normalized card names (e.g. "Charizard ex", "Iono", "Ultra Ball").
- Skip pure hype/banter; emit it as type "note" only if it states board info.
- Never output anything except the JSON array.
"""


def build_user_prompt(
    window_text: str,
    players: List[str],
    running_turn: int,
    tail_context: str = "",
) -> str:
    name_map = "\n".join(f"  {i} = {n}" for i, n in enumerate(players))
    ctx = f"\nEvents already extracted just before this window (for continuity):\n{tail_context}\n" if tail_context else ""
    return (
        f"Player index map:\n{name_map}\n\n"
        f"Current running turn number: {running_turn}\n"
        f"{ctx}\n"
        f"Transcript window:\n{window_text}\n\n"
        f"Return the JSON array of events for THIS window only."
    )
