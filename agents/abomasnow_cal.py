"""Data-calibrated Mega Abomasnow opponent (Q1).

Replaces abomasnow_real's hand-set `_mega_delay=7` (slow Mega EVERY game) with the
empirical Mega-online distribution measured from the 48 real ladder games
(abomasnow_pack/CALIBRATION.md, generated on the mini). Key findings:

  - Among the 23/48 games where Mega was observed active, it came online at the
    opponent's OWN turn 2 (mode, 57%), median 2 — NOT turn 7.
  - 25/48 games (52%) never showed Mega active at all (short games the opp lost
    before setup + long games where Mega never reached front).

So the faithful marginal over all 48 games (opp-own-turn that Mega comes online):
    turn 2: 13   turn 3: 4   turn 4: 3   turn 7: 1   turn 8: 1   turn 9: 1
    never/late (within window): 25
This deck is therefore BIMODAL — sometimes a turn-2 monster, often no Mega — not a
uniform turn-7 setup. We sample that marginal per game.

MODELING CHOICE / CAVEAT: the 52% "never" bucket is right-censored — in the real
games it partly reflects games that ENDED early (opp lost before setup), not a
deliberate "don't evolve" policy. Modeling it as "defer Mega to a late turn" is the
faithful marginal but risks mild circularity (it bakes in some no-Mega games rather
than letting honchkrow's pressure produce them). We keep it because it reproduces the
real distribution; the conditional-only variant (always fast Mega) is the natural A/B.

Turn frame: we count the OPPONENT'S OWN decision-turns (distinct obs.current.turn
values this agent acts on), which matches the data's opp-turn numbering and is robust
to the engine's global-vs-per-player turn counter. Everything else (energy attach,
Frost Barrier / Hammer-lanche attacking) is left to bare_agent, unchanged.
"""
import os
import random

os.environ.setdefault("BARE_DECK", "data/decks/deck_loss_abomasnow.json")

from cg.api import OptionType, to_observation_class
import agents.bare_agent as B

my_deck = B.my_deck
MEGA_ABOM = 723

# Empirical marginal (opp own-turn at which Mega comes online), counts over 48 games.
# "never/late" is modeled as a far target so Mega effectively does not come online.
_NEVER = 99
_MEGA_ONLINE_POP = (
    [2] * 13 + [3] * 4 + [4] * 3 + [7] * 1 + [8] * 1 + [9] * 1 + [_NEVER] * 25
)

_rng = random.Random(20260623)
_last_raw_turn = -1          # detect new game (turn counter resets/decreases)
_last_seen_turn = None       # last distinct obs.current.turn value we acted on
_my_turn_ordinal = 0         # opponent's OWN turn count this game (1-based)
_mega_target = 7             # this game's own-turn to bring Mega online


def _hand_card_id(obs, opt):
    """The card a PLAY/EVOLVE option brings in (from hand)."""
    try:
        return obs.current.players[obs.current.yourIndex].hand[opt.index].id
    except (AttributeError, IndexError, TypeError):
        return None


def agent(obs_dict):
    global _last_raw_turn, _last_seen_turn, _my_turn_ordinal, _mega_target
    if obs_dict.get("select") is None or obs_dict.get("current") is None:
        return B.agent(obs_dict)

    obs = to_observation_class(obs_dict)
    turn = obs.current.turn or 0

    # New game -> reset own-turn counter and roll a fresh setup speed from the marginal.
    if turn < _last_raw_turn:
        _my_turn_ordinal = 0
        _last_seen_turn = None
        _mega_target = _rng.choice(_MEGA_ONLINE_POP)
    _last_raw_turn = turn

    # Count the opponent's OWN turns (each new distinct turn value we act on).
    if turn != _last_seen_turn:
        _my_turn_ordinal += 1
        _last_seen_turn = turn

    choice = B.agent(obs_dict)            # what the strong proxy would do
    opts = obs.select.option or []

    # Defer the Snover -> Mega Abomasnow evolution until our own-turn reaches the
    # sampled target (so most games go fast, ~half go late/never).
    if _my_turn_ordinal < _mega_target and isinstance(choice, list):
        chose_mega_evo = any(0 <= i < len(opts) and opts[i].type == OptionType.EVOLVE
                             and _hand_card_id(obs, opts[i]) == MEGA_ABOM for i in choice)
        if chose_mega_evo:
            # Defer ONLY the Mega evolution but KEEP DEVELOPING this turn (don't waste
            # it on END): attach energy > play > non-Mega evolve > ability > anything.
            PRI = {OptionType.ATTACH: 6, OptionType.PLAY: 5, OptionType.EVOLVE: 4,
                   OptionType.ABILITY: 3, OptionType.ATTACK: 2}
            cand = [k for k, o in enumerate(opts)
                    if not (o.type == OptionType.EVOLVE and _hand_card_id(obs, o) == MEGA_ABOM)
                    and o.type != OptionType.END]
            if cand:
                cand.sort(key=lambda k: PRI.get(opts[k].type, 1), reverse=True)
                return cand[:1]
            return choice                 # only END / Mega-evolve remain -> don't stall

    return choice
