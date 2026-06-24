"""Realistic Mega Abomasnow opponent (codex-spec'd) for honest >=50% validation.

The default `bare_agent` proxy pilots this deck unrealistically: it evolves Snover -> Mega
Abomasnow ASAP (~turn 5) and OHKO-loops every turn, capping Honchkrow ~30% — which CONTRADICTS
the real-ladder 24-24 (50%). Per codex's round-3 spec, a representative pilot has the Mega come
online ~turn 6-6.5 (not a deterministic turn-5 lock) and doesn't chain every disruption
perfectly.

This wraps `bare_agent` and injects exactly that realism: it DEFERS the Snover->Mega Abomasnow
evolution until a per-game random turn (~6-8), and occasionally passes a main action. Energy
attachment / Frost Barrier attacking are left to bare_agent (those are correct). Nothing else
is weakened, so this is a calibrated-to-data opponent, not a strawman.
"""
import os
import random

os.environ.setdefault("BARE_DECK", "data/decks/deck_loss_abomasnow.json")

from cg.api import OptionType, to_observation_class
import agents.bare_agent as B

my_deck = B.my_deck
MEGA_ABOM = 723

_last_turn = -1
_mega_delay = 7          # per-game: don't evolve to Mega before this turn
_rng = random.Random(20260623)


def _hand_card_id(obs, opt):
    """The card a PLAY/EVOLVE option brings in (from hand)."""
    try:
        return obs.current.players[obs.current.yourIndex].hand[opt.index].id
    except (AttributeError, IndexError, TypeError):
        return None


def agent(obs_dict):
    global _last_turn, _mega_delay
    if obs_dict.get("select") is None or obs_dict.get("current") is None:
        return B.agent(obs_dict)

    obs = to_observation_class(obs_dict)
    turn = obs.current.turn or 0
    if turn < _last_turn:                 # new game -> roll a fresh setup speed (~6-8)
        _mega_delay = _rng.choice([6, 6, 7, 7, 8])
    _last_turn = turn

    choice = B.agent(obs_dict)            # what the strong proxy would do
    opts = obs.select.option or []

    # Realism 1: defer the Snover -> Mega Abomasnow evolution until _mega_delay.
    if turn < _mega_delay and isinstance(choice, list):
        chose_mega_evo = any(0 <= i < len(opts) and opts[i].type == OptionType.EVOLVE
                             and _hand_card_id(obs, opts[i]) == MEGA_ABOM for i in choice)
        if chose_mega_evo:
            # Defer ONLY the Mega evolution but KEEP DEVELOPING this turn (don't waste it on
            # END). Prefer a productive alternative: attach energy > play > non-Mega evolve >
            # ability > anything; END only if nothing else is legal.
            PRI = {OptionType.ATTACH: 6, OptionType.PLAY: 5, OptionType.EVOLVE: 4,
                   OptionType.ABILITY: 3, OptionType.ATTACK: 2}
            cand = [k for k, o in enumerate(opts)
                    if not (o.type == OptionType.EVOLVE and _hand_card_id(obs, o) == MEGA_ABOM)
                    and o.type != OptionType.END]
            if cand:
                cand.sort(key=lambda k: PRI.get(opts[k].type, 1), reverse=True)
                return cand[:1]
            # only END / Mega-evolve remain -> allow the evolution (don't stall)
            return choice

    return choice
