"""Walrein — v4 = bare_agent + Walrein-#2 RESERVE only (isolated). No attack-bias (v1/v2/v3 all
regressed by distorting the Fangs/Megaton balance). The single non-attack lever: when our active
is Walrein, ensure a backup Walrein line (Spheal/Sealeo) is on the bench before it faints, since
Megaton self-damages and the race/deck-out wins need a 2nd body. Data: bench-backup at faint was
only 37-62% in baseline."""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_walrein.json"))
import agents.bare_agent as B
from agents.bare_agent import (
    to_observation_class, card_table, BASIC_MONS, get_card, AreaType, OptionType,
    SelectContext, Pokemon, card_type, AttackPlan,
)
from collections import defaultdict
my_deck = B.my_deck
SPHEAL, SEALEO, WALREIN = 941, 942, 943
plan = AttackPlan(); pre_turn = -1; turn_actions = 0

def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None: return my_deck
    state = obs.current; select = obs.select; context = select.context
    my_index = state.yourIndex; my_state = state.players[my_index]
    global plan, pre_turn, turn_actions
    if pre_turn != state.turn: pre_turn = state.turn; plan = AttackPlan(); turn_actions = 0
    turn_actions += 1
    # delegate the whole decision to bare_agent, then OVERRIDE only the bench-backup case
    chosen = B.agent(obs_dict)
    # If we're being asked to PLAY a Pokemon from hand and our active is Walrein with no bench
    # Walrein-line, force-bench a Spheal to secure the backup (only when minCount allows a play
    # and the option exists).
    if context == 0:  # MAIN
        active_is_walrein = bool(my_state.active and my_state.active[0] and my_state.active[0].id == WALREIN)
        bench_has_line = any(c is not None and c.id in (SPHEAL, SEALEO, WALREIN)
                             for c in (my_state.bench or []))
        if active_is_walrein and not bench_has_line:
            # find a PLAY option for a Spheal/Sealeo in hand; if the agent didn't pick it, override
            opts = select.option or []
            spheal_play_idx = None
            for i, o in enumerate(opts):
                if o.type == OptionType.PLAY:
                    c = get_card(obs, AreaType.HAND, o.index, my_index)
                    if c is not None and c.id == SPHEAL:
                        spheal_play_idx = i; break
            if spheal_play_idx is not None and spheal_play_idx not in chosen:
                # prefer the Spheal play (secures backup) — replace the top choice
                return [spheal_play_idx] if select.minCount <= 1 else chosen
    return chosen
