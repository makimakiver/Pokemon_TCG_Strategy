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
    # v6 RARE CANDY PRIORITY: force-play Rare Candy (1079) to skip Sealeo (Spheal->Walrein direct).
    # bare_agent NEVER uses it (0 plays / 652 opportunities). Lands Walrein ~3 turns earlier.
    RARE_CANDY = 1079
    if context == 0:  # MAIN
        op_state = state.players[1 - my_index]   # v15: needed by closed-loop override + reserve gate
        hand_ids = [c.id for c in (my_state.hand or [])]
        spheal_in_play = any(c is not None and c.id == SPHEAL for c in (list(my_state.active) + list(my_state.bench)))
        walrein_in_hand = WALREIN in hand_ids
        rc_in_hand = RARE_CANDY in hand_ids
        walrein_in_play = any(c is not None and c.id == WALREIN for c in (list(my_state.active) + list(my_state.bench)))
        if spheal_in_play and walrein_in_hand and rc_in_hand and not walrein_in_play:
            opts = select.option or []
            for i, o in enumerate(opts):
                if o.type == OptionType.PLAY:
                    c = get_card(obs, AreaType.HAND, o.index, my_index)
                    if c is not None and c.id == RARE_CANDY:
                        return [i] if select.minCount <= 1 else [i]
        # v12 DAWN FORCE-PLAY ONLY: Dawn (1231) tutors Basic+Stage1+Stage2 -> assembles Walrein line.
        # (v11 showed Dawn lifts hydrapple/zoroark/ogerapon without regressing bolt, unlike Crushing.)
        line_complete = walrein_in_play or (walrein_in_hand and spheal_in_play)
        if not line_complete and 1231 in hand_ids:
            opts = select.option or []
            for i, o in enumerate(opts):
                if o.type == OptionType.PLAY:
                    c = get_card(obs, AreaType.HAND, o.index, my_index)
                    if c is not None and c.id == 1231:
                        return [i] if select.minCount <= 1 else [i]
        # v15 CLOSED-LOOP Fangs/Megaton (planner spec), gated to weak matchups {dragapult,bolt,hydrapple}.
        # Override bare's ATTACK pick when our active is Walrein in a gated matchup.
        WEAK_OPP_IDS = {120, 121, 918, 710, 919, 347, 150, 63, 108}  # distinctive attackers
        op_all_ids = [c.id for c in ([op_state.active[0]] if op_state.active and op_state.active[0] else []) +
                      [c for c in (op_state.bench or []) if c is not None]]
        gated = any(i in WEAK_OPP_IDS for i in op_all_ids)
        active_is_walrein = bool(my_state.active and my_state.active[0] and my_state.active[0].id == WALREIN)
        if gated and active_is_walrein:
            # find ATTACK options (type 13) in MAIN, map index -> attackId -> name
            opts = select.option or []
            atk_idx = {}  # name -> option index
            for i, o in enumerate(opts):
                if o.type == OptionType.ATTACK:
                    aid = getattr(o, "attackId", None)
                    a = B.attack_table.get(aid) if hasattr(B, "attack_table") else None
                    nm = getattr(a, "name", "") if a else ""
                    if nm in ("Frigid Fangs", "Megaton Fall"):
                        atk_idx[nm] = i
            if atk_idx:
                # compute closed-loop state
                _oa = op_state.active[0] if op_state.active else None
                op_active_energy = len(_oa.energies) if (_oa and hasattr(_oa, "energies")) else 0
                walrein_hp = my_state.active[0].hp if (my_state.active and my_state.active[0]) else 170
                # opp_can_OHKO_walrein: opp active's best attack damage (apply weakness to Walrein=weak Metal)
                opp_best_dmg = 0
                _ocd = B.card_table.get(_oa.id) if (_oa and hasattr(B, "card_table")) else None
                if _ocd:
                    for oa in (_ocd.attacks or []):
                        _oatk = B.attack_table.get(oa)
                        if _oatk:
                            d = _oatk.damage or 0
                            # Walrein weak=Metal(8); apply x2 only if opp energyType==Metal
                            if getattr(_ocd, "energyType", None) == 8:
                                d *= 2
                            if d > opp_best_dmg: opp_best_dmg = d
                opp_can_ohko = opp_best_dmg >= walrein_hp
                backup_on_bench = any(c is not None and c.id in (SPHEAL, SEALEO, WALREIN)
                                      for c in (my_state.bench or []))
                # CLOSED-LOOP RULE (planner spec)
                want = None
                if opp_can_ohko and op_active_energy <= 2:
                    want = "Frigid Fangs"        # prevent the KO turn
                elif op_active_energy >= 3:
                    want = "Megaton Fall"         # lock dead, race
                elif walrein_hp > 50 and backup_on_bench:
                    want = "Megaton Fall"         # afford self-dmg, start 2HKO clock
                else:
                    want = "Frigid Fangs"         # stall while developing/vulnerable
                # v18 FORCE-ATTACK: even if bare wanted to PLAY a card, force the ATTACK when the
                # lock prevents an OHKO (the logic hole fix — v15 was no-op on setup turns).
                if want in atk_idx:
                    want_i = atk_idx[want]
                    bare_atk = None
                    for idx in chosen:
                        if 0 <= idx < len(opts) and opts[idx].type == OptionType.ATTACK:
                            bare_atk = idx; break
                    if bare_atk is not None and bare_atk != want_i:
                        # swap the attack choice
                        chosen = [want_i if idx == bare_atk else idx for idx in chosen]
                    elif bare_atk is None and opp_can_ohko and op_active_energy <= 2 and want == "Frigid Fangs":
                        # bare wanted to play/attach/evolve instead of attacking, but Fangs prevents
                        # a KO this turn -> OVERRIDE: force the attack (survival > development).
                        chosen = [want_i] if select.minCount <= 1 else [want_i]

        active_is_walrein = bool(my_state.active and my_state.active[0] and my_state.active[0].id == WALREIN)
        bench_has_line = any(c is not None and c.id in (SPHEAL, SEALEO, WALREIN)
                             for c in (my_state.bench or []))
        op_state = state.players[1 - my_index]
        my_prizes_lost = 6 - len(my_state.prize)
        opp_prizes_lost = 6 - len(op_state.prize)
        # v5 PRIZE-GATE: only force the backup when we're NOT badly behind (>=2 prizes behind vs a
        # ramp deck = benching a 70hp Spheal just gifts them a prize). When behind, let bare_agent
        # play normally (don't gift the easy KO).
        behind = my_prizes_lost >= opp_prizes_lost + 2
        if active_is_walrein and not bench_has_line and not behind:
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
