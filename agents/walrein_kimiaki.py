# Kimiaki's rank-6 Walrein list (deck_kimiaki) + v23 tuned pilot.
"""Walrein v22 = v21 with the force-gate TIMING FIX.

PROBLEM (measured, n=100 docker sim, seat-swapped):
  opponent            v15   v20
  meta_opp (field)    65%   30%   <- v20 CATASTROPHIC -35pt regression
  mega_lucario        59%   70%
  meta_starmie        58%   66%
  bench_starmie_cind  54%   74%
v20's broad gate `gated = (opp_best_dmg>=walrein_hp) or (op_active_hp>=200)` fires against almost
every meta EX, so v20 FORCES Fangs/Megaton every turn vs everyone — abandoning v15's develop/race/
deck-out win-con. Loss data vs meta_opp: 65/69 losses by all-prizes (out-raced, never developed a
2nd Walrein because every turn was a forced attack instead of attach/evolve).

ROOT CAUSE: forcing the attack is ONLY good against FAST OHKO TEMPO decks (Lucario/Starmie/Cinderace)
that OHKO Walrein(170hp); v15's swap-only (nudge bare_agent's attack pick, let it develop) is correct
for the general field. v21 gated force on `opp_can_ohko`, but Starmie/Cinderace need 3 energy to OHKO
— by the time opp_can_ohko is True Walrein is already gone (v21 starmie 58% / cind 54% = no gain over
v15). v20 won those (66/74%) because its `op_hp>=200` clause forced the Fangs lock PREEMPTIVELY the
moment the 320hp ex became active, BEFORE it charged. That early lock is the real win mechanism.

FIX — two-mode gate (v22 timing fix):
  FORCE mode (v20 behavior, return [want]):  opp ACTIVE id in TEMPO_OHKO_IDS (NO opp_can_ohko req —
    start the lock the instant the tempo ex is active, preemptively, like v20's hp clause did).
    Scoped to TEMPO_OHKO_IDS so the general field is untouched.
    TEMPO_OHKO_IDS = the archetype-defining OHKO ex basics that 2-3 shot the race:
      678 Mega Lucario ex (Mega Brave 270 @ 2F), 153 Cinderace ex (Flare Strike 280 @ 3),
      666 Cinderace, 1031 Mega Starmie ex (Nebula Beam 210 @ 3), 361 Misty's Starmie.
    ID-gate (not energy-gate): Nebula Beam/Flare Strike cost 3 energy, so an `energy<=2` gate (the
    planner's option B) would NOT fire when Starmie/Cinderace are ready — wrong. And a `<200hp`
    tiebreaker is wrong too (all three are 320-340hp ex). IDs are robust to list TUNING (the user's
    "tuned mega lucario" still runs 678), which an HP/damage heuristic is not.
  SWAP mode (v15 behavior, nudge only if bare already attacked):  v15's WEAK_OPP_IDS set
    {dragapult/bolt/hydrapple/etc.} — preserves the 65% field winrate.
  else: no attack override (bare_agent decides) — the safe general-field default.

`want` selection inside both modes is unchanged from v20/v15 (planner-verified sound)."""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_kimiaki.json"))
import agents.bare_agent as B
from agents.bare_agent import (
    to_observation_class, card_table, BASIC_MONS, get_card, AreaType, OptionType,
    SelectContext, Pokemon, card_type, AttackPlan,
)
from collections import defaultdict
my_deck = B.my_deck
SPHEAL, SEALEO, WALREIN = 941, 942, 943
plan = AttackPlan(); pre_turn = -1; turn_actions = 0

# Tempo OHKO EX attackers (the real 200+ dmg ex basics where the preemptive Fangs-lock wins):
#   678 Mega Lucario ex (Mega Brave 270), 153 Cinderace ex (Flare Strike 280), 1031 Mega Starmie ex
#   (Nebula Beam 210). NOT 666 Cinderace (50dmg ramp) or 361 Misty's Starmie (60dmg) — those are
#   harmless non-OHKO attackers; forcing the lock vs them wastes tempo (v22 regressed cind to 46%).
TEMPO_OHKO_IDS = {678, 153, 1031}
# v15's swap-mode set (dragapult/bolt/hydrapple/ogerpon/etc.) — swap-only, never force.
WEAK_OPP_IDS = {120, 121, 918, 710, 919, 347, 150, 63, 108}

def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None: return my_deck
    state = obs.current; select = obs.select; context = select.context
    my_index = state.yourIndex; my_state = state.players[my_index]
    global plan, pre_turn, turn_actions
    if pre_turn != state.turn: pre_turn = state.turn; plan = AttackPlan(); turn_actions = 0
    turn_actions += 1
    chosen = B.agent(obs_dict)
    RARE_CANDY = 1079
    if context == 0:  # MAIN
        op_state = state.players[1 - my_index]
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
        walrein_seen = walrein_in_play or any(c is not None and c.id == WALREIN
                                              for c in (my_state.discard or []))
        if spheal_in_play and rc_in_hand and not walrein_seen and not walrein_in_hand:
            opts = select.option or []
            for i, o in enumerate(opts):
                if o.type == OptionType.PLAY:
                    c = get_card(obs, AreaType.HAND, o.index, my_index)
                    if c is not None and c.id == RARE_CANDY:
                        return [i] if select.minCount <= 1 else [i]
        line_complete = walrein_in_play or (walrein_in_hand and spheal_in_play)
        if not line_complete and 1231 in hand_ids:
            opts = select.option or []
            for i, o in enumerate(opts):
                if o.type == OptionType.PLAY:
                    c = get_card(obs, AreaType.HAND, o.index, my_index)
                    if c is not None and c.id == 1231:
                        return [i] if select.minCount <= 1 else [i]
        # ---- closed-loop attack override (v21 two-mode gate) ----
        op_active_card = op_state.active[0] if (op_state.active and op_state.active[0]) else None
        op_all = ([op_active_card] if op_active_card else []) + \
                 [c for c in (op_state.bench or []) if c is not None]
        op_all_ids = [c.id for c in op_all]
        op_active_id = op_active_card.id if op_active_card else None
        opp_best_dmg_gate = 0
        _ocd = B.card_table.get(op_active_card.id) if (op_active_card and hasattr(B, "card_table")) else None
        if _ocd:
            for oa in (_ocd.attacks or []):
                _oatk = B.attack_table.get(oa)
                if _oatk:
                    d = _oatk.damage or 0
                    if getattr(_ocd, "energyType", None) == 8:   # Walrein weak=Metal(8)
                        d *= 2
                    if d > opp_best_dmg_gate: opp_best_dmg_gate = d
        walrein_hp_now = my_state.active[0].hp if (my_state.active and my_state.active[0]) else 170
        opp_can_ohko = opp_best_dmg_gate >= walrein_hp_now
        active_is_walrein = bool(my_state.active and my_state.active[0] and my_state.active[0].id == WALREIN)
        # v22 two-mode gate: force the INSTANT a tempo ex is active (preemptive lock, like v20's
        # op_hp>=200 clause), scoped to TEMPO_OHKO_IDS so the field is untouched. opp_can_ohko stays
        # in the `want` selection below (Fangs to survive vs Megaton to race) but NOT in the gate.
        force_mode = active_is_walrein and (op_active_id in TEMPO_OHKO_IDS)
        swap_mode = (not force_mode) and active_is_walrein and any(i in WEAK_OPP_IDS for i in op_all_ids)
        if force_mode or swap_mode:
            opts = select.option or []
            atk_idx = {}
            for i, o in enumerate(opts):
                if o.type == OptionType.ATTACK:
                    aid = getattr(o, "attackId", None)
                    a = B.attack_table.get(aid) if hasattr(B, "attack_table") else None
                    nm = getattr(a, "name", "") if a else ""
                    if nm in ("Frigid Fangs", "Megaton Fall"):
                        atk_idx[nm] = i
            if atk_idx:
                _oa = op_state.active[0] if op_state.active else None
                op_active_energy = len(_oa.energies) if (_oa and hasattr(_oa, "energies")) else 0
                walrein_hp = walrein_hp_now
                backup_on_bench = any(c is not None and c.id in (SPHEAL, SEALEO, WALREIN)
                                      for c in (my_state.bench or []))
                want = None
                if opp_can_ohko and op_active_energy <= 2:
                    want = "Frigid Fangs"
                elif op_active_energy >= 3:
                    want = "Megaton Fall"
                elif walrein_hp > 50 and backup_on_bench:
                    want = "Megaton Fall"
                else:
                    want = "Frigid Fangs"
                if want in atk_idx:
                    want_i = atk_idx[want]
                    if force_mode:
                        # v20: FORCE the attack every turn (hold the lock vs tempo OHKO).
                        return [want_i]
                    else:
                        # v15: SWAP only — nudge bare's pick if it already chose to attack.
                        bare_atk = None
                        for idx in chosen:
                            if 0 <= idx < len(opts) and opts[idx].type == OptionType.ATTACK:
                                bare_atk = idx; break
                        if bare_atk is not None and bare_atk != want_i:
                            chosen = [want_i if idx == bare_atk else idx for idx in chosen]

        active_is_walrein = bool(my_state.active and my_state.active[0] and my_state.active[0].id == WALREIN)
        bench_has_line = any(c is not None and c.id in (SPHEAL, SEALEO, WALREIN)
                             for c in (my_state.bench or []))
        op_state = state.players[1 - my_index]
        my_prizes_lost = 6 - len(my_state.prize)
        opp_prizes_lost = 6 - len(op_state.prize)
        behind = my_prizes_lost >= opp_prizes_lost + 2
        if active_is_walrein and not bench_has_line and not behind:
            opts = select.option or []
            spheal_play_idx = None
            for i, o in enumerate(opts):
                if o.type == OptionType.PLAY:
                    c = get_card(obs, AreaType.HAND, o.index, my_index)
                    if c is not None and c.id == SPHEAL:
                        spheal_play_idx = i; break
            if spheal_play_idx is not None and spheal_play_idx not in chosen:
                return [spheal_play_idx] if select.minCount <= 1 else chosen
    return chosen
