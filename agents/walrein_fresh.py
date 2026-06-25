"""Walrein — FRESH, tunable pilot (Kimiaki's rank-6 list + the current v23 strategy, logged).

Clean reimplementation of the existing Walrein strategy so it's easy to read and tune. Every
override logs a `STRAT| T<turn> <move> — <why>` line to stderr, so you can watch what it decides
(and see which lever fires where). Deck = Kimiaki's rank-6 Walrein (deck_kimiaki.json).

STRATEGY (each block is an independently-tunable lever; KNOBS marked):
  L1 RARE-CANDY     — force Rare Candy (Spheal -> Walrein, skip Sealeo) to land the Stage-2 fast.
  L2 RARE-CANDY-DIG — force RC even when Walrein isn't in hand yet (RC tutors it from deck).
  L3 DAWN           — force Dawn (1231) to assemble the Walrein line when incomplete.
  L4 ATTACK GATE    — two-mode Frigid Fangs / Megaton Fall control:
       FORCE mode  vs TEMPO_OHKO_IDS (Lucario/Cinderace ex/Mega Starmie ex): attack EVERY turn (lock).
       SWAP  mode  vs WEAK_OPP_IDS (dragapult/bolt/hydrapple/...): only nudge bare_agent's attack pick.
       want = Fangs (survive a <=2-energy OHKO) | Megaton (race once opp has 3+ energy or we have backup).
  L5 BENCH-BACKUP   — bench a Spheal so a Walrein KO doesn't end the game (unless badly behind).

  >>> KNOWN GAP (not yet implemented — the diagnosed weakness): NO Enhanced/Crushing Hammer denial.
      The deck runs 4 Enhanced Hammer + 2 Crushing Hammer and the pilot never plays them. Add an
      L0 HAMMER lever here to strip the opponent's special/attached energy — that's the #1 tune. <<<
"""
import os
import sys

os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_kimiaki.json"))
import agents.bare_agent as B
from agents.bare_agent import to_observation_class, get_card, AreaType, OptionType

my_deck = B.my_deck

SPHEAL, SEALEO, WALREIN = 941, 942, 943
RARE_CANDY, DAWN = 1079, 1231
ENHANCED_HAMMER, CRUSHING_HAMMER = 1081, 1120     # <- the unused denial tools (KNOB for L0)

# KNOB: ex attackers where Walrein FORCES the Fangs/Megaton lock every turn (200+ OHKO threats).
TEMPO_OHKO_IDS = {678, 153, 1031}                 # Mega Lucario ex / Cinderace ex / Mega Starmie ex
# KNOB: opponents where Walrein only SWAPS bare_agent's attack pick (never forces).
WEAK_OPP_IDS = {120, 121, 918, 710, 919, 347, 150, 63, 108}

_pre_turn = -1


def _log(turn, move, why):
    sys.stderr.write(f"STRAT| T{turn:<2} {move:<16} — {why}\n"); sys.stderr.flush()


def _ids(cards):
    return [c.id for c in (cards or []) if c is not None]


def _play_idx(obs, opts, my_index, card_id):
    for i, o in enumerate(opts):
        if o.type == OptionType.PLAY:
            c = get_card(obs, AreaType.HAND, o.index, my_index)
            if c is not None and c.id == card_id:
                return i
    return None


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck
    st = obs.current
    sel = obs.select
    if sel.context != 0:                     # only steer MAIN-phase decisions
        return B.agent(obs_dict)
    chosen = B.agent(obs_dict)
    me = st.players[st.yourIndex]
    opp = st.players[1 - st.yourIndex]
    opts = sel.option or []
    turn = st.turn
    one = sel.minCount <= 1

    in_play = _ids((list(me.active) + list(me.bench)))
    hand = _ids(me.hand)
    spheal_in_play = SPHEAL in in_play
    walrein_in_play = WALREIN in in_play
    walrein_in_hand = WALREIN in hand
    walrein_seen = walrein_in_play or WALREIN in _ids(me.discard)

    # ── L1 RARE CANDY (Spheal + Walrein in hand → skip to Stage 2) ──
    if one and spheal_in_play and walrein_in_hand and RARE_CANDY in hand and not walrein_in_play:
        i = _play_idx(obs, opts, st.yourIndex, RARE_CANDY)
        if i is not None:
            _log(turn, "Rare Candy", "Spheal+Walrein in hand → skip Sealeo, land Walrein now"); return [i]

    # ── L2 RARE-CANDY-DIG (RC tutors Walrein from deck even when not in hand) ──
    if one and spheal_in_play and RARE_CANDY in hand and not walrein_seen and not walrein_in_hand:
        i = _play_idx(obs, opts, st.yourIndex, RARE_CANDY)
        if i is not None:
            _log(turn, "Rare Candy(dig)", "Spheal up, Walrein unseen → RC to fetch the Stage-2"); return [i]

    # ── L3 DAWN (assemble the Walrein line when incomplete) ──
    line_complete = walrein_in_play or (walrein_in_hand and spheal_in_play)
    if one and not line_complete and DAWN in hand:
        i = _play_idx(obs, opts, st.yourIndex, DAWN)
        if i is not None:
            _log(turn, "Dawn", "Walrein line incomplete → Dawn to tutor the pieces"); return [i]

    # ── L4 ATTACK GATE (two-mode Frigid Fangs / Megaton Fall) ──
    active_is_walrein = bool(me.active and me.active[0] and me.active[0].id == WALREIN)
    oa = opp.active[0] if (opp.active and opp.active[0]) else None
    op_ids = ([oa.id] if oa else []) + _ids(opp.bench)
    opp_best = 0
    ocd = B.card_table.get(oa.id) if oa else None
    if ocd:
        for aid in (ocd.attacks or []):
            atk = B.attack_table.get(aid) if hasattr(B, "attack_table") else None
            d = (getattr(atk, "damage", 0) or 0) if atk else 0
            if getattr(ocd, "energyType", None) == 8: d *= 2   # Walrein weak = Metal
            opp_best = max(opp_best, d)
    wal_hp = me.active[0].hp if (me.active and me.active[0]) else 170
    opp_can_ohko = opp_best >= wal_hp
    force_mode = active_is_walrein and (oa.id in TEMPO_OHKO_IDS if oa else False)
    swap_mode = (not force_mode) and active_is_walrein and any(i in WEAK_OPP_IDS for i in op_ids)

    if force_mode or swap_mode:
        atk_idx = {}
        for i, o in enumerate(opts):
            if o.type == OptionType.ATTACK:
                a = B.attack_table.get(getattr(o, "attackId", None)) if hasattr(B, "attack_table") else None
                nm = getattr(a, "name", "") if a else ""
                if nm in ("Frigid Fangs", "Megaton Fall"): atk_idx[nm] = i
        if atk_idx:
            op_energy = len(oa.energies) if (oa and hasattr(oa, "energies")) else 0
            backup = any(c is not None and c.id in (SPHEAL, SEALEO, WALREIN) for c in (me.bench or []))
            if opp_can_ohko and op_energy <= 2:   want = "Frigid Fangs"   # survive the lock turn
            elif op_energy >= 3:                  want = "Megaton Fall"   # lock dead → race
            elif wal_hp > 50 and backup:          want = "Megaton Fall"   # afford recoil, 2HKO clock
            else:                                 want = "Frigid Fangs"   # stall / develop
            if want in atk_idx:
                wi = atk_idx[want]
                if force_mode:
                    _log(turn, want, f"FORCE vs tempo-OHKO (opp {op_energy}e, walrein {wal_hp}hp)"); return [wi]
                bare_atk = next((idx for idx in chosen if 0 <= idx < len(opts)
                                 and opts[idx].type == OptionType.ATTACK), None)
                if bare_atk is not None and bare_atk != wi:
                    _log(turn, want, "SWAP bare's attack pick"); chosen = [wi if x == bare_atk else x for x in chosen]

    # ── L5 BENCH-BACKUP (keep a Walrein-line body so a KO doesn't end the game) ──
    bench_line = any(c is not None and c.id in (SPHEAL, SEALEO, WALREIN) for c in (me.bench or []))
    behind = (6 - len(me.prize)) >= (6 - len(opp.prize)) + 2
    if one and active_is_walrein and not bench_line and not behind:
        i = _play_idx(obs, opts, st.yourIndex, SPHEAL)
        if i is not None and i not in chosen:
            _log(turn, "bench Spheal", "no Walrein-line backup on bench → secure a 2nd body"); return [i]

    return chosen
