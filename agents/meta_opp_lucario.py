"""Mega Lucario ex — HARDENED ladder-realistic opponent (collision-safe, reads META_DECK).

WHY: pokemon_bench runs `--opponents` via the GENERIC meta_opp pilot, which leaves the Lucario
deck's whole gameplan on the table — it never evolves into Mega Lucario ex, never energizes it,
never plays the boost items, and so NEVER fires Mega Brave. Result: sim said we beat Lucario 77%
while the REAL ladder Lucario beats us ~70% (3W-7L). This module reproduces the real ladder line by
FORCING the engine on top of meta_opp: evolve to Mega Lucario ex -> ramp {F} energy + boost items
(Premium Power Pro / Fighting Gong) + Carmine to dig -> fire Mega Brave (270) every payable turn,
Aura Jab on the cooldown. Use as side B with META_DECK=deck_bench_megalucario.json (no BARE_DECK
collision, so it can face starmie_cind_* honestly).
"""
import os

os.environ.setdefault("META_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_bench_megalucario.json"))
import agents.meta_opp as M
from agents.meta_opp import to_observation_class, get_card, AreaType, OptionType

my_deck = M.my_deck

MEGA_LUCARIO = 678
MEGA_BRAVE = 983          # 270, 2x{F} — the payoff; can't be used 2 turns running
AURA_JAB = 982            # 130, 1x{F} — cooldown chip + {F} accel
PREMIUM_POWER_PRO = 1141  # damage-boost item
FIGHTING_GONG = 1142      # damage-boost item
CARMINE = 1192            # ramp/draw supporter (real Lucario plays it every game)
F_ENERGY = 6              # Basic {F} Energy — Mega Brave needs 2 on the active attacker
RIOLU_IDS = {333, 677, 974}


def _play_idx(obs, opts, mi, card_ids):
    for i, o in enumerate(opts):
        if o.type == OptionType.PLAY:
            c = get_card(obs, AreaType.HAND, o.index, mi)
            if c is not None and c.id in card_ids:
                return i
    return None


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck
    chosen = M.agent(obs_dict)
    sel = obs.select
    if sel.context != 0:
        return chosen
    opts = sel.option or []
    st = obs.current
    mi = st.yourIndex
    one = sel.minCount <= 1
    has_brave = any(o.type == OptionType.ATTACK and getattr(o, "attackId", None) == MEGA_BRAVE for o in opts)

    # 1. OHKO/payoff first: fire Mega Brave whenever it's a payable option.
    for i, o in enumerate(opts):
        if o.type == OptionType.ATTACK and getattr(o, "attackId", None) == MEGA_BRAVE:
            return [i]

    # 2. SET UP THE ATTACKER (the part meta_opp skips): evolve Riolu -> ... -> Mega Lucario ex ASAP.
    for i, o in enumerate(opts):
        if o.type == OptionType.EVOLVE:
            if one:
                return [i]

    # 2b. ENERGIZE THE ACTIVE (the actual missing piece — it evolved + boosted but never got 2 {F}
    #     onto the active, so Mega Brave was never payable). Force {F} energy onto an active Mega
    #     Lucario ex that still needs it.
    me = st.players[mi]
    active = me.active[0] if (me.active and me.active[0]) else None
    if one and active is not None and active.id == MEGA_LUCARIO and len(getattr(active, "energies", []) or []) < 2:
        for i, o in enumerate(opts):
            if o.type == OptionType.ATTACH and getattr(o, "inPlayArea", None) == AreaType.ACTIVE:
                c = get_card(obs, AreaType.HAND, o.index, mi)
                if c is not None and c.id == F_ENERGY:
                    return [i]

    # 3. RAMP: force the boost items + Carmine to power Mega Brave a turn earlier (the speed that
    #    wins the real ladder race — meta_opp never plays these).
    if one:
        i = _play_idx(obs, opts, mi, {PREMIUM_POWER_PRO, FIGHTING_GONG})
        if i is not None:
            return [i]
        if not has_brave and not getattr(st, "supporterPlayed", False):
            i = _play_idx(obs, opts, mi, {CARMINE})
            if i is not None:
                return [i]

    # 4. Energy attachment: meta_opp already targets its strongest attacker (Mega Lucario ex), so
    #    we let it place {F} energy — but on the cooldown turn, chip + accel with Aura Jab.
    for i, o in enumerate(opts):
        if o.type == OptionType.ATTACK and getattr(o, "attackId", None) == AURA_JAB:
            return [i]

    return chosen
