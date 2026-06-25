"""Dragapult ex — v2 = fresh pilot + ENERGY-TO-ACTIVE fix (the crustle-floor lever).

v0 baseline (n=100/deck, 10-deck field): 69.4% mean. STRONG zoroark97/honchkrow91/ogerpon82/alakazam79/
bolt74/slowking68; MID mirror62/clefairy60/hydrapple50; WEAK crustle 31%. Trace: "energy often
stranded on bench, active Dragapult starved." Crustle = 4x Crushing Hammer denial wall; Dragapult's
Phantom Dive(200) OHKOs Crustle(150) IF it has energy — but Hammers strip it + bench-routed attaches
starve the active, so Dragapult can't pay [Fire+Psychic] and loses the race.

FIX (this file, on top of the v0 strategy):
  6. FORCE the manual turn energy-ATTACH onto the active under-energised Dragapult (bare_agent was
     routing it to benched planned-attackers). Single safe MAIN ATTACH force.
  7. ROUTE Crispin/attach-target selects to the active Dragapult (the Crispin force existed but its
     follow-up target pick defaulted to bare_agent, which picked the bench). Handles ATTACH_TO/TO_FIELD
     selects by picking the option referencing the active Dragapult when it is starved.
Levers 1-5 (Rare Candy, Crispin force, Phantom Dive, Boss gust, Budew lock) unchanged.

--- ORIGINAL v0 docstring below ---
Dragapult ex — FRESH strategic pilot (mimic of Kaggle sub 54012154 deck).

A real strategy on top of the generic bare engine, NOT a plain wrapper. Plan:
  1. SET UP FAST  — force Rare Candy (Dreepy -> Dragapult ex, skip Drakloak) the turn it's possible.
  2. FUEL         — force Crispin (dual {R}/{P} energy acceleration) while Dragapult is under-energised.
  3. SPREAD       — attack with Phantom Dive (200 to active + 6 counters / 60 dmg onto the bench),
                    setting up multi-prize turns.
  4. GUST KO      — once the spread has softened a benched Pokémon, force Boss's Orders to drag it up
                    so Phantom Dive / a cheap hit closes the prize (the deck's core combo).
  5. DISRUPT      — Budew item-lock + Crushing Hammer when not yet set up / opponent is charging.

Every override is logged to stderr as `STRAT| T<turn> <move> — <why>` so the decision-making is visible.
"""
import os
import sys

os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_dragapult_mimic.json"))
import agents.bare_agent as B
from agents.bare_agent import to_observation_class, get_card, AreaType, OptionType

my_deck = B.my_deck

DRAGAPULT, DREEPY, DRAKLOAK = 121, 119, 120
RARE_CANDY, CRISPIN, BOSS, BUDEW, CRUSHING_HAMMER = 1079, 1198, 1182, 235, 1120
PHANTOM_DIVE, JET_HEADBUTT = 154, 153
# Attack-target / attach-target select contexts we steer (energy routing to active Dragapult).
ATTACH_TO_CTX, TO_FIELD_CTX = 22, 6

_pre_turn = -1


def _log(turn, move, why):
    sys.stderr.write(f"STRAT| T{turn:<2} {move:<22} — {why}\n"); sys.stderr.flush()


def _ids(cards):
    return [c.id for c in (cards or []) if c is not None]


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck
    chosen = B.agent(obs_dict)
    sel = obs.select
    st = obs.current
    me = st.players[st.yourIndex]
    opp = st.players[1 - st.yourIndex]
    # ---- v2 lever 7: route attach-target selects (e.g. Crispin's "attach to 1 of your Pokémon") to
    # the active under-energised Dragapult, so Crispin/manual-accel energy lands on the attacker,
    # not the bench. Fires only in attach-like non-MAIN contexts when active Dragapult is starved.
    if sel.context in (ATTACH_TO_CTX, TO_FIELD_CTX):
        active = (me.active or [None])[0]
        if active is not None and active.id == DRAGAPULT and len(active.energies) < 2:
            opts = sel.option or []
            for i, o in enumerate(opts):
                # CARD-type options referencing a Pokémon carry area/index/playerIndex
                oarea = getattr(o, "area", None)
                opidx = getattr(o, "playerIndex", None)
                oarea2 = getattr(o, "inPlayArea", None)   # ATTACH-style options use inPlayArea
                if opidx == st.yourIndex and (oarea == 4 or oarea2 == 4):   # AreaType.ACTIVE == 4
                    _log(st.turn, "attach->active Dragapult", f"Crispin/route energy to starved active ({len(active.energies)}E)")
                    return [i] if sel.minCount <= 1 else [i]
    if sel.context != 0:                      # only steer MAIN-phase decisions below
        return chosen
    opts = sel.option or []
    turn = st.turn
    global _pre_turn

    active = (me.active or [None])[0]
    in_play = _ids(([active] if active else []) + list(me.bench or []))
    hand = _ids(me.hand)
    drag_in_play = DRAGAPULT in in_play
    active_is_drag = active is not None and active.id == DRAGAPULT

    def play_idx(card_id):
        for i, o in enumerate(opts):
            if o.type == OptionType.PLAY:
                c = get_card(obs, AreaType.HAND, o.index, st.yourIndex)
                if c is not None and c.id == card_id:
                    return i
        return None

    # 1. RARE CANDY — land Dragapult ex ASAP (bare_agent never plays it)
    if not drag_in_play and DREEPY in in_play and DRAGAPULT in hand and RARE_CANDY in hand:
        i = play_idx(RARE_CANDY)
        if i is not None and sel.minCount <= 1:
            _log(turn, "Rare Candy", "Dreepy in play + Dragapult in hand → skip to Stage 2 now")
            return [i]

    # 6. v2 ENERGY-TO-ACTIVE — force the manual turn energy-ATTACH onto the active under-energised
    #    Dragapult. bare_agent routes the free attach to benched planned-attackers, starving the
    #    active so it can never pay Phantom Dive's [Fire+Psychic] (the crustle-floor root cause).
    #    Only fires when Dragapult is active with <2 energy; the ATTACK option for Phantom Dive only
    #    appears once payable, so this just gets us there faster. (AreaType.ACTIVE == 4.)
    if active_is_drag and len(active.energies) < 2:
        for i, o in enumerate(opts):
            if o.type == OptionType.ATTACH and getattr(o, "inPlayArea", None) == 4:
                tgt = get_card(obs, AreaType.ACTIVE, getattr(o, "inPlayIndex", 0), st.yourIndex)
                if tgt is not None and tgt.id == DRAGAPULT:
                    _log(turn, "attach->active Dragapult", f"manual energy to starved active ({len(active.energies)}E) — beat Hammer denial")
                    return [i] if sel.minCount <= 1 else [i]

    # 2. CRISPIN — accelerate {R}/{P} energy onto the line when Dragapult is short on energy
    drag_energy = len(active.energies) if active_is_drag else 0
    if drag_in_play and drag_energy < 2 and CRISPIN in hand and not st.supporterPlayed:
        i = play_idx(CRISPIN)
        if i is not None and sel.minCount <= 1:
            _log(turn, "Crispin", f"Dragapult has {drag_energy} energy (<2 for Phantom Dive) → ramp")
            return [i]

    # 3. PHANTOM DIVE — the spread attack; force it over Jet Headbutt when payable
    if active_is_drag:
        for i, o in enumerate(opts):
            if o.type == OptionType.ATTACK and getattr(o, "attackId", None) == PHANTOM_DIVE:
                _log(turn, "Phantom Dive", "200 to active + 6 counters onto bench → set up gust KOs")
                return [i]

    # 4. BOSS'S ORDERS — drag up a benched Pokémon the spread has already softened (combo close-out)
    opp_bench = [c for c in (opp.bench or []) if c is not None]
    softened = [c for c in opp_bench if getattr(c, "hp", 999) <= 120]  # in Phantom-Dive/Jet range after spread
    if active_is_drag and drag_energy >= 2 and softened and BOSS in hand and not st.supporterPlayed:
        i = play_idx(BOSS)
        if i is not None and sel.minCount <= 1:
            tgt = min(softened, key=lambda c: getattr(c, "hp", 999))
            _log(turn, "Boss's Orders", f"gust softened bench mon ({getattr(tgt,'hp','?')}hp) → Phantom Dive KO")
            return [i]

    # 5. BUDEW item-lock when we have NOT set Dragapult up yet (slow the opponent's items)
    if not drag_in_play and BUDEW in hand and turn <= 4:
        i = play_idx(BUDEW)
        if i is not None and sel.minCount <= 1 and BUDEW not in in_play:
            _log(turn, "Budew (bench)", "not set up yet → bench Budew for Itchy Pollen item-lock")
            return [i]

    return chosen
