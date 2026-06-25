"""Dragapult ex — v4 = fresh pilot + EARLY WALL promotion (the starmie-cind setup-speed fix).

DIAGNOSTIC (n=60/deck) overturned the fire-fold premise: fire_ceruledge "0%" was a DATA BUG (illegal
battle_start), and Dragapult BEATS real fire decks (78-93%). The REAL fold is starmie_cinderace (33%):
SETUP-SPEED — Phantom Dive NEVER fired in 53% of losses, board-at-loss median=2. Mega Starmie ex
(210/turn) sweeps the squishy Dragapult line (Dreepy 70hp, Drakloak 90hp) before Stage-2 lands.

FIX: the deck runs two 210hp BASIC walls (Fezandipiti ex retreat-1, Latias ex retreat-2) the agent never
uses. v4 PROMOTES a 210hp wall as the early active when (a) the active is a squishy Dragapult-line
basic (Dreepy/Drakloak, <100hp) AND (b) Dragapult is NOT yet in play AND (c) a wall is on the bench.
The wall absorbs Starmie's hits (forcing 2+ turns per wall instead of an instant OHKO on Dreepy),
buying the turns Stage-2 Dragapult needs to land + energise.

ISOLATION (protects strong matchups from a v3-style regression): the lever is gated on
`not drag_in_play`, so it fires ONLY during the setup phase. In matchups where Dragapult lands fast
(starmie 72%, nighttime 71%), the condition is never true → the lever is dormant → no regression.

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
# v4 wall-promotion: 210hp Basic walls that absorb hits during Dragapult setup.
WALL_IDS = (140, 184)            # Fezandipiti ex (210hp, retreat 1), Latias ex (210hp, retreat 2)
SQUISHY_IDS = (119, 120)         # Dreepy (70hp), Drakloak (90hp) — OHKO'd by 210-dmg sweepers
SWITCH_ITEM = 1123               # Switch item (promotes a benched mon without retreat-cost energy)

_pre_turn = -1


def _log(turn, move, why):
    sys.stderr.write(f"STRAT| T{turn:<2} {move:<22} — {why}\n"); sys.stderr.flush()


def _ids(cards):
    return [c.id for c in (cards or []) if c is not None]


def card_name(cid):
    cd = B.card_table.get(cid)
    return getattr(cd, "name", str(cid)) if cd else str(cid)


def card_retreat(cid):
    cd = B.card_table.get(cid)
    return getattr(cd, "retreatCost", 9) if cd else 9


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck
    chosen = B.agent(obs_dict)
    sel = obs.select
    st = obs.current
    me = st.players[st.yourIndex]
    opp = st.players[1 - st.yourIndex]

    # ---- v4 lever 10: WALL PROMOTION follow-up (SWITCH context). If we just retreated/played Switch
    # to escape a squishy active, pick a 210hp wall (Fez/Latias) as the new active — not another squishy
    # basic. Fires in the SWITCH/TO_ACTIVE select that follows a RETREAT or Switch-item play.
    if sel.context in (3, 4):   # SelectContext.SWITCH=3, TO_ACTIVE=4
        bench = [c for c in (me.bench or []) if c is not None]
        walls = [c for c in bench if c.id in WALL_IDS]
        if walls:
            # pick the lowest-retreat wall (Fez=1 preferred over Latias=2)
            tgt = min(walls, key=lambda c: card_retreat(c.id))
            opts_sw = sel.option or []
            for i, o in enumerate(opts_sw):
                if o.type in (OptionType.CARD, OptionType.PLAY):
                    oarea = getattr(o, "area", None) or getattr(o, "inPlayArea", None)
                    opidx = getattr(o, "index", None)
                    if oarea == 5 and opidx is not None:   # AreaType.BENCH == 5
                        bc = get_card(obs, AreaType.BENCH, opidx, st.yourIndex)
                        if bc is not None and bc.id == tgt.id:
                            _log(st.turn, "promote wall", f"{card_name(tgt.id)} (210hp) active to absorb hits while Dragapult sets up")
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

    # 11. v4 WALL PROMOTION (MAIN trigger) — if the active is a squishy Dragapult-line basic (<100hp)
    #    AND Dragapult is NOT yet in play AND a 210hp wall (Fez/Latias) is benched → escape the squishy
    #    so it doesn't get OHKO'd for free. Prefer Switch-item (no retreat-cost energy); else RETREAT
    #    (if the squishy has energy to pay its retreat). The follow-up SWITCH/TO_ACTIVE select (lever
    #    10 above) picks the wall as the new active. ISOLATED to the setup phase via `not drag_in_play`.
    if not drag_in_play and active is not None and active.id in SQUISHY_IDS:
        bench_walls = [c for c in (me.bench or []) if c is not None and c.id in WALL_IDS]
        if bench_walls:
            # (a) prefer Switch-item (1123) if in hand — promotes a benched mon, no retreat energy
            si = play_idx(SWITCH_ITEM)
            if si is not None and sel.minCount <= 1:
                _log(turn, "Switch (wall)", f"active {card_name(active.id)} ({active.hp}hp) squishy → Switch to 210hp wall")
                return [si]
            # (b) else RETREAT if the squishy can pay its retreat cost (has >= retreat energy)
            rcost = card_retreat(active.id)
            if len(active.energies) >= rcost:
                for i, o in enumerate(opts):
                    if o.type == OptionType.RETREAT:
                        _log(turn, "Retreat (wall)", f"active {card_name(active.id)} ({active.hp}hp) squishy → retreat to 210hp wall")
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
