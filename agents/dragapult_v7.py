"""Dragapult ex — FRESH strategic pilot (mimic of Kaggle sub 54012154 deck).

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
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_dragapult_v5.json"))
import agents.bare_agent as B
from agents.bare_agent import to_observation_class, get_card, AreaType, OptionType

my_deck = B.my_deck

DRAGAPULT, DREEPY, DRAKLOAK = 121, 119, 120
RARE_CANDY, CRISPIN, BOSS, BUDEW, CRUSHING_HAMMER = 1079, 1198, 1182, 235, 1120
PHANTOM_DIVE, JET_HEADBUTT = 154, 153

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
    if sel.context != 0:                      # only steer MAIN-phase decisions
        return chosen
    st = obs.current
    me = st.players[st.yourIndex]
    opp = st.players[1 - st.yourIndex]
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

    # 12. v6 ENERGY-COLOR-COMPLETENESS — force the manual energy-ATTACH onto the active Dragapult to
    #    COMPLETE the Phantom-Dive cost [Fire(2)=2, Psychic(5)=5]. bare_agent attaches by score with no
    #    color-awareness, so it strands a 2nd Fire when Psychic is the missing half -> Phantom Dive
    #    unpayable -> loses the 2HKO race. Pick the ATTACK option whose card supplies the missing color.
    #    (EnergyType 2=Fire, 5=Psychic; Phantom Dive cost ids [2,5] = one of each.)
    if active_is_drag:
        etypes_have = set(int(e) for e in (active.energies or []))
        need_fire = 2 not in etypes_have
        need_psychic = 5 not in etypes_have
        if need_fire or need_psychic:
            best_i = None; best_prio = -1
            for i, o in enumerate(opts):
                if o.type != OptionType.ATTACH:
                    continue
                tgt = get_card(obs, getattr(o, "inPlayArea", None), getattr(o, "inPlayIndex", 0), st.yourIndex)
                if tgt is None or tgt.id != DRAGAPULT:
                    continue
                card = get_card(obs, AreaType.HAND, getattr(o, "index", 0), st.yourIndex)
                if card is None:
                    continue
                etype = getattr(B.card_table.get(card.id), "energyType", None)
                if etype == 5 and need_psychic:
                    prio = 2
                elif etype == 2 and need_fire:
                    prio = 2
                elif etype in (2, 5):
                    prio = 1            # correct color but other half already present (still useful if <2 total)
                else:
                    prio = 0
                if prio > best_prio:
                    best_prio = prio; best_i = i
            if best_i is not None and best_prio >= 1:
                missing = ("Fire" if need_fire else "") + ("+" if need_fire and need_psychic else "") + ("Psychic" if need_psychic else "")
                _log(turn, "attach-color->Dragapult", f"active Dragapult needs {missing} for Phantom Dive → route the missing color")
                return [best_i] if sel.minCount <= 1 else [best_i]

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
