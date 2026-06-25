"""Dragapult ex — v3 = fresh pilot + BUDEW ITEM-LOCK ACTIVATION (the fire-fold lever).

v0 baseline (FULL 37-deck, n=100/deck): MEAN(current meta 10) = 56.5%. HARD COUNTERS:
fire_ceruledge 0%, crustle 28%, starmie_cinderace 33%, typhlosion 33%. Planner verdict: the fire
fold is SETUP-SPEED (Stage-1 Ceruledge out-racing Stage-2 Dragapult), not energy count. Budew's Itchy
Pollen ("opponent can't play Items next turn") buys the extra setup turn — and it's the only lever
that hits ALL the item-heavy hard counters (fire=Poffin/Ultra Ball/Poke Pad, crustle=Hammer spam,
typhlosion=item ramp).

ROOT BUG in v0: lever 5 BENCHES Budew but never ACTIVATES the lock. Itchy Pollen requires Budew to be
ACTIVE and ATTACK; a benched Budew locks nothing. v0 leaves the lock entirely unused.

FIX (this file, on top of the v0 strategy):
  8. WHEN BUDEW IS ACTIVE + Itchy Pollen payable + Dragapult NOT yet online → FORCE Itchy Pollen
     (attack) to lock the opponent's items. Budew (30hp) will get KO'd next turn (1 prize) but the
     lock stalls their setup 1 turn — the tempo that lets Stage-2 Dragapult land. Specifically NOT
     retreating Budew away (v0/bare_agent would retreat the 30hp Basic); we WANT the lock.
  9. KEEP Budew active until Dragapult is online (don't retreat it to bench a development mon that
     can't attack yet). Once Dragapult is evolved+energised on the bench, retreat/switch to promote it.
Levers 1-5 unchanged. Built from v0 (dragapult_fresh) for CLEAN isolation of the C lever.

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
ITCHY_POLLEN = 323   # Budew 'Itchy Pollen' attackId ("opponent can't play Items next turn")

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

    # 8/9. v3 BUDEW ITEM-LOCK ACTIVATION — when Budew is ACTIVE (often the opening Basic) and Dragapult
    #    is NOT online yet, FORCE Itchy Pollen to lock the opponent's items. The whole point of Budew;
    #    v0 benched it but never used the lock. Budew dies next turn (30hp, 1 prize) but the 1-turn
    #    item-lock stalls fire/crustle/typhlosion setup long enough for Stage-2 Dragapult to land.
    #    Do NOT retreat Budew while it can still lock and Dragapult isn't ready.
    if active is not None and active.id == BUDEW and not drag_in_play:
        for i, o in enumerate(opts):
            if o.type == OptionType.ATTACK and getattr(o, "attackId", None) == ITCHY_POLLEN:
                _log(turn, "Itchy Pollen (lock)", "Budew active, Dragapult not ready → lock opp items (sacrifice Budew, buy setup turn)")
                return [i] if sel.minCount <= 1 else [i]

    return chosen
