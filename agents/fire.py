"""Tuned anti-Crustle Fire deck agent.

Main's Crustle deck is a fragile 8-Pokemon, single-line, **all-Grass** deck whose
Pokemon are weak to Fire. This deck is fast NON-ex Fire aggro: every attacker OHKOs
Crustle (>=75 base x2 weakness = >=150) and survives Crustle's 120, so the prize race
is ~2:1 in our favor (we KO every turn; main needs two hits on our 130-HP bodies and
gives us only 1 prize since nothing is an ex).

The generic bare engine spreads energy and wastes turns on cheap 0-damage attacks
(Singe/Flap). This agent layers on:
  - ENERGY CONCENTRATION: pile energy on ONE attacker until it can fire its [RRC]
    OHKO, instead of dribbling it across the bench.
  - skip 0-damage distractor attacks when we're close to powering the real one.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

from cg.api import (
    AreaType, Card, CardType, EnergyType, Observation, OptionType, Pokemon,
    SelectContext, all_attack, all_card_data, to_observation_class,
)

# ---- Load the Fire deck (deck-specific agent) ----
_default_deck = Path(__file__).resolve().parents[1] / "data" / "decks" / "deck_fire.json"
_deck_path = os.environ.get("BARE_DECK", "") or str(_default_deck)
if os.path.exists(_deck_path):
    my_deck = json.load(open(_deck_path))
else:
    my_deck = [318]*4 + [663]*3 + [358]*3 + [2]*16 + [1121]*4 + [1152]*4 + [1182]*4 + \
              [1119]*4 + [1123]*3 + [1227]*4 + [1212]*4 + [1235]*3 + [1159]*1 + [1147]*3
assert len(my_deck) == 60, f"deck has {len(my_deck)} cards"

# ---- Card / attack databases (read at runtime on the competition image) ----
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
try:
    attack_table = {a.attackId: a for a in all_attack()}
except Exception:
    attack_table = {}


def _stage(cid):
    c = card_table.get(cid)
    if c is None:
        return 0
    return 2 if getattr(c, "stage2", False) else 1 if getattr(c, "stage1", False) else 0


def _best_dmg(cid):
    c = card_table.get(cid)
    if c is None:
        return 0
    return max((getattr(attack_table.get(a), "damage", 0) or 0
                for a in (c.attacks or []) if a in attack_table), default=0)


# ---- Derive deck roles automatically (no archetype knowledge) ----
_deck_set = set(my_deck)
_poke = [cid for cid in _deck_set if card_table.get(cid) and card_table[cid].cardType == CardType.POKEMON]

BASIC_MONS = {cid for cid in _poke if _stage(cid) == 0}
ATTACKERS = {cid for cid in _poke if _stage(cid) >= 1}                  # evolutions carry the attack
ATTACKERS |= {cid for cid in BASIC_MONS if _best_dmg(cid) >= 60}        # all real Fire attackers
if not ATTACKERS:
    ATTACKERS = set(_poke)

ENERGY_BASICS = {cid for cid in _deck_set if card_table.get(cid) and card_table[cid].cardType == CardType.BASIC_ENERGY}
SPECIAL_ENERGY = {cid for cid in _deck_set if card_table.get(cid) and card_table[cid].cardType == CardType.SPECIAL_ENERGY}
ALL_ENERGY = ENERGY_BASICS | SPECIAL_ENERGY
TOOLS = {cid for cid in _deck_set if card_table.get(cid) and card_table[cid].cardType == CardType.TOOL}

# Heal cards (Cook +70, Jumbo Ice Cream +80) — only worth playing on a damaged active,
# and never at the cost of our one Supporter-per-turn when we're at full HP.
def _heals(cid):
    c = card_table.get(cid)
    return bool(c) and any("heal" in (getattr(s, "text", "") or "").lower()
                           for s in (getattr(c, "skills", None) or []))
HEAL_CARDS = {cid for cid in _deck_set if _heals(cid)}

# Energy types the deck can attach (used to test "attach one more this turn" plans).
DECK_ETYPES = {card_table[e].energyType for e in ENERGY_BASICS if card_table.get(e)}
# Primary basic energy id (most common), for the "+1 energy" lookahead.
_ec = defaultdict(int)
for cid in my_deck:
    if cid in ENERGY_BASICS:
        _ec[cid] += 1
PRIMARY_BASIC_ENERGY = max(_ec, key=_ec.get) if _ec else (next(iter(ENERGY_BASICS), None))


class AttackPlan:
    attacker = -1
    target = -1
    attack_id = -1
    remain_hp = -1
    needs_energy = False


plan = AttackPlan()
pre_turn = -1
turn_actions = 0      # actions taken in the current turn (loop guard)
turn_abilities = 0    # abilities used in the current turn (loop guard)


def get_card(obs, area, index, player_index):
    try:
        ps = obs.current.players[player_index]
        match area:
            case AreaType.DECK: return obs.select.deck[index]
            case AreaType.HAND: return ps.hand[index]
            case AreaType.DISCARD: return ps.discard[index]
            case AreaType.ACTIVE: return ps.active[index]
            case AreaType.BENCH: return ps.bench[index]
            case AreaType.PRIZE: return ps.prize[index]
            case AreaType.STADIUM: return obs.current.stadium[index]
            case AreaType.LOOKING: return obs.current.looking[index]
            case _: return None
    except (IndexError, TypeError, AttributeError):
        return None


def card_type(cid):
    cd = card_table.get(cid)
    return cd.cardType if cd else None


def prize_count(pokemon):
    data = card_table.get(pokemon.id)
    if data is None:
        return 1
    return 3 if data.megaEx else 2 if data.ex else 1


def can_pay(attached, cost):
    need = list(cost)
    pool = list(attached)
    for c in [e for e in need if e != EnergyType.COLORLESS]:
        match = next((p for p in pool if p == c or p == EnergyType.RAINBOW), None)
        if match is None:
            return False
        pool.remove(match)
    colorless = sum(1 for e in need if e == EnergyType.COLORLESS)
    return len(pool) >= colorless


def attack_damage(attacker, atk, target):
    dmg = getattr(atk, "damage", 0) or 0
    if dmg <= 0:
        return 0
    my_type = card_table[attacker.id].energyType if attacker.id in card_table else None
    tdata = card_table.get(target.id)
    if tdata and my_type is not None:
        if tdata.weakness == my_type:
            dmg *= 2
        elif tdata.resistance == my_type:
            dmg = max(0, dmg - 30)
    return dmg


def pokemon_score(pokemon):
    data = card_table.get(pokemon.id)
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 120
    score += len(pokemon.tools) * 80
    if data:
        if data.stage2:
            score += 250
        elif data.stage1:
            score += 130
    score += pokemon.hp
    return score


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    op_prize_left = len(op_state.prize)

    global plan, pre_turn, turn_actions
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()
        turn_actions = 0
    turn_actions += 1
    # Loop guard: ability-engine decks can offer a repeatable ABILITY forever, stalling
    # the game to the step cap (a false "draw"). Suppress abilities after a handful of
    # actions, and force END once a turn has churned far too long.
    stalling = turn_actions > 80

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    for c in my_state.active + list(my_state.bench):
        if c is not None:
            field_counts[c.id] += 1
    for c in (my_state.hand or []):
        hand_counts[c.id] += 1

    n_bodies = sum(1 for c in my_state.active + list(my_state.bench) if c is not None)
    have_spare_energy = any(hand_counts[e] >= 1 for e in ALL_ENERGY)
    _act = my_state.active[0] if my_state.active else None
    active_dmg = (_act.maxHp - _act.hp) if (_act is not None) else 0   # damage on our active

    can_switch = False
    if context == SelectContext.MAIN:
        for o in select.option:
            if o.type in (OptionType.RETREAT,):
                can_switch = True

        my_cards = [my_state.active[0] if my_state.active else None] + list(my_state.bench)
        op_cards = [op_state.active[0] if op_state.active else None] + list(op_state.bench)

        best = -1.0
        for i, mine in enumerate(my_cards):
            if mine is None:
                continue
            if i != 0 and not can_switch:
                continue
            cd = card_table.get(mine.id)
            if cd is None:
                continue
            for aid in (cd.attacks or []):
                atk = attack_table.get(aid)
                if atk is None:
                    continue
                cost = atk.energies or []
                attached_types = list(mine.energies)
                more_energy = False
                if not can_pay(attached_types, cost):
                    # Try assuming we attach one more energy (any deck type) this turn.
                    if have_spare_energy and not state.energyAttached and DECK_ETYPES and \
                       any(can_pay(attached_types + [et], cost) for et in DECK_ETYPES):
                        more_energy = True
                    else:
                        continue
                for j, opp in enumerate(op_cards):
                    if opp is None or j != 0:
                        continue
                    dmg = attack_damage(mine, atk, opp)
                    if dmg <= 0:
                        continue
                    score = pokemon_score(opp)
                    if opp.hp <= dmg:
                        prize = prize_count(opp)
                        score += 500
                        if op_prize_left <= prize:
                            score = 50000
                    else:
                        score *= dmg / max(1, opp.hp)
                    if i == 0:
                        score += 200
                    score += len(mine.energies)
                    if score > best:
                        best = score
                        plan.attacker = i
                        plan.target = j
                        plan.attack_id = aid
                        plan.remain_hp = opp.hp - dmg
                        plan.needs_energy = more_energy

    def energy_score(pokemon, is_active):
        if pokemon is None:
            return -1
        score = 8000
        e = len(pokemon.energies)
        if pokemon.id in ATTACKERS:
            score += 300
            # CONCENTRATE: finish powering the most-loaded attacker to [RRC]=3, rather
            # than dribbling energy across the bench. Once it has 3, spread to a fresh one.
            if e >= 3:
                score -= 400
            else:
                score += e * 200
        else:
            score += 20            # don't strand energy on non-attacker bodies
        if is_active:
            score += 120           # power the body that's actually doing the attacking
        return score

    scores = []
    for o in select.option:
        score = 0.0

        if o.type == OptionType.NUMBER:
            score = o.number if o.number is not None else 0

        elif o.type == OptionType.YES:
            score = 1

        elif o.type == OptionType.NO:
            score = 0

        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                    if o.playerIndex == my_index:
                        energy_count = len(card.energies) if isinstance(card, Pokemon) else 0
                        score = 10 + energy_count * 2
                        if o.index == plan.attacker - 1:
                            score += 100
                        if card.id in ATTACKERS:
                            score += 40
                        elif card.id in BASIC_MONS:
                            score += 10
                    else:
                        # Gust target (Boss's Orders): we OHKO anything via Fire weakness,
                        # so drag up their most-developed benched threat (most energy /
                        # evolved) to KO it and break their thin board.
                        if isinstance(card, Pokemon):
                            score = 50 + len(card.energies) * 12
                            cd = card_table.get(card.id)
                            if cd and (getattr(cd, "stage1", False) or getattr(cd, "stage2", False)):
                                score += 40
                        if o.index == plan.target - 1:
                            score += 100
                elif context in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE):
                    # The ACTIVE must survive Crustle's 120, so prefer a 130-HP body
                    # (Ho-Oh / Volcanion). Keep the 110-HP Hearthflame benched as an
                    # accelerator — active, it just gets OHKO'd for a free prize.
                    cd = card_table.get(card.id)
                    hp = getattr(cd, "hp", 0) if cd else 0
                    score = (100 if card.id in BASIC_MONS else 50) + hp
                elif context in (
                    SelectContext.SETUP_BENCH_POKEMON,
                    SelectContext.TO_FIELD, SelectContext.TO_BENCH,
                ):
                    score = 100 if card.id in BASIC_MONS else 50
                elif context == SelectContext.TO_HAND:
                    cid = card.id
                    if cid in ATTACKERS:
                        score = 300 - field_counts[cid] * 40
                    elif cid in BASIC_MONS:
                        score = 260 - (field_counts[cid] + hand_counts[cid]) * 60
                    elif cid in ALL_ENERGY:
                        score = 120 if not state.energyAttached else 60
                    elif card_type(cid) == CardType.SUPPORTER:
                        score = 140
                    elif card_type(cid) == CardType.ITEM:
                        score = 130
                    else:
                        score = 100 - hand_counts[cid] * 20
                elif context == SelectContext.ATTACH_FROM:
                    if isinstance(card, Pokemon):
                        score = energy_score(card, o.area == AreaType.ACTIVE)
                elif context in (
                    SelectContext.DISCARD, SelectContext.TO_DECK,
                    SelectContext.TO_DECK_BOTTOM, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
                ):
                    cid = card.id
                    if cid in ATTACKERS or cid in BASIC_MONS:
                        score = -50
                    elif cid in ALL_ENERGY:
                        score = 10 + max(0, hand_counts[cid] - 1) * 20
                    else:
                        score = 30 + max(0, hand_counts[cid] - 1) * 25
                else:
                    score = 1

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is not None:
                ct = card_type(card.id)
                if ct == CardType.POKEMON:
                    score = 20000
                    if card.id in BASIC_MONS and n_bodies >= 3:
                        score = 30
                elif card.id in HEAL_CARDS and active_dmg < 30:
                    score = 200          # don't waste a heal (or our supporter) at ~full HP
                elif ct == CardType.ITEM:
                    score = 3500
                elif ct == CardType.SUPPORTER:
                    score = 3000
                elif ct == CardType.STADIUM:
                    score = 1500
                else:
                    score = 1000

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is not None and card.id in TOOLS:
                score = 7000
                if pokemon is not None and pokemon.id in ATTACKERS:
                    score += 200
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                if plan.needs_energy:
                    if plan.attacker == 0 and o.inPlayArea == AreaType.ACTIVE:
                        score += 300
                    elif plan.attacker == 1 + (o.inPlayIndex or 0) and o.inPlayArea == AreaType.BENCH:
                        score += 300

        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + (len(pokemon.energies) if isinstance(pokemon, Pokemon) else 0) * 10

        elif o.type == OptionType.ABILITY:
            # Use abilities, but stop after several actions in a turn to avoid loops.
            score = 15000 if turn_actions <= 8 else -2000

        elif o.type == OptionType.RETREAT:
            score = 2000 if plan.attacker >= 1 else -1

        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == plan.attack_id:
                score += 500

        elif o.type == OptionType.END:
            score = 200000 if stalling else -1000

        else:
            score = 0

        scores.append(score)

    desc = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    chosen = desc[: max(select.minCount, min(select.maxCount, len(desc)))]
    if len(chosen) < select.minCount:
        chosen = desc[: select.minCount]
    return chosen
