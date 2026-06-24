"""Dudunsparce/Alakazam Psychic toolbox pilot (#1 top-50 deck) on the bare engine.

Ships the most-common winning Psychic-toolbox 60 and reuses the archetype-agnostic
bare_agent engine. Originally: a generic "bare setup" agent piloting ANY deck.

It reuses the same generic engine as main.py (runtime card/attack stats, evolve-ASAP,
feed-the-attacker energy heuristics) but DERIVES the deck roles automatically:
  - basics            = every Stage-0 Pokemon in the deck
  - attackers         = every evolution + any basic with a real (>=90) attack
  - energy            = whatever basic/special energy the deck runs
  - tools             = any Pokemon Tool (attached to an attacker, like Hero's Cape)

The 60-card deck is loaded from the BARE_DECK env var (path to a JSON list of ids),
so one module plays all 10 meta decks across separate runner invocations.
"""
import json
import os
from collections import defaultdict

from cg.api import (
    AreaType, Card, CardType, EnergyType, Observation, OptionType, Pokemon,
    SelectContext, all_attack, all_card_data, to_observation_class,
)

# ---- Dudunsparce/Alakazam Psychic toolbox (#1 deck in the top 50) ----
# Alakazam Stage-2 attacker (Abra->Kadabra->Alakazam + Rare Candy) on a Dunsparce
# ->Dudunsparce draw engine; Telepath Psychic Energy, Enhanced Hammer disruption,
# Boss's Orders gust. Most-common winning list among top-team Psychic decks (8x;
# out/dudunsparce_deck.json). Overridable via BARE_DECK.
_deck_path = os.environ.get("BARE_DECK", "")
if _deck_path and os.path.exists(_deck_path):
    my_deck = json.load(open(_deck_path))
else:
    my_deck = (
        [5]*3 + [13] + [19]*4 +          # Basic {P} / Enriching / Telepath Psychic Energy
        [741]*4 + [742]*4 + [743]*4 +    # Abra -> Kadabra -> Alakazam (attacker)
        [305]*3 + [66]*3 +               # Dunsparce -> Dudunsparce (draw engine)
        [1079]*3 + [1086]*4 + [1152]*4 + # Rare Candy / Buddy-Buddy Poffin / Poke Pad
        [1231]*4 + [1225]*3 +            # Dawn / Hilda (draw supporters)
        [1081]*4 +                       # Enhanced Hammer (energy disruption)
        [1182]*3 + [1097]*3 + [1264]*3 + # Boss's Orders / Night Stretcher / Battle Cage
        [1129] + [1146] + [1184]         # Sacred Ash / Wondrous Patch / Lana's Aid
    )
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
ATTACKERS |= {cid for cid in BASIC_MONS if _best_dmg(cid) >= 90}        # strong standalone basics
if not ATTACKERS:
    ATTACKERS = set(_poke)

ENERGY_BASICS = {cid for cid in _deck_set if card_table.get(cid) and card_table[cid].cardType == CardType.BASIC_ENERGY}
SPECIAL_ENERGY = {cid for cid in _deck_set if card_table.get(cid) and card_table[cid].cardType == CardType.SPECIAL_ENERGY}
ALL_ENERGY = ENERGY_BASICS | SPECIAL_ENERGY
TOOLS = {cid for cid in _deck_set if card_table.get(cid) and card_table[cid].cardType == CardType.TOOL}

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
        score = 8000
        if pokemon is None:
            return -1
        if pokemon.id in ATTACKERS:
            score += 300
        elif pokemon.id in BASIC_MONS:
            score += 150
        if is_active:
            score += 40
        if len(pokemon.energies) >= 4:
            score -= 60
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
                        if o.index == plan.target - 1:
                            score += 100
                elif context in (
                    SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON,
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
