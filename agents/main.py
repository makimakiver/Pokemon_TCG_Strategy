import os
import sys
from collections import defaultdict

from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_attack,
    all_card_data,
    to_observation_class,
)

"""
"Palace" Single-Line Evolution Deck (ported from Shun's winning list in palace_1/2.json).

This deck is built around ONE evolution line:
    344 (Basic, 70 HP)  --evolve-->  345 (Stage1, 150 HP, main attacker)
backed by a very heavy energy count and a 25-card trainer engine that digs for
the line and refuels the attacker.

IMPORTANT: card *names/effects* are not known offline (the card DB lives in the
native binary that only runs on the competition servers). So this agent reads all
card/attack STATS at runtime via all_card_data()/all_attack() and only hardcodes
the deck-specific STRATEGY (which Pokemon attacks, energy priorities, evolve ASAP).
Trainers whose effects we can't introspect are played via cardType-based heuristics.
"""

# Winning deck, copied verbatim (60 card IDs) from palace_1.json / palace_2.json (Shun).
my_deck = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    11, 11, 11, 11,
    14, 14, 14, 14,
    18, 18, 18, 18,
    344, 344, 344, 344,
    345, 345, 345, 345,
    1086, 1086, 1086, 1086,
    1147, 1147, 1147, 1147,
    1212, 1212, 1212, 1212,
    1227, 1227, 1227, 1227,
    1235, 1235, 1235, 1235,
    1159,
]

# ---- Deck constants (roles inferred from replays) ----
BASIC_MON = 344       # only Basic Pokemon in the deck
ATTACKER = 345        # Stage1 evolution; carries the attack
BASIC_ENERGY = 1      # main energy (x19)
SPECIAL_ENERGY = {11, 14}  # special energies, 1-of each onto the attacker
HERO_CAPE = 1159      # Pokemon Tool -> goes on the attacker
LILLIE_DET = 1227     # known draw Supporter (from the original decklist comments)
POKE_SEARCH = 1086    # Item: Buddy-Buddy Poffin (search 2 Basics<=70hp to bench)
COOK = 1212           # Supporter: Cook (heal) — Shun only plays it when damaged

# Card / attack databases (work on the competition servers; read at runtime).
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
try:
    attack_table = {a.attackId: a for a in all_attack()}
except Exception:
    attack_table = {}


class AttackPlan:
    attacker = -1      # index into [active] + bench (0 == active)
    target = -1        # index into opponent [active] + bench (0 == active)
    attack_id = -1
    remain_hp = -1     # opponent HP left after the planned attack
    needs_energy = False  # True if the plan requires one more energy attach this turn


plan = AttackPlan()
pre_turn = -1


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    """Safely extract a Card or Pokemon object from a specific zone."""
    try:
        ps = obs.current.players[player_index]
        match area:
            case AreaType.DECK:
                return obs.select.deck[index]
            case AreaType.HAND:
                return ps.hand[index]
            case AreaType.DISCARD:
                return ps.discard[index]
            case AreaType.ACTIVE:
                return ps.active[index]
            case AreaType.BENCH:
                return ps.bench[index]
            case AreaType.PRIZE:
                return ps.prize[index]
            case AreaType.STADIUM:
                return obs.current.stadium[index]
            case AreaType.LOOKING:
                return obs.current.looking[index]
            case _:
                return None
    except (IndexError, TypeError, AttributeError):
        return None


def card_type(cid: int):
    cd = card_table.get(cid)
    return cd.cardType if cd else None


def prize_count(pokemon: Pokemon) -> int:
    """Prizes yielded when this Pokemon is Knocked Out."""
    data = card_table.get(pokemon.id)
    if data is None:
        return 1
    return 3 if data.megaEx else 2 if data.ex else 1


def can_pay(attached: list, cost: list) -> bool:
    """True if `attached` energy types can satisfy `cost` (colorless = any)."""
    need = list(cost)
    pool = list(attached)
    # First satisfy the specific (non-colorless) requirements.
    for c in [e for e in need if e != EnergyType.COLORLESS]:
        match = next((p for p in pool if p == c or p == EnergyType.RAINBOW), None)
        if match is None:
            return False
        pool.remove(match)
    # Remaining requirements are colorless -> any leftover energy pays them.
    colorless = sum(1 for e in need if e == EnergyType.COLORLESS)
    return len(pool) >= colorless


def attack_damage(attacker: Pokemon, atk, target: Pokemon) -> int:
    """Raw attack damage adjusted for the target's weakness/resistance."""
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


def pokemon_score(pokemon: Pokemon) -> int:
    """Heuristic value of knocking out / pressuring a given opponent Pokemon."""
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


def agent(obs_dict: dict) -> list[int]:
    """Main agent. Returns a list of option indices within [minCount, maxCount]."""
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        # Initial selection: return the 60-card deck.
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize_left = len(my_state.prize)
    op_prize_left = len(op_state.prize)

    global plan, pre_turn
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()

    # Inventory of our own field / hand.
    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    for c in my_state.active + list(my_state.bench):
        if c is not None:
            field_counts[c.id] += 1
    for c in (my_state.hand or []):
        hand_counts[c.id] += 1

    # Total Pokemon bodies in play (drives anti-no-active bench insurance).
    total_bodies = sum(1 for c in my_state.active + list(my_state.bench) if c is not None)

    have_spare_energy = (
        hand_counts[BASIC_ENERGY] >= 1
        or any(hand_counts[e] >= 1 for e in SPECIAL_ENERGY)
    )

    # ---------------------------------------------------------------
    # Build an attack plan during the MAIN selection.
    # ---------------------------------------------------------------
    can_attack_now = False
    can_switch = False
    if context == SelectContext.MAIN:
        for o in select.option:
            if o.type == OptionType.ATTACK:
                can_attack_now = True
            elif o.type in (OptionType.RETREAT,):
                can_switch = True
            elif o.type == OptionType.PLAY:
                c = get_card(obs, AreaType.HAND, o.index, my_index)
                if c is not None and card_type(c.id) == CardType.ITEM:
                    # Switch-like items exist; treat as possible switch enabler.
                    pass

        my_cards = [my_state.active[0] if my_state.active else None] + list(my_state.bench)
        op_cards = [op_state.active[0] if op_state.active else None] + list(op_state.bench)

        best = -1.0
        for i, mine in enumerate(my_cards):
            if mine is None:
                continue
            if i != 0 and not can_switch:
                # Can only attack with a benched Pokemon if we can promote it.
                continue
            cd = card_table.get(mine.id)
            if cd is None:
                continue
            for aid in (cd.attacks or []):
                atk = attack_table.get(aid)
                if atk is None:
                    continue
                cost = atk.energies or []
                need = len(cost)
                attached_types = list(mine.energies)
                more_energy = False
                if not can_pay(attached_types, cost):
                    # Try assuming we attach one more basic energy this turn.
                    if have_spare_energy and not state.energyAttached:
                        if can_pay(attached_types + [card_table[BASIC_ENERGY].energyType], cost):
                            more_energy = True
                        else:
                            continue
                    else:
                        continue
                for j, opp in enumerate(op_cards):
                    if opp is None:
                        continue
                    if j != 0:
                        # We can't reliably gust the bench (unknown trainer effects).
                        continue
                    dmg = attack_damage(mine, atk, opp)
                    if dmg <= 0:
                        continue
                    score = pokemon_score(opp)
                    if opp.hp <= dmg:
                        # Knockout.
                        prize = prize_count(opp)
                        score += 500
                        if op_prize_left <= prize:
                            score = 50000  # this attack wins the game
                    else:
                        score *= dmg / max(1, opp.hp)
                    if i == 0:
                        score += 200  # prefer attacking with the active (no switch cost)
                    score += len(mine.energies)
                    if score > best:
                        best = score
                        plan.attacker = i
                        plan.target = j
                        plan.attack_id = aid
                        plan.remain_hp = opp.hp - dmg
                        plan.needs_energy = more_energy

    # ---------------------------------------------------------------
    # Energy attachment scoring: feed the attacker until it can swing.
    # ---------------------------------------------------------------
    def energy_score(pokemon: Pokemon, is_active: bool) -> int:
        score = 8000
        if pokemon is None:
            return -1
        if pokemon.id == ATTACKER:
            score += 300
        elif pokemon.id == BASIC_MON:
            score += 150  # pre-evolution that will become the attacker
        if is_active:
            score += 40
        # Don't massively overload one body once it can already attack.
        if len(pokemon.energies) >= 4:
            score -= 60
        return score

    # ---------------------------------------------------------------
    # Score every option.
    # ---------------------------------------------------------------
    scores: list[float] = []
    for o in select.option:
        score = 0.0

        if o.type == OptionType.NUMBER:
            score = o.number if o.number is not None else 0  # e.g. draw as many as allowed

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
                        if card.id == ATTACKER:
                            score += 40
                        elif card.id == BASIC_MON:
                            score += 10
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context in (
                    SelectContext.SETUP_ACTIVE_POKEMON,
                    SelectContext.SETUP_BENCH_POKEMON,
                    SelectContext.TO_FIELD,
                    SelectContext.TO_BENCH,
                ):
                    # Only basics are legal here; play them.
                    score = 100 if card.id == BASIC_MON else 50
                elif context == SelectContext.TO_HAND:
                    # Searching/drawing to hand: prioritize the line, then energy.
                    cid = card.id
                    if cid == ATTACKER:
                        score = 300 - field_counts[cid] * 40
                    elif cid == BASIC_MON:
                        score = 260 - (field_counts[cid] + hand_counts[cid]) * 60
                    elif cid == BASIC_ENERGY or cid in SPECIAL_ENERGY:
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
                    SelectContext.DISCARD,
                    SelectContext.TO_DECK,
                    SelectContext.TO_DECK_BOTTOM,
                    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
                ):
                    # Higher score == more willing to let this card go.
                    cid = card.id
                    if cid == ATTACKER or cid == BASIC_MON:
                        score = -50  # keep the line
                    elif cid == BASIC_ENERGY or cid in SPECIAL_ENERGY:
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
                    # Build the board, but don't flood the bench with basics.
                    score = 20000
                    if card.id == BASIC_MON and field_counts[BASIC_MON] + field_counts[ATTACKER] >= 3:
                        score = 30  # already have enough bodies
                elif ct == CardType.SUPPORTER:
                    score = 3000  # draw/search engine — generally play one each turn
                elif ct == CardType.ITEM:
                    score = 2500
                    # Buddy-Buddy Poffin (1086) searches 2 Basics to bench. Losses are
                    # `no-active` (active KO'd with empty bench). Refilling is matchup-
                    # adaptive: when we're being RACED (opponent has taken >=2 of our prizes)
                    # keep a deeper bench (<=2) to survive the KO pressure; otherwise refill
                    # only when the bench is empty (<=1), since proactive searching thins our
                    # own deck and over-exposes bodies -> regresses the slow mill matchups.
                    under_race_pressure = my_prize_left <= 5
                    refill_threshold = 2 if under_race_pressure else 1
                    if card.id == POKE_SEARCH and total_bodies <= refill_threshold:
                        score = 19000
                elif ct == CardType.STADIUM:
                    score = 1500
                else:
                    score = 1000

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is not None and card.id == HERO_CAPE:
                score = 7000
                if pokemon is not None and pokemon.id in (ATTACKER, BASIC_MON):
                    score += 200
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                # Strongly prefer fulfilling the planned attack's energy need.
                if plan.needs_energy:
                    if plan.attacker == 0 and o.inPlayArea == AreaType.ACTIVE:
                        score += 300
                    elif plan.attacker == 1 + (o.inPlayIndex or 0) and o.inPlayArea == AreaType.BENCH:
                        score += 300

        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            # Evolving 344 -> 345 is almost always correct.
            score = 9000 + (len(pokemon.energies) if isinstance(pokemon, Pokemon) else 0) * 10

        elif o.type == OptionType.ABILITY:
            score = 15000  # use abilities when offered

        elif o.type == OptionType.RETREAT:
            # Only retreat to promote a better attacker.
            score = 2000 if plan.attacker >= 1 else -1

        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == plan.attack_id:
                score += 500

        elif o.type == OptionType.END:
            score = -1000  # only end when nothing better remains

        else:
            score = 0

        scores.append(score)

    # Pick the highest-scoring options, respecting maxCount / minCount.
    desc = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    chosen = desc[: max(select.minCount, min(select.maxCount, len(desc)))]
    if len(chosen) < select.minCount:
        chosen = desc[: select.minCount]
    return chosen
