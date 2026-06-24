"""Slowking control-toolbox — v2 = v1 deck + research-driven pilot overlays.

Built on agents.bare_agent's generic decision function, with two targeted changes derived
from docs/SLOWKING_ROSS_NAIC_RESEARCH.md (Ross Cawthon NAIC 2026):

  LEVER 1 — PRIZE-MAP DISCIPLINE (the deck's #1 stated rule):
    "offer one-Prizers, take two-Prize turns; do NOT over-bench Rule-Box ex's."
    The generic pilot benches any Pokemon at 20000. We ration the five 2-prize ex's
    (Mega Kangaskhan ex, Latias ex, Meowth ex, Fezandipiti ex, Lillie's Clefairy ex):
    keep at most ONE on board (the Kanga draw/pivot engine); deprioritize benching more.

  LEVER 2 — SLOWKING LINE (the one-prize attacker is the deck's identity):
    boost evolving Slowpoke -> Slowking so the Seek-Inspiration attacker actually lands.

NOT yet implemented: LEVER 3 (Seek-Inspiration top-card setup via Ciphermaniac/Academy +
copied-attack selection) — the core engine and the real ceiling; needs dedicated engine work.

Same deck as v1 (data/decks/deck_slowking.json). Measured in the gauntlet vs v1's baseline.
"""
import os

os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_slowking.json"))

import agents.bare_agent as B
from agents.bare_agent import (
    to_observation_class, card_table, attack_table, ATTACKERS, BASIC_MONS, ALL_ENERGY,
    DECK_ETYPES, can_pay, attack_damage, pokemon_score, prize_count, get_card,
    AreaType, OptionType, SelectContext, Pokemon, card_type, AttackPlan,
)
from collections import defaultdict

my_deck = B.my_deck

# Five 2-prize Rule-Box ex's in this deck. Keep ~1 (Kanga engine); don't gift the rest.
EX_RATION = {756, 184, 1071, 140, 272}   # Kanga, Latias, Meowth, Fezandipiti, Clefairy
SLOWPOKE, SLOWKING = 162, 163

plan = AttackPlan()
pre_turn = -1
turn_actions = 0


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
    stalling = turn_actions > 80

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    for c in my_state.active + list(my_state.bench):
        if c is not None:
            field_counts[c.id] += 1
    for c in (my_state.hand or []):
        hand_counts[c.id] += 1

    n_bodies = sum(1 for c in my_state.active + list(my_state.bench) if c is not None)
    n_ex_in_play = sum(field_counts[e] for e in EX_RATION)          # v2: prize-discipline counter
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
                    if mine.id == SLOWKING:               # v2 lever 2: prefer the 1-prize identity attacker
                        score += 120
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
        if pokemon.id == SLOWKING:                        # v2: feed the Slowking attacker first
            score += 120
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
                    # v2 lever 1: setting up the bench, ration extra ex's
                    if card.id in EX_RATION and n_ex_in_play >= 1:
                        score = 20
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
                    # v2 lever 1: PRIZE DISCIPLINE — don't bench a 2nd+ Rule-Box ex.
                    if card.id in EX_RATION and n_ex_in_play >= 1:
                        score = 25
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
            if card is not None and card.id in B.TOOLS:
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
            if pokemon is not None and pokemon.id == SLOWPOKE:    # v2 lever 2: land Slowking
                score += 800

        elif o.type == OptionType.ABILITY:
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


# CardType is referenced via card_type()'s results above; import it explicitly for the
# TO_HAND / PLAY branches that compare against CardType.* members.
from cg.api import CardType
