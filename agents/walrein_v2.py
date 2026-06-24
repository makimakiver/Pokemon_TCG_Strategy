"""Walrein/Spheal — v2 = bare_agent + MEGATON-priority (data-driven: more Megaton = more wins) (Frigid Fangs is the win condition).

Built on agents.bare_agent. The deck's identity attack, Walrein 'Frigid Fangs' [W] 60dmg, LOCKS
the opponent: their Pokemon with <=2 energy can't attack next turn. The lock IS the card — the
60 damage is irrelevant. bare_agent scores Fangs by its 60 printed damage and picks Megaton Fall
(170) instead, throwing the lock away. v1 fixes that:

  L1 LOCK-PRIORITY: when Walrein can fire Frigid Fangs and the opp active has <=2 energy, score
     Fangs by LOCK VALUE (it removes the opponent's next turn), not 60dmg. The lock is INFINITE
     when the opp threatens a KO we can't otherwise survive.
  L2 EVOLUTION ACCEL: boost finding the Spheal->Sealeo->Walrein stage pieces (Ultra Ball/draw)
     so the 170hp Stage-2 lands by turn 2-3.
  L3 MEGATON FALL RULE: prefer Megaton (170 +50 self) ONLY when the opp is locked this turn, or
     it KOs and Fangs doesn't, or Walrein will die anyway. Otherwise maintain the Fangs lock.

Deck = data/decks/deck_walrein.json (4 Spheal/3 Sealeo/4 Walrein, Water energy, draw/search).
Baseline walrein(bare) = 33% mean, zero draws. Planner estimate: 70% reachable with tuning.
"""
import os

os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_walrein.json"))

import agents.bare_agent as B
from agents.bare_agent import (
    to_observation_class, card_table, attack_table, ATTACKERS, BASIC_MONS, ALL_ENERGY,
    DECK_ETYPES, can_pay, attack_damage, pokemon_score, prize_count, get_card,
    AreaType, OptionType, SelectContext, Pokemon, card_type, AttackPlan,
)
from collections import defaultdict

my_deck = B.my_deck

# Walrein line + identity attacks
SPHEAL, SEALEO, WALREIN = 941, 942, 943
FRIGID_FANGS = None   # resolved at runtime below
MEGATON_FALL = None
for _aid in (card_table.get(WALREIN, None).attacks or []):
    _a = attack_table.get(_aid)
    if _a and getattr(_a, "name", "") == "Frigid Fangs":
        FRIGID_FANGS = _aid
    elif _a and getattr(_a, "name", "") == "Megaton Fall":
        MEGATON_FALL = _aid

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
    n_ex_in_play = 0   # (no ex-ration in this deck; placeholder for parity)
    have_spare_energy = any(hand_counts[e] >= 1 for e in ALL_ENERGY)
    # v1 L2 EVO-ACCEL: do we have an un-evolved Spheal/Sealeo in play without its next stage in hand?
    hand_ids = set(hand_counts.keys())
    spheal_in_play = any(c is not None and c.id == SPHEAL for c in (list(my_state.active) + list(my_state.bench)))
    sealeo_in_play = any(c is not None and c.id == SEALEO for c in (list(my_state.active) + list(my_state.bench)))
    walrein_in_play = any(c is not None and c.id == WALREIN for c in (list(my_state.active) + list(my_state.bench)))
    need_evo = ((spheal_in_play and SEALEO not in hand_ids) or
                (sealeo_in_play and WALREIN not in hand_ids)) and not walrein_in_play

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
                    # v2 MEGATON-PRIORITY (data-driven: more Megaton fired = more wins).
                    is_fangs = (mine.id == WALREIN and aid == FRIGID_FANGS)
                    is_megaton = (mine.id == WALREIN and aid == MEGATON_FALL)
                    if is_megaton:
                        # Megaton = 170 KO power. Strongly prefer it when payable (Walrein has 2E).
                        # The race is won by 170 KOs, not 60-dmg Fangs chips.
                        walrein_has_2e = len(mine.energies) >= 2
                        if walrein_has_2e:
                            score += 700
                            # guard: don't self-KO Walrein wastefully (Megaton does 50 self).
                            # If Walrein hp <= 50, Megaton self-KOs -> only if it wins or KOs a 2-prizer.
                            walrein_hp = mine.hp if hasattr(mine, 'hp') else 170
                            if walrein_hp <= 50 and not (opp.hp <= dmg and op_prize_left <= prize_count(opp)):
                                score -= 600   # wasteful self-KO; prefer Fangs/retreat instead
                    if is_fangs:
                        # Fangs is the cheap fallback (1E) / lock. DON'T over-prioritize it (v1's bug).
                        # Only boost if Megaton is NOT payable this turn AND Fangs locks a threat.
                        walrein_has_2e = len(mine.energies) >= 2
                        op_en = len(opp.energies) if hasattr(opp, 'energies') else 0
                        if not walrein_has_2e and op_en <= 2:
                            score += 200   # cheap lock while we dig for the 2nd energy
                    if i == 0:
                        score += 200
                    score += len(mine.energies)
                    # v1 L3 MEGATON RULE: prefer Megaton only when safe (opp locked) or it KOs
                    if is_megaton:
                        op_en = len(opp.energies) if hasattr(opp, 'energies') else 0
                        if opp.hp <= dmg:
                            score += 250            # KO with Megaton is fine
                        elif op_en <= 2:
                            score += 100            # opp would be locked next turn -> self-50 is free
                        else:
                            score -= 400             # risky: take 50 self, then get hit (opp has >=3E)
                    if mine.id == WALREIN:                       # identity attacker priority
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
        if pokemon.id == WALREIN:
            score += 200
            # v2 E1 ENERGY-CONSOLIDATION: push the ACTIVE Walrein to 2 energy hard so Megaton
            # [WW] is payable every turn (data: Megaton frequency correlates w/ wins).
            if is_active and len(pokemon.energies) < 2:
                score += 600
        elif pokemon.id in (SPHEAL, SEALEO):
            score += 80
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
                    pass   # (walrein deck: no ex-ration discipline needed)
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
                    # v2 E3 WALREIN #2 RESERVE: if our active is Walrein, bench a Spheal to start
                    # the backup line before the active faints (Megaton self-damages -> it will).
                    # Keep priority modest (don't override attacking/attaching in MAIN).
                    if card.id == SPHEAL and any(c is not None and c.id == WALREIN for c in (list(my_state.active))):
                        if not any(c is not None and (c.id==SPHEAL or c.id==SEALEO or c.id==WALREIN) for c in (my_state.bench or [])):
                            score = max(score, 21000)
                    # v2 lever 1: PRIZE DISCIPLINE — don't bench a 2nd+ Rule-Box ex.
                    pass   # (walrein deck: no ex-ration discipline needed)
                elif ct == CardType.ITEM:
                    score = 3500
                    # v1 L2 EVO-ACCEL: if we need a stage piece (active Spheal/Sealeo without its
                    # evo in hand), boost Ultra Ball / Poke Pad to find it.
                    if card.id in (1121, 1152) and need_evo:
                        score = 9000
                elif ct == CardType.SUPPORTER:
                    score = 3000
                    if need_evo:           # draw supporters find the stage piece too
                        score = 6000
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
            if pokemon is not None and pokemon.id in (SPHEAL, SEALEO):  # v1 L2: land Walrein
                score += 800
            if pokemon is not None and pokemon.id == SEALEO and WALREIN in (my_state.hand and [c.id for c in my_state.hand] or []):
                score += 400

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
