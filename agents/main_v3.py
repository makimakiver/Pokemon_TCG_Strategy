"""Improved Typhlosion/Crustle agent derived from lost-replay analysis.

This is main_v2.py with an archetype-aware overlay.  The lost-game replays
(``data/loser/*.json``) revealed five recurring failure modes that the generic
"bare" engine cannot see:

1. **Buddy Blast blindness.**  Ethan's Typhlosion's signature attack "Buddy
   Blast" deals *40 + 60 per Ethan's Adventure in the discard* but only costs
   a single {R} energy.  The generic engine only reads base ``damage=40`` from
   the card table, so it never picks Buddy Blast.  v3 computes the real
   fuel-aware damage.

2. **No discard-fuelling.**  Because Buddy Blast looked weak, the agent had no
   reason to play "Ethan's Adventure" supporters.  v3 boosts their priority
   so each copy adds +60 to the follow-up KO.

3. **Slow Typhlosion setup.**  Rare Candy lets a basic Cyndaquil leap straight
   to Typhlosion (skipping Quilava).  The generic engine evolved one stage per
   turn.  v3 explicitly prefers the Candy -> Typhlosion path.

4. **No gust KOs.**  Several losses left a prize on a tanky benched ex the
   agent never forced active.  v3 adds Boss's-Orders gust scoring for
   guaranteed knockouts from the bench.

5. **No-active losses.**  Several games ended with the active KO'd and an
   empty bench.  v3 keeps the bench fuller and retreats a dying active to
   promote a healthy Typhlosion.
"""
import json
import os
from collections import defaultdict

from cg.api import (
    AreaType, Card, CardType, EnergyType, Observation, OptionType, Pokemon,
    SelectContext, all_attack, all_card_data, to_observation_class,
)

# ---- Crustle / Ethan's Typhlosion two-line deck (from winners.json) ----
my_deck = (
    [1]*4 + [2]*8 + [18]*4 +                              # energy: Grass / Fire / Grow Grass
    [344]*4 + [345]*4 +                                   # Dwebble -> Crustle
    [352]*4 + [353]*4 + [354]*4 +                         # Cyndaquil -> Quilava -> Typhlosion
    [1079]*2 + [1121]*3 + [1147]*2 + [1152]*2 + [1159] +  # Rare Candy/Ultra Ball/Ice Cream/Pad/Cape
    [1182]*4 + [1192]*2 + [1215]*4 + [1227]*4             # Boss/Carmine/Ethan's Adv./Lillie's
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


# =============================================================================
# Archetype overlay -- Ethan's Typhlosion "Buddy Blast" awareness
# =============================================================================
HAS_TYPHLOSION = 354 in _deck_set
TYPHLOSION_ID = 354
QUILAVA_ID = 353
CYNDAQUIL_ID = 352
ETHANS_ADVENTURE = 1215      # supporter: draw AND +60 Buddy Blast fuel per copy in discard
RARE_CANDY = 1079            # item: basic -> stage2 skip
ULTRA_BALL = 1121            # item: search any pokemon
BOSS_ORDERS = 1182           # supporter: gust opponent's benched pokemon

# Resolve Typhlosion attack IDs at import time.
_BUDDY_BLAST_AID = None      # aid 490: dmg=40, cost=[R] -- scales with discard
_STEAM_ARTILLERY_AID = None  # aid 491: dmg=160, cost=[R,R,C]
if HAS_TYPHLOSION:
    _typh = card_table.get(TYPHLOSION_ID)
    if _typh and _typh.attacks:
        for _aid in _typh.attacks:
            _a = attack_table.get(_aid)
            if _a is None:
                continue
            _d = getattr(_a, "damage", 0) or 0
            if _d == 40 and _BUDDY_BLAST_AID is None:
                _BUDDY_BLAST_AID = _aid
            elif _d >= 150 and _STEAM_ARTILLERY_AID is None:
                _STEAM_ARTILLERY_AID = _aid


def _ethan_in_discard(my_state):
    """Count Ethan's Adventure cards currently in my discard pile."""
    return sum(1 for c in (my_state.discard or []) if c.id == ETHANS_ADVENTURE)


def _buddy_blast_damage(fuel):
    """Buddy Blast: 40 + 60 per Ethan's Adventure in the discard."""
    return 40 + 60 * fuel


def _is_buddy_blast(atk):
    """True if this attack is Typhlosion's Buddy Blast (base 40 damage)."""
    return (HAS_TYPHLOSION
            and atk is not None
            and (getattr(atk, "damage", 0) or 0) == 40
            and _BUDDY_BLAST_AID is not None)


def _hand_has(my_state, cid):
    return any(c.id == cid for c in (my_state.hand or []))


def _field_has(my_state, cid):
    """True if cid is active or on the bench."""
    for c in my_state.active:
        if c is not None and c.id == cid:
            return True
    for c in my_state.bench:
        if c is not None and c.id == cid:
            return True
    return False


class AttackPlan:
    attacker = -1
    target = -1
    attack_id = -1
    remain_hp = -1
    needs_energy = False
    gust = False          # True when this KO requires a Boss's Orders gust


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


def attack_damage(attacker, atk, target, ethan_fuel=0):
    """Damage of *atk* from *attacker* into *target*.

    For Buddy Blast (Typhlosion only), the real damage is 40 + 60*ethan_fuel.
    """
    # Only Typhlosion's own 40-damage attack gets the Buddy Blast scaling --
    # Quilava's Combustion also has base 40 but must NOT be treated as Buddy Blast.
    if HAS_TYPHLOSION and attacker.id == TYPHLOSION_ID and _is_buddy_blast(atk):
        dmg = _buddy_blast_damage(ethan_fuel)
    else:
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

    # ---- Archetype: count Ethan's Adventure fuel in the discard ----
    ethan_fuel = _ethan_in_discard(my_state) if HAS_TYPHLOSION else 0
    typhlosion_fielded = _field_has(my_state, TYPHLOSION_ID) if HAS_TYPHLOSION else False
    have_boss = _hand_has(my_state, BOSS_ORDERS)

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
                    if opp is None:
                        continue
                    # Benched targets (j!=0) require a Boss's Orders gust.
                    needs_gust = (j != 0)
                    if needs_gust and (not have_boss or state.supporterPlayed):
                        continue
                    dmg = attack_damage(mine, atk, opp, ethan_fuel)
                    if dmg <= 0:
                        continue
                    score = pokemon_score(opp)
                    if opp.hp <= dmg:
                        prize = prize_count(opp)
                        score += 500
                        if op_prize_left <= prize:
                            score = 50000
                        if needs_gust:
                            score -= 150      # gust costs a supporter slot
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
                        plan.gust = needs_gust

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
        # ---- Feed the chosen attacker ----
        if plan.needs_energy:
            if plan.attacker == 0 and is_active:
                score += 400
            elif plan.attacker >= 1 and plan.attacker - 1 < len(my_state.bench) and \
                    pokemon is my_state.bench[plan.attacker - 1]:
                score += 400
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
                        # ---- Prefer promoting a healthy Typhlosion ----
                        if card.id == TYPHLOSION_ID:
                            score += 300
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
                        # ---- Search priority: Typhlosion first ----
                        if cid == TYPHLOSION_ID:
                            score += 300
                        elif cid == QUILAVA_ID:
                            score += 80
                    elif cid in BASIC_MONS:
                        score = 260 - (field_counts[cid] + hand_counts[cid]) * 60
                    elif cid in ALL_ENERGY:
                        score = 120 if not state.energyAttached else 60
                    elif card_type(cid) == CardType.SUPPORTER:
                        score = 140
                        # ---- Ethan's Adventure is premium search target ----
                        if cid == ETHANS_ADVENTURE:
                            score += 200
                    elif card_type(cid) == CardType.ITEM:
                        score = 130
                        # ---- Rare Candy if we can Candy into Typhlosion ----
                        if cid == RARE_CANDY and _field_has(my_state, CYNDAQUIL_ID):
                            score += 150
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
                    # ---- Keep bench fuller to prevent no-active losses ----
                    if card.id in BASIC_MONS and n_bodies >= 4:
                        score = 30
                    # ---- Early game: bench basics ASAP when thin ----
                    if card.id in BASIC_MONS and n_bodies <= 2:
                        score = 25000
                elif ct == CardType.ITEM:
                    score = 3500
                    # ---- Rare Candy: top priority when Candy -> Typhlosion is live ----
                    # But never Candy when the bench is thin (causes no-active losses).
                    if card.id == RARE_CANDY and _hand_has(my_state, TYPHLOSION_ID) \
                            and _field_has(my_state, CYNDAQUIL_ID):
                        score = 60000 if n_bodies >= 3 else 5000
                    # ---- Ultra Ball: search for missing Typhlosion ----
                    if card.id == ULTRA_BALL and hand_counts.get(TYPHLOSION_ID, 0) == 0 \
                            and not typhlosion_fielded:
                        score += 2000
                elif ct == CardType.SUPPORTER:
                    score = 3000
                    # ---- Boss's Orders for gust KO ----
                    if card.id == BOSS_ORDERS and plan.gust:
                        score = 50000
                    # ---- Ethan's Adventure: draw AND +60 Buddy Blast fuel ----
                    if card.id == ETHANS_ADVENTURE:
                        score += 2000
                        if typhlosion_fielded:
                            score += 1500
                elif ct == CardType.STADIUM:
                    score = 1500
                else:
                    score = 1000

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is not None and card.id in TOOLS:
                score = 7000
                # ---- Hero's Cape on Typhlosion for survivability ----
                if pokemon is not None and pokemon.id == TYPHLOSION_ID:
                    score += 400
                elif pokemon is not None and pokemon.id in ATTACKERS:
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
            # ---- Evolving into Typhlosion is the most valuable evolve ----
            # But never when the bench is thin (avoid no-active losses).
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is not None and card.id == TYPHLOSION_ID:
                score = 55000 if n_bodies >= 3 else 8000

        elif o.type == OptionType.ABILITY:
            score = 15000 if turn_actions <= 8 else -2000

        elif o.type == OptionType.RETREAT:
            score = 2000 if plan.attacker >= 1 else -1
            # ---- Retreat a dying active to promote a healthy Typhlosion ----
            active = my_state.active[0] if my_state.active else None
            if active is not None and isinstance(active, Pokemon):
                cd = card_table.get(active.id)
                ret_cost = getattr(cd, "retreatCost", 0) if cd else 0
                if active.hp <= 40:
                    can_ret = can_pay(list(active.energies),
                                      [EnergyType.COLORLESS] * ret_cost) if ret_cost else True
                    if can_ret and any(b is not None and b.id == TYPHLOSION_ID and b.hp >= 60
                                       for b in my_state.bench):
                        score = 40000
                    elif can_ret and any(b is not None and b.id in ATTACKERS and b.hp > active.hp
                                         for b in my_state.bench):
                        score = 12000

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
