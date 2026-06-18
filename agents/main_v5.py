"""main_v5 — the scripted INITIAL POINT for the RL Solver agent.

The competition submission is a (deck + agent) pair. `rl/` learns a pointer-net
Solver, but a net trained from scratch on this engine wanders for a long time
before it discovers the deck's actual win condition. `main_v5` is the strong
hand-tuned pilot the net is *bootstrapped from*: behavioral-cloned into a fresh
`PointerPolicy` (`rl.sft`) to reach ~scripted strength, then handed to
`rl.train_solver(init_policy=...)` as the warm start. It also doubles as the
same-deck anchor opponent (`RL_OPP=agents.main_v5`) and the reference pilot for
the prior residual.

Two things make it the right initial point (vs `agents.bare_agent` or the
`rl.llm_agent` Claude teacher):

  1. It pilots the EXACT pinned Solver deck — `rl.config.solver_deck_path()`
     (Team Rocket's Honchkrow, limitless 26267, the Ultra-Ball flex variant).
     The bare engine would load a different list; cloning a different-deck pilot
     teaches the net the wrong board.
  2. It encodes the combo the deck actually wins with, which a damage-stat pilot
     is blind to (see below). No external API calls, so the bootstrap is free and
     reproducible inside the Docker engine image.

Win condition encoded — a DUAL-attacker, one-prize archetype that pivots between
an early burst and a late-game scaling sweep (per AzulGG's deck breakdown; only
the cards actually present in the pinned 60 are modelled — no Porygon Z / Brave
Bangle in this list, so they're omitted):

  EARLY (hand-fuel burst) — Team Rocket's Honchkrow 'Rocket Feathers' [CC]:
    listed 0, deals 60 x (Team Rocket Supporters discarded FROM HAND). 4 disc =
    240, 6 = 360 -> OHKOs any ex.
  LATE (discard-fuel scaling) — Team Rocket's Porygon2 'R Command' [CC]:
    listed 0, deals 20 x (Team Rocket Supporters in your DISCARD PILE). Ignores
    hand size, so it keeps firing after Iono / Roxy shrink the hand — the fuel is
    everything you already discarded. Reads the pile WITHOUT spending it (free,
    repeatable). The two attackers share one resource pool: every supporter
    discarded for Rocket Feathers (or played for its text) becomes R Command fuel.

Plan:
  1. Evolve Murkrow -> Honchkrow and Porygon -> Porygon2; power the active line
     with special energy (Team Rocket's Energy = 2 units; Ignition = scaling).
  2. HOARD Team Rocket Supporters in hand as Rocket Feathers ammo — fetched by
     Murkrow 'Deceit', Transceiver, Petrel, Roto-Stick, Proton (basics), refilled
     by Ariana's draw-to-8; Team Rocket's Factory draws 2 per supporter played.
  3. Burst with Rocket Feathers, discarding EXACTLY enough ammo to KO (over-
     discard -> redraw -> deck-out, this deck's main self-loss); as the discard
     fills, hand the late game to Porygon2's R Command.
  Hammer In [DCC]=100 is the vanilla fallback; Articuno is a Psychic tech body.

The pilot computes the TRUE damage of both 0-listed scaling attacks and lets the
existing arg-max pick the bigger one each turn, so the early<->late pivot is
emergent, not hand-scripted. Built on the generic bare_agent engine (runtime
card/attack stats, evolve-ASAP, feed-the-attacker energy). Imports `cg` (Linux-
only native lib) -> runs inside the Docker engine image, like the rest of `rl/`.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

from cg.api import (
    AreaType, Card, CardType, EnergyType, Observation, OptionType, Pokemon,
    SelectContext, all_attack, all_card_data, to_observation_class,
)


# ---- Bind to the EXACT pinned Solver deck (the whole point of main_v5) -------
def _resolve_solver_deck_path() -> str:
    """BARE_DECK override -> rl.config pinned solver deck -> default file path."""
    override = os.environ.get("BARE_DECK", "")
    if override and os.path.exists(override):
        return override
    try:  # rl.config is pure-Python and honors RL_SOLVER_DECK
        from rl.config import solver_deck_path
        p = str(solver_deck_path())
        if os.path.exists(p):
            return p
    except Exception:
        pass
    return str(Path(__file__).resolve().parents[1] / "data" / "decks" / "deck_solver_honchkrow.json")


_deck_path = _resolve_solver_deck_path()
if os.path.exists(_deck_path):
    my_deck = json.load(open(_deck_path))
else:  # inline fallback == limitless 26267 (Ultra Ball 1121 flex vs Air Balloon 1174)
    my_deck = [463]*4 + [891]*3 + [473]*2 + [474]*1 + [414]*2 + \
              [1220]*4 + [1216]*4 + [1218]*4 + [1219]*4 + [1217]*4 + \
              [1152]*4 + [1077]*4 + [1134]*4 + [1097]*3 + [1109]*1 + \
              [1121]*1 + [1257]*3 + [15]*4 + [17]*4
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


# ---- Honchkrow combo specifics ----
def _card_named(name):
    return next((cid for cid in _deck_set
                 if card_table.get(cid) and card_table[cid].name == name), None)


def _attack_named(card_id, atk_name):
    c = card_table.get(card_id)
    for aid in (c.attacks or []) if c else []:
        a = attack_table.get(aid)
        if a and a.name == atk_name:
            return aid
    return None


HONCHKROW = _card_named("Team Rocket's Honchkrow")
MURKROW = _card_named("Team Rocket's Murkrow")
PORYGON = _card_named("Team Rocket's Porygon")             # basic; evolves to Porygon2
PORYGON2 = _card_named("Team Rocket's Porygon2")           # Stage 1; R Command scaling sweeper
RF_ATTACK = _attack_named(HONCHKROW, "Rocket Feathers")    # 60 x TR-Supporters discarded from HAND
HAMMER_IN = _attack_named(HONCHKROW, "Hammer In")          # vanilla [DCC] = 100
R_COMMAND = _attack_named(PORYGON2, "R Command")           # 20 x TR-Supporters in DISCARD pile
DECEIT = _attack_named(MURKROW, "Deceit")                  # search a Supporter into hand
RF_PER_CARD = 60
RC_PER_CARD = 20

# The Honchkrow & Porygon2 lines are the two scaling attackers; energy goes here.
COMBO_ATTACKERS = {cid for cid in (HONCHKROW, PORYGON2, MURKROW, PORYGON) if cid}

# Energy split: Team Rocket's Energy (the scarce enabler) vs Ignition Energy (clogs the
# hand). Guide: aggressively pitch SURPLUS Ignition mid-game to keep draws live.
TR_ENERGY = _card_named("Team Rocket's Energy")
IGNITION_ENERGY = _card_named("Ignition Energy")
# Team Rocket's Factory: stadium that draws 2 each time a TR Supporter is played (engine).
FACTORY = _card_named("Team Rocket's Factory")

# Team Rocket Supporters = ammunition for Rocket Feathers (name contains "Team Rocket").
TR_SUPPORTERS = {cid for cid in _deck_set
                 if card_table.get(cid)
                 and int(card_table[cid].cardType) == int(CardType.SUPPORTER)
                 and "Team Rocket" in card_table[cid].name}

# Both scaling attackers list 0 damage, so the generic >=90 detector misses them
# (Porygon2 is a Stage-1 so already in ATTACKERS, but be explicit).
if HONCHKROW:
    ATTACKERS.add(HONCHKROW)
if PORYGON2:
    ATTACKERS.add(PORYGON2)

# This deck runs only special energy; give the "+1 energy" lookahead something to test.
if not DECK_ETYPES:
    DECK_ETYPES = {EnergyType.COLORLESS, EnergyType.DARKNESS, EnergyType.PSYCHIC}


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


def _apply_weakness(attacker, base, target):
    my_type = card_table[attacker.id].energyType if attacker.id in card_table else None
    tdata = card_table.get(target.id)
    if tdata and my_type is not None:
        if tdata.weakness == my_type:
            return base * 2
        if tdata.resistance == my_type:
            return max(0, base - 30)
    return base


def rocket_feathers_damage(attacker, target, ammo):
    """Honchkrow 'Rocket Feathers': 60 x (TR Supporters discarded from hand), with weakness."""
    return _apply_weakness(attacker, RF_PER_CARD * max(0, ammo), target)


def _tr_in_discard(my_state):
    """R Command fuel: count of Team Rocket Supporters sitting in our discard pile."""
    return sum(1 for c in (my_state.discard or []) if c.id in TR_SUPPORTERS)


def r_command_damage(attacker, target, fuel):
    """Porygon2 'R Command': 20 x (TR Supporters in discard pile), with weakness.

    Unlike Rocket Feathers this does NOT consume the fuel — it just reads the pile,
    so it is free and repeatable once the discard is stocked (the late-game engine).
    """
    return _apply_weakness(attacker, RC_PER_CARD * max(0, fuel), target)


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
    have_special_energy = any(hand_counts[e] >= 1 for e in SPECIAL_ENERGY)
    # Team Rocket Supporters in hand = Rocket Feathers ammo (60 each, consumed on use).
    tr_ammo = sum(hand_counts[c] for c in TR_SUPPORTERS)
    # Team Rocket Supporters in discard = R Command fuel (20 each, NOT consumed).
    rc_fuel = _tr_in_discard(my_state)

    # Stop digging only when the deck is genuinely about to run out (over-search is this
    # deck's main self-inflicted loss). Conserving Rocket Feathers ammo at discard time
    # (below) does most of the deck-out prevention; this is just a final backstop.
    deck_count = getattr(my_state, "deckCount", 99)

    # Rocket Feathers' discard step: the engine offers only Team Rocket Supporters from
    # hand ("discard any number ... 60 each"). Discard EXACTLY enough to KO the target and
    # keep the rest as ammo for next turn (over-discarding causes redraws -> deck-out).
    if context == SelectContext.DISCARD and select.minCount == 0 and select.option:
        opts = select.option
        tr_idx = []
        for k, o in enumerate(opts):
            c = get_card(obs, o.area, o.index, o.playerIndex) if o.type == OptionType.CARD else None
            if c is not None and c.id in TR_SUPPORTERS and o.playerIndex == my_index:
                tr_idx.append(k)
        if tr_idx and len(tr_idx) == sum(1 for o in opts if o.type == OptionType.CARD):
            opp_active = op_state.active[0] if op_state.active else None
            need = len(tr_idx)
            if opp_active is not None:
                per = RF_PER_CARD
                tdata = card_table.get(opp_active.id)
                htype = card_table[HONCHKROW].energyType if HONCHKROW in card_table else None
                if tdata and htype is not None and tdata.weakness == htype:
                    per *= 2
                need = (opp_active.hp + per - 1) // per      # ceil(hp / per) to KO
            need = max(select.minCount, min(need, len(tr_idx), select.maxCount))
            return tr_idx[:need]

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
                # Porygon2 plays R Command for [CC] (1 Team Rocket's Energy = 2 units pays it);
                # the shared attackId may carry Porygon-Z's [CCC] cost, so override for Porygon2.
                if mine.id == PORYGON2 and aid == R_COMMAND:
                    cost = [EnergyType.COLORLESS, EnergyType.COLORLESS]
                attached_types = list(mine.energies)
                more_energy = False
                if not can_pay(attached_types, cost):
                    # Try assuming we attach one more energy this turn. A special energy
                    # provides 2+ units (TR Energy = 2 P/D, Ignition on an Evolution = 3),
                    # so approximate it as two colorless when testing colorless costs.
                    if not state.energyAttached:
                        extras = [[et] for et in DECK_ETYPES]
                        if have_special_energy:
                            extras.append([EnergyType.COLORLESS, EnergyType.COLORLESS])
                        if (have_spare_energy or have_special_energy) and \
                           any(can_pay(attached_types + ex, cost) for ex in extras):
                            more_energy = True
                        else:
                            continue
                    else:
                        continue
                # Both scaling attacks list 0 damage; compute their REAL output.
                #   Rocket Feathers: 60 per TR Supporter discarded from hand (consumes ammo).
                #   R Command:       20 per TR Supporter already in the discard pile (free).
                is_rf = (mine.id == HONCHKROW and aid == RF_ATTACK)
                is_rc = (mine.id == PORYGON2 and aid == R_COMMAND)
                if is_rf and tr_ammo <= 0:
                    continue
                if is_rc and rc_fuel <= 0:
                    continue
                for j, opp in enumerate(op_cards):
                    if opp is None or j != 0:
                        continue
                    if is_rf:
                        dmg = rocket_feathers_damage(mine, opp, tr_ammo)
                    elif is_rc:
                        dmg = r_command_damage(mine, opp, rc_fuel)
                    else:
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
                    # Conserve ammo: on equal footing prefer the vanilla attack, and don't
                    # over-discard supporters for an already-lethal Rocket Feathers. R Command
                    # spends nothing, so it carries no such penalty — once the discard is
                    # stocked it is the cheaper, hand-independent sweep (the late-game pivot).
                    if is_rf:
                        score -= 6 * tr_ammo
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
        # Energy is scarce (all special), so concentrate it on the two scaling attackers.
        # Porygon2 is a real late-game sweeper (R Command), NOT a wall — feed it. Articuno
        # is a Psychic tech body that rarely needs our special energy.
        need = 2                  # Rocket Feathers costs [CC]
        if pokemon.id == HONCHKROW:
            score += 400
        elif pokemon.id == PORYGON2:
            score += 380          # R Command sweeper — the late-game win condition
            need = 2              # Porygon2 plays R Command for [CC] (not Porygon-Z's [CCC])
        elif pokemon.id == MURKROW:
            score += 250          # becomes Honchkrow — ready to fire on evolve
        elif pokemon.id == PORYGON:
            score += 230          # becomes Porygon2
        elif pokemon.id in ATTACKERS:
            score += 300
        else:
            score += 20           # Articuno etc: don't strand special energy here
        if is_active:
            score += 40
        # Once an attacker can pay its scaling attack, spread the next energy to a backup.
        if len(pokemon.energies) >= need:
            score -= 220
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
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    # The active must become Honchkrow, so start with Murkrow (it can also
                    # Deceit for [C] turn 1). Articuno/Porygon as the active stall us — they
                    # can't attack on one special energy and force a wasteful retreat later.
                    if card.id == MURKROW:
                        score = 300
                    elif card.id in BASIC_MONS:
                        score = 100
                    else:
                        score = 50
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
                    elif cid in TR_SUPPORTERS:
                        score = 200          # fetch ammo for Rocket Feathers / R Command fuel
                    elif cid == FACTORY:
                        # Factory is the draw engine (draw 2 per TR Supporter). Grab it hard
                        # when none is in play; less urgent if one already is.
                        score = 210 if not (state.stadium or []) else 90
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
                    elif cid in TR_SUPPORTERS:
                        score = 60           # ammo for Rocket Feathers; pile feeds R Command
                    elif cid in ALL_ENERGY:
                        score = 10 + max(0, hand_counts[cid] - 1) * 20
                        # Guide: dump SURPLUS Ignition (it clogs draws); never the last one.
                        if cid == IGNITION_ENERGY and hand_counts[cid] >= 2:
                            score += 50
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
                    # Stop digging only when the deck is genuinely about to run out (the
                    # only time over-search actually loses us the game).
                    score = 600 if deck_count <= 8 else 3500
                elif ct == CardType.SUPPORTER:
                    score = 3000
                    # Team Rocket Supporters are Rocket Feathers ammo. Hold them only when a
                    # LETHAL Rocket Feathers is already lined up this turn (discard them in
                    # the attack). Otherwise play one to dig/refuel (Ariana draws to 8).
                    lethal_rf = plan.attack_id == RF_ATTACK and plan.remain_hp <= 0
                    if card.id in TR_SUPPORTERS and lethal_rf:
                        score = 120
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
            # Prefer evolving the ACTIVE Murkrow so Honchkrow ends up active (no retreat).
            if o.inPlayArea == AreaType.ACTIVE:
                score += 500

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
