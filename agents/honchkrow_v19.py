"""Team Rocket's Honchkrow — research-driven heuristic pilot (ground-up v4).

Encodes the competitive piloting research (DarkRye Kyoto / Vanoverschelde Utrecht
lists + Limitless Labs matchup data) onto the engine, cross-checked against the
data/Crow replays and the live cg card database. Deck = limitlesstcg 26267
(Utrecht, Porygon2-only) — the same 60 the Crow games were played on.

ENGINE-TRUE CORRECTIONS over submission_v5 (verified against cg.all_attack and the
data/Crow energy expansions):
  * R Command (atk 670) costs **CCC**, NOT CC. v5 hard-overrode it to CC, so it
    planned R Commands the engine refuses to pay. Here we use the real cost and
    rely on the engine ALREADY expanding special energy by context:
        - Ignition on an EVOLUTION  -> [C,C,C]  (pays R Command alone)   [Crow win_1 s74+]
        - Ignition on a BASIC       -> [C]
        - Team Rocket's Energy      -> 2 units of P/D
    so 1 Ignition on Porygon2 powers R Command; 1 TR Energy alone (2) does not.

ARCHETYPE PLAN — one-prize dual-attacker race (research §A):
  EARLY  Honchkrow 'Rocket Feathers' [CC] = 60 x (TR Supporters discarded FROM HAND).
  LATE   Porygon2  'R Command'      [CCC]= 20 x (TR Supporters in your DISCARD pile).
  Every supporter pitched for Rocket Feathers becomes R Command fuel — one pool.

KEY RESEARCH RULES ENCODED (section refs from the piloting doc):
  C4/C5/C6  attack preference: when each KOs, R Command > Hammer In > Rocket Feathers
            (preserve hand supporters = future ammo/fuel); RF discards the MINIMUM.
  C8        D_TR >= 9  (R Command >= 180) is the pivot into the Porygon2 plan.
  D1-D9     value-ordered discard: dead Proton > dead Archer > extra Ariana >
            extra Petrel > extra Giovanni > singles > live Archer/last Giovanni last.
  C16-C20   supporter logic: Giovanni gust-KO first; Proton T1-going-first; Petrel
            for a specific Trainer; Ariana draw-to-8 (only if all in-play are TR);
            Archer only when its KO-condition is live and the hand is dead.
  C12-C15   energy: Ignition -> evolution attacking NOW (1-card enabler); TR Energy
            -> Honchkrow line (persistent + the Darkness for Hammer In); never
            Ignition on a non-attacking basic; never invest in Articuno (no Water).
  Crustle   protect Fighting-weak Porygon2 (90 HP); keep Lightning-weak Honchkrow
            (safe vs Fighting) in front; single-prize race.

Built on the generic bare_agent engine (runtime card/attack stats, evolve-ASAP,
feed-the-attacker energy) so non-combo decisions still play legally. Imports cg
(Linux native lib) -> runs inside the Docker engine image.
"""
from collections import defaultdict

from cg.api import (
    AreaType, CardType, EnergyType, OptionType, Pokemon,
    SelectContext, all_attack, all_card_data, to_observation_class,
)

# ---- Deck (inlined) == limitlesstcg 26267, Utrecht / Crow list ----
my_deck = ([463] * 4 + [891] * 3 + [473] * 2 + [474] * 2 + [475] * 1 +
           [1220] * 4 + [1216] * 4 + [1218] * 4 + [1219] * 4 + [1217] * 4 +
           [1152] * 3 + [1077] * 3 + [1134] * 4 + [1097] * 2 + [1109] * 1 +
           [1121] * 1 + [1257] * 4 + [1175] * 1 + [1176] * 1 + [15] * 4 + [17] * 4)
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


# ---- Derive deck roles automatically ----
_deck_set = set(my_deck)
_poke = [cid for cid in _deck_set if card_table.get(cid) and card_table[cid].cardType == CardType.POKEMON]

BASIC_MONS = {cid for cid in _poke if _stage(cid) == 0}
ATTACKERS = {cid for cid in _poke if _stage(cid) >= 1}
ATTACKERS |= {cid for cid in BASIC_MONS if _best_dmg(cid) >= 90}
if not ATTACKERS:
    ATTACKERS = set(_poke)

ENERGY_BASICS = {cid for cid in _deck_set if card_table.get(cid) and card_table[cid].cardType == CardType.BASIC_ENERGY}
SPECIAL_ENERGY = {cid for cid in _deck_set if card_table.get(cid) and card_table[cid].cardType == CardType.SPECIAL_ENERGY}
ALL_ENERGY = ENERGY_BASICS | SPECIAL_ENERGY
TOOLS = {cid for cid in _deck_set if card_table.get(cid) and card_table[cid].cardType == CardType.TOOL}
DECK_ETYPES = {card_table[e].energyType for e in ENERGY_BASICS if card_table.get(e)}


# ---- Combo card / attack resolution (by name, print-robust) ----
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
PORYGON = _card_named("Team Rocket's Porygon")
PORYGON2 = _card_named("Team Rocket's Porygon2")
PORYGONZ = _card_named("Team Rocket's Porygon-Z")          # NEW: R Command at CC (cheaper)
ARTICUNO = _card_named("Team Rocket's Articuno")
BRAVE_BANGLE = _card_named("Brave Bangle")                 # +30 vs Active ex (non-Rule-Box attacker)
PUNK_HELMET = _card_named("Punk Helmet")                   # attacker that KOs the holder takes 40
MAXIMUM_BELT = _card_named("Maximum Belt")                 # +50 vs Active ex (if present)

RF_ATTACK = _attack_named(HONCHKROW, "Rocket Feathers")   # CC, 60 x TR supporters discarded from hand
HAMMER_IN = _attack_named(HONCHKROW, "Hammer In")          # DCC, 100 flat
R_COMMAND = _attack_named(PORYGON2, "R Command")           # CCC, 20 x TR supporters in discard
R_COMMAND_Z = _attack_named(PORYGONZ, "R Command")         # CC on Porygon-Z, 20 x TR supporters in discard
DECEIT = _attack_named(MURKROW, "Deceit")                  # C, search a Supporter to hand
RF_PER_CARD = 60
RC_PER_CARD = 20
# Both R Command attackers share the scaling; treat Porygon-Z like Porygon2.
RCOMMANDERS = {cid for cid in (PORYGON2, PORYGONZ) if cid}

# Supporters / engine pieces (by name).
PROTON = _card_named("Team Rocket's Proton")
ARIANA = _card_named("Team Rocket's Ariana")
GIOVANNI = _card_named("Team Rocket's Giovanni")
PETREL = _card_named("Team Rocket's Petrel")
ARCHER = _card_named("Team Rocket's Archer")
TRANSCEIVER = _card_named("Team Rocket's Transceiver")
FACTORY = _card_named("Team Rocket's Factory")
TR_ENERGY = _card_named("Team Rocket's Energy")
IGNITION_ENERGY = _card_named("Ignition Energy")

# Team Rocket Supporters = Rocket Feathers ammo / R Command fuel.
TR_SUPPORTERS = {cid for cid in _deck_set
                 if card_table.get(cid)
                 and int(card_table[cid].cardType) == int(CardType.SUPPORTER)
                 and "Team Rocket" in card_table[cid].name}

# Both scaling attackers list 0 damage; make sure they count as attackers.
for _x in (HONCHKROW, PORYGON2):
    if _x:
        ATTACKERS.add(_x)
# The two scaling attackers we feed energy to.
COMBO_ATTACKERS = {cid for cid in (HONCHKROW, PORYGON2, MURKROW, PORYGON) if cid}

if not DECK_ETYPES:
    DECK_ETYPES = {EnergyType.COLORLESS, EnergyType.DARKNESS, EnergyType.PSYCHIC}

# Discard VALUE rank (D1-D9): lower = pitch first. Singles handled dynamically.
_DISCARD_RANK = {PROTON: 0, ARCHER: 1, ARIANA: 2, PETREL: 3, GIOVANNI: 4}


class AttackPlan:
    attacker = -1
    target = -1
    attack_id = -1
    remain_hp = -1
    needs_energy = False


plan = AttackPlan()
pre_turn = -1
turn_actions = 0


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
    """Engine pre-expands special energy into `attached` (a list[EnergyType]); with
    the REAL attack cost this is exact (no v5-style override needed)."""
    need = list(cost)
    pool = list(attached)
    for c in [e for e in need if e != EnergyType.COLORLESS]:
        match = next((p for p in pool if p == c or p == EnergyType.RAINBOW), None)
        if match is None:
            return False
        pool.remove(match)
    colorless = sum(1 for e in need if e == EnergyType.COLORLESS)
    return len(pool) >= colorless


def _apply_weakness(attacker, base, target):
    my_type = card_table[attacker.id].energyType if attacker.id in card_table else None
    tdata = card_table.get(target.id)
    if tdata and my_type is not None:
        if tdata.weakness == my_type:
            return base * 2
        if tdata.resistance == my_type:
            return max(0, base - 30)
    return base


def attack_damage(attacker, atk, target):
    dmg = getattr(atk, "damage", 0) or 0
    if dmg <= 0:
        return 0
    return _apply_weakness(attacker, dmg, target)


def _tool_bonus(attacker, target):
    """Brave Bangle (+30) / Maximum Belt (+50) vs an Active ex (our attackers have no Rule
    Box). This is the threshold fix that lets a 6-discard Rocket Feathers (360+30=390) KO a
    Frost-Barriered Mega Abomasnow (350+30 shield) — 390-30 = 360 >= 350."""
    tdata = card_table.get(target.id)
    if not (tdata and (tdata.ex or tdata.megaEx)):
        return 0
    tools = {t.id for t in getattr(attacker, "tools", []) or []}
    bonus = 0
    if BRAVE_BANGLE in tools:
        bonus += 30
    if MAXIMUM_BELT in tools:
        bonus += 50
    return bonus


def rocket_feathers_damage(attacker, target, ammo):
    base = RF_PER_CARD * max(0, ammo)
    return _apply_weakness(attacker, base, target) + _tool_bonus(attacker, target)


def r_command_damage(attacker, target, fuel):
    base = RC_PER_CARD * max(0, fuel)
    return _apply_weakness(attacker, base, target) + _tool_bonus(attacker, target)


def _tr_in_discard(my_state):
    return sum(1 for c in (my_state.discard or []) if c.id in TR_SUPPORTERS)


def pokemon_score(pokemon):
    data = card_table.get(pokemon.id)
    score = prize_count(pokemon) * 1000 + len(pokemon.energies) * 120 + pokemon.hp
    if data and data.stage2:
        score += 250
    elif data and data.stage1:
        score += 130
    return score


def _discard_value(cid, hand_counts, kept):
    """Lower = pitch earlier (D1-D9). `kept` tracks how many of each we've already
    decided to keep, so the FIRST copy of a keeper ranks high but extras rank low."""
    base = _DISCARD_RANK.get(cid, 2)
    # extras of any supporter are cheap; the last copy of a keeper is precious.
    remaining = hand_counts.get(cid, 0) - kept.get(cid, 0)
    is_last = remaining <= 1
    if cid in (GIOVANNI, PETREL, ARIANA) and is_last:
        base += 6          # keep the last tutor/switch/refill
    return base


def _ordered_rf_discards(obs, my_state, opts, opt_idxs, k):
    """Pick the k LOWEST-value Team Rocket supporters to pitch for Rocket Feathers
    (research §D). `opt_idxs` are option indices that are TR supporters in hand."""
    hand_counts = defaultdict(int)
    for c in (my_state.hand or []):
        hand_counts[c.id] += 1
    scored = []
    for oi in opt_idxs:
        o = opts[oi]
        c = get_card(obs, o.area, o.index, o.playerIndex)
        if c is not None:
            scored.append((oi, c.id))
    kept = defaultdict(int)
    # rank by value ascending; tie-break keeps singles of keepers
    scored.sort(key=lambda t: _discard_value(t[1], hand_counts, kept))
    return [oi for oi, _ in scored[:k]]


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
    op_active = op_state.active[0] if op_state.active else None
    # H8: is the opponent a Fighting deck (e.g. Crustle)? Our Porygon2 is Fighting-weak
    # (90 HP -> OHKO'd), while Honchkrow is only Lightning-weak (safe vs Fighting). So
    # against Fighting, lean on Honchkrow and stop exposing the fragile Porygon2.
    opp_fighting = any(c is not None and card_table.get(c.id) is not None
                       and card_table[c.id].energyType == EnergyType.FIGHTING
                       for c in ([op_active] + list(op_state.bench)))

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
    have_special_energy = any(hand_counts[e] >= 1 for e in SPECIAL_ENERGY)
    tr_ammo = sum(hand_counts[c] for c in TR_SUPPORTERS)          # H_TR
    rc_fuel = _tr_in_discard(my_state)                            # D_TR
    deck_count = getattr(my_state, "deckCount", 99)
    # H3: is our active scaling-attacker starved (can't pay its attack right now)?
    # 80%+ of our energy-starved loss turns had ZERO energy in hand -> a SUPPLY problem,
    # so when starved we dig for energy hard (boost it in search/draw selections below).
    active_starved = False
    _a0 = my_state.active[0] if my_state.active else None
    if _a0 is not None and _a0.id in (HONCHKROW, PORYGON2):
        _cd0 = card_table.get(_a0.id)
        for _aid in (_cd0.attacks or []) if _cd0 else []:
            _a = attack_table.get(_aid)
            if _a and _a.name in ("Rocket Feathers", "R Command"):
                if not can_pay(list(_a0.energies), _a.energies or []):
                    active_starved = True
    all_play_tr = all(c.id in TR_SUPPORTERS or card_table.get(c.id) and "Team Rocket" in card_table[c.id].name
                      for c in (my_state.active + list(my_state.bench)) if c is not None)
    # going first on our first turn (Proton legality)
    first_turn_going_first = (state.turn == 1 and state.firstPlayer == my_index)
    # did the opponent KO one of our TR Pokemon last turn? (Archer legality) — approx via logs
    archer_live = False
    for lg in (obs.logs or []):
        if getattr(lg, "result", None) is None and getattr(lg, "playerIndex", None) == my_index:
            pass
    # robust check: Archer is only OFFERED by the engine when legal, so trust the option list.

    # ---- Rocket Feathers discard step (research §D, C4): pitch the MINIMUM, lowest-value ----
    if context == SelectContext.DISCARD and select.minCount == 0 and select.option:
        opts = select.option
        tr_idx = []
        n_card_opts = 0
        for k, o in enumerate(opts):
            if o.type == OptionType.CARD:
                n_card_opts += 1
                c = get_card(obs, o.area, o.index, o.playerIndex)
                if c is not None and c.id in TR_SUPPORTERS and o.playerIndex == my_index:
                    tr_idx.append(k)
        if tr_idx and len(tr_idx) == n_card_opts:    # this is the RF discard menu
            need = len(tr_idx)
            if op_active is not None:
                per = RF_PER_CARD
                htype = card_table[HONCHKROW].energyType if HONCHKROW in card_table else None
                tdata = card_table.get(op_active.id)
                if tdata and htype is not None and tdata.weakness == htype:
                    per *= 2
                need = (op_active.hp + per - 1) // per          # ceil(hp/per) to KO
            need = max(select.minCount, min(need, len(tr_idx), select.maxCount))
            return _ordered_rf_discards(obs, my_state, opts, tr_idx, need)

    # ---- MAIN: choose the attack plan (research C4/C5/C6/C8) ----
    can_switch = False
    if context == SelectContext.MAIN:
        for o in select.option:
            if o.type == OptionType.RETREAT:
                can_switch = True

        my_cards = [my_state.active[0] if my_state.active else None] + list(my_state.bench)
        op_cards = [op_active] + list(op_state.bench)

        best = -1.0
        for i, mine in enumerate(my_cards):
            if mine is None or (i != 0 and not can_switch):
                continue
            cd = card_table.get(mine.id)
            if cd is None:
                continue
            for aid in (cd.attacks or []):
                atk = attack_table.get(aid)
                if atk is None:
                    continue
                cost = atk.energies or []                       # REAL cost (R Command = CCC)
                attached = list(mine.energies)
                more_energy = False
                if not can_pay(attached, cost):
                    # "+1 attach this turn" lookahead with engine-true special-energy units:
                    #   Ignition on an evolution -> 3 colorless; TR Energy -> 2 (P/D).
                    if not state.energyAttached:
                        extras = [[et] for et in DECK_ETYPES]
                        if have_special_energy:
                            is_evo = _stage(mine.id) >= 1
                            if IGNITION_ENERGY and hand_counts[IGNITION_ENERGY] >= 1:
                                extras.append([EnergyType.COLORLESS] * (3 if is_evo else 1))
                            if TR_ENERGY and hand_counts[TR_ENERGY] >= 1:
                                extras.append([EnergyType.DARKNESS, EnergyType.PSYCHIC])
                        if (have_spare_energy or have_special_energy) and \
                           any(can_pay(attached + ex, cost) for ex in extras):
                            more_energy = True
                        else:
                            continue
                    else:
                        continue

                is_rf = (mine.id == HONCHKROW and aid == RF_ATTACK)
                is_rc = (mine.id in RCOMMANDERS and aid in (R_COMMAND, R_COMMAND_Z))
                is_hammer = (mine.id == HONCHKROW and aid == HAMMER_IN)
                if is_rf and tr_ammo <= 0:
                    continue
                if is_rc and rc_fuel <= 0:
                    continue

                opp = op_cards[0]
                if opp is None:
                    continue
                # v14 WALL DOCTRINE: vs a big multi-prize ex/mega wall (HP > 180, e.g. Mega
                # Lucario 340 / Mega Abomasnow 350), chipping with Rocket Feathers just BURNS
                # the ammo we need for the one-shot. Every wall here is 2-3 prizes, so a single
                # 6-discard OHKO (360) banks 2-3 prizes; two of those win. So: do NOT fire a
                # NON-lethal Rocket Feathers at a wall — hoard the ammo for the kill. (Small
                # targets are unaffected: only HP>180 triggers the hoard.)
                if is_rf and opp.hp > 180 and rocket_feathers_damage(mine, opp, tr_ammo) < opp.hp:
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
                kos = opp.hp <= dmg
                if kos:
                    score += 500
                    if op_prize_left <= prize_count(opp):
                        score = 50000
                else:
                    score *= dmg / max(1, opp.hp)

                # Research C4/C5/C6 resource-preservation preference when each KOs:
                #   R Command (spends nothing) > Hammer In (spends nothing) > Rocket Feathers.
                if kos:
                    if is_rc:
                        score += 60                  # free + hand-independent: best finisher
                    elif is_hammer:
                        score += 40                  # flat 100, preserves supporters
                    elif is_rf:
                        score -= 8 * tr_ammo          # discourage over-discard ammo burn
                if is_rf:
                    score -= 4 * tr_ammo

                # Crustle / Fighting tuning: Porygon2 is Fighting-weak + 90 HP. Don't expose
                # it as the active unless it KOs THIS turn; keep Honchkrow (Lightning-weak,
                # safe vs Fighting) in front otherwise.
                if mine.id == PORYGON2 and not kos and i != 0:
                    score -= 150
                # H8: vs a Fighting deck, don't commit the Fighting-weak Porygon2 unless it
                # KOs this turn (it just dies and we lose tempo) — keep racing with Honchkrow.
                if mine.id == PORYGON2 and opp_fighting and not kos:
                    score -= 400

                if i == 0:
                    score += 200
                score += len(mine.energies)
                if score > best:
                    best = score
                    plan.attacker = i
                    plan.target = 0
                    plan.attack_id = aid
                    plan.remain_hp = opp.hp - dmg
                    plan.needs_energy = more_energy

    # ---- energy attachment scoring (research C12-C15) ----
    def energy_score(pokemon, is_active, is_ignition):
        if pokemon is None:
            return -1
        score = 8000
        is_evo = _stage(pokemon.id) >= 1
        # Ignition: only worth it on an evolution attacking now (1-card enabler). On a
        # basic it is 1 colorless and self-discards — near useless (C15).
        if is_ignition and not is_evo:
            return 200
        if pokemon.id == HONCHKROW:
            score += 400
        elif pokemon.id == PORYGONZ:
            score += 390            # R Command at CC — cheapest scaling sweeper
        elif pokemon.id == PORYGON2:
            score += 380
        elif pokemon.id == MURKROW:
            score += 250
        elif pokemon.id == PORYGON:
            score += 230
        elif pokemon.id == ARTICUNO:
            return 20            # never invest: no Water energy in the list (C/§A)
        elif pokemon.id in ATTACKERS:
            score += 300
        else:
            score += 20
        if is_active:
            score += 40
        need = 3 if pokemon.id == PORYGON2 else 2   # Porygon-Z R Command is CC; Honchkrow CC
        if len(pokemon.energies) >= need:
            score -= 220
        return score

    # ---- option scoring ladder ----
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
                        if card.id == HONCHKROW:
                            score += 60          # prefer Honchkrow active (Fighting-safe)
                        elif card.id in ATTACKERS:
                            score += 40
                        elif card.id in BASIC_MONS:
                            score += 10
                        if card.id == ARTICUNO:
                            score -= 50          # don't promote the dead body
                    else:
                        # opponent target (gust destination): aim at plan.target / juicy KO
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if card.id == MURKROW:
                        score = 300              # Murkrow active -> Deceit T1, evolves to Honchkrow
                    elif card.id in BASIC_MONS and card.id != ARTICUNO:
                        score = 100
                    else:
                        score = 50
                elif context in (SelectContext.SETUP_BENCH_POKEMON,
                                 SelectContext.TO_FIELD, SelectContext.TO_BENCH):
                    # bench priority: Murkrow > Porygon > Articuno (C17). Keep all-TR for Ariana.
                    if card.id == MURKROW:
                        score = 120
                    elif card.id == PORYGON:
                        score = 90
                    elif card.id == ARTICUNO:
                        score = 30
                    elif card.id in BASIC_MONS:
                        score = 80
                    else:
                        score = 50
                elif context == SelectContext.TO_HAND:
                    cid = card.id
                    if cid in ATTACKERS:
                        score = 300 - field_counts[cid] * 40
                    elif cid in BASIC_MONS:
                        score = 260 - (field_counts[cid] + hand_counts[cid]) * 60
                    elif cid in ALL_ENERGY:
                        # H3: when the attacker is starved, energy is the #1 grab (above
                        # attackers/basics) — it's the binding constraint on racing.
                        score = 330 if active_starved else (120 if not state.energyAttached else 60)
                    elif cid in TR_SUPPORTERS:
                        score = 200          # ammo / fuel
                    elif cid == FACTORY:
                        score = 210 if not (state.stadium or []) else 90
                    elif card_type(cid) == CardType.SUPPORTER:
                        score = 140
                    elif card_type(cid) == CardType.ITEM:
                        score = 130
                    else:
                        score = 100 - hand_counts[cid] * 20
                elif context == SelectContext.ATTACH_FROM:
                    if isinstance(card, Pokemon):
                        score = energy_score(card, o.area == AreaType.ACTIVE, is_ignition=False)
                elif context in (SelectContext.DISCARD, SelectContext.TO_DECK,
                                 SelectContext.TO_DECK_BOTTOM, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
                    cid = card.id
                    if cid in ATTACKERS or cid in BASIC_MONS:
                        score = -50
                    elif cid in TR_SUPPORTERS:
                        # value-ordered: pitch low-value supporters first (D1-D9)
                        score = 40 + (4 - _DISCARD_RANK.get(cid, 2)) * 6 + max(0, hand_counts[cid] - 1) * 20
                    elif cid in ALL_ENERGY:
                        score = 10 + max(0, hand_counts[cid] - 1) * 20
                        if cid == IGNITION_ENERGY and hand_counts[cid] >= 2:
                            score += 50          # dump surplus Ignition (clogs draws)
                    else:
                        score = 30 + max(0, hand_counts[cid] - 1) * 25
                else:
                    score = 1

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is not None:
                ct = card_type(card.id)
                cid = card.id
                if ct == CardType.POKEMON:
                    score = 20000
                    # H9: keep benching bodies up to 5 (was 3). Board collapse ("no-active")
                    # is ~24% of our losses; a deeper bench gives replacement attackers and
                    # Ariana keeps drawing to 8 (all-TR) so long as every body is a TR Pokemon.
                    if cid in BASIC_MONS and n_bodies >= 5:
                        score = 30
                elif ct == CardType.SUPPORTER:
                    score = 3000
                    # ---- research supporter logic (C16-C20) ----
                    if cid == GIOVANNI:
                        # H2: NEVER swap away a lethal active. Giovanni moves our active to
                        # the bench; if our active already KOs the opp this turn, playing it
                        # throws the kill away (the #1 missed-lethal bug).
                        active_lethal = (plan.attacker == 0 and plan.attack_id not in (-1, None)
                                         and plan.remain_hp <= 0)
                        if active_lethal:
                            score = 50
                        else:
                            # gust+switch: high if a benched opp target can be KO'd after the pull.
                            gusts = any(b is not None and b.hp <= max(
                                rocket_feathers_damage(my_state.active[0], b, tr_ammo) if (my_state.active and my_state.active[0] and my_state.active[0].id == HONCHKROW) else 0,
                                100 if (my_state.active and my_state.active[0] and my_state.active[0].id == HONCHKROW) else 0)
                                for b in (op_state.bench or []))
                            score = 6000 if gusts else 2600
                    elif cid == PROTON:
                        # only legal/best T1-going-first while board is thin
                        score = 5200 if (first_turn_going_first and n_bodies < 3) else 2500
                    elif cid == PETREL:
                        score = 3300            # broad tutor: usually fine
                    elif cid == ARIANA:
                        # draw-to-8 only if all in-play are TR; worth more when hand is small
                        small = len(my_state.hand or []) <= 4
                        score = (3400 if all_play_tr else 2900) + (300 if small else 0)
                        # H4: Ariana refuels by DRAWING, which mills us toward the turn-21
                        # deck-out that is our #1 loss. Throttle hard as the deck thins, and
                        # never fire it on a non-desperate hand when the deck is short.
                        if deck_count <= 8:
                            score = 150
                        elif deck_count <= 16 and len(my_state.hand or []) > 2:
                            score = 1200
                    elif cid == ARCHER:
                        # the engine only OFFERS Archer when its KO-condition is legal; if it's
                        # here and our hand is weak, it's a strong reset, else low.
                        weak_hand = len(my_state.hand or []) <= 3
                        score = 3600 if weak_hand else 1200
                elif ct == CardType.ITEM:
                    # H1 anti-deck-out: every item in this deck digs, milling us toward the
                    # turn-19 deck-out that is our #1 loss. Taper hard as the deck thins, and
                    # stop digging entirely once a lethal attack is already lined up this turn.
                    if deck_count <= 12:
                        score = 200
                    elif deck_count <= 20:
                        score = 1500
                    else:
                        score = 3500
                    if plan.attack_id not in (-1, None) and plan.remain_hp <= 0:
                        score = min(score, 300)
                elif ct == CardType.STADIUM:
                    score = 1500
                else:
                    score = 1000

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is not None and card.id in TOOLS:
                score = 7000 + (200 if pokemon is not None and pokemon.id in ATTACKERS else 0)
                # v19: Brave Bangle is the Abomasnow threshold-fix — put it on a Honchkrow
                # (our ex-killer) so a 6-discard Rocket Feathers clears a Mega through Frost
                # Barrier. Punk Helmet goes on whatever is taking hits (active), for 40 recoil.
                if card.id == BRAVE_BANGLE:
                    score = 9000 if (pokemon is not None and pokemon.id == HONCHKROW) else 1500
                elif card.id == PUNK_HELMET:
                    score = 6500 if (o.inPlayArea == AreaType.ACTIVE) else 1500
            else:
                is_ign = card is not None and card.id == IGNITION_ENERGY
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE, is_ignition=is_ign)
                if plan.needs_energy:
                    if plan.attacker == 0 and o.inPlayArea == AreaType.ACTIVE:
                        score += 300
                    elif plan.attacker == 1 + (o.inPlayIndex or 0) and o.inPlayArea == AreaType.BENCH:
                        score += 300

        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + (len(pokemon.energies) if isinstance(pokemon, Pokemon) else 0) * 10
            if o.inPlayArea == AreaType.ACTIVE:
                score += 500             # evolve active Murkrow -> Honchkrow ends active

        elif o.type == OptionType.ABILITY:
            score = 15000 if turn_actions <= 8 else -2000

        elif o.type == OptionType.RETREAT:
            score = 2000 if plan.attacker >= 1 else -1

        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == plan.attack_id:
                score += 500
            # Murkrow Deceit fallback: search the best supporter when nothing better.
            if o.attackId == DECEIT and plan.attack_id in (-1, DECEIT):
                score = 900

        elif o.type == OptionType.END:
            score = 200000 if stalling else -1000

        else:
            score = 0

        scores.append(score)

    desc = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    chosen = desc[: max(select.minCount, min(select.maxCount, len(desc)))]
    if len(chosen) < select.minCount:
        chosen = desc[: select.minCount]
    return chosen
