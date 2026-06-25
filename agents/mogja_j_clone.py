"""mogja_j clone — behavioral clone of the 83%-ladder Starmie/Cinderace pilot.

Extracted by decision-by-decision analysis of all 47 replays in data/opponents/mogja_j/ (39W/8L = 83%).
mogja_j plays deck_bench_starmie_cinderace.json (Mega Starmie ex + Cinderace + Budew). This agent
replicates its MEASURED playbook rather than a hand-tuned theory:

  MEASURED mogja_j patterns (from 47-replay decision parse):
  - T1-2 SETUP: attach Basic Water/Hero's Cape to active, draw (Pokégear 3.0 / Ultra Ball / Lillie's
    Determination), bench basics via Buddy-Buddy Poffin. Rarely attacks before T3.
  - T2-3 EVOLVE: evolves Staryu->Mega Starmie ex via Salvatore (same-turn evolve of a just-benched Basic)
    or natural evolution. Salvatore is core to the real playbook (used T2/T3 in wins).
  - PROMOTE Starmie ex to active the moment it's evolved (RETREAT the setup basic).
  - RELENTLESS ATTACK (the key 83% behavior): once Starmie ex is active, ATTACK EVERY TURN.
      * Jetting Blow (120 + 50 bench spread, 1 Water): fires 4x more than Nebula Beam; mogja_j attacks
        the INSTANT Starmie has 1 energy (69/120 JettingBlow attacks at exactly 1 energy). Pressure.
      * Nebula Beam (210 piercing, 3 colorless): the closer — fires at 3 energy to OHKO threats.
      (120x JettingBlow vs 33x NebulaBeam in the measured data.)
  - Cinderace Turbo Flare: ramps 3 basic energy from deck onto benched Starmie-line each turn.

The generic bare_agent undervalues Jetting Blow (low base dmg) and won't prioritize the relentless
attack pattern. This clone forces the measured rules.
"""
import os
import sys

os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "decks", "deck_bench_starmie_cinderace.json"))
import agents.bare_agent as B
from agents.bare_agent import to_observation_class, get_card, AreaType, OptionType, card_table, attack_table

my_deck = B.my_deck

STARYU, STARMIE_EX, CINDERACE, BUDEW = 1030, 1031, 666, 235
TURBO_FLARE = 965       # Cinderace: 50 dmg + attach 3 basic energy from deck to benched Pokemon
JETTING_BLOW = 1487     # Mega Starmie ex: 120 dmg + 50 to 1 benched opp (1 Water cost)
NEBULA_BEAM = 1488      # Mega Starmie ex: 210 piercing (3 colorless cost)
SALVATORE = 1189        # Supporter: evolve a no-Ability Stage-2 from a just-benched Basic (same-turn)
HEROS_CAPE = 1159       # +100HP tool (mogja_j attaches T1)
CRUSHING_HAMMER = 1120

_pre_turn = -1


def _log(turn, move, why):
    sys.stderr.write(f"MOGJA| T{turn:<2} {move:<22} — {why}\n"); sys.stderr.flush()


def _ids(cards):
    return [c.id for c in (cards or []) if c is not None]


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck
    chosen = B.agent(obs_dict)
    sel = obs.select
    st = obs.current
    me = st.players[st.yourIndex]
    opp = st.players[1 - st.yourIndex]
    opts = sel.option or []
    turn = st.turn
    global _pre_turn

    active = (me.active or [None])[0]
    in_play = _ids(([active] if active else []) + list(me.bench or []))
    hand = _ids(me.hand)
    active_is_starmie = active is not None and active.id == STARMIE_EX
    active_is_cind = active is not None and active.id == CINDERACE
    starmie_in_play = STARMIE_EX in in_play

    def play_idx(card_id):
        for i, o in enumerate(opts):
            if o.type == OptionType.PLAY:
                c = get_card(obs, AreaType.HAND, o.index, st.yourIndex)
                if c is not None and c.id == card_id:
                    return i
        return None

    def atk_idx(aid):
        for i, o in enumerate(opts):
            if o.type == OptionType.ATTACK and getattr(o, "attackId", None) == aid:
                return i
        return None

    # ---- SWITCH/TO_ACTIVE: promote Starmie ex to active when evolved on the bench (the relentless pattern) ----
    if sel.context in (3, 4):
        bench_starmie = [c for c in (me.bench or []) if c is not None and c.id == STARMIE_EX]
        if bench_starmie:
            for i, o in enumerate(sel.option or []):
                if o.type in (OptionType.CARD, OptionType.PLAY):
                    oarea = getattr(o, "area", None) or getattr(o, "inPlayArea", None)
                    opidx = getattr(o, "index", None)
                    if oarea == 5 and opidx is not None:   # AreaType.BENCH == 5
                        bc = get_card(obs, AreaType.BENCH, opidx, st.yourIndex)
                        if bc is not None and bc.id == STARMIE_EX:
                            _log(st.turn, "promote Starmie ex", "evolved on bench → make it active (relentless-attack setup)")
                            return [i] if sel.minCount <= 1 else [i]
    if sel.context != 0:
        return chosen

    # ---- MAIN ----

    # 1. SALVATORE — evolve Staryu->Mega Starmie ex the SAME turn Staryu is benched (mogja_j's T2-3 move).
    #    Salvatore evolves a no-Ability Stage-2 from a just-played Basic; Rare Candy cannot. Core to the playbook.
    if not starmie_in_play and STARYU in in_play and SALVATORE in hand and not st.supporterPlayed:
        i = play_idx(SALVATORE)
        if i is not None and sel.minCount <= 1:
            _log(turn, "Salvatore", "Staryu in play + Salvatore → evolve Starmie ex same-turn (the real T2-3 setup move)")
            return [i]

    # 2. PROMOTE TRIGGER — if active is NOT Starmie ex but a benched Starmie ex exists → RETREAT/Switch
    #    so Starmie ex becomes the attacker. mogja_j promotes Starmie ex the moment it's evolved.
    if active is not None and active.id != STARMIE_EX and starmie_in_play:
        # try Switch-item-free RETREAT if the current active can pay
        rcost = getattr(B.card_table.get(active.id), "retreatCost", 9)
        if len(active.energies) >= rcost:
            for i, o in enumerate(opts):
                if o.type == OptionType.RETREAT:
                    _log(turn, "Retreat (→Starmie)", f"active {active.id} not Starmie → retreat, promote Starmie ex to attack")
                    return [i] if sel.minCount <= 1 else [i]

    # 3. TURBO FLARE — Cinderace active + benched Starmie-line under-energised → ramp 3 energy (the combo).
    if active_is_cind:
        bench_line = [c for c in (me.bench or []) if c is not None and c.id in (STARYU, STARMIE_EX)]
        under = any(len(c.energies) < 2 for c in bench_line)
        if under and starmie_in_play:
            i = atk_idx(TURBO_FLARE)
            if i is not None:
                _log(turn, "Turbo Flare", "Cinderace active + benched Starmie <2 energy → ramp 3 energy from deck")
                return [i] if sel.minCount <= 1 else [i]

    # 4. RELENTLESS ATTACK — the core 83% behavior. Once Starmie ex is active, ATTACK EVERY TURN
    #    STARTING T3 (mogja_j measures: attacks begin T3+, after Staryu evolved + energized; T2 is setup-only).
    #    mogja_j measured: 120x JettingBlow (1-energy pressure) + 33x NebulaBeam (3-energy closer).
    #    Gate on turn>=3 so we don't waste T2 attacking an un-energized just-evolved Starmie (the v0-clone bug).
    if active_is_starmie and turn >= 3:
        opp_active = (opp.active or [None])[0]
        opp_hp = getattr(opp_active, "hp", 0) if opp_active else 0
        neb = atk_idx(NEBULA_BEAM)
        jet = atk_idx(JETTING_BLOW)
        # Nebula Beam (210 piercing) when: payable, AND (it KOs opp active <=210 OR no Jetting available
        # OR opp active is a high-HP threat we want to grind). mogja_j uses Nebula as the closer.
        if neb is not None and (opp_hp > 0 and (opp_hp <= 210 or jet is None)):
            _log(turn, "Nebula Beam", f"210 piercing closer on opp active ({opp_hp}hp)")
            return [neb] if sel.minCount <= 1 else [neb]
        # else Jetting Blow — the relentless 120+50-spread pressure (mogja_j fires this at 1 energy)
        if jet is not None:
            _log(turn, "Jetting Blow", "120 to active + 50 spread → relentless pressure (mogja_j's 4x-preferred attack)")
            return [jet] if sel.minCount <= 1 else [jet]

    # 5. HERO'S CAPE — attach +100HP to active Starmie ex (mogja_j's T1 move). Starmie 330->430hp.
    if active_is_starmie and HEROS_CAPE in hand:
        already = any(getattr(t, "id", None) == HEROS_CAPE for t in (active.tools or [])) if hasattr(active, "tools") else False
        if not already:
            for i, o in enumerate(opts):
                if o.type == OptionType.ATTACH:
                    card = get_card(obs, AreaType.HAND, o.index, st.yourIndex)
                    if card is not None and card.id == HEROS_CAPE and getattr(o, "inPlayArea", None) == 4:
                        tgt = get_card(obs, AreaType.ACTIVE, getattr(o, "inPlayIndex", 0), st.yourIndex)
                        if tgt is not None and tgt.id == STARMIE_EX:
                            _log(turn, "attach Hero's Cape", "Starmie active → +100HP (mogja_j's T1 move)")
                            return [i] if sel.minCount <= 1 else [i]

    return chosen
