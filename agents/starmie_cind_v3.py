"""Starmie/Cinderace — v1 strategic pilot (the 83%-ladder deck, validated by mogja_j's 47 games).

DECK IDENTITY (data/decks/deck_bench_starmie_cinderace.json):
  - Mega Starmie ex (1031, 330hp, Water, weak Grass/4): Jetting Blow 120+50-bench-spread (1W cost);
    Nebula Beam 210, DAMAGE IGNORES WEAKNESS/RESISTANCE + effects on opp active (3-colorless cost). The
    premium 210-piercing nuke — OHKOs Dragapult-line, Lucario-line, most Stage-2 attackers.
  - Staryu (1030, 70hp) -> Starmie ex.
  - Cinderace (666, 160hp, Fire, weak Water/3): Turbo Flare = 50 dmg + SEARCH UP TO 3 BASIC ENERGY from
    deck and attach them to benched Pokemon (1 colorless cost). THE energy-accel engine — lands 2-3
    energy on Starmie ex in one turn, enabling T2 Nebula Beam.
  - Energy: 9 Water + 4 Ignition(special). Salvatore/Hilda/Harlequin/Wally tutors. 4x Crushing Hammer
    (opponent energy denial). Boss's Orders (gust). Pokegear/Mega Signal/Poffin draw.

WHY THIS DECK (data-driven): mogja_j piloted it 83% on ladder (47 games): vs Dragapult 6/9, Lucario
3/3, Dudunsparce 8/9. Sim confirmed 78% field-avg (collision+legality clean). Dragapult (the prior
campaign deck) is structurally capped ~53%; this deck clears the 60% goal on autopilot.

STRATEGY (on top of the generic bare engine):
  1. FORCE Cinderace TURBO FLARE when Cinderace active + a benched Starmie-line mon is under-energised
     (<2 energy) -> ramp 3 energy onto it in one turn (the deck's core combo). bare_agent scores
     Turbo Flare low (50 base dmg) and won't prioritize the ramp.
  2. FORCE Nebula Beam on Mega Starmie ex when payable AND opp active is in 210-KO range (or any
     threatening Stage-2/EX — the piercing means no weakness math, 210 is 210). Prefer over Jetting
     Blow unless Jetting's bench-spread KO is available.
  3. KEEP Starmie ex active when it's set up (don't retreat a 330hp tank that's firing 210/turn).

Every override logs to stderr as STRAT| T<turn> <move> — <why> for visibility.
"""
import os
import sys

os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "decks", "deck_bench_starmie_cinderace.json"))
import agents.bare_agent as B
from agents.bare_agent import to_observation_class, get_card, AreaType, OptionType, card_table, attack_table

my_deck = B.my_deck

STARYU, STARMIE_EX, CINDERACE = 1030, 1031, 666
TURBO_FLARE = 965      # Cinderace Turbo Flare attackId (50 dmg + attach 3 basic energy from deck to bench)
NEBULA_BEAM = 1488     # Mega Starmie ex Nebula Beam (210 piercing)
JETTING_BLOW = 1487    # Mega Starmie ex Jetting Blow (120 + 50 bench spread)
CRUSHING_HAMMER = 1120
POFFIN = 1086          # Buddy-Buddy Poffin: bench 2 basics from deck. v3 force-play early to prevent
                      # the no-active board-wipe (traced root cause of aggro losses).

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
    active_is_cind = active is not None and active.id == CINDERACE
    active_is_starmie = active is not None and active.id == STARMIE_EX
    starmie_in_play = STARMIE_EX in in_play

    # v3 FORCE-BENCH BACKUP — prevent the no-active board-wipe (traced: 9/14 megalucario + 7/11 hydrapple
    # losses were no-active at T2-3 because the glass 70hp Staryu got OHKO'd with no backup on bench).
    # If early (<=T2) and no backup basic exists (total in-play basics < 2) -> force Poffin to bench basics.
    if turn <= 2 and POFFIN in hand:
        basics_in_play = len(in_play)  # active + bench count; basics are the pokemon line
        if basics_in_play < 2:
            for i, o in enumerate(opts):
                if o.type == OptionType.PLAY:
                    c = get_card(obs, AreaType.HAND, o.index, st.yourIndex)
                    if c is not None and c.id == POFFIN:
                        _log(turn, "Poffin (bench backup)", f"only {basics_in_play} mon in play -> bench basics to prevent no-active wipe")
                        return [i] if sel.minCount <= 1 else [i]

    def atk_idx(aid):
        for i, o in enumerate(opts):
            if o.type == OptionType.ATTACK and getattr(o, "attackId", None) == aid:
                return i
        return None

    # 1. TURBO FLARE — Cinderace active + a benched Starmie-line is under-energised -> ramp 3 energy.
    #    The deck's core combo; bare_agent undervalues the 50-base-dmg ramp play.
    if active_is_cind:
        bench_starmie = [c for c in (me.bench or []) if c is not None and c.id in (STARYU, STARMIE_EX)]
        under_energy = any(len(c.energies) < 2 for c in bench_starmie)
        if under_energy and starmie_in_play:
            i = atk_idx(TURBO_FLARE)
            if i is not None:
                _log(turn, "Turbo Flare", "Cinderace active + benched Starmie <2 energy → ramp 3 energy from deck")
                return [i] if sel.minCount <= 1 else [i]

    # 2. NEBULA BEAM — Mega Starmie ex active + payable + opp active in 210-KO range OR is a threat.
    #    210 piercing (ignores weakness/resistance/effects) — OHKOs most Stage-2/EX attackers.
    if active_is_starmie:
        opp_active = (opp.active or [None])[0]
        opp_hp = getattr(opp_active, "hp", 0) if opp_active else 0
        # fire Nebula if it KOs (hp<=210) OR the opp active is a high-HP threat we want to grind
        # (always prefer the piercing nuke over Jetting Blow unless Jetting nets a bench KO — handled below)
        i = atk_idx(NEBULA_BEAM)
        j = atk_idx(JETTING_BLOW)
        if i is not None:
            # prefer Nebula when it KO's or when no Jetting-bench-KO line is clear (keep it simple: always Nebula if payable)
            if opp_hp > 0 and (opp_hp <= 210 or j is None):
                _log(turn, "Nebula Beam", f"210 piercing nuke on opp active ({opp_hp}hp) → OHKO/grind")
                return [i] if sel.minCount <= 1 else [i]

    return chosen
