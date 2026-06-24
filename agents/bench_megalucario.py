"""Mega Lucario ex — BASELINE BENCHMARK (a walrein vulnerability), AGGRESSIVE PILOT.

Exact 60-card list piloted by Kaggle opponent `shun_otoko` (rank 508) that beats our walrein
submissions (data/decks/deck_bench_megalucario.json, from data/wlren/lost_ep81547157.json).

WHY A TUNED PILOT (not raw bare_agent): the generic bare pilot left the deck's whole damage
engine on the table — it chipped for ~130 (Aura Jab) and let walrein's Frigid-Fangs lock stall
it, so walrein "won" 71% locally while LOSING to this deck on Kaggle (Lucario hit 270-330 there).
This pilot reproduces the real ladder behaviour: evolve Mega Lucario ASAP, play the damage-boost
trainers (Premium Power Pro / Fighting Gong), and FORCE Mega Brave (270, OHKOs Walrein's 170 HP)
every turn it is payable — falling back to Aura Jab (130 + Basic {F} energy acceleration to the
bench) on Mega Brave's cooldown turn. Delegates everything else to the generic bare engine.
"""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "decks", "deck_bench_megalucario.json"))
import agents.bare_agent as B
from agents.bare_agent import to_observation_class, get_card, AreaType, OptionType

my_deck = B.my_deck

MEGA_LUCARIO = 678
MEGA_BRAVE = 983          # 270 dmg, 2x{F} — the OHKO; can't use 2 turns running
AURA_JAB = 982            # 130 dmg, 1x{F} — accelerates Basic {F} from discard to bench
PREMIUM_POWER_PRO = 1141  # damage-boost item
FIGHTING_GONG = 1142      # damage-boost item


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck
    chosen = B.agent(obs_dict)
    select = obs.select
    if select.context != 0:          # only override MAIN-phase decisions
        return chosen
    opts = select.option or []
    my_index = obs.current.yourIndex

    # 1. OHKO first: force Mega Brave whenever it is a legal (payable) attack option.
    for i, o in enumerate(opts):
        if o.type == OptionType.ATTACK and getattr(o, "attackId", None) == MEGA_BRAVE:
            return [i]

    # 2. Stack damage before attacking: force-play the boost trainers (bare pilot never does).
    for i, o in enumerate(opts):
        if o.type == OptionType.PLAY:
            c = get_card(obs, AreaType.HAND, o.index, my_index)
            if c is not None and c.id in (PREMIUM_POWER_PRO, FIGHTING_GONG):
                return [i] if select.minCount <= 1 else chosen

    # 3. Get the attacker online: force any available evolution (Riolu -> Mega Lucario, etc.).
    for i, o in enumerate(opts):
        if o.type == OptionType.EVOLVE:
            return [i] if select.minCount <= 1 else chosen

    # 4. Cooldown turn (Mega Brave unavailable): chip + accelerate with Aura Jab if offered.
    for i, o in enumerate(opts):
        if o.type == OptionType.ATTACK and getattr(o, "attackId", None) == AURA_JAB:
            return [i]

    return chosen
