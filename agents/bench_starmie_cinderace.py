"""Mega Starmie ex / Cinderace ("fireball guy") — BASELINE BENCHMARK (a walrein vulnerability).

Exact 60-card combo list (Mega Starmie ex + Cinderace, Turbo Flare fire engine) extracted from a
real walrein loss replay (data/wlren/lost_ep81590873.json). Generic bare_agent pilot — no
archetype heuristics. This is the second of the two decks walrein is structurally weak to
(Starmie + Cinderace fire pressure + Crushing Hammer energy denial). Fixed gauntlet opponent.
"""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "decks", "deck_bench_starmie_cinderace.json"))
import agents.bare_agent as B
my_deck = B.my_deck
agent = B.agent
