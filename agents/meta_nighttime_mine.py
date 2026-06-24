"""Nighttime Mine / Fezandipiti ex — bare-deck agent (Kaggle-meta deck + generic pilot).
Deck from ptcg-kaggle-meta /api/archetype?slug=nighttime-mine-fezandipiti-ex-d91498 (date 2026-06-22): ladder 168-107 = 61.1% winrate,
meta share 2.7%. Built by avgCopies; generic bare_agent pilot (no archetype heuristics).
"""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_nighttime_mine.json"))
import agents.bare_agent as B
my_deck = B.my_deck
agent = B.agent
