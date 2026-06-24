"""Mega Starmie ex / Staryu — bare-deck agent (Kaggle-meta deck + generic pilot).
Deck from ptcg-kaggle-meta /api/archetype?slug=mega-starmie-ex-staryu-ac2267 (date 2026-06-22): ladder 52-29 = 64.2% winrate,
meta share 0.8%. Built by avgCopies; generic bare_agent pilot (no archetype heuristics).
"""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_starmie.json"))
import agents.bare_agent as B
my_deck = B.my_deck
agent = B.agent
