"""Dragapult ex / Drakloak — bare-deck agent (Kaggle-meta deck + generic pilot).
Deck from ptcg-kaggle-meta /api/archetype?slug=dragapult-ex-drakloak-5e6d8a (date 2026-06-22): ladder 337-226 = 59.9% winrate,
meta share 5.6%. Built by avgCopies; generic bare_agent pilot (no archetype heuristics).
"""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_dragapult.json"))
import agents.bare_agent as B
my_deck = B.my_deck
agent = B.agent
