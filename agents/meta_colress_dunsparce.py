"""Colress’s Tenacity / Dunsparce — bare-deck agent (Kaggle-meta deck + generic pilot).
Deck from ptcg-kaggle-meta /api/archetype?slug=colresss-tenacity-dunsparce-4a3db3 (date 2026-06-22): ladder 1051-1007 = 51.1% winrate,
meta share 20.4%. Built by avgCopies; generic bare_agent pilot (no archetype heuristics).
"""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_colress_dunsparce.json"))
import agents.bare_agent as B
my_deck = B.my_deck
agent = B.agent
