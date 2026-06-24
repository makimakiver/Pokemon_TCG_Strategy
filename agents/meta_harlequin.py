"""Harlequin / Cinderace — bare-deck agent (Kaggle-meta deck + generic pilot).
Deck from ptcg-kaggle-meta /api/archetype?slug=harlequin-cinderace-855158 (date 2026-06-22): ladder 107-62 = 63.3% winrate,
meta share 1.7%. Built by avgCopies; generic bare_agent pilot (no archetype heuristics).
"""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_harlequin.json"))
import agents.bare_agent as B
my_deck = B.my_deck
agent = B.agent
