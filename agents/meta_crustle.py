"""Crustle — bare-deck agent (Kaggle-meta deck + generic pilot)."""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_crustle.json"))
import agents.bare_agent as B
my_deck = B.my_deck
agent = B.agent
