"""Walrein/Spheal — bare-deck agent (Kaggle 74% deck + generic pilot). Clean ranking track,
separate from the co-developed agents/walrein_v1.py. Deck data/decks/deck_walrein.json."""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_walrein.json"))
import agents.bare_agent as B
my_deck = B.my_deck
agent = B.agent
