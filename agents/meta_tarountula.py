"""Team Rocket's Tarountula / Team Rocket's Spidops — bare-deck agent (Kaggle-meta deck + generic pilot).
Deck from ptcg-kaggle-meta /api/archetype?slug=team-rocket-s-tarountula-team-rocket-s-spidops-dc95e1 (date 2026-06-22): ladder 69-36 = 65.7% winrate,
meta share 1.0%. Built by avgCopies; generic bare_agent pilot (no archetype heuristics).
"""
import os
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_tarountula.json"))
import agents.bare_agent as B
my_deck = B.my_deck
agent = B.agent
