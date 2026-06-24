"""Slowking control-toolbox — FIRST agent (v1 = real Kaggle deck + generic pilot).

DECK (data/decks/deck_slowking.json): the exact 60 that the ladder's strongest Slowking
pilot ran on Kaggle (team "Shun Tsukuda / しゅん", rank ~1633), extracted from a winning
episode replay (data/slowking_kaggle/_slowking_deck.json):
  Pokemon: 4 Slowpoke, 3 Slowking, 4 Mega Kangaskhan ex, 2 Latias ex, 2 Kyurem,
           2 Conkeldurr, 1 Fezandipiti ex, 1 Lillie's Clefairy ex, 1 Meowth ex
  Trainers: 4 Poke Pad, 4 Academy at Night, 4 Hilda, 4 Ciphermaniac's Codebreaking,
            4 Lillie's Determination, 4 Ultra Ball, 3 Wondrous Patch, 2 Night Stretcher,
            1 Prime Catcher, 1 Switch
  Energy: 4 Telepathic Psychic, 4 Basic Psychic, 1 Boomerang
(NB: this Kaggle list differs from Ross Cawthon's NAIC 4th-place 28251 list — no
Metagross/Zeraora/Brave Bangle/Lucky Helmet/Secret Box — so the engine ceiling here is
lower than the paper deck's; see research notes below.)

HOW THE WINNING DATA PERFORMED (data/slowking_kaggle/_stats.json, this pilot, n=95):
  overall 49-46 (~52%). Strong: Crustle/Typhlosion 5-0, Dragapult 3-0, Mega Abomasnow 6-3.
  Weak: Dragapult/Dusknoir 0-3, Mega Lucario 10-12, Alakazam/Dudunsparce 6-7.
  => Slowking is a coin-flip ~52% deck on the Kaggle ladder (fringe; absent from top-50),
     NOT the top-tier control deck it is in the physical NAIC meta.

RESEARCH-DERIVED GAME PLAN (Ross Cawthon NAIC writeup — encode as heuristics in v2):
  - Win condition: single-PRIZE control-toolbox. Slowking "Seek Inspiration" discards the
    top deck card and copies a non-Rule-Box Pokemon's attack -> so the TOP CARD is a resource.
    Set it with Ciphermaniac's Codebreaking / Academy at Night, then copy the right attacker.
  - Pivot heartbeat: Active Mega Kangaskhan ex draws 2 (Run Errand) -> Latias ex gives Basics
    free retreat -> pivot to Slowking and attack. Latias does NOT free-retreat Slowking itself.
  - Prize map: present one-Prizers, take two-Prize turns; do NOT over-bench Rule-Box ex's
    (Kanga/Latias/Meowth/Fezandipiti/Clefairy) or you gift the opponent an easy 3-KO path.
  - Boomerang Energy returns after a discard-all-energy copied attack (Kyurem spread, etc.).

v1 SCOPE: this is a measurable BASELINE. It pilots the real deck with the repo's generic
engine (agents.bare_agent) — the same wrap pattern as agents.abomasnow_cal. The generic pilot
does NOT yet execute the Seek-Inspiration topdeck-setup combo (the deck's core engine), so v1
will under-perform the deck's ceiling. v2 will add the prize-discipline + Kanga->Slowking pivot
+ copied-attack-selection overlays above. Gauntlet vs the meta is the next step.
"""
import os

# Pilot the extracted Kaggle Slowking deck via the generic engine.
os.environ.setdefault("BARE_DECK", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "decks", "deck_slowking.json"))

import agents.bare_agent as B

my_deck = B.my_deck
agent = B.agent
