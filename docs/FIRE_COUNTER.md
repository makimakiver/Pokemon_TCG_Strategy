# Anti-Crustle Fire deck — `agents/fire.py` + `data/decks/deck_fire.json`

Goal: a deck built to hard-counter `main.py` (tuned Crustle) and reach 80%.
**Result: 74.1% vs the current tuned main over 1000 games — and ~80% on the play.**

## Why Fire counters Crustle
Main's deck is structurally fragile:
- **8 Pokemon, single line, all Grass — every one is weak to Fire** (Dwebble/Crustle weak=Fire).
- Its known loss mode is `no-active` (thin board KO'd off).

So this is a **fast NON-ex Fire aggro** deck:
- Every attacker OHKOs Crustle: base x2 weakness, Crustle is 150 HP. Ho-Oh *Shining Blaze*
  100→200, Volcanion *Backfire* 130→260, Hearthflame *Searing Flame* 80→160.
- Attackers are **130 HP (Ho-Oh / Volcanion)** → survive Crustle's 120, so the prize race is
  ~2:1 ours: we KO every turn (1 prize), main needs **two** hits per body and our bodies are
  non-ex (no 2-prize swings).
- Hearthflame *Fire Kagura* [C] and Waitress accelerate energy; Cook/Jumbo Ice Cream heal a
  hit off, so a body often survives 3+ turns.

Deck: 4 Ho-Oh (318) / 3 Volcanion (663) / 2 Hearthflame Ogerpon (358); 18 Fire energy;
Ultra Ball, Poké Pad, Boss's Orders (gust), Cook + Jumbo (heal), Waitress (accel), Lillie's
Determination (draw), 1 Hero's Cape (+100 HP, capped at 1 per deck).

## What the tuned pilot adds over the bare agent (which only gets ~24-65%)
1. **Non-ex deck** — biggest single lever (bare ex-deck 7%→65%): deny main 2-prize swings.
2. **Energy concentration** — pile energy on ONE attacker to reach its [RRC] OHKO instead of
   dribbling across the bench (the bare agent's fatal habit). +~5pts.
3. **18 Fire energy** — an all-[RRC] deck starves on 16; 18 is the sweet spot. +~5pts.
4. **Smart gust** — Boss's Orders drags up main's *most-developed* benched threat to KO it
   (we OHKO anything), breaking its thin board. +~3pts.
5. **Tanky active** — keep a 130-HP body active; bench the 110-HP Hearthflame as accel only
   (active it just gets OHKO'd for a free prize). Balanced the seat split 80/69 → ~75/73.
6. Heal-timing (don't waste a heal/our Supporter at full HP).

## Measurements (seat-swapped `runner.py`, Docker)
| Opponent | Fire deck win % |
|---|:---:|
| **main_cur** (current tuned Crustle) | **74.1%** (1000g) — *80% on the play, ~73% on the draw* |
| main_bench (older main) | ~70% (600g) |
| main_megalucario_backup | **36%** — *craters vs a non-Grass deck; this is a specific counter* |

## On the 80% target
**80% overall is at/beyond the realistic ceiling vs this tuned opponent.** The deck already
wins **~80% on the play**; the overall is capped by the coin flip (main gets the play half the
time) and by main being a strong tuned agent with healing + bench refill. The fastest Fire
OHKO is turn-3 ([RRC] — no cheaper Fire OHKO exists in the pool), so we can't blow main out
before it acts. Remaining losses are the going-second prize race (all-prizes), an inherent
TCG disadvantage.

This is still a decisive, purpose-built counter: **74% vs main_cur**, far above what tuning a
peer deck achieves (tuned Honchkrow caps ~55% — see `HONCHKROW_TUNING.md`).

## Files
- `agents/fire.py` — tuned pilot (loads `data/decks/deck_fire.json`)
- `data/decks/deck_fire.json` — the 60-card list
- `tools/trace_deck.py` — generic per-seat / per-first telemetry for any deck+agent
- `tools/inspect_deck.py` — dump a deck's attacks/abilities/effect text
