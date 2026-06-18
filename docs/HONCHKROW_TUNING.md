# Tuning Rocket's Honchkrow — `agents/honchkrow.py`

A deck-specific pilot for Rocket's Honchkrow, the best meta deck under the bare auto-pilot.
Goal: beat `main.py` (the tuned Crustle agent). **Result: 23% → 56% vs the current `main.py`.**

## The deck is a combo the bare agent can't see

| Card | Why it matters |
|------|----------------|
| **Team Rocket's Honchkrow** — *Rocket Feathers* `[CC]` | listed **damage = 0**, but does **60 × (Team Rocket Supporters discarded from hand)**. The real nuke (120–240+). |
| Team Rocket's Murkrow — *Deceit* `[C]` | search a Supporter into hand (loads ammo) |
| 20× Team Rocket **Supporters** | **ammunition**, not just effects (Proton/Ariana/Giovanni/Petrel/Archer) |
| Transceiver / Roto-Stick / Petrel / Poké Pad | fetch supporters → more ammo, no supporter-for-turn cost |
| Team Rocket's Energy (=2) / Ignition (=3 on an Evolution) | one special energy powers Rocket Feathers' `[CC]` |
| Articuno *Repelling Veil* | walls Basic TR Pokémon from attack effects |

The bare agent's planner only considers attacks with `damage > 0`, so it **never planned
Rocket Feathers** and **played the Team Rocket supporters for their text** — burning its own
ammunition. It scraped ~23% purely from opportunistic Rocket Feathers when nothing else was
payable.

## What the tuned agent changes

1. **Model Rocket Feathers** — effective damage = `60 × (TR supporters in hand)` (with
   weakness), so the planner values and prioritizes it.
2. **Hoard ammo** — Team Rocket supporters are held in hand; only played for their text when
   no lethal Rocket Feathers is already lined up (you can only play 1 supporter/turn anyway).
3. **Fetch ammo** — Deceit / Transceiver / supporter searches are biased toward Team Rocket
   supporters; `TO_HAND` prioritizes them.
4. **Energy awareness** — the "+1 energy" look-ahead knows a special energy provides 2+ units
   (this deck runs *no* basic energy), so it correctly sees Honchkrow as ready to attack.
5. **Precise ammo-discard** *(the biggest single lever)* — at Rocket Feathers' discard step,
   ditch **only enough supporters to KO** the target and keep the rest. Over-discarding forced
   constant redraws → self-deck-out; conserving ammo gives **sustained** KOs.

## Measurements (seat-swapped `runner.py`, Docker)

Opponents: `agents/main_bench.py` (earlier frozen main) and `agents/main_cur.py` (frozen
snapshot of the current, further-tuned `main.py`).

| Pilot | vs main_bench | vs main_cur |
|-------|:---:|:---:|
| `bare_agent` (no tuning) | ~22% | ~23% (300g) |
| **`honchkrow` (tuned)** | **~57–60%** | **55.7% (300g)** |

Behavior after tuning: Honchkrow online ~turn 6–7, Rocket Feathers fires ~5×/game at ~150
avg damage (ammo-efficient), deck-out losses cut sharply, most wins by KO (`no-active`).
Secondary: Porygon2's *R Command* scales off the supporters Rocket Feathers banks in the
discard (~188 avg).

## Tried and rejected
- **Throttle draw/search to avoid deck-out** → *worse* (lost the prize race). The precise
  ammo-discard fixed deck-out without sacrificing tempo.
- **Deeper bench (cap 4 bodies)** → neutral: fewer `no-active` losses but more `all-prizes`
  (extra bodies = extra prizes for the opponent). Same trade the Crustle log found.

## Round 2 — pushing for higher win rate (going-second + speed)

Diagnosed with `tools/trace_turns.py` (per-turn action log) + `trace_honch.py` (per-seat,
loss-mode telemetry). Findings and changes:

- The "80% seat0 / 33% seat1" split was **small-sample noise**. At 300g the real gap is
  ~52% (first) vs ~46% (second); the engine makes **seat 0 always go first**.
- **Bug found:** setup put *any* basic active (often Articuno/Porygon), which can't attack on
  one special energy → the agent durdled its first two turns and retreated later to get
  Honchkrow online. **Fix:** prefer **Murkrow** as the setup active + prefer evolving the
  **active** Murkrow → Honchkrow ends up active, no retreat. (going-second ~46%→~53%.)
- **Energy was getting stranded** on Articuno/Porygon (walls). **Fix:** concentrate special
  energy on the Murkrow→Honchkrow line and spread to a backup once an attacker has 2 (Rocket
  Feathers only costs `[CC]`). First Honchkrow attack ~turn 8 → ~turn 7.
- **Rejected:** "stop searching once combo-ready" throttle (deck-out is *downstream* of long
  losing games, not a root cause — no measurable change at 800g).

### The ceiling (measured, large samples)
| Opponent | tuned Honchkrow win % |
|---|:---:|
| **main_cur** (current tuned Crustle) | **54.9%** (1000g) |
| main_bench (older main) | ~54% (600g) |
| main_megalucario_backup | ~46% (600g) — *unfavorable* |

**80% is not reachable by strategy tuning.** This is a peer matchup: tuned Honchkrow vs tuned
Crustle is ~even-to-slightly-favored (~55%), and Honchkrow is an underdog vs Mega Lucario.
~52% of remaining losses are the **prize race** — structural, because Honchkrow is a setup
combo deck that comes online ~turn 7 while Crustle attacks from turn ~3. The real-world meta
agrees (Crustle 6.1% share > Honchkrow 2.1%). The big, real win was Round 1 (23% → ~55%);
Round 2's micro-tuning is within noise (~+1pt) but makes the agent play correctly.

To actually reach 80% you'd need a *deck* change (a build that hard-counters Crustle) or a
weaker opponent — not further pilot tuning.

## Files
- `agents/honchkrow.py` — the tuned agent (loads `data/decks/deck_rocket_s_honchkrow.json`)
- `agents/main_cur.py` — frozen snapshot of current `main.py` used as the benchmark opponent
- `tools/inspect_deck.py` — dump a deck's attacks/abilities/effect text
- `tools/trace_honch.py` — per-seat win rates + attack-usage/damage telemetry
