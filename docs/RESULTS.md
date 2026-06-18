# Crustle / "Palace" Agent — Results & Tuning Log

`main.py` is a tuned pilot of Shun's winning **Crustle** deck (Dwebble `344` → Crustle `345`,
ported from `palace/palace_*.json`). It reads all card/attack stats at runtime and hardcodes
deck strategy. This file logs **every tuning attempt this session and the reasoning behind it**,
plus the current meta-gauntlet standings.

---

## Part 1 — Current gauntlet (post-tuning `main.py`)

**Setup.** 10 top TEF-CRI meta decks (`data/decks/deck_<slug>.json`), each piloted by
`bare_agent.py` (the generic auto-piloting engine, zero archetype tuning). 40 seat-swapped
games per deck via `runner.py`. Win % below is **main's** win rate.

| # | Deck (opponent) | main win % | How main wins / loses |
|---|------|:---:|---|
| 1 | Dragapult ex | **100%** | prizes 25 / deck-out 11 |
| 1 | N's Zoroark ex | **100%** | prizes 21 / deck-out 19 |
| 1 | Lillie's Clefairy ex | **100%** | prizes 39 |
| 1 | Ogerpon Box | **100%** | prizes 39 |
| 1 | Raging Bolt ex | **100%** | prizes 38 |
| 6 | Hydrapple ex | 97.5% | prizes 24; 1 loss no-active |
| 7 | Slowking | 95.0% | prizes 32; ~2 losses no-active |
| 8 | Alakazam | 92.5% | prizes 21 / no-active 13; 3 losses no-active |
| 9 | Crustle (mirror-ish) | 87.5% | deck-out 26; 4 losses no-active |
| 10 | **Rocket's Honchkrow** | **67.5%**\* | prizes 15 / deck-out 12; 13 losses no-active |

\* Honchkrow is the toughest matchup and the gauntlet's 40-game sample is noisy; the dedicated
**100-game** measurement is **82.0%** (see Attempt #12). Every loss across the gauntlet is the
same failure mode: `no-active` (main's thin 8-Pokemon line gets KO'd off the board).

**vs the original Mega Lucario agent** (`main_megalucario_backup.py`): **~88%** (100 games).

---

## Part 2 — Session tuning log (2026-06-17)

**Method.** Each change was tested in Docker (`linux/amd64`, the engine `.so` is Linux-only)
against a **frozen baseline** and the relevant matchups. Rule: *keep a change only if it
demonstrably helps and does not regress other matchups.* Mirror tests use a frozen snapshot so
concurrent edits don't move the target.

### Phase 0 — Build & validate the agent
- **Ported Shun's Crustle deck** into `main.py`; agent reads `CardData`/`Attack` at runtime
  (so it knows HP / weakness / attack cost / 120-dmg Superb Scissors without hardcoding).
- **Validation:** beat the old Mega Lucario agent **25-5 (83%)**, 0 crashes / 30 games. ✓ kept.

### Phase 1 — First tuning round (levers tested blind, all dead-ends)
| # | Change | Reasoning | Result | Decision |
|---|--------|-----------|--------|----------|
| 1 | Prioritize special energy (Mist/Spiky) attach | Experts attach specials readily | 45% mirror (noise) | reverted |
| 2 | More bench bodies (cap 6) | no-active is the loss mode → more bodies = insurance | **41.7% mirror (worse)** | reverted |
| 3 | Fewer bench bodies (cap 2) | fewer exposed Pokemon = fewer free prizes | **45% mirror (worse)** | reverted |
| 4 | Go second on the coin (`IS_FIRST`) | test tempo trade | **38% head-to-head (worse)** | reverted; **go first is correct** |

**Takeaway:** the baseline (cap 3 bodies, go first) is a robust local optimum; blind levers
moved nothing or regressed.

### Phase 2 — Diagnostics (instrument *why* games are won/lost)
- Added win-condition + loss-board-state capture to `runner.py`.
- **vs Mega Lucario:** main wins by **milling the opponent out** (deck-out); **every loss is
  `no-active`**, dominated by **turn-2 opening snipes** — going second, the lone Dwebble is KO'd
  before main ever acts. *Unactionable by in-game play.*

### Phase 3 — Ground truth (read the card DB from the binary)
Ran `all_card_data()` inside Docker. Corrected several offline guesses:
`344`=Dwebble (Basic 70), `345`=Crustle (St1 150, 120 dmg, Grass+2), `1086`=Buddy-Buddy Poffin
(search **2 Basics** to bench), `1147`=Jumbo Ice Cream (heal), **`1212`/`1227`/`1235` are all
Supporters** (Cook heal / Lillie's draw / Waitress energy — only ONE per turn), `18`=Grow Grass
Energy. Deck has only **4 Basics / 8 Pokemon total** → structural fragility.

### Phase 4 — Second tuning round (card-accurate, but REGRESSED)
| # | Change | Reasoning | Result | Decision |
|---|--------|-----------|--------|----------|
| 5 | Items before hand-shuffle supporter (ITEM 3500 > SUP 3000) | Lillie's shuffles hand into deck → don't shuffle unplayed items away | 52.5% / 160 mirror (noise) | folded into stack |
| 6 | 1086 board insurance (bodies≤1 → 19000) | prevent no-active | — | folded into stack |
| 7 | Situational supporters (Waitress/Lillie/Cook by state) | only 1 supporter/turn → pick best | — | folded into stack |
| — | **Full stack (5+6+7)** | sum of above | **LOST 42.5% vs baseline / 120 games** | **reverted all** |

**Takeaway:** "smart" situational logic *regressed* the simple baseline. The robust heuristic
won. **True baseline = 88% vs Mega Lucario (100 games).** 90% solid vs Mega Lucario isn't
reachable by tuning — the ~12% loss floor is structural (turn-2 opening snipes, 4-Basic deck).

### Phase 5 — Rocket's Honchkrow matchup (the win)
Honchkrow is main's worst matchup — an **aggressive prize race** (not a mill). All main's
losses are `no-active`, but **mid-game** (turns 5-22) — meaning the agent has turns to act, so
this matchup is *tunable* (unlike Mega Lucario's pre-turn snipes).

| # | Change | Reasoning | Result | Decision |
|---|--------|-----------|--------|----------|
| 8 | Retreat-to-save a dying active | retreat the ≤40%-HP active to deny the opponent a prize | **66.7% (worse)** — loses tempo in a race | reverted |
| 9 | Bench refill ALWAYS-ON (1086 @ bodies≤2 → 19000) | free item action refills bench, prevents no-active | Honchkrow **83% (+12%)** but Mega Lucario **78% (−10%)**, mirror **45% (−5%)** | **failed guard** — overfit |
| 10 | Refill empty-bench only (bodies≤1) | surgical: only when truly empty | Honchkrow 67% (no gain) — too reactive | reverted |
| 11 | Adaptive refill, pressure = `my_prize_left ≤ 4` | deep refill only when raced (opp took 2 prizes) | Honchkrow 73-77%, Mega 86.7%, mirror 53.8% | close but marginal |
| 12 | **Adaptive refill, pressure = `my_prize_left ≤ 5`** | trigger the deep refill as soon as the opponent takes **1** prize — catch the race earlier; stay conservative (bodies≤1) in mill matchups so we don't thin our own deck | **Honchkrow 82.0% (100g)**, **Mega Lucario 86.2%**, **mirror 56.0%** | ✅ **SHIPPED** |

**Why #12 works:** the no-active losses come from the active being KO'd with an empty bench.
Buddy-Buddy Poffin refills the bench as a *free* item action (no tempo cost). Doing it
**always** (#9) helps the race but thins our own deck and over-exposes bodies, which loses the
*mill* matchups (where we win by decking the OPPONENT out). Gating the deep refill on
race-pressure (opponent has started taking prizes) keeps the mill matchups clean while still
saving the race games. The prize convention was confirmed empirically (Honchkrow improved →
`my_prize_left` decreasing does mean we're being KO'd).

**Net effect of the session:** Rocket's Honchkrow **71.7% → 82.0%** (+10pts), with **no
regression** (Mega Lucario ~88%→86%, mirror 50%→56%, all other meta decks 92-100%).

### Dead-ends worth not retrying
Special-energy priority, bench-size changes (cap 2 / cap 6), go-second, retreat-to-save, and
the situational-supporter stack all tested neutral-or-worse. The baseline heuristic is strong;
only the **matchup-adaptive bench refill** (Attempt #12) beat it.

---

## Files
- `agents/main.py` — shipped agent (Crustle, with adaptive bench refill)
- `agents/main_honch_base.py` — frozen baseline used for the Phase-5 mirror A/B
- `agents/main_megalucario_backup.py` — original Mega Lucario agent (external benchmark)
- `agents/bare_agent.py` — generic auto-pilot (`BARE_DECK` env selects the deck)
- `runner.py` — match runner (win-condition + loss-board-state instrumentation)
- `data/decks/deck_<slug>.json` — the 10 resolved meta decks
- `results/gauntlet_new.txt` — raw output of the Part-1 gauntlet

---
---

# Part 3 — Counter-deck work log (opponent-side R&D)

This half of the session attacked the problem from the *other* side: building decks/agents to
**beat** `main.py`. All numbers from `runner.py` under Docker (`linux/amd64`), seat-swapped.
**Measurement caveat used throughout:** win-rate variance is ±3-4% at 500-1000 games — every
conclusion below is taken at ≥1500-2000 games (smaller samples repeatedly produced deceptive
"80%" readings that were really ~76%).

## 3.0 — Meta gauntlet (top-10 limitlesstcg decks vs main)
Built the 10 current top meta lists into the cabt pool (`tools/build_decks.py`, name→id
resolver) and ran each (bare auto-pilot) vs main. **main wins all 10**; Rocket's Honchkrow was
the toughest (~30%). Infra: `data/cards.json` (full pool dump WITH names — card names ARE
readable offline), `tools/inspect_deck.py`, `tools/trace_deck.py`. Writeup: `FIRE_COUNTER.md`
is Part-3's main artifact; gauntlet detail in `meta-gauntlet` memory.

## 3.1 — Tuning Rocket's Honchkrow (combo deck) → ~55% vs main
Honchkrow's win condition is invisible to a damage-stat pilot: **'Rocket Feathers' [CC] lists
0 damage but does 60 × (Team Rocket Supporters discarded from hand)**. Built `agents/honchkrow.py`.
| Lever | vs main |
|---|:---:|
| bare pilot (can't see Rocket Feathers) | 23% |
| + model Rocket Feathers, hoard TR-supporter ammo | ~33% |
| + **precise ammo-discard** (discard only enough to KO; conserve the rest) | **55%** |
| + setup-active=Murkrow, energy-focus (round 2) | ~55% (within noise) |
Result: **23% → 55.7%** (1000g). Conclusion: a *peer* matchup (tuned Honchkrow vs tuned Crustle)
caps at ~55% — Honchkrow sets up ~turn 7 vs Crustle's ~turn 3. **80% by piloting is impossible**
for a peer deck. (See `HONCHKROW_TUNING.md`.)

## 3.2 — Anti-Crustle Fire counter (the winning idea)
Crustle is **mono-Grass, 8 Pokemon, all weak to Fire**. A fast **NON-ex Fire** deck OHKOs
Crustle (≥100 base ×2 = ≥200 vs 150 HP) with 130-HP bodies that survive Crustle's 120 → ~2:1
prize race. Build-up (`agents/fire.py` + `data/decks/deck_fire.json`):
| Change | vs main_cur |
|---|:---:|
| ex-heavy Fire (gives easy prize swings) | 6.7% |
| **non-ex** Fire (Ho-Oh 318 / Volcanion 663), bare pilot | 65% |
| + energy CONCENTRATION on one attacker | 69% |
| + **18 Fire energy** (all-[RRC] deck starves on 16) | 74% |
| + smart gust (Boss's Orders → most-developed benched threat) | 74% |
| + tanky-active (bench the fragile 110-HP Hearthflame) | **74.7%** (≈80% on the play) |
Key gotchas found: Mega Kangaskhan's "200" is a coin-flip averaging ~22 dmg; damage tools
(Maximum Belt / Brave Bangle / Black Belt's) only boost vs opponent **ex** (useless here);
Hero's Cape is 1-per-deck (errorType=4).

## 3.3 — Goal ">80% vs BOTH Crustle AND the Crustle/Typhlosion hybrid": INFEASIBLE
Tested **7 archetypes**; mapped the Pareto frontier (each point raising Hybrid lowers Crustle):
| archetype | Crustle | Hybrid |
|---|:---:|:---:|
| pure non-ex Fire | **74.7%** | 48% |
| tank (Mega Kangaskhan) | 65% | **57%** |
| Volcanion ex + Premium Power Pro | 62% | 50% |
| non-ex toolbox (Fire+Water, Prism energy) | 65% | 58.5% |
| Fire+Water toolbox v1 | 52% | 45% |
| Zangoose ex (180 [CCC]) | 50% | 50% |
| Water ex (Keldeo/Kyurem) | 8.5% | 37% |
**No deck > 60% vs the hybrid.** Structural reason: Crustle needs *non-ex* (prize denial) but
Typhlosion (170 HP Fire) needs *ex/Water power* to OHKO — mutually exclusive; the two opponents
have complementary Grass/Fire weaknesses; the hybrid is the format's strongest deck (80.7%
gauntlet avg, built type-resilient). (See `beat-both-infeasible` memory.)

## 3.4 — The "Rebel" submission (final deliverable): ~77-78% vs Crustle
Packaged the Fire counter as the self-contained `submission_rebel.py` (deck inlined) and pushed
the Crustle number with three pilot upgrades, each from telemetry:
| Change | vs main_cur (≥1500g) |
|---|:---:|
| ported fire.py → main_v2_rebel | ~74% |
| + survival-heal (heal a 130-HP body one hit from death so main gets no prize) | ~76% |
| + clean-active (keep Ho-Oh active; its Flap chips 50 vs Volcanion's 0-dmg Singe spam — Singe was fired 1382×) | ~78% |
| + accel-rush (Waitress energy-accel when an attacker is mid-power-up) | **~77-78%** |
**Going FIRST is ~80-84%**; the going-second coin flip holds the *overall* just under a stable
80%. A 10-Pokemon variant tested worse (76.1%) — kept 9 Pokemon / 18 energy.

### Rebel vs the deck zoo (1000g each)
| Opponent | Deck | Rebel win% |
|---|---|:---:|
| `main` / `main_cur` | **Crustle** (mono-Grass) | **76-77%** |
| `main_v3` | Crustle/Typhlosion hybrid (tuned) | 52.6% |
| `main_v4` | hybrid (+Rare Candy) | 46.3% |
| `main_v2` | hybrid (generic) | 45.7% |
| `main_typhlosion` | hybrid | ~50% |

And — **which agent beats Crustle best** (vs `main_cur`, 1000g): `main_v2_rebel` **74.3%** >
`main_typhlosion` 64.9% > `main_v4` 63.2% > `main_v2` 61.8% > `main_v3` 60.0%. The hybrids also
beat Crustle (their Typhlosion is a Fire attacker hitting Crustle's weakness), but **pure Fire
does it best**.

## Part-3 deliverables / files
- `submission_rebel.tar.gz` — competition submission (structure matches the other gangs:
  `main.py` + `deck.csv` + `cg/`); extract-and-run verified ~76% vs Crustle. Staged in `build/rebel/`.
- `submission_rebel.py` / `agents/main_v2_rebel.py` — the Rebel agent (deck inlined)
- `data/decks/deck_fire.json` — the Fire deck list
- `agents/fire.py`, `agents/honchkrow.py` — tuned counter agents
- `tools/trace_deck.py` (per-seat / per-first telemetry for any deck), `tools/inspect_deck.py`,
  `tools/build_decks.py`
- Companion writeups: `FIRE_COUNTER.md`, `HONCHKROW_TUNING.md`

---
---

# Part 4 — Agent-side deck/strategy R&D (later session)

This half built the *main* agent forward: from the single-line Crustle (Part 2) to the
Crustle/Typhlosion hybrid, driven by analysing **ladder losses** and **winners' replays**.
Same discipline: a change ships only if it holds up on the **bare-pilot gauntlet** (40 games ×
11 meta decks), never on a head-to-head mirror alone (the mirror misled *seven* times).

## 4.0 — Ladder-loss diagnosis (`data/loser/*.json`)
- **`lost.json`** (Crustle mirror vs Tensa.bit): lost turn 11 by `no-active` — lone Crustle at
  30 HP, **empty bench, only 1–2 energy**. Structural thinness (4 Basics / 8 Pokemon).
- **`lost_1.json`** (vs `seven`'s Mega Abomasnow ex 350-HP Water deck): lost turn 4, stuck at
  **Quilava (mid-evolution), empty bench**, never drew Ultra Ball / Rare Candy / 2nd Basic —
  a brick punished by an aggressive deck. Both losses = same root cause: **under-development**.

## 4.1 — The Typhlosion pivot (`winners.json` → `main_v2`)
`palace_typhlosion/winners.json` showed the winning build is a **Crustle + Ethan's Typhlosion**
two-line deck (20 Pokemon / 16 energy vs the Palace list's 8 / 31). Adopted it as `main_v2`
(generic `bare_agent` engine + deck inlined).
| Test | Result |
|---|:---:|
| `main_v2` (Typhlosion) vs `main` (single-line Crustle), head-to-head | **67.5%** |
| `main_v2` vs Rocket's Honchkrow | 72.5% (only 4/40 no-active vs Crustle's ~28%) |
| `main_v2` bare-pilot gauntlet average | **80.7%** |
Verdict: the two-line deck is a real upgrade — fixes the empty-bench brick.

## 4.2 — Non-ex Fire monster experiment (rejected)
Tested swapping the monster core for a non-`ex` Fire attacker (avoid the 2-prize `ex`
liability). Surveyed the pool: **Ceruledge `797`** (220 dmg for **1** energy), **Reshiram `794`**
(240/4, Basic). Built 3 variants on a shared shell.
- **Two deck-legality bugs found** (saved to memory): Hero's Cape `1159` is **ACE SPEC = max 1**
  (`errorType=4`); the 4-copy limit is **by card NAME not ID** (8 Charcadet via two printings →
  `errorType=2`).
- After fixing: `fire_reshiram` 35%, `fire_both` 10% **vs `main_v2`** — **none beat Typhlosion.**
  Ceruledge's 220/1 doesn't carry; the line is capped at 8 Pokemon (still bricks). **Abandoned.**

## 4.3 — `main_v3` archetype overlay → pure Buddy Blast (the 2nd real win)
`main_v3` = `main_v2` + a 5-part overlay (Buddy Blast damage scaling, Ethan's-fuel priority,
energy concentration, Rare Candy skip, Boss's-Orders gust + retreat-to-save). The key insight:
**Buddy Blast lists 40 dmg but really deals 40 + 60 × (Ethan's Adventure in discard)** — a fact
the generic engine literally cannot read.
| Version | Gauntlet avg |
|---|:---:|
| `main_v3` full overlay (deck bug: 64 cards → fixed to 60) | **66.1%** (−14.5) |
| `main_v3` ablated (removed gust + energy-concentration + retreat) | 75.5% (still −5) |
| **`main_v3_pure`** = `main_v2` + **ONLY** the Buddy Blast damage fix, zero preference bonuses | **83.2%** (+2.5 vs v2) ✅ |
`main_v3_pure` beat v2 on 8/11 decks (raging-bolt/lillie's/honchkrow +7.5), mirror 50% (no
regression). **Lesson confirmed:** the *factual damage correction* helps; every play-style
heuristic bolted alongside it (gust, energy-concentration, retreat, preference bonuses) *cost*
14 points. Isolate fixes; never bundle heuristics.

## 4.4 — `palace_3.json` Cook replication (rejected, 7th heuristic failure)
Mined Shun's winning Crustle play in `palace_3.json`: **Cook (heal) only when damaged**;
Lillie's/Poffin proactively; never Waitress; attacks aggressively (ATTACK ×10). Replicated the
clean rule (Cook scored low unless a Pokemon is ≥60 damaged):
| Test | Result |
|---|:---:|
| Cook-rule vs no-Cook baseline, head-to-head | +52.5% (looked good) |
| Cook-rule vs no-Cook, **bare-pilot gauntlet** | **87.0% vs 90.0% (−3.0)** — REVERTED |
Seventh time a supporter heuristic looked good head-to-head and **regressed the field**. Shun's
human read doesn't generalise into a rule.

## 4.5 — Final standings (the two leaders are a near-tie)
| Agent | Deck / pilot | Bare-field gauntlet | Head-to-head |
|---|---|:---:|:---:|
| **`agents/main.py`** | tuned Crustle + adaptive refill | **90.0%** | loses to Typhlosion 46.2% |
| **`main_v3_pure`** | generic engine + Typhlosion + Buddy Blast | 83.2% | **beats Crustle 53.8%** |
Metrics disagree: tuned Crustle crushes the *untuned* bare field; the Typhlosion deck wins the
direct duel and is intrinsically stronger. Recommendation: **submit both** and let the ladder
break the tie; if one, `main_v3_pure`.

## Overarching lesson of the whole session
Across ~9 phases and **~20 changes**, only **TWO** improved the gauntlet — both **factual
corrections the engine cannot read from card stats**:
1. Matchup-adaptive bench refill, gated on real prize-state (Part 2 #12): Honchkrow +10.
2. Buddy Blast real-damage calculation (Part 4.3): gauntlet +2.5.

Every **play-style heuristic** — special-energy priority, bench-size, go-second, retreat-to-save,
situational supporters (×2), gust scoring, energy concentration, Cook-when-damaged — tested
neutral-or-worse on the diverse field, even when it won a head-to-head. **The generic engine is a
robust local optimum; beat it only with facts it can't see, never with "smarter" play.**

## Part-4 files
- `agents/main_v2.py` — generic Typhlosion (80.7%)
- `agents/main_v3.py` — full overlay (regressed; kept for reference)
- `agents/main_v3_pure.py` — **v2 + pure Buddy Blast fix (83.2%, current best deck)**
- `agents/main_crustle_base.py` — Crustle no-Cook baseline (Phase 4.4 A/B)
- `data/decks/deck_typhlosion.json`, `deck_fire_{both,ceruledge,reshiram}.json`
- `results/gauntlet_{typhlosion,v3,v3pure,cook}.txt` — raw gauntlet outputs
