# Strategy Simulation Results & Suggestions

## Methodology
- **Engine**: cabt-sim Docker container (linux/amd64), deterministic seeds
- **H2H**: Round-robin, sides swapped each game, 40 games/pair
- **Meta**: Each agent vs bare_agent on all 10 meta decks, 20 games/deck (200 total)
- **Agents tested**: main_v2, main_v2_baseline, main_v3 (full overlay),
  main_v3_pure (clean Buddy Blast fix), main_v4 (v2+BB fix v2), main_typh_base,
  main_typhlosion

---

## PART 1: HEAD-TO-HEAD RANKING (40 games/pair, vs other strong agents)

| Agent          | H2H Total | vs v2  | vs v3  | vs v3_pure | vs typhlosion |
|----------------|-----------|--------|--------|------------|---------------|
| **v3_pure**    | **BEST**  | 58% ✅ | 62% ✅ | —          | 58% ✅        |
| typhlosion     | 2nd       | 55%    | 50%    | 42%        | —             |
| v3             | 3rd       | 50%    | —      | 38%        | 50%           |
| v2             | 4th       | —      | 50%    | 42%        | 45%           |
| v4             | ≈ v2      | 50%    | 52%    | 52%        | —             |

**Winner: main_v3_pure** — beats every other variant in direct head-to-head.

---

## PART 2: META RANKING (200 games vs bare_agent, all 10 decks)

| Agent       | DRAG | CRUS | HYDR | OGER | ZORO | SLOW | ALAK | BOLT | LILL | HONC | TOTAL |
|-------------|------|------|------|------|------|------|------|------|------|------|-------|
| v2          | 85%  | 75%  | 60%  | 85%  | 95%  | 60%  | 95%  | 90%  | 95%  | 80%  | **83%** |
| v3          | 75%  | 65%  | 75%  | 95%  | 95%  | 75%  | 80%  | 90%  | 90%  | 80%  | **82%** |
| v3_pure     | 90%  | 72%  | 80%  | 85%  | 100% | 60%  | 90%  | 75%  | 80%  | 70%  | **78%** |

**Winner: main_v2** for meta breadth, but the gap is within noise (±5% per cell).

---

## PART 3: LOSS-REPLAY ROOT CAUSE (from data/loser/*.json)

The #1 bug across ALL replays: **Buddy Blast deals 40 + 60×N (N = Ethan's
Adventure in discard), but the engine only reads base damage=40.** With 1
fuel → 100 dmg, with 2 fuel → 160 dmg, for just 1 {R} energy.

```
fuel=0:  BB=40 dmg   (worse than Crustle's 120)
fuel=1:  BB=100 dmg  (2x weak = 200 → OHKOs most non-ex)
fuel=2:  BB=160 dmg  (matches Steam Artillery at 1/3 the energy cost)
fuel=3:  BB=220 dmg  (OHKOs 210hp ex pokemon)
```

---

## SUGGESTIONS (ranked by impact)

### 1. 🏆 USE main_v3_pure AS YOUR COMPETITION AGENT
It's the **only variant that beats v2 in head-to-head (58%)**. Competition
opponents are strong pilots (zakopuro, Tensa.bit, etc.), so H2H is the most
predictive metric. The Buddy Blast fix is a 1-line conceptual change that
makes Typhlosion a real attacker.

### 2. ❌ DROP main_v3 (full overlay) — IT'S OVER-ENGINEERED
Despite adding gust KOs, Rare Candy priority, retreat logic, and Ethan's
fueling, **v3 is WORSE than v3_pure** (H2H 38%). The extra scoring constants
create competing priorities that confuse the greedy scorer. Lesson: **one
correct fix beats five heuristics.**

### 3. 🔧 FIX THE SLOWKING MATCHUP (worst for all agents: 50-60%)
Slowking is the only deck all agents struggle with. Investigate:
- Slowking likely runs ability-lock or energy-disruption effects
- Check if the agent's loop guard (`turn_actions > 80`) is being triggered
  by Slowking's abilities, causing premature END turns

### 4. ⚡ ADD EARLY-GAME BENCH SAFETY (prevents no-active losses)
Several losses end at turn 4-8 with an empty bench after the active is KO'd.
Fix: in the first 3 turns, **always play a basic to bench before evolving or
attaching**. Current code already has `n_bodies >= 4 → score=30` for basics,
but the threshold should be `>= 3` and basics should score **higher** early.

### 5. 🔥 CONCENTRATE ENERGY ON ONE ATTACKER
The generic engine spreads energy evenly. Steam Artillery needs [R,R,C] = 3
energy. Track which Typhlosion has the most energy and keep feeding it until
3, instead of spreading 1 energy across 3 bodies.

### 6. 🎯 PRIORITIZE ETHAN'S ADVENTURE PLAY (Buddy Blast fuel)
Each Ethan's Adventure in discard = +60 damage to every future Buddy Blast.
The agent should play Ethan's Adventure supporters **earlier** (before
attacking), not just when it happens to draw them. Current supporter scoring
(3000 flat) undervalues them vs items (3500).

### 7. 🛡️ DON'T TRUST THE META BENCHMARK TOO MUCH
The meta benchmark uses bare_agent (weak pilot) as the opponent. Real
competition agents are much smarter. The 83% vs 78% meta gap between v2 and
v3_pure is **less important** than the 58% H2H advantage of v3_pure.

---

## RECOMMENDED ACTION

```bash
# Promote v3_pure to your main submission agent
cp agents/main_v3_pure.py agents/main_v2.py
```

Then iterate on suggestions #3-#6 one at a time, re-running the 40-game H2H
confirmation after each change:

```bash
docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
  cabt-sim --a agents.main_v2 --b agents.main_v2_baseline -n 40
```
