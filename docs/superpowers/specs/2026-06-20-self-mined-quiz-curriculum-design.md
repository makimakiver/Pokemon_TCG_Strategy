# Self-Mined, Win-Rate-Driven Quiz Curriculum — Design

**Date:** 2026-06-20
**Status:** Approved direction; pending spec review → implementation plan
**Related:** `rl/outer_loop.py` (`run_sgs`), `rl/targets.py`, `rl/eval_sgs_mcts.py`, `rl/quiz_miner.py` (new), memory `sgs-rl-plan`, `crow-dataset-sgs`

## 1. Goal

Make the act of **solving quizzes directly raise the solver's win-rate in full
simulation games.** The solver plays full games; its **lost** games are mined into
new "win-from-here" quizzes spanning **all phases (opening → endgame)**; it drills
those positions; **live simulation win-rate is the headline metric and the
curriculum driver.** This closes the SGS loop into a self-improving system whose
quizzes come from the solver's own weaknesses, not a frozen position set.

### Motivating failure (why this design)
The prior SGS run reached 29% solve-rate on the fixed 652-position set D but **0%
full-game win-rate** — because D is **mid-game-only**, so the opening was never in
the quiz distribution. Mining the solver's own games across all phases puts the
exact decisions that win/lose real games into the curriculum.

### Decisions captured (brainstorming)
1. **Self-improvement source:** mine quizzes from the solver's **own games** (close the loop).
2. **Mining criterion:** decision points in games the solver **LOST**.
3. **Lifecycle:** **augment** the pool with mined quizzes; **retire** a quiz once `solve_rate ≥ τ`; **cap** the mined buffer (evict oldest).
4. **Win-rate link:** mine **all phases** (incl. opening); **live simulation win-rate** is the headline metric + curriculum driver (NOT solve-rate).
5. **Per-game mining cap:** small **K ≈ 5** decision points per lost game, sampled across the trajectory's phases (bounds compute).

## 2. Non-goals
- Per-move blame/credit-assignment weighting of quizzes (a possible later enhancement; v1 mines uniformly across phases).
- Replacing D entirely — D stays as the grounded anchor; mined quizzes augment it.
- A new objective — reuses the AlphaZero (π,z) solver objective and the seed-first conjecturer already in `run_sgs`.

## 3. Architecture

```
                 ┌──────────────── run_selfmined (per generation) ────────────────┐
 active pool ──► sample batch ─► seed-first conjecturer ─► MCTS solver ─► AZ update │
 (unsolved D                                                       ▲                │
  ∪ mined)                                                         │                │
       ▲                                                           │                │
       │            live full games (win-gated rollouts) ──────────┘                │
       │                    │                                                       │
       │              LOST game? ──► mine ≤K decision points (ALL phases)           │
       │                    │         determinize (rl/determinize) → ScenarioSpecs  │
       └──── retire solved ─┴──► add to mined buffer (cap RL_MINE_CAP, evict oldest)│
                 headline metric = rolling LIVE WIN-RATE ◄──────────────────────────┘
```

## 4. Components

### 4.1 `rl/determinize.py` (new, shared)
Lift the hidden-info determinization out of `eval_sgs_mcts.py` so the miner and the
live-MCTS harness share ONE implementation.
- `build_spec(obs, deck_me, deck_op, poke_ids) -> ScenarioSpec` — partition the
  hidden pool (reuse `targets._hidden_pool/_visible_ids/_take`) into
  your_deck/your_prize/opp_deck/opp_prize/opp_hand/opp_active for the live obs.
- `eval_sgs_mcts.py` is updated to import `build_spec` from here (no behavior change).

### 4.2 `rl/quiz_miner.py` (new)
- `collect_live_for_mining(policy, env, n_games, prior_weight, actor, k=5) -> (rollouts, mined, wins)`:
  - plays `n_games` live full games vs the env's scripted opponent, recording the
    **raw obs** (`_to_dict(env._obs)`) at every solver decision alongside the rollout;
  - for each **lost** game, calls `mine_from_trajectory(obs_seq, deck_me, deck_op, poke_ids, k)`;
  - returns the live `Rollout`s (for training), the list of mined `ScenarioSpec`s,
    and per-game win bools (for the win-rate metric).
- `mine_from_trajectory(obs_seq, deck_me, deck_op, poke_ids, k) -> list[ScenarioSpec]`:
  - candidate positions = solver decision points that are **mid-result** (`result==-1`),
    have a real `select` + `search_begin_input`, and are **non-degenerate**
    (`guide.non_degeneracy > 0`); **all phases including the opening qualify**;
  - **sample ≤ k** of them spread across the trajectory (evenly spaced indices) to
    bound compute;
  - `determinize.build_spec` each → `ScenarioSpec` (target_id = `selfmined_<gen>_<game>_<i>`,
    source = `selfplay`); `validate_shapes()`; drop on failure.

### 4.3 `rl/outer_loop.py` → new `run_selfmined(...)`
A new driver (keeps `run_sgs` untouched) reusing its machinery.
- Signature: `run_selfmined(generations, batch_size, run_name, init_ckpt=None, live_games=8, mine_k=5, mine_cap=200, problem_set=None, objective="alphazero", conjecture_after=2)`.
- State: `mined: list[ScenarioSpec]` (capped), `solve_rate`, `stale`.
- **Active pool** each gen = `[t for t in (D + mined) if solve_rate[t.target_id] < τ]`
  (retire solved). Sample `batch_size` from it; if empty, fall back to a random D sample.
- Per gen:
  1. batch → seed-first conjecturer → MCTS rollouts → (collected for the update);
  2. `collect_live_for_mining(...)` → win-gated live rollouts (losing live rollouts get
     `pi=None`, value-only) added to the update; `mined.extend(new_quizzes)`; evict oldest
     beyond `mine_cap`;
  3. AlphaZero update on all rollouts; conjecturer update on R_synth;
  4. record `live_winrate` (rolling), `pool_size`, `mined_total`, `retired`.
- Checkpoints + `history.json` (with the win-rate spine) every 5 gens.

## 5. Data flow + the win-rate spine
`live_winrate` is THE metric: logged every generation and used to retire mastered
quizzes (so the pool always reflects *current* weaknesses). The trajectory graph =
**live win-rate (headline)** + active-pool size + cumulative mined count — not
solve-rate.

## 6. Error handling / risks
- **Won-everything generation:** mines nothing — fine, the pool just doesn't grow.
- **Determinization failure:** that position is skipped (logged at debug); never fatal.
- **Opening quizzes are nearly full-length games** (expensive): bounded by `mine_k`
  (≤5/game) + `mine_cap` (≤200 buffer).
- **Genuinely-lost mined positions:** the seed-first conjecturer weakens them into
  solvable lemmas; retire+cap keep the pool current and bounded.
- **Distribution drift:** D remains in the pool as the grounded Crow-loss anchor.

## 7. Testing
- **Unit (host, no engine):** `mine_from_trajectory` on a synthetic obs sequence
  (stub obs dicts with `result`, `select`, `search_begin_input`, player states) →
  asserts ≤k specs, all-phase coverage, degenerate/terminal skipped. `determinize.build_spec`
  round-trip on a synthetic obs.
- **Docker smoke (`rl/smoke_selfmined.py`):** 2 gens, `live_games=4, mine_k=3,
  mine_cap=20`, warm-start from a checkpoint → assert: mined buffer grows, a solved
  quiz retires, `live_winrate` present in history, loss finite, checkpoint written.
- **Metric:** over a real run, **live win-rate trend** (should climb) is the success
  criterion — explicitly NOT solve-rate.
