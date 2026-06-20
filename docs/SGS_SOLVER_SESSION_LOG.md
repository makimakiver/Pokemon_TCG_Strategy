# SGS Solver — Session Log (NN+MCTS solver env → win-rate reality check → self-learning roadmap)

Single-document record of the session: what was built, every training experiment and
its result, the bugs found and fixed, the honest findings, the design specs produced,
and the staged roadmap. Companion to `docs/SGS_CONJECTURER_SESSION_SUMMARY.md` (which
covered the SmolLM conjecturer side).

---

## TL;DR

- **Built** the full NN+MCTS **solver environment** for the SGS stack: problem-set IO,
  MCTS that emits training targets, AlphaZero objective, the SGS training driver, and a
  determinized **live-MCTS harness** so the search-mode solver can play full games.
- **Generated the real problem set** from the SmolLM conjecturer on the Mac mini:
  **652 problems, 90.3% valid** LLM edits (`rl/runs/problems.jsonl`).
- **Fixed 3 real bugs** surfaced by running the engine: illegal-action `log(0)` → NaN/+inf;
  an MCTS search-node leak (14 GiB → 1 GiB); and zero-gradient starvation (→ switched the
  solver to an AlphaZero `(π,z)` objective).
- **Trained the solver** (AlphaZero, 150 gens): **0 → 29% solve-rate** on the conjecturer's
  problems. **But that net is 0% in full games** — even with MCTS — because it only ever
  trained on **mid-game** positions and never learned the opening.
- **The hard finding:** none of the RL/MCTS approaches beat the hand-tuned scripted agents.
  Best stable result ≈ **30% in a favorable matchup**, and that 30% came from a **separate
  pre-existing self-play net (`kaggle_mcts`), NOT from SGS.**
- **Diagnosed the cause** (train/eval distribution mismatch + cold-start trap) and **specced
  the fix:** a self-mined, **win-rate-driven** quiz curriculum (Stage 1) feeding a conjecturer
  auto-generation + fine-tune + difficulty-ramp self-learning loop (Stage 2).

---

## 1. Original goal

> "Create an environment for the solver agent (NN + MCTS) to solve the problem set the
> conjecturer proposed (`docs/SGS_CONJECTURER_SESSION_SUMMARY.md`)."

Solver = `PointerPolicy` (pointer net + value head, deck-agnostic). The conjecturer
(SmolLM, trained separately) proposes mid-game "lemma" positions; the solver must win
from them. MCTS sharpens the net using the engine's `search_begin`/`search_step` tree.

## 2. What was built (committed on `feat/nn-mcts-solver-env`)

| File | Purpose |
|---|---|
| `rl/problem_set.py` | load/seed the conjecturer's lemmas (target_id + edit-script), engine-free |
| `rl/conjecturer/export_problems.py` | export the conjecturer's problem set to JSONL (LLM on mini / parametric on CPU) |
| `rl/mcts.py` | + `search_policy` (visit-distribution targets) + **search-node release (leak fix)** |
| `rl/env.py` | + `search_state()` so the MCTS actor roots at the live search node |
| `rl/train_solver.py` | pluggable `actor`; **MCTS actor returns only legal actions** + full π target |
| `rl/solver_objectives.py` | + **AlphaZero `(π,z)`** objective (dense; alongside reinforce_half/ppo/cispo) |
| `rl/outer_loop.py` | `run_sgs(use_mcts, problem_set, objective, conjecture_after, init_ckpt, live_games)` — seed-first curriculum + live-game mixing |
| `rl/eval_mcts_vs_agent.py` | eval the `kaggle_mcts` net vs any `agents.<name>` |
| `rl/eval_sgs_mcts.py` | **determinized live-MCTS harness**: SGS PointerPolicy plays full games |
| `rl/train_multi.py` | multi-opponent live PPO from a warm-start |
| `tests/rl/*` | problem_set, export, visit_policy, actor plumbing, AlphaZero objective |

Plus the conjecturer problem set on the mini and the eval checkpoints under `rl/runs/eval_ckpts/`.

## 3. Infrastructure

- **Engine** (`cg/libcg.so`) is Linux x86-64 only → Docker `linux/amd64` (`cabt-rl` image).
- **Mac mini** `takayukimac-mini` (Tailscale, user `makimakiver`, repo `~/Pokemon_TCG_Strategy`):
  Colima VM **16 GiB, vz + rosetta** → runs the engine **and** the SmolLM on MPS. The
  **preferred training box** (the laptop's Docker is only **7.6 GiB and OOM-kills MCTS**).
  Non-interactive ssh needs `export PATH=/opt/homebrew/bin:$PATH`; always pass
  `-e PYTHONUNBUFFERED=1` to `docker run` or the log stays empty (block-buffered).
- **Problem set export:** `RL_CONJ_DEVICE=mps RL_CONJ_MODEL=…SmolLM2-360M-Instruct
  RL_CONJ_LORA=…/conjecturer_lora_sft` → 652 rows, **`{llm:589, fallback:63}` = 90.3% valid**.

## 4. Bugs found & fixed (by running the engine for real)

1. **Illegal-action → NaN/+inf loss.** The MCTS actor sampled over all engine option edges,
   ignoring the env's per-micro-step `pending` mask → returned already-picked (masked) options
   → `policy.evaluate` `log_prob = log(0) = -inf` → CISPO `0·-inf = NaN`, REINFORCE `-Σ = +inf`,
   plus a no-op stall loop. **Fix:** mask MCTS π by `obs["mask"]`, renormalize, defer to the
   masked net when no legal option has MCTS mass. Diagnostic: nonfinite_logp **8 → 0** / 1256 steps.
2. **MCTS search-node leak → OOM.** Every simulation's `search_step` created an engine node that
   was never freed → **13.8 GiB in one generation** → SIGKILL (137). **Fix:** record created node
   ids and release the exploration subtree after each search (keep the env root). Mem **14 → 1 GiB**.
3. **Zero-gradient starvation.** CISPO/REINFORCE-½ need within-problem reward variance; MCTS makes
   per-problem outcomes near-deterministic (variance ≈ 0) → degenerate groups → `loss 0.000, used=0`,
   net never trains. **Fix:** **AlphaZero `(π,z)`** — policy CE to MCTS visits + value MSE to outcome,
   CE restricted to legal slots (no `0·-inf`). Dense per-step signal regardless of wins.

**Objective history:** solver CISPO → (user) REINFORCE-½ → **AlphaZero** (dense, what worked).
For the **ExpRL conjecturer** fine-tune the user later chose **GRPO** (scoped override; CISPO stays
for the solver + the R_synth conjecturer path).

## 5. Experiment log (every training run + result)

| Run | Setup | Result |
|---|---|---|
| `train_az` (mini) | AlphaZero, 150 gens, 8 sims, on the 652 problems | **0 → 29% solve-rate** on D (187/652 peak @ gen 133); mem bounded ~1 GiB. **`az_final` = 0% full-game** |
| BC bootstrap | `bootstrap_solver`: clone `main_v5` into a net (2,579 state→move pairs) | floor 16.7% → **BC'd net 50%** vs bare (prior-assisted) |
| `live_final` | BC clone + live PPO vs bare | deployable **29% bare / 37.5% main** (noisy ~30%) |
| `train_multi` | BC clone + multi-opponent PPO, anneal prior→0 | **flat ~0.15–0.25**, collapses as prior anneals (raw net can't stand alone) |
| `model_best` (kaggle_mcts) | **separate** NN+MCTS self-play net (transformer, 50 MB) | **30% vs main_v3_pure (stable, 40 games)**, 0% vs bare/main (matchup-bound) |
| more-on-main_v3_pure | continue `model_best` self-play, mix 25% vs main_v3_pure | **flat 18/12/12** — training more did **not** help |
| `az_final` + live-MCTS | the SGS net in the determinized full-game harness | **0% vs all 3 agents** (never learned the opening; MCTS can't rescue a bad-at-root net) |
| `train_sgs_live` | BC clone + problems + win-gated live games | **`live_win` starts 0.50** (warm-start works) |
| `train_sgs_b` | **`az_final`** + problems + win-gated live games | solve-rate climbs (14%) but **`live_win 0.00` flat over 59 gens** — cold-start trap confirmed |

## 6. Key findings (the honest read)

1. **SGS trains a mid-game specialist, not a full-game agent.** 29% solve-rate on D is real, but
   D is **mid-game-only**, so the net is **0% in full games even with MCTS** — it never saw the
   opening and loses every game by `no-active` at ~turn 5.
2. **Cold-start trap.** Live-RL from a 0%-opening net never wins → win-gated signal is always empty
   → `live_win` stuck at 0 (proven over 59 gens). You can't bootstrap the opening from live RL alone.
3. **BC warm-start fixes the opening** (0 → 50% vs bare) — imitating a tuned pilot teaches setup.
4. **No RL/MCTS net beat the hand-tuned scripts.** Best stable ≈ **30% in a favorable matchup**.
   The scripted pilots are strong; surpassing them needs GPU-scale self-play, a bigger net,
   deck-matched training, and a denser-than-win/loss reward.
5. **The 30% "NN+MCTS" is NOT from SGS** — it's the separate pre-existing `kaggle_mcts` net.
   SGS's own net is 0% full-game.
6. **16-game evals are noise** (37% vs 12% = the same ~30%). Use **40+ games** for any real number.

## 7. Design specs produced (under `docs/superpowers/specs/`)

- `2026-06-20-nn-mcts-solver-env-design.md` — the solver env (built, this session).
- `2026-06-20-exprl-conjecturer-reasoning-design.md` — **Stage 2**: fine-tune the SmolLM
  conjecturer's chain-of-thought via ExpRL (arXiv 2606.17024): GLM-5.2 reference (cached) +
  a per-sample **LLM rubric judge** (1–5 → `(s−1)/4`) on `(problem, SmolLM CoT, reference)`,
  optimized with **GRPO + KL** on K-sample groups.
- `2026-06-20-self-mined-quiz-curriculum-design.md` — **Stage 1**: mine the solver's **lost**
  games into "win-from-here" quizzes across **all phases** (≤K≈5/game), active pool =
  unsolved(D ∪ mined) with retire-on-solved + a cap, and **live simulation win-rate as the
  headline metric and curriculum driver** (fixes the mid-game-only transfer gap).

## 8. Roadmap — the self-learning loop (user's intended flow)

**Stage 1 — Log data → trainable data → win-rate vs `bare_agent`** *(build & prove first)*
- Turn logged games (Crow/loss replays **+ the solver's own self-play**) into all-phase
  "win-from-here" quizzes; train the solver; **gate on measurable win-rate improvement vs
  `bare_agent`.** No automation yet — first prove the data→training→win-rate chain.
- = the `self-mined-quiz-curriculum` spec.

**Stage 2 — Automate + self-improve** *(after Stage 1 wins)*
- SmolLM conjecturer **auto-generates** quizzes; **fine-tune** it (ExpRL/GLM judge) to **improve
  quiz quality**; **ramp difficulty** as the solver improves → the co-evolving self-learning loop.
- = the `exprl-conjecturer-reasoning` spec + a difficulty ramp.

## 9. Where things live

- **Branch:** `feat/nn-mcts-solver-env` (all solver-env code + the 3 specs + this log).
- **Problem set:** `rl/runs/problems.jsonl` (laptop + mini), 652 rows, 90.3% valid.
- **Checkpoints (`rl/runs/eval_ckpts/`):** `az_final.pt` (SGS solver), `bc_init.pt`/`solver_init.pt`
  (BC clone), `live_final.pt` (BC+PPO, current best ~30% main), `multi_160.pt`; `out/model_best.pth`
  (separate kaggle_mcts net, 30% vs main_v3_pure).
- **Memory updated:** `objective-is-cispo` (solver→AlphaZero, ExpRL→GRPO), `macmini-deploy`
  (Colima 16 GiB runs the engine).

## 10. Open items / next steps

1. **Implement Stage 1** (self-mined, win-rate-driven curriculum) and **prove win-rate vs
   `bare_agent` rises** — the gate before any automation.
2. Stop `train_sgs_b` (confirmed: `live_win 0.00`, the cold-start dead end).
3. Then **Stage 2**: conjecturer auto-gen + ExpRL/GRPO fine-tune + difficulty ramp.
4. If chasing real strength vs the scripts: GPU-scale self-play + bigger net + deck-matched
   training (the emulated-CPU ceiling is the binding constraint).
