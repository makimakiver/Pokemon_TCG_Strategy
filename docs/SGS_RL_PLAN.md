# Self-Guided Self-Play RL for cabt Pokémon TCG — Design Plan

**Goal:** train the strongest possible competition submission = `(deck + agent)` for the
cabt Pokémon TCG engine, using **Self-Guided Self-Play (SGS)**: a learned **Solver net**
that plays the game, trained on a **fixed target set `D` of hard scenarios** that it
currently loses, with a **Conjecturer** that generates *simpler related scenarios*
(stepping-stones toward each target) and a **Guide** that keeps those scenarios clean and
non-degenerate.

Adapted from *Scaling Self-Play with Self-Guidance* (Bailey, Wen, Dong, Hashimoto, Ma,
arXiv:2604.20209).

---

## 0. Framing note (read first — SGS, not PSRO)

Earlier drafts of this plan described a PSRO/PFSP **league** (growing never-deleted
archive, opponent-pool sampling, open competitive strength). That is a different algorithm
from SGS and optimizes a different objective. **This plan is now SGS:**

| | SGS (this plan) | PSRO (old draft) |
|---|---|---|
| Target set | **Fixed `D`** of hard scenarios the Solver loses | Growing league archive |
| Conjecturer generates | **Simpler related scenarios** (lemmas) via mid-game edits | Opponent decks |
| Sampling | One synth per unsolved target; REINFORCE½ on `s ≤ 0.5` | PFSP (~50% win-rate) + meta-share |
| Headline metric | Cumulative solve-rate on `D` plateaus higher | Open Elo / win-rate |

SGS is the right fit when the goal is "crack these specific hard scenarios"; PSRO is the
right fit for "open competitive strength." The competition scores `(deck+agent)` head-to-head,
which leans PSRO — but the paper's headline result (7B beating 671B) is SGS on a fixed hard
set. **We adopt SGS** because: (a) the engine gives us the exact primitive SGS needs — a
**state-injection API** (`cg/api.py:517` `search_begin`) that loads arbitrary mid-game
positions, so a "lemma" = an edited losing position, not just a different deck; and (b)
`data/loser/lost_*.json` already ship serialized losing decision points → a free grounded `D`.

If late in training the fixed-`D` solve-rate saturates and we want open strength, we can
layer PFSP on top without changing the Solver — see §3.6 (optional).

---

## 1. Decision ledger (locked)

| Component | Decision |
|---|---|
| **Objective** | Strongest competition submission = `(deck + agent)` |
| **Target set `D`** | **Fixed**: losing decision points from `data/loser/lost_*.json` + self-play losses vs anchors, across the top-20 meta decks. A target is *solved* when Solver win-rate ≥ τ (0.8). |
| **Solver** | Small RL **net**: micro-step pointer policy + action masking; **behavioral-cloned from `main_v1`/`fire`/`honchkrow` traces (offline SFT) before RL**; LLM-written `option_prior` residual (annealed → 0). |
| **Solver reward** | Sparse ±1 (win/loss, draw=0) **primary** + small **annealed** LLM shaping; stall/max-step games = 0. |
| **Solver objective** | **PPO vs CISPO ablation** — identical reward + curriculum across arms; entropy + solve-rate-spread instrumented; **REINFORCE½ as the reference arm + CISPO-with-entropy-control rescue arm**. |
| **Conjecturer** | **Parametric policy over scenario edit-scripts (default)**, emitting edits loaded via `search_begin`. **LLM (7–8B, LoRA+GRPO) is an OPTIONAL P4 upgrade** — not on the critical path. |
| **Opponent pilot** | `bare_agent` (auto-roles) + tuned pilots (`honchkrow`, `fire`) as fixed anchors; checkpoint league at eval only. |
| **Guide** | **Rule-based v1** (legality × non-degeneracy × relevance) **with a defined LLM-Guide upgrade path** (semantic elegance judge mirroring paper §E.3). |
| **Deck pool** | Top-10 already built; build top-11–20 via `build_decks.py` + `battle_start` validation; **Beedrill AND Festival Lead are absent from the pool — excluded**, see §3.7. |
| **Eval** | Frozen **held-out meta gauntlet** (original unmutated decks, meta-share weighted), no leakage. |
| **Infra** | Single box + 1 GPU, Docker linux/amd64; N CPU env-workers + GPU learner. |
| **Solver deck** | **Candidate set pinned at P1** (not deferred to P5); dev-default = v3_pure Crustle+Typhlosion. |
| **Inference-time search** | `search_begin`/`search_step` **MCTS pulled forward to P1** (the engine's standout feature); off during training. |

---

## 2. Architecture

```
   ┌──────────────── SGS Algorithm 1 loop (GPU time-shared) ────────────────┐
   │                                                                         │
   │   1. Sample batch B ⊆ D (fixed target set)                              │
   │   2. Split B → B_solved / B_unsolved                                    │
   │   3. For each x ∈ B_unsolved: Conjecturer emits edit_script → x̃        │
   │        (loaded into the engine via search_begin; legality-gated)         │
   │   4. Solver plays k games on B ∪ B_synth; engine verifies → win/loss     │
   │   5. R_solve(x̃) = ind · (1 − s);  R_guide(x̃) = ρ(x, x̃)               │
   │        R_synth = R_solve · R_guide   (batch-normalized to [0,1])         │
   │   6. Update Solver πθ with PPO|CISPO|REINFORCE½ on v(y)                 │
   │   7. Update Conjecturer gϕ with REINFORCE on R_synth                     │
   │              ▲                                    │                      │
   │         Guide ρ (rule v1, LLM upgrade path)       Solver net = submission│
   └────────────────────────────────────────────────────────────────────────-┘
```

- **The GPU is never contended:** Solver net-RL and (optional) Conjecturer LLM-RL run in
  separate phases. The **parametric Conjecturer (default) needs no GPU** — it runs on CPU
  alongside the env workers, so P0–P3 are single-RL-stack.
- **Submission strength comes from the Solver net + inference-time MCTS.** The Conjecturer
  only improves the curriculum.
- **Guidance is asymmetric:** the Conjecturer improves the curriculum; the Solver never
  plays the Conjecturer. The in-game adversary is part of the scenario spec (anchor pilot).

---

## 3. Component specs

### 3.1 Environment wrapper (`rl/env.py`)
Single-seat self-play view of one cabt battle, grounded in `cg/game.py`
(`battle_start → battle_select → battle_finish`).

- **Micro-stepping:** the policy picks **one** option index per RL step; the env buffers
  picks and only calls `battle_select` once `minCount..maxCount` is satisfied (+ a `STOP`
  action for optional-length selects). Turns the variable-length subset selection into
  ordinary masked-discrete RL.
- **Scenario loading (NEW):** `reset(scenario)` calls `search_begin(...)` with the scenario's
  predicted deck/prize/hand/active args + its `search_begin_input` blob, so an episode can
  start from **any mid-game position**, not just turn-0. This is the primitive that makes
  scenario-based curriculum possible.
- **Opponent in-env:** `_advance()` plays the opponent (and skips non-decisions) until it's
  the Solver's turn, so the learner sees a clean single-agent MDP.
- **One battle per process** (`Battle.battle_ptr` is a global singleton, `cg/sim.py:67`);
  parallelism = subprocess vec-env workers, all inside the Docker linux/amd64 image. Always
  `battle_finish()` in `reset`/on error.
- Reward: terminal ±1 from `current.result`; 0 for stalls/`max_steps`.

### 3.2 Solver net (`rl/policy.py`, `rl/encode.py`)
- **Offline SFT bootstrap (NEW, before RL):** behavioral-clone the net on game traces from
  `main_v1` / `fire` / `honchkrow` so it starts at ~scripted strength. Stronger and more
  stable than an annealed prior alone.
- **Pointer network over options:** encode each `select.option` (its type; for `CARD`
  options the target card's runtime stats from `card_table`/`attack_table` — HP, weakness,
  energy cost, prize value, stage), score with `dot(query, option_embed)`, mask, softmax.
  Deck-agnostic (reads stats at runtime, like `bare_agent`).
- **`option_prior` residual:** LLM-authored heuristic prior distilled from `main_v1`'s
  scoring; added to policy logits (or KL-anchor). Prior weight **annealed → 0** so the net
  surpasses the heuristic.
- **Value head** (for PPO/GAE).

### 3.3 Solver objectives (`rl/solver_objectives.py`)
Swappable behind one interface; **run as sequential SGS runs** sharing identical frozen
reward + curriculum for a fair comparison.
- **REINFORCE½ (reference arm — paper's winner):** log-likelihood over winning rollouts on
  problems with win-rate ≤ 0.5. Cheapest, keeps entropy alive, feeds the Conjecturer.
- **PPO:** clipped surrogate + GAE (γ≈0.999, λ≈0.95) + value loss + entropy bonus.
- **CISPO:** critic-free, group-normalized advantage (G rollouts/scenario), clipped
  stop-grad IS weight × advantage × logπ; entropy bonus.
- **CISPO degenerate-group handling (NEW):** with hard targets (win-rate ≤ 0.5) many CISPO
  groups are all-loss or all-win → `std≈0` → group advantage undefined. **Filter those
  groups** (|std| < ε), and **fall back to REINFORCE½ on them** — this is precisely the
  regime the paper shows breaks vanilla CISPO. Without this, CISPO-vanilla just re-derives
  the paper's negative result.
- **Length-normalize** per-rollout loss by # decisions (long games ≠ more weight).
- **Collapse diagnostics (mandatory):** mean Solver entropy, histogram of per-scenario win
  rates (collapse = mass at 0/1), `R_synth`, KL-to-reference, grad-norm, value
  explained-variance (PPO).

### 3.4 Conjecturer (`rl/conjecturer/`) — scenario edit-scripts, not deck edits
- **Action space = edit-scripts over a target scenario.** Given a target `x` (a losing
  mid-game position from `D`), the Conjecturer emits a declarative list of edits that carve
  out a sub-skill, e.g.:
  - *reduce the threat:* "opponent's active is Quilava (Stage-1) instead of Typhlosion; 2 {R}"
  - *partial credit:* "pre-attach one extra {G} to our active Crustle"
  - *isolate a tactic:* "place a Boss's Orders in hand; opponent active is a gustable target"
  - *prune complexity:* "opponent bench reduced to 1 Pokémon"
- **Why edits, not from-scratch states / not deck edits:** a lemma in the paper isn't "a
  different theorem," it's "the same theorem with one assumption weakened." The TCG analogue
  is **the same matchup in an easier mid-game position that isolates a sub-skill** — not a
  different deck (which changes the *matchup*, not the *difficulty within a matchup*).
  Editing a *real* target state also keeps legality high and keeps `x̃` meaningfully
  "related to `x`" (the whole point of conditioning — paper's "No Problem Conditioning"
  ablation showed unconditional generation gains nothing).
- **Loading:** the edit-script is translated (deterministic code) into the
  predicted-hand/deck/active args passed to `search_begin` alongside the target's
  `search_begin_input` blob. The engine validates legality; illegal edits are rejected +
  resampled.
- **Default = parametric policy** over edit-script tokens (fast, CPU, no GPU). Optional
  upgrade = 7–8B LLM (LoRA + GRPO on `R_synth`) emitting a CoT strategy + structured
  edit-script (§5 P4, off the critical path).

### 3.5 Guide (`rl/guide.py`) — rule-based v1 + LLM upgrade path
`R_guide = legality × non_degeneracy × relevance × (·)`

**v1 (rule-based, default):**
- **legality:** run `cg.game.battle_start`, check `errorType` (4-copy-by-name, no-Basic,
  ACE-SPEC-over-limit). Hard gate.
- **non_degeneracy:** the edited state still has a real win-condition for both sides, enough
  Pokémon, isn't an auto-win by rule (opponent active present, prize count sane).
- **relevance:** edit-distance of the *edit-script* from the empty edit (budget bound) —
  rewards minimal, targeted edits over wholesale rewrites.
- Map product to [0,1].

**LLM upgrade path (paper §E.3, the decisive mechanism):** the rule-based Guide catches
*obvious* failures but **cannot catch the subtle "superficially related but inelegant /
useless" failure** — e.g. a legal, on-archetype edit that removes the exact skill the Solver
needs to learn. The paper needed a *semantic* judge for precisely this (Fig. 2: without the
LLM Guide, Conjecturer collapses to disjunction-spammed theorems). The TCG analogue is
collapse to trivial auto-win states. **Upgrade plan:**
- **Relevance (0–5):** is `x̃` on the critical path to mastering `x`? (0 unrelated/identical
  → 5 a faithful lemma).
- **Degeneracy (0–4)** — the "complexity" analogue, inverted: 0 realistic → 4 auto-win by
  construction.
- **Redundancy (0/1):** edits irrelevant to `x`'s difficulty.
- **Combine:** `R_guide = max(0, relevance + (2 − degeneracy) + (1 − redundancy))`,
  **forced to 0 if degeneracy ≥ 3** — mirroring the paper's complexity≥3 → 0 rule.
- Frozen (+ small SFT for output format, paper §4.1). Trigger the upgrade if the rule-based
  Guide shows collapse symptoms (auto-win rate rising, relevance-spread shrinking).

### 3.6 Pool / league (optional, layered on if `D` saturates)
If the fixed-`D` solve-rate plateaus and we want open strength, add PSRO on top **without
changing the Solver**: growing never-deleted archive of `{deck, pilot}`, PFSP
(prioritize ~50%-winrate opponents) × meta-share weighting. Anchors: `bare_agent`-piloted
imported decks + `honchkrow.py` / `fire.py`. This is a toggle, not the core algorithm.

### 3.7 Deck pool (`data/decks/`)
- **Top-10 already built** (Dragapult, N's Zoroark, Crustle, Slowking, Hydrapple, Alakazam,
  Raging Bolt, Ogerpon, Lillie's Clefairy, Honchkrow).
- **Build the top-11–20** via `tools/build_decks.py` + `battle_start` validation. **Pool
  validation is mandatory before commit** — two named archetypes are **absent from the cabt
  card pool and must be excluded/replaced:**
  - **Beedrill / Beedrill ex** — absent.
  - **Festival Lead** — absent (same class of error; verified).
  - Viable **present** alternatives: Mega Lucario ex, Team Rocket's Mewtwo ex, Hop's
    Trevenant, Ethan's Typhlosion, Cynthia's Garchomp ex, Metagross / Steven's Metagross ex,
    Mega Lopunny ex, Marnie's Grimmsnarl ex, Ceruledge ex, Gholdengo, Roaring Moon,
    Squawkabilly, Noctowl, Mega Gardevoir ex.
- Meta shares from limitlesstcg (TEF-CRI) drive sampling weights (only relevant if §3.6 is
  enabled).

---

## 4. Evaluation protocol

- **Held-out frozen gauntlet:** the **original unmutated** top-N meta decks, piloted by
  `bare_agent` + tuned anchors + frozen past Solvers. **Never** used for conjecturer training.
- **Headline metric:** meta-share-weighted win-rate vs the held-out gauntlet, plotted vs
  generations — this is the **PPO-vs-CISPO curve axis** and the **ship trigger**. Secondary
  SGS metric: cumulative solve-rate on `D`.
- **Leakage guard:** conjecturer may only *edit* target scenarios, never *reproduce* a
  held-out position; bar state-hash duplicates of the held-out set.
- **Secondary:** head-to-head Elo vs existing agents (`main`, `v3_pure`, `fire`, `honchkrow`)
  via the gauntlet runner — directly comparable to `docs/RESULTS.md`.
- **Ship when:** held-out win-rate beats current `main` and plateaus.

---

## 5. Phased build plan (de-risked order)

- **P0 — Env + smoke test.** `TCGEnv` (micro-step + masking + **scenario loading via
  `search_begin`**) in Docker; random-masked opponent; verify episodes reconcile with
  `runner.py` win-attribution; no `battle_ptr` leaks; confirm you can clone + edit a
  position from `data/loser/lost_*.json` and replay it to terminal. **This de-risks the
  scenario primitive that everything else depends on.**
- **P1 — Solver learns vs a FIXED anchor.** Pointer net + **offline SFT from scripted
  traces** + `option_prior` + features; **pin `SOLVER_DECK` candidate set here** (not P5);
  **enable inference-time `search_begin`/`search_step` MCTS**; train (REINFORCE½ or
  entropy-reg PPO) vs frozen `main_v1`; success = win-rate climbs past 50%, entropy stays
  alive.
- **P2 — SGS loop, FROZEN + PARAMETRIC conjecturer.** Build `D` from loss replays +
  gauntlet losses; seed pool (imported top-20 + anchors); parametric Conjecturer emitting
  edit-scripts; rule-based Guide; held-out gauntlet eval. De-risks all SGS plumbing
  **before any LLM-RL**.
- **P3 — PPO vs CISPO ablation.** Two sequential SGS runs, identical reward/curriculum;
  full collapse diagnostics; **CISPO degenerate-group fallback enabled**; pick the winning
  objective (+ CISPO-with-entropy-control arm, + REINFORCE½ reference).
- **P4 — OPTIONAL: LLM conjecturer + LLM Guide.** Only if the parametric Conjecturer /
  rule-based Guide plateaus. Swap to LoRA-GRPO 7–8B Conjecturer on `R_synth` and/or the LLM
  Guide (§3.5). Confirm curriculum improves held-out win-rate.
- **P5 — Submission polish.** Finalize `SOLVER_DECK`; tune inference-time MCTS budget;
  package `(deck + agent)`.

---

## 6. Proposed layout

```
rl/
  env.py            # TCGEnv: micro-step, masking, opponent-in-env, scenario loading (search_begin)
  scenario.py       # ScenarioSpec + edit_script → search_begin kwargs (legality-checked)   [NEW]
  targets.py        # build fixed D from data/loser/*.json + gauntlet + self-play losses    [NEW]
  encode.py         # obs + per-option featurizer (runtime card stats)
  policy.py         # pointer net + value head + option_prior residual
  solver_objectives.py   # reinforce_half | ppo | cispo (+ degenerate-group fallback) behind one interface
  vec.py            # subprocess vec-env (1 battle/process)
  league.py         # OPTIONAL PSRO archive + PFSP + meta-share sampling (§3.6 toggle)
  guide.py          # rule-based v1 + LLM-Guide upgrade path (§3.5)
  mcts.py           # inference-time search_begin/search_step MCTS                           [P1]
  conjecturer/
    parametric.py   # edit-script policy (default, CPU)                                     [NEW]
    author.py       # strategy→edit-script generation (LLM, P4 optional)
    grpo_train.py   # LoRA-GRPO fine-tune on R_synth (P4 optional)
  sft.py            # offline behavioral cloning of scripted traces                         [NEW]
  train_solver.py   # inner net-RL loop
  outer_loop.py     # SGS Algorithm 1 driver + eval
  eval.py           # held-out gauntlet (no leakage)
```

---

## 7. Risks & mitigations

- **CISPO entropy collapse** (paper §4.5) — instrument entropy; include entropy/KL-control
  arm + degenerate-group fallback; keep REINFORCE½ as the reference.
- **Conjecturer collapse to auto-wins** (the paper's Fig. 2 failure mode) — the rule-based
  Guide is a v1 only; the LLM-Guide upgrade path (§3.5) is the durable fix. Monitor auto-win
  rate as a collapse signal.
- **Two RL stacks on one GPU** (net-RL + LLM-RL) — *mitigated by default:* the parametric
  Conjecturer (P0–P3) needs no GPU. LLM-RL is optional P4 and PSRO-time-shares the GPU.
- **MCTS inference cost** — bound the search budget at eval; keep it off during training.
- **Curriculum overfit** — strict held-out gauntlet + state-hash leakage guard.
- **Mac can't run `libcg.so`** — all env workers run in the Docker linux/amd64 image.

---

## 8. Open items

- [ ] Build top-11–20 decks via `build_decks.py` + `battle_start`; **exclude Beedrill +
  Festival Lead** (absent from pool); validate every list resolves before commit.
- [ ] Pin the `SOLVER_DECK` candidate set at P1.
- [ ] Confirm GPU VRAM (gates the optional P4 7–8B conjecturer; assumed ≥24GB).
- [ ] Decide τ (target solved threshold; default 0.8) and k (rollouts/scenario; paper uses 8).

## 9. Defaults assumed (override anytime)

- Target set `D` = losing decision points (`data/loser/lost_*.json`) + gauntlet/self-play
  losses, across the top-20; fixed (not growing).
- Conjecturer = parametric edit-script policy (CPU) through P3; LLM is optional P4.
- Guide = rule-based v1, with the §3.5 LLM upgrade path available when collapse appears.
- Inference-time MCTS on at submission (P1), off in training.
- Offline SFT of the Solver before RL (P1).
- BR phase = train Solver to held-out plateau before each (optional) conjecturer-RL phase.
