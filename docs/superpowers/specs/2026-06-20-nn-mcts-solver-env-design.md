# NN+MCTS Solver Environment for the Conjecturer's Problem Set — Design

**Date:** 2026-06-20
**Status:** Approved direction; pending spec review → implementation plan
**Related:** `docs/SGS_CONJECTURER_SESSION_SUMMARY.md`, `docs/SGS_RL_PLAN.md`, `rl/` package

## 1. Goal

Train the SGS **Solver** with an **NN-guided MCTS** actor on the **problem set the
conjecturer (SmolLM) proposed**, updating the net with the project's **CISPO**
objective. The conjecturer's lemmas are produced once by the SmolLM on the Mac
mini, persisted to a file, then solved (and co-evolved against a CPU parametric
conjecturer) by the solver loop running in the Docker engine image on the laptop.

This realizes open-item #6 of the conjecturer session summary ("actually train the
Solver") with the NN+MCTS architecture, while reusing the existing, tested SGS
machinery rather than rewriting it.

### Decisions captured (from brainstorming)

1. **Deliverable:** train the solver (not just an eval harness).
2. **Problem source:** a **persisted problem-set file** (decouples the GPU/MPS
   conjecturer from the Docker solver).
3. **MCTS foundation:** **extend `rl/mcts.py`** (engine-search-tree PUCT with
   `PointerPolicy` prior + value), not `kaggle_mcts.py`.
4. **Train loss:** **CISPO with MCTS as the improved behavior/proposal policy** —
   keep the single stack objective; MCTS sharpens the rollouts.
5. **Loop scope:** **co-evolving** conjecturer, warm-started from the persisted set.
6. **Conjecturer split:** the **SmolLM (mini, good `conjecturer_lora_sft` adapter)
   seeds** the problem set; the **in-loop conjecturer during co-evolution is the
   CPU parametric one** (runs in Docker, no GPU contention, no cross-host calls).

## 2. Non-goals

- Re-running the LLM conjecturer inside the Docker solve loop (cross-host/GPU
  infeasible). Optional periodic LLM re-export is a future extension, not v1.
- Classic AlphaZero `(π, z)` distillation loss (explicitly rejected in favor of
  CISPO-with-MCTS-prior).
- Fixing MCTS's variable-length multi-pick **STOP-timing** semantics (see §6).
- Board-state edits (already out of scope in `scenario.py`).

## 3. Architecture

```
SmolLM on mini (MPS, conjecturer_lora_sft)
        │  rl/conjecturer/export_problems.py  --backend llm
        ▼
  rl/runs/problems.jsonl   (target_id + edit-script; small, engine-free)
        │  scp to laptop
        ▼  rl/problem_set.py: load_problem_set + seed_scenarios(D, ps)
  Docker engine image (linux/amd64)
        │  rl/outer_loop.run_sgs(use_mcts=True, problem_set=…, objective=cispo)
        │     • MCTS(PointerPolicy) actor plays k engine-verified rollouts / lemma
        │     • CISPO updates net with log(π_mcts) as behavior log-prob
        │     • parametric conjecturer re-proposes for still-unsolved targets
        ▼
  rl/runs/<run>/solver_*.pt + history.json
```

## 4. Components

### 4.1 `rl/problem_set.py` (new, pure-Python)
Persisted problem-set IO. No `cg` import → loads on any host.

- **Row schema (JSONL):**
  `{"target_id": str, "source": str, "backend": str, "edits": [{"kind","card_id","slot"}]}`.
  Stores only the **edit-script + provenance**. The heavy `obs` blob is rebuilt
  from `D` (via `targets.build_target_set`) keyed by `target_id`, so the file is
  small and engine-free to read.
- `load_problem_set(path) -> dict[str, EditScript]` — parse rows into
  `EditScript`s keyed by `target_id`; skip malformed rows (logged).
- `seed_scenarios(D, ps) -> list[ScenarioSpec]` — for each target in `D`, apply
  its `EditScript` (identity if missing/empty, with a `log` warning). Returns the
  warm-started lemma list.
- `write_problem_set(rows, path)` — used by the exporter.

### 4.2 `rl/conjecturer/export_problems.py` (new, thin CLI)
Materializes the conjecturer's problem set.

- `python -m rl.conjecturer.export_problems --backend {llm,parametric} --out rl/runs/problems.jsonl [--lora DIR]`
- Build `D = build_target_set()`; for each target call
  `get_conjecturer(backend).propose(target, rng)`; persist `{target_id, source,
  backend, edits}` via `write_problem_set`.
- **Must point the LLM backend at the good adapter:** default `--lora` to the SFT
  adapter (`conjecturer_lora_sft`), overriding `CONFIG.conj_llm_lora_dir` which
  defaults to the degraded `conjecturer_lora`. Sets `RL_CONJ_LORA` accordingly.
- Runs `llm` on the mini (MPS), `parametric` anywhere (CPU). CPU `parametric`
  export also serves as a Docker fallback when the mini file is unavailable.

### 4.3 `rl/mcts.py` (extend)
Add a training-facing search that returns CISPO's behavior policy.

- `search_policy(root_state, temperature=1.0) -> (action:int, pi:np.ndarray, root_value:float)`:
  run the existing PUCT sims from `root_state`, then
  - `pi` = visit counts `root.N` normalized over the **root option edges**
    (length = `n_options`);
  - `action` = sampled from `pi` (temperature-scaled) at train time, or
    `argmax` for submission;
  - `root_value` = the value-head estimate at the root (diagnostics).
- Keep the existing `search()` (argmax-only) for submission.
- MCTS selects among **option edges only**; STOP/variable-length closing is left
  to the env (see §6). No change to the existing tree/opponent logic.

### 4.4 `rl/env.py` (extend)
- `search_state() -> SimpleNamespace(searchId, observation)` returning
  `(self._search_id, self._obs)`; valid only in search mode (asserts otherwise).
  Lets the MCTS actor root itself at the env's live search node without the env
  leaking internals.

### 4.5 `rl/train_solver.py` (extend)
- Generalize `collect_rollouts(..., actor=None)`. Default `actor` = the current
  `policy.act` path (unchanged). An **MCTS actor** is a small callable that, per
  solver decision: reads `env.search_state()`, calls `mcts.search_policy`, and
  returns `(action, logp=log(pi[action]), value=root_value)`.
- The recorded `logp` is the **MCTS behavior log-prob** CISPO consumes as
  `s["logp"]`. Reward/grouping are unchanged → `CISPO.compute_loss` is reused
  verbatim. Forced-STOP (no legal option edge) records `logp=0.0`.

### 4.6 `rl/outer_loop.py` (extend)
- `run_sgs(generations=…, batch_size=…, run_name=…, use_mcts=False, problem_set=None, objective=None)`.
  - `objective` default for this path = `"cispo"` (per memory; overridable).
  - `problem_set`: warm-start `current_scenario[target_id]` via `seed_scenarios`;
    a target's first-seen lemma is the SmolLM one rather than the raw target.
  - `use_mcts`: build `MCTS(policy, opponent)` once; pass the MCTS actor into
    `collect_rollouts`.
  - Co-evolution (conjecturer.propose for unsolved, guide score, solve-rate τ,
    REINFORCE on `R_synth`) is the **existing loop, unchanged**; the in-loop
    conjecturer is `get_conjecturer("parametric")`.

### 4.7 Tests
- `rl/tests/test_problem_set.py` (new, pure-Python, no engine): build a tiny `D`
  (or a synthetic target stub), write → load → `seed_scenarios` round-trip;
  assert edit-scripts apply and missing targets fall back to identity.
- `rl/smoke_test_mcts_sgs.py` (new, Docker): export a tiny **parametric** problem
  set → load → `run_sgs(use_mcts=True, problem_set=…, generations=2,
  RL_MCTS_SIMS=8, RL_K=2)`. Assert: problems load; `pi` sums to ~1 and is
  legal-masked; rollouts collected; CISPO loss finite; checkpoint written.

## 5. Data flow (end to end)

1. **(mini)** `export_problems --backend llm --lora …/conjecturer_lora_sft`
   → `problems.jsonl`; `scp` to laptop `rl/runs/`.
2. **(Docker)** `run_sgs(use_mcts=True, problem_set="rl/runs/problems.jsonl")`:
   MCTS(net) plays k rollouts per lemma → engine verifies wins → CISPO updates
   net with `log π_mcts` behavior → parametric conjecturer re-proposes for
   unsolved → solve-rate tracked → checkpoints to `rl/runs/<run>/`.
3. Headline metric: held-out gauntlet (`eval.py`); secondary: cumulative
   solve-rate on `D`.

## 6. Risks & mitigations

- **MCTS multi-pick STOP timing (v1 limitation):** MCTS returns one option edge
  per call; the env's existing STOP logic closes variable-length selects. Exact
  for single-pick decisions (the TCG majority: attack/target/play-one-card);
  multi-pick uses the env heuristic. Documented, not silently dropped.
- **CISPO faithfulness:** importance weight `exp(net_logp − log π_mcts)` pulls the
  net toward MCTS-sharpened, group-advantage-weighted actions; degenerate-group
  REINFORCE½ fallback still applies. MCTS is off at `mcts_enabled=0` so the
  non-MCTS reference path in `outer_loop` is preserved for ablation.
- **Search-node validity:** engine search nodes are immutable/persistent (each
  `search_step` yields a new id), so MCTS exploring children does not invalidate
  the env's root `searchId` before the committed `env.step`.
- **Live mode:** MCTS is search-mode only; the problem set is entirely
  search-mode lemmas. If a `None` (live) scenario is passed, the actor falls back
  to `policy.act`.
- **Engine constraints:** one-battle-per-process preserved by `env.reset`;
  everything that imports `cg` runs only in Docker linux/amd64. Validation is the
  Docker smoke, matching the project's sim-harness convention.
- **Wrong adapter footgun:** `CONFIG.conj_llm_lora_dir` defaults to the degraded
  `conjecturer_lora`; the exporter must default `--lora` to `conjecturer_lora_sft`
  and log which adapter it loaded.

## 7. Reuse summary

New surface is intentionally small: the problem-set IO layer
(`problem_set.py`, `export_problems.py`), one MCTS method (`search_policy`), one
env accessor (`search_state`), one `collect_rollouts` parameter (`actor`), and two
`run_sgs` flags. `CISPO`, the guide, the parametric conjecturer, the curriculum,
and the SGS driver are reused intact.
