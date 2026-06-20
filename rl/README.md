# `rl/` — Self-Guided Self-Play for cabt Pokémon TCG

Implements `docs/SGS_RL_PLAN.md`. The competition submission is a `(deck + agent)`
pair: the **deck** is pinned to Team Rocket's Honchkrow (limitlesstcg list 26267,
`data/decks/deck_solver_honchkrow.json`) and the **agent** is a learned pointer-net
Solver trained with SGS.

> The engine (`cg/libcg.so`) is Linux x86-64 only. Everything that imports `cg`
> runs inside the Docker image below. Pure-Python modules (`config`, `scenario`,
> `targets`, `guide`, `conjecturer/parametric`) import on the Mac host too.

## Module map (plan §6)

| File | Role | Status |
|---|---|---|
| `config.py` | central knobs; pins the Honchkrow 26267 solver deck | ✅ |
| `env.py` | `TCGEnv`: micro-step + masking + opponent-in-env + scenario loading | ✅ P0 |
| `scenario.py` | `ScenarioSpec` + edit-script → `search_begin` kwargs | ✅ P0 |
| `targets.py` | build fixed `D` from `data/loser/*.json` | ✅ P2 |
| `encode.py` | observation + per-option featurizer + scripted prior | ✅ |
| `policy.py` | pointer net + value head + annealed prior residual | ✅ P1 |
| `solver_objectives.py` | `reinforce_half` \| `ppo` \| `cispo` (+ degenerate fallback) | ✅ P3 |
| `sft.py` | offline behavioral cloning of scripted pilots | ✅ P1 |
| `train_solver.py` | inner net-RL loop + diagnostics | ✅ P1 |
| `conjecturer/parametric.py` | edit-script policy (CPU default) | ✅ P2 |
| `guide.py` | rule-based `R_guide` v1 | ✅ P2 |
| `outer_loop.py` | SGS Algorithm 1 driver | ✅ P2 |
| `eval.py` | held-out gauntlet (headline metric) | ✅ |
| `mcts.py` | inference-time `search_begin`/`search_step` PUCT | ✅ P1 (engine-validate) |
| `llm_agent.py` | LLM-driven policy (Claude picks moves live); agent-interface; SFT teacher | ✅ (research/eval/teacher) |
| `vec.py` | subprocess vec-env (1 battle/process) | ✅ |
| `league.py` | OPTIONAL PSRO archive (§3.6 toggle) | stub |
| `conjecturer/author.py` | OPTIONAL P4 LLM conjecturer (SmolLM2-1.7B; CISPO buffer logging; parametric fallback) | ✅ P4 (pod) |
| `conjecturer/cispo_train.py` | OPTIONAL P4 offline **CISPO** LoRA fine-tune on `R_synth` (not GRPO) | ✅ P4 (pod) |

> **P4 LLM conjecturer** is built but GPU-pod-only — see `docs/P4_LLM_CONJECTURER.md`.
> Objective is **CISPO across the whole stack** (Solver + conjecturer); `grpo_train.py`
> is a deprecated redirect. It degrades to the parametric conjecturer on a CPU host.

## Build & run (Docker)

```bash
# from repo root
docker build --platform=linux/amd64 -f rl/Dockerfile -t cabt-rl .

# P0 — smoke test (scenario primitive + env + target set)
docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app cabt-rl \
  python -m rl.smoke.smoke_test

# P1 — train the Solver vs a fixed anchor (bare_agent on the same deck)
docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app cabt-rl \
  python -m rl.sgs.train_solver

# P2 — SGS loop (fixed D + parametric conjecturer + rule guide)
docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app cabt-rl \
  python -m rl.sgs.outer_loop

# Eval a checkpoint on the held-out gauntlet
docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app cabt-rl \
  python -m rl.shared.eval.eval rl/runs/p1/solver_final.pt
```

### LLM-driven policy (Claude picks moves live)

`rl/llm_agent.py` is an alternative Solver backend driven by Claude per decision.
It needs `ANTHROPIC_API_KEY` (falls back to the scripted prior without one).
Defaults to `claude-opus-4-8`; override via `RL_LLM_MODEL` (e.g. `claude-haiku-4-5`
for cheap, fast play). It's a research / eval / **SFT-teacher** tool — NOT the
shipped agent (the competition forbids external API calls at inference).

```bash
# Claude vs the bare pilot, head-to-head
docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
  -e ANTHROPIC_API_KEY=sk-... --entrypoint python cabt-rl \
  runner.py --a rl.llm_agent --b agents.bare_agent -n 10

# Distill Claude into the shippable net (SFT teacher)
#   sft.collect_traces(deck, opp_deck, "rl.shared.agents.llm_agent", "agents.bare_agent", n_games=...)
```

`RL_LLM_ALL=1` consults the LLM on every selection (default: only MAIN decisions,
to bound cost); `RL_LLM_THINKING=1` enables adaptive thinking.

Key env overrides (see `config.py`): `RL_SOLVER_DECK` (`honchkrow`|`crustle`|…),
`RL_OBJECTIVE` (`reinforce_half`|`ppo`|`cispo`), `RL_K`, `RL_TAU`, `RL_MCTS=1`.

## Status & next steps

- **Validated by reasoning against the engine API** (`cg/api.py`, `cg/game.py`),
  not yet executed — the native lib needs the linux/amd64 image. Run `rl.smoke.smoke_test`
  first; it de-risks the `search_begin` scenario primitive (plan §5 P0).
- The biggest engine-contract item to confirm in Docker: search-mode opponent
  observations are reconstructed to a dict via `env._to_dict`; verify a scripted
  opponent pilots correctly inside `search_step`.
- Hidden-info prediction in `targets.py` is approximate (multiset subtraction);
  fine for a belief state but revisit if `search_begin` rejects scenarios.
