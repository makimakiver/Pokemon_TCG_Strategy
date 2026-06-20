# NN+MCTS Solver Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the SGS Solver with an NN-guided MCTS actor on the SmolLM conjecturer's persisted problem set, using the existing CISPO objective with the MCTS visit-distribution as the behavior policy.

**Architecture:** A small pure-Python IO layer persists the conjecturer's proposed lemmas (`target_id` + edit-script) to a JSONL the engine isn't needed to read. The Docker solver loop loads that file as a warm-start, then trains `PointerPolicy` via MCTS-guided rollouts (MCTS visit distribution `π` becomes CISPO's behavior log-prob), with a CPU parametric conjecturer co-evolving for unsolved targets. New surface is deliberately thin: IO layer, one MCTS method, one env accessor, one `collect_rollouts` parameter, two `run_sgs` flags.

**Tech Stack:** Python 3, NumPy, PyTorch, the cabt engine (`cg`, Linux x86-64, Docker `linux/amd64` image `cabt-rl`), pytest.

## Global Constraints

- Everything that imports `cg` runs ONLY inside Docker `cabt-rl` (`--platform=linux/amd64`). Pure-Python modules (`config`, `scenario`, `targets`, `problem_set`, `guide`) import on any host.
- The engine is one-battle-per-process (`Battle.battle_ptr` singleton); `env.reset` tears down the prior battle. Never run two battles per process.
- CISPO is THE solver objective for this path (`RL_OBJECTIVE=cispo` / explicit default); do NOT substitute GRPO or an AlphaZero `(π,z)` distillation loss.
- The LLM conjecturer export MUST use the good SFT adapter `conjecturer_lora_sft` (88.5% valid), NOT the degraded `conjecturer_lora` that `CONFIG.conj_llm_lora_dir` defaults to.
- Edit-scripts may only use the four legal kinds: `set_opponent_active`, `stack_your_deck_top`, `set_opponent_hand`, `weaken_opponent_hand`. No board-state edits.
- Tests live in repo-root `tests/` and import via `from rl...`; run with `pytest` from the repo root.
- Conjecturer interface: `propose(target, rng) -> (edited_spec, EditScript, idx)`; `EditScript.ops` is `list[EditOp]` with `.kind: str, .card_id: int, .slot: int`.

---

### Task 1: Problem-set IO layer (`rl/problem_set.py`)

Pure-Python persistence of the conjecturer's proposed lemmas. No `cg` import.

**Files:**
- Create: `rl/problem_set.py`
- Test: `tests/rl/__init__.py` (empty), `tests/rl/test_problem_set.py`

**Interfaces:**
- Consumes: `rl.scenario.ScenarioSpec`, `rl.scenario.EditScript`, `rl.scenario.EditOp`.
- Produces:
  - `problem_row(target_id: str, source: str, backend: str, edits: EditScript) -> dict`
  - `write_problem_set(rows: list[dict], path) -> None`
  - `load_problem_set(path) -> dict[str, EditScript]`  (keyed by `target_id`)
  - `seed_scenarios(D: list[ScenarioSpec], ps: dict[str, EditScript]) -> list[ScenarioSpec]`

- [ ] **Step 1: Create the test package marker**

Create `tests/rl/__init__.py` as an empty file.

```python
```

- [ ] **Step 2: Write the failing test**

Create `tests/rl/test_problem_set.py`:

```python
import json

from rl.scenario import ScenarioSpec, EditScript, EditOp
from rl.problem_set import (
    problem_row, write_problem_set, load_problem_set, seed_scenarios,
)


def _spec(target_id, your_deck=(10, 11, 12), opp_hand=(20, 21)):
    # Minimal obs that satisfies ScenarioSpec.my_index / apply(); no engine needed.
    obs = {"current": {"yourIndex": 0,
                       "players": [{"prize": [1, 2], "deckCount": 0, "handCount": 2,
                                    "active": [{"id": 99}]},
                                   {"prize": [1, 2], "deckCount": 0, "handCount": 2,
                                    "active": [{"id": 88}]}]}}
    return ScenarioSpec(
        obs=obs, your_deck=list(your_deck), your_prize=[],
        opponent_deck=[], opponent_prize=[], opponent_hand=list(opp_hand),
        opponent_active=[], source="t.json#step1", target_id=target_id)


def test_problem_row_shape():
    es = EditScript(ops=[EditOp(kind="stack_your_deck_top", card_id=11, slot=0)], budget=4)
    row = problem_row("t_1", "t.json#step1", "parametric", es)
    assert row["target_id"] == "t_1"
    assert row["backend"] == "parametric"
    assert row["edits"] == [{"kind": "stack_your_deck_top", "card_id": 11, "slot": 0}]


def test_write_then_load_roundtrip(tmp_path):
    es = EditScript(ops=[EditOp(kind="weaken_opponent_hand", card_id=20)], budget=4)
    rows = [problem_row("t_1", "t.json#step1", "parametric", es)]
    p = tmp_path / "problems.jsonl"
    write_problem_set(rows, p)
    loaded = load_problem_set(p)
    assert set(loaded) == {"t_1"}
    assert [op.kind for op in loaded["t_1"].ops] == ["weaken_opponent_hand"]
    assert loaded["t_1"].ops[0].card_id == 20


def test_seed_scenarios_applies_edits_and_falls_back(tmp_path):
    D = [_spec("t_1"), _spec("t_missing")]
    ps = {"t_1": EditScript(ops=[EditOp(kind="stack_your_deck_top", card_id=12)], budget=4)}
    seeded = seed_scenarios(D, ps)
    # t_1: card 12 moved to deck top.
    s1 = next(s for s in seeded if s.target_id == "t_1")
    assert s1.your_deck[0] == 12
    # t_missing: no edit -> identity (unchanged order).
    s2 = next(s for s in seeded if s.target_id == "t_missing")
    assert s2.your_deck == [10, 11, 12]


def test_load_skips_malformed_rows(tmp_path):
    p = tmp_path / "bad.jsonl"
    with open(p, "w") as f:
        f.write("not json\n")
        f.write(json.dumps({"no_target_id": True}) + "\n")
        f.write(json.dumps({"target_id": "ok", "edits": []}) + "\n")
    loaded = load_problem_set(p)
    assert set(loaded) == {"ok"}
    assert loaded["ok"].ops == []
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/rl/test_problem_set.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.problem_set'`.

- [ ] **Step 4: Implement `rl/problem_set.py`**

```python
"""Persisted problem set: the conjecturer's proposed lemmas, engine-free IO.

A *problem* is a target position (recoverable from the fixed target set D by
``target_id``) plus an edit-script the Conjecturer proposed to weaken it. We
persist ONLY ``(target_id, source, backend, edits)`` — the heavy observation
blob is rebuilt from D at load time — so the file is small and parses on any
host (no ``cg`` dependency). See docs/.../2026-06-20-nn-mcts-solver-env-design.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from .scenario import ScenarioSpec, EditScript, EditOp


def problem_row(target_id: str, source: str, backend: str, edits: EditScript) -> dict:
    return {
        "target_id": target_id,
        "source": source,
        "backend": backend,
        "edits": [{"kind": op.kind, "card_id": op.card_id, "slot": op.slot}
                  for op in edits.ops],
    }


def write_problem_set(rows: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row_to_editscript(row: dict) -> EditScript | None:
    edits = row.get("edits")
    if not isinstance(edits, list):
        return None
    ops: list[EditOp] = []
    for e in edits:
        if not isinstance(e, dict) or "kind" not in e:
            continue
        try:
            ops.append(EditOp(kind=e["kind"],
                              card_id=int(e.get("card_id", 0)),
                              slot=int(e.get("slot", 0))))
        except (ValueError, TypeError):
            continue
    return EditScript(ops=ops)


def load_problem_set(path) -> dict[str, EditScript]:
    out: dict[str, EditScript] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = row.get("target_id")
            es = _row_to_editscript(row)
            if not tid or es is None:
                continue
            out[tid] = es
    return out


def seed_scenarios(D: list[ScenarioSpec], ps: dict[str, EditScript]) -> list[ScenarioSpec]:
    """Apply each target's proposed edit-script; identity (with a warning) when a
    target has no entry, so the loop still has a lemma for every target in D."""
    seeded: list[ScenarioSpec] = []
    missing = 0
    for target in D:
        es = ps.get(target.target_id)
        if es is None:
            missing += 1
            seeded.append(target)
        else:
            seeded.append(es.apply(target))
    if missing:
        print(f"[problem_set] {missing}/{len(D)} targets had no proposed lemma "
              f"(seeded as identity)")
    return seeded
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/rl/test_problem_set.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add rl/problem_set.py tests/rl/__init__.py tests/rl/test_problem_set.py
git commit -m "feat(rl): persisted problem-set IO layer (problem_set.py)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Problem-set exporter CLI (`rl/conjecturer/export_problems.py`)

Materializes the conjecturer's problem set into the JSONL Task 1 reads.

**Files:**
- Create: `rl/conjecturer/export_problems.py`
- Test: `tests/rl/test_export_problems.py`

**Interfaces:**
- Consumes: `rl.problem_set.problem_row/write_problem_set`, `rl.targets.build_target_set`, `rl.conjecturer.get_conjecturer`.
- Produces:
  - `export_problems(conjecturer, D, rng, backend: str) -> list[dict]`
  - `main(argv=None) -> int`  (CLI entry; `--backend`, `--out`, `--lora`, `--seed`)

- [ ] **Step 1: Write the failing test**

Create `tests/rl/test_export_problems.py`:

```python
import random

from rl.scenario import ScenarioSpec, EditScript, EditOp
from rl.conjecturer.export_problems import export_problems


class _StubConj:
    def propose(self, target, rng):
        es = EditScript(ops=[EditOp(kind="weaken_opponent_hand", card_id=20)])
        return es.apply(target), es, 0


def _spec(tid):
    obs = {"current": {"yourIndex": 0,
                       "players": [{"prize": [1, 2], "deckCount": 0, "handCount": 1,
                                    "active": [{"id": 99}]},
                                   {"prize": [1, 2], "deckCount": 0, "handCount": 1,
                                    "active": [{"id": 88}]}]}}
    return ScenarioSpec(obs=obs, your_deck=[10], your_prize=[], opponent_deck=[],
                        opponent_prize=[], opponent_hand=[20], opponent_active=[],
                        source=f"{tid}.json#step1", target_id=tid)


def test_export_produces_one_row_per_target():
    D = [_spec("t_1"), _spec("t_2")]
    rows = export_problems(_StubConj(), D, random.Random(0), backend="parametric")
    assert len(rows) == 2
    assert {r["target_id"] for r in rows} == {"t_1", "t_2"}
    assert rows[0]["backend"] == "parametric"
    assert rows[0]["edits"][0]["kind"] == "weaken_opponent_hand"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/rl/test_export_problems.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.conjecturer.export_problems'`.

- [ ] **Step 3: Implement `rl/conjecturer/export_problems.py`**

```python
"""Export the conjecturer's proposed problem set to a JSONL (plan Task 2).

Run the parametric backend anywhere (CPU, no engine needed — targets parse from
data/loser/*.json). Run the LLM backend on the GPU/MPS host (the Mac mini),
pointing --lora at the GOOD SFT adapter ``conjecturer_lora_sft`` (88.5% valid),
NOT the degraded default ``conjecturer_lora``.

Examples:
  # CPU / Docker fallback
  python -m rl.conjecturer.export_problems --backend parametric --out rl/runs/problems.jsonl
  # Mac mini (good adapter)
  python -m rl.conjecturer.export_problems --backend llm \
      --lora ~/Pokemon_TCG_Strategy/rl/runs/conjecturer_lora_sft \
      --out rl/runs/problems.jsonl
"""
from __future__ import annotations

import argparse
import os
import random
import sys

from ..problem_set import problem_row, write_problem_set


def export_problems(conjecturer, D, rng, backend: str) -> list[dict]:
    rows: list[dict] = []
    for target in D:
        _, edits, _ = conjecturer.propose(target, rng)
        rows.append(problem_row(target.target_id or target.source,
                                target.source, backend, edits))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export the conjecturer's problem set.")
    ap.add_argument("--backend", choices=["parametric", "llm"], default="parametric")
    ap.add_argument("--out", default="rl/runs/problems.jsonl")
    ap.add_argument("--lora", default=None,
                    help="LoRA adapter dir for --backend llm (use conjecturer_lora_sft).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # Point the LLM backend at the good adapter BEFORE constructing CONFIG-readers.
    if args.backend == "llm" and args.lora:
        os.environ["RL_CONJ_LORA"] = args.lora
    os.environ["RL_CONJ"] = args.backend

    from ..targets import build_target_set
    from . import get_conjecturer

    D = build_target_set()
    if not D:
        print("[export] empty target set D; check data/loser/*.json", file=sys.stderr)
        return 1
    conj = get_conjecturer(args.backend)
    if args.backend == "llm":
        adapter = os.environ.get("RL_CONJ_LORA", "(default conjecturer_lora — DEGRADED!)")
        print(f"[export] llm backend; adapter = {adapter}")
    rows = export_problems(conj, D, random.Random(args.seed), args.backend)
    write_problem_set(rows, args.out)
    print(f"[export] wrote {len(rows)} problems -> {args.out} (backend={args.backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/rl/test_export_problems.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add rl/conjecturer/export_problems.py tests/rl/test_export_problems.py
git commit -m "feat(rl): conjecturer problem-set exporter CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: MCTS visit-policy + env search-state accessor

Give MCTS a training-facing search that returns the visit distribution (CISPO's behavior policy), and let the env hand MCTS its live search node.

**Files:**
- Modify: `rl/mcts.py` (add `visit_policy` module function + `MCTS.search_policy`)
- Modify: `rl/env.py` (add `TCGEnv.search_state`)
- Test: `tests/rl/test_visit_policy.py`

**Interfaces:**
- Consumes: existing `rl.mcts._Node`, `MCTS._expand`, `MCTS._simulate`, `MCTS._priors`.
- Produces:
  - `rl.mcts.visit_policy(N: dict, n_options: int, temperature: float = 1.0) -> np.ndarray`
  - `MCTS.search_policy(root_state, temperature: float = 1.0, greedy: bool = False) -> tuple[int | None, np.ndarray, float]`
  - `TCGEnv.search_state() -> SimpleNamespace(searchId, observation)`

- [ ] **Step 1: Write the failing test** (pure-Python; no engine — exercises the normalization math)

Create `tests/rl/test_visit_policy.py`:

```python
import numpy as np

from rl.mcts import visit_policy


def test_visit_policy_normalizes_over_options():
    pi = visit_policy({0: 3, 1: 1}, n_options=2)
    assert np.allclose(pi, [0.75, 0.25])
    assert abs(pi.sum() - 1.0) < 1e-9


def test_visit_policy_ignores_out_of_range_edges():
    # A STOP edge (== n_options) must not appear in the option distribution.
    pi = visit_policy({0: 2, 2: 5}, n_options=2)
    assert np.allclose(pi, [1.0, 0.0])


def test_visit_policy_zero_visits_is_uniform_zero():
    pi = visit_policy({}, n_options=3)
    assert pi.shape == (3,)
    assert pi.sum() == 0.0


def test_visit_policy_temperature_sharpens():
    cold = visit_policy({0: 3, 1: 1}, n_options=2, temperature=0.5)
    # τ<1 sharpens toward the most-visited edge.
    assert cold[0] > 0.75
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/rl/test_visit_policy.py -v`
Expected: FAIL with `ImportError: cannot import name 'visit_policy'`.

- [ ] **Step 3: Add `visit_policy` and `search_policy` to `rl/mcts.py`**

Add this module-level function after the imports (after the `_result_value` function):

```python
def visit_policy(N: dict, n_options: int, temperature: float = 1.0) -> np.ndarray:
    """Normalized MCTS visit distribution over the root OPTION edges (length
    ``n_options``). Edges >= n_options (e.g. a STOP edge) are ignored. Returns an
    all-zero vector when there are no visits (caller treats that as forced-STOP).
    ``temperature`` < 1 sharpens toward the most-visited edge (counts ** 1/τ)."""
    pi = np.zeros(int(n_options), dtype=np.float64)
    for e, c in N.items():
        if 0 <= e < n_options:
            pi[e] = c
    if temperature != 1.0:
        pi = np.power(pi, 1.0 / max(1e-6, temperature))
    s = pi.sum()
    if s > 0:
        pi /= s
    return pi
```

Add this method to the `MCTS` class (right after `search`):

```python
    def search_policy(self, root_state, temperature: float = 1.0,
                      greedy: bool = False):
        """Training-facing search: return (action, pi, root_value).

        ``pi`` is the normalized visit distribution over the root option edges
        (CISPO's behavior policy). ``action`` is sampled from ``pi`` (or argmax
        when ``greedy``); ``None`` when the root has no option edges/visits
        (the caller then closes the select with STOP). ``root_value`` is the
        net's value-head estimate at the root (diagnostics)."""
        root = _Node(root_state.searchId, root_state.observation)
        self._expand(root)
        for _ in range(self.cfg.mcts_simulations):
            self._simulate(root)
        sel = root.obs.select
        n = len(sel.option) if sel else 0
        _, root_value = self._priors(root.obs) if n else (None, 0.0)
        pi = visit_policy(root.N, n, temperature)
        if n == 0 or pi.sum() <= 0.0:
            return None, pi, float(root_value)
        action = int(pi.argmax()) if greedy else int(np.random.choice(n, p=pi))
        return action, pi, float(root_value)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/rl/test_visit_policy.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Add `search_state()` to `rl/env.py`**

Add this method to `TCGEnv` (right after the `reset` method, before `close`):

```python
    def search_state(self):
        """The env's current engine search node, for an MCTS actor to root at.
        Valid only in search mode (mid-game scenarios loaded via search_begin)."""
        if self.mode != "search" or self._search_id is None:
            raise RuntimeError("search_state() is only valid in search mode")
        from types import SimpleNamespace
        return SimpleNamespace(searchId=self._search_id, observation=self._obs)
```

- [ ] **Step 6: Verify the env still imports cleanly (syntax)**

Run: `python -c "import ast; ast.parse(open('rl/env.py').read()); ast.parse(open('rl/mcts.py').read()); print('ok')"`
Expected: `ok` (engine import is NOT exercised here; behavioral check is the Docker smoke in Task 6).

- [ ] **Step 7: Commit**

```bash
git add rl/mcts.py rl/env.py tests/rl/test_visit_policy.py
git commit -m "feat(rl): MCTS search_policy (visit dist) + env.search_state accessor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Pluggable actor in `collect_rollouts` + actor factories

Let rollout collection take an `actor`; add the default (policy) and MCTS actors. CISPO consumes the recorded `logp` as the behavior log-prob unchanged.

**Files:**
- Modify: `rl/train_solver.py` (`collect_rollouts` signature + `make_policy_actor`, `make_mcts_actor`)
- Test: `tests/rl/test_actor_plumbing.py`

**Interfaces:**
- Consumes: `TCGEnv` (`reset`, `step`, `done`, `mode`, `search_state`), `PointerPolicy.act`, `MCTS.search_policy`, `rl.solver_objectives.Rollout`.
- Produces:
  - `actor(env, obs, prior_weight) -> tuple[int, float, float]`  (action, behavior logp, value)
  - `make_policy_actor(policy) -> actor`
  - `make_mcts_actor(mcts, policy) -> actor`
  - `collect_rollouts(policy, env, scenarios, k, prior_weight, actor=None) -> list[Rollout]`  (default actor wraps `policy.act`)

- [ ] **Step 1: Write the failing test** (uses fakes — no engine)

Create `tests/rl/test_actor_plumbing.py`:

```python
import numpy as np

from rl.train_solver import collect_rollouts, make_mcts_actor


class _FakeEnv:
    """Two-step search-mode episode; records actions; reward via _reward()."""
    mode = "search"

    def __init__(self):
        self._t = 0
        self.done = False
        self.actions = []

    def _obs(self):
        return {"global": np.zeros(2, np.float32),
                "options": np.zeros((2, 2), np.float32),
                "mask": np.array([1, 1, 1], np.float32), "n_options": 2}

    def reset(self, scenario=None):
        self._t = 0
        self.done = False
        self.actions = []
        return self._obs()

    def search_state(self):
        return object()

    def step(self, action):
        self.actions.append(action)
        self._t += 1
        self.done = self._t >= 2
        return self._obs(), 0.0, self.done, {}

    def _reward(self):
        return 1.0


class _FakeMCTS:
    def search_policy(self, root_state, temperature=1.0, greedy=False):
        # Deterministic: always pick option 1 with a 0.25/0.75 visit split.
        return 1, np.array([0.25, 0.75]), 0.5


class _Spec:
    target_id = "t_1"


def test_mcts_actor_records_visit_logp():
    env = _FakeEnv()
    actor = make_mcts_actor(_FakeMCTS(), policy=None)
    rollouts = collect_rollouts(policy=None, env=env, scenarios=[_Spec()],
                                k=1, prior_weight=0.0, actor=actor)
    assert len(rollouts) == 1
    r = rollouts[0]
    assert r.scenario_id == "t_1"
    assert r.win is True
    # behavior logp == log(pi[action]) == log(0.75)
    assert abs(r.steps[0]["logp"] - float(np.log(0.75))) < 1e-6
    assert env.actions == [1, 1]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/rl/test_actor_plumbing.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_mcts_actor'`.

- [ ] **Step 3: Edit `rl/train_solver.py`**

Add `import numpy as np` is already present. Add the actor factories above `collect_rollouts`:

```python
def make_policy_actor(policy):
    """Default actor: sample directly from the net (no search)."""
    def actor(env, obs, prior_weight):
        return policy.act(obs, prior_weight)
    return actor


def make_mcts_actor(mcts, policy):
    """MCTS actor: pick via MCTS in search mode; record log(pi_mcts[action]) as the
    behavior log-prob CISPO consumes. Falls back to the net in live (turn-0) mode."""
    def actor(env, obs, prior_weight):
        if getattr(env, "mode", None) != "search":
            return policy.act(obs, prior_weight)
        action, pi, value = mcts.search_policy(env.search_state())
        n = obs["n_options"]
        if action is None:                      # no option edge -> close with STOP
            return n, 0.0, float(value)
        logp = float(np.log(pi[action] + 1e-12))
        return int(action), logp, float(value)
    return actor
```

Then change `collect_rollouts` to accept and use `actor`:

```python
def collect_rollouts(policy, env: TCGEnv, scenarios, k: int, prior_weight: float,
                     actor=None):
    """k rollouts per scenario (None == a live turn-0 game). Returns list[Rollout].

    ``actor(env, obs, prior_weight) -> (action, logp, value)`` selects each move;
    defaults to sampling from ``policy`` directly. An MCTS actor records the
    MCTS visit-distribution log-prob as ``logp`` (CISPO's behavior policy)."""
    if actor is None:
        actor = make_policy_actor(policy)
    rollouts: list[Rollout] = []
    for sc in scenarios:
        sid = sc.target_id if sc is not None else "live"
        for _ in range(k):
            obs = env.reset(sc)
            steps = []
            guard = 0
            while not env.done and guard < CONFIG.max_steps:
                guard += 1
                action, logp, value = actor(env, obs, prior_weight)
                steps.append({"obs": obs, "action": action, "logp": float(logp),
                              "value": float(value)})
                obs, _, done, _ = env.step(action)
            reward = env._reward()
            rollouts.append(Rollout(scenario_id=sid, steps=steps,
                                    reward=reward, win=(reward > 0)))
    return rollouts
```

- [ ] **Step 4: Run the new test AND the existing-behavior guard**

Run: `pytest tests/rl/test_actor_plumbing.py -v`
Expected: PASS (1 passed).

Run: `python -c "import ast; ast.parse(open('rl/train_solver.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add rl/train_solver.py tests/rl/test_actor_plumbing.py
git commit -m "feat(rl): pluggable actor in collect_rollouts + MCTS/policy actor factories

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Wire MCTS + problem-set seed into `run_sgs`

Add the two flags and route rollouts through the MCTS actor; warm-start lemmas from the persisted problem set.

**Files:**
- Modify: `rl/outer_loop.py` (`_winrate`, `run_sgs`)
- Test: covered by the Docker smoke in Task 6 (engine-coupled); add a syntax/plumbing check here.

**Interfaces:**
- Consumes: `rl.problem_set.load_problem_set/seed_scenarios`, `rl.targets.build_target_set`, `rl.mcts.MCTS`, `rl.train_solver.collect_rollouts/make_mcts_actor`, `rl.solver_objectives.get_objective`.
- Produces:
  - `run_sgs(generations=20, batch_size=8, run_name="p2_sgs", use_mcts=False, problem_set=None, objective=None) -> (policy, history)`

- [ ] **Step 1: Edit `_winrate` to thread an actor**

In `rl/outer_loop.py`, replace the `_winrate` helper:

```python
def _winrate(policy, env, scenario, k, pw, actor=None):
    rs = collect_rollouts(policy, env, [scenario], k, pw, actor=actor)
    return np.mean([r.win for r in rs]), rs
```

- [ ] **Step 2: Edit `run_sgs` signature, imports, and setup**

Change the import line near the top of `outer_loop.py`:

```python
from .train_solver import collect_rollouts, _load_deck, make_mcts_actor
```

Add these imports with the other `rl` imports:

```python
from .mcts import MCTS
from .solver_objectives import get_objective
from .problem_set import load_problem_set, seed_scenarios
```

Replace the `run_sgs` signature and the lines that build `objective`, plus add seeding + MCTS-actor setup. The new head of `run_sgs` (down to the `for gen` loop) becomes:

```python
def run_sgs(generations: int = 20, batch_size: int = 8, run_name: str = "p2_sgs",
            use_mcts: bool = False, problem_set=None, objective: str | None = None):
    random.seed(CONFIG.seed); np.random.seed(CONFIG.seed); torch.manual_seed(CONFIG.seed)
    rng = random.Random(CONFIG.seed)

    D = build_target_set()
    if not D:
        raise RuntimeError("empty target set D; check data/loser/*.json")
    print(f"[sgs] target set |D| = {len(D)}")

    # Warm-start each target's current lemma from the conjecturer's problem set.
    seeded: dict[str, ScenarioSpec] = {}
    if problem_set:
        ps = load_problem_set(problem_set)
        for s in seed_scenarios(D, ps):
            seeded[s.target_id] = s
        print(f"[sgs] seeded {len(ps)} lemmas from {problem_set}")

    solver_deck = _load_deck(solver_deck_path())
    import importlib
    opponent = importlib.import_module(CONFIG.opponent_module)
    opp_deck = list(getattr(opponent, "my_deck", solver_deck))
    env = TCGEnv(solver_deck, opp_deck, opponent)

    policy = PointerPolicy()
    objective = get_objective(objective or ("cispo" if use_mcts else None))
    conjecturer = get_conjecturer()
    opt = torch.optim.Adam(policy.parameters(), lr=CONFIG.lr)

    actor = None
    if use_mcts:
        mcts = MCTS(policy, opponent, solver_seat=CONFIG.seat)
        actor = make_mcts_actor(mcts, policy)
        print(f"[sgs] MCTS actor ON ({CONFIG.mcts_simulations} sims/decision)")

    solve_rate = defaultdict(float)
    out = Path(RUNS_DIR) / run_name
    out.mkdir(parents=True, exist_ok=True)
    history = []
```

Add the `ScenarioSpec` import at the top with the other imports:

```python
from .scenario import ScenarioSpec
```

- [ ] **Step 3: Use the seeded lemma + actor inside the generation loop**

In the `for target in batch:` loop, replace the block that sets `scenario = target` and computes `wr`:

```python
        for target in batch:
            solved = solve_rate[target.target_id] >= CONFIG.tau
            scenario = seeded.get(target.target_id, target)   # warm-started lemma
            edits = None
            conj_idx = None
            if not solved:
                # Conjecture an easier lemma for an unsolved target (parametric in-loop).
                edited, edits, conj_idx = conjecturer.propose(target, rng)
                scenario = edited
            try:
                wr, rs = _winrate(policy, env, scenario, CONFIG.k_rollouts, pw, actor=actor)
                legal = True
            except Exception:
                wr, rs, legal = 0.0, [], False
```

(The rest of the loop body — `rollouts.extend`, `solve_rate[...]`, the `synth_updates` block, the solver/conjecturer updates, logging, and checkpointing — is unchanged.)

- [ ] **Step 4: Verify it parses and imports on a pure-Python host**

Run: `python -c "import ast; ast.parse(open('rl/outer_loop.py').read()); print('ok')"`
Expected: `ok`.

(Full import exercises `cg` via `env`/`mcts`; that is checked in the Docker smoke.)

- [ ] **Step 5: Commit**

```bash
git add rl/outer_loop.py
git commit -m "feat(rl): run_sgs use_mcts + problem_set seed (MCTS+CISPO solver loop)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Docker end-to-end smoke test

Prove the whole path runs inside the engine image: export → load → MCTS+CISPO SGS generations → checkpoint.

**Files:**
- Create: `rl/smoke_test_mcts_sgs.py`

**Interfaces:**
- Consumes: everything above, plus the Docker `cabt-rl` image and `data/loser/*.json`.
- Produces: a runnable module `python -m rl.smoke_test_mcts_sgs` that exits 0 on success.

- [ ] **Step 1: Implement the smoke module**

Create `rl/smoke_test_mcts_sgs.py`:

```python
"""End-to-end smoke for the NN+MCTS solver env (plan Task 6). Docker only.

  docker build --platform=linux/amd64 -f rl/Dockerfile -t cabt-rl .
  docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app cabt-rl \
      python -m rl.smoke_test_mcts_sgs

Verifies: parametric problem-set export -> load/seed -> 2 SGS generations with the
MCTS actor under CISPO -> a finite loss and a written checkpoint. Tiny knobs so it
finishes fast; correctness of training quality is out of scope for a smoke.
"""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

# Tiny, fast settings BEFORE importing CONFIG-consumers.
os.environ.setdefault("RL_MCTS_SIMS", "8")
os.environ.setdefault("RL_K", "2")
os.environ.setdefault("RL_OBJECTIVE", "cispo")

from .targets import build_target_set
from .problem_set import load_problem_set, seed_scenarios, write_problem_set
from .conjecturer import get_conjecturer
from .conjecturer.export_problems import export_problems
from .outer_loop import run_sgs


def main() -> int:
    D = build_target_set()
    assert D, "empty target set D; need data/loser/*.json in the image/mount"
    print(f"[smoke] |D| = {len(D)}")

    # 1) Export a parametric problem set, then load + seed it.
    rows = export_problems(get_conjecturer("parametric"), D, random.Random(0),
                           backend="parametric")
    tmp = Path(tempfile.mkdtemp()) / "problems.jsonl"
    write_problem_set(rows, tmp)
    ps = load_problem_set(tmp)
    seeded = seed_scenarios(D, ps)
    assert len(seeded) == len(D)
    assert len(ps) >= 1, "no problems exported"
    print(f"[smoke] exported+loaded {len(ps)} problems")

    # 2) Run 2 SGS generations with the MCTS actor under CISPO.
    policy, history = run_sgs(generations=2, batch_size=min(4, len(D)),
                              run_name="smoke_mcts_sgs", use_mcts=True,
                              problem_set=str(tmp), objective="cispo")
    assert len(history) == 2, history
    for rec in history:
        assert rec["loss"] == rec["loss"], "loss is NaN"   # NaN != NaN
    ckpt = Path("rl/runs/smoke_mcts_sgs/solver_final.pt")
    assert ckpt.exists(), f"missing checkpoint {ckpt}"
    print(f"[smoke] OK — 2 gens ran, checkpoint at {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Build the Docker image (if not already built)**

Run: `docker build --platform=linux/amd64 -f rl/Dockerfile -t cabt-rl .`
Expected: builds successfully (image `cabt-rl`).

- [ ] **Step 3: Run the smoke inside Docker**

Run:
```bash
docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app cabt-rl \
  python -m rl.smoke_test_mcts_sgs
```
Expected: ends with `[smoke] OK — 2 gens ran, checkpoint at rl/runs/smoke_mcts_sgs/solver_final.pt` and exit code 0. The per-gen `[sgs] gen ...` lines should print a finite `loss` and a non-empty `synth` count.

- [ ] **Step 4: Run the full pure-Python test suite (no engine) to confirm no regressions**

Run: `pytest tests/rl -v`
Expected: all tests from Tasks 1–4 PASS.

- [ ] **Step 5: Commit**

```bash
git add rl/smoke_test_mcts_sgs.py
git commit -m "test(rl): Docker end-to-end smoke for NN+MCTS solver env

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Runbook docs

Document the two-host workflow so the next operator can reproduce it.

**Files:**
- Modify: `rl/README.md` (add a "NN+MCTS solver over the conjecturer's problem set" section)

**Interfaces:** none (docs only).

- [ ] **Step 1: Append the runbook section to `rl/README.md`**

Add at the end of `rl/README.md`:

```markdown
## NN+MCTS solver over the conjecturer's problem set

Train the Solver with an MCTS actor (NN prior+value) on the SmolLM conjecturer's
proposed lemmas, under CISPO. See
`docs/superpowers/specs/2026-06-20-nn-mcts-solver-env-design.md`.

1. **Export the problem set from the SmolLM (Mac mini, MPS).** Use the GOOD SFT
   adapter — NOT the degraded `conjecturer_lora` default:

   ```bash
   python -m rl.conjecturer.export_problems --backend llm \
     --lora ~/Pokemon_TCG_Strategy/rl/runs/conjecturer_lora_sft \
     --out rl/runs/problems.jsonl
   ```

   (CPU/Docker fallback: `--backend parametric`.) Copy `rl/runs/problems.jsonl`
   to the laptop.

2. **Train the solver (laptop, Docker engine image).**

   ```bash
   docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app cabt-rl \
     python -c "from rl.outer_loop import run_sgs; \
       run_sgs(generations=200, use_mcts=True, \
               problem_set='rl/runs/problems.jsonl', objective='cispo')"
   ```

   The in-loop conjecturer is the CPU parametric one (co-evolves for unsolved
   targets); the SmolLM lemmas are the gen-0 warm-start. Checkpoints + per-gen
   solve-rate land in `rl/runs/p2_sgs/`.

3. **Smoke test the path first:** `python -m rl.smoke_test_mcts_sgs` (in Docker).
```

- [ ] **Step 2: Commit**

```bash
git add rl/README.md
git commit -m "docs(rl): runbook for NN+MCTS solver over conjecturer problem set

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (§ → task):**
- §4.1 `problem_set.py` → Task 1 ✓
- §4.2 `export_problems.py` (good-adapter default) → Task 2 ✓ (+ runbook Task 7)
- §4.3 `mcts.search_policy` → Task 3 ✓
- §4.4 `env.search_state` → Task 3 ✓
- §4.5 `collect_rollouts(actor=…)` + MCTS actor → Task 4 ✓
- §4.6 `run_sgs(use_mcts, problem_set, objective)` + parametric co-evolve → Task 5 ✓
- §4.7 unit + Docker smoke → Tasks 1–4 (unit) + Task 6 (smoke) ✓
- §5 data flow / two-host runbook → Task 7 ✓
- §6 risks: search-mode-only actor (Task 4 fallback), forced-STOP logp=0 (Task 4), wrong-adapter footgun (Task 2 warning + Task 7) ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test step shows the assertions.

**Type consistency:** `actor(env, obs, prior_weight) -> (int, float, float)` is consistent across Tasks 4–5; `search_policy(...) -> (int|None, np.ndarray, float)` consistent between Tasks 3 and 4; `seed_scenarios(D, ps)`/`load_problem_set` signatures consistent across Tasks 1, 5, 6; `problem_row`/`write_problem_set` consistent across Tasks 1, 2, 6; `EditScript.ops[].kind/card_id/slot` matches the verified source.

**Note for executor:** Tasks 1–4 are host-agnostic (pytest, no engine). Tasks 5–6 require the Docker `cabt-rl` image and `data/loser/*.json`. If the mini's `problems.jsonl` is not yet on the laptop, the smoke (Task 6) generates a parametric one itself, so the path is verifiable without the mini.
