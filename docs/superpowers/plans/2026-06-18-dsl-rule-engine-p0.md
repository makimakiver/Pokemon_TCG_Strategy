# DSL Rule Engine (P0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the constrained JSON DSL, its validator, and an interpreter that compiles a rule-set into a sim agent — then prove (the P0 gate) that a hand-authored rule-set matches or beats `agents/bare_agent` on the meta gauntlet, *before* any LLM/GPU work.

**Architecture:** A rule-set is JSON (`condition → weight`). `grammar.py` validates it (the legality gate). `predicates.py` is pure Python — small dataclasses (`RuleContext`, `OptInfo`, `CardInfo`, `AttackInfo`) plus a predicate registry and the scorer — with **no `cg` import**, so it unit-tests on the Mac host. `interpret.py` (engine side) builds those dataclasses from a live observation via `rl/encode.py` and compiles a rule-set into an `agent(obs_dict)->list[int]`. `rl/dsl_agent.py` loads a rule-set from a path and exposes the harness agent interface.

**Tech Stack:** Python 3.9+ (host) / 3.11 (Docker engine image), `pytest` for unit tests, the existing `cg` engine + `runner.py` for integration, no new runtime deps (hand-rolled validation, stdlib `json`).

## Global Constraints

- `predicates.py` and `grammar.py` MUST NOT import `cg` (or `rl.encode`, which imports `cg`) — they must import on the Mac host. Only `interpret.py` and `dsl_agent.py` may import `cg`/`rl.encode`.
- The engine native lib is Linux x86-64 only; all integration runs go through Docker `cabt-rl` (`docker build --platform=linux/amd64 -f rl/Dockerfile -t cabt-rl .`).
- DSL limits (the legality gate, copied from spec §4.3): `rules` ≤ 40; each rule `when` is 1–6 predicates; `weight` ∈ `[-100000, 100000]`; `default_weight` numeric; `pred` must be in the registry; integer `n` arg within `[0, 1000]` when required.
- Agent contract: `agent(obs_dict) -> list[int]`; when `obs["select"] is None`, return the 60-card deck; otherwise return option indices honoring `select.minCount`/`maxCount` with no duplicates (same as `agents/bare_agent.py`).
- The repo is **not** a git repo yet. Run `git init` once before Task 1 if you want the commit cadence below; otherwise skip the commit steps.

---

### Task 1: DSL grammar + validator

**Files:**
- Create: `rl/dsl/__init__.py` (empty)
- Create: `rl/dsl/grammar.py`
- Test: `tests/dsl/test_grammar.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PREDICATE_ARITY: dict[str, int]` — predicate name → 0 or 1 (whether it needs `n`).
  - `validate(ruleset: dict) -> tuple[bool, list[str]]` — `(ok, errors)`; `ok` is True only when `errors` is empty.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsl/test_grammar.py
from rl.dsl.grammar import validate, PREDICATE_ARITY

GOOD = {
    "name": "x", "default_weight": 0.0,
    "rules": [
        {"when": [{"pred": "is_attack"}, {"pred": "attack_is_lethal"}], "weight": 800},
        {"when": [{"pred": "target_hp_ge", "n": 100}], "weight": 50},
    ],
}

def test_valid_ruleset_passes():
    ok, errors = validate(GOOD)
    assert ok and errors == []

def test_unknown_predicate_rejected():
    bad = {"rules": [{"when": [{"pred": "nope"}], "weight": 1}]}
    ok, errors = validate(bad)
    assert not ok and any("nope" in e for e in errors)

def test_missing_required_n_rejected():
    bad = {"rules": [{"when": [{"pred": "target_hp_ge"}], "weight": 1}]}
    ok, errors = validate(bad)
    assert not ok and any("target_hp_ge" in e for e in errors)

def test_weight_out_of_range_rejected():
    bad = {"rules": [{"when": [{"pred": "is_end"}], "weight": 10**9}]}
    ok, errors = validate(bad)
    assert not ok

def test_too_many_rules_rejected():
    bad = {"rules": [{"when": [{"pred": "is_end"}], "weight": 1}] * 41}
    ok, errors = validate(bad)
    assert not ok

def test_bool_weight_rejected():
    # Python bool is an int subclass; reject it explicitly for parity with `n`.
    bad = {"rules": [{"when": [{"pred": "is_end"}], "weight": True}]}
    ok, errors = validate(bad)
    assert not ok and any("weight" in e for e in errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/dsl/test_grammar.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.dsl.grammar'`

- [ ] **Step 3: Write minimal implementation**

```python
# rl/dsl/grammar.py
"""Constrained-DSL validation (the legality gate). Pure Python, NO cg import."""
from __future__ import annotations

# Predicate name -> arity (0 = no arg, 1 = needs integer "n"). Mirrors spec §4.2.
PREDICATE_ARITY: dict[str, int] = {
    # option type
    "is_attack": 0, "is_play_pokemon": 0, "is_play_item": 0, "is_play_supporter": 0,
    "is_play_stadium": 0, "is_attach_energy": 0, "is_attach_tool": 0, "is_evolve": 0,
    "is_ability": 0, "is_retreat": 0, "is_end": 0, "is_card_select": 0,
    "is_yes": 0, "is_no": 0, "is_number": 0,
    # target card
    "target_is_pokemon": 0, "target_is_basic": 0, "target_is_stage1": 0,
    "target_is_stage2": 0, "target_is_ex": 0, "target_is_energy": 0,
    "target_is_item": 0, "target_is_supporter": 0, "target_is_tool": 0,
    "target_is_attacker": 0, "target_is_mine": 0, "target_is_opponent": 0,
    "target_hp_ge": 1, "target_energy_ge": 1,
    # attack
    "attack_is_lethal": 0, "attack_hits_weakness": 0, "attack_damage_ge": 1,
    "attack_affordable": 0,
    # board / global
    "my_prizes_le": 1, "opp_prizes_le": 1, "opp_active_hp_le": 1,
    "my_active_at_risk": 0, "bench_has_attacker": 0, "hand_has_energy": 0,
    "energy_unused": 0, "supporter_unused": 0, "bench_count_ge": 1,
    # context
    "ctx_setup": 0, "ctx_main": 0, "ctx_to_hand": 0, "ctx_attach_from": 0,
    "ctx_discard": 0, "ctx_switch": 0, "ctx_choose_count": 0,
}

MAX_RULES = 40
MAX_WHEN = 6
WEIGHT_MIN, WEIGHT_MAX = -100000, 100000
N_MIN, N_MAX = 0, 1000


def validate(ruleset: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(ruleset, dict):
        return False, ["ruleset must be a JSON object"]
    if "default_weight" in ruleset and not isinstance(ruleset["default_weight"], (int, float)):
        errors.append("default_weight must be a number")
    rules = ruleset.get("rules")
    if not isinstance(rules, list):
        return False, ["ruleset.rules must be a list"]
    if len(rules) > MAX_RULES:
        errors.append(f"too many rules: {len(rules)} > {MAX_RULES}")
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"rule[{i}] must be an object"); continue
        w = rule.get("weight")
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            errors.append(f"rule[{i}].weight must be a number")
        elif not (WEIGHT_MIN <= w <= WEIGHT_MAX):
            errors.append(f"rule[{i}].weight {w} out of range")
        when = rule.get("when")
        if not isinstance(when, list) or not (1 <= len(when) <= MAX_WHEN):
            errors.append(f"rule[{i}].when must be a list of 1..{MAX_WHEN} predicates"); continue
        for j, p in enumerate(when):
            if not isinstance(p, dict) or "pred" not in p:
                errors.append(f"rule[{i}].when[{j}] must have a 'pred'"); continue
            name = p["pred"]
            if name not in PREDICATE_ARITY:
                errors.append(f"rule[{i}].when[{j}] unknown predicate '{name}'"); continue
            if PREDICATE_ARITY[name] == 1:
                n = p.get("n")
                if not isinstance(n, int) or isinstance(n, bool) or not (N_MIN <= n <= N_MAX):
                    errors.append(f"rule[{i}].when[{j}] '{name}' needs integer n in [{N_MIN},{N_MAX}]")
            elif "n" in p:
                errors.append(f"rule[{i}].when[{j}] '{name}' takes no n")
    return (len(errors) == 0), errors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/dsl/test_grammar.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add rl/dsl/__init__.py rl/dsl/grammar.py tests/dsl/test_grammar.py
git commit -m "feat(dsl): rule-set grammar + validator (legality gate)"
```

---

### Task 2: Predicates + scorer (pure, host-testable)

**Files:**
- Create: `rl/dsl/predicates.py`
- Test: `tests/dsl/test_predicates.py`

**Interfaces:**
- Consumes: `rl.dsl.grammar.PREDICATE_ARITY`.
- Produces:
  - Dataclasses `CardInfo`, `AttackInfo`, `OptInfo`, `RuleContext` (field names below — `interpret.py` builds these).
  - `PREDICATES: dict[str, callable]` — name → `fn(ctx, opt)` (arity 0) or `fn(ctx, opt, n)` (arity 1).
  - `rule_matches(rule: dict, ctx: RuleContext, opt: OptInfo) -> bool`.
  - `score_options(ruleset: dict, ctx: RuleContext, opts: list[OptInfo]) -> list[float]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsl/test_predicates.py
from rl.dsl.predicates import (
    CardInfo, AttackInfo, OptInfo, RuleContext, PREDICATES, score_options,
)

def ctx(**kw):
    base = dict(my_prizes=6, opp_prizes=6, opp_active_hp=120, energy_unused=True,
                supporter_unused=True, bench_count=2, bench_has_attacker=True,
                hand_has_energy=True, my_active_at_risk=False, context="main")
    base.update(kw)
    return RuleContext(**base)

def opt(kind="end", is_mine=True, target=None, attack=None):
    return OptInfo(kind=kind, is_mine=is_mine, target=target, attack=attack)

def test_is_attack_predicate():
    assert PREDICATES["is_attack"](ctx(), opt(kind="attack"))
    assert not PREDICATES["is_attack"](ctx(), opt(kind="end"))

def test_attack_lethal_and_param_predicates():
    a = AttackInfo(damage=120, lethal=True, hits_weakness=False, affordable=True)
    o = opt(kind="attack", attack=a)
    assert PREDICATES["attack_is_lethal"](ctx(), o)
    assert PREDICATES["attack_damage_ge"](ctx(), o, 100)
    assert not PREDICATES["attack_damage_ge"](ctx(), o, 130)

def test_target_and_board_predicates():
    c = CardInfo(is_pokemon=True, is_ex=True, is_attacker=True, hp=180, energy_count=2)
    o = opt(kind="attach_energy", target=c)
    assert PREDICATES["target_is_ex"](ctx(), o)
    assert PREDICATES["target_hp_ge"](ctx(), o, 150)
    assert PREDICATES["my_prizes_le"](ctx(my_prizes=1), o, 2)
    assert not PREDICATES["my_prizes_le"](ctx(my_prizes=3), o, 2)

def test_score_options_sums_matching_rules():
    rs = {"default_weight": 1.0, "rules": [
        {"when": [{"pred": "is_attack"}], "weight": 10},
        {"when": [{"pred": "is_attack"}, {"pred": "attack_is_lethal"}], "weight": 100},
    ]}
    a = AttackInfo(damage=200, lethal=True, hits_weakness=False, affordable=True)
    opts = [opt(kind="attack", attack=a), opt(kind="end")]
    scores = score_options(rs, ctx(), opts)
    assert scores[0] == 1.0 + 10 + 100   # attack+lethal
    assert scores[1] == 1.0              # end: only default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/dsl/test_predicates.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.dsl.predicates'`

- [ ] **Step 3: Write minimal implementation**

```python
# rl/dsl/predicates.py
"""Pure predicate registry + scorer over prebuilt dataclasses. NO cg import.

interpret.py (engine side) builds the dataclasses from a live observation; tests
build them directly. Each predicate is a pure fn of (RuleContext, OptInfo[, n]).
"""
from __future__ import annotations

from dataclasses import dataclass

from .grammar import PREDICATE_ARITY


@dataclass
class CardInfo:
    is_pokemon: bool = False
    is_basic: bool = False
    is_stage1: bool = False
    is_stage2: bool = False
    is_ex: bool = False
    is_energy: bool = False
    is_item: bool = False
    is_supporter: bool = False
    is_tool: bool = False
    is_attacker: bool = False
    hp: int = 0
    energy_count: int = 0


@dataclass
class AttackInfo:
    damage: int = 0
    lethal: bool = False
    hits_weakness: bool = False
    affordable: bool = False


@dataclass
class OptInfo:
    kind: str = "other"              # attack|play_pokemon|play_item|play_supporter|
                                     # play_stadium|attach_energy|attach_tool|evolve|
                                     # ability|retreat|end|card_select|yes|no|number|other
    is_mine: bool = True
    target: CardInfo | None = None
    attack: AttackInfo | None = None


@dataclass
class RuleContext:
    my_prizes: int = 6
    opp_prizes: int = 6
    opp_active_hp: int = 999          # large sentinel when no opp active
    energy_unused: bool = True
    supporter_unused: bool = True
    bench_count: int = 0
    bench_has_attacker: bool = False
    hand_has_energy: bool = False
    my_active_at_risk: bool = False
    context: str = "main"            # setup|main|to_hand|attach_from|discard|switch|choose_count|other


def _t(opt):  # target or empty CardInfo
    return opt.target if opt.target is not None else CardInfo()


def _a(opt):  # attack or empty AttackInfo
    return opt.attack if opt.attack is not None else AttackInfo()


PREDICATES = {
    # option type
    "is_attack": lambda c, o: o.kind == "attack",
    "is_play_pokemon": lambda c, o: o.kind == "play_pokemon",
    "is_play_item": lambda c, o: o.kind == "play_item",
    "is_play_supporter": lambda c, o: o.kind == "play_supporter",
    "is_play_stadium": lambda c, o: o.kind == "play_stadium",
    "is_attach_energy": lambda c, o: o.kind == "attach_energy",
    "is_attach_tool": lambda c, o: o.kind == "attach_tool",
    "is_evolve": lambda c, o: o.kind == "evolve",
    "is_ability": lambda c, o: o.kind == "ability",
    "is_retreat": lambda c, o: o.kind == "retreat",
    "is_end": lambda c, o: o.kind == "end",
    "is_card_select": lambda c, o: o.kind == "card_select",
    "is_yes": lambda c, o: o.kind == "yes",
    "is_no": lambda c, o: o.kind == "no",
    "is_number": lambda c, o: o.kind == "number",
    # target card
    "target_is_pokemon": lambda c, o: _t(o).is_pokemon,
    "target_is_basic": lambda c, o: _t(o).is_basic,
    "target_is_stage1": lambda c, o: _t(o).is_stage1,
    "target_is_stage2": lambda c, o: _t(o).is_stage2,
    "target_is_ex": lambda c, o: _t(o).is_ex,
    "target_is_energy": lambda c, o: _t(o).is_energy,
    "target_is_item": lambda c, o: _t(o).is_item,
    "target_is_supporter": lambda c, o: _t(o).is_supporter,
    "target_is_tool": lambda c, o: _t(o).is_tool,
    "target_is_attacker": lambda c, o: _t(o).is_attacker,
    "target_is_mine": lambda c, o: o.is_mine,
    "target_is_opponent": lambda c, o: not o.is_mine,
    "target_hp_ge": lambda c, o, n: _t(o).hp >= n,
    "target_energy_ge": lambda c, o, n: _t(o).energy_count >= n,
    # attack
    "attack_is_lethal": lambda c, o: _a(o).lethal,
    "attack_hits_weakness": lambda c, o: _a(o).hits_weakness,
    "attack_damage_ge": lambda c, o, n: _a(o).damage >= n,
    "attack_affordable": lambda c, o: _a(o).affordable,
    # board / global
    "my_prizes_le": lambda c, o, n: c.my_prizes <= n,
    "opp_prizes_le": lambda c, o, n: c.opp_prizes <= n,
    "opp_active_hp_le": lambda c, o, n: c.opp_active_hp <= n,
    "my_active_at_risk": lambda c, o: c.my_active_at_risk,
    "bench_has_attacker": lambda c, o: c.bench_has_attacker,
    "hand_has_energy": lambda c, o: c.hand_has_energy,
    "energy_unused": lambda c, o: c.energy_unused,
    "supporter_unused": lambda c, o: c.supporter_unused,
    "bench_count_ge": lambda c, o, n: c.bench_count >= n,
    # context
    "ctx_setup": lambda c, o: c.context == "setup",
    "ctx_main": lambda c, o: c.context == "main",
    "ctx_to_hand": lambda c, o: c.context == "to_hand",
    "ctx_attach_from": lambda c, o: c.context == "attach_from",
    "ctx_discard": lambda c, o: c.context == "discard",
    "ctx_switch": lambda c, o: c.context == "switch",
    "ctx_choose_count": lambda c, o: c.context == "choose_count",
}
assert set(PREDICATES) == set(PREDICATE_ARITY), "predicate registry / arity mismatch"


def _eval(pred: dict, ctx: RuleContext, opt: OptInfo) -> bool:
    name = pred["pred"]
    fn = PREDICATES[name]
    try:
        return bool(fn(ctx, opt, pred["n"]) if PREDICATE_ARITY[name] == 1 else fn(ctx, opt))
    except Exception:
        return False


def rule_matches(rule: dict, ctx: RuleContext, opt: OptInfo) -> bool:
    return all(_eval(p, ctx, opt) for p in rule["when"])


def score_options(ruleset: dict, ctx: RuleContext, opts: list[OptInfo]) -> list[float]:
    base = float(ruleset.get("default_weight", 0.0))
    rules = ruleset.get("rules", [])
    scores = []
    for opt in opts:
        s = base
        for rule in rules:
            if rule_matches(rule, ctx, opt):
                s += float(rule["weight"])
        scores.append(s)
    return scores
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/dsl/test_predicates.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add rl/dsl/predicates.py tests/dsl/test_predicates.py
git commit -m "feat(dsl): pure predicate registry + option scorer"
```

---

### Task 3: Interpreter + shippable agent (engine side)

**Files:**
- Create: `rl/dsl/interpret.py`
- Create: `rl/dsl_agent.py`
- Test: `tests/dsl/test_interpret_build.py` (host-safe parts only)

**Interfaces:**
- Consumes: `rl.dsl.predicates` (dataclasses + `score_options`), `rl.dsl.grammar.validate`, `rl.encode` (card/attack tables, `get_card`, `_card_of_option`), `cg.api`.
- Produces:
  - `build_context(obs) -> RuleContext`
  - `build_optinfo(obs, option) -> OptInfo`
  - `compile(ruleset: dict) -> callable` returning `agent(obs_dict)->list[int]`.
  - `rl/dsl_agent.py`: module-level `my_deck` + `agent(obs_dict)`, reading the rule-set path from `RL_RULESET` (default `rl/dsl/examples/handcrafted.json`).

- [ ] **Step 1: Write the failing test (host-safe: scorer→picks logic only)**

```python
# tests/dsl/test_interpret_build.py
# Host-safe: verifies the score->picks selection logic without importing cg.
from rl.dsl.predicates import OptInfo, RuleContext, AttackInfo, score_options
from rl.dsl.interpret import pick_from_scores

def test_pick_from_scores_respects_min_max():
    scores = [5.0, 1.0, 9.0, 3.0]
    # maxCount 1 -> single best index
    assert pick_from_scores(scores, min_count=1, max_count=1) == [2]
    # maxCount 2 -> top two by score, descending
    assert pick_from_scores(scores, min_count=1, max_count=2) == [2, 0]
    # minCount 3 -> exactly 3 even if it wants fewer
    assert pick_from_scores(scores, min_count=3, max_count=4) == [2, 0, 3]

def test_pick_no_options():
    assert pick_from_scores([], min_count=0, max_count=0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/dsl/test_interpret_build.py -q`
Expected: FAIL with `ImportError: cannot import name 'pick_from_scores'` (or ModuleNotFound)

> Note: `tests/dsl/test_interpret_build.py` imports `rl.dsl.interpret`, which imports `cg`. To keep this test host-runnable, `pick_from_scores` lives in a cg-free helper. Put it in `predicates.py` and re-export from `interpret.py`. Adjust the import in Step 1 to `from rl.dsl.predicates import pick_from_scores` and add the function to `predicates.py` instead. (See Step 3.)

- [ ] **Step 3: Write minimal implementation**

First add the cg-free selector to `predicates.py`:

```python
# append to rl/dsl/predicates.py
def pick_from_scores(scores: list[float], min_count: int, max_count: int) -> list[int]:
    """Indices chosen by descending score, honoring the engine's min/max."""
    n = len(scores)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    k = max(min_count, min(max_count, n))
    picks = order[:k]
    if len(picks) < min_count:
        picks = order[:min_count]
    return picks
```

Update the test import to `from rl.dsl.predicates import pick_from_scores` (it is re-exported by `interpret.py` for engine callers).

Then the engine-side interpreter:

```python
# rl/dsl/interpret.py
"""Compile a rule-set into a sim agent. Engine side (imports cg via rl.encode)."""
from __future__ import annotations

import json

from cg.api import (
    AreaType, CardType, OptionType, Pokemon, SelectContext, to_observation_class,
)
from .. import encode
from .grammar import validate
from .predicates import (
    CardInfo, AttackInfo, OptInfo, RuleContext, score_options, pick_from_scores,
)

_CTX_MAP = {
    int(SelectContext.MAIN): "main",
    int(SelectContext.SETUP_ACTIVE_POKEMON): "setup",
    int(SelectContext.SETUP_BENCH_POKEMON): "setup",
    int(SelectContext.TO_HAND): "to_hand",
    int(SelectContext.ATTACH_FROM): "attach_from",
    int(SelectContext.DISCARD): "discard",
    int(SelectContext.SWITCH): "switch",
    int(SelectContext.DRAW_COUNT): "choose_count",
    int(SelectContext.DAMAGE_COUNTER_COUNT): "choose_count",
}

_KIND = {
    int(OptionType.ATTACK): "attack", int(OptionType.EVOLVE): "evolve",
    int(OptionType.ABILITY): "ability", int(OptionType.RETREAT): "retreat",
    int(OptionType.END): "end", int(OptionType.YES): "yes", int(OptionType.NO): "no",
    int(OptionType.NUMBER): "number",
    int(OptionType.CARD): "card_select", int(OptionType.TOOL_CARD): "card_select",
    int(OptionType.ENERGY_CARD): "card_select", int(OptionType.ENERGY): "card_select",
}


def _card_info(cid, pokemon) -> CardInfo:
    cd = encode.CARD_TABLE.get(cid)
    if cd is None:
        return CardInfo()
    stage = encode._stage(cid)
    is_poke = cd.cardType == CardType.POKEMON
    return CardInfo(
        is_pokemon=is_poke,
        is_basic=is_poke and stage == 0,
        is_stage1=bool(getattr(cd, "stage1", False)),
        is_stage2=bool(getattr(cd, "stage2", False)),
        is_ex=bool(getattr(cd, "ex", False) or getattr(cd, "megaEx", False)),
        is_energy=cd.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY),
        is_item=cd.cardType == CardType.ITEM,
        is_supporter=cd.cardType == CardType.SUPPORTER,
        is_tool=cd.cardType == CardType.TOOL,
        is_attacker=encode._best_attack(cid) is not None and (
            stage >= 1 or (getattr(encode._best_attack(cid), "damage", 0) or 0) >= 90),
        hp=cd.hp or 0,
        energy_count=len(pokemon.energies) if isinstance(pokemon, Pokemon) else 0,
    )


def _play_kind(cid) -> str:
    cd = encode.CARD_TABLE.get(cid)
    if cd is None:
        return "other"
    return {CardType.POKEMON: "play_pokemon", CardType.ITEM: "play_item",
            CardType.SUPPORTER: "play_supporter", CardType.STADIUM: "play_stadium"}.get(
                cd.cardType, "other")


def build_context(obs) -> RuleContext:
    st = obs.current
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]
    op_active = op.active[0] if op.active else None
    hand = me.hand or []
    has_energy = any(encode.CARD_TABLE.get(c.id) and encode.CARD_TABLE[c.id].cardType
                     in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY) for c in hand)
    bench_attacker = any(encode._best_attack(p.id) for p in me.bench)
    return RuleContext(
        my_prizes=len(me.prize), opp_prizes=len(op.prize),
        opp_active_hp=op_active.hp if op_active else 999,
        energy_unused=not st.energyAttached, supporter_unused=not st.supporterPlayed,
        bench_count=len(me.bench), bench_has_attacker=bench_attacker,
        hand_has_energy=has_energy, my_active_at_risk=False,
        context=_CTX_MAP.get(int(obs.select.context), "other"),
    )


def build_optinfo(obs, o) -> OptInfo:
    my_index = obs.current.yourIndex
    t = int(o.type)
    kind = _KIND.get(t, "other")
    if t == int(OptionType.PLAY):
        card = encode.get_card(obs, AreaType.HAND, o.index, my_index)
        kind = _play_kind(card.id) if card is not None else "other"
    elif t == int(OptionType.ATTACH):
        card = encode.get_card(obs, AreaType.HAND, o.index, my_index)
        cd = encode.CARD_TABLE.get(getattr(card, "id", None))
        kind = "attach_tool" if cd and cd.cardType == CardType.TOOL else "attach_energy"

    card = encode._card_of_option(obs, o, my_index)
    target = None
    is_mine = True
    if card is not None:
        target = _card_info(card.id, card)
        if o.playerIndex is not None:
            is_mine = (o.playerIndex == my_index)

    attack = None
    if t == int(OptionType.ATTACK) and o.attackId in encode.ATTACK_TABLE:
        atk = encode.ATTACK_TABLE[o.attackId]
        op = obs.current.players[1 - my_index]
        op_active = op.active[0] if op.active else None
        dmg = atk.damage or 0
        mine = obs.current.players[my_index].active
        mtype = (encode.CARD_TABLE[mine[0].id].energyType
                 if mine and mine[0] and mine[0].id in encode.CARD_TABLE else None)
        tdata = encode.CARD_TABLE.get(op_active.id) if op_active else None
        attack = AttackInfo(
            damage=dmg,
            lethal=bool(op_active and dmg >= op_active.hp),
            hits_weakness=bool(tdata and mtype is not None and tdata.weakness == mtype),
            affordable=True,
        )
    return OptInfo(kind=kind, is_mine=is_mine, target=target, attack=attack)


def compile(ruleset: dict):
    ok, errors = validate(ruleset)
    if not ok:
        raise ValueError("invalid ruleset: " + "; ".join(errors))

    def agent(obs_dict, _deck=None):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return _deck if _deck is not None else []
        sel = obs.select
        ctx = build_context(obs)
        opts = [build_optinfo(obs, o) for o in sel.option]
        scores = score_options(ruleset, ctx, opts)
        return pick_from_scores(scores, sel.minCount, sel.maxCount)

    return agent


def load(path: str) -> dict:
    return json.load(open(path))
```

Then the shippable agent module:

```python
# rl/dsl_agent.py
"""Run a saved DSL rule-set as a harness agent (zero API). Engine side."""
from __future__ import annotations

import json
import os

from .config import solver_deck_path, REPO_ROOT
from .dsl.interpret import compile, load

my_deck = json.load(open(solver_deck_path()))
assert len(my_deck) == 60

_RULESET = os.environ.get("RL_RULESET",
                          str(REPO_ROOT / "rl" / "dsl" / "examples" / "handcrafted.json"))
_agent = compile(load(_RULESET))


def agent(obs_dict):
    return _agent(obs_dict, _deck=my_deck)
```

- [ ] **Step 4: Run test to verify it passes (host)**

Run: `python3 -m pytest tests/dsl/test_interpret_build.py -q`
Expected: PASS (2 passed). (This exercises only `pick_from_scores`, which is cg-free.)

- [ ] **Step 5: Commit**

```bash
git add rl/dsl/interpret.py rl/dsl_agent.py tests/dsl/test_interpret_build.py
git commit -m "feat(dsl): interpreter (obs->dataclasses->scores->picks) + dsl_agent"
```

---

### Task 4: Hand-authored rule-set + P0 gate (Docker integration)

**Files:**
- Create: `rl/dsl/examples/handcrafted.json`
- Create: `rl/dsl/gate.py` (gate runner: dsl_agent vs bare_agent on the gauntlet)

**Interfaces:**
- Consumes: `rl.dsl_agent` (agent module), `rl.config.EVAL_GAUNTLET`, `rl.eval` patterns (`_bare_for`), the Docker `cabt-rl` image.
- Produces: `rl/dsl/gate.py` `main()` printing per-deck and average win-rate of `dsl_agent` vs `bare_agent`, and a PASS/FAIL vs the bare baseline.

- [ ] **Step 1: Author the rule-set (the strategy to beat the baseline)**

```json
{
  "name": "handcrafted-p0",
  "default_weight": 0.0,
  "rules": [
    {"when": [{"pred": "is_attack"}, {"pred": "attack_is_lethal"}, {"pred": "opp_prizes_le", "n": 1}], "weight": 50000},
    {"when": [{"pred": "is_attack"}, {"pred": "attack_is_lethal"}], "weight": 1500},
    {"when": [{"pred": "is_attack"}, {"pred": "attack_hits_weakness"}], "weight": 1200},
    {"when": [{"pred": "is_attack"}, {"pred": "attack_damage_ge", "n": 90}], "weight": 1000},
    {"when": [{"pred": "is_attack"}], "weight": 800},
    {"when": [{"pred": "is_evolve"}], "weight": 900},
    {"when": [{"pred": "is_play_pokemon"}], "weight": 600},
    {"when": [{"pred": "is_play_pokemon"}, {"pred": "bench_count_ge", "n": 3}], "weight": -570},
    {"when": [{"pred": "is_play_item"}], "weight": 350},
    {"when": [{"pred": "is_play_supporter"}], "weight": 300},
    {"when": [{"pred": "is_play_stadium"}], "weight": 150},
    {"when": [{"pred": "is_ability"}], "weight": 1500},
    {"when": [{"pred": "is_attach_energy"}, {"pred": "target_is_mine"}, {"pred": "target_is_attacker"}], "weight": 700},
    {"when": [{"pred": "is_attach_energy"}, {"pred": "target_is_mine"}], "weight": 500},
    {"when": [{"pred": "is_attach_tool"}, {"pred": "target_is_attacker"}], "weight": 700},
    {"when": [{"pred": "is_card_select"}, {"pred": "ctx_to_hand"}, {"pred": "target_is_attacker"}], "weight": 300},
    {"when": [{"pred": "is_card_select"}, {"pred": "ctx_to_hand"}, {"pred": "target_is_pokemon"}], "weight": 260},
    {"when": [{"pred": "is_card_select"}, {"pred": "ctx_setup"}, {"pred": "target_is_basic"}], "weight": 100},
    {"when": [{"pred": "is_card_select"}, {"pred": "ctx_discard"}, {"pred": "target_is_pokemon"}], "weight": -50},
    {"when": [{"pred": "is_yes"}], "weight": 30},
    {"when": [{"pred": "is_end"}], "weight": 1}
  ]
}
```

- [ ] **Step 2: Write the gate runner**

```python
# rl/dsl/gate.py
"""P0 gate: does the handcrafted DSL rule-set match/beat bare_agent on the gauntlet?

Run (Docker): docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
  --entrypoint python cabt-rl -m rl.dsl.gate
"""
from __future__ import annotations

import importlib
import json
import os

from ..config import DECKS_DIR, EVAL_GAUNTLET, solver_deck_path
from ..env import TCGEnv, STOP   # reuse the env for play; STOP imported for parity
from .. import dsl_agent


def _bare_for(deck_path):
    os.environ["BARE_DECK"] = str(deck_path)
    import agents.bare_agent as ba
    importlib.reload(ba)
    return ba


def _play(solver_mod, opp_mod, solver_deck, opp_deck, games):
    """solver_mod pilots solver_deck vs opp_mod; seats swapped; returns win-rate.
    Drives both via their agent(obs_dict) interface through the engine."""
    from cg.game import battle_start, battle_select, battle_finish
    wins = 0
    for g in range(games):
        seat = g % 2
        decks = (solver_deck, opp_deck) if seat == 0 else (opp_deck, solver_deck)
        mods = (solver_mod, opp_mod) if seat == 0 else (opp_mod, solver_mod)
        obs, start = battle_start(decks[0], decks[1])
        if obs is None:
            continue
        try:
            for _ in range(20000):
                cur = obs.get("current")
                if cur is None or cur.get("result", -1) != -1 or obs.get("select") is None:
                    break
                who = cur["yourIndex"]
                action = mods[who].agent(obs)
                obs = battle_select(action if isinstance(action, list) else list(action))
            res = (obs.get("current") or {}).get("result", -1)
        finally:
            battle_finish()
        if res == seat:
            wins += 1
    return wins / games


def main():
    games = int(os.environ.get("RL_GATE_GAMES", "20"))
    solver_deck = json.load(open(solver_deck_path()))
    print(f"P0 gate: dsl_agent vs bare_agent, {games} games/deck, seats swapped\n")
    dsl_total, bare_total = [], []
    for deck_name in EVAL_GAUNTLET:
        bare = _bare_for(DECKS_DIR / deck_name)
        opp_deck = list(bare.my_deck)
        # dsl_agent (our strategy) vs bare opponent
        dsl_wr = _play(dsl_agent, bare, solver_deck, opp_deck, games)
        # baseline: bare pilot (same solver deck) vs bare opponent
        bare_solver = _bare_for(solver_deck_path())   # bare piloting OUR deck
        bare_wr = _play(bare_solver, bare, solver_deck, opp_deck, games)
        dsl_total.append(dsl_wr); bare_total.append(bare_wr)
        print(f"  {deck_name:34s} dsl {dsl_wr:5.1%} | bare {bare_wr:5.1%}")
    d, b = sum(dsl_total) / len(dsl_total), sum(bare_total) / len(bare_total)
    print(f"\n  AVG  dsl {d:.1%} | bare {b:.1%}   ->  "
          f"{'PASS' if d >= b else 'FAIL'} (gate: dsl >= bare)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Build the image and validate the rule-set loads + agent runs**

Run:
```bash
docker build --platform=linux/amd64 -f rl/Dockerfile -t cabt-rl .
docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app --entrypoint python cabt-rl \
  -c "import rl.dsl_agent as d; from rl.dsl.grammar import validate; import json; \
print(validate(json.load(open('rl/dsl/examples/handcrafted.json')))); \
print('deck', len(d.my_deck))"
```
Expected: `(True, [])` and `deck 60`.

- [ ] **Step 4: Run the P0 gate**

Run:
```bash
docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app --entrypoint python cabt-rl \
  -m rl.dsl.gate
```
Expected: a per-deck table and a final `AVG dsl XX.X% | bare YY.Y% -> PASS`. If `FAIL`, iterate the weights/predicates in `handcrafted.json` (and widen `predicates.py` vocabulary only if no weighting can close the gap) until `dsl >= bare`. **This PASS is the P0 gate that authorizes P1+ (the SmolLM3 loop).**

- [ ] **Step 5: Commit**

```bash
git add rl/dsl/examples/handcrafted.json rl/dsl/gate.py
git commit -m "feat(dsl): handcrafted rule-set + P0 gauntlet gate (vs bare_agent)"
```

---

## Self-Review

**Spec coverage (spec §):**
- §3 (DSL not code) → Task 1 grammar/validate gate. ✓
- §4.1/§4.2/§4.3 (DSL shape, vocabulary, schema) → Task 1 (`PREDICATE_ARITY`, `validate`) + Task 2 (`PREDICATES`). ✓
- §5 (`grammar.py`, `predicates.py`, `interpret.py`, `dsl_agent.py`) → Tasks 1–3. ✓ (`rl/rulegen/*`, `prompts`, `author`, `evaluate`, `train`, `dataset`, `pod-requirements.txt` are P1+ — deferred to the follow-on plan, by design.)
- §9 P0 (hand-authored rule-set beats `bare_agent`) → Task 4 gate. ✓
- §10 testing (host unit + Docker integration) → host tests Tasks 1–3, Docker gate Task 4. ✓
- §11 risk "DSL too narrow" → Task 4 Step 4 instructs widening `predicates.py` before LLM work. ✓
- §12 config knobs `RL_RULESET` → Task 3 `dsl_agent.py`; `RL_GATE_GAMES`/`RL_GAUNTLET_GAMES` → Task 4. ✓
- P1–P4 (SmolLM3 author/evaluate/train, expert-iteration, CISPO) → **out of scope for this plan**; follow-on plan after the P0 gate passes.

**Placeholder scan:** No TBD/TODO; every code step has full code; the only "iterate until PASS" is Task 4 Step 4, which is the intended human-in-the-loop tuning of a *data* file, not a code placeholder. ✓

**Type consistency:** `RuleContext`/`OptInfo`/`CardInfo`/`AttackInfo` field names are defined in Task 2 and consumed identically by `interpret.build_context`/`build_optinfo` in Task 3. `score_options(ruleset, ctx, opts)` and `pick_from_scores(scores, min_count, max_count)` signatures match across Tasks 2–4. `PREDICATES`/`PREDICATE_ARITY` key sets are asserted equal in Task 2. ✓

---

## Follow-on (not this plan)

After the P0 gate PASSes, a second plan covers P1–P4: `rl/rulegen/prompts.py`,
`author.py` (SmolLM3 schema-constrained generation), `evaluate.py` (quick-screen →
gauntlet reward), `dataset.py`, and `train.py` (expert iteration → CISPO with LoRA),
plus `rl/pod-requirements.txt` — all run on the GPU pod.
