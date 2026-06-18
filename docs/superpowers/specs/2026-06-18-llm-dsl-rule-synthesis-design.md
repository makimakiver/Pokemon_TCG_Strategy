# LLM-Authored DSL Rules, Sim-Tested, LLM Fine-Tuned — Design Spec

**Date:** 2026-06-18
**Status:** Approved for planning (brainstorm complete)
**Track:** Parallel to the solver-net RL stack (`rl/policy.py`, `rl/train_solver.py`).
Reuses `rl/encode.py`, the sim harness (`runner.py`/`cg`), and `rl/eval.py`. Does
not modify the net track.

---

## 1. Goal & non-goals

**Goal.** A closed loop in which a small local LLM (SmolLM3-3B) *authors decision
strategies* for the cabt Pokémon TCG agent, expressed in a constrained,
non-executable DSL. Each strategy is compiled to a sim agent, scored by actually
playing games, and the win-rates are used to **fine-tune the LLM** (expert
iteration, then CISPO) so it writes better strategies over time.

**Two deliverables:**
1. **The best rule-set** — a JSON strategy that runs as a zero-API agent
   (`rl/dsl_agent.py`); this is shippable as a competition submission.
2. **The fine-tuned rule-author LLM** — a LoRA adapter over SmolLM3-3B that
   emits strong rule-sets on demand.

**Non-goals.**
- Not fine-tuning Claude (impossible via API) and not the net-distillation track.
- Not free-form code generation by the LLM (rejected for safety — see §3).
- Not replacing the solver net; this is a separate *symbolic policy* track that
  may later feed priors into the net, but ships independently.

---

## 2. Decision ledger (locked in brainstorm)

| # | Decision | Choice |
|---|---|---|
| 1 | Compute | GPU **runner pods** (Linux/CUDA). Sim runs *native* there (no Docker emulation); LLM inference + LoRA train on GPU. Dev locally. |
| 2 | Rule form | **Constrained JSON DSL** (`condition → weight` rules). Data, not code → no arbitrary execution. |
| 3 | Fine-tune | **Expert iteration → CISPO** (critic-free; degenerate-group fallback), **LoRA**. |
| 4 | Reward | **Quick screen** (1–2 fixed decks) → **full 10-deck gauntlet** for finalists. |
| 5 | Model | **`HuggingFaceTB/SmolLM3-3B`** (Apache-2.0, 64k ctx, TRL/PEFT LoRA, `/think`·`/no_think` modes). Config knob. |
| 6 | De-risk gate | **P0:** a *hand-authored* rule-set must match/beat `bare_agent` in sim **before** any GPU/LLM spend. |

---

## 3. Why a DSL, not code

The LLM emits **JSON data** validated against a schema, never Python. The
interpreter only performs arithmetic over a fixed, vetted feature vocabulary, so
a malformed or adversarial rule-set cannot execute anything — it is rejected by
schema validation (the legality gate). This is the safety basis for running
thousands of LLM-authored strategies unattended on a pod.

---

## 4. The DSL

### 4.1 Shape
A rule-set scores each legal option in the current selection; the agent then picks
options by descending score, respecting the engine's `minCount`/`maxCount` (same
contract as `agents/bare_agent.py`).

```json
{
  "name": "honchkrow-aggro-v1",
  "default_weight": 0.0,
  "rules": [
    {"when": [{"pred": "is_attack"}, {"pred": "attack_is_lethal"}], "weight": 800},
    {"when": [{"pred": "is_attack"}, {"pred": "attack_is_lethal"},
              {"pred": "opp_prizes_le", "n": 1}], "weight": 5000},
    {"when": [{"pred": "is_play_pokemon"}], "weight": 600},
    {"when": [{"pred": "is_attach_energy"}, {"pred": "target_is_mine"},
              {"pred": "target_is_attacker"}], "weight": 400},
    {"when": [{"pred": "is_evolve"}], "weight": 500},
    {"when": [{"pred": "is_end"}], "weight": 1}
  ]
}
```

`score(option) = default_weight + Σ weight` over every rule whose **`when`
predicates all match** that option in the current state. A rule's `when` is an
AND of predicates; multiple rules OR together additively.

### 4.2 Predicate vocabulary (fixed, vetted)
All predicates are pure functions of `(observation, option)` implemented on top of
`rl/encode.py` helpers + the runtime `card_table`/`attack_table`. `n` is an
integer arg where noted.

- **Option type:** `is_attack`, `is_play_pokemon`, `is_play_item`,
  `is_play_supporter`, `is_play_stadium`, `is_attach_energy`, `is_attach_tool`,
  `is_evolve`, `is_ability`, `is_retreat`, `is_end`, `is_card_select`,
  `is_yes`, `is_no`, `is_number`.
- **Target card** (the card/Pokémon the option concerns): `target_is_pokemon`,
  `target_is_basic`, `target_is_stage1`, `target_is_stage2`, `target_is_ex`,
  `target_is_energy`, `target_is_item`, `target_is_supporter`, `target_is_tool`,
  `target_is_attacker`, `target_is_mine`, `target_is_opponent`,
  `target_hp_ge{n}`, `target_energy_ge{n}`.
- **Attack:** `attack_is_lethal`, `attack_hits_weakness`, `attack_damage_ge{n}`,
  `attack_affordable`.
- **Board / global:** `my_prizes_le{n}`, `opp_prizes_le{n}`, `opp_active_hp_le{n}`,
  `my_active_at_risk` (opp can KO us next turn, heuristic), `bench_has_attacker`,
  `hand_has_energy`, `energy_unused`, `supporter_unused`, `bench_count_ge{n}`.
- **Context** (groups `SelectContext`): `ctx_setup`, `ctx_main`, `ctx_to_hand`,
  `ctx_attach_from`, `ctx_discard`, `ctx_switch`, `ctx_choose_count`.

New predicates are added in `grammar.py` only (human-vetted), never by the LLM.

### 4.3 Schema / legality gate
JSON Schema enforces: `rules` is a list (≤ 40); each rule has `when`
(list, 1–6 predicate objects) and numeric `weight` in `[-100000, 100000]`; each
predicate's `pred` is in the registry enum and `n` (when required) is an integer
in a bounded range. `default_weight` numeric. Anything failing → reward 0,
excluded from training (mirrors the engine-legality gate in `rl/guide.py`).

---

## 5. Components & interfaces

```
rl/dsl/
  grammar.py        # predicate registry + JSON schema + validate(ruleset)->ok/errors
  interpret.py      # compile(ruleset) -> agent(obs_dict)->list[int]
  predicates.py     # the predicate implementations over encode.py
  examples/handcrafted.json   # P0 baseline rule-set (human-authored)
rl/rulegen/
  prompts.py        # DSL spec text + few-shot + board/meta context -> prompt
  author.py         # SmolLM3 wrapper: prompt -> JSON ruleset (schema-constrained)
  evaluate.py       # ruleset -> quick screen -> gauntlet -> reward (win-rate)
  dataset.py        # build LoRA-SFT pairs (prompt -> top ruleset)
  train.py          # outer loop: expert iteration, then CISPO
rl/dsl_agent.py     # loads RL_RULESET json -> shippable agent (zero API)
rl/pod-requirements.txt   # transformers>=4.53, peft, trl, accelerate, (vllm optional)
```

- **`grammar.validate(ruleset) -> (bool, list[str])`** — pure, host-testable.
- **`interpret.compile(ruleset) -> callable`** — returns an `agent(obs_dict)`
  identical in contract to `bare_agent.agent`; reuses `encode.get_card` etc.
- **`author.RuleAuthor.propose(context, n, mode) -> list[ruleset]`** — samples N
  rule-sets from SmolLM3 with schema-constrained decoding; `mode` selects
  `/no_think` (fast, bulk) or `/think` (deep). Pod-only.
- **`evaluate.score(ruleset, screen_decks, gauntlet, games) -> {reward, detail}`**
  — compile → screen → (if above threshold) gauntlet; reuses `runner`/`rl.eval`.
  Parallel across candidates via process pool (sim is the bottleneck).
- **`train.expert_iteration(...)` / `train.cispo(...)`** — outer loops (pod).

---

## 6. Data flow

```
SmolLM3 (LoRA) --propose--> JSON ruleset --validate(grammar)--> compile(interpret)
   ^                                                               |
   |                                                          agent in sim
   |                                                               |
   |                                       quick screen (1-2 decks) -> [gate]
   |                                                               |
   |                                              full 10-deck gauntlet (finalists)
   |                                                               |
   +-----  fine-tune (expert-iter SFT | CISPO)  <---- reward = win-rate
```

Best ruleset (by gauntlet) → `rl/dsl_agent.py` (shippable). LoRA adapter saved
per round.

---

## 7. Training

### 7.1 Expert iteration (bootstrap)
Each round: `propose` N rule-sets (mostly `/no_think`) → `evaluate` → keep top-k by
reward → LoRA-SFT SmolLM3 on `(prompt → top ruleset)` pairs (`dataset.py` + TRL
`SFTTrainer`). Repeat until the model reliably emits **valid** (schema-passing) and
**decent** (beats a floor) rule-sets. Robust to reward noise; learns from winners
only.

### 7.2 CISPO (push past imitation)
Sample **G** rule-sets per prompt, score each, advantage = `(r − group_mean) /
group_std`; CISPO update (clip the IS-weight, keep the REINFORCE term — preserves
gradient on rare-but-important tokens) via a pod LLM-RL stack (TRL/verl) with LoRA.
**Degenerate groups** (std≈0, common when most rule-sets lose) fall back to
REINFORCE½ on the winning samples — mirroring `rl/solver_objectives.py::CISPO`.
Instrument entropy, group-reward spread, KL-to-reference, fraction-valid.

### 7.3 Reward (the expensive part)
- **Quick screen:** vs 1–2 fixed `bare_agent` opponents (default Crustle +
  Dragapult), small game count (e.g. 10, seats swapped). Cheap, kills bad
  candidates early.
- **Gauntlet gate:** candidates above a screen threshold play the full held-out
  10-deck gauntlet (`rl/eval.py`, meta-share weighted) with more games. This
  win-rate is the selection/ship metric.
- Tunables: `screen_decks`, `screen_games`, `screen_threshold`, `gauntlet_games`.
  Use SmolLM3 `/no_think` to keep candidate volume cheap so sim, not the LLM, is
  the bottleneck.

---

## 8. Where it runs

- **Pod (Linux/CUDA):** everything — sim native (no emulation), SmolLM3 inference,
  LoRA training. `rl/pod-requirements.txt` + the engine.
- **Local (Mac):** develop and unit-test the *non-LLM* pieces — `grammar`,
  `predicates`, `interpret`, and `evaluate` against the Docker sim. The GPU pieces
  (`author`, `train`) are exercised on the pod.

---

## 9. Phased plan (de-risked)

- **P0 — DSL + interpreter, NO LLM (the gate).** Implement `grammar`,
  `predicates`, `interpret`. Hand-author `examples/handcrafted.json` and show via
  the sim that it **matches or beats `bare_agent`** on the gauntlet. If a human
  can't beat the baseline with the DSL, widen the vocabulary before any LLM work.
- **P1 — loop with a FROZEN LLM (or random-ruleset sampler).** Build `prompts`,
  `author`, `evaluate`, and an evolutionary keep-the-best driver — *no training*.
  Validates the whole pipeline (propose → validate → compile → screen → gauntlet)
  on a pod.
- **P2 — expert-iteration LoRA-SFT** (`dataset`, `train.expert_iteration`).
- **P3 — CISPO** (`train.cispo`) with full diagnostics + degenerate-group fallback.
- **P4 — ship.** Freeze the best ruleset → `rl/dsl_agent.py`; optionally export top
  rules as a prior into the net track.

---

## 10. Testing

- **Unit (host, no engine):** `grammar.validate` accepts/rejects fixtures;
  predicate truth tables on synthetic obs dicts; `interpret` picks the expected
  option on hand-built selects.
- **Integration (Docker sim):** the handcrafted ruleset agent reconciles with
  `runner.py` win attribution; `evaluate.score` is reproducible (fixed seeds).
- **Loop smoke (pod or Docker):** random/templated rule-sets flow through
  propose→screen→gauntlet without crashes; degenerate groups handled.
- **Gate test:** P0 handcrafted ruleset ≥ `bare_agent` gauntlet win-rate.

---

## 11. Risks & mitigations

- **DSL too narrow** (no strong rule-set exists) → P0 gate catches it before GPU
  spend; widen `predicates.py` vocabulary, re-run P0.
- **Reward noise/cost** → quick-screen funnel; fixed seeds; `/no_think` bulk
  sampling; parallel sim workers.
- **CISPO entropy collapse / degenerate groups** → instrument entropy + spread;
  REINFORCE½ fallback (already proven in the net track).
- **Schema gaming / invalid output** → hard schema gate (reward 0); reward
  fraction-valid; expert-iteration teaches valid format first.
- **Overfit to screen decks** → gauntlet gate on held-out decks is the selection
  metric, never the screen.
- **Pod portability** → all paths via `rl/config.py` + env; no Mac-only assumptions.

---

## 12. Config knobs (env, via `rl/config.py`)

`RL_RULEGEN_MODEL` (default `HuggingFaceTB/SmolLM3-3B`), `RL_RULESET` (path for
`dsl_agent`), `RL_SCREEN_DECKS`, `RL_SCREEN_GAMES`, `RL_SCREEN_THRESHOLD`,
`RL_GAUNTLET_GAMES`, `RL_EI_ROUNDS`, `RL_EI_SAMPLES`, `RL_EI_TOPK`,
`RL_CISPO_GROUP` (G), plus the existing `RL_SOLVER_DECK` for the deck piloted.

---

## 13. Open items (defaults chosen; override anytime)

- [ ] Exact screen decks (default Crustle + Dragapult) and game counts.
- [ ] LoRA rank/targets for SmolLM3 (start r=16 on attn+MLP proj).
- [ ] CISPO group size G vs sim budget (start G=8, matching the net track).
- [ ] Whether P4 exports rules as net priors (deferred; not on critical path).
- [ ] `git init` if version control / spec commit is wanted (repo is not git yet).
