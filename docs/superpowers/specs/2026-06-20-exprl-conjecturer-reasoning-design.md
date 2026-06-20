# ExpRL Reasoning-Training for the SmolLM Conjecturer — Design

**Date:** 2026-06-20
**Status:** Approved direction; pending spec review → implementation plan
**Paper:** ExpRL — Exploratory RL for LLM Mid-Training (arXiv 2606.17024)
**Related:** `rl/conjecturer/{author,cispo_train}.py`, `docs/SGS_CONJECTURER_SESSION_SUMMARY.md`, memory `objective-is-cispo`

## 1. Goal

Train the SmolLM conjecturer's **chain-of-thought reasoning** (not just its edit-script output) by rewarding it for reasoning that a frontier model (GLM-5.2) would judge as **aligned with a strong reference solution** — the ExpRL method. This is an explicit **stage-1 "mid-training"** that builds reasoning coverage before the engine-grounded sparse-reward RL (stage 2, out of scope here).

### Decisions captured (from brainstorming)

1. **Method:** faithful ExpRL — an **LLM-judge rubric**, NOT embedding/cosine similarity. ExpRL uses no vector metric; alignment is an LLM reading text and emitting an integer.
2. **Reference model:** **GLM-5.2 via the z.ai/Zhipu API** generates the reference CoT + answer per target (cached).
3. **Per-sample judge:** a **local 7–14B model on the mini (MPS)** scores each SmolLM completion 1–5 against the cached reference — no API call in the RL hot loop.
4. **Reward:** `R = (s̃ − 1)/4 ∈ [0,1]`, `s̃ ∈ {1..5}` (ExpRL normalization), scoring both the answer and the reasoning toward it.
5. **Optimization:** **GRPO + KL-to-base** — faithful to ExpRL (user override 2026-06-20, scoped to THIS path only). GRPO = group-normalized advantage `(r−mean)/std` + PPO-style clipped ratio `min(ρ·A, clip(ρ,1±ε)·A)` + `β·KL(π_θ‖π_0)`. The group-advantage computation is **shared with CISPO** (`cispo_train.group_advantages`); only the update wrapper differs (clipped ratio vs CISPO's clipped stop-grad IS weight). **CISPO remains the objective for the solver and the existing R_synth conjecturer path** — this GRPO override applies only to ExpRL reasoning-training.
6. **Baseline:** **K completions per target (group)**, group-normalized advantage (GRPO's "group relative") — a single isolated score has no baseline. Degenerate groups (all K scores equal) fall back to REINFORCE½ (existing `group_advantages` behavior).
7. **Concurrency:** runs on the mini's MPS, concurrent with the engine/solver training (Docker/CPU) — no resource contention.

## 2. Non-goals

- **Stage 2 (engine grounding):** rewarding whether the conjecture actually produces a winnable-but-non-trivial position (`R_solve`). Designed-for (reward interface is pluggable) but not built here.
- **Embedding/cosine similarity:** explicitly rejected — it is not ExpRL and rewards vocabulary mimicry.
- Verifying GLM-5.2 availability/pricing is an implementation pre-step, not a design assumption.

## 3. Architecture (3 models)

```
target position (from D)
   │
   ├─(once/target, cached)─► GLM-5.2 (z.ai API) ──► reference CoT + answer  y*
   │
   └─(K samples, blind)────► SmolLM conjecturer ──► {CoT_k (### steps) + answer_k}
                                   │                         │
                                   │      local 7–14B judge (MPS): reads (x, y_k, y*)
                                   │      "verify, don't solve" → s̃_k ∈ 1..5
                                   ▼                         ▼
                          buffer rows {target_id, prompt, completion_k,
                                       old_logp_k, reward_k=(s̃_k−1)/4}
                                   │
                                   ▼  group by target → group advantage → GRPO + KL-to-base
                          SmolLM LoRA update (reason_train.py / grpo_train.py)
```

ExpRL fidelity: the reference `y*` is shown **only to the judge**, never to SmolLM (on-policy exploration preserved; reference is a reward scaffold, not an imitation target). The judge is instructed to **verify, not solve** ("Do not solve the problem yourself; do not fill in omitted steps").

## 4. Components

### 4.1 `rl/conjecturer/reason_reference.py` (new)
GLM-5.2 reference generator + cache.
- `build_references(targets, out_path)` — for each target, prompt GLM-5.2 (OpenAI-compatible z.ai client; key `GLM_API_KEY`, model `RL_GLM_MODEL` default `glm-5.2`, endpoint `RL_GLM_BASE_URL`) for a CoT + edit-script answer. Persist `{target_id: {"cot": str, "answer": str}}` to JSONL.
- Pluggable client; on API failure logs + skips the target (its samples get no reward → filtered out of training, never a crash).
- Caches: re-run reuses existing rows; only missing targets call the API.

### 4.2 `rl/conjecturer/reason_judge.py` (new)
Local rubric judge (the per-sample scorer).
- Loads a quantized 7–14B instruct model on MPS (`RL_JUDGE_MODEL`, default e.g. `Qwen2.5-7B-Instruct` 4-bit; `RL_JUDGE_DEVICE=mps`).
- `score(problem_text, smollm_trace, reference) -> int` in `{1..5}` using the ExpRL rubric (1 = wrong/far … 5 = sound, complete, directly implies the reference). System prompt = "verify, don't solve."
- `reward(s̃) = (s̃ − 1)/4`. Robust parse of the integer (default to 1 = lowest on unparseable output, logged).
- Optional **ExpRL-Process**: split `smollm_trace` on `###`, score each prefix, return per-step `s̃_t`; advantage `A_t = s̃_t − s̃_{t−1}` (`A_1 = s̃_1 − s̃_T`). v1 ships outcome scoring; process is a flagged extension.

### 4.3 `rl/conjecturer/reason_train.py` (new, thin driver) + GRPO update
Stage-1 ExpRL loop with a **GRPO** update (scoped here only).
- For each target in a batch: SmolLM (`author.LLMConjecturer`, reused) generates **K** completions, each already logging `{prompt, completion, old_logp}` (the behavior sequence log-prob both GRPO and CISPO need).
- For each completion: `reward_k = reason_judge.reward(score(x, trace_k, reference[target]))`; write into the buffer row's `reward`.
- **Group advantages:** reuse `cispo_train.group_advantages` (identical for GRPO — group-normalized `(r−mean)/std`, degenerate→REINFORCE½).
- **GRPO update** (the only new objective code): per row, ratio `ρ = exp(lp − old_logp)`; loss `−min(ρ·A, clip(ρ, 1±ε)·A) + β·KL(π_θ‖π_0)`, length-normalized; `ε = RL_GRPO_CLIP` (reuse `ppo_clip` default 0.2), `β = RL_EXPRL_KL_BETA`. Lives in `reason_train.py` (or flesh out the existing `rl/conjecturer/grpo_train.py` stub) — **does NOT touch `cispo_train.py`'s objective**, so the CISPO conjecturer/solver paths are untouched.
- Saves a LoRA adapter to `rl/runs/conjecturer_lora_exprl/`.

### 4.4 Reuse (no rewrite)
- `author.LLMConjecturer` — SmolLM generation + `old_logp` logging + buffer rows (already exists; may need a `k`/format knob to emit `###`-delimited CoT).
- `cispo_train.py` — **only** `group_advantages` is reused (groups = target prompt, `(r−mean)/std`, degenerate→REINFORCE½; identical for GRPO). Its CISPO update is NOT used by this path (ExpRL uses the GRPO update in §4.3); `cispo_train.py` is left unmodified.

## 5. Data flow (end to end)

1. **(mini, once)** `reason_reference.build_references(D)` → GLM-5.2 → `rl/runs/exprl_references.jsonl`. ~|D| API calls, cached.
2. **(mini, RL loop, concurrent with solver Docker)** per generation: sample K SmolLM completions/target → local judge scores each vs cached reference → reward → group-normalize → GRPO+KL LoRA step.
3. Output: `conjecturer_lora_exprl/` adapter; eval = valid-edit-rate (existing) + judge-score trend.

## 6. Error handling

- **GLM API failure / missing key:** that target gets no reference → its samples are dropped from training (filtered like rows without a reward). The run continues; logged. A dry-run mode (no key) builds an empty reference set and warns.
- **Judge unparseable output:** default `s̃ = 1` (lowest), logged; never crashes the step.
- **Degenerate group** (all K scores equal): existing REINFORCE½ fallback in `group_advantages`.
- **KL term** is additive and flag-gated so it cannot silently change the existing conjecturer-CISPO path.

## 7. Risks

- **Imitation / reward-gaming:** even with a rubric (not cosine), a model can learn to *look* aligned. ExpRL mitigates by (a) reference seen only by the judge, (b) "verify don't solve" judge, (c) KL-to-base. The real correction is **stage 2 engine grounding** — kept as the explicit next phase.
- **Small-judge reliability:** a 7–14B local judge is weaker than GLM. Mitigation: optionally have GLM **calibrate** the local judge on a sample (compare scores) before the run; if divergence is high, escalate that subset to API scoring. (Diagnostic, not v1-blocking.)
- **GLM-5.2 access:** model id / endpoint / key must be live; verified as the first implementation step, with Claude (`distill_claude.py` path) as a drop-in reference fallback if GLM is unavailable.
- **MPS memory:** the 7–14B judge (4-bit ~4–9 GB) + SmolLM (small) fit the mini's 24 GB; GLM is API so never resident. The judge loads in the RL phase; the (separate) reference phase only needs the API client.

## 8. Testing

- **Pure-Python (no model/API):** `reward(s̃)` normalization; rubric integer parsing (incl. malformed); `group_advantages` already tested. Reference-cache read/write round-trip with a stub client.
- **Small live smoke (mini):** 3–5 targets → GLM references (or Claude fallback) → K=4 SmolLM samples → local judge scores → one CISPO+KL step → assert a finite loss, a saved adapter, and rewards in [0,1].
- Metric: judge-score trend over generations (should rise) + the existing valid-edit-rate eval (should not regress).
