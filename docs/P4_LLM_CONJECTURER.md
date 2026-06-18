# P4 — LLM Conjecturer (CISPO LoRA), GPU-pod runbook

Optional upgrade path for the SGS Conjecturer (plan §3.4 / §5 P4). Swaps the
parametric edit-script policy for a small open instruct model that emits a
chain-of-thought + structured edit-script, **LoRA fine-tuned with CISPO** — the
same objective the Solver net uses (`rl/solver_objectives.CISPO`), **not GRPO**.

It is OFF the critical path: only reach for it if the parametric conjecturer /
rule-based guide plateau. Everything degrades to the parametric conjecturer when
`transformers` / a GPU / the weights are absent, so the stack still imports and
plumbing-tests on a CPU host.

## Pieces

| File | Role |
|---|---|
| `rl/conjecturer/author.py` | `LLMConjecturer`: prompt → sample edit-script → parse/legality-clamp → apply; records the **behavior log-prob** (`old_logp`) + reward to a replay buffer (CISPO's IS weight needs `old_logp`). Falls back to parametric without a model. |
| `rl/conjecturer/cispo_train.py` | Offline **CISPO** LoRA fine-tune over the buffer: group by target prompt, advantage `(R−mean)/std`, clipped stop-grad IS weight, REINFORCE½ degenerate-group fallback. Saves the adapter. |
| `rl/conjecturer/grpo_train.py` | Deprecated shim → redirects to `cispo_train` (we do not use GRPO). |
| `rl/requirements-llm.txt` | `transformers/peft/accelerate/...` — installed on top of the base image. |
| `rl/Dockerfile.llm` | `cabt-rl` + the LLM deps → `cabt-rl-llm` (GPU pod image). |

Config knobs (all `RL_CONJ_*`, see `rl/config.py`): `RL_CONJ_MODEL`
(default `HuggingFaceTB/SmolLM2-1.7B-Instruct`), `RL_CONJ_LORA`, `RL_CONJ_BUFFER`,
`RL_CONJ_DEVICE`, `RL_CONJ_4BIT`, `RL_CONJ_LR/EPOCHS/BATCH`, `RL_CONJ_LORA_R/ALPHA/DROPOUT`.
The CISPO clip / degenerate-eps are the **shared** `cispo_clip` / `cispo_std_eps`.

## CISPO objective (sequence-level)

```
group by target prompt
  a = (R_synth − mean_g) / std_g                       # group-normalized advantage
  per (prompt, completion):
    lp   = Σ log π_θ(token)                             # current LoRA policy
    is_w = clamp(exp(lp − old_logp), max=cispo_clip).detach()
    loss += − is_w · a.detach() · lp  − entropy_coef · H̄
  degenerate group (std_g < cispo_std_eps) → REINFORCE½:  a = R_synth − 0.5
```

Identical semantics to the Solver's `CISPO`, just with the LLM completion log-prob
in place of the pointer-net option log-prob.

## Build (once)

```bash
docker build --platform=linux/amd64 -f rl/Dockerfile -t cabt-rl .          # base (if not built)
docker build --platform=linux/amd64 -f rl/Dockerfile.llm -t cabt-rl-llm .   # + LLM deps
```

## Run on the GPU pod

```bash
# 1) Collect a CISPO buffer: the model-backed conjecturer drives an SGS generation.
docker run --rm --gpus all -v "$PWD":/app -w /app \
  -e RL_CONJ=llm -e RL_CONJ_DEVICE=cuda cabt-rl-llm -m rl.outer_loop
#    -> appends rewarded (prompt, completion, old_logp, R_synth) rows to
#       rl/runs/conjecturer_buffer.jsonl

# 2) Offline CISPO LoRA fine-tune on the buffer -> rl/runs/conjecturer_lora/
docker run --rm --gpus all -v "$PWD":/app -w /app cabt-rl-llm \
  -m rl.conjecturer.cispo_train --epochs 2

# 3) Next generation's author auto-loads the adapter from RL_CONJ_LORA. Repeat 1–2
#    (alternate buffer-collection and CISPO updates) as the P4 curriculum loop.
```

The two RL stacks never share the GPU: net-RL (Solver) and LLM-RL (this
conjecturer) run in **separate phases**, exactly as the plan specifies.

## Plumbing dry-runs (no GPU, CPU host)

```bash
# Fallback proposal path (no model) + buffer logging:
RL_CONJ=llm python -c "from rl.conjecturer import get_conjecturer; print(get_conjecturer('llm').snapshot())"
# CISPO group-advantage math, no model loaded:
python -m rl.conjecturer.cispo_train --dry-run
```

(Fallback buffer rows carry `old_logp=None` and are skipped by the trainer — only
real model generations are CISPO-trainable.)
