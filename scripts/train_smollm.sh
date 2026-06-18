#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train_smollm.sh — one-shot P4 SmolLM conjecturer training on an Apple-Silicon
# Mac mini. Run this NATIVELY on the mini (over Tailscale SSH), from the repo root.
#
#   Phase A (buffer):  Docker linux/amd64  — needs the cg engine (search_begin).
#                      Apple GPU is NOT visible to Docker, so SmolLM generates on
#                      the (emulated) CPU here → keep the model small + gens few.
#   Phase B (LoRA):    NATIVE macOS + MPS  — pure transformers/peft, no engine.
#
# The model MUST be identical in both phases (CISPO's importance weight compares
# Phase-B log-probs against the Phase-A `old_logp`).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."                       # repo root

# ---- knobs (override via env) --------------------------------------------
MODEL="${RL_CONJ_MODEL:-HuggingFaceTB/SmolLM2-360M-Instruct}"  # 360M is tractable on emulated CPU; bump to -1.7B- once it works
GENERATIONS="${GENERATIONS:-3}"               # Phase-A SGS generations (start small; CPU LLM gen is slow)
BATCH="${BATCH:-4}"                           # targets sampled per generation
EPOCHS="${EPOCHS:-2}"                         # Phase-B CISPO epochs
DEVICE_B="${RL_CONJ_DEVICE:-mps}"             # Phase-B device (mps | cpu). Falls back to CPU per-op if an op is unsupported.
HF_VOL="${HF_VOL:-$HOME/.cache/huggingface}"  # shared HF weight cache (native + Docker mount the same dir)
export HF_HOME="$HF_VOL"

echo "== config =="
echo "  model       = $MODEL"
echo "  phase A     = $GENERATIONS gens x $BATCH targets (Docker, CPU)"
echo "  phase B     = $EPOCHS epochs on $DEVICE_B (native)"
echo "  hf cache    = $HF_VOL"

# ---- 0. build the LLM image once -----------------------------------------
if ! docker image inspect cabt-rl-llm:latest >/dev/null 2>&1; then
  echo "== building cabt-rl-llm (one-time) =="
  docker image inspect cabt-rl:latest >/dev/null 2>&1 || \
    docker build --platform=linux/amd64 -f rl/Dockerfile     -t cabt-rl     .
  docker build  --platform=linux/amd64 -f rl/Dockerfile.llm -t cabt-rl-llm .
fi

# ---- A. collect the CISPO buffer (Docker; engine + SmolLM on CPU) ---------
# D is built from data/loser/lost_*.json — already includes the 5 Crow losses
# (lost_crow1..5.json). To train ONLY on Crow, move the other lost_*.json aside.
echo "== Phase A: collecting conjecturer buffer (this is the slow part) =="
docker run --rm --platform=linux/amd64 \
  -v "$PWD":/app -w /app -e PYTHONPATH=/app \
  -v "$HF_VOL":/root/.cache/huggingface \
  -e RL_CONJ=llm -e RL_CONJ_DEVICE=cpu \
  -e RL_CONJ_MODEL="$MODEL" \
  -e RL_CONJ_MAXNEW="${RL_CONJ_MAXNEW:-160}" -e RL_K="${RL_K:-2}" \
  --entrypoint python cabt-rl-llm \
  -c "from rl.outer_loop import run_sgs; run_sgs(generations=$GENERATIONS, batch_size=$BATCH, run_name='p4_smollm')"

echo "== buffer rows =="
python3 -m rl.conjecturer.cispo_train --dry-run   # trainable rows / groups (no model)

# ---- B. CISPO LoRA fine-tune (native macOS + MPS) ------------------------
echo "== Phase B: native venv + CISPO LoRA on $DEVICE_B =="
VENV=".venv_llm"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r rl/requirements.txt -r rl/requirements-llm.txt  # bitsandbytes is Linux-only (marker skips it)
fi
# PYTORCH_ENABLE_MPS_FALLBACK lets any op SmolLM/peft needs that MPS lacks (incl.
# some bf16 paths) fall back to CPU instead of crashing. If MPS still errors,
# re-run with RL_CONJ_DEVICE=cpu.
RL_CONJ_DEVICE="$DEVICE_B" RL_CONJ_MODEL="$MODEL" \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  "$VENV/bin/python" -m rl.conjecturer.cispo_train --epochs "$EPOCHS"

echo "== done: LoRA adapter -> rl/runs/conjecturer_lora/ =="
echo "Next generation's author auto-loads it. Re-run this script to alternate"
echo "buffer-collection (A) and CISPO updates (B) — that's the P4 curriculum loop."
