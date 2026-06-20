"""Offline CISPO LoRA fine-tune of the LLM Conjecturer on R_synth (OPTIONAL P4).

The project objective is **CISPO** — the same one the Solver net uses
(``rl/solver_objectives.CISPO``), NOT GRPO. This module lifts that objective to the
sequence level for the LLM conjecturer and trains it OFFLINE from the replay buffer
``rl.sgs.conjecturer.author.LLMConjecturer`` wrote during an ``outer_loop`` generation:

  group by target prompt  ->  advantage  a = (R_synth - mean_g) / std_g
  per (prompt, completion):  lp = Σ log π_θ(token)            (current LoRA policy)
                             is_w = clamp(exp(lp - old_logp), max=cispo_clip).detach()
                             loss += - is_w · a.detach() · lp   - entropy_coef · H̄
  degenerate group (std_g < cispo_std_eps)  ->  REINFORCE½:  a = (R_synth - 0.5)

``old_logp`` is the BEHAVIOR sequence log-prob the author recorded at sampling time
(rows without it — parametric-fallback rows — are skipped). The two RL stacks never
share a GPU: the Solver and this conjecturer train in separate phases, so this runs
as its own pod step. Saves the LoRA adapter to ``CONFIG.conj_llm_lora_dir`` for the
next generation's ``author`` to load.

Run on the GPU pod (needs rl/requirements-llm.txt):
  docker run --rm --gpus all -v "$PWD":/app -w /app cabt-rl-llm \
    -m rl.sgs.conjecturer.cispo_train --epochs 2
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from rl.config import CONFIG


# --- buffer loading + CISPO group advantages (pure-python, host-testable) ----
def load_rows(path: str | None = None) -> list[dict]:
    path = path or CONFIG.conj_llm_buffer
    p = Path(path)
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    # only rows with a reward AND a behavior log-prob are trainable under CISPO
    return [r for r in rows if r.get("reward") is not None and r.get("old_logp") is not None]


def group_advantages(rows: list[dict]) -> list[dict]:
    """Attach CISPO group-normalized advantage to each row (groups = target prompt).

    Mirrors rl.solver_objectives.CISPO: a = (r - mean)/std per group; degenerate
    groups (std < cispo_std_eps) fall back to REINFORCE½ a = (r - 0.5).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["target_id"]].append(r)

    eps = CONFIG.cispo_std_eps
    out: list[dict] = []
    for sid, rs in groups.items():
        if len(rs) < CONFIG.conj_cispo_min_group:
            continue
        rewards = [float(r["reward"]) for r in rs]
        mean = sum(rewards) / len(rewards)
        var = sum((x - mean) ** 2 for x in rewards) / len(rewards)
        std = math.sqrt(var)
        for r, rw in zip(rs, rewards):
            if std < eps:
                adv = rw - 0.5                     # REINFORCE½ degenerate-group fallback
                r["_degenerate"] = True
            else:
                adv = (rw - mean) / (std + 1e-8)
                r["_degenerate"] = False
            r["_adv"] = adv
            out.append(r)
    return out


# --- the GPU training step (transformers/peft; pod only) ---------------------
def train_conjecturer_cispo(buffer_path: str | None = None, epochs: int | None = None):
    rows = group_advantages(load_rows(buffer_path))
    if not rows:
        raise SystemExit(
            "[cispo_train] no trainable rows (need reward + old_logp; parametric-"
            "fallback rows are skipped). Run an outer_loop generation with a real "
            "model (RL_CONJ=llm on the GPU pod) to populate the buffer first.")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise SystemExit(f"[cispo_train] LLM deps missing ({e}); install "
                         f"rl/requirements-llm.txt on the GPU pod.")

    epochs = epochs or CONFIG.conj_cispo_epochs
    name, device = CONFIG.conj_llm_model, CONFIG.conj_llm_device
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # bf16 only on CUDA; MPS/CPU use fp32 (bf16 on MPS falls back to CPU per-op and thrashes).
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    lora = LoraConfig(r=CONFIG.conj_lora_r, lora_alpha=CONFIG.conj_lora_alpha,
                      lora_dropout=CONFIG.conj_lora_dropout, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=CONFIG.conj_cispo_lr)
    clip = CONFIG.cispo_clip

    from rl.sgs.conjecturer.author import _SYSTEM   # same system prompt the buffer rows were sampled under

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def batch_logp_entropy(batch_rows):
        """Per-row Σ logπ_θ(completion) and mean entropy via ONE padded forward over
        the whole batch (vs one forward per row — fixes the MPS slowdown + memory hang
        that wedged the per-row loop on large buffers)."""
        seqs, spans = [], []
        for r in batch_rows:
            msgs = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": r["prompt"]}]
            pre = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            if hasattr(pre, "input_ids"):          # newer transformers returns a BatchEncoding
                pre = pre.input_ids
            comp = tok(r["completion"], return_tensors="pt", add_special_tokens=False).input_ids
            ids = torch.cat([pre, comp], dim=1)[0]              # [T]
            seqs.append(ids); spans.append((pre.shape[1], ids.shape[0]))   # (start, T)
        maxlen, B = max(s.shape[0] for s in seqs), len(seqs)
        input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        for b, s in enumerate(seqs):
            input_ids[b, :s.shape[0]] = s; attn[b, :s.shape[0]] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits   # [B, maxlen, V]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        lps, ents = [], []
        for b, (start, T) in enumerate(spans):
            if T <= start:                                      # no completion tokens
                z = torch.zeros((), device=device); lps.append(z); ents.append(z); continue
            pred = logprobs[b, start - 1:T - 1]                 # [L,V] dist predicting tokens start..T-1
            tgt = input_ids[b, start:T]                         # [L] actual completion tokens
            lps.append(pred.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum())
            ents.append((-(pred.exp() * pred).sum(-1)).mean())  # mean per-token entropy
        return lps, ents

    bs = CONFIG.conj_cispo_batch
    is_mps = str(device).startswith("mps")
    print(f"[cispo_train] {len(rows)} rows | {epochs} epochs | model {name} | clip {clip}")
    for ep in range(epochs):
        total, n, deg = 0.0, 0, 0
        for i in range(0, len(rows), bs):
            batch = rows[i:i + bs]
            opt.zero_grad()
            lps, ents = batch_logp_entropy(batch)               # ONE forward for the whole batch
            loss = torch.zeros((), device=device)
            for r, lp, ent in zip(batch, lps, ents):
                is_w = torch.clamp(torch.exp(lp.detach() - float(r["old_logp"])), max=clip)
                loss = loss - is_w * float(r["_adv"]) * lp - CONFIG.entropy_coef * ent
                deg += int(r["_degenerate"])
            loss = loss / max(1, len(batch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), CONFIG.grad_clip)
            opt.step()
            if is_mps:
                torch.mps.empty_cache()                         # bound memory growth across batches
            total += float(loss.detach()); n += 1
        print(f"[cispo_train] epoch {ep+1}/{epochs}  loss={total/max(1,n):+.4f}  "
              f"batches={n}  degenerate_rows={deg}")

    out_dir = Path(CONFIG.conj_llm_lora_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    print(f"[cispo_train] saved LoRA adapter -> {out_dir}")
    return str(out_dir)


def main():
    ap = argparse.ArgumentParser(description="Offline CISPO LoRA fine-tune of the conjecturer")
    ap.add_argument("--buffer", default=CONFIG.conj_llm_buffer)
    ap.add_argument("--epochs", type=int, default=CONFIG.conj_cispo_epochs)
    ap.add_argument("--dry-run", action="store_true",
                    help="load buffer + compute CISPO group advantages, no GPU/training")
    args = ap.parse_args()
    if args.dry_run:
        rows = group_advantages(load_rows(args.buffer))
        groups = len({r["target_id"] for r in rows})
        deg = sum(1 for r in rows if r["_degenerate"])
        print(f"[cispo_train:dry-run] trainable rows={len(rows)} groups={groups} "
              f"degenerate_rows={deg} (adv computed; no model loaded)")
        return
    train_conjecturer_cispo(args.buffer, args.epochs)


if __name__ == "__main__":
    main()
