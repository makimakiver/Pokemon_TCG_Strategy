"""Behavior-clone the conjecturer on VALID edit-scripts only (SFT).

Fix for the CISPO-on-garbage degradation (13.5% -> 0% valid): plain next-token
SFT on the well-formed `source:llm` completions teaches the structured format
directly, instead of RL-reinforcing the model's own 96% fallback failures.
Reuses the batched-forward pattern from cispo_train (MPS-safe, fp32, empty_cache).
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from rl.config import CONFIG
from rl.conjecturer.author import _SYSTEM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default="rl/runs/buffer_valid.jsonl")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="rl/runs/conjecturer_lora_sft")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    rows = [json.loads(l) for l in open(args.buffer) if l.strip()]
    print(f"[sft] {len(rows)} valid (prompt,completion) pairs | epochs={args.epochs} lr={args.lr}", flush=True)
    device, name = CONFIG.conj_llm_device, CONFIG.conj_llm_model
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    lora = LoraConfig(r=CONFIG.conj_lora_r, lora_alpha=CONFIG.conj_lora_alpha,
                      lora_dropout=CONFIG.conj_lora_dropout, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora); model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    is_mps = str(device).startswith("mps")
    bs = CONFIG.conj_cispo_batch

    def nll(batch):
        seqs, spans = [], []
        for r in batch:
            pre = tok.apply_chat_template([{"role": "system", "content": _SYSTEM},
                                           {"role": "user", "content": r["prompt"]}],
                                          add_generation_prompt=True, return_tensors="pt")
            if hasattr(pre, "input_ids"): pre = pre.input_ids
            comp = tok(r["completion"], return_tensors="pt", add_special_tokens=False).input_ids
            ids = torch.cat([pre, comp], dim=1)[0]; seqs.append(ids); spans.append((pre.shape[1], ids.shape[0]))
        maxlen, B = max(s.shape[0] for s in seqs), len(seqs)
        inp = torch.full((B, maxlen), pad_id, dtype=torch.long); attn = torch.zeros((B, maxlen), dtype=torch.long)
        for b, s in enumerate(seqs): inp[b, :s.shape[0]] = s; attn[b, :s.shape[0]] = 1
        inp, attn = inp.to(device), attn.to(device)
        lp = torch.log_softmax(model(input_ids=inp, attention_mask=attn).logits.float(), dim=-1)
        losses = []
        for b, (start, T) in enumerate(spans):
            if T <= start: continue
            pred = lp[b, start - 1:T - 1]; tgt = inp[b, start:T]
            losses.append(-pred.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean())  # mean token NLL
        return torch.stack(losses).mean()

    for ep in range(args.epochs):
        random.shuffle(rows); tot, n = 0.0, 0
        for i in range(0, len(rows), bs):
            batch = rows[i:i + bs]
            opt.zero_grad(); loss = nll(batch); loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), CONFIG.grad_clip)
            opt.step()
            if is_mps: torch.mps.empty_cache()
            tot += float(loss.detach()); n += 1
        print(f"[sft] epoch {ep+1}/{args.epochs}  nll={tot/max(1,n):.4f}", flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    print(f"[sft] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
