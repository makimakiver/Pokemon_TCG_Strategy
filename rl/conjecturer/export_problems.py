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
