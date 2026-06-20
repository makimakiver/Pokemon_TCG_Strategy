"""DEPRECATED — the conjecturer fine-tune uses CISPO, not GRPO.

The plan doc (§3.4 / §5 P4) said "LoRA + GRPO", but this project's objective is
**CISPO across the whole stack** (Solver net AND conjecturer); the real trainer is
``rl/conjecturer/cispo_train.py``. This shim only redirects so old references don't
silently run the wrong objective.
"""
from __future__ import annotations

from rl.conjecturer.cispo_train import train_conjecturer_cispo


def train_conjecturer_grpo(*_, **__):
    raise NotImplementedError(
        "GRPO is not used here — the conjecturer fine-tune is CISPO. "
        "Use rl.conjecturer.cispo_train.train_conjecturer_cispo "
        "(or `python -m rl.conjecturer.cispo_train`).")
