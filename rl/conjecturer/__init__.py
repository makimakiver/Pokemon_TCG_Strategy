"""Conjecturer: emits scenario edit-scripts that carve out sub-skills of a target.

Default = ``parametric.ParametricConjecturer`` (CPU, no GPU). The LLM upgrade
(``author``/``grpo_train``) is the optional P4 path and is intentionally left as
documented stubs.
"""
from __future__ import annotations

from .parametric import ParametricConjecturer

__all__ = ["ParametricConjecturer", "get_conjecturer"]


def get_conjecturer(name: str | None = None):
    from ..config import CONFIG
    name = name or CONFIG.conjecturer
    if name == "parametric":
        return ParametricConjecturer()
    if name == "llm":
        from .author import LLMConjecturer
        return LLMConjecturer()
    raise ValueError(f"unknown conjecturer '{name}'")
