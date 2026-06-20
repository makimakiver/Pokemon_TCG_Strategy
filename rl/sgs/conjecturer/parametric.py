"""Parametric edit-script policy (the default Conjecturer, plan §3.4).

A small categorical policy over edit-script *templates*, trained with REINFORCE on
``R_synth = R_solve · R_guide``. No GPU: it's a handful of logits over a fixed
template set, updated with a plain policy-gradient step. Given a target scenario
it samples an edit-script that weakens one assumption (so the position becomes an
easier lemma), conditioned on the target — never an unconditional rewrite (the
paper's "No Problem Conditioning" ablation showed unconditional generation gains
nothing).

The action space is deliberately tiny and tied to what ``search_begin`` can
actually express (see ``scenario.py``): predict a weak opponent hand, set the
face-down active to a fragile basic, or stack our deck top toward a known answer.
"""
from __future__ import annotations

import math

from rl.config import CONFIG
from rl.shared.engine.scenario import ScenarioSpec, EditScript, EditOp


# Edit-script templates. Each maps (target -> list[EditOp]); card ids are filled
# from the target's own predicted pools so edits stay on-distribution.
TEMPLATES = [
    "noop",                  # identity (the empty lemma; baseline difficulty)
    "weaken_opp_hand",       # opponent predicted to hold a benign card -> no answer
    "fragile_active",        # face-down opp active predicted as a low-HP basic
    "stack_our_top",         # move a strong attacker prediction to our deck top
    "weaken_and_stack",      # combine the two most useful weakenings
]


def _benign_card(spec: ScenarioSpec) -> int:
    """Pick a low-impact id the opponent could 'hold' to defang their hand.
    Prefers a basic energy id already in their predicted pool."""
    pool = spec.opponent_hand + spec.opponent_deck
    return pool[0] if pool else 0


def _fragile_basic(spec: ScenarioSpec) -> int:
    return spec.opponent_active[0] if spec.opponent_active else 0


def _our_top_card(spec: ScenarioSpec) -> int:
    return spec.your_deck[0] if spec.your_deck else 0


def _template_to_edits(name: str, spec: ScenarioSpec) -> EditScript:
    ops: list[EditOp] = []
    if name == "weaken_opp_hand":
        ops = [EditOp("weaken_opponent_hand", card_id=_benign_card(spec))]
    elif name == "fragile_active":
        if spec.opponent_active:
            ops = [EditOp("set_opponent_active", card_id=_fragile_basic(spec))]
    elif name == "stack_our_top":
        ops = [EditOp("stack_your_deck_top", card_id=_our_top_card(spec))]
    elif name == "weaken_and_stack":
        ops = [EditOp("weaken_opponent_hand", card_id=_benign_card(spec)),
               EditOp("stack_your_deck_top", card_id=_our_top_card(spec))]
    return EditScript(ops=ops, budget=CONFIG.edit_budget)


class ParametricConjecturer:
    def __init__(self, lr: float = 0.05):
        self.logits = [0.0] * len(TEMPLATES)
        self.lr = lr
        self._last = {}    # template_index -> chosen, for the REINFORCE update

    def _probs(self) -> list[float]:
        m = max(self.logits)
        e = [math.exp(x - m) for x in self.logits]
        z = sum(e)
        return [x / z for x in e]

    def propose(self, target: ScenarioSpec, rng) -> tuple[ScenarioSpec, EditScript, int]:
        """Sample an edit-script for ``target``; return (edited_spec, edits, idx)."""
        probs = self._probs()
        r, acc, idx = rng.random(), 0.0, len(probs) - 1
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                idx = i
                break
        edits = _template_to_edits(TEMPLATES[idx], target)
        edited = edits.apply(target)
        return edited, edits, idx

    def update(self, idx: int, r_synth: float) -> None:
        """REINFORCE on the template logits: ∇ = (R - baseline) · ∇logπ(idx)."""
        probs = self._probs()
        baseline = 0.5
        adv = r_synth - baseline
        for i in range(len(self.logits)):
            grad = (1.0 if i == idx else 0.0) - probs[i]
            self.logits[i] += self.lr * adv * grad

    def snapshot(self) -> dict:
        return {"templates": TEMPLATES, "logits": list(self.logits),
                "probs": self._probs()}
