"""Pure predicate registry + scorer over prebuilt dataclasses. NO cg import.

interpret.py (engine side) builds the dataclasses from a live observation; tests
build them directly. Each predicate is a pure fn of (RuleContext, OptInfo[, n]).
"""
from __future__ import annotations

from dataclasses import dataclass

from rl.shared.dsl.grammar import PREDICATE_ARITY


@dataclass
class CardInfo:
    is_pokemon: bool = False
    is_basic: bool = False
    is_stage1: bool = False
    is_stage2: bool = False
    is_ex: bool = False
    is_energy: bool = False
    is_item: bool = False
    is_supporter: bool = False
    is_tool: bool = False
    is_attacker: bool = False
    hp: int = 0
    energy_count: int = 0


@dataclass
class AttackInfo:
    damage: int = 0
    lethal: bool = False
    hits_weakness: bool = False
    affordable: bool = False


@dataclass
class OptInfo:
    kind: str = "other"              # attack|play_pokemon|play_item|play_supporter|
                                     # play_stadium|attach_energy|attach_tool|evolve|
                                     # ability|retreat|end|card_select|yes|no|number|other
    is_mine: bool = True
    target: CardInfo | None = None
    attack: AttackInfo | None = None


@dataclass
class RuleContext:
    my_prizes: int = 6
    opp_prizes: int = 6
    opp_active_hp: int = 999          # large sentinel when no opp active
    energy_unused: bool = True
    supporter_unused: bool = True
    bench_count: int = 0
    bench_has_attacker: bool = False
    hand_has_energy: bool = False
    my_active_at_risk: bool = False
    context: str = "main"            # setup|main|to_hand|attach_from|discard|switch|choose_count|other


def _t(opt):  # target or empty CardInfo
    return opt.target if opt.target is not None else CardInfo()


def _a(opt):  # attack or empty AttackInfo
    return opt.attack if opt.attack is not None else AttackInfo()


PREDICATES = {
    # option type
    "is_attack": lambda c, o: o.kind == "attack",
    "is_play_pokemon": lambda c, o: o.kind == "play_pokemon",
    "is_play_item": lambda c, o: o.kind == "play_item",
    "is_play_supporter": lambda c, o: o.kind == "play_supporter",
    "is_play_stadium": lambda c, o: o.kind == "play_stadium",
    "is_attach_energy": lambda c, o: o.kind == "attach_energy",
    "is_attach_tool": lambda c, o: o.kind == "attach_tool",
    "is_evolve": lambda c, o: o.kind == "evolve",
    "is_ability": lambda c, o: o.kind == "ability",
    "is_retreat": lambda c, o: o.kind == "retreat",
    "is_end": lambda c, o: o.kind == "end",
    "is_card_select": lambda c, o: o.kind == "card_select",
    "is_yes": lambda c, o: o.kind == "yes",
    "is_no": lambda c, o: o.kind == "no",
    "is_number": lambda c, o: o.kind == "number",
    # target card
    "target_is_pokemon": lambda c, o: _t(o).is_pokemon,
    "target_is_basic": lambda c, o: _t(o).is_basic,
    "target_is_stage1": lambda c, o: _t(o).is_stage1,
    "target_is_stage2": lambda c, o: _t(o).is_stage2,
    "target_is_ex": lambda c, o: _t(o).is_ex,
    "target_is_energy": lambda c, o: _t(o).is_energy,
    "target_is_item": lambda c, o: _t(o).is_item,
    "target_is_supporter": lambda c, o: _t(o).is_supporter,
    "target_is_tool": lambda c, o: _t(o).is_tool,
    "target_is_attacker": lambda c, o: _t(o).is_attacker,
    "target_is_mine": lambda c, o: o.is_mine,
    "target_is_opponent": lambda c, o: not o.is_mine,
    "target_hp_ge": lambda c, o, n: _t(o).hp >= n,
    "target_energy_ge": lambda c, o, n: _t(o).energy_count >= n,
    # attack
    "attack_is_lethal": lambda c, o: _a(o).lethal,
    "attack_hits_weakness": lambda c, o: _a(o).hits_weakness,
    "attack_damage_ge": lambda c, o, n: _a(o).damage >= n,
    "attack_affordable": lambda c, o: _a(o).affordable,
    # board / global
    "my_prizes_le": lambda c, o, n: c.my_prizes <= n,
    "opp_prizes_le": lambda c, o, n: c.opp_prizes <= n,
    "opp_active_hp_le": lambda c, o, n: c.opp_active_hp <= n,
    "my_active_at_risk": lambda c, o: c.my_active_at_risk,
    "bench_has_attacker": lambda c, o: c.bench_has_attacker,
    "hand_has_energy": lambda c, o: c.hand_has_energy,
    "energy_unused": lambda c, o: c.energy_unused,
    "supporter_unused": lambda c, o: c.supporter_unused,
    "bench_count_ge": lambda c, o, n: c.bench_count >= n,
    # context
    "ctx_setup": lambda c, o: c.context == "setup",
    "ctx_main": lambda c, o: c.context == "main",
    "ctx_to_hand": lambda c, o: c.context == "to_hand",
    "ctx_attach_from": lambda c, o: c.context == "attach_from",
    "ctx_discard": lambda c, o: c.context == "discard",
    "ctx_switch": lambda c, o: c.context == "switch",
    "ctx_choose_count": lambda c, o: c.context == "choose_count",
}
assert set(PREDICATES) == set(PREDICATE_ARITY), "predicate registry / arity mismatch"


def _eval(pred: dict, ctx: RuleContext, opt: OptInfo) -> bool:
    name = pred["pred"]
    fn = PREDICATES[name]
    try:
        return bool(fn(ctx, opt, pred["n"]) if PREDICATE_ARITY[name] == 1 else fn(ctx, opt))
    except Exception:
        return False


def rule_matches(rule: dict, ctx: RuleContext, opt: OptInfo) -> bool:
    return all(_eval(p, ctx, opt) for p in rule["when"])


def score_options(ruleset: dict, ctx: RuleContext, opts: list[OptInfo]) -> list[float]:
    base = float(ruleset.get("default_weight", 0.0))
    rules = ruleset.get("rules", [])
    scores = []
    for opt in opts:
        s = base
        for rule in rules:
            if rule_matches(rule, ctx, opt):
                s += float(rule["weight"])
        scores.append(s)
    return scores


def pick_from_scores(scores: list, min_count: int, max_count: int) -> list:
    """Indices chosen by descending score, honoring the engine's min/max.

    Matches agents/bare_agent selection: pick max(min_count, min(max_count, n))
    top-scored options; fall back to min_count if short. [] when no options.
    """
    n = len(scores)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    k = max(min_count, min(max_count, n))
    picks = order[:k]
    if len(picks) < min_count:
        picks = order[:min_count]
    return picks
