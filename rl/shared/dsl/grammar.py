"""Constrained-DSL validation (the legality gate). Pure Python, NO cg import."""
from __future__ import annotations

# Predicate name -> arity (0 = no arg, 1 = needs integer "n"). Mirrors spec §4.2.
PREDICATE_ARITY: dict[str, int] = {
    # option type
    "is_attack": 0, "is_play_pokemon": 0, "is_play_item": 0, "is_play_supporter": 0,
    "is_play_stadium": 0, "is_attach_energy": 0, "is_attach_tool": 0, "is_evolve": 0,
    "is_ability": 0, "is_retreat": 0, "is_end": 0, "is_card_select": 0,
    "is_yes": 0, "is_no": 0, "is_number": 0,
    # target card
    "target_is_pokemon": 0, "target_is_basic": 0, "target_is_stage1": 0,
    "target_is_stage2": 0, "target_is_ex": 0, "target_is_energy": 0,
    "target_is_item": 0, "target_is_supporter": 0, "target_is_tool": 0,
    "target_is_attacker": 0, "target_is_mine": 0, "target_is_opponent": 0,
    "target_hp_ge": 1, "target_energy_ge": 1,
    # attack
    "attack_is_lethal": 0, "attack_hits_weakness": 0, "attack_damage_ge": 1,
    "attack_affordable": 0,
    # board / global
    "my_prizes_le": 1, "opp_prizes_le": 1, "opp_active_hp_le": 1,
    "my_active_at_risk": 0, "bench_has_attacker": 0, "hand_has_energy": 0,
    "energy_unused": 0, "supporter_unused": 0, "bench_count_ge": 1,
    # context
    "ctx_setup": 0, "ctx_main": 0, "ctx_to_hand": 0, "ctx_attach_from": 0,
    "ctx_discard": 0, "ctx_switch": 0, "ctx_choose_count": 0,
}

MAX_RULES = 40
MAX_WHEN = 6
WEIGHT_MIN, WEIGHT_MAX = -100000, 100000
N_MIN, N_MAX = 0, 1000


def validate(ruleset: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(ruleset, dict):
        return False, ["ruleset must be a JSON object"]
    dw = ruleset.get("default_weight")
    if dw is not None and (isinstance(dw, bool) or not isinstance(dw, (int, float))):
        errors.append("default_weight must be a number")
    rules = ruleset.get("rules")
    if not isinstance(rules, list):
        return False, ["ruleset.rules must be a list"]
    if len(rules) > MAX_RULES:
        errors.append(f"too many rules: {len(rules)} > {MAX_RULES}")
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"rule[{i}] must be an object"); continue
        w = rule.get("weight")
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            errors.append(f"rule[{i}].weight must be a number")
        elif not (WEIGHT_MIN <= w <= WEIGHT_MAX):
            errors.append(f"rule[{i}].weight {w} out of range")
        when = rule.get("when")
        if not isinstance(when, list) or not (1 <= len(when) <= MAX_WHEN):
            errors.append(f"rule[{i}].when must be a list of 1..{MAX_WHEN} predicates"); continue
        for j, p in enumerate(when):
            if not isinstance(p, dict) or "pred" not in p:
                errors.append(f"rule[{i}].when[{j}] must have a 'pred'"); continue
            name = p["pred"]
            if name not in PREDICATE_ARITY:
                errors.append(f"rule[{i}].when[{j}] unknown predicate '{name}'"); continue
            if PREDICATE_ARITY[name] == 1:
                n = p.get("n")
                if not isinstance(n, int) or isinstance(n, bool) or not (N_MIN <= n <= N_MAX):
                    errors.append(f"rule[{i}].when[{j}] '{name}' needs integer n in [{N_MIN},{N_MAX}]")
            elif "n" in p:
                errors.append(f"rule[{i}].when[{j}] '{name}' takes no n")
    return (len(errors) == 0), errors
