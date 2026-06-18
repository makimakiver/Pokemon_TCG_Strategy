from rl.dsl.predicates import (
    CardInfo, AttackInfo, OptInfo, RuleContext, PREDICATES, score_options,
)

def ctx(**kw):
    base = dict(my_prizes=6, opp_prizes=6, opp_active_hp=120, energy_unused=True,
                supporter_unused=True, bench_count=2, bench_has_attacker=True,
                hand_has_energy=True, my_active_at_risk=False, context="main")
    base.update(kw)
    return RuleContext(**base)

def opt(kind="end", is_mine=True, target=None, attack=None):
    return OptInfo(kind=kind, is_mine=is_mine, target=target, attack=attack)

def test_is_attack_predicate():
    assert PREDICATES["is_attack"](ctx(), opt(kind="attack"))
    assert not PREDICATES["is_attack"](ctx(), opt(kind="end"))

def test_attack_lethal_and_param_predicates():
    a = AttackInfo(damage=120, lethal=True, hits_weakness=False, affordable=True)
    o = opt(kind="attack", attack=a)
    assert PREDICATES["attack_is_lethal"](ctx(), o)
    assert PREDICATES["attack_damage_ge"](ctx(), o, 100)
    assert not PREDICATES["attack_damage_ge"](ctx(), o, 130)

def test_target_and_board_predicates():
    c = CardInfo(is_pokemon=True, is_ex=True, is_attacker=True, hp=180, energy_count=2)
    o = opt(kind="attach_energy", target=c)
    assert PREDICATES["target_is_ex"](ctx(), o)
    assert PREDICATES["target_hp_ge"](ctx(), o, 150)
    assert PREDICATES["my_prizes_le"](ctx(my_prizes=1), o, 2)
    assert not PREDICATES["my_prizes_le"](ctx(my_prizes=3), o, 2)

def test_score_options_sums_matching_rules():
    rs = {"default_weight": 1.0, "rules": [
        {"when": [{"pred": "is_attack"}], "weight": 10},
        {"when": [{"pred": "is_attack"}, {"pred": "attack_is_lethal"}], "weight": 100},
    ]}
    a = AttackInfo(damage=200, lethal=True, hits_weakness=False, affordable=True)
    opts = [opt(kind="attack", attack=a), opt(kind="end")]
    scores = score_options(rs, ctx(), opts)
    assert scores[0] == 1.0 + 10 + 100   # attack+lethal
    assert scores[1] == 1.0              # end: only default
