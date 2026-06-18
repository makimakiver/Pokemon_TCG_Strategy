from rl.dsl.grammar import validate, PREDICATE_ARITY

GOOD = {
    "name": "x", "default_weight": 0.0,
    "rules": [
        {"when": [{"pred": "is_attack"}, {"pred": "attack_is_lethal"}], "weight": 800},
        {"when": [{"pred": "target_hp_ge", "n": 100}], "weight": 50},
    ],
}

def test_valid_ruleset_passes():
    ok, errors = validate(GOOD)
    assert ok and errors == []

def test_unknown_predicate_rejected():
    bad = {"rules": [{"when": [{"pred": "nope"}], "weight": 1}]}
    ok, errors = validate(bad)
    assert not ok and any("nope" in e for e in errors)

def test_missing_required_n_rejected():
    bad = {"rules": [{"when": [{"pred": "target_hp_ge"}], "weight": 1}]}
    ok, errors = validate(bad)
    assert not ok and any("target_hp_ge" in e for e in errors)

def test_weight_out_of_range_rejected():
    bad = {"rules": [{"when": [{"pred": "is_end"}], "weight": 10**9}]}
    ok, errors = validate(bad)
    assert not ok

def test_too_many_rules_rejected():
    bad = {"rules": [{"when": [{"pred": "is_end"}], "weight": 1}] * 41}
    ok, errors = validate(bad)
    assert not ok

def test_bool_weight_rejected():
    # Python bool is an int subclass; reject it explicitly for parity with `n`.
    bad = {"rules": [{"when": [{"pred": "is_end"}], "weight": True}]}
    ok, errors = validate(bad)
    assert not ok and any("weight" in e for e in errors)
