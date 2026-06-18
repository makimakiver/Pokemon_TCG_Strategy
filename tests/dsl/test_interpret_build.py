# tests/dsl/test_interpret_build.py
# Host-safe: verifies the score->picks selection logic without importing cg.
from rl.dsl.predicates import OptInfo, RuleContext, AttackInfo, score_options
from rl.dsl.predicates import pick_from_scores


def test_pick_from_scores_respects_min_max():
    scores = [5.0, 1.0, 9.0, 3.0]
    # maxCount 1 -> single best index
    assert pick_from_scores(scores, min_count=1, max_count=1) == [2]
    # maxCount 2 -> top two by score, descending
    assert pick_from_scores(scores, min_count=1, max_count=2) == [2, 0]
    # minCount 3 -> exactly 3 even if it wants fewer
    assert pick_from_scores(scores, min_count=3, max_count=4) == [2, 0, 3]


def test_pick_no_options():
    assert pick_from_scores([], min_count=0, max_count=0) == []
