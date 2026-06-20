import json

from rl.core.scenario import ScenarioSpec, EditScript, EditOp
from rl.core.problem_set import (
    problem_row, write_problem_set, load_problem_set, seed_scenarios,
)


def _spec(target_id, your_deck=(10, 11, 12), opp_hand=(20, 21)):
    # Minimal obs that satisfies ScenarioSpec.my_index / apply(); no engine needed.
    obs = {"current": {"yourIndex": 0,
                       "players": [{"prize": [1, 2], "deckCount": 0, "handCount": 2,
                                    "active": [{"id": 99}]},
                                   {"prize": [1, 2], "deckCount": 0, "handCount": 2,
                                    "active": [{"id": 88}]}]}}
    return ScenarioSpec(
        obs=obs, your_deck=list(your_deck), your_prize=[],
        opponent_deck=[], opponent_prize=[], opponent_hand=list(opp_hand),
        opponent_active=[], source="t.json#step1", target_id=target_id)


def test_problem_row_shape():
    es = EditScript(ops=[EditOp(kind="stack_your_deck_top", card_id=11, slot=0)], budget=4)
    row = problem_row("t_1", "t.json#step1", "parametric", es)
    assert row["target_id"] == "t_1"
    assert row["backend"] == "parametric"
    assert row["edits"] == [{"kind": "stack_your_deck_top", "card_id": 11, "slot": 0}]


def test_write_then_load_roundtrip(tmp_path):
    es = EditScript(ops=[EditOp(kind="weaken_opponent_hand", card_id=20)], budget=4)
    rows = [problem_row("t_1", "t.json#step1", "parametric", es)]
    p = tmp_path / "problems.jsonl"
    write_problem_set(rows, p)
    loaded = load_problem_set(p)
    assert set(loaded) == {"t_1"}
    assert [op.kind for op in loaded["t_1"].ops] == ["weaken_opponent_hand"]
    assert loaded["t_1"].ops[0].card_id == 20


def test_seed_scenarios_applies_edits_and_falls_back(tmp_path):
    D = [_spec("t_1"), _spec("t_missing")]
    ps = {"t_1": EditScript(ops=[EditOp(kind="stack_your_deck_top", card_id=12)], budget=4)}
    seeded = seed_scenarios(D, ps)
    # t_1: card 12 moved to deck top.
    s1 = next(s for s in seeded if s.target_id == "t_1")
    assert s1.your_deck[0] == 12
    # t_missing: no edit -> identity (unchanged order).
    s2 = next(s for s in seeded if s.target_id == "t_missing")
    assert s2.your_deck == [10, 11, 12]


def test_load_skips_malformed_rows(tmp_path):
    p = tmp_path / "bad.jsonl"
    with open(p, "w") as f:
        f.write("not json\n")
        f.write(json.dumps({"no_target_id": True}) + "\n")
        f.write(json.dumps({"target_id": "ok", "edits": []}) + "\n")
    loaded = load_problem_set(p)
    assert set(loaded) == {"ok"}
    assert loaded["ok"].ops == []
