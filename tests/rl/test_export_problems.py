import random

from rl.scenario import ScenarioSpec, EditScript, EditOp
from rl.conjecturer.export_problems import export_problems


class _StubConj:
    def propose(self, target, rng):
        es = EditScript(ops=[EditOp(kind="weaken_opponent_hand", card_id=20)])
        return es.apply(target), es, 0


def _spec(tid):
    obs = {"current": {"yourIndex": 0,
                       "players": [{"prize": [1, 2], "deckCount": 0, "handCount": 1,
                                    "active": [{"id": 99}]},
                                   {"prize": [1, 2], "deckCount": 0, "handCount": 1,
                                    "active": [{"id": 88}]}]}}
    return ScenarioSpec(obs=obs, your_deck=[10], your_prize=[], opponent_deck=[],
                        opponent_prize=[], opponent_hand=[20], opponent_active=[],
                        source=f"{tid}.json#step1", target_id=tid)


def test_export_produces_one_row_per_target():
    D = [_spec("t_1"), _spec("t_2")]
    rows = export_problems(_StubConj(), D, random.Random(0), backend="parametric")
    assert len(rows) == 2
    assert {r["target_id"] for r in rows} == {"t_1", "t_2"}
    assert rows[0]["backend"] == "parametric"
    assert rows[0]["edits"][0]["kind"] == "weaken_opponent_hand"
