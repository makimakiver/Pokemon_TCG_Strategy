import numpy as np

from rl.train_solver import collect_rollouts, make_mcts_actor


class _FakeEnv:
    """Two-step search-mode episode; records actions; reward via _reward()."""
    mode = "search"

    def __init__(self):
        self._t = 0
        self.done = False
        self.actions = []

    def _obs(self):
        return {"global": np.zeros(2, np.float32),
                "options": np.zeros((2, 2), np.float32),
                "mask": np.array([1, 1, 1], np.float32), "n_options": 2}

    def reset(self, scenario=None):
        self._t = 0
        self.done = False
        self.actions = []
        return self._obs()

    def search_state(self):
        return object()

    def step(self, action):
        self.actions.append(action)
        self._t += 1
        self.done = self._t >= 2
        return self._obs(), 0.0, self.done, {}

    def _reward(self):
        return 1.0


class _FakeMCTS:
    def search_policy(self, root_state, temperature=1.0, greedy=False):
        # Deterministic: always pick option 1 with a 0.25/0.75 visit split.
        return 1, np.array([0.25, 0.75]), 0.5


class _Spec:
    target_id = "t_1"


def test_mcts_actor_records_visit_logp():
    env = _FakeEnv()
    actor = make_mcts_actor(_FakeMCTS(), policy=None)
    rollouts = collect_rollouts(policy=None, env=env, scenarios=[_Spec()],
                                k=1, prior_weight=0.0, actor=actor)
    assert len(rollouts) == 1
    r = rollouts[0]
    assert r.scenario_id == "t_1"
    assert r.win is True
    # behavior logp == log(pi[action]) == log(0.75)
    assert abs(r.steps[0]["logp"] - float(np.log(0.75))) < 1e-6
    assert env.actions == [1, 1]
