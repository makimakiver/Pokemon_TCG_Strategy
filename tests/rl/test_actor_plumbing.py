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


class _MaskedEnv(_FakeEnv):
    """Option 0 is masked (e.g. already picked mid multi-select); only option 1
    (and STOP) are legal."""
    def _obs(self):
        return {"global": np.zeros(2, np.float32),
                "options": np.zeros((2, 2), np.float32),
                "mask": np.array([0, 1, 1], np.float32), "n_options": 2}


class _MaskedPrefMCTS:
    def search_policy(self, root_state, temperature=1.0, greedy=False):
        # MCTS visit-argmax is the MASKED option 0; option 1 carries some mass.
        return 0, np.array([0.9, 0.1]), 0.5


class _AllMaskedMassMCTS:
    def search_policy(self, root_state, temperature=1.0, greedy=False):
        # ALL MCTS mass sits on the masked option 0.
        return 0, np.array([1.0, 0.0]), 0.5


class _FakePolicy:
    def act(self, obs, prior_weight):
        # Masked-aware fallback: pick the first legal option.
        mask, n = obs["mask"], obs["n_options"]
        a = next(i for i in range(n) if mask[i] > 0.5)
        return a, -0.5, 0.1


def test_mcts_actor_never_returns_a_masked_option():
    # Regression: MCTS visit-argmax is illegal (option 0 masked). The actor must
    # mask+renormalize and return the LEGAL option 1 (not the masked 0). An illegal
    # action would re-eval to log_prob=-inf and poison the loss (NaN / +inf).
    env = _MaskedEnv()
    actor = make_mcts_actor(_MaskedPrefMCTS(), policy=None)
    rollouts = collect_rollouts(policy=None, env=env, scenarios=[_Spec()],
                                k=1, prior_weight=0.0, actor=actor)
    r = rollouts[0]
    assert env.actions == [1, 1]                       # NOT the masked option 0
    # legal_pi = [0, 0.1] -> renormalized [0, 1] -> logp = log(1.0) = 0
    assert abs(r.steps[0]["logp"] - 0.0) < 1e-6


def test_mcts_actor_defers_to_policy_when_no_legal_mcts_mass():
    # When all MCTS mass sits on the (illegal) masked option, defer to the masked
    # net policy, which always returns a legal action.
    env = _MaskedEnv()
    actor = make_mcts_actor(_AllMaskedMassMCTS(), policy=_FakePolicy())
    rollouts = collect_rollouts(policy=_FakePolicy(), env=env, scenarios=[_Spec()],
                                k=1, prior_weight=0.0, actor=actor)
    r = rollouts[0]
    assert env.actions == [1, 1]                       # policy.act picked legal option 1
    assert abs(r.steps[0]["logp"] - (-0.5)) < 1e-6     # logp from policy.act
