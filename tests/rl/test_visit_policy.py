import numpy as np

from rl.sgs.mcts import visit_policy


def test_visit_policy_normalizes_over_options():
    pi = visit_policy({0: 3, 1: 1}, n_options=2)
    assert np.allclose(pi, [0.75, 0.25])
    assert abs(pi.sum() - 1.0) < 1e-9


def test_visit_policy_ignores_out_of_range_edges():
    # A STOP edge (== n_options) must not appear in the option distribution.
    pi = visit_policy({0: 2, 2: 5}, n_options=2)
    assert np.allclose(pi, [1.0, 0.0])


def test_visit_policy_zero_visits_is_uniform_zero():
    pi = visit_policy({}, n_options=3)
    assert pi.shape == (3,)
    assert pi.sum() == 0.0


def test_visit_policy_temperature_sharpens():
    cold = visit_policy({0: 3, 1: 1}, n_options=2, temperature=0.5)
    # τ<1 sharpens toward the most-visited edge.
    assert cold[0] > 0.75
