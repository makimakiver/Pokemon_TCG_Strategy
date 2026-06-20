"""AlphaZero objective: dense (pi, z) signal that does not starve on low-variance
groups (unlike REINFORCE-half / CISPO). Pure-Python (real PointerPolicy, no engine)."""
import numpy as np
import torch

from rl.shared.policy import PointerPolicy
from rl.sgs.solver_objectives import get_objective, Rollout


def _obs(n=3):
    return {"global": np.zeros(24, np.float32),
            "options": np.zeros((n, 32), np.float32),
            "mask": np.ones(n + 1, np.float32)}


def _step(n=3, action=0, pi=None):
    if pi is None:
        pi = np.ones(n + 1) / (n + 1)
    return {"obs": _obs(n), "action": action, "logp": -1.0, "value": 0.0, "pi": pi}


def test_alphazero_loss_finite_and_nonzero_gradient_on_all_loss_group():
    # A group where EVERY rollout loses (reward -1): REINFORCE-half/CISPO give zero
    # gradient here, but AlphaZero must still learn (value target + pi CE).
    pol = PointerPolicy()
    R = [Rollout("A", [_step(3, 1)], -1.0, False),
         Rollout("A", [_step(3, 2)], -1.0, False)]
    az = get_objective("alphazero")
    loss, m = az.compute_loss(pol, R, prior_weight=0.0)
    assert torch.isfinite(loss), loss
    assert m["used"] == 2
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in pol.parameters() if p.grad is not None)
    assert gnorm > 0.0, "AlphaZero produced a zero gradient on an all-loss group"


def test_alphazero_masked_pi_has_no_nan():
    # Target pi puts 0 on the (masked) STOP slot; CE must stay finite (no 0*-inf).
    pol = PointerPolicy()
    s = _step(2, 0, pi=np.array([0.5, 0.5, 0.0]))   # STOP target 0
    s["obs"]["mask"] = np.array([1, 1, 0], np.float32)  # STOP illegal
    loss, m = get_objective("alphazero").compute_loss(
        pol, [Rollout("A", [s], 1.0, True)], prior_weight=0.0)
    assert torch.isfinite(loss), loss
