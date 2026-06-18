"""Inference-time MCTS over the engine's search tree (plan §1, P1).

The engine exposes exactly the primitive MCTS needs: ``search_begin`` injects a
root position and ``search_step(search_id, picks)`` returns a *child* search state
with its own id — so siblings are obtained by stepping the same parent with
different picks, and ``search_release`` frees a subtree. We run PUCT with the
PointerPolicy as the prior and its value head as the leaf evaluator. Opponent
decision nodes inside the tree are filled by the scripted opponent (the scenario's
anchor pilot), matching how the opponent is modelled during training.

This is OFF during training (``CONFIG.mcts_enabled``) and ON at submission, where
it sharpens the trained net. Variable-length selects are handled by treating each
engine select as a node and each legal option index as an edge (STOP is an extra
edge when the select's min is already met). Runs in the Docker engine image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cg.api import search_step, search_release
from . import encode
from .config import CONFIG


@dataclass
class _Node:
    search_id: int
    obs: object                     # Observation dataclass for this node
    pending: tuple = ()             # picks buffered for a multi-select
    children: dict = field(default_factory=dict)   # edge(int) -> _Node
    N: dict = field(default_factory=dict)          # edge -> visit count
    W: dict = field(default_factory=dict)          # edge -> total value
    P: dict = field(default_factory=dict)          # edge -> prior
    expanded: bool = False
    terminal_value: float | None = None


def _result_value(obs, solver_seat) -> float | None:
    st = getattr(obs, "current", None)
    if st is None or st.result == -1:
        return None
    return 1.0 if st.result == solver_seat else -1.0 if st.result == 1 - solver_seat else 0.0


class MCTS:
    def __init__(self, policy, opponent_agent, solver_seat=0, cfg=CONFIG):
        self.policy = policy
        self.opponent_agent = opponent_agent
        self.solver_seat = solver_seat
        self.cfg = cfg

    # --- public: choose an action at the env's current solver decision ------
    def search(self, root_state) -> int:
        """Run N simulations from ``root_state`` (a SearchState) and return the
        most-visited legal option index at the root."""
        root = _Node(root_state.searchId, root_state.observation)
        self._expand(root)
        for _ in range(self.cfg.mcts_simulations):
            self._simulate(root)
        # Greedy by visit count over root edges.
        if not root.N:
            return 0
        return max(root.N, key=root.N.get)

    # --- internals ----------------------------------------------------------
    def _priors(self, obs):
        g, opts, _ = encode.featurize(obs)
        import torch
        with torch.no_grad():
            mask = np.ones(len(opts) + 1, np.float32)
            logits, value = self.policy.forward(
                torch.as_tensor(g), torch.as_tensor(opts), torch.as_tensor(mask), 0.0)
            p = torch.softmax(logits, dim=0).cpu().numpy()
        return p, float(value)

    def _expand(self, node: _Node):
        node.terminal_value = _result_value(node.obs, self.solver_seat)
        if node.terminal_value is not None:
            node.expanded = True
            return 0.0
        sel = node.obs.select
        n = len(sel.option) if sel else 0
        priors, value = self._priors(node.obs)
        for e in range(n):
            node.P[e] = float(priors[e]) if e < len(priors) else 1.0 / max(1, n)
            node.N[e] = 0
            node.W[e] = 0.0
        node.expanded = True
        return value

    def _simulate(self, node: _Node) -> float:
        if node.terminal_value is not None:
            return node.terminal_value
        sel = node.obs.select
        if sel is None:
            return 0.0
        # PUCT select.
        total_N = sum(node.N.values()) + 1
        best_edge, best_u = None, -1e9
        for e in node.N:
            q = node.W[e] / node.N[e] if node.N[e] > 0 else 0.0
            u = q + self.cfg.mcts_c_puct * node.P[e] * math.sqrt(total_N) / (1 + node.N[e])
            if u > best_u:
                best_u, best_edge = u, e
        if best_edge is None:
            return 0.0

        child = node.children.get(best_edge)
        if child is None:
            picks = list(node.pending) + [best_edge]
            # Commit when the select is satisfied; else keep buffering on a child.
            if len(picks) >= sel.minCount and len(picks) >= 1:
                next_state = search_step(node.search_id, picks)
                child_obs = next_state.observation
                child = _Node(next_state.searchId, child_obs)
                child = self._advance_opponent(child)
                self._expand(child)
            else:
                child = _Node(node.search_id, node.obs, pending=tuple(picks))
                self._expand_inherit(node, child)
            node.children[best_edge] = child

        value = self._simulate(child)
        node.N[best_edge] += 1
        node.W[best_edge] += value
        return value

    def _expand_inherit(self, parent: _Node, child: _Node):
        # Mid-multiselect: same engine node, fewer legal edges (drop used picks).
        sel = parent.obs.select
        n = len(sel.option)
        priors, _ = self._priors(parent.obs)
        for e in range(n):
            if e in child.pending:
                continue
            child.P[e] = float(priors[e]) if e < len(priors) else 1.0 / n
            child.N[e] = 0
            child.W[e] = 0.0
        child.expanded = True

    def _advance_opponent(self, node: _Node) -> _Node:
        """Step the scripted opponent (and forced selects) until it's the solver's
        turn or the game ends — mirroring TCGEnv._advance_to_solver inside search."""
        import dataclasses
        from .env import _to_dict
        guard = 0
        while True:
            guard += 1
            if guard > 10000:
                break
            if _result_value(node.obs, self.solver_seat) is not None:
                break
            sel = node.obs.select
            if sel is None:
                break
            if len(sel.option) <= 1 or sel.minCount >= len(sel.option):
                picks = [0] if len(sel.option) else []
                if sel.minCount >= len(sel.option):
                    picks = list(range(len(sel.option)))[:max(sel.minCount, 1)]
                state = search_step(node.search_id, picks)
                node = _Node(state.searchId, state.observation)
                continue
            if node.obs.current.yourIndex == self.solver_seat:
                break
            obs_dict = _to_dict(node.obs)
            try:
                action = self.opponent_agent.agent(obs_dict)
            except Exception:
                action = list(range(sel.minCount or 1))
            action = [a for a in action if 0 <= a < len(sel.option)][: sel.maxCount or 1]
            state = search_step(node.search_id, action or [0])
            node = _Node(state.searchId, state.observation)
        return node
