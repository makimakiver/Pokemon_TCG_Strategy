"""TCGEnv — single-seat self-play view of one cabt battle.

Two load paths, one RL interface:

* **live mode** (``reset(scenario=None)``): ``battle_start`` a fresh game from the
  solver deck vs the opponent deck. Turn-0 curriculum.
* **search mode** (``reset(scenario=spec)``): inject a mid-game position with
  ``search_begin`` and advance it with ``search_step``. This is the SGS scenario
  primitive — see ``scenario.py``.

The RL contract is **micro-stepped + masked**: the policy picks ONE option index
per ``step`` call. The env buffers picks for the current ``select`` and only
commits to the engine once ``minCount <= len(picks) <= maxCount`` (a STOP action
closes a variable-length select early). Between solver decisions the env
auto-plays the opponent seat and auto-resolves forced selects, so the learner
sees a clean single-agent MDP. Reward is terminal ±1 (0 for stalls/max_steps).

Engine constraint: ``Battle.battle_ptr`` is a process-global singleton
(cg/sim.py:67), so exactly one battle exists per process. ``reset`` always tears
the previous one down with ``battle_finish``; parallelism is subprocess workers
(``vec.py``). Runs only inside the Docker linux/amd64 image.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Optional

import numpy as np

from cg.api import to_observation_class, search_begin, search_step, search_release
from cg.game import battle_start, battle_select, battle_finish

from rl.config import CONFIG
from rl.shared.engine import encode
from rl.shared.engine.scenario import ScenarioSpec


STOP = "STOP"  # sentinel action: close a variable-length select


def _to_dict(obj):
    """Recursively convert an Observation dataclass (+ enums) back to a plain
    dict so a scripted opponent agent (which expects the raw obs dict) can run on
    search-mode observations. Live-mode observations are already dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, enum.Enum):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


class TCGEnv:
    def __init__(self, solver_deck: list[int], opponent_deck: list[int],
                 opponent_agent, solver_seat: int = 0, max_steps: Optional[int] = None):
        assert len(solver_deck) == 60 and len(opponent_deck) == 60
        self.solver_deck = list(solver_deck)
        self.opponent_deck = list(opponent_deck)
        self.opponent_agent = opponent_agent      # module/obj exposing .agent(obs_dict)
        self.solver_seat = solver_seat
        self.max_steps = max_steps or CONFIG.max_steps

        self.mode = None            # "live" | "search"
        self._search_id = None
        self._obs = None            # Observation dataclass (current decision)
        self._pending: list[int] = []
        self._steps = 0
        self._done = True
        self._result = -1

    # --- lifecycle ---------------------------------------------------------
    def reset(self, scenario: Optional[ScenarioSpec] = None):
        self._safe_finish()
        self._pending = []
        self._steps = 0
        self._done = False
        self._result = -1

        if scenario is None:
            self.mode = "live"
            obs_dict, start = battle_start(self.solver_deck, self.opponent_deck)
            if obs_dict is None:
                raise RuntimeError(
                    f"battle_start failed: errorPlayer={start.errorPlayer} "
                    f"errorType={start.errorType}")
            self._obs = to_observation_class(obs_dict)
        else:
            self.mode = "search"
            scenario.validate_shapes()
            src_obs = to_observation_class(scenario.obs)
            state = search_begin(src_obs, **scenario.search_begin_kwargs())
            self._search_id = state.searchId
            self._obs = state.observation

        self._advance_to_solver()
        return self._encode()

    def search_state(self):
        """The env's current engine search node, for an MCTS actor to root at.
        Valid only in search mode (mid-game scenarios loaded via search_begin)."""
        if self.mode != "search" or self._search_id is None:
            raise RuntimeError("search_state() is only valid in search mode")
        from types import SimpleNamespace
        return SimpleNamespace(searchId=self._search_id, observation=self._obs)

    def close(self):
        self._safe_finish()

    def _safe_finish(self):
        try:
            if self.mode == "search" and self._search_id is not None:
                search_release(self._search_id)
        except Exception:
            pass
        try:
            if self.mode == "live":
                battle_finish()
        except Exception:
            pass
        self._search_id = None

    # --- engine select application ----------------------------------------
    def _select(self, picks: list[int]):
        """Apply a committed selection to the engine; refresh self._obs."""
        if self.mode == "live":
            obs_dict = battle_select(picks)
            self._obs = to_observation_class(obs_dict)
        else:
            state = search_step(self._search_id, picks)
            self._search_id = state.searchId    # each step yields a NEW node id
            self._obs = state.observation

    def _result_now(self) -> int:
        st = self._obs.current if self._obs else None
        return st.result if st is not None else -1

    # --- auto-play opponent + forced selects ------------------------------
    def _advance_to_solver(self):
        """Advance the engine until the solver faces a real decision (or done)."""
        guard = 0
        while True:
            guard += 1
            if guard > 100000:
                raise RuntimeError("advance loop guard tripped")
            if self._obs is None or self._obs.current is None:
                self._finish(self._result_now())
                return
            if self._result_now() != -1:
                self._finish(self._result_now())
                return
            sel = self._obs.select
            if sel is None:
                self._finish(self._result_now())
                return

            who = self._obs.current.yourIndex
            # Forced (no real choice): resolve for either seat without an RL step.
            if self._is_forced(sel):
                self._select(self._forced_picks(sel))
                continue
            if who == self.solver_seat:
                return                       # hand control to the RL policy
            # Opponent's decision: let the scripted opponent choose the full list.
            picks = self._opponent_pick(sel)
            self._select(picks)

    @staticmethod
    def _is_forced(sel) -> bool:
        n = len(sel.option)
        return n <= 1 or sel.minCount >= n

    @staticmethod
    def _forced_picks(sel) -> list[int]:
        n = len(sel.option)
        if n == 0:
            return []
        return list(range(max(1, sel.minCount) if sel.minCount else min(1, n) or n))[:max(sel.minCount, 1)] \
            if sel.minCount >= n else [0]

    def _opponent_pick(self, sel) -> list[int]:
        obs_dict = _to_dict(self._obs)
        try:
            action = self.opponent_agent.agent(obs_dict)
        except Exception:
            action = list(range(sel.minCount or 1))
        if not isinstance(action, list):
            action = list(action)
        # Clamp to legal length.
        action = [a for a in action if 0 <= a < len(sel.option)]
        if len(action) < sel.minCount:
            extra = [i for i in range(len(sel.option)) if i not in action]
            action += extra[: sel.minCount - len(action)]
        return action[: sel.maxCount] if sel.maxCount else action

    # --- RL step -----------------------------------------------------------
    def step(self, action):
        """Micro-step. ``action`` is an option index in [0, N) or ``STOP``."""
        if self._done:
            raise RuntimeError("step() after done; call reset()")
        sel = self._obs.select
        n = len(sel.option)

        if action == STOP or (isinstance(action, int) and action >= n):
            if len(self._pending) < sel.minCount:
                # Illegal STOP -> pick the first unused legal option instead.
                action = next((i for i in range(n) if i not in self._pending), 0)
            else:
                return self._commit(sel)

        if action in self._pending:                # duplicate -> ignore, no-op step
            return self._encode(), 0.0, False, {"noop": True}
        self._pending.append(int(action))
        if len(self._pending) >= sel.maxCount:
            return self._commit(sel)
        # Still building a multi-pick selection; same select, updated mask.
        return self._encode(), 0.0, False, {"building": True}

    def _commit(self, sel):
        picks = list(self._pending)
        self._pending = []
        self._select(picks)
        self._advance_to_solver()
        self._steps += 1
        if self._done:
            return self._encode(), float(self._reward()), True, {"result": self._result}
        if self._steps >= self.max_steps:
            self._finish(-1)               # stall -> draw -> 0 reward
            return self._encode(), 0.0, True, {"result": -1, "stall": True}
        return self._encode(), 0.0, False, {}

    def _finish(self, result: int):
        self._result = result
        self._done = True

    def _reward(self) -> float:
        if self._result == self.solver_seat:
            return 1.0
        if self._result == 1 - self.solver_seat:
            return -1.0
        return 0.0                          # draw / stall

    # --- observation encoding ---------------------------------------------
    def legal_mask(self):
        """Mask over [options..., STOP]. STOP legal iff minCount satisfied."""
        if self._done or self._obs is None or self._obs.select is None:
            return np.zeros(1, np.float32)
        n = len(self._obs.select.option)
        m = np.ones(n + 1, np.float32)
        for i in self._pending:
            m[i] = 0.0
        m[n] = 1.0 if len(self._pending) >= self._obs.select.minCount else 0.0
        return m

    def _encode(self):
        if self._done or self._obs is None or self._obs.select is None:
            G = np.zeros(CONFIG.global_feat_dim, np.float32)
            return {"global": G,
                    "options": np.zeros((0, CONFIG.option_feat_dim), np.float32),
                    "mask": np.zeros(1, np.float32),
                    "n_options": 0}
        g, opts, _ = encode.featurize(self._obs)
        return {"global": g, "options": opts, "mask": self.legal_mask(),
                "n_options": len(self._obs.select.option)}

    @property
    def done(self) -> bool:
        return self._done

    @property
    def result(self) -> int:
        return self._result
