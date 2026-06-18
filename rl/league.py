"""OPTIONAL PSRO archive (plan §3.6) — layered on only if fixed-D saturates.

SGS trains the Solver on a fixed target set; this league is a *toggle*, not the
core algorithm. If the held-out win-rate plateaus and we want open competitive
strength, add a growing never-deleted archive of ``{deck, pilot}`` and sample
opponents with PFSP (prioritize ~50%-win-rate) × meta-share weighting, WITHOUT
changing the Solver. Kept minimal and off by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Opponent:
    deck_path: str
    pilot_module: str           # e.g. "agents.bare_agent", "agents.honchkrow"
    meta_share: float = 1.0
    games: int = 0
    solver_wins: int = 0

    def winrate(self) -> float:
        return self.solver_wins / self.games if self.games else 0.5


@dataclass
class League:
    archive: list[Opponent] = field(default_factory=list)
    enabled: bool = False

    def add(self, opp: Opponent):
        self.archive.append(opp)

    def pfsp_weights(self) -> list[float]:
        """Prioritized fictitious self-play: weight opponents the solver wins
        ~50% of the time the most, scaled by meta-share."""
        ws = []
        for o in self.archive:
            p = o.winrate()
            ws.append(max(1e-3, (1.0 - abs(p - 0.5) * 2.0)) * o.meta_share)
        z = sum(ws) or 1.0
        return [w / z for w in ws]

    def sample(self, rng) -> Opponent | None:
        if not self.archive:
            return None
        weights = self.pfsp_weights()
        r, acc = rng.random(), 0.0
        for o, w in zip(self.archive, weights):
            acc += w
            if r <= acc:
                return o
        return self.archive[-1]
