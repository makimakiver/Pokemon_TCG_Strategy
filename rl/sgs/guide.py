"""Guide ρ — keeps conjectured scenarios clean and non-degenerate.

``R_guide = legality × non_degeneracy × relevance`` mapped to [0, 1] (plan §3.5).

v1 is rule-based (default). The LLM upgrade path (a semantic elegance judge,
paper §E.3) is sketched in ``llm_guide_score`` and triggered when collapse
symptoms appear (auto-win rate rising / relevance-spread shrinking).

The legality term is the only part that touches the engine, and only lazily:
scenario legality is ultimately enforced by ``search_begin`` raising in
``env.reset``; the cheap structural checks here let the Conjecturer prune
obviously-degenerate edits before paying for a rollout.
"""
from __future__ import annotations

from rl.config import CONFIG
from rl.shared.engine.scenario import ScenarioSpec, EditScript


def non_degeneracy(spec: ScenarioSpec) -> float:
    """1.0 if the edited position is a real, two-sided game; lower if lopsided.

    Degenerate == auto-win by construction: opponent has no active, or a prize
    count that means the game is already decided.
    """
    st = spec.obs["current"]
    me = st["players"][spec.my_index]
    op = st["players"][1 - spec.my_index]
    score = 1.0
    op_active = op.get("active") or []
    has_op_active = len(op_active) > 0 and (op_active[0] is not None or spec.opponent_active)
    if not has_op_active:
        return 0.0
    if len(op["prize"]) == 0 or len(me["prize"]) == 0:
        return 0.0                       # someone is on their last prize -> trivial
    # Mild penalty for very lopsided prize races (still a game, but easy).
    diff = abs(len(me["prize"]) - len(op["prize"]))
    score *= max(0.3, 1.0 - 0.15 * diff)
    return score


def relevance(edits: EditScript) -> float:
    """Reward minimal, targeted edit-scripts over wholesale rewrites.

    Edit-distance from the empty edit, bounded by the budget. A lemma should
    weaken ONE assumption, not rebuild the position.
    """
    k = edits.size()
    budget = max(1, edits.budget)
    return max(0.0, 1.0 - k / (budget + 1))


def guide_score(spec: ScenarioSpec, edits: EditScript, legal: bool = True) -> float:
    """Rule-based R_guide in [0, 1]. ``legal`` comes from whether search_begin
    accepted the compiled scenario (env.reset)."""
    if not legal:
        return 0.0
    nd = non_degeneracy(spec)
    rel = relevance(edits)
    # Hard cutoff mirroring the paper's complexity>=3 -> 0 rule.
    if nd <= 0.0:
        return 0.0
    return float(nd * rel)


# --- deck legality (battle_start errorType), imported lazily ----------------
def deck_is_legal(deck: list[int]) -> tuple[bool, int]:
    """Validate a 60-card deck via the engine's battle_start error codes.

    errorType: 2 => >4 copies by NAME, 3 => no Basic Pokémon, 4 => ACE SPEC over
    limit (see memory 'Deck legality rules'). Returns (legal, errorType).
    Lazily imports cg so this module loads on a non-engine host.
    """
    from cg.game import battle_start, battle_finish
    if len(deck) != 60:
        return False, -1
    try:
        obs, start = battle_start(deck, deck)
        ok = obs is not None
        et = getattr(start, "errorType", 0)
        return ok, et
    finally:
        try:
            battle_finish()
        except Exception:
            pass


# --- LLM upgrade path (P4, off the critical path) ---------------------------
def llm_guide_score(spec: ScenarioSpec, edits: EditScript, target: ScenarioSpec) -> float:
    """Semantic elegance judge (paper §E.3). UNIMPLEMENTED v1.

    Plan: relevance(0-5) + (2 - degeneracy(0-4)) + (1 - redundancy(0/1)),
    forced to 0 if degeneracy >= guide_degeneracy_cutoff. Trigger the swap when
    auto-win rate rises or relevance-spread shrinks.
    """
    raise NotImplementedError(
        "LLM Guide is the P4 upgrade path; rule-based guide_score is the v1 default.")
