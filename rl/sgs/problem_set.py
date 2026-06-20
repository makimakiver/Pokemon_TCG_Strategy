"""Persisted problem set: the conjecturer's proposed lemmas, engine-free IO.

A *problem* is a target position (recoverable from the fixed target set D by
``target_id``) plus an edit-script the Conjecturer proposed to weaken it. We
persist ONLY ``(target_id, source, backend, edits)`` — the heavy observation
blob is rebuilt from D at load time — so the file is small and parses on any
host (no ``cg`` dependency). See docs/.../2026-06-20-nn-mcts-solver-env-design.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from rl.shared.engine.scenario import ScenarioSpec, EditScript, EditOp


def problem_row(target_id: str, source: str, backend: str, edits: EditScript) -> dict:
    return {
        "target_id": target_id,
        "source": source,
        "backend": backend,
        "edits": [{"kind": op.kind, "card_id": op.card_id, "slot": op.slot}
                  for op in edits.ops],
    }


def write_problem_set(rows: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row_to_editscript(row: dict) -> EditScript | None:
    edits = row.get("edits")
    if not isinstance(edits, list):
        return None
    ops: list[EditOp] = []
    for e in edits:
        if not isinstance(e, dict) or "kind" not in e:
            continue
        try:
            ops.append(EditOp(kind=e["kind"],
                              card_id=int(e.get("card_id", 0)),
                              slot=int(e.get("slot", 0))))
        except (ValueError, TypeError):
            continue
    return EditScript(ops=ops)


def load_problem_set(path) -> dict[str, EditScript]:
    out: dict[str, EditScript] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = row.get("target_id")
            es = _row_to_editscript(row)
            if not tid or es is None:
                continue
            out[tid] = es
    return out


def seed_scenarios(D: list[ScenarioSpec], ps: dict[str, EditScript]) -> list[ScenarioSpec]:
    """Apply each target's proposed edit-script; identity (with a warning) when a
    target has no entry, so the loop still has a lemma for every target in D."""
    seeded: list[ScenarioSpec] = []
    missing = 0
    for target in D:
        es = ps.get(target.target_id)
        if es is None:
            missing += 1
            seeded.append(target)
        else:
            seeded.append(es.apply(target))
    if missing:
        print(f"[problem_set] {missing}/{len(D)} targets had no proposed lemma "
              f"(seeded as identity)")
    return seeded
