"""Chunk a transcript and extract validated SemanticEvents via the Anthropic API."""
from __future__ import annotations
import json
from typing import List, Dict

from pydantic import ValidationError

from .schema import SemanticEvent, EventType
from .prompts import SYSTEM, build_user_prompt

DEFAULT_MODEL = "claude-sonnet-4-6"


def _window(segments: List[Dict], win_s: float, overlap_s: float) -> List[List[Dict]]:
    """Slide a time window over segments with overlap so no turn is split."""
    if not segments:
        return []
    windows, i = [], 0
    while i < len(segments):
        t0 = segments[i]["start"]
        j = i
        while j < len(segments) and segments[j]["start"] - t0 < win_s:
            j += 1
        windows.append(segments[i:j])
        # step forward, leaving `overlap_s` of tail in the next window
        step_t = t0 + win_s - overlap_s
        nxt = i
        while nxt < len(segments) and segments[nxt]["start"] < step_t:
            nxt += 1
        i = max(nxt, i + 1)
    return windows


def _fmt(window: List[Dict]) -> str:
    return "\n".join(f"[{s['start']:.1f}] {s['text']}" for s in window)


def _extract_json_array(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("["):]
    a, b = text.find("["), text.rfind("]")
    if a == -1 or b == -1:
        return []
    return json.loads(text[a:b + 1])


def extract_events(
    segments: List[Dict],
    players: List[str],
    model: str = DEFAULT_MODEL,
    win_s: float = 90.0,
    overlap_s: float = 15.0,
    max_tokens: int = 4000,
    verbose: bool = True,
) -> List[SemanticEvent]:
    import anthropic
    client = anthropic.Anthropic()

    events: List[SemanticEvent] = []
    seen = set()
    running_turn = 0

    for w_idx, window in enumerate(_window(segments, win_s, overlap_s)):
        tail = "\n".join(
            json.dumps(e.model_dump(exclude_none=True, mode="json"))
            for e in events[-6:]
        )
        user = build_user_prompt(_fmt(window), players, running_turn, tail)
        try:
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, system=SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  window {w_idx}: API error {e}")
            continue

        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            raw = _extract_json_array(text)
        except json.JSONDecodeError:
            if verbose:
                print(f"  window {w_idx}: bad JSON, skipped")
            continue

        kept = 0
        for obj in raw:
            try:
                ev = SemanticEvent(**obj)
            except ValidationError:
                continue
            k = ev.dedupe_key()
            if k in seen:
                continue
            seen.add(k)
            events.append(ev)
            running_turn = max(running_turn, ev.turn)
            kept += 1
        if verbose:
            print(f"  window {w_idx}: +{kept} events (turn~{running_turn})")

    events.sort(key=lambda e: (e.turn, e.t_start or 0.0))
    return events
