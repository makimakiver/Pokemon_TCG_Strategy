"""LLM-driven policy: Claude picks the moves, live, per decision.

This is an alternative Solver backend to the trained pointer net (``policy.py``) —
the policy is *dynamically controlled by an LLM* at decision time. It conforms to
the harness agent interface (``agent(obs_dict) -> list[int]`` + module-level
``my_deck``), so it drops into every existing seam unchanged:

  * opponent in ``TCGEnv``            → ``RL_OPP=rl.llm_agent``
  * head-to-head in the runner        → ``python runner.py --a rl.llm_agent --b agents.bare_agent``
  * **SFT teacher** (the best use)    → ``sft.collect_traces(..., scripted_module="rl.llm_agent")``
    lets Claude play and clones its moves into the shippable net.

How it decides: for a real (non-forced) selection it renders the board + the
enumerated legal options to text, asks Claude (structured JSON output) for the
option index(es) to pick, validates against the engine's min/max + legality, and
applies them. Trivial / forced selects are resolved with the scripted prior
(``encode.option_prior_scores``) so we don't pay an API call per non-decision.

Robustness: if the ``anthropic`` SDK is missing, ``ANTHROPIC_API_KEY`` is unset,
or any call errors, it **falls back to the scripted prior** — so it always
returns a legal action and runs (in fallback form) even without a key.

NOTE: the competition submission cannot call external APIs at inference
(sandboxed, per-step timeout). Use this for research / evaluation / as an SFT
teacher, not as the shipped agent. Requires ``cg`` (Linux) -> Docker image.
"""
from __future__ import annotations

import json
import os

from cg.api import (
    AreaType, CardType, OptionType, Pokemon, SelectContext, to_observation_class,
)
from . import encode
from .config import solver_deck_path

# ---- Deck this LLM agent pilots (the Honchkrow 26267 solver deck) ----
my_deck = json.load(open(solver_deck_path()))
assert len(my_deck) == 60, f"deck has {len(my_deck)} cards"

# ---- LLM config (claude-api skill defaults: opus-4-8, override via env) ----
LLM_MODEL = os.environ.get("RL_LLM_MODEL", "claude-opus-4-8")
LLM_MAX_TOKENS = int(os.environ.get("RL_LLM_MAX_TOKENS", "1024"))
LLM_THINKING = os.environ.get("RL_LLM_THINKING", "0") == "1"
# Only consult the LLM for these contexts by default (the real strategic
# decisions); everything else uses the scripted prior to bound cost.
LLM_CONTEXTS = {int(SelectContext.MAIN)}
LLM_ALL = os.environ.get("RL_LLM_ALL", "0") == "1"   # call the LLM on every select

_client = None
_client_failed = False


def _get_client():
    """Lazily construct the Anthropic client; None if unavailable (-> fallback)."""
    global _client, _client_failed
    if _client is not None or _client_failed:
        return _client
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _client_failed = True
        print("[llm_agent] ANTHROPIC_API_KEY unset -> using scripted-prior fallback")
        return None
    try:
        import anthropic
        _client = anthropic.Anthropic()
    except Exception as e:                      # SDK missing / construction error
        _client_failed = True
        print(f"[llm_agent] anthropic unavailable ({e}) -> scripted-prior fallback")
    return _client


# ---- scripted-prior fallback (shared with the net's prior) ------------------
def _prior_picks(obs) -> list[int]:
    sel = obs.select
    scores = encode.option_prior_scores(obs)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n = max(sel.minCount, min(sel.maxCount, len(order)))
    picks = order[:n]
    if len(picks) < sel.minCount:
        picks = order[: sel.minCount]
    return picks


# ---- human-readable rendering for the LLM -----------------------------------
def _name(cid) -> str:
    cd = encode.CARD_TABLE.get(cid)
    return cd.name if cd else f"card#{cid}"


def _poke_str(p) -> str:
    if p is None:
        return "(facedown)"
    cd = encode.CARD_TABLE.get(p.id)
    return (f"{_name(p.id)} hp{p.hp}/{getattr(cd,'hp',0) or p.hp} "
            f"E{len(p.energies)}" + (f" T{len(p.tools)}" if p.tools else ""))


def _option_str(obs, o, idx, my_index) -> str:
    t = OptionType(o.type)
    base = f"[{idx}] {t.name}"
    if t == OptionType.ATTACK:
        atk = encode.ATTACK_TABLE.get(o.attackId)
        if atk:
            base += f" {atk.name} ({atk.damage or 0} dmg, cost {len(atk.energies or [])})"
    elif t in (OptionType.PLAY, OptionType.EVOLVE, OptionType.ATTACH):
        card = encode._card_of_option(obs, o, my_index)
        if card is not None:
            base += f" {_name(card.id)}"
    elif t in (OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD,
               OptionType.ENERGY):
        card = encode.get_card(obs, o.area, o.index,
                               o.playerIndex if o.playerIndex is not None else my_index)
        if card is not None:
            owner = "mine" if o.playerIndex == my_index else "opp" \
                if o.playerIndex is not None else "?"
            base += f" {_name(card.id)} ({owner})"
    elif t == OptionType.NUMBER:
        base += f" = {o.number}"
    return base


def _render(obs) -> str:
    st = obs.current
    sel = obs.select
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]
    lines = [
        f"Turn {st.turn}. You have {len(me.prize)} prizes left, opponent {len(op.prize)}.",
        f"Your active: {_poke_str(me.active[0] if me.active else None)}",
        f"Your bench: {', '.join(_poke_str(p) for p in me.bench) or '(empty)'}",
        f"Opp active: {_poke_str(op.active[0] if op.active else None)}",
        f"Opp bench: {len(op.bench)} Pokemon",
        f"Your hand: {', '.join(_name(c.id) for c in (me.hand or [])) or '(hidden)'}",
        f"Energy attached this turn: {st.energyAttached}; supporter used: {st.supporterPlayed}",
        f"\nSelection: {SelectContext(sel.context).name} "
        f"(choose between {sel.minCount} and {sel.maxCount} options, no duplicates).",
        "Legal options:",
    ]
    lines += [_option_str(obs, o, i, st.yourIndex) for i, o in enumerate(sel.option)]
    return "\n".join(lines)


_SYSTEM = (
    "You are an expert Pokemon Trading Card Game player piloting a Team Rocket's "
    "Honchkrow deck in the cabt engine. You will be shown the current board state "
    "and a list of legal option indices. Choose the option index(es) that best "
    "advance toward taking all prizes (knock out the opponent's Pokemon, set up "
    "your attackers, and play efficiently). Respect the min/max count and never "
    "pick duplicate indices. Pick the strongest concrete line; when unsure prefer "
    "playing Pokemon and items, attaching energy to your attacker, and attacking "
    "for a knockout."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "picks": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["picks"],
    "additionalProperties": False,
}


def _ask_llm(obs) -> list[int] | None:
    client = _get_client()
    if client is None:
        return None
    sel = obs.select
    try:
        kwargs = dict(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _render(obs)}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
        if LLM_THINKING:
            kwargs["thinking"] = {"type": "adaptive"}
        resp = client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            return None
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if not text:
            return None
        picks = json.loads(text).get("picks", [])
    except Exception as e:
        print(f"[llm_agent] LLM call failed ({type(e).__name__}: {e}) -> fallback")
        return None
    return _sanitize(picks, sel)


def _sanitize(picks, sel) -> list[int] | None:
    n = len(sel.option)
    seen, clean = set(), []
    for p in picks:
        if isinstance(p, int) and 0 <= p < n and p not in seen:
            seen.add(p)
            clean.append(p)
    if len(clean) < sel.minCount:
        for i in range(n):
            if i not in seen:
                clean.append(i)
                seen.add(i)
            if len(clean) >= sel.minCount:
                break
    clean = clean[: sel.maxCount] if sel.maxCount else clean
    return clean if len(clean) >= sel.minCount else None


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:                       # deck request
        return list(my_deck)

    sel = obs.select
    n = len(sel.option)
    # Forced / trivial selections: never worth an API call.
    if n <= 1 or sel.minCount >= n:
        return _prior_picks(obs)
    use_llm = LLM_ALL or int(sel.context) in LLM_CONTEXTS
    if use_llm:
        picks = _ask_llm(obs)
        if picks is not None:
            return picks
    return _prior_picks(obs)
