"""LLM Conjecturer (OPTIONAL P4 upgrade, plan §3.4 / §5 P4).

A small open instruct model (default ``SmolLM2-1.7B-Instruct``, swap via
``RL_CONJ_MODEL``) that, given a target *losing* position, emits a chain-of-thought
strategy plus a structured edit-script. It carves an easier sub-skill (lemma) out
of a hard target by weakening ONE hidden-info assumption — exactly the four
``search_begin`` prediction-mutations ``scenario.py`` can express (no board
rewrites). The model is LoRA fine-tuned **offline** with the project's CISPO
objective on ``R_synth`` by ``rl/conjecturer/cispo_train.py`` (separate phase, GPU pod).

Objective note: this stack uses **CISPO everywhere** (Solver net AND this
conjecturer), NOT GRPO — see ``rl/solver_objectives.CISPO``. CISPO's clipped
importance weight ``clamp(exp(lp - old_logp), max=cispo_clip)`` needs the BEHAVIOR
log-prob of the sampled completion, so ``propose`` records ``old_logp`` in every
buffer row alongside the reward attached later by ``update``.

Conjecturer interface used by ``outer_loop``:
  propose(target, rng) -> (edited_spec, EditScript, idx)   # logs a CISPO buffer row
  update(idx, r_synth)                                      # labels row idx with R_synth
  snapshot() -> dict                                        # flush rewarded rows -> jsonl

GRACEFUL DEGRADATION: if ``transformers`` / a usable device / the weights are
unavailable, it falls back to ``ParametricConjecturer`` for the *proposal* and
still logs a (untrained-source) buffer row, so the whole stack imports and
plumbing-tests on a CPU host with ``RL_CONJ=llm`` — the real model only loads on
the pod. Fallback rows carry ``old_logp=None`` and are skipped by the CISPO trainer.
"""
from __future__ import annotations

import json
from pathlib import Path

from rl.config import CONFIG
from rl.core.scenario import ScenarioSpec, EditScript, EditOp
from rl.conjecturer.parametric import ParametricConjecturer


# Structured-output contract (kinds MUST match scenario.EditKind exactly).
EDIT_SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"enum": ["set_opponent_active", "stack_your_deck_top",
                                       "set_opponent_hand", "weaken_opponent_hand"]},
                    "card_id": {"type": "integer"},
                    "slot": {"type": "integer"},
                },
                "required": ["kind"],
            },
        },
    },
    "required": ["edits"],
}

_LEGAL_KINDS = {"set_opponent_active", "stack_your_deck_top",
                "set_opponent_hand", "weaken_opponent_hand"}


# --- card-name context (best-effort, pure data; enriches the prompt) ---------
def _load_card_names() -> dict[int, str]:
    p = Path(__file__).resolve().parents[2] / "data" / "cards.json"
    try:
        raw = json.load(open(p))
        recs = raw if isinstance(raw, list) else raw.get("cards", list(raw.values()))
        return {r.get("cardId", r.get("id")): r.get("name", "?")
                for r in recs if isinstance(r, dict)}
    except Exception:
        return {}


_CARD_NAMES = _load_card_names()


def _nm(cid: int) -> str:
    return _CARD_NAMES.get(cid, f"card#{cid}")


_SYSTEM = (
    "You are a curriculum designer for a Pokemon TCG reinforcement-learning agent "
    "(\"the Solver\"). You are given a mid-game position the Solver currently LOSES. "
    "Propose a small edit-script that turns it into an EASIER but still non-trivial "
    "LEMMA, by weakening exactly ONE of the opponent's hidden-information advantages "
    "— never by rewriting the board. Prefer the smallest edit (1 op) that makes the "
    "position winnable-but-not-free; keep both sides with prizes and a real active.\n\n"
    "You may ONLY use these edit kinds (engine hidden-info predictions):\n"
    "  set_opponent_active {card_id}    - predict the FACE-DOWN opp active is this basic\n"
    "  stack_your_deck_top {card_id}    - put one of YOUR cards on top of your deck\n"
    "  set_opponent_hand {card_id, slot}- set one slot of the opp's predicted hand\n"
    "  weaken_opponent_hand {card_id}   - fill the opp's predicted hand with a benign card\n"
    "Reply with ONLY a JSON object: {\"reasoning\":\"...\",\"edits\":[{\"kind\":...,"
    "\"card_id\":...,\"slot\":...}]}. Use card_ids from the pools listed below."
)


def _basic_pool(spec: ScenarioSpec) -> list[int]:
    return list(spec.opponent_active) or list(spec.opponent_deck)[:6]


def _build_prompt(spec: ScenarioSpec) -> str:
    st = spec.obs["current"]
    me = st["players"][spec.my_index]
    op = st["players"][1 - spec.my_index]

    def _ids(label, ids, k=12):
        ids = list(ids)[:k]
        return (f"  {label}: " + ", ".join(f"{c}({_nm(c)})" for c in ids)) if ids \
            else f"  {label}: (none)"

    facedown = bool(spec.opponent_active)
    return "\n".join([
        f"TARGET (Solver loses this). Turn {st.get('turn')}. "
        f"You have {len(me['prize'])} prizes left, opponent {len(op['prize'])}.",
        f"Opponent active is "
        f"{'FACE-DOWN (set_opponent_active is legal)' if facedown else 'face-up (set_opponent_active is a no-op)'}.",
        "Card-id pools you may reference (id(name)):",
        _ids("your_deck (stack_your_deck_top)", spec.your_deck),
        _ids("opponent_hand (set/weaken_opponent_hand)",
             spec.opponent_hand or list(spec.opponent_deck)[:6]),
        _ids("opponent_active candidates (basics)", _basic_pool(spec)),
        f"edit budget: {CONFIG.edit_budget} ops max. Propose the MINIMAL weakening.",
    ])


# --- completion -> EditScript (parse + legality clamp to on-distribution) -----
def _extract_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        seg = text.split("```")[1]
        text = seg[4:] if seg.startswith("json") else seg
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        return {}
    try:
        return json.loads(text[a:b + 1])
    except Exception:
        return {}


def _as_int(v) -> int:
    """Coerce a model-emitted value to int; small models often return strings/garbage."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _clamp_ops(parsed: dict, spec: ScenarioSpec) -> list[EditOp]:
    """Keep only legal kinds with on-distribution card ids (mirrors parametric)."""
    ops: list[EditOp] = []
    your_deck = set(spec.your_deck)
    opp_hand_pool = set(spec.opponent_hand) | set(spec.opponent_deck)
    raw_edits = parsed.get("edits") if isinstance(parsed, dict) else None
    if not isinstance(raw_edits, list):          # model may emit a non-list "edits"
        raw_edits = []
    for e in raw_edits[: CONFIG.edit_budget]:
        if not isinstance(e, dict):              # ...or list items that aren't dicts
            continue
        kind = e.get("kind")
        if kind not in _LEGAL_KINDS:
            continue
        cid = _as_int(e.get("card_id", 0))
        slot = _as_int(e.get("slot", 0))
        if kind == "stack_your_deck_top" and cid not in your_deck:
            continue                                  # must be a card we actually own
        if kind in ("set_opponent_hand", "weaken_opponent_hand") and opp_hand_pool \
                and cid not in opp_hand_pool:
            cid = next(iter(opp_hand_pool))           # snap to an on-distribution id
        if kind == "set_opponent_active" and not spec.opponent_active:
            continue                                  # only legal vs a face-down active
        ops.append(EditOp(kind=kind, card_id=cid, slot=slot))
    return ops


class LLMConjecturer:
    """SmolLM-backed conjecturer; CISPO buffer logging; parametric fallback."""

    def __init__(self, *, load_model: bool | None = None):
        self.cfg = CONFIG
        self._buffer: list[dict] = []      # rows: idx,target_id,prompt,completion,old_logp,edits,reward,source
        self._fallback = ParametricConjecturer()
        self._tok = None
        self._model = None
        self._ok = False
        want = (CONFIG.conjecturer == "llm") if load_model is None else load_model
        if want:
            self._try_load_model()

    # -- model loading (lazy, optional) --------------------------------------
    def _try_load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            print(f"[conjecturer.llm] transformers unavailable ({e}); "
                  f"parametric fallback (CPU plumbing mode).")
            return
        try:
            name = self.cfg.conj_llm_model
            self._tok = AutoTokenizer.from_pretrained(name)
            kw = {}
            if self.cfg.conj_llm_load_4bit:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            else:
                kw["torch_dtype"] = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(name, **kw)
            lora = Path(self.cfg.conj_llm_lora_dir)
            if (lora / "adapter_config.json").exists():
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(lora))
                print(f"[conjecturer.llm] loaded LoRA adapter from {lora}")
            self._model = model.to(self.cfg.conj_llm_device).eval()
            self._ok = True
            print(f"[conjecturer.llm] loaded {name} on {self.cfg.conj_llm_device}")
        except Exception as e:
            print(f"[conjecturer.llm] model load failed ({e}); parametric fallback.")
            self._tok = self._model = None
            self._ok = False

    # -- generation + behavior log-prob (CISPO old_logp) ---------------------
    def _ids_for(self, prompt: str):
        msgs = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt}]
        enc = self._tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt")
        if hasattr(enc, "input_ids"):          # newer transformers returns a BatchEncoding
            enc = enc.input_ids
        return enc.to(self._model.device)

    def _generate(self, prompt: str):
        """Sample a completion; return (text, old_logp) where old_logp is the summed
        token log-prob of the sampled completion under the current policy."""
        import torch
        ids = self._ids_for(prompt)
        with torch.no_grad():
            out = self._model.generate(
                ids, max_new_tokens=self.cfg.conj_llm_max_new_tokens, do_sample=True,
                temperature=self.cfg.conj_llm_temperature, top_p=self.cfg.conj_llm_top_p,
                pad_token_id=self._tok.eos_token_id,
                return_dict_in_generate=True, output_scores=True)
        seq = out.sequences[0, ids.shape[1]:]
        # per-step token log-prob from the generation scores (= behavior policy)
        old_logp = 0.0
        for tok, step_scores in zip(seq, out.scores):
            logp = torch.log_softmax(step_scores[0].float(), dim=-1)
            old_logp += float(logp[tok])
        text = self._tok.decode(seq, skip_special_tokens=True)
        return text, old_logp

    # -- Conjecturer interface ----------------------------------------------
    def propose(self, target: ScenarioSpec, rng):
        prompt = _build_prompt(target)
        old_logp = None
        if self._ok:
            completion, old_logp = self._generate(prompt)
            ops = _clamp_ops(_extract_json(completion), target)
            source = "llm"
            if not ops:                              # empty/invalid -> safe parametric edit
                _, fb_edits, _ = self._fallback.propose(target, rng)
                ops = fb_edits.ops
                source = "llm_empty_fallback"
        else:
            _, fb_edits, _ = self._fallback.propose(target, rng)
            ops = fb_edits.ops
            completion = json.dumps({"reasoning": "parametric fallback (no model)",
                                     "edits": [op.__dict__ for op in ops]})
            source = "fallback"

        edits = EditScript(ops=ops, budget=CONFIG.edit_budget)
        edited = edits.apply(target)
        idx = len(self._buffer)
        self._buffer.append({
            "idx": idx, "target_id": target.target_id or target.source,
            "prompt": prompt, "completion": completion, "old_logp": old_logp,
            "edits": [op.__dict__ for op in edits.ops],
            "reward": None, "source": source,
        })
        return edited, edits, idx

    def update(self, idx: int, r_synth: float) -> None:
        if 0 <= idx < len(self._buffer):
            self._buffer[idx]["reward"] = float(r_synth)

    def snapshot(self) -> dict:
        """Append rewarded rows to the CISPO replay jsonl; return stats."""
        rows = [r for r in self._buffer if r["reward"] is not None]
        if rows:
            path = Path(self.cfg.conj_llm_buffer)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            self._buffer = [r for r in self._buffer if r["reward"] is None]
        srcs: dict[str, int] = {}
        for r in rows:
            srcs[r["source"]] = srcs.get(r["source"], 0) + 1
        return {"backend": "llm" if self._ok else "parametric_fallback",
                "model": self.cfg.conj_llm_model, "flushed": len(rows),
                "trainable": sum(1 for r in rows if r["old_logp"] is not None),
                "sources": srcs}
