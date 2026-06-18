"""Central configuration for the SGS RL stack.

Pure-Python and dependency-free so it imports on the Mac host as well as inside
the Docker engine image. Every other module reads its knobs from here.

The decision ledger in ``docs/SGS_RL_PLAN.md`` §1 is the source of truth; the
values below are its concrete encoding. Override at runtime via the ``RL_*``
environment variables documented next to each field.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- Repo paths -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
DECKS_DIR = DATA / "decks"
LOSER_DIR = DATA / "loser"
RUNS_DIR = REPO_ROOT / "rl" / "runs"        # checkpoints + logs (gitignored)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Solver deck candidate set (pinned at P1, ledger §1 "Solver deck") ------
# The competition submission is (deck + agent). We pin the candidate set here.
# Dev-default for the headline run is the Team Rocket's Honchkrow list scraped
# from limitlesstcg deck 26267 (= the existing meta list with the single flex
# slot as Ultra Ball instead of Air Balloon). The Crustle/Typhlosion "v3_pure"
# list is kept as the secondary candidate per the ledger.
SOLVER_DECKS: dict[str, str] = {
    "honchkrow": "deck_solver_honchkrow.json",   # limitless 26267 (Ultra Ball flex)
    "honchkrow_meta": "deck_rocket_s_honchkrow.json",  # the Air-Balloon meta variant
    "crustle": "deck_crustle.json",
    "typhlosion": "deck_typhlosion.json",
}
# Which candidate is the live solver deck for this run.
SOLVER_DECK_KEY = _env_str("RL_SOLVER_DECK", "honchkrow")


def solver_deck_path() -> Path:
    return DECKS_DIR / SOLVER_DECKS[SOLVER_DECK_KEY]


# --- Held-out evaluation gauntlet (frozen, never used for training) ---------
# The original unmutated top-10 meta decks, piloted by bare_agent. Meta-share
# weights drive the headline metric. Keys are deck json slugs under data/decks/.
EVAL_GAUNTLET: list[str] = [
    "deck_dragapult_ex.json",
    "deck_n_s_zoroark_ex.json",
    "deck_crustle.json",
    "deck_slowking.json",
    "deck_hydrapple_ex.json",
    "deck_alakazam.json",
    "deck_raging_bolt_ex.json",
    "deck_ogerpon_box.json",
    "deck_lillie_s_clefairy_ex.json",
    "deck_rocket_s_honchkrow.json",
]


@dataclass
class SGSConfig:
    # ---- Target set D (ledger §1) ----
    tau: float = field(default_factory=lambda: _env_float("RL_TAU", 0.8))   # solved threshold
    k_rollouts: int = field(default_factory=lambda: _env_int("RL_K", 8))    # rollouts per scenario

    # ---- Env ----
    max_steps: int = field(default_factory=lambda: _env_int("RL_MAX_STEPS", 2000))
    opponent_module: str = field(default_factory=lambda: _env_str("RL_OPP", "agents.bare_agent"))
    seat: int = 0                # the solver always views itself as a single-seat MDP

    # ---- Solver net ----
    option_feat_dim: int = 32    # per-option feature width (see encode.OPTION_FEATURE_NAMES)
    global_feat_dim: int = 24    # per-state global feature width
    hidden_dim: int = field(default_factory=lambda: _env_int("RL_HIDDEN", 128))
    prior_weight_start: float = 1.0     # option_prior residual weight (annealed -> 0)
    prior_weight_end: float = 0.0
    prior_anneal_updates: int = field(default_factory=lambda: _env_int("RL_PRIOR_ANNEAL", 2000))

    # ---- RL inner loop ----
    objective: str = field(default_factory=lambda: _env_str("RL_OBJECTIVE", "reinforce_half"))
    lr: float = field(default_factory=lambda: _env_float("RL_LR", 3e-4))
    gamma: float = 0.999
    gae_lambda: float = 0.95
    entropy_coef: float = field(default_factory=lambda: _env_float("RL_ENT", 0.01))
    value_coef: float = 0.5
    ppo_clip: float = 0.2
    cispo_clip: float = 4.0       # clip on the stop-grad IS weight
    cispo_std_eps: float = 1e-3   # below this group std -> degenerate -> REINFORCE-half fallback
    grad_clip: float = 1.0
    n_workers: int = field(default_factory=lambda: _env_int("RL_WORKERS", 8))
    rollout_games: int = field(default_factory=lambda: _env_int("RL_ROLLOUT_GAMES", 64))
    updates_per_gen: int = field(default_factory=lambda: _env_int("RL_UPDATES", 50))

    # ---- Inference-time MCTS (off during training; on at submission) ----
    mcts_enabled: bool = field(default_factory=lambda: _env_str("RL_MCTS", "0") == "1")
    mcts_simulations: int = field(default_factory=lambda: _env_int("RL_MCTS_SIMS", 64))
    mcts_c_puct: float = 1.5

    # ---- Conjecturer / Guide ----
    conjecturer: str = field(default_factory=lambda: _env_str("RL_CONJ", "parametric"))
    edit_budget: int = field(default_factory=lambda: _env_int("RL_EDIT_BUDGET", 4))
    guide_degeneracy_cutoff: int = 3   # degeneracy >= this -> R_guide forced to 0

    # ---- P4 LLM Conjecturer (OPTIONAL; off the critical path, GPU pod only) ----
    # A small open instruct model (default SmolLM2-1.7B) emits a CoT + structured
    # edit-script, LoRA fine-tuned offline on R_synth with the SAME CISPO objective as
    # the Solver (rl/conjecturer/cispo_train.py reuses cispo_clip / cispo_std_eps).
    # Everything degrades to the parametric conjecturer when transformers/torch-GPU or
    # the weights are unavailable, so the stack still imports/plumbing-tests on CPU.
    conj_llm_model: str = field(default_factory=lambda: _env_str(
        "RL_CONJ_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct"))
    conj_llm_lora_dir: str = field(default_factory=lambda: _env_str(
        "RL_CONJ_LORA", str(RUNS_DIR / "conjecturer_lora")))   # adapter load/save dir
    conj_llm_buffer: str = field(default_factory=lambda: _env_str(
        "RL_CONJ_BUFFER", str(RUNS_DIR / "conjecturer_buffer.jsonl")))  # CISPO replay data
    conj_llm_temperature: float = field(default_factory=lambda: _env_float("RL_CONJ_TEMP", 0.9))
    conj_llm_top_p: float = field(default_factory=lambda: _env_float("RL_CONJ_TOPP", 0.95))
    conj_llm_max_new_tokens: int = field(default_factory=lambda: _env_int("RL_CONJ_MAXNEW", 320))
    conj_llm_device: str = field(default_factory=lambda: _env_str("RL_CONJ_DEVICE", "cuda"))
    conj_llm_load_4bit: bool = field(default_factory=lambda: _env_str("RL_CONJ_4BIT", "0") == "1")
    # CISPO fine-tune (the project objective; groups = completions sharing a target prompt).
    # Reuses cispo_clip / cispo_std_eps / entropy_coef above — one objective across the stack.
    conj_cispo_lr: float = field(default_factory=lambda: _env_float("RL_CONJ_LR", 1e-5))
    conj_cispo_epochs: int = field(default_factory=lambda: _env_int("RL_CONJ_EPOCHS", 2))
    conj_cispo_batch: int = field(default_factory=lambda: _env_int("RL_CONJ_BATCH", 4))
    conj_cispo_min_group: int = field(default_factory=lambda: _env_int("RL_CONJ_MINGRP", 2))
    conj_lora_r: int = field(default_factory=lambda: _env_int("RL_CONJ_LORA_R", 16))
    conj_lora_alpha: int = field(default_factory=lambda: _env_int("RL_CONJ_LORA_ALPHA", 32))
    conj_lora_dropout: float = field(default_factory=lambda: _env_float("RL_CONJ_LORA_DROPOUT", 0.05))

    # ---- Misc ----
    seed: int = field(default_factory=lambda: _env_int("RL_SEED", 0))
    device: str = field(default_factory=lambda: _env_str("RL_DEVICE", "cpu"))

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG = SGSConfig()
