"""Self-Guided Self-Play (SGS) RL stack for the cabt Pokémon TCG engine.

Implements the design in ``docs/SGS_RL_PLAN.md``. Organised by **methodology**, with a
shared infra base. All intra-package imports are absolute (e.g. ``rl.sgs.outer_loop``):

    config.py                 central config (knobs via RL_* env vars); imports anywhere

    shared/                   infra used by ALL methodologies
      engine/                 interface to the game engine
        env                   TCGEnv: micro-step, action masking, opponent-in-env, scenarios
        scenario              ScenarioSpec + EditScript -> search_begin kwargs
        encode                observation + per-option featurizer; vec = subprocess vec-env
      policy.py               the pointer net + value head (shared model)
      agents/                 shippable wrappers (net_agent, llm_agent, dsl_agent)
      eval/                   eval harnesses (eval gauntlet, eval_mcts_vs_agent, eval_sgs_mcts)
      bootstrap/              warm-starts (bootstrap_solver, distill_claude, sft)
      dsl/                    rule-synthesis engine

    sgs/                      METHODOLOGY: Self-Guided Self-play + MCTS
      outer_loop              SGS Algorithm 1 driver (seed-first curriculum, live games)
      train_solver            inner net-RL loop + actor factories
      solver_objectives       reinforce_half | ppo | cispo | alphazero (the AlphaZero objective)
      mcts                    search_begin/search_step MCTS (prior+value, node release)
      guide                   rule-based Guide v1 (R_guide); league = OPTIONAL PSRO archive
      targets, problem_set    the curriculum (D from data/loser + persisted conjecturer problems)
      conjecturer/            quiz generators (parametric default; SmolLM LoRA optional)

    alphazero/                METHODOLOGY: vanilla AlphaZero self-play (deck-agnostic)
      kaggle_mcts             transformer value/policy net + determinized MCTS self-play
      kaggle_eval, train_multi   eval + multi-opponent driver

    smoke/                    smoke tests (P0 + MCTS-SGS)

The native engine (``cg/libcg.so``) is Linux x86-64 only, so everything that imports
``cg`` runs only inside the Docker linux/amd64 image (rl/Dockerfile). Pure-Python modules
(config, shared.engine.scenario, sgs.targets/problem_set) import on any host.
"""

__all__ = ["config"]
