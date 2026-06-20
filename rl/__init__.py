"""Self-Guided Self-Play (SGS) RL stack for the cabt Pokémon TCG engine.

Implements the design in ``docs/SGS_RL_PLAN.md``. Organised folders-of-folders by
domain; all intra-package imports are absolute (e.g. ``rl.training.solver.policy``):

    config.py                 central config (knobs via RL_* env vars); imports anywhere

    engine/                   interface to the game engine
      env                     TCGEnv: micro-step, action masking, opponent-in-env, scenarios
      scenario                ScenarioSpec + EditScript -> search_begin kwargs (legality-checked)
      encode                  observation + per-option featurizer
      vec                     subprocess vec-env (1 battle per process)

    training/                 everything about TRAINING the solver
      solver/                 the learner
        policy                pointer net + value head + option_prior residual
        mcts                  search_begin/search_step MCTS (prior+value, node release)
        solver_objectives     reinforce_half | ppo | cispo | alphazero behind one interface
        train_solver          inner net-RL loop + actor factories
        outer_loop            SGS Algorithm 1 driver (seed-first curriculum, live games)
        guide                 rule-based Guide v1 (R_guide); league = OPTIONAL PSRO archive
      curriculum/             WHAT the solver trains on
        targets               build the target set D from data/loser/*.json
        problem_set           persisted conjecturer problem set (target_id + edit-script)
        conjecturer/          quiz generators (parametric default; SmolLM LoRA optional)
      bootstrap/              warm-starts (bootstrap_solver, distill_claude, sft)

    inference/                USING a trained model
      agents/                 shippable wrappers (net_agent, llm_agent, dsl_agent)
      eval/                   SGS eval harnesses (eval gauntlet, eval_mcts_vs_agent, eval_sgs_mcts)

    selfplay/                 the SEPARATE kaggle_mcts lineage (kaggle_mcts, train_multi, kaggle_eval)
    dsl/                      rule-synthesis engine (standalone)
    smoke/                    smoke tests (P0 + MCTS-SGS)

The native engine (``cg/libcg.so``) is Linux x86-64 only, so everything that imports
``cg`` runs only inside the Docker linux/amd64 image (rl/Dockerfile). Pure-Python modules
(config, engine.scenario, training.curriculum.targets/problem_set) import on any host.
"""

__all__ = ["config"]
