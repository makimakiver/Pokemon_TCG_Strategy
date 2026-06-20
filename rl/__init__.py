"""Self-Guided Self-Play (SGS) RL stack for the cabt Pokémon TCG engine.

Implements the design in ``docs/SGS_RL_PLAN.md``. Organised into subpackages by
responsibility (all intra-package imports are absolute, e.g. ``rl.solver.policy``):

    config.py            central config (knobs via RL_* env vars); imports anywhere

    core/                engine-facing primitives + scenario/problem types
      env                TCGEnv: micro-step, action masking, opponent-in-env, scenario loading
      scenario           ScenarioSpec + EditScript -> search_begin kwargs (legality-checked)
      encode             observation + per-option featurizer (runtime card stats)
      targets            build the target set D from data/loser/*.json
      problem_set        persisted conjecturer problem set (target_id + edit-script)
      vec                subprocess vec-env (1 battle per process)

    solver/              the solver net + RL training
      policy             pointer net + value head + option_prior residual
      solver_objectives  reinforce_half | ppo | cispo | alphazero behind one interface
      mcts               search_begin/search_step MCTS (prior+value, search-node release)
      train_solver       inner net-RL loop + actor factories (policy / MCTS)
      outer_loop         SGS Algorithm 1 driver (+ seed-first curriculum, live games)
      guide              rule-based Guide v1 (R_guide)
      league             OPTIONAL PSRO archive

    conjecturer/         scenario edit-script policy (parametric default; LLM optional)
    dsl/                 DSL rule-engine for scripted-rule synthesis

    agents/              shippable agent wrappers (net_agent, llm_agent, dsl_agent)
    eval/                evaluation harnesses (eval, kaggle_eval, eval_mcts_vs_agent, eval_sgs_mcts)
    bootstrap/           warm-starts (bootstrap_solver, distill_claude, sft)
    kaggle/              kaggle_mcts self-play trainer + train_multi
    smoke/               smoke_test (P0) + smoke_test_mcts_sgs

The native engine (``cg/libcg.so``) is Linux x86-64 only, so everything that imports
``cg`` must run inside the Docker linux/amd64 image (see rl/Dockerfile). Pure-Python
modules (config, core.scenario, core.targets, problem_set) import on any host.
"""

__all__ = ["config"]
