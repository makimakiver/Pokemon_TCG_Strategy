"""Self-Guided Self-Play (SGS) RL stack for the cabt Pokémon TCG engine.

This package implements the design in ``docs/SGS_RL_PLAN.md``. The module layout
mirrors §6 of that plan:

    env.py            TCGEnv: micro-step, action masking, opponent-in-env, scenario loading
    scenario.py       ScenarioSpec + edit_script -> search_begin kwargs (legality-checked)
    targets.py        build the fixed target set D from data/loser/*.json + gauntlet losses
    encode.py         observation + per-option featurizer (runtime card stats)
    policy.py         pointer net + value head + option_prior residual
    solver_objectives.py   reinforce_half | ppo | cispo behind one interface
    vec.py            subprocess vec-env (1 battle per process)
    guide.py          rule-based Guide v1 (+ LLM upgrade path)
    mcts.py           inference-time search_begin/search_step MCTS
    league.py         OPTIONAL PSRO archive (§3.6 toggle)
    sft.py            offline behavioral cloning of scripted traces
    conjecturer/      scenario edit-script policy (parametric default; LLM optional)
    train_solver.py   inner net-RL loop
    outer_loop.py     SGS Algorithm 1 driver + eval
    eval.py           held-out gauntlet (no leakage)

The native engine (``cg/libcg.so``) is Linux x86-64 only, so everything that
imports ``cg`` must run inside the Docker linux/amd64 image (see rl/Dockerfile).
Pure-Python modules (config, scenario types, targets parsing) import on any host.
"""

__all__ = ["config"]
