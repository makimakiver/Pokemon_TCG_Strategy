# rl/dsl_agent.py
"""Run a saved DSL rule-set as a harness agent (zero API). Engine side."""
from __future__ import annotations

import json
import os

from rl.config import solver_deck_path, REPO_ROOT
from rl.dsl.interpret import compile, load

with open(solver_deck_path()) as _f:
    my_deck = json.load(_f)
assert len(my_deck) == 60

_RULESET = os.environ.get("RL_RULESET",
                          str(REPO_ROOT / "rl" / "dsl" / "examples" / "handcrafted.json"))
_agent = compile(load(_RULESET))


def agent(obs_dict):
    return _agent(obs_dict, _deck=my_deck)
