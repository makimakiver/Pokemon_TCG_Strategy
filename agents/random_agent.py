"""Random-pilot baseline opponent (diagnostic only).

Returns a legal 60-card deck on registration, then picks a random legal subset of
the offered options (respecting minCount/maxCount) on every selection. Used to
sanity-check whether a trained net+MCTS can beat a non-strategic opponent; if it
can't beat random, the net/harness is broken rather than merely outmatched.
"""
import random

from cg.api import to_observation_class

# Mirror the MCTS net's training deck so deck strength is NOT a confounder.
# (net's sample_deck non-energy core + basic energy filler, padded to exactly 60.)
_core = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262]
my_deck = (_core + [3] * 60)[:60]
assert len(my_deck) == 60


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck
    select = obs.select
    n = len(select.option)
    lo = getattr(select, "minCount", 1) or 0
    hi = getattr(select, "maxCount", 1) or 1
    hi = min(hi, n)
    lo = min(lo, hi)
    k = random.randint(lo, hi) if hi >= lo else 0
    return random.sample(range(n), k)
