"""Prove data/Crow losses feed the SGS RL stack end-to-end:
   replay JSON -> ScenarioSpec -> env.reset(search_begin) -> Solver faces a decision."""
import json
from rl.targets import replay_to_targets, _deck_from_step1
from rl.env import TCGEnv
import agents.bare_agent as opp

for path in ['data/Crow/lost_2.json','data/Crow/lost_1.json']:
    replay=json.load(open(path))
    rewards=replay['rewards']; loser=1 if rewards[0]>=rewards[1] else 0
    deck_me=_deck_from_step1(replay,loser); deck_op=_deck_from_step1(replay,1-loser)
    specs=replay_to_targets(path)
    # pick a mid-game target (skip the early deck/setup steps)
    mid=[s for s in specs if int(s.source.split('step')[1])>=15]
    env=TCGEnv(deck_me, deck_op, opp)
    loaded=0; sample=None
    for s in mid[:25]:
        try:
            enc=env.reset(scenario=s)
            loaded+=1
            if sample is None:
                sample=(s.source, enc['n_options'])
        except Exception as e:
            pass
    env.close()
    print(f"{path}: loser=P{loser} | {loaded}/{min(25,len(mid))} mid-game targets loaded into engine via search_begin")
    print(f"   e.g. {sample[0]} -> Solver sees {sample[1]} legal options to choose from")
