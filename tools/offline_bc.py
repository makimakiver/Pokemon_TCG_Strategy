"""Prove ALL Crow files are consumable by the rl/ stack:
 - lost_* (Crow lost)  -> Crow-view D-targets (search_begin scenarios)  [path 1]
 - win_*  (Crow won)   -> offline behavioral-cloning traces of Crow's winning play [path 2]
Crow's SEAT varies per file, so we detect it by deck signature and read that seat.
"""
import json, glob
from collections import Counter
from cg.api import to_observation_class
from rl import encode

CROW_SIG={891,463,15,1216,1217,1218,1219,1220}
def crow_seat(d):
    def deck(seat):
        for s in d['steps'][:3]:
            a=s[seat].get('action') or []
            if len(a)==60: return set(a)
        return set()
    return 0 if len(CROW_SIG&deck(0))>=len(CROW_SIG&deck(1)) else 1

def offline_traces(path):
    d=json.load(open(path)); seat=crow_seat(d); rw=d['rewards']
    crow_won = rw[seat]==1
    traces=0; games_ok=0
    for s in d['steps']:
        ag=s[seat]
        if ag.get('status')!='ACTIVE': continue
        obs=ag.get('observation')
        if not isinstance(obs,dict) or obs.get('select') is None: continue
        cur=obs.get('current')
        if not cur or cur.get('result',-1)!=-1: continue
        action=ag.get('action') or []
        try:
            o=to_observation_class(obs)
            g,opts,_=encode.featurize(o)            # pure featurization, no engine
        except Exception:
            continue
        nopt=len(o.select.option)
        picks=[a for a in action if isinstance(a,int) and 0<=a<nopt]
        traces+=len(picks)
        games_ok+=1
    return seat, crow_won, traces, games_ok

print(f"{'file':<12} {'crow_seat':>9} {'crow_won':>8} {'BC_picks':>9} {'decisions':>9}")
tot_win=tot_loss=0
for f in sorted(glob.glob('data/Crow/*.json')):
    seat,won,tr,dec=offline_traces(f)
    print(f"{f.split('/')[-1]:<12} {seat:>9} {str(won):>8} {tr:>9} {dec:>9}")
    if won: tot_win+=tr
    else: tot_loss+=tr
print(f"\nBC picks from CROW WINS (usable warm-start traces): {tot_win}")
print(f"BC picks from CROW LOSSES (avoid cloning; use as D instead): {tot_loss}")
