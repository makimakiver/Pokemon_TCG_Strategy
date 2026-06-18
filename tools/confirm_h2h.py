import sys, importlib
from collections import Counter
sys.path.insert(0, "/app")
from cg.game import battle_start, battle_select, battle_finish

VARIANTS = ["main_v2", "main_v3_pure", "main_v3", "main_typhlosion", "main_v4"]
mods = {v: importlib.import_module("agents." + v) for v in VARIANTS}

def deck_request(mod):
    try:
        d = mod.agent({"select": None, "logs": [], "current": None})
        if isinstance(d, list) and len(d) == 60: return d
    except: pass
    return list(mod.my_deck)

def play_one(m0, m1, max_steps=10000):
    obs, start = battle_start(deck_request(m0), deck_request(m1))
    if obs is None: return -1
    pair = (m0, m1)
    try:
        for _ in range(max_steps):
            cur = obs.get("current")
            if cur is None: break
            if cur.get("result",-1)!=-1: return cur["result"]
            if obs.get("select") is None: break
            who = cur["yourIndex"]
            try: action = pair[who].agent(obs)
            except: return 1-who
            if not isinstance(action, list): action = list(action)
            obs = battle_select(action)
        return (obs.get("current") or {}).get("result", -1)
    finally: battle_finish()

def h2h(a, b, n):
    w0=w1=dr=0
    for g in range(n):
        if g%2==0:
            r = play_one(mods[a], mods[b])
            if r==0: w0+=1
            elif r==1: w1+=1
            else: dr+=1
        else:
            r = play_one(mods[b], mods[a])
            if r==1: w0+=1
            elif r==0: w1+=1
            else: dr+=1
    return w0, w1, dr

print("=== CONFIRMATION H2H (40 games per pair) ===", flush=True)
for a, b in [("main_v3_pure","main_v2"), ("main_v3_pure","main_v3"),
             ("main_v3_pure","main_typhlosion"), ("main_v3_pure","main_v4"),
             ("main_v3","main_v2"), ("main_v3","main_v4"),
             ("main_v2","main_v4")]:
    w0,w1,dr = h2h(a,b,40)
    wr = 100*w0/40
    tag = " <<<" if wr>=60 else (" <" if wr>52 else "")
    print(f"{a:<16} vs {b:<16}  {w0:2d}-{w1:2d}-{dr}  ({wr:.0f}% A){tag}", flush=True)
