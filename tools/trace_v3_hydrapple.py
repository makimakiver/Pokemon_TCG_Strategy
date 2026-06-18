"""Trace a single game of main_v3 vs bare_agent(hydrapple) inside Docker."""
import os, sys
os.environ["BARE_DECK"] = "data/decks/deck_hydrapple_ex.json"
sys.path.insert(0, "/app")
from cg.game import battle_start, battle_select, battle_finish
from agents import main_v3, bare_agent
from cg.api import to_observation_class

CTX = {0:"MAIN",1:"SETUP_ACT",2:"SETUP_BENCH",3:"SWITCH",4:"TO_ACTIVE",5:"TO_BENCH",
       6:"TO_FIELD",7:"TO_HAND",8:"DISCARD",21:"ATTACH_FROM",18:"EVOL_FROM",19:"EVOL_TO",12:"NOT_MOVE"}
OPT = {0:"NUM",1:"YES",2:"NO",3:"CARD",7:"PLAY",8:"ATTACH",9:"EVOLVE",10:"ABILITY",
       11:"DISCARD",12:"RETREAT",13:"ATTACK",14:"END",15:"SKILL"}

def bstr(ps, s):
    a = ps["active"]
    act = f"{a[0]['id']}({a[0]['hp']}h)" if a and a[0] else "-"
    bn = ",".join(f"{b['id']}({b['hp']}h)" for b in ps["bench"])
    return f"act={act} bench=[{bn}] pr={len(ps['prize'])} dk={ps.get('deckCount')} hd={ps.get('handCount')}"

deck0 = main_v3.agent({"select": None, "logs": [], "current": None})
deck1 = bare_agent.agent({"select": None, "logs": [], "current": None})
obs, start = battle_start(deck0, deck1)
print("start:", start.errorType)
mods = (main_v3, bare_agent)
last_turn = -1
for step in range(10000):
    cur = obs.get("current")
    if cur is None:
        print("cur None at", step); break
    res = cur.get("result", -1)
    if res != -1:
        print(f"RESULT={res} at step={step} turn={cur.get('turn')}")
        p0, p1 = cur["players"]
        print(f"  P0(v3): {bstr(p0,0)}")
        print(f"  P1(bare): {bstr(p1,1)}")
        break
    sel = obs.get("select")
    if sel is None:
        print("select None at", step); break
    who = cur["yourIndex"]
    t = cur.get("turn")
    if who == 0 and t != last_turn:
        p0, p1 = cur["players"]
        print(f"\n[T{t}] v3: {bstr(p0,0)}")
        print(f"     bare: {bstr(p1,1)}")
        last_turn = t
    if who == 0:
        ctx = sel.get("context")
        opts = sel.get("option", [])
        action = mods[who].agent(obs)
        chosen = [OPT.get(opts[i].get("type"), opts[i].get("type")) for i in action if 0 <= i < len(opts)]
        print(f"  s{step:3d} ctx={CTX.get(ctx,ctx)} chose={chosen}")
    else:
        action = mods[who].agent(obs)
    obs = battle_select(action)
battle_finish()
