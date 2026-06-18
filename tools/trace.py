import os
from _paths import DECKS_DIR     # puts repo root on sys.path so `cg`/`agents` resolve
os.environ.setdefault("BARE_DECK", str(DECKS_DIR / "deck_dragapult_ex.json"))
from cg.game import battle_start, battle_select, battle_finish
from agents import bare_agent, main_bench

mods = (bare_agent, main_bench)
d0 = bare_agent.agent({"select": None, "logs": [], "current": None})
d1 = main_bench.agent({"select": None, "logs": [], "current": None})
obs, start = battle_start(d0, d1)
print("start ok:", obs is not None, getattr(start, "errorType", None))
from collections import Counter
optc = Counter()
last_turn = -1
turn_steps = 0
for step in range(10000):
    cur = obs.get("current")
    if cur is None:
        print("cur None at step", step); break
    if cur.get("result", -1) != -1:
        print("RESULT", cur["result"], "at step", step, "turn", cur.get("turn")); break
    sel = obs.get("select")
    if sel is None:
        print("select None at step", step, "turn", cur.get("turn")); break
    who = cur["yourIndex"]
    t = cur.get("turn")
    if t != last_turn:
        if last_turn != -1 and t > 60 and t % 20 == 0:
            print(f"  turn {t}: who={who} ctx={sel.get('context')} nopts={len(sel.get('option',[]))} steps_so_far={step}")
        last_turn = t
    try:
        action = mods[who].agent(obs)
    except Exception as e:
        import traceback; traceback.print_exc(); print("crash who", who); break
    # record option types chosen by bare_agent (who==0 when seat0)
    if who == 0:
        opts = sel.get("option", [])
        for i in action:
            if 0 <= i < len(opts):
                optc[opts[i].get("type")] += 1
    obs = battle_select(action)
else:
    print("HIT STEP CAP 10000 -> draw. final turn", obs.get("current", {}).get("turn"))
print("bare_agent chosen option-type counts:", dict(optc))
battle_finish()
