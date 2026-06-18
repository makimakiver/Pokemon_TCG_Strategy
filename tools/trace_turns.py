"""Print Honchkrow(A)'s actions turn-by-turn for one game (A going second by default).
Usage: trace_turns.py [opponent] [a_seat] [max_actions]"""
import sys, importlib
import _paths  # noqa: F401
from cg.game import battle_start, battle_select, battle_finish
from cg.api import all_card_data, all_attack, OptionType, SelectContext, to_observation_class
from agents import honchkrow as A

opp = sys.argv[1] if len(sys.argv) > 1 else "agents.main_cur"
a_seat = int(sys.argv[2]) if len(sys.argv) > 2 else 1     # 1 = A goes second
maxA = int(sys.argv[3]) if len(sys.argv) > 3 else 40
B = importlib.import_module(opp)
cards = {c.cardId: c.name for c in all_card_data()}
atks = {a.attackId: a.name for a in all_attack()}
OT = {int(v): v.name for v in OptionType}

dA = A.agent({"select": None, "logs": [], "current": None})
dB = B.agent({"select": None, "logs": [], "current": None})
(d0, d1) = (dA, dB) if a_seat == 0 else (dB, dA)
obs, _ = battle_start(d0, d1)
mods = (A, B) if a_seat == 0 else (B, A)
shown = 0
while shown < maxA:
    cur = obs.get("current")
    if cur is None or cur.get("result", -1) != -1:
        print("RESULT:", cur.get("result") if cur else None); break
    if obs.get("select") is None: break
    who = cur["yourIndex"]
    sel = obs["select"]
    action = mods[who].agent(obs)
    action = action if isinstance(action, list) else list(action)
    if who == a_seat:
        o = to_observation_class(obs).select
        ctx = SelectContext(int(o.context)).name
        picked = []
        for idx in action:
            if 0 <= idx < len(o.option):
                op = o.option[idx]
                tag = OT.get(int(op.type), str(op.type))
                cid = op.cardId or op.attackId
                nm = atks.get(op.attackId) if op.type == OptionType.ATTACK else None
                picked.append(f"{tag}{('/'+nm) if nm else ''}")
        ms = cur.get("players")[a_seat]
        nb = (1 if ms["active"] and ms["active"][0] else 0) + len(ms["bench"])
        if ctx == "MAIN" or picked:
            print(f"  t{cur.get('turn'):2} {ctx:16s} bodies={nb} deck={ms.get('deckCount')} hand={ms.get('handCount')} -> {picked}")
            shown += 1
    obs = battle_select(action)
battle_finish()
