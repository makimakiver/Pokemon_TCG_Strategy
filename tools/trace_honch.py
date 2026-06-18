"""Tally what the Honchkrow agent actually does: attacks used, damage, end reasons.
Usage: trace_honch.py [opponent_module] [n_games]"""
import sys, importlib
from collections import Counter
import _paths  # noqa: F401
from cg.game import battle_start, battle_select, battle_finish
from cg.api import all_card_data, all_attack
from agents import honchkrow as A

opp_name = sys.argv[1] if len(sys.argv) > 1 else "agents.main_bench"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
B = importlib.import_module(opp_name)

cards = {c.cardId: c.name for c in all_card_data()}
atks = {a.attackId: a.name for a in all_attack()}
REASON = {1: "all-prizes", 2: "deck-out", 3: "no-active", 4: "card-effect"}

wins = Counter(); reasons = Counter()
atk_use = Counter()      # attackId -> times A used it
atk_dmg = Counter()      # attackId -> total damage dealt by A
first_honch = []         # turn A first has Honchkrow attack

seat_wins = {0: [0, 0], 1: [0, 0]}   # seat -> [A wins, games]
firstbuck = {"first": [0, 0], "second": [0, 0]}   # A went first/second -> [wins, games]
lossbuck = {"first": Counter(), "second": Counter()}
for g in range(N):
    a_seat = g % 2                    # alternate A's seat
    dA = A.agent({"select": None, "logs": [], "current": None})
    dB = B.agent({"select": None, "logs": [], "current": None})
    (d0, d1) = (dA, dB) if a_seat == 0 else (dB, dA)
    obs, start = battle_start(d0, d1)
    if obs is None:
        reasons["start-fail"] += 1; continue
    mods = (A, B) if a_seat == 0 else (B, A)
    last_reason = None; last_attack_by_A = None; saw_honch = None; a_first = None
    seat_wins[a_seat][1] += 1
    for _ in range(10000):
        cur = obs.get("current")
        if cur is None: break
        if a_first is None and cur.get("firstPlayer", -1) in (0, 1):
            a_first = (cur["firstPlayer"] == a_seat)
        for lg in (obs.get("logs") or []):
            t = lg.get("type")
            if t == 23 and lg.get("reason") is not None:
                last_reason = lg["reason"]
            if t == 15 and lg.get("playerIndex") == a_seat:     # ATTACK by A
                aid = lg.get("attackId"); atk_use[aid] += 1; last_attack_by_A = aid
                if cards.get(lg.get("cardId")) == "Team Rocket's Honchkrow" and saw_honch is None:
                    saw_honch = cur.get("turn")
            if t == 16 and last_attack_by_A is not None and lg.get("playerIndex") == (1 - a_seat):
                v = lg.get("value") or 0
                if v < 0: atk_dmg[last_attack_by_A] += -v
        if cur.get("result", -1) != -1:
            res = cur["result"]; a_won = (res == a_seat)
            wins["A" if a_won else "B"] += 1
            seat_wins[a_seat][0] += int(a_won)
            key = "first" if a_first else "second"
            firstbuck[key][0] += int(a_won); firstbuck[key][1] += 1
            reasons[("A:" if a_won else "B:") + REASON.get(last_reason, str(last_reason))] += 1
            lossbuck[key][REASON.get(last_reason, str(last_reason)) if not a_won else "WIN"] += 1
            if saw_honch: first_honch.append(saw_honch)
            break
        if obs.get("select") is None: break
        who = cur["yourIndex"]
        try: action = mods[who].agent(obs)
        except Exception:
            import traceback; traceback.print_exc(); break
        obs = battle_select(action if isinstance(action, list) else list(action))
    battle_finish()

print(f"Honchkrow(A) vs {opp_name}: {N} games (seats alternated)")
print(f"  wins: A={wins['A']} B={wins['B']}   ({100*wins['A']/max(1,N):.0f}% A)")
for s in (0, 1):
    w, n = seat_wins[s]
    print(f"    A as seat {s}: {w}/{n}  ({100*w/max(1,n):.0f}%)")
for k in ("first", "second"):
    w, n = firstbuck[k]
    print(f"    A went {k:6s}: {w}/{n}  ({100*w/max(1,n):.0f}%)   losses: {dict(lossbuck[k])}")
print(f"  end reasons: {dict(reasons)}")
avg = sum(first_honch)/len(first_honch) if first_honch else None
print(f"  games A got Honchkrow attacking: {len(first_honch)}/{N}, avg first at turn {avg}")
print("  A attack usage (id name: count, total dmg, avg):")
for aid, n in atk_use.most_common():
    d = atk_dmg[aid]
    print(f"    {aid:5d} {atks.get(aid,'?'):20s}: used {n:3d}, dmg {d:5d}, avg {d/max(1,n):.0f}")
