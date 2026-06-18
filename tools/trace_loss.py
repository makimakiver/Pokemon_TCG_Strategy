"""Detailed turn-by-turn trace of a single lost replay.

Usage: python trace_loss.py <file.json>
Prints, for each turn where makimakiver acted:
  turn | select context | chosen option types | my board (hp) | opp board (hp) | prizes
Also hunts for the RESULT log anywhere in the file.
"""
import json
import sys
from collections import Counter

ME = "makimakiver"
CTX = {
    0: "MAIN", 1: "SETUP_ACTIVE", 2: "SETUP_BENCH", 3: "SWITCH", 4: "TO_ACTIVE",
    5: "TO_BENCH", 6: "TO_FIELD", 7: "TO_HAND", 8: "DISCARD", 9: "TO_DECK",
    21: "ATTACH_FROM", 18: "EVOLVES_FROM", 19: "EVOLVES_TO", 12: "NOT_MOVE",
    14: "DAMAGE_COUNTER_ANY", 25: "EFFECT_TARGET",
}
OPT = {
    0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD", 5: "ENERGY_CARD",
    6: "ENERGY", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD",
    12: "RETREAT", 13: "ATTACK", 14: "END", 15: "SKILL", 16: "SPECIAL_CONDITION",
}
REASON = {1: "all-prizes", 2: "deck-out", 3: "no-active", 4: "card-effect"}


def board_str(ps, seat):
    p = ps[seat]
    act = p["active"]
    a = f"{act[0]['id']}({act[0]['hp']}h)" if act and act[0] else "-"
    bench = ",".join(f"{b['id']}({b['hp']}h)" for b in p["bench"])
    return f"act={a} bench=[{bench}] pr={len(p['prize'])} dk={p.get('deckCount')} hd={p.get('handCount')}"


def main():
    path = sys.argv[1]
    with open(path) as f:
        d = json.load(f)
    agents = d["info"]["Agents"]
    my_seat = next(i for i, a in enumerate(agents) if a["Name"] == ME)
    opp = 1 - my_seat
    opp_name = agents[opp]["Name"]
    steps = d["steps"]
    print(f"== {path.split('/')[-1]}  me=seat{my_seat}  opp={opp_name} ==")

    # hunt RESULT log across ALL seats/steps
    result = reason = None
    for pair in steps:
        for el in pair:
            for lg in (el.get("observation") or {}).get("logs", []) or []:
                if lg.get("type") == 23:
                    result = lg.get("result")
                    reason = lg.get("reason")
    print(f"RESULT log: result={result} reason={REASON.get(reason, reason)}")
    print(f"rewards: {d.get('rewards')}\n")

    last_turn = -1
    for i, pair in enumerate(steps):
        if my_seat >= len(pair):
            continue
        e = pair[my_seat]
        if e["status"] != "ACTIVE":
            continue
        obs = e["observation"]
        cur = obs.get("current")
        if cur is None:
            continue
        sel = obs.get("select")
        t = cur.get("turn")
        ps = cur["players"]
        ctx = sel.get("context") if sel else None
        chosen = []
        if sel:
            opts = sel.get("option", [])
            for idx in (e.get("action") or []):
                if 0 <= idx < len(opts):
                    chosen.append(OPT.get(opts[idx].get("type"), opts[idx].get("type")))
        # print a header only on new turn, but list every action
        marker = f"\n--- turn {t} ---" if t != last_turn else ""
        if t != last_turn:
            print(f"\n[T{t}] me: {board_str(ps, my_seat)}")
            print(f"     opp: {board_str(ps, opp)}")
            last_turn = t
        print(f"  step{i:3d} ctx={CTX.get(ctx, ctx)} chose={chosen}")


if __name__ == "__main__":
    main()
