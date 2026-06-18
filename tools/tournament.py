"""Comprehensive tournament: round-robin + meta benchmark for all Typhlosion/Crustle
agent variants. Runs in one process to avoid per-game Docker startup cost."""
import sys, importlib, traceback
from collections import Counter, defaultdict
sys.path.insert(0, "/app")
from cg.game import battle_start, battle_select, battle_finish

VARIANTS = ["main_v2", "main_v3", "main_v3_pure", "main_v4",
            "main_typh_base", "main_typhlosion", "main_v2_baseline"]
META_DECKS = ["dragapult_ex", "crustle", "hydrapple_ex", "ogerpon_box",
              "n_s_zoroark_ex", "slowking", "alakazam", "raging_bolt_ex",
              "lillie_s_clefairy_ex", "rocket_s_honchkrow"]
GAMES_H2H = 20
GAMES_META = 12

# Load all variant modules
mods = {}
for v in VARIANTS:
    try:
        mods[v] = importlib.import_module("agents." + v)
        print(f"loaded {v}: deck={len(mods[v].my_deck)}", flush=True)
    except Exception as e:
        print(f"FAILED {v}: {e}", flush=True)

# bare_agent needs a deck file per meta matchup; we make per-deck proxies by
# setting BARE_DECK before import. Simpler: import bare_agent fresh per deck.
def load_bare(deck_path):
    import os
    os.environ["BARE_DECK"] = deck_path
    import importlib
    import agents.bare_agent as ba
    importlib.reload(ba)
    return ba

def deck_request(mod):
    try:
        d = mod.agent({"select": None, "logs": [], "current": None})
        if isinstance(d, list) and len(d) == 60:
            return d
    except Exception:
        pass
    return list(mod.my_deck)

REASON = {1: "prizes", 2: "deckout", 3: "no-active", 4: "effect"}

def play_one(mod0, mod1, max_steps=10000):
    deck0, deck1 = deck_request(mod0), deck_request(mod1)
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        return -1, "start-fail"
    mods_pair = (mod0, mod1)
    last_reason = None
    try:
        for _ in range(max_steps):
            cur = obs.get("current")
            if cur is None:
                break
            for lg in (obs.get("logs") or []):
                if lg.get("type") == 23 and lg.get("reason") is not None:
                    last_reason = lg["reason"]
            if cur.get("result", -1) != -1:
                return cur["result"], REASON.get(last_reason, last_reason)
            if obs.get("select") is None:
                break
            who = cur["yourIndex"]
            try:
                action = mods_pair[who].agent(obs)
            except Exception:
                return 1 - who, "crash"
            if not isinstance(action, list):
                action = list(action)
            obs = battle_select(action)
        cur = obs.get("current") or {}
        return cur.get("result", -1), REASON.get(last_reason, last_reason) or "cap"
    finally:
        battle_finish()

def h2h(m0name, m1name, n):
    m0, m1 = mods[m0name], mods[m1name]
    w0 = w1 = draw = 0
    reasons = Counter()
    for g in range(n):
        if g % 2 == 0:
            res, rsn = play_one(m0, m1)
            if res == 0: w0 += 1; reasons["A:"+str(rsn)] += 1
            elif res == 1: w1 += 1; reasons["B:"+str(rsn)] += 1
            else: draw += 1
        else:
            res, rsn = play_one(m1, m0)
            if res == 1: w0 += 1; reasons["A:"+str(rsn)] += 1
            elif res == 0: w1 += 1; reasons["B:"+str(rsn)] += 1
            else: draw += 1
    return w0, w1, draw, reasons

def meta_bench(vname, deck_slug, n):
    """Variant vs bare_agent piloting deck_slug. Returns wins, losses, draws."""
    deck_path = f"data/decks/deck_{deck_slug}.json"
    bare = load_bare(deck_path)
    me = mods[vname]
    w = l = d = 0
    for g in range(n):
        if g % 2 == 0:
            res, _ = play_one(me, bare)
            if res == 0: w += 1
            elif res == 1: l += 1
            else: d += 1
        else:
            res, _ = play_one(bare, me)
            if res == 1: w += 1
            elif res == 0: l += 1
            else: d += 1
    return w, l, d

print("\n" + "="*70)
print("PART 1: ROUND-ROBIN HEAD-TO-HEAD (sides swapped, %d games each pair)" % GAMES_H2H)
print("="*70)
h2h_results = {}
h2h_score = defaultdict(int)  # total wins
pairs = [(a,b) for i,a in enumerate(VARIANTS) for b in VARIANTS[i+1:]]
for a, b in pairs:
    if a not in mods or b not in mods:
        continue
    w0, w1, draw, reasons = h2h(a, b, GAMES_H2H)
    h2h_results[(a,b)] = (w0, w1, draw)
    h2h_score[a] += w0
    h2h_score[b] += w1
    wr = 100*w0/GAMES_H2H
    tag = " <<<" if wr >= 60 else (" <" if wr > 52 else "")
    print(f"{a:<16} vs {b:<16}  {w0:2d}-{w1:2d}-{draw}  ({wr:.0f}% A){tag}", flush=True)

print("\nH2H total wins (each agent vs all others):")
for v, s in sorted(h2h_score.items(), key=lambda x:-x[1]):
    print(f"  {v:<20} {s} wins")

print("\n" + "="*70)
print("PART 2: META BENCHMARK vs bare_agent (%d games/deck x 10 decks)" % GAMES_META)
print("="*70)
# Header
hdr = "agent".ljust(18) + "".join(d.replace("_ex","").replace("_","")[:5].ljust(7) for d in META_DECKS) + "  TOTAL"
print(hdr)
meta_totals = {}
for v in VARIANTS:
    if v not in mods:
        continue
    row = v.ljust(18)
    tw = tl = 0
    cells = []
    for slug in META_DECKS:
        w, l, d = meta_bench(v, slug, GAMES_META)
        tw += w; tl += l
        cells.append(f"{w}/{GAMES_META}")
    meta_totals[v] = (tw, tl)
    row += "".join(c.rjust(7) for c in cells)
    total = GAMES_META * len(META_DECKS)
    row += f"  {tw}/{total} ({100*tw/total:.0f}%)"
    print(row, flush=True)

print("\n" + "="*70)
print("FINAL RANKING (by meta win %)")
print("="*70)
for v, (tw, tl) in sorted(meta_totals.items(), key=lambda x: -(x[1][0]/max(1,x[1][0]+x[1][1]))):
    total = tw + tl
    print(f"  {v:<20} meta={tw}/{total} ({100*tw/max(1,total):.0f}%)   h2h_wins={h2h_score.get(v,0)}")
