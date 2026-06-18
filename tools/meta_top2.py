import sys, importlib, os
sys.path.insert(0, "/app")
from cg.game import battle_start, battle_select, battle_finish

VARIANTS = ["main_v2", "main_v3_pure", "main_v3"]
META = ["dragapult_ex","crustle","hydrapple_ex","ogerpon_box","n_s_zoroark_ex",
        "slowking","alakazam","raging_bolt_ex","lillie_s_clefairy_ex","rocket_s_honchkrow"]
GAMES = 20

def load_bare(deck_path):
    os.environ["BARE_DECK"] = deck_path
    import agents.bare_agent as ba
    importlib.reload(ba)
    return ba

def deck_request(mod):
    try:
        d = mod.agent({"select": None, "logs": [], "current": None})
        if isinstance(d, list) and len(d) == 60: return d
    except: pass
    return list(mod.my_deck)

def play_one(m0, m1, max_steps=10000):
    obs, st = battle_start(deck_request(m0), deck_request(m1))
    if obs is None: return -1
    pair = (m0,m1)
    try:
        for _ in range(max_steps):
            cur = obs.get("current")
            if cur is None: break
            if cur.get("result",-1)!=-1: return cur["result"]
            if obs.get("select") is None: break
            who = cur["yourIndex"]
            try: action = pair[who].agent(obs)
            except: return 1-who
            if not isinstance(action,list): action=list(action)
            obs = battle_select(action)
        return (obs.get("current") or {}).get("result",-1)
    finally: battle_finish()

mods = {v: importlib.import_module("agents."+v) for v in VARIANTS}
print(f"=== META BENCHMARK ({GAMES} games/deck x 10 decks) ===", flush=True)
short = {"dragapult_ex":"DRAG","crustle":"CRUS","hydrapple_ex":"HYDR","ogerpon_box":"OGER",
         "n_s_zoroark_ex":"ZORO","slowking":"SLOW","alakazam":"ALAK","raging_bolt_ex":"BOLT",
         "lillie_s_clefairy_ex":"LILL","rocket_s_honchkrow":"HONC"}
print(f"{'agent':<16}" + "".join(short[d].rjust(6) for d in META) + "  TOTAL")
for v in VARIANTS:
    me = mods[v]; tw=0
    cells=[]
    for slug in META:
        bare = load_bare(f"data/decks/deck_{slug}.json")
        w=0
        for g in range(GAMES):
            if g%2==0:
                r=play_one(me,bare)
                if r==0: w+=1
            else:
                r=play_one(bare,me)
                if r==1: w+=1
        tw+=w; cells.append(f"{w}/{GAMES}")
    total=GAMES*len(META)
    print(v.ljust(16)+"".join(c.rjust(6) for c in cells)+f"  {tw}/{total} ({100*tw/total:.0f}%)", flush=True)
