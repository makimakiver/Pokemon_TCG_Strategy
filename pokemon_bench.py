"""pokemon_bench.py — single-file benchmark harness for the cabt sim.

Measures a set of CANDIDATE agents against the full fixed OPPONENT roster (every validated
benchmark deck) via the Docker runner, seat-swapped, and prints a win-rate matrix.

STEP 1 (this commit): define the roster — every deck/agent that goes into the benchmark — and a
deck-validity guard. The guard exists because bare_agent SILENTLY falls back to the Palace deck when
its BARE_DECK file is missing (this already corrupted a whole gauntlet once: bench_* decks vanished on
a branch-switch and every bench cell secretly tested Palace). `validate_roster()` fails loudly if any
opponent's deck file is missing or has degenerated into the Palace fallback.

Run:
  python3 pokemon_bench.py --validate        # step 1: prove the roster decks are real (no Docker)
  python3 pokemon_bench.py --run -n 100      # step 2 (next): run the full matrix in Docker
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
DECKS = os.path.join(REPO, "data", "decks")

# The Palace fallback deck baked into bare_agent.py / meta_opp.py — if an opponent's deck equals this,
# its real deck file is missing and the benchmark would silently test the wrong thing.
PALACE_FALLBACK = sorted([1]*19 + [11]*4 + [14]*4 + [18]*4 + [344]*4 + [345]*4 +
                         [1086]*4 + [1147]*4 + [1212]*4 + [1227]*4 + [1235]*4 + [1159])

# ─────────────────────────────────────────────────────────────────────────────
# OPPONENT ROSTER — the fixed "field" we measure candidates against.
# Each opponent is a self-loading agent MODULE (no shared BARE_DECK env → no deck collision).
# `deck` is the JSON the module loads (for the validity guard); None = intentionally the generic
# Palace field deck (meta_opp), which is expected and not a bug.
#   key       : dotted module path passed to runner.py --b
#   archetype : human label
#   deck      : deck file under data/decks/ (or None for the Palace generic field)
#   note      : why it's in the roster
# ─────────────────────────────────────────────────────────────────────────────
OPPONENTS = [
    # --- walrein's known hard matchups (measured holes) ---
    {"key": "agents.bench_starmie_cinderace", "archetype": "Mega Starmie ex / Cinderace", "deck": "deck_bench_starmie_cinderace.json", "note": "worst matchup (~95% vs walrein); real ladder deck"},
    {"key": "agents.meta_harlequin",          "archetype": "Harlequin",                   "deck": "deck_harlequin.json",                "note": "second 95% hole surfaced in gauntlet"},
    {"key": "agents.meta_starmie",            "archetype": "Mega Starmie ex",             "deck": "deck_starmie.json",                  "note": "water/Starmie family (~75%)"},
    {"key": "agents.bench_megalucario",       "archetype": "Mega Lucario ex (tuned pilot)","deck": "deck_bench_megalucario.json",        "note": "tuned-pilot Lucario (faithful to ladder); the REAL Lucario benchmark"},
    {"key": "agents.meta_dragapult",          "archetype": "Dragapult ex",                "deck": "deck_dragapult.json",                "note": "coin-flip tempo deck"},
    # --- favorable / mid matchups (range coverage) ---
    {"key": "agents.meta_tarountula",         "archetype": "Tarountula",                  "deck": "deck_tarountula.json",               "note": "near-even"},
    {"key": "agents.meta_walrein",            "archetype": "Walrein (mirror)",            "deck": "deck_walrein.json",                  "note": "mirror sanity check"},
    {"key": "agents.meta_crustle",            "archetype": "Crustle",                     "deck": "deck_crustle.json",                  "note": "favorable"},
    {"key": "agents.meta_colress_dunsparce",  "archetype": "Colress / Dunsparce",         "deck": "deck_colress_dunsparce.json",        "note": "favorable; top meta-share deck"},
    {"key": "agents.meta_bellibolt",          "archetype": "Iono's Bellibolt",            "deck": "deck_bellibolt.json",                "note": "favorable (combo, bare pilot under-plays it)"},
    {"key": "agents.meta_nighttime_mine",     "archetype": "Nighttime Mine (combo)",      "deck": "deck_nighttime_mine.json",           "note": "weak benchmark (combo needs its engine)"},
    {"key": "agents.meta_opp",                "archetype": "Generic field (Palace)",      "deck": None,                                 "note": "the loop's 'field' yardstick; intentionally the Palace generic deck"},
]
# Deliberately EXCLUDED: agents.mega_lucario — it's an untuned-pilot duplicate of bench_megalucario
# and caused exactly the tuned/untuned confusion we just debugged. bench_megalucario is the canonical
# Lucario benchmark. (Add it back as a separate weak-pilot baseline only if explicitly wanted.)

# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE ROSTER — the agents under test. Includes BOTH engine families on purpose:
#   agents.walrein_*           = the real bare_agent engine (what the loop optimizes)
#   submissions.submission_*   = the self-contained inline engine (what actually shipped to Kaggle)
# Keeping both exposes the ~44pt engine divergence vs the water decks.
# ─────────────────────────────────────────────────────────────────────────────
CANDIDATES = [
    {"key": "agents.walrein_v15", "code": "v15", "label": "v15 (bare_agent, sim champion)"},
    {"key": "agents.walrein_v21", "code": "v21", "label": "v21 (bare_agent, force/swap hybrid)"},
    {"key": "agents.walrein_v23", "code": "v23", "label": "v23 (bare_agent, ID-gated two-mode)"},
    {"key": "agents.walrein_v24", "code": "v24", "label": "v24 (bare_agent, race-during-charge)"},
    {"key": "submissions.submission_walrein_v22", "code": "sub22", "label": "sub_v22 (inline engine, on ladder)"},
    {"key": "submissions.submission_walrein_v23", "code": "sub23", "label": "sub_v23 (inline engine, faithful port)"},
]


def _load_deck(deckfile):
    with open(os.path.join(DECKS, deckfile)) as f:
        return json.load(f)


def validate_roster():
    """Fail loudly if any opponent deck is missing or has degenerated to the Palace fallback."""
    ok = True
    print(f"{'opponent module':<38}{'archetype':<30}{'deck':<34}{'status'}")
    print("-" * 120)
    for o in OPPONENTS:
        deckfile = o["deck"]
        if deckfile is None:
            status = "OK (Palace field, by design)"
        else:
            path = os.path.join(DECKS, deckfile)
            if not os.path.exists(path):
                status = "❌ MISSING DECK → would Palace-fallback"; ok = False
            else:
                deck = _load_deck(deckfile)
                if len(deck) != 60:
                    status = f"❌ {len(deck)} cards (expected 60)"; ok = False
                elif sorted(deck) == PALACE_FALLBACK:
                    status = "❌ IS the Palace fallback deck"; ok = False
                else:
                    status = "✅ ok (60, real deck)"
        print(f"{o['key']:<38}{o['archetype']:<30}{str(deckfile):<34}{status}")
    print("-" * 120)
    print(f"opponents: {len(OPPONENTS)}   candidates: {len(CANDIDATES)}   "
          f"matrix: {len(OPPONENTS) * len(CANDIDATES)} matchups")
    print("ROSTER VALID ✅" if ok else "ROSTER INVALID ❌ — fix decks before benchmarking")
    return ok


import re

IMAGE = "cabt-sim"
RESULT_RE = re.compile(r"\(([\d.]+)%\)")


def run_matchup(cand_key, opp_key, n, timeout=1800):
    """Run one candidate-vs-opponent match (n games, seat-swapped) in Docker.
    Returns {wins, games, pct, draws} from the CANDIDATE's perspective (side A), or None on error."""
    cmd = ["docker", "run", "--rm", "--platform=linux/amd64", "-v", f"{REPO}:/app", IMAGE,
           "--a", cand_key, "--b", opp_key, "-n", str(n)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return None
    wins = pct = draws = None
    # candidate is side A: line looks like "  A (agents.walrein_v23): 61  (61.0%)"
    a_re = re.compile(r"A \(" + re.escape(cand_key) + r"\):\s*(\d+)\s*\(([\d.]+)%\)")
    for line in out.splitlines():
        m = a_re.search(line)
        if m:
            wins, pct = int(m.group(1)), float(m.group(2))
        if line.strip().startswith("draws:"):
            try:
                draws = int(line.split(":")[1])
            except (ValueError, IndexError):
                pass
    if pct is None:
        return None
    return {"wins": wins, "games": n, "pct": pct, "draws": draws}


def run_matrix(n, candidates, opponents, out_path=None):
    """Run the full candidate × opponent matrix, print a win-rate table, return results."""
    results = {}  # (cand_code, opp_key) -> result dict
    total = len(candidates) * len(opponents)
    done = 0
    print(f"Running {total} matchups @ n={n} each ({total * n} games). Candidate = side A (win% shown).\n")
    for opp in opponents:
        for c in candidates:
            done += 1
            print(f"  [{done}/{total}] {c['code']:>6} vs {opp['key'].split('.')[-1]:<26} ", end="", flush=True)
            r = run_matchup(c["key"], opp["key"], n)
            results[(c["code"], opp["key"])] = r
            print(f"{r['pct']:.0f}%" if r else "ERR")

    # ── matrix table: rows = opponents, cols = candidate codes ──
    codes = [c["code"] for c in candidates]
    w_opp = max(len(o["archetype"]) for o in opponents) + 1
    header = "opponent".ljust(w_opp) + "".join(f"{c:>7}" for c in codes)
    print("\n" + "=" * len(header)); print(header); print("-" * len(header))
    col_sum = {c: [] for c in codes}
    for opp in opponents:
        row = opp["archetype"].ljust(w_opp)
        for c in candidates:
            r = results[(c["code"], opp["key"])]
            if r:
                row += f"{r['pct']:>6.0f}%"; col_sum[c["code"]].append(r["pct"])
            else:
                row += f"{'ERR':>7}"
        print(row)
    print("-" * len(header))
    avg_row = "AVG (mean win%)".ljust(w_opp)
    for c in codes:
        vals = col_sum[c]
        avg_row += (f"{sum(vals)/len(vals):>6.0f}%" if vals else f"{'-':>7}")
    print(avg_row); print("=" * len(header))

    if out_path:
        flat = [{"candidate": code, "opponent": opp, **(r or {"error": True})}
                for (code, opp), r in results.items()]
        with open(out_path, "w") as f:
            json.dump({"n": n, "results": flat}, f, indent=1)
        print(f"\nfull results → {out_path}")
    return results


def _filter(roster, codes_or_keys, field):
    if not codes_or_keys:
        return roster
    wanted = set(codes_or_keys.split(","))
    return [r for r in roster if r[field] in wanted or r["key"] in wanted or r["key"].split(".")[-1] in wanted]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="cabt benchmark matrix")
    ap.add_argument("--validate", action="store_true", help="step 1: check the roster decks are real")
    ap.add_argument("--run", action="store_true", help="step 2: run the full candidate × opponent matrix in Docker")
    ap.add_argument("-n", type=int, default=100, help="games per matchup (default 100)")
    ap.add_argument("--candidates", help="comma-separated candidate codes/keys to limit to (e.g. v23,sub23)")
    ap.add_argument("--opponents", help="comma-separated opponent keys/module-names to limit to")
    ap.add_argument("--out", help="write full results JSON here")
    args = ap.parse_args()

    if args.run:
        if not validate_roster():
            print("\nrefusing to run — fix the roster decks first."); sys.exit(1)
        print()
        cands = _filter(CANDIDATES, args.candidates, "code")
        opps = _filter(OPPONENTS, args.opponents, "archetype")
        run_matrix(args.n, cands, opps, out_path=args.out)
        sys.exit(0)
    # default + --validate: print and validate the chosen roster
    sys.exit(0 if validate_roster() else 1)
