"""pokemon_bench.py — collision-proof deck benchmark for the cabt sim.

Measures how well each CANDIDATE deck performs against the OPPONENT field, with a UNIFORM generic
pilot on BOTH sides — so the result reflects DECK strength, not pilot skill. Answers: "which deck,
if we adopt it, clears ~X% vs the meta field?"

──────────────────────────────────────────────────────────────────────────────────────────────
WHY THIS DESIGN (read before trusting any number):
  The naive approach — gauntlet two bare-deck agents (agents.meta_X vs agents.meta_Y) — is BROKEN.
  Both read the same BARE_DECK env var at import; the runner imports both into one process, so the
  SECOND agent silently loads the FIRST agent's deck (a mirror). This corrupted earlier gauntlets
  (a walrein candidate made every "opponent" play walrein → fake 58% "wins"). Proven via import test.

  FIX: each side is a DISTINCT module reading a DISTINCT env var:
     side A = agents._bench_a   (reads BENCH_A_DECK)   ← candidate deck
     side B = agents._bench_b   (reads BENCH_B_DECK)   ← opponent deck
  Different module objects + different env vars = no shared deck state. Both are copies of the
  self-contained generic pilot (meta_opp.py), so the pilot is identical on both sides and only the
  DECK differs. `validate_collision()` asserts the two sides load different decks before any run.

  CAVEAT: combo decks (Nighttime Mine, etc.) are UNDER-piloted by a generic engine, so their sim
  number understates their real-pilot ceiling. Cross-check with the Kaggle-meta tracker win rate
  (the `meta_wr` field below) which reflects real-pilot ladder performance.
──────────────────────────────────────────────────────────────────────────────────────────────

Run:
  python3 pokemon_bench.py --validate              # decks legal + collision-proof self-test (Docker)
  python3 pokemon_bench.py --run -n 100            # full candidate × opponent matrix
  python3 pokemon_bench.py --run -n 100 --candidates hoptrevenant,walrein --out out/bench.json
"""
import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
DECKS = os.path.join(REPO, "data", "decks")
IMAGE = "cabt-sim"
SIDE_A, SIDE_B = "agents._bench_a", "agents._bench_b"
PALACE = sorted([1]*19 + [11]*4 + [14]*4 + [18]*4 + [344]*4 + [345]*4 +
                [1086]*4 + [1147]*4 + [1212]*4 + [1227]*4 + [1235]*4 + [1159])

# ── CANDIDATE decks: what WE could adopt and run with the generic engine ──
#   name : key used on the CLI and as the matrix row
#   deck : JSON under data/decks/
#   meta_wr : Kaggle-meta tracker ladder win rate (2026-06-23) — real-pilot signal, sim-independent
CANDIDATES = [
    {"name": "walrein",          "deck": "deck_walrein.json",                 "meta_wr": "74% (our current sub)"},
    {"name": "hoptrevenant",     "deck": "deck_cand_hoptrevenant.json",       "meta_wr": "62.3% / 848g (top robust)"},
    {"name": "kadabra_alakazam", "deck": "deck_cand_kadabra_alakazam.json",   "meta_wr": "61.4% / 207g"},
    {"name": "abra_kadabra",     "deck": "deck_cand_abra_kadabra.json",        "meta_wr": "59.6% / 617g"},
    {"name": "enrich_nighttime", "deck": "deck_cand_enrich_nighttime.json",   "meta_wr": "64.7% / 102g (combo)"},
    {"name": "colress_dunsparce","deck": "deck_colress_dunsparce.json",        "meta_wr": "prior bare champ"},
    {"name": "harlequin",        "deck": "deck_harlequin.json",                "meta_wr": "beats walrein 95%"},
]

# ── OPPONENT field: the meta we'd face on the ladder ──
OPPONENTS = [
    {"name": "starmie_cinderace", "deck": "deck_bench_starmie_cinderace.json", "label": "Mega Starmie / Cinderace"},
    {"name": "starmie",           "deck": "deck_starmie.json",                 "label": "Mega Starmie ex"},
    {"name": "harlequin",         "deck": "deck_harlequin.json",               "label": "Harlequin"},
    {"name": "megalucario",       "deck": "deck_bench_megalucario.json",       "label": "Mega Lucario ex"},
    {"name": "dragapult",         "deck": "deck_dragapult.json",               "label": "Dragapult ex"},
    {"name": "crustle",           "deck": "deck_crustle.json",                 "label": "Crustle"},
    {"name": "colress_dunsparce", "deck": "deck_colress_dunsparce.json",       "label": "Colress / Dunsparce"},
    {"name": "bellibolt",         "deck": "deck_bellibolt.json",               "label": "Iono's Bellibolt"},
    {"name": "tarountula",        "deck": "deck_tarountula.json",              "label": "Tarountula"},
    {"name": "walrein",           "deck": "deck_walrein.json",                 "label": "Walrein (mirror)"},
]


def _deck_path(deckfile):
    return os.path.join(DECKS, deckfile)


def _check_deck(deckfile):
    p = _deck_path(deckfile)
    if not os.path.exists(p):
        return "MISSING"
    deck = json.load(open(p))
    if len(deck) != 60:
        return f"{len(deck)}≠60"
    if sorted(deck) == PALACE:
        return "PALACE-FALLBACK"
    return "ok"


def validate_decks():
    ok = True
    seen = {}
    print(f"{'deck file':<38}{'status':<18}role")
    print("-" * 70)
    for role, roster in (("candidate", CANDIDATES), ("opponent", OPPONENTS)):
        for d in roster:
            st = seen.get(d["deck"]) or _check_deck(d["deck"])
            seen[d["deck"]] = st
            if st != "ok":
                ok = False
            print(f"{d['deck']:<38}{('✅ '+st if st=='ok' else '❌ '+st):<18}{role}")
    print("-" * 70)
    return ok


def validate_collision():
    """Prove side A and side B load DIFFERENT decks (the whole point of the redesign)."""
    code = (
        "import importlib,json,os;"
        "CARDS={c['id']:c['name'] for c in json.load(open('/app/data/cards.json'))['cards']};"
        "A=importlib.import_module('agents._bench_a');B=importlib.import_module('agents._bench_b');"
        "print('A_len',len(A.my_deck),'B_len',len(B.my_deck),'SAME' if A.my_deck==B.my_deck else 'DIFFERENT')"
    )
    out = subprocess.run(
        ["docker", "run", "--rm", "--platform=linux/amd64", "-v", f"{REPO}:/app",
         "-e", f"BENCH_A_DECK=/app/data/decks/{CANDIDATES[0]['deck']}",
         "-e", f"BENCH_B_DECK=/app/data/decks/{OPPONENTS[0]['deck']}",
         "--entrypoint", "python", IMAGE, "-c", code],
        capture_output=True, text=True, timeout=300).stdout
    passed = "DIFFERENT" in out
    print(f"collision self-test: {out.strip().splitlines()[-1] if out.strip() else 'no output'}  "
          f"→ {'✅ collision-proof' if passed else '❌ COLLISION'}")
    return passed


def run_matchup(deck_a, deck_b, n, timeout=2400):
    cmd = ["docker", "run", "--rm", "--platform=linux/amd64", "-v", f"{REPO}:/app",
           "-e", f"BENCH_A_DECK=/app/data/decks/{deck_a}",
           "-e", f"BENCH_B_DECK=/app/data/decks/{deck_b}",
           IMAGE, "--a", SIDE_A, "--b", SIDE_B, "-n", str(n)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r"A \(" + re.escape(SIDE_A) + r"\):\s*(\d+)\s*\(([\d.]+)%\)", out)
    return {"wins": int(m.group(1)), "pct": float(m.group(2)), "n": n} if m else None


def run_matrix(n, candidates, opponents, out_path=None):
    total = len(candidates) * len(opponents)
    done = 0
    results = {}
    print(f"Matrix: {len(candidates)} candidate decks × {len(opponents)} opponents @ n={n} "
          f"({total} matchups, {total*n} games). Uniform generic pilot both sides → DECK strength.\n")
    for c in candidates:
        for o in opponents:
            done += 1
            print(f"  [{done}/{total}] {c['name']:>16} vs {o['name']:<20} ", end="", flush=True)
            r = run_matchup(c["deck"], o["deck"], n)
            results[(c["name"], o["name"])] = r
            print(f"{r['pct']:.0f}%" if r else "ERR")

    w = max(len(c["name"]) for c in candidates) + 1
    onames = [o["name"][:9] for o in opponents]
    print("\n" + "=" * (w + 9*len(onames) + 8))
    print("candidate deck".ljust(w) + "".join(f"{x:>10}" for x in onames) + f"{'AVG':>8}")
    print("-" * (w + 9*len(onames) + 8))
    for c in candidates:
        vals = []
        row = c["name"].ljust(w)
        for o in opponents:
            r = results[(c["name"], o["name"])]
            if r:
                row += f"{r['pct']:>9.0f}%"; vals.append(r["pct"])
            else:
                row += f"{'ERR':>10}"
        row += f"{(sum(vals)/len(vals) if vals else 0):>7.0f}%"
        print(row)
    print("=" * (w + 9*len(onames) + 8))
    print("AVG = candidate's mean win% across the field. Target: a deck with AVG ≥ 70%.")

    if out_path:
        flat = [{"candidate": c, "opponent": o, **(r or {"error": True})} for (c, o), r in results.items()]
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        json.dump({"n": n, "results": flat}, open(out_path, "w"), indent=1)
        print(f"full results → {out_path}")
    return results


def _filter(roster, csv):
    if not csv:
        return roster
    want = set(csv.split(","))
    return [r for r in roster if r["name"] in want]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="collision-proof cabt deck benchmark")
    ap.add_argument("--validate", action="store_true", help="check decks + collision self-test")
    ap.add_argument("--run", action="store_true", help="run the candidate × opponent matrix")
    ap.add_argument("-n", type=int, default=100, help="games per matchup")
    ap.add_argument("--candidates", help="comma-separated candidate names to limit to")
    ap.add_argument("--opponents", help="comma-separated opponent names to limit to")
    ap.add_argument("--out", help="write full results JSON here")
    args = ap.parse_args()

    decks_ok = validate_decks()
    if args.run or args.validate:
        coll_ok = validate_collision()
    else:
        coll_ok = True
    if args.run:
        if not (decks_ok and coll_ok):
            print("\n❌ refusing to run — fix decks/collision first."); sys.exit(1)
        print()
        run_matrix(args.n, _filter(CANDIDATES, args.candidates), _filter(OPPONENTS, args.opponents), args.out)
        sys.exit(0)
    sys.exit(0 if decks_ok and coll_ok else 1)
