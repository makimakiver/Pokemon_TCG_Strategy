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

# ── ROSTER: auto-discover EVERY distinct, legal 60-card deck in data/decks/ ──
# Every bare-agent deck is included automatically. Exact-content duplicates are collapsed (the
# first filename wins) so we never waste runs on identical decks (e.g. deck_walrein == deck_walrein_real).
# Candidates and opponents are the SAME full set → every deck is scored against the whole field.
def discover_decks():
    import glob
    seen, roster = {}, []
    for f in sorted(glob.glob(os.path.join(DECKS, "*.json"))):
        base = os.path.basename(f)
        if base.endswith(".meta.json"):
            continue
        try:
            deck = json.load(open(f))
        except Exception:
            continue
        if not isinstance(deck, list) or len(deck) != 60 or sorted(deck) == PALACE:
            continue
        key = tuple(sorted(deck))
        if key in seen:        # exact duplicate of an already-included deck → skip
            continue
        seen[key] = True
        name = base[:-5].replace("deck_", "")
        roster.append({"name": name, "deck": base})
    return roster

ROSTER = discover_decks()
CANDIDATES = ROSTER
OPPONENTS = ROSTER


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
    bad = []
    for d in ROSTER:
        st = _check_deck(d["deck"])
        if st != "ok":
            ok = False; bad.append((d["deck"], st))
    print(f"roster: {len(ROSTER)} distinct legal decks → {len(ROSTER)*len(ROSTER)} matchups")
    if bad:
        for f, st in bad:
            print(f"  ❌ {f}: {st}")
    else:
        print("  ✅ all decks legal (60 cards, no Palace fallback)")
    return ok


def validate_legality():
    """Run battle_start on every roster deck in Docker — catches the illegal decks that host-side
    checks (60 cards / not-Palace) miss. An illegal deck fails battle_start and silently scores 0%
    (this exact bug made deck_fire_ceruledge look like a 'hard counter' — errorType 2, 8 Charcadet)."""
    code = """
import json
from cg.game import battle_start, battle_finish
bad=[]
for b in {decks!r}.split(','):
    d=json.load(open('/app/data/decks/'+b))
    o,s=battle_start(list(d),list(d))
    try:
        battle_finish()
    except Exception:
        pass
    if o is None:
        bad.append((b, getattr(s,'errorType',None)))
print('ILLEGAL:', bad if bad else 'NONE')
""".format(decks=",".join(d["deck"] for d in ROSTER))
    out = subprocess.run(
        ["docker", "run", "--rm", "--platform=linux/amd64", "-v", f"{REPO}:/app",
         "--entrypoint", "python", IMAGE, "-c", code],
        capture_output=True, text=True, timeout=600).stdout
    line = next((l for l in out.splitlines() if l.startswith("ILLEGAL:")), "ILLEGAL: ?")
    passed = "NONE" in line
    print(f"legality (battle_start): {'✅ all decks legal' if passed else '❌ ' + line}")
    return passed


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
         "-e", f"BENCH_A_DECK=/app/data/decks/{ROSTER[0]['deck']}",
         "-e", f"BENCH_B_DECK=/app/data/decks/{ROSTER[1]['deck']}",
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
        legal_ok = validate_legality()
        coll_ok = validate_collision()
    else:
        legal_ok = coll_ok = True
    if args.run:
        if not (decks_ok and legal_ok and coll_ok):
            print("\n❌ refusing to run — fix decks/legality/collision first."); sys.exit(1)
        print()
        run_matrix(args.n, _filter(CANDIDATES, args.candidates), _filter(OPPONENTS, args.opponents), args.out)
        sys.exit(0)
    sys.exit(0 if decks_ok and legal_ok and coll_ok else 1)
