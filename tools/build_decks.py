"""Resolve the 10 meta decklists into cabt card-id decks. Report ambiguity/misses."""
import json, re, sys, unicodedata
from collections import defaultdict
from _paths import DATA, DECKS_DIR
from decklists import DECKS

cards = json.load(open(DATA / "cards.json"))["cards"]

def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)   # drop apostrophes/punct
    s = re.sub(r"\s+", " ", s).strip()
    return s

by_name = defaultdict(list)
for c in cards:
    by_name[norm(c["name"])].append(c)

# Energy names in the meta lists -> cabt energy card ids.
ENERGY_ID = {
    "grass energy": 1, "fire energy": 2, "water energy": 3, "lightning energy": 4,
    "psychic energy": 5, "fighting energy": 6, "darkness energy": 7, "metal energy": 8,
    "growing grass energy": 18, "telepathic psychic energy": 19, "rocky fighting energy": 20,
    # special energies whose names already match are resolved normally
}

# Trainers absent from the cabt pool -> nearest functional substitute (documented).
SUBSTITUTE = {
    "Special Red Card": "Ultra Ball",          # disruption item -> generic search item
    "Transformation Tome": "Ultra Ball",       # N's draw-engine item -> generic search item
    "Prism Tower": "Area Zero Underdepths",     # stadium -> stadium already in that deck
}

OVERRIDE = {}   # (deck, name) -> id   OR   name -> id  (global)

def stage(c): return 2 if c["stage2"] else 1 if c["stage1"] else 0
def desc(c):
    ba = max((a for a in c["attacks"] if a.get("damage") is not None),
             key=lambda a: a.get("damage",0), default=None)
    return (f"id={c['id']} {c['cardTypeName'][:3]} st{stage(c)} hp{c['hp']} "
            f"ty{c['energyType']} {'EX' if c['ex'] else ('MEX' if c['megaEx'] else '')}"
            f" atk{ba.get('damage') if ba else '-'}")

def best_dmg(c):
    return max((a.get("damage",0) or 0 for a in c["attacks"]), default=0)

card_by_id = {c["id"]: c for c in cards}

def pick(deck, name, deck_etypes=frozenset()):
    name = SUBSTITUTE.get(name, name)
    key = norm(name)
    if key in ENERGY_ID:
        return ENERGY_ID[key], "energy", []
    if (deck, name) in OVERRIDE:
        return OVERRIDE[(deck, name)], "override", []
    if name in OVERRIDE:
        return OVERRIDE[name], "override", []
    cands = by_name.get(key, [])
    if not cands:
        return None, "MISS", []
    if len(cands) == 1:
        return cands[0]["id"], "ok", cands
    # Ambiguous print: prefer one whose energyType the deck actually runs, then
    # the higher-damage / higher-HP print (the "real" attacker version).
    ranked = sorted(cands, key=lambda c: (
        c.get("energyType") in deck_etypes, best_dmg(c), c.get("hp",0)), reverse=True)
    return ranked[0]["id"], "ambig", ranked

def deck_energy_types(lines):
    ets = set()
    for _, name in lines:
        k = norm(SUBSTITUTE.get(name, name))
        if k in ENERGY_ID:
            ets.add(card_by_id[ENERGY_ID[k]].get("energyType"))
        else:
            for c in by_name.get(k, []):
                if c["cardTypeName"] in ("BASIC_ENERGY","SPECIAL_ENERGY"):
                    ets.add(c.get("energyType"))
    return frozenset(ets)

def run(verbose=True):
    decks = {}
    problems = []
    for deck, lines in DECKS.items():
        ids = []
        etypes = deck_energy_types(lines)
        for count, name in lines:
            cid, status, cands = pick(deck, name, etypes)
            if cid is None:
                problems.append((deck, name, "MISS", []))
                if verbose: print(f"  [MISS ] {deck:22s} {name}")
                continue
            if status == "ambig":
                problems.append((deck, name, "ambig", [desc(c) for c in cands]))
                if verbose: print(f"  [ambig] {deck:22s} {name!r:30s} -> {cid}  ({len(cands)} prints: "
                                  + " | ".join(desc(c) for c in cands) + ")")
            ids += [cid]*count
        # Legality fix: cabt allows max 4 of any non-basic-energy card. Substitutions can
        # push a card over 4; convert the excess into the deck's primary basic energy.
        basic_e = [i for i in ids if card_by_id.get(i, {}).get("cardTypeName") == "BASIC_ENERGY"]
        prim = max(set(basic_e), key=basic_e.count) if basic_e else 1
        cnt = defaultdict(int)
        fixed = []
        for i in ids:
            is_basic_e = card_by_id.get(i, {}).get("cardTypeName") == "BASIC_ENERGY"
            cnt[i] += 1
            if not is_basic_e and cnt[i] > 4:
                fixed.append(prim)
                if verbose: print(f"  [legal] {deck:22s} {card_by_id[i]['name']} >4 -> +1 basic energy")
            else:
                fixed.append(i)
        ids = fixed

        decks[deck] = ids
        total = len(ids)
        miss = sum(1 for c,n in lines if pick(deck,n)[0] is None)
        if verbose: print(f"==> {deck:22s} total={total}  ({'OK' if total==60 else 'NOT 60'})  misses={miss}\n")
    return decks, problems

if __name__ == "__main__":
    decks, problems = run(verbose=True)
    DECKS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(decks, open(DATA / "decks.json", "w"), indent=0)
    # Split into per-deck files (consumed by bare_agent via BARE_DECK) + a slug map.
    slugs = {}
    for name, ids in decks.items():
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        assert len(ids) == 60, (name, len(ids))
        json.dump(ids, open(DECKS_DIR / f"deck_{slug}.json", "w"))
        slugs[name] = slug
    json.dump(slugs, open(DATA / "deck_slugs.json", "w"), indent=2)
    miss = [p for p in problems if p[2]=="MISS"]
    print(f"\nTOTAL MISSES: {len(miss)}")
    for d,n,_,_ in miss: print(f"   {d}: {n}")
    print(f"wrote decks.json + {len(slugs)} per-deck files in {DECKS_DIR}")
