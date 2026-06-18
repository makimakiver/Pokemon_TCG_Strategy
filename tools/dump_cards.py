"""Dump the cabt engine card pool with NAMES so we can match meta archetypes."""
import _paths  # noqa: F401  (puts repo root on sys.path so `cg` resolves)
from cg.api import all_card_data, all_attack, CardType
from collections import defaultdict

cards = all_card_data()
attacks = {a.attackId: a for a in all_attack()}
print(f"TOTAL CARDS: {len(cards)}  ATTACKS: {len(attacks)}")

def is_pokemon(c): return int(c.cardType) == int(CardType.POKEMON)
def stage_of(c):
    if getattr(c, "stage2", False): return 2
    if getattr(c, "stage1", False): return 1
    return 0
def best_atk(c):
    best = None
    for aid in (getattr(c, "attacks", []) or []):
        a = attacks.get(aid)
        if a is None: continue
        d = getattr(a, "damage", 0) or 0
        ne = len(getattr(a, "energies", []) or [])
        if best is None or d > best[1]:
            best = (aid, d, ne, getattr(a, "name", ""))
    return best

# Do names exist?
named = [c for c in cards if getattr(c, "name", "")]
print(f"cards with non-empty name: {len(named)}")
print("sample names:", [getattr(c,'name','') for c in cards[200:210]])

# Index pokemon by name
print("\n######## ALL POKEMON NAMES (id stage hp type ex bestAtk) ########")
poke = sorted([c for c in cards if is_pokemon(c)], key=lambda c: getattr(c,'name','') or '')
for c in poke:
    nm = getattr(c, "name", "") or "?"
    ba = best_atk(c)
    ex = "EX" if getattr(c,"ex",False) else ("MEX" if getattr(c,"megaEx",False) else "")
    print(f"  id={c.cardId:5d} st{stage_of(c)} hp={getattr(c,'hp',0):4} ty={getattr(c,'energyType',None)} "
          f"{ex:3s} from={getattr(c,'evolvesFrom',None)} | {nm:24s} | atk={ba}")

print("\n######## TRAINERS / ENERGY NAMES ########")
for c in cards:
    if is_pokemon(c): continue
    print(f"  id={c.cardId:5d} type={CardType(int(c.cardType)).name:14s} | {getattr(c,'name','') or '?'}")
