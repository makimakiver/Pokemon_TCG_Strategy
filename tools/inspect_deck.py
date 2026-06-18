"""Full detail of a deck's cards: attacks (dmg/cost/text) + abilities. Usage: inspect_deck.py <deckfile>"""
import sys, json
from collections import Counter
import _paths  # noqa: F401
from cg.api import all_card_data, all_attack, CardType, EnergyType

deckfile = sys.argv[1] if len(sys.argv) > 1 else str(_paths.DECKS_DIR / "deck_rocket_s_honchkrow.json")
ids = json.load(open(deckfile))
cnt = Counter(ids)
cards = {c.cardId: c for c in all_card_data()}
atk = {a.attackId: a for a in all_attack()}
ET = {0:'C',1:'G',2:'R',3:'W',4:'L',5:'P',6:'F',7:'D',8:'M',9:'N',10:'Rainbow',11:'TR'}

def et(e): return ET.get(int(e), str(e)) if e is not None else '-'

print(f"DECK {deckfile}  ({len(ids)} cards, {len(cnt)} unique)\n")
order = sorted(cnt, key=lambda i: (cards[i].cardType, -cards[i].hp, i))
for i in order:
    c = cards[i]
    n = cnt[i]
    if c.cardType == CardType.POKEMON:
        stg = 'Basic' if c.basic else 'St1' if c.stage1 else 'St2' if c.stage2 else '?'
        print(f"{n}x id={i} '{c.name}' [{stg} hp={c.hp} type={et(c.energyType)} weak={et(c.weakness)} "
              f"ex={c.ex} retreat={c.retreatCost} from={c.evolvesFrom!r}]")
        for aid in c.attacks:
            a = atk.get(aid)
            if a:
                cost = ''.join(et(e) for e in a.energies)
                print(f"      ATK '{a.name}' dmg={a.damage} cost=[{cost}] :: {a.text}")
        for s in (c.skills or []):
            print(f"      ABILITY '{s.name}' :: {s.text}")
    else:
        print(f"{n}x id={i} '{c.name}' [{CardType(int(c.cardType)).name}]")
        for s in (c.skills or []):
            print(f"      :: {s.text}")
