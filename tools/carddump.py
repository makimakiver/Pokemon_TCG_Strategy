import json
from cg.api import all_card_data, all_attack
cards={c.cardId:c for c in all_card_data()}
atks={a.attackId:a for a in all_attack()}
ids=json.load(open('tools/_ids.json'))
print("=== CARDS ===")
for cid in ids['cards']:
    c=cards.get(cid)
    if not c: print(cid,"??"); continue
    tag='ex' if c.ex else ('mEX' if c.megaEx else '')
    print(f"{cid:>5} {c.name:<24}ct{str(c.cardType):<3} hp{str(c.hp):<4} {tag:<4} atks={c.attacks}")
print("=== ATTACKS ===")
for aid in ids['atks']:
    a=atks.get(aid)
    if not a: print(aid,"??"); continue
    es=''.join(e.name[0] for e in a.energies)
    print(f"{aid:>5} {a.name:<22} dmg{str(a.damage):<5} [{es}] :: {a.text[:95]}")
