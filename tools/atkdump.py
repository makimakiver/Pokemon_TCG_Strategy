import json
from cg.api import all_attack
atks={a.attackId:a for a in all_attack()}
ids=json.load(open('tools/_ids.json'))
# include Crow's attack ids explicitly
crow=[652,653,1285,1286,583,669,670]
for aid in sorted(set(ids['atks'])|set(crow)):
    a=atks.get(aid)
    if not a: print(aid,"??"); continue
    es=''.join(str(e) for e in a.energies)
    print(f"{aid:>5} {a.name:<22} dmg={str(a.damage):<4} cost[{es}]\n        {a.text}")
