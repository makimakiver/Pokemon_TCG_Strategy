import _paths  # noqa: F401  (puts repo root on sys.path so `cg` resolves)
from cg.api import all_card_data
ct={0:'POKEMON',1:'ITEM',2:'TOOL',3:'SUPPORTER',4:'STADIUM',5:'BASIC_ENERGY',6:'SPECIAL_ENERGY'}
cards=all_card_data()
for c in cards:
    n=c.name.lower()
    if 'rocket' in n or 'honchkrow' in n or 'murkrow' in n:
        extra=''
        if c.cardType==0:
            stage='Basic' if c.basic else 'St1' if c.stage1 else 'St2' if c.stage2 else '?'
            extra=f" {stage} hp={c.hp} ex={c.ex} evolvesFrom={c.evolvesFrom!r} attacks={c.attacks}"
        print(f"id={c.cardId:5d} {ct.get(c.cardType,''):12s} '{c.name}'{extra}")
