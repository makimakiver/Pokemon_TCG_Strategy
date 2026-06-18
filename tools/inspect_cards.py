import _paths  # noqa: F401  (puts repo root on sys.path so `cg` resolves)
from cg.api import all_card_data, all_attack
ct={0:'POKEMON',1:'ITEM',2:'TOOL',3:'SUPPORTER',4:'STADIUM',5:'BASIC_ENERGY',6:'SPECIAL_ENERGY'}
et={0:'COLORLESS',1:'GRASS',2:'FIRE',3:'WATER',4:'LIGHTNING',5:'PSYCHIC',6:'FIGHTING',7:'DARK',8:'METAL',9:'DRAGON',10:'RAINBOW',11:'TEAM_ROCKET'}
cards={c.cardId:c for c in all_card_data()}
atks={a.attackId:a for a in all_attack()}
deck=[1,11,14,18,344,345,1086,1147,1212,1227,1235,1159]
for cid in deck:
    c=cards.get(cid)
    if not c: print(cid,'MISSING'); continue
    line=f"id={cid:5d} {ct.get(c.cardType,c.cardType):13s} '{c.name}'"
    if c.cardType==0:
        stage='Basic' if c.basic else 'Stage1' if c.stage1 else 'Stage2' if c.stage2 else '?'
        line+=f" {stage} hp={c.hp} weak={et.get(c.weakness)} ex={c.ex} megaEx={c.megaEx}"
        line+=f" attacks={[(a, atks[a].name if a in atks else '?', atks[a].damage if a in atks else '?', [et.get(e) for e in (atks[a].energies if a in atks else [])]) for a in c.attacks]}"
    elif c.cardType in (5,6):
        line+=f" energyType={et.get(c.energyType)}"
    print(line)
