"""Export the full cabt card pool to cards.json for offline deck mapping."""
import json
from _paths import DATA           # puts repo root on sys.path so `cg` resolves
from cg.api import all_card_data, all_attack, CardType

attacks = {}
for a in all_attack():
    attacks[a.attackId] = {
        "id": a.attackId,
        "name": getattr(a, "name", ""),
        "damage": getattr(a, "damage", 0),
        "energies": [int(e) for e in (getattr(a, "energies", []) or [])],
        "text": getattr(a, "text", ""),
    }

out = []
for c in all_card_data():
    out.append({
        "id": c.cardId,
        "name": getattr(c, "name", ""),
        "cardType": int(c.cardType),
        "cardTypeName": CardType(int(c.cardType)).name,
        "hp": getattr(c, "hp", 0),
        "energyType": (int(c.energyType) if getattr(c, "energyType", None) is not None else None),
        "weakness": (int(c.weakness) if getattr(c, "weakness", None) is not None else None),
        "resistance": (int(c.resistance) if getattr(c, "resistance", None) is not None else None),
        "retreatCost": getattr(c, "retreatCost", 0),
        "stage0": not (getattr(c, "stage1", False) or getattr(c, "stage2", False)),
        "stage1": bool(getattr(c, "stage1", False)),
        "stage2": bool(getattr(c, "stage2", False)),
        "ex": bool(getattr(c, "ex", False)),
        "megaEx": bool(getattr(c, "megaEx", False)),
        "evolvesFrom": getattr(c, "evolvesFrom", None),
        "attacks": [attacks.get(aid, {"id": aid}) for aid in (getattr(c, "attacks", []) or [])],
    })

with open(DATA / "cards.json", "w") as f:
    json.dump({"cards": out, "attacks": attacks}, f)
print(f"wrote {DATA / 'cards.json'}: {len(out)} cards, {len(attacks)} attacks")
