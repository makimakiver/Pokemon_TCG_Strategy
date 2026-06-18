#!/usr/bin/env python3
"""Generate an Excalidraw flowchart of the whole agents/main_v1.py control flow.

Writes docs/diagrams/agents-main_v1-flow.excalidraw; import at excalidraw.com.
Shapes: ellipse = start/end, rectangle = process, diamond = decision,
dashed box = expanded sub-detail (attack-plan loop + per-option scoring switch).
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "diagrams", "agents-main_v1-flow.excalidraw")
elements = []
_seed = 3000


def _ns():
    global _seed
    _seed += 7
    return _seed


def _base(eid, etype, x, y, w, h, **kw):
    el = {
        "id": eid, "type": etype, "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": "#1e1e1e", "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": _ns(), "version": 1, "versionNonce": _ns(),
        "isDeleted": False, "boundElements": [], "updated": 1, "link": None,
        "locked": False,
    }
    el.update(kw)
    return el


def _text(eid, x, y, w, h, s, size, color, align="center", font=2):
    lines = s.split("\n")
    th = len(lines) * size * 1.25
    ty = y + (h - th) / 2
    tx = x + (8 if align == "left" else 0)
    elements.append(_base(eid, "text", tx, ty, w - (16 if align == "left" else 0),
                          len(lines) * size * 1.25, strokeColor=color, text=s,
                          fontSize=size, fontFamily=font, textAlign=align,
                          verticalAlign="top", baseline=int(size * 0.9),
                          containerId=None, originalText=s, lineHeight=1.25))


RECTS = {}


def NODE(eid, kind, x, y, w, h, label, stroke, bg, tsize=13, tcolor="#1e1e1e",
         dashed=False, align="center", font=2):
    etype = {"start": "ellipse", "end": "ellipse", "proc": "rectangle",
             "dec": "diamond", "box": "rectangle"}[kind]
    rnd = {"type": 3} if etype == "rectangle" else None
    elements.append(_base(eid, etype, x, y, w, h, strokeColor=stroke,
                          backgroundColor=bg, fillStyle="solid",
                          strokeStyle="dashed" if dashed else "solid",
                          roundness=rnd, strokeWidth=2))
    if label:
        _text("t-" + eid, x, y, w, h, label, tsize, tcolor, align=align, font=font)
    RECTS[eid] = (x, y, w, h)
    return eid


def C(eid):
    x, y, w, h = RECTS[eid]
    return {"t": (x + w / 2, y), "b": (x + w / 2, y + h),
            "l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "c": (x + w / 2, y + h / 2)}


def ARROW(eid, p1, p2, color="#495057", dashed=False):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    elements.append(_base(eid, "arrow", x1, y1, abs(dx) or 1, abs(dy) or 1,
                          strokeColor=color, strokeWidth=2, roundness={"type": 2},
                          strokeStyle="dashed" if dashed else "solid",
                          points=[[0, 0], [dx, dy]], lastCommittedPoint=None,
                          startBinding=None, endBinding=None,
                          startArrowhead=None, endArrowhead="arrow"))


def LBL(eid, x, y, s, size=12, color="#868e96"):
    elements.append(_base(eid, "text", x, y, 160, size * 1.25, strokeColor=color,
                          text=s, fontSize=size, fontFamily=2, textAlign="left",
                          verticalAlign="top", baseline=int(size * 0.9),
                          containerId=None, originalText=s, lineHeight=1.25))


GREEN, BLUE, AMBER, GRAY = "#2f9e44", "#1971c2", "#e8590c", "#868e96"
BG_G, BG_B, BG_A, BG_GR = "#ebfbee", "#e7f5ff", "#fff9db", "#f1f3f5"

# ---- title + legend ----
LBL("title", 150, 30, "agents/main_v1.py  —  agent(obs_dict) control flow", size=26, color="#1e1e1e")
NODE("lg", "box", 560, 28, 430, 70, "LEGEND   ( ) start/end   [ ] process   <> decision\n- - - dashed box = expanded sub-detail",
     GRAY, "#ffffff", tsize=11, tcolor=GRAY, dashed=True, align="left")

# ---- module-load note ----
NODE("mod", "box", 150, 120, 300, 60,
     "module load: my_deck (60 ids),\nrole constants, card_table / attack_table",
     GRAY, BG_GR, tsize=11, tcolor=GRAY, dashed=True)

# ---- main spine (left column, x 150..450) ----
NODE("n1", "start", 200, 210, 200, 60, "agent(obs_dict)", GREEN, BG_G, tsize=15)
NODE("n2", "proc", 150, 300, 300, 54, "obs = to_observation_class(obs_dict)", BLUE, BG_B, tsize=12)
NODE("d1", "dec", 175, 388, 250, 104, "obs.select\nis None?", AMBER, BG_A, tsize=13)
NODE("n3", "proc", 150, 532, 300, 66, "read state, select, context,\nmy/opp index, prize counts", BLUE, BG_B, tsize=12)
NODE("d2", "dec", 175, 632, 250, 104, "pre_turn !=\nstate.turn?", AMBER, BG_A, tsize=13)
NODE("n4", "proc", 150, 776, 300, 66, "count field + hand cards;\ncompute have_spare_energy", BLUE, BG_B, tsize=12)
NODE("d3", "dec", 175, 876, 250, 104, "context ==\nMAIN?", AMBER, BG_A, tsize=13)
NODE("n5", "proc", 150, 1020, 300, 54, "define energy_score()\n(feed the attacker)", BLUE, BG_B, tsize=12)
NODE("n6", "proc", 150, 1110, 300, 66, "SCORE EVERY OPTION\nswitch on o.type", BLUE, BG_B, tsize=13)
NODE("n7", "proc", 150, 1212, 300, 52, "sort options by score (desc)", BLUE, BG_B, tsize=12)
NODE("n8", "proc", 150, 1300, 300, 56, "take top within\n[minCount, maxCount]", BLUE, BG_B, tsize=12)
NODE("n9", "end", 200, 1396, 200, 60, "return chosen", GREEN, BG_G, tsize=15)

# branch exits (left)
NODE("ret_deck", "end", -150, 400, 270, 64, "return my_deck\n(submit 60-card deck)", GREEN, BG_G, tsize=12)
NODE("reset", "box", -150, 650, 270, 64, "plan = AttackPlan()\npre_turn = state.turn", GRAY, BG_GR, tsize=12)

# ---- spine arrows ----
ARROW("a-mod", C("mod")["b"], C("n1")["t"], color=GRAY, dashed=True)
ARROW("a12", C("n1")["b"], C("n2")["t"])
ARROW("a2d1", C("n2")["b"], C("d1")["t"])
ARROW("ad1yes", C("d1")["l"], C("ret_deck")["r"], color=GREEN)
ARROW("ad1no", C("d1")["b"], C("n3")["t"])
ARROW("a3d2", C("n3")["b"], C("d2")["t"])
ARROW("ad2yes", C("d2")["l"], C("reset")["r"], color=GRAY)
ARROW("areset", C("reset")["b"], (RECTS["n4"][0] + 40, RECTS["n4"][1]), color=GRAY)
ARROW("ad2no", C("d2")["b"], C("n4")["t"])
ARROW("a4d3", C("n4")["b"], C("d3")["t"])
ARROW("ad3no", C("d3")["b"], C("n5")["t"])
ARROW("a56", C("n5")["b"], C("n6")["t"])
ARROW("a67", C("n6")["b"], C("n7")["t"])
ARROW("a78", C("n7")["b"], C("n8")["t"])
ARROW("a89", C("n8")["b"], C("n9")["t"])
LBL("l-d1y", 70, 415, "yes (first call)", color=GREEN)
LBL("l-d1n", 305, 498, "no", color=AMBER)
LBL("l-d2y", 70, 660, "yes (new turn)", color=GRAY)
LBL("l-d2n", 305, 742, "no", color=AMBER)
LBL("l-d3y", 432, 905, "yes", color=GREEN)
LBL("l-d3n", 305, 986, "no", color=AMBER)

# ============ DETAIL 1: BUILD ATTACK PLAN (context == MAIN) ============
NODE("ap", "box", 560, 300, 440, 520, "", AMBER, "#fffaf2", dashed=True)
LBL("ap-h", 574, 312, "BUILD ATTACK PLAN   (context == MAIN)", size=13, color=AMBER)


def inner(eid, y, h, label, x=580, w=400, stroke=AMBER, bg="#ffffff", size=11):
    NODE(eid, "box", x, y, w, h, label, stroke, bg, tsize=size, align="left")


inner("ap1", 344, 40, "for each of MY Pokemon (active + bench):")
inner("ap2", 392, 40, "  skip a benched one unless can_switch (RETREAT offered)")
inner("ap3", 440, 40, "  for each attack of that Pokemon:")
inner("ap4", 488, 52, "    can_pay(cost)?  else assume +1 basic energy\n    (only if have_spare_energy and not energyAttached)")
inner("ap5", 548, 52, "    consider OPPONENT ACTIVE only\n    (cannot reliably gust the bench)")
inner("ap6", 608, 52, "    dmg = attack_damage()  (x2 weakness, -30 resist)\n    skip if dmg <= 0")
inner("ap7", 668, 64, "    score = pokemon_score(opp); +500 if KO;\n    50000 if KO wins the game; else x dmg/hp; +200 if active")
inner("ap8", 740, 64, "    if score > best: keep ->\n    plan{attacker, target, attack_id, remain_hp, needs_energy}")
ARROW("a-d3ap", C("d3")["r"], (560, RECTS["d3"][1] + 52), color=GREEN)
ARROW("ap12", C("ap1")["b"], C("ap2")["t"], color=AMBER)
ARROW("ap23", C("ap2")["b"], C("ap3")["t"], color=AMBER)
ARROW("ap34", C("ap3")["b"], C("ap4")["t"], color=AMBER)
ARROW("ap45", C("ap4")["b"], C("ap5")["t"], color=AMBER)
ARROW("ap56", C("ap5")["b"], C("ap6")["t"], color=AMBER)
ARROW("ap67", C("ap6")["b"], C("ap7")["t"], color=AMBER)
ARROW("ap78", C("ap7")["b"], C("ap8")["t"], color=AMBER)
# loop-back (next attack / next Pokemon)
ARROW("ap-loop", (RECTS["ap8"][0] + RECTS["ap8"][2], RECTS["ap8"][1] + 20),
      (RECTS["ap3"][0] + RECTS["ap3"][2], RECTS["ap3"][1] + 20), color="#adb5bd")
LBL("ap-loopl", 988, 560, "loop", size=10, color="#adb5bd")

# ============ DETAIL 2: SCORE EVERY OPTION — switch(o.type) ============
NODE("sw", "box", 560, 860, 440, 560, "", BLUE, "#f3f7ff", dashed=True)
LBL("sw-h", 574, 872, "SCORE EVERY OPTION   switch(o.type)  -> scores[]", size=13, color=BLUE)


def srow(eid, y, h, label, size=11):
    NODE(eid, "box", 580, y, 400, h, label, BLUE, "#ffffff", tsize=size, align="left")


srow("sw1", 904, 28, "NUMBER -> o.number    YES -> 1    NO -> 0")
srow("sw2", 938, 118,
     "CARD -> by context:\n  SWITCH/TO_ACTIVE: energy x2, +100 planned, +40 attacker\n  SETUP/TO_FIELD: basic 100 else 50\n  TO_HAND: attacker 300, basic 260, energy 120, supp 140, item 130\n  ATTACH_FROM: energy_score()\n  DISCARD/TO_DECK: keep line (-50), dump spare dupes")
srow("sw3", 1062, 46, "PLAY -> Pokemon 20000 (basic -> 30 if >= 3 bodies),\n  Supporter 3000, Item 2500, Stadium 1500")
srow("sw4", 1112, 46, "ATTACH -> Hero Cape 7000;  else energy_score()\n  (+300 to satisfy plan.needs_energy)")
srow("sw5", 1162, 28, "EVOLVE -> 9000 (+10 x energies)   [344 -> 345]")
srow("sw6", 1194, 28, "ABILITY -> 15000      RETREAT -> 2000 if plan.attacker>=1 else -1")
srow("sw7", 1226, 28, "ATTACK -> 1000 (+500 if == plan.attack_id)")
srow("sw8", 1258, 28, "END -> -1000 (last resort)")
ARROW("a-n6sw", C("n6")["r"], (560, RECTS["n6"][1] + 33), color=BLUE)

# ---- energy_score note ----
NODE("escore", "box", 560, 1300, 440, 96,
     "energy_score(pokemon, is_active):\n  base 8000; +300 if ATTACKER (345); +150 if BASIC (344);\n  +40 if active;  -60 if already has >= 4 energy",
     BLUE, "#eef4ff", tsize=11, tcolor=BLUE, dashed=True, align="left")
ARROW("a-n5e", C("n5")["r"], C("escore")["l"], color=BLUE, dashed=True)

# ---- HELPERS reference ----
NODE("help", "box", -150, 1500, 1150, 150, "", GRAY, "#ffffff", dashed=True)
LBL("help-h", -136, 1512, "HELPERS  (all card/attack STATS read at runtime via all_card_data / all_attack)", size=13, color=GRAY)
LBL("help-b", -136, 1544,
    "get_card(): safe zone lookup        card_type(): cardType from DB        prize_count(): 3 megaEx / 2 ex / 1\n"
    "can_pay(): match cost incl. colorless / rainbow        attack_damage(): base x2 weakness, -30 resistance\n"
    "pokemon_score(): prize x1000 + energies x120 + tools x80 + stage(250/130) + hp", size=12, color="#495057")

doc = {
    "type": "excalidraw", "version": 2, "source": "claude-code",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}
with open(OUT, "w") as f:
    json.dump(doc, f, indent=2)
print(f"wrote {os.path.realpath(OUT)}  ({len(elements)} elements)")
