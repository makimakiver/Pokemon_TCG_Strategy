#!/usr/bin/env python3
"""Generate an Excalidraw board of EVERY distinct main* agent strategy.

One card per game plan (deck, win-condition, 3-step plan, key levers, result).
Writes docs/diagrams/agents-main-strats.excalidraw; import at excalidraw.com.
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "diagrams", "agents-main-strats.excalidraw")
elements = []
_seed = 2000


def _ns():
    global _seed
    _seed += 7
    return _seed


def base(eid, etype, x, y, w, h, **kw):
    el = {
        "id": eid, "type": etype, "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": "#1e1e1e", "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": {"type": 3}, "seed": _ns(), "version": 1,
        "versionNonce": _ns(), "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False,
    }
    el.update(kw)
    return el


def R(eid, x, y, w, h, stroke, bg, sw=2, dashed=False, round_=True):
    e = base(eid, "rectangle", x, y, w, h, strokeColor=stroke, backgroundColor=bg,
             fillStyle="solid", strokeWidth=sw,
             strokeStyle="dashed" if dashed else "solid",
             roundness={"type": 3} if round_ else None)
    elements.append(e)
    return e


def T(eid, x, y, w, s, size=13, color="#1e1e1e", align="left", font=2):
    lines = s.split("\n")
    h = int(len(lines) * size * 1.25)
    e = base(eid, "text", x, y, w, h, strokeColor=color, roundness=None,
             text=s, fontSize=size, fontFamily=font, textAlign=align,
             verticalAlign="top", baseline=int(size * 0.9),
             containerId=None, originalText=s, lineHeight=1.25)
    elements.append(e)
    return e


def ARROW(eid, x1, y1, x2, y2, color="#adb5bd"):
    dx, dy = x2 - x1, y2 - y1
    elements.append(base(eid, "arrow", x1, y1, abs(dx) or 1, abs(dy) or 1,
                         strokeColor=color, strokeWidth=2, roundness={"type": 2},
                         points=[[0, 0], [dx, dy]], lastCommittedPoint=None,
                         startBinding=None, endBinding=None,
                         startArrowhead=None, endArrowhead="arrow"))


CW, CH = 380, 318


def card(idx, cx, cy, color, chip_bg, title, files, deck, win, flow, levers, result):
    p = f"c{idx}"
    R(p + "-body", cx, cy, CW, CH, color, "#ffffff", sw=2)
    R(p + "-hdr", cx, cy, CW, 44, color, color, sw=2)
    T(p + "-title", cx + 14, cy + 12, CW - 28, title, size=17, color="#ffffff", font=1)
    T(p + "-files", cx + 14, cy + 52, CW - 28, files, size=11, color="#868e96")
    T(p + "-deck", cx + 14, cy + 74, CW - 28, "DECK  " + deck, size=12, color="#343a40")
    T(p + "-win", cx + 14, cy + 118, CW - 28, win, size=12, color=color)
    # 3-step game plan flow
    inner = CW - 28
    gap = 16
    bw = (inner - 2 * gap) // 3
    fy = cy + 168
    bh = 44
    for i, step in enumerate(flow):
        bx = cx + 14 + i * (bw + gap)
        R(f"{p}-f{i}", bx, fy, bw, bh, "#adb5bd", "#f8f9fa", sw=1)
        lines = step.split("\n")
        th = len(lines) * 10 * 1.25
        T(f"{p}-ft{i}", bx, fy + (bh - th) / 2, bw, step, size=10,
          color="#495057", align="center")
        if i < len(flow) - 1:
            ARROW(f"{p}-fa{i}", bx + bw + 1, fy + bh / 2, bx + bw + gap - 1, fy + bh / 2)
    T(p + "-lev", cx + 14, cy + 226, CW - 28, "LEVERS  " + levers, size=11, color="#868e96")
    R(p + "-chip", cx + 14, cy + CH - 38, inner, 26, color, chip_bg, sw=1)
    T(p + "-res", cx + 14, cy + CH - 33, inner, result, size=13, color=color,
      align="center", font=1)


# ---- title ----
T("title", 40, 30, 1000, "agents/  —  all main* strategies", size=30, color="#1e1e1e", font=1)
T("sub", 40, 74, 1100,
  "Each card = one distinct game plan among the main* agents: deck · win-condition · 3-step plan · key levers · result",
  size=14, color="#868e96")

col = [40, 450, 860]
row = [120, 470]

# Row 1
card(1, col[0], row[0], "#0c8599", "#c5f6fa",
     "Palace Crustle — MILL",
     "main.py · main_v1 · main_cur · main_bench · *_base",
     "1 line: Dwebble 344 -> Crustle 345 (120)\n8 Pokemon / 31 energy  (thin, structural)",
     "WIN: deck the opponent OUT (mill)\nLOSE: no-active (thin board wiped)",
     ["Crustle 345", "Scissors 120", "Mill / deck-out"],
     "Flat scoring; evolve-ASAP; adaptive bench refill\n(Poffin) only when being raced.",
     "~88% vs Mega Lucario")

card(2, col[1], row[0], "#3b5bdb", "#dbe4ff",
     "Crustle + Typhlosion — RACE",
     "main_v2 · main_v4 · main_typhlosion · main_typh_base",
     "2 lines: Crustle 345 (Grass 120) +\nCyndaquil->Quilava->Typhlosion 354 (Fire St2 160)\n20 Pokemon / 16 energy  (consistent)",
     "WIN: prize race with TWO attackers\n(Rare Candy, Ultra Ball, Boss's Orders)",
     ["Rare Candy", "Typhlosion 160", "Take 6 prizes"],
     "Untuned GENERIC pilot is best; pilot tuning regresses it.",
     "80.7% gauntlet avg")

card(3, col[2], row[0], "#2f9e44", "#d3f9d8",
     "main_v3_pure — STRONGEST",
     "= v2 deck + exactly ONE overlay",
     "Same Crustle + Ethan's Typhlosion deck",
     "WIN: prize race, now USING Buddy Blast\n40 + 60 x Ethan's Adventure in discard",
     ["v2 engine", "+Buddy Blast\ndamage fix", "picks real KO"],
     "ONLY the damage-correction the engine can't\nread off a card. No behavioral overlays.",
     "83.2% gauntlet  (+2.5 vs v2)  *")

# Row 2
card(4, col[0], row[1], "#f08c00", "#fff3bf",
     "main_v3 — 5-FIX OVERLAY",
     "v2 + archetype overlay (from lost replays)",
     "Same v2 Crustle + Typhlosion deck",
     "FIXES: 1 Buddy Blast  2 discard-fuel\n3 Rare-Candy skip  4 Boss gust  5 bench/retreat",
     ["Buddy Blast", "+gust +fuel", "+bench/retreat"],
     "Bundling heuristics HURT vs v3_pure's single fix.",
     "REGRESSED vs main_v3_pure")

card(5, col[1], row[1], "#e03131", "#ffe3e3",
     "main_v2_rebel — FIRE COUNTER",
     "self-contained anti-Crustle Fire aggro",
     "Ho-Oh 318 / Volcanion 663 (130hp) +\nHearthflame Ogerpon 358 accel; 18 Fire, NON-ex",
     "WIN: OHKO Crustle on Fire weakness (x2 >= 150),\nsurvive its 120 -> ~2:1 prize race",
     ["Concentrate [RRC]", "OHKO Crustle", "2:1 prize race"],
     "Energy concentration; Boss gust; tanky active; heal timing.",
     "74% vs main  (80% on the play)")

card(6, col[2], row[1], "#868e96", "#f1f3f5",
     "Mega Lucario ex — ORIGIN",
     "main_megalucario_backup (replaced by Palace)",
     "Mega Lucario ex 678 main + Hariyama 674 /\nSolrock 676 secondary; 13 Fighting energy",
     "WIN: switch attackers; Mega Lucario ex hits hard",
     ["Riolu->M.Lucario", "switch attackers", "take prizes"],
     "The original main.py, superseded by Palace Crustle.",
     "baseline (beaten ~83%)")

doc = {
    "type": "excalidraw", "version": 2, "source": "claude-code",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}
with open(OUT, "w") as f:
    json.dump(doc, f, indent=2)
print(f"wrote {os.path.realpath(OUT)}  ({len(elements)} elements)")
