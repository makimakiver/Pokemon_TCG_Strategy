#!/usr/bin/env python3
"""Generate an Excalidraw drawing of the agents/ strategy lineage.

Writes docs/diagrams/agents-strategy.excalidraw; import it at excalidraw.com
(Menu -> Open) or in the VS Code Excalidraw extension.
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "diagrams", "agents-strategy.excalidraw")

elements = []
_seed = 1000


def _next_seed():
    global _seed
    _seed += 7
    return _seed


def base(eid, etype, x, y, w, h, **kw):
    el = {
        "id": eid, "type": etype, "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": "#1e1e1e", "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": {"type": 3}, "seed": _next_seed(), "version": 1,
        "versionNonce": _next_seed(), "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False,
    }
    el.update(kw)
    return el


def rect(eid, x, y, w, h, stroke, bg, dashed=False, sw=2):
    return base(eid, "rectangle", x, y, w, h, strokeColor=stroke,
                backgroundColor=bg, fillStyle="solid", strokeWidth=sw,
                strokeStyle="dashed" if dashed else "solid")


def text(eid, x, y, w, s, size=16, color="#1e1e1e", align="center"):
    lines = s.split("\n")
    h = int(len(lines) * size * 1.25)
    return base(eid, "text", x, y, w, h, strokeColor=color, roundness=None,
                text=s, fontSize=size, fontFamily=1, textAlign=align,
                verticalAlign="middle", baseline=int(size * 0.9),
                containerId=None, originalText=s, lineHeight=1.25)


def R(*a, **k):
    e = rect(*a, **k); elements.append(e); return e


def T(*a, **k):
    e = text(*a, **k); elements.append(e); return e


def BOX(eid, x, y, w, h, label, stroke, bg, size=16):
    R("r-" + eid, x, y, w, h, stroke, bg)
    lines = label.split("\n")
    th = len(lines) * size * 1.25
    T("t-" + eid, x, y + (h - th) / 2, w, label, size=size, color=stroke)
    return (x, y, w, h)


def ARROW(eid, x1, y1, x2, y2, color="#495057"):
    dx, dy = x2 - x1, y2 - y1
    e = base(eid, "arrow", x1, y1, abs(dx), abs(dy), strokeColor=color,
             strokeWidth=2, roundness={"type": 2},
             points=[[0, 0], [dx, dy]], lastCommittedPoint=None,
             startBinding=None, endBinding=None,
             startArrowhead=None, endArrowhead="arrow")
    elements.append(e)
    return e


def LABEL(eid, x, y, s, size=13, color="#868e96"):
    T(eid, x, y, 240, s, size=size, color=color, align="left")


# ---- title ----
T("title", 60, 34, 760, "agents/  —  strategy lineage", size=30, color="#1e1e1e", align="left")
T("subtitle", 60, 78, 900,
  "One generic engine swapped across decks (left)  vs  specialist counter-decks built to beat main (right)",
  size=14, color="#868e96", align="left")

# ---- family containers (behind) ----
R("c-left", 70, 250, 330, 430, "#1971c2", "transparent", dashed=True, sw=2)
T("cl-label", 84, 260, 300, "GENERIC ENGINE  ·  swap the deck", size=13, color="#1971c2", align="left")
R("c-right", 735, 320, 330, 310, "#e03131", "transparent", dashed=True, sw=2)
T("cr-label", 749, 330, 300, "SPECIALIST COUNTERS  ·  beat main", size=13, color="#e03131", align="left")

# ---- top row ----
BOX("origin", 70, 120, 230, 72, "Mega Lucario ex\n(original main → backup)", "#495057", "#f1f3f5")
BOX("engine", 430, 120, 240, 72, "Generic scoring engine\nruntime stats · evolve-ASAP", "#343a40", "#e9ecef")
BOX("bare", 800, 120, 230, 72, "bare_agent.py\npilots ANY deck (gauntlet)", "#495057", "#f1f3f5")

# ---- left column: generic-engine line ----
BOX("palace", 120, 300, 240, 80, "main.py — Palace Crustle\nsingle line · mill · ~88%", "#1971c2", "#d0ebff")
BOX("v2", 120, 442, 240, 80, "main_v2\nCrustle + Typhlosion · 80.7%", "#1971c2", "#d0ebff")
BOX("v3", 120, 584, 240, 88, "main_v3_pure  ★ STRONGEST\n+Buddy Blast fix · 83.2%", "#2f9e44", "#d3f9d8", size=16)

# ---- right column: counters ----
BOX("honch", 775, 362, 250, 88, "honchkrow.py\nRocket Feathers combo · ~56%", "#e03131", "#ffe3e3")
BOX("fire", 775, 502, 250, 92, "fire.py / main_v2_rebel\nFire counter · 74% (80% on play)", "#e8590c", "#ffe8cc")

# ---- arrows ----
ARROW("a1", 185, 192, 230, 298)            # origin -> palace
ARROW("a2", 520, 192, 265, 300)            # engine -> palace
ARROW("a3", 672, 150, 796, 150)            # engine -> bare
ARROW("a4", 240, 380, 240, 440)            # palace -> v2
ARROW("a5", 240, 522, 240, 582)            # v2 -> v3
ARROW("a6", 915, 192, 905, 360)            # bare -> honch
ARROW("a7", 900, 450, 900, 500)            # honch -> fire

# ---- arrow labels ----
LABEL("l1", 120, 232, "swapped deck")
LABEL("l2", 360, 250, "powers")
LABEL("l3", 690, 124, "generic pilot")
LABEL("l4", 248, 400, "more consistent deck")
LABEL("l5", 248, 542, "+ damage fix only")
LABEL("l6", 920, 250, "best meta deck found")
LABEL("l7", 908, 462, "peer plateau → counter")

doc = {
    "type": "excalidraw", "version": 2, "source": "claude-code",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}

with open(OUT, "w") as f:
    json.dump(doc, f, indent=2)
print(f"wrote {os.path.realpath(OUT)}  ({len(elements)} elements)")
