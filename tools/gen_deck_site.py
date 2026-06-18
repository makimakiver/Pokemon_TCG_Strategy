#!/usr/bin/env python3
"""Generate a self-contained HTML visualizer for the current deck.

Parses the live `my_deck` from main.py (so it stays in sync), resolves every card
against data/cards.json (names, HP, type, weakness, retreat, attacks), and emits a
standalone HTML page (no external deps).

Usage:
    python3 tools/gen_deck_site.py                       # visualize root main.py
    python3 tools/gen_deck_site.py --deck data/decks/deck_dragapult_ex.json --title "Dragapult ex"
"""
import argparse
import json
import os
import re
from collections import Counter, OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ETYPE = {0: ("Colorless", "C", "#9aa0a6"), 1: ("Grass", "G", "#3fa34d"),
         2: ("Fire", "R", "#e8590c"), 3: ("Water", "W", "#1c7ed6"),
         4: ("Lightning", "L", "#f1c40f"), 5: ("Psychic", "P", "#9c36b5"),
         6: ("Fighting", "F", "#a9744f"), 7: ("Darkness", "D", "#343a40"),
         8: ("Metal", "M", "#868e96"), 9: ("Dragon", "N", "#c9a227"),
         10: ("Rainbow", "✦", "#7048e8"), 11: ("Team Rocket", "TR", "#6a1b9a")}
TRAINER_COLOR = {"ITEM": "#0ca678", "SUPPORTER": "#f08c00",
                 "STADIUM": "#1971c2", "TOOL": "#7048e8"}


def load_cards():
    raw = json.load(open(os.path.join(ROOT, "data", "cards.json")))
    cards = raw["cards"] if isinstance(raw, dict) and "cards" in raw else (
        list(raw.values()) if isinstance(raw, dict) else raw)
    return {c["id"]: c for c in cards if isinstance(c, dict) and "id" in c}


def load_deck_from_main():
    src = open(os.path.join(ROOT, "main.py")).read()
    m = re.search(r"my_deck\s*=\s*(\(.*?\))\s*\n\s*assert", src, re.S)
    if not m:
        m = re.search(r"my_deck\s*=\s*(\[.*?\])\s*\n", src, re.S)
    return eval(m.group(1), {"__builtins__": {}}, {})  # arithmetic list literal only


def pips(energies):
    out = []
    for e in (energies or []):
        name, letter, col = ETYPE.get(e, ETYPE[0])
        out.append(f'<span class="pip" style="background:{col}" title="{name}">{letter}</span>')
    return "".join(out) or '<span class="pip none">–</span>'


def attack_html(c, cards):
    rows = []
    for a in (c.get("attacks") or []):
        if not isinstance(a, dict):
            continue
        dmg = a.get("damage") or 0
        dmg_s = f'<span class="dmg">{dmg}</span>' if dmg else ""
        txt = (a.get("text") or "").strip()
        txt_s = f'<div class="atxt">{txt}</div>' if txt else ""
        rows.append(f'<div class="atk"><div class="aline"><span class="cost">{pips(a.get("energies"))}</span>'
                    f'<span class="aname">{a.get("name","")}</span>{dmg_s}</div>{txt_s}</div>')
    return "".join(rows)


def card_tile(cid, n, cards):
    c = cards.get(cid, {"name": f"#{cid}", "cardTypeName": "?"})
    ct = c.get("cardTypeName", "?")
    name = c.get("name", f"#{cid}")
    if os.path.exists(os.path.join(ROOT, "docs", "cards", f"{cid}.png")):
        return (f'<figure class="card img"><span class="count">×{n}</span>'
                f'<img loading="lazy" src="cards/{cid}.png" alt="{name}">'
                f'<figcaption>{name}</figcaption></figure>')
    if ct == "POKEMON":
        et = c.get("energyType", 0)
        accent = ETYPE.get(et, ETYPE[0])[2]
        stage = "Stage 2" if c.get("stage2") else "Stage 1" if c.get("stage1") else "Basic"
        ex = ' <span class="ex">ex</span>' if c.get("ex") else (' <span class="ex">MEGA ex</span>' if c.get("megaEx") else "")
        meta = [f'<span class="chip" style="--c:{accent}">{ETYPE.get(et,ETYPE[0])[0]}</span>',
                f'<span class="chip ghost">{stage}</span>']
        if c.get("hp"):
            meta.append(f'<span class="hp">{c.get("hp")} HP</span>')
        w = c.get("weakness")
        if w is not None:
            wn, _, wc = ETYPE.get(w, ETYPE[0])
            meta.append(f'<span class="chip ghost" title="weakness">weak {wn} ×2</span>')
        rc = c.get("retreatCost")
        if rc:
            meta.append(f'<span class="chip ghost">retreat {rc}</span>')
        body = attack_html(c, cards)
    else:
        accent = TRAINER_COLOR.get(ct, ETYPE.get(c.get("energyType", 0), ETYPE[0])[2])
        if ct in ("BASIC_ENERGY", "SPECIAL_ENERGY"):
            et = c.get("energyType", 0)
            accent = ETYPE.get(et, ETYPE[0])[2]
            kind = "Special Energy" if ct == "SPECIAL_ENERGY" else "Basic Energy"
            meta = [f'<span class="chip" style="--c:{accent}">{kind}</span>']
        else:
            meta = [f'<span class="chip" style="--c:{accent}">{ct.title()}</span>']
        body = ""
    return (f'<div class="card" style="--accent:{accent}">'
            f'<div class="count">×{n}</div>'
            f'<div class="cname">{name}</div>'
            f'<div class="meta">{"".join(meta)}</div>'
            f'{body}</div>')


def section(title, sub, ids, n_total, cards):
    tiles = "".join(card_tile(cid, n, cards) for cid, n in ids)
    return (f'<section><div class="shead"><h2>{title}</h2>'
            f'<span class="scount">{n_total} cards · {sub}</span></div>'
            f'<div class="grid">{tiles}</div></section>')


def build(deck, title, cards):
    cnt = Counter(deck)
    poke, trainers, energy = [], [], []
    for cid, n in cnt.items():
        ct = cards.get(cid, {}).get("cardTypeName", "?")
        if ct == "POKEMON":
            poke.append((cid, n))
        elif ct in ("BASIC_ENERGY", "SPECIAL_ENERGY"):
            energy.append((cid, n))
        else:
            trainers.append((cid, n))
    # order Pokemon by (energyType, stage) so evolution lines cluster
    poke.sort(key=lambda x: (cards.get(x[0], {}).get("energyType", 0),
                             (cards.get(x[0], {}).get("stage2") and 2) or
                             (cards.get(x[0], {}).get("stage1") and 1) or 0))
    order = {"SUPPORTER": 0, "ITEM": 1, "TOOL": 2, "STADIUM": 3}
    trainers.sort(key=lambda x: order.get(cards.get(x[0], {}).get("cardTypeName", ""), 9))
    energy.sort(key=lambda x: (cards.get(x[0], {}).get("cardTypeName") == "SPECIAL_ENERGY",
                               cards.get(x[0], {}).get("energyType", 0)))
    np = sum(n for _, n in poke); nt = sum(n for _, n in trainers); ne = sum(n for _, n in energy)
    # type identity
    types = sorted({cards.get(c, {}).get("energyType") for c, _ in poke
                    if cards.get(c, {}).get("energyType") is not None})
    type_chips = " ".join(f'<span class="chip" style="--c:{ETYPE[t][2]}">{ETYPE[t][0]}</span>' for t in types)

    secs = (section("Pokémon", f"{len({cards[c].get('evolvesFrom') or c for c,_ in poke})} lines", poke, np, cards) +
            section("Trainers", "items · supporters · tools", trainers, nt, cards) +
            section("Energy", "basic + special", energy, ne, cards))

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — deck</title>
<style>
:root{{--bg:#0f1115;--panel:#171a21;--line:#262b35;--text:#e8eaed;--muted:#9aa0a6;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:40px 24px 80px}}
header h1{{margin:0 0 6px;font-size:34px;letter-spacing:-.02em}}
.sub{{color:var(--muted);margin:0 0 16px}}
.tot{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:8px}}
.tot b{{font-size:15px}}
.pill{{border:1px solid var(--line);border-radius:999px;padding:5px 12px;color:var(--muted);font-size:13px}}
section{{margin-top:34px}}
.shead{{display:flex;align-items:baseline;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:16px}}
.shead h2{{margin:0;font-size:20px}}
.scount{{color:var(--muted);font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(168px,1fr));gap:16px}}
.card.img{{margin:0;background:transparent;border:none;padding:0;position:relative}}
.card.img img{{width:100%;display:block;border-radius:10px;box-shadow:0 4px 14px rgba(0,0,0,.45)}}
.card.img .count{{position:absolute;top:8px;right:8px;background:rgba(15,17,21,.88);border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-weight:800;color:#fff;font-size:13px}}
.card.img figcaption{{margin-top:7px;font-size:12.5px;color:var(--muted);text-align:center;line-height:1.3}}
.card{{position:relative;background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:12px;padding:14px 14px 12px}}
.count{{position:absolute;top:12px;right:12px;font-weight:700;color:var(--accent)}}
.cname{{font-weight:650;font-size:15.5px;padding-right:36px}}
.ex{{color:#ffd43b;font-weight:800;font-size:11px;vertical-align:super}}
.meta{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 2px}}
.chip{{font-size:11px;padding:2px 9px;border-radius:999px;background:color-mix(in srgb,var(--c,#888) 22%,transparent);color:#fff;border:1px solid color-mix(in srgb,var(--c,#888) 55%,transparent)}}
.chip.ghost{{background:transparent;color:var(--muted);border:1px solid var(--line)}}
.hp{{font-size:11px;color:var(--muted);align-self:center}}
.atk{{border-top:1px dashed var(--line);margin-top:10px;padding-top:8px}}
.aline{{display:flex;align-items:center;gap:8px}}
.cost{{display:inline-flex;gap:2px}}
.pip{{width:18px;height:18px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;color:#fff;text-shadow:0 1px 1px rgba(0,0,0,.4)}}
.pip.none{{background:transparent;color:var(--muted)}}
.aname{{font-weight:600;font-size:13.5px}}
.dmg{{margin-left:auto;font-weight:800}}
.atxt{{color:var(--muted);font-size:12px;margin-top:4px}}
footer{{margin-top:50px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:14px}}
</style></head><body><div class="wrap">
<header>
<h1>{title}</h1>
<p class="sub">cabt competition submission deck · 60 cards</p>
<div class="tot"><span class="pill"><b>{np}</b> Pokémon</span><span class="pill"><b>{nt}</b> Trainers</span><span class="pill"><b>{ne}</b> Energy</span>&nbsp; {type_chips}</div>
</header>
{secs}
<footer>Generated from <code>main.py</code> (live deck) + <code>data/cards.json</code> by <code>tools/gen_deck_site.py</code>. Card stats/attacks read from the cabt engine pool.</footer>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", help="path to a deck JSON id-list (default: parse main.py)")
    ap.add_argument("--title", default="main.py — Crustle + Ethan's Typhlosion")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "deck-viewer.html"))
    args = ap.parse_args()
    cards = load_cards()
    deck = json.load(open(args.deck)) if args.deck else load_deck_from_main()
    html = build(deck, args.title, cards)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(html)
    print(f"wrote {args.out}  ({len(deck)} cards)")


if __name__ == "__main__":
    main()
