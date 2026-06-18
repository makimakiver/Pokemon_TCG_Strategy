#!/usr/bin/env python3
"""Interactive gallery: one clickable component per main* agent deck.

Click a component -> a modal shows that agent's full 60-card deck (card art,
grouped Pokemon / Trainers / Energy, with counts). Self-contained HTML; card
images live in docs/cards/<id>.png (downloaded by the image-fetch step).

Run: python3 tools/gen_deck_gallery.py   ->  docs/deck-gallery.html
"""
import glob, json, os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETYPE = {0: "#9aa0a6", 1: "#3fa34d", 2: "#e8590c", 3: "#1c7ed6", 4: "#f1c40f",
         5: "#9c36b5", 6: "#a9744f", 7: "#343a40", 8: "#868e96", 9: "#c9a227",
         10: "#7048e8", 11: "#6a1b9a"}


def extract_deck(src):
    i = src.index('my_deck'); j = src.index('=', i) + 1
    k = next(p for p in range(j, len(src)) if src[p] in '([')
    op = src[k]; cl = ')' if op == '(' else ']'; d = 0
    for p in range(k, len(src)):
        if src[p] == op: d += 1
        elif src[p] == cl:
            d -= 1
            if d == 0: return eval(src[k:p+1], {"__builtins__": {}}, {})


def load_cards():
    raw = json.load(open(os.path.join(ROOT, "data", "cards.json")))
    cards = raw["cards"] if isinstance(raw, dict) and "cards" in raw else (
        list(raw.values()) if isinstance(raw, dict) else raw)
    return {c["id"]: c for c in cards if isinstance(c, dict) and "id" in c}


def archetype(deck, cards):
    s = set(deck)
    if 678 in s: return ("Mega Lucario ex", "#868e96", 678)
    if 318 in s or 663 in s: return ("Rebel Fire (anti-Crustle)", "#e8590c", 318 if 318 in s else 663)
    if 354 in s: return ("Crustle + Ethan's Typhlosion", "#3b5bdb", 354)
    if 345 in s: return ("Palace single-line Crustle", "#0c8599", 345)
    # fallback: highest-stage / highest-hp pokemon
    pk = [c for c in s if cards.get(c, {}).get("cardTypeName") == "POKEMON"]
    hero = max(pk, key=lambda c: (cards[c].get("stage2", 0), cards[c].get("stage1", 0), cards[c].get("hp", 0)), default=(pk[0] if pk else deck[0]))
    return ("Custom", "#7048e8", hero)


def label_for(path):
    b = os.path.basename(path)
    if path == "main.py": return "main.py · root (submission)"
    if path == "agents/main.py": return "main.py · agents"
    return b


def main():
    files = sorted(set(glob.glob(os.path.join(ROOT, "agents", "main*.py")) + [os.path.join(ROOT, "main.py")]))
    cards = load_cards()
    decks, used_ids = [], set()
    for f in files:
        try:
            dk = extract_deck(open(f).read())
        except Exception:
            dk = None
        if not (isinstance(dk, list) and len(dk) == 60):
            continue
        rel = os.path.relpath(f, ROOT)
        arch, accent, hero = archetype(dk, cards)
        if not os.path.exists(os.path.join(ROOT, "docs", "cards", f"{hero}.png")):
            hero = next((c for c in dk if os.path.exists(os.path.join(ROOT, "docs", "cards", f"{c}.png"))), dk[0])
        used_ids.update(dk)
        decks.append({"label": label_for(rel), "arch": arch, "accent": accent,
                      "hero": hero, "n": len(dk), "cards": dict(Counter(dk))})
    # group identical decklists -> sort so same archetype clusters
    decks.sort(key=lambda d: (d["arch"], d["label"]))

    CARDS = {}
    for cid in used_ids:
        c = cards.get(cid, {})
        stage = 2 if c.get("stage2") else 1 if c.get("stage1") else 0
        CARDS[cid] = {"n": c.get("name", f"#{cid}"), "t": c.get("cardTypeName", "?"),
                      "e": c.get("energyType", 0), "s": stage,
                      "img": os.path.exists(os.path.join(ROOT, "docs", "cards", f"{cid}.png"))}

    data = json.dumps({"decks": decks, "cards": CARDS, "etype": ETYPE})
    html = HTML.replace("/*DATA*/", data)
    out = os.path.join(ROOT, "docs", "deck-gallery.html")
    open(out, "w").write(html)
    print(f"wrote {out}  ({len(decks)} components, {len(used_ids)} unique cards)")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>main* deck gallery</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b35;--text:#e8eaed;--muted:#9aa0a6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:40px 24px 80px}
h1{margin:0 0 4px;font-size:32px;letter-spacing:-.02em}.sub{color:var(--muted);margin:0 0 26px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:18px}
.comp{cursor:pointer;background:var(--panel);border:1px solid var(--line);border-top:3px solid var(--accent);border-radius:14px;padding:14px;transition:transform .12s,box-shadow .12s;display:flex;gap:13px;align-items:center}
.comp:hover{transform:translateY(-3px);box-shadow:0 10px 26px rgba(0,0,0,.5)}
.comp img{width:62px;border-radius:7px;flex:none;box-shadow:0 3px 8px rgba(0,0,0,.5)}
.comp .nm{font-weight:700;font-size:15px;word-break:break-word}
.comp .ar{color:var(--accent);font-size:12px;font-weight:600;margin-top:3px}
.comp .ct{color:var(--muted);font-size:11.5px;margin-top:5px}
.modal{position:fixed;inset:0;background:rgba(6,8,12,.85);display:none;align-items:flex-start;justify-content:center;padding:40px 16px;overflow:auto;z-index:9}
.modal.open{display:flex}
.sheet{background:var(--bg);border:1px solid var(--line);border-radius:16px;max-width:1100px;width:100%;padding:26px}
.mhead{display:flex;align-items:baseline;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:6px}
.mhead h2{margin:0;font-size:22px}.mhead .a{color:var(--accent);font-weight:600}
.x{cursor:pointer;border:1px solid var(--line);border-radius:8px;padding:5px 12px;color:var(--muted);background:transparent;font-size:18px;line-height:1}
.sec h3{margin:20px 0 10px;font-size:15px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:14px}
figure.card{margin:0;position:relative}
figure.card img{width:100%;border-radius:9px;display:block;box-shadow:0 4px 12px rgba(0,0,0,.45)}
figure.card .cnt{position:absolute;top:7px;right:7px;background:rgba(15,17,21,.9);border:1px solid var(--line);border-radius:999px;padding:1px 9px;font-weight:800;font-size:12.5px}
figure.card figcaption{margin-top:5px;font-size:11.5px;color:var(--muted);text-align:center}
.noimg{aspect-ratio:.72;border:1px dashed var(--line);border-radius:9px;display:flex;align-items:center;justify-content:center;text-align:center;font-size:12px;color:var(--muted);padding:6px}
</style></head><body><div class="wrap">
<h1>main* deck gallery</h1>
<p class="sub">Each component is a <code>main*</code> agent. Click one to see its full 60-card deck.</p>
<div class="gallery" id="gallery"></div></div>
<div class="modal" id="modal"><div class="sheet" id="sheet"></div></div>
<script>
const D=/*DATA*/;
const ec=D.etype, C=D.cards;
function tile(id,n){const c=C[id]||{n:'#'+id};
  const img=c.img?`<img loading="lazy" src="cards/${id}.png" alt="${c.n}">`:`<div class="noimg">${c.n}</div>`;
  return `<figure class="card"><span class="cnt">×${n}</span>${img}<figcaption>${c.n}</figcaption></figure>`;}
function group(cardsObj){const P=[],T=[],E=[];
  for(const id in cardsObj){const c=C[id]||{t:'?'};const t=c.t;
    if(t==='POKEMON')P.push(id);else if(t==='BASIC_ENERGY'||t==='SPECIAL_ENERGY')E.push(id);else T.push(id);}
  P.sort((a,b)=>(C[a].e-C[b].e)||(C[a].s-C[b].s));
  const ord={SUPPORTER:0,ITEM:1,TOOL:2,STADIUM:3};
  T.sort((a,b)=>(ord[C[a].t]??9)-(ord[C[b].t]??9));
  E.sort((a,b)=>(C[a].t==='SPECIAL_ENERGY')-(C[b].t==='SPECIAL_ENERGY'));
  const sum=a=>a.reduce((s,id)=>s+cardsObj[id],0);
  const sec=(t,a)=>a.length?`<div class="sec"><h3>${t} · ${sum(a)}</h3><div class="grid">${a.map(id=>tile(id,cardsObj[id])).join('')}</div></div>`:'';
  return sec('Pokémon',P)+sec('Trainers',T)+sec('Energy',E);}
function showDeck(i){const d=D.decks[i];const sh=document.getElementById('sheet');
  sh.style.setProperty('--accent',d.accent);
  sh.innerHTML=`<div class="mhead"><div><h2>${d.label}</h2><div class="a">${d.arch} · ${d.n} cards</div></div>
    <button class="x" onclick="close_()">✕</button></div>${group(d.cards)}`;
  document.getElementById('modal').classList.add('open');}
function close_(){document.getElementById('modal').classList.remove('open');}
document.getElementById('modal').addEventListener('click',e=>{if(e.target.id==='modal')close_();});
document.addEventListener('keydown',e=>{if(e.key==='Escape')close_();});
document.getElementById('gallery').innerHTML=D.decks.map((d,i)=>
  `<div class="comp" data-i="${i}" style="--accent:${d.accent}">
     <img src="cards/${d.hero}.png" alt="">
     <div><div class="nm">${d.label}</div><div class="ar">${d.arch}</div><div class="ct">${d.n} cards · click to view</div></div>
   </div>`).join('');
document.querySelectorAll('.comp').forEach(el=>el.addEventListener('click',()=>showDeck(+el.dataset.i)));
</script></body></html>"""

if __name__ == "__main__":
    main()
