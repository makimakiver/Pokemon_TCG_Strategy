# Self-training loop prompt — push starmie_cind_v4 to 80% (Mega Lucario is the hole)

Paste into a Claude Code `/loop` (the builder/runner side); the Strategist + Planner pi agents
drive proposals through `discussion/discussion.log` per `discussion/PROTOCOL.md`. The team
self-trains: Strategist proposes a lever → Planner refines it into an experiment → Claude Code
BUILDS + SIMULATES it via `pokemon_bench.py` → result goes back to the log → repeat.

    /loop 60s  <paste the PROMPT block below>

## CURRENT STATE (the anchor — do not re-derive)
- SHIP = `agents/starmie_cind_v4.py` on `deck_starmie_cind_v4.json` (Mega Starmie ex / Cinderace).
  Live on Kaggle (sub 54042598): 6W-3L (66%), climbing 600→786 score. Collision-proof sim field ≈ 81%
  (Starmie 80, Dragapult 78, Crustle 70, **Mega Lucario 67 ← weakest**).
- GOAL: lift the field mean to **>= 80%**, gated on fixing the ONE hole: **Mega Lucario**.
- LADDER EVIDENCE: 2 of 3 live losses are to Mega Lucario. This is the bottleneck — fix it and 80% follows.

## THE MEGA LUCARIO MATCHUP (the prevention to engineer)
Mega Lucario ex: 340 HP; **Mega Brave 270** (2 {F} energy; can't use 2 turns running); Aura Jab 130 (+{F} accel).
Why we lose: Mega Brave 270 OHKOs our Starmie line; we can't OHKO its 340 HP back (Nebula Beam 210 = 2 hits).
PREVENTION levers the deck already has (propose pilot tuning around THESE, measure each in isolation):
  1. ENERGY DENIAL — force Crushing Hammer (and any Hammer) onto Lucario's active to strip its {F} energy
     so it can't reach Mega Brave's 2-energy cost (it's a 2-turn build; denial buys the race).
  2. SURVIVE THE OHKO — force Hero's Cape onto the active Starmie so +100 HP survives a 270 Mega Brave,
     turning Lucario's OHKO into a 2HKO and flipping the trade.
  3. RACE WITH NEBULA BEAM (210) on the off-turns — Mega Brave can't fire twice in a row; punish the
     cooldown turn with a 210 + the spread/gust to take prizes while it can't OHKO back.
  4. GUST the support (Boss's Orders) — on the Mega-Brave-cooldown turn, drag up a benched Riolu/support
     and KO it for a cheap prize instead of trading into the 340hp wall.

## STRATEGIST SELF-TRAINING CYCLE (the proposer agent — run this continuously)
KEEP MONITORING `discussion/discussion.log` and ACTIVATE YOURSELF whenever a fresh
`@discussion RESULT:` (or `@discussion ERROR:`) from Claude Code appears. As always, each activation:
  1. OBSERVE the result and ANALYZE it, then form exactly ONE hypothesis (what single lever, focused on
     the Mega Lucario hole, would lift the field toward 80% — e.g. "forcing Crushing Hammer on Lucario's
     active before it reaches 2 {F} energy converts the OHKO race").
  2. PLAN it, then REFINE it with the Planner agent via discussion (post the idea; let Planner pressure-test
     scope/conditions; converge on a clean, isolated change).
  3. COMMAND Claude Code to modify the code and run the simulation — post `@claude IMPLEMENT:` (base/new/
     change), then after BUILT post `@claude EXPERIMENT:` + `@claude GO:` so it simulates via pokemon_bench.
  4. ANALYZE the returned data and decide whether the hypothesis is TRUE (kept) or FALSE (reverted) —
     compare the candidate's Mega Lucario cell + field-mean vs starmie_cind_v4; isolate, never bundle.
  5. REPEAT this process (back to step 1 on the next result) until the goal is achieved
     (field-mean >= 80% AND Mega Lucario >= 75%), then append `@discussion FINISH:` with the winning module.
Only ONE hypothesis in flight at a time. Never bundle levers. Let the result decide; the data is the judge.

## PROMPT (Claude Code builder/runner side — act on the NEWEST control marker, ONE action per tick)
Read ./discussion/PROTOCOL.md, then ./discussion/discussion.log.

A) `@claude IMPLEMENT:` (no `@claude BUILT:`/`@discussion ERROR:` after it) → BUILD the pilot change:
   derive `agents/<new>.py` from `base:` applying EXACTLY `change:`; keep `agent` + `my_deck` exported;
   `python3 -m py_compile`; if the change adds/edits a deck, GATE it: `python3 pokemon_bench.py --validate`
   (must be legal — Hero's Cape + Legacy are BOTH ACE SPEC, max 1). Append `@claude BUILT:` (summary +
   `syntax: OK`/`legality: OK`). Stop.

B) `@claude EXPERIMENT:` (no `@claude PLAN:`) → append `@claude PLAN:` naming candidate, baseline
   (`agents.starmie_cind_v4`), and the cells — ALWAYS include `bench_megalucario` as the PRIMARY, plus
   regression watches `starmie, dragapult_ex, crustle, bellibolt, nighttime_mine`. Stop; wait for GO.

C) `@claude EXPERIMENT:` + your `@claude PLAN:` + a later `@claude GO:` → SIMULATE via the module
   (NOT ad-hoc docker — pokemon_bench has the collision/Palace/legality guards):
     python3 pokemon_bench.py --agent agents.<cand>,agents.starmie_cind_v4 \
        --opponents bench_megalucario,starmie,dragapult_ex,crustle,bellibolt,nighttime_mine,hydrapple_ex,tarountula -n 100
   Append `@discussion RESULT:` — per-cell %, field mean, and VERDICT vs starmie_cind_v4.
   SUCCESS = candidate `bench_megalucario` >= 75% AND field-mean >= 80% AND no watch drops > 5pt.

D) Nothing actionable → "no pending trigger".

On any failure (compile/legality/illegal-deck/crash) append `@discussion ERROR:` with the cause.
Only APPEND to discussion.log; never rewrite earlier lines. When field-mean >= 80% with Mega Lucario
>= 75%, append `@discussion FINISH:` naming the winning module — that's the new ship.
