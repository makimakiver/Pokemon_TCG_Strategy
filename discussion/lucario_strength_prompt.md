# Self-training loop prompt — STRENGTHEN the Mega Lucario pilot until sim ≈ ladder

Goal: build a Mega Lucario agent that PILOTS the deck like the real rank-6 ladder player — so it
beats `starmie_cind_v14` at ~70% (the real ladder rate), not the ~22% our scripted pilot manages.
A strong Lucario is needed as an HONEST benchmark opponent; right now `pokemon_bench` pilots Lucario
with the generic `meta_opp`, which never fires Mega Brave, so the sim says we beat Lucario 77% while
the real ladder beats us 70% (3W-7L). Close that sim↔ladder gap by fixing the PILOT, not the list.

    /loop 60s  <paste the PROMPT block below>

## CURRENT STATE (the anchor — do not re-derive)
- Deck = `deck_bench_megalucario.json` (exact real meta list; DO NOT change the list — pilot only).
- Candidate base = `agents/meta_opp_lucario.py` (collision-safe: reads META_DECK, not BARE_DECK).
  It already forces evolve + 6 boosts/game + {F} energy routing, yet still loses to starmie_cind_v14
  ~78%-22% and fires Mega Brave only 8/20 games. THAT is the problem to fix.
- Ladder truth: real Lucario beats starmie_cind ~70%; hits 240-300 (NOT a single OHKO — it wins the
  RACE through fast, consistent Mega Brave); median 6 boosts/game; first Mega Brave very early.
- GOAL: candidate Lucario beats `agents.starmie_cind_v14` >= 60% (stretch 70%) AND fires Mega Brave
  >= 1.5/game (median), WITHOUT a hand-coded line that regresses vs the rest of the field.

## THE MEGA LUCARIO GAMEPLAN (what the pilot must reliably execute)
Mega Lucario ex: 340 HP; **Mega Brave 983 = 270 dmg, 2x{F}, can't be used 2 turns running**; **Aura
Jab 982 = 130 dmg + accelerates Basic {F} from discard**. Win condition = land Mega Brave every other
turn and chip with Aura Jab between. The pilot keeps WHIFFING because it can't keep 2 {F} on the active
attacker. Levers to engineer (propose ONE at a time, measure each in isolation):
  1. ENERGY ROUTING — guarantee 2 {F} on the ACTIVE Mega Lucario ex before anything else (the #1 leak:
     Mega Brave is never a payable option, so it never fires). Use Aura Jab's discard→bench accel +
     manual attach + retreat the charged body into the active slot.
  2. MEGA BRAVE CADENCE — fire Mega Brave when payable; on its forced cooldown turn fire Aura Jab
     (which BOTH chips 130 AND accelerates {F} for next turn's Mega Brave). Never idle the turn.
  3. ATTACKER CONTINUITY — keep a 2nd Riolu/Lucario charging on the bench so a KO'd attacker is
     replaced instantly (the real Lucario never loses tempo to a dead board).
  4. RAMP TO SET UP A TURN EARLIER — Carmine + Premium Power Pro + Fighting Gong to dig/accelerate;
     evolve Riolu → Mega Lucario ex ASAP. (Already partly done — refine timing, don't over-force.)
  5. PRIZE RACE — Boss's Orders to gust a softened/benched attacker for a clean prize when trading
     into starmie's 320 HP wall is unfavorable.

## STRATEGIST SELF-TRAINING CYCLE (the proposer agent — run this continuously)
KEEP MONITORING `discussion/discussion.log` and ACTIVATE YOURSELF whenever a fresh `@discussion
RESULT:`/`@discussion ERROR:` from Claude Code appears. As always, each activation:
  1. OBSERVE the result and ANALYZE it, then form exactly ONE hypothesis (the single Lucario-pilot
     lever — e.g. "force-attach {F} to the active until Mega Brave is payable" — that raises the
     Mega-Brave fire-rate and the win vs starmie_cind_v14).
  2. PLAN it, then REFINE it with the Planner agent via discussion (Planner pressure-tests scope; keep
     it isolated; never bundle two levers).
  3. COMMAND Claude Code to modify the code and simulate — `@claude IMPLEMENT:` (base/new/change),
     then after BUILT `@claude EXPERIMENT:` + `@claude GO:`.
  4. ANALYZE the returned data: did the Mega-Brave fire-rate AND the win-rate vs starmie_cind_v14 go
     UP? Hypothesis TRUE (keep) or FALSE (revert). The data is the judge.
  5. REPEAT until the goal is met (Lucario beats starmie_cind_v14 >= 60% AND Mega Brave >= 1.5/game),
     then append `@discussion FINISH:` with the winning module = the new honest benchmark Lucario.
Only ONE hypothesis in flight. Never bundle levers. Pilot-only — do NOT edit the decklist.

## PROMPT (Claude Code builder/runner side — act on the NEWEST control marker, ONE action per tick)
Read ./discussion/PROTOCOL.md, then ./discussion/discussion.log.

A) `@claude IMPLEMENT:` (no later `@claude BUILT:`/`@discussion ERROR:`) → BUILD the pilot change:
   derive `agents/<new>.py` from `base:` (default base `agents/meta_opp_lucario.py`) applying EXACTLY
   `change:`; it MUST read its deck from `META_DECK` (collision-safe — never BARE_DECK), export
   `agent` + `my_deck`; `python3 -m py_compile`. Append `@claude BUILT:` (summary + `syntax: OK`). Stop.

B) `@claude EXPERIMENT:` (no `@claude PLAN:`) → append `@claude PLAN:` naming candidate, baseline
   (`agents.meta_opp_lucario`), and the head-to-head (the candidate Lucario vs `agents.starmie_cind_v14`,
   n=100/200/300, seat-swapped), PLUS the Mega-Brave fire-rate diagnostic. Stop; wait for GO.

C) `@claude EXPERIMENT:` + your `@claude PLAN:` + a later `@claude GO:` → SIMULATE the head-to-head
   (TWO AGENTS FIGHTING — this is the honest test, NOT meta_opp on the deck):
     docker run --rm --platform=linux/amd64 -v "$(pwd)":/app -e META_DECK="data/decks/deck_bench_megalucario.json" \
       cabt-sim --a agents.<cand> --b agents.starmie_cind_v14 -n 100   # then 200, 300
   Parse the candidate Lucario (side A) win-rate each round + pooled. Also report Mega-Braves/game
   (count attackId 983 by the Lucario seat). Append `@discussion RESULT:` — per-round %, pooled %,
   Mega-Brave fire-rate, and VERDICT vs `meta_opp_lucario`.
   SUCCESS = candidate beats starmie_cind_v14 >= 60% AND Mega Brave >= 1.5/game.

D) Nothing actionable → "no pending trigger".

On any failure (compile error / BARE_DECK collision / crash / never fires Mega Brave) append
`@discussion ERROR:` with the cause. Only APPEND to discussion.log; never rewrite earlier lines.
When the goal is met, append `@discussion FINISH:` naming the module — wire it into pokemon_bench as
the Lucario benchmark opponent so starmie_cind tunes against an HONEST (ladder-realistic) Lucario.
