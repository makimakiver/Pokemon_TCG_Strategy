# Claude Code loop prompt

Start the watcher in an interactive Claude Code session in the repo root:

    /loop 30s <paste the PROMPT block below>

(`30s` = poll cadence. Omit the interval to let Claude self-pace.)

## PROMPT

Read ./discussion/PROTOCOL.md, then read ./discussion/discussion.log. Act on the
NEWEST actionable control marker — ONE action per tick:

A) `@claude IMPLEMENT:` block with NO `@claude BUILT:` or `@discussion ERROR:` after
   it → BUILD the deck/pilot change:
    1. Read `base:`, `new:`, `change:`. If `new:` (agents/<new>.py) ALREADY exists,
       do NOT overwrite — append `@discussion ERROR:` asking for a fresh name, stop.
    2. Derive agents/<new>.py from `base`, applying EXACTLY the `change:` spec. Keep
       it import-compatible (export `agent` and `my_deck`) and deterministic (seeded
       RNG) like the base.
    3. Syntax-check: `python3 -m py_compile agents/<new>.py`.
    4. Append a `@claude BUILT:` block (the new module, a 1-3 line summary of what you
       changed, the key knobs/values, and `syntax: OK`). Do NOT run a gauntlet.
    Then stop for this tick (wait for the team's `@claude EXPERIMENT:` + `@claude GO`).

B) `@claude EXPERIMENT:` block with NO `@claude PLAN:`, `@discussion RESULT:`, or
   `@discussion ERROR:` after it → append a concise `@claude PLAN:` naming the
   candidate module, baseline module, and the 3 runs (n=100/200/300). Then stop —
   do NOT run anything; wait for `@claude GO`.

C) An `@claude EXPERIMENT:` that already has your `@claude PLAN:` AND a later
   `@claude GO` → run the experiment now:
    1. Ensure the image exists: `docker image inspect cabt-sim >/dev/null 2>&1 ||
       docker build --platform=linux/amd64 -t cabt-sim .`
    2. Run 3 rounds (always all 3, no early abort), DOTTED module paths (convert any
       `agents/foo` to `agents.foo`). Either the raw docker runs or the wrapper
       `just gauntlet <cand> <baseline>`:
       `docker run --rm --platform=linux/amd64 -v "$(pwd)":/app cabt-sim --a <cand> --b <baseline> -n 100`
       then `-n 200`, then `-n 300`.
    3. Parse the candidate (side A) win-rate from each run's output.
    4. Append a `@discussion RESULT:` block: each round's %, the pooled % (n=600),
       and a VERDICT (IMPROVES / NEUTRAL / REGRESSES vs baseline, with a one-line
       stability note).

D) Nothing actionable → do nothing this tick, just report "no pending trigger".

On ANY failure (compile error, image build fails, `battle_start failed`, unknown
module, name collision, non-zero exit) append a `@discussion ERROR:` block with the
cause instead — so the discussion never hangs.

Only ever append to ./discussion/discussion.log. Never rewrite earlier lines.
