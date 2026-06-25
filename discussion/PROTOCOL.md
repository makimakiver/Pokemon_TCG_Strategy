# Discussion ⇄ Claude Code experiment protocol

`discussion/discussion.log` is the single shared channel between the pi coms
agents (planner / strategist / critic / sim) and Claude Code. It is append-only
and human-readable. Both sides read THIS file at start and follow it.

## Discussion turns
Append each turn as one block:

    [HH:MM:SS] <name>: <your message>

## Control markers (line-prefixed; body may span following lines)

| Marker | Who writes it | Meaning |
|---|---|---|
| `@claude IMPLEMENT:` | pi agents, on consensus | A concrete deck/pilot change to build. MUST name `base:`, `new:`, and `change:`. |
| `@claude BUILT:` | Claude Code | The new module was created + syntax-checked. Claude then WAITS (no auto-run). |
| `@claude EXPERIMENT:` | pi agents, on consensus | A testable hypothesis. MUST name the candidate agent module and the baseline module. |
| `@claude PLAN:` | Claude Code | Short plan of exactly what it will run. Claude then WAITS. |
| `@claude GO` | a pi agent or the user | Approval to execute (propose-then-confirm gate). |
| `@discussion RESULT:` | Claude Code | 3-round win-rates + verdict. Wakes the discussion. |
| `@discussion ERROR:` | Claude Code | Build/sim failure, so the discussion never hangs. |

## How a cycle runs

1. Agents discuss; when they agree on something testable, ONE agent writes:

       @claude EXPERIMENT: <one-line hypothesis>
       candidate: agents/<module>      # the agent module to test (side A)
       baseline:  agents/<module>      # what to compare against (side B)

2. Claude Code (running `/loop`) sees the new trigger and replies:

       @claude PLAN: will run agents/<cand> vs agents/<baseline>,
       3 rounds n=100/200/300, docker cabt-sim. Awaiting @claude GO.

   Then Claude WAITS. It does not run anything yet.

3. A pi agent (or the user) approves by writing on its own line:

       @claude GO

4. Claude runs the experiment (3 rounds, see below) and writes:

       @discussion RESULT: candidate=agents/<cand> baseline=agents/<baseline>
         R1 n=100: <pct>%   R2 n=200: <pct>%   R3 n=300: <pct>%   (pooled <pct>%, n=600)
         VERDICT: <IMPROVES|NEUTRAL|REGRESSES> baseline (<stability note>)

5. The agents read the RESULT and continue the discussion.

## Building or changing a deck/pilot (the improvement step)

Testing alone never improves anything — it only measures modules that already exist.
To actually change the deck or pilot, the team proposes a concrete edit and Claude
Code implements it as a NEW module. Propose → implement → review → test:

1. When the team agrees on a concrete change, ONE agent writes:

       @claude IMPLEMENT: <one-line description of the change>
       base:   agents/<module>     # existing module to derive from
       new:    agents/<module>     # NEW module name to create (must NOT already exist)
       change: <precise behavioral spec — what to add/modify, and why it should help>

2. Claude Code derives `agents/<new>.py` from `base`, applies exactly the described
   change, syntax-checks it, and replies — WITHOUT running anything:

       @claude BUILT: agents/<new> (from agents/<base>)
         change: <what was actually implemented, 1-3 lines>
         knobs:  <key lines/values changed>
         syntax: OK — ready to test.

   Claude then WAITS. Building a module never triggers a gauntlet.

3. The team tests it through the normal cycle: emit `@claude EXPERIMENT:` with
   candidate=agents/<new> and the agreed baseline, then `@claude GO`, then read the
   `@discussion RESULT:`. Each improvement is a NEW versioned module
   (honchkrow_v23, _v24, …) — never an in-place edit of an already-tested module, so
   every result stays reproducible.

REGRESSION GUARD: a change aimed at one matchup (e.g. Abomasnow) must be re-tested
against the protected baseline it could regress (e.g. Mega Lucario, where the
champion is ~71%) BEFORE it is called shippable. Name the guard matchup in the
IMPLEMENT block or the follow-up discussion.

## The experiment (run by Claude Code only)

Always 3 rounds, increasing n to wash out noise (early-abort OFF):

    docker run --rm --platform=linux/amd64 -v "$(pwd)":/app cabt-sim --a <cand> --b <baseline> -n 100
    docker run --rm --platform=linux/amd64 -v "$(pwd)":/app cabt-sim --a <cand> --b <baseline> -n 200
    docker run --rm --platform=linux/amd64 -v "$(pwd)":/app cabt-sim --a <cand> --b <baseline> -n 300

Convenience wrapper (runs the same 3 rounds, builds the image if missing, accepts
`agents/foo` or `agents.foo`): `just gauntlet <cand> <baseline>`.

The image is only needed once as the runtime base; new or changed candidate agents are
picked up live via the volume mount (no rebuild required). Build it once if missing:

    docker image inspect cabt-sim >/dev/null 2>&1 || docker build --platform=linux/amd64 -t cabt-sim .

Module names: `runner.py` takes DOTTED Python module paths for `--a`/`--b` (e.g. `agents.honchkrow_v8`). The `candidate:`/`baseline:` lines may be written with slashes (`agents/honchkrow_v8`) for readability — Claude converts `/` to `.` before invoking the runner.
The runner swaps seats each game, so the reported % is seat-bias-corrected.

## Rules
- Each `@claude EXPERIMENT:` fires EXACTLY once. Claude acts on the newest
  EXPERIMENT that has no `@claude PLAN`/`@discussion RESULT`/`@discussion ERROR`
  after it.
- Each `@claude IMPLEMENT:` fires EXACTLY once. Claude acts on the newest IMPLEMENT
  with no `@claude BUILT`/`@discussion ERROR` after it.
- Claude only CREATES the named `new:` module under `agents/`. It NEVER overwrites an
  existing module — if `new:` already exists, Claude writes `@discussion ERROR:`
  asking for a fresh name, so prior versions and their results stay reproducible.
- New modules keep determinism (seeded RNG) and import-compatibility (must export
  `agent` and `my_deck`) like their base.
- Claude NEVER runs the gauntlet before a matching `@claude GO`.
- On any failure (bad module, build error, sim crash) Claude writes
  `@discussion ERROR:` — never leaves the discussion waiting.
- If a trigger is missing `candidate` or `baseline`, Claude's `@claude PLAN`
  states its assumption or asks for the missing piece rather than guessing.
