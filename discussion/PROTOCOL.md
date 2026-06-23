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

## The experiment (run by Claude Code only)

Always 3 rounds, increasing n to wash out noise (early-abort OFF):

    docker run --rm --platform=linux/amd64 cabt-sim --a <cand> --b <baseline> -n 100
    docker run --rm --platform=linux/amd64 cabt-sim --a <cand> --b <baseline> -n 200
    docker run --rm --platform=linux/amd64 cabt-sim --a <cand> --b <baseline> -n 300

If the `cabt-sim` image is missing, build first:

    docker build --platform=linux/amd64 -t cabt-sim .

Module names use the harness convention (`runner.py` resolves `--a`/`--b`).
The runner swaps seats each game, so the reported % is seat-bias-corrected.

## Rules
- Each `@claude EXPERIMENT:` fires EXACTLY once. Claude acts on the newest
  EXPERIMENT that has no `@claude PLAN`/`@discussion RESULT`/`@discussion ERROR`
  after it.
- Claude NEVER runs the gauntlet before a matching `@claude GO`.
- On any failure (bad module, build error, sim crash) Claude writes
  `@discussion ERROR:` — never leaves the discussion waiting.
- If a trigger is missing `candidate` or `baseline`, Claude's `@claude PLAN`
  states its assumption or asks for the missing piece rather than guessing.
