# pi ⇄ Claude Code experiment loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a file-mediated loop where pi coms agents discuss a hypothesis, Claude Code (`/loop`) runs it as a 3-round gauntlet experiment on confirmation, and writes the result back to re-activate the discussion.

**Architecture:** A single append-only file `discussion/discussion.log` is the whole channel. Pi agents append their turns and emit `@claude EXPERIMENT:` triggers; Claude Code polls the file via the `/loop` skill, proposes a plan, waits for `@claude GO`, runs the gauntlet 3× at n=100/200/300 in Docker, and writes `@discussion RESULT:`. Both sides read the shared `discussion/PROTOCOL.md` contract. No pi/`coms.ts` code changes.

**Tech Stack:** Markdown protocol files, `just` recipes (bash), Docker (`cabt-sim` image, `runner.py`), Claude Code `/loop` skill.

## Global Constraints

- File location: everything lives under `discussion/` in the repo root (`/Users/makimakiver/pokemon-tcg/discussion/`).
- Pi agents run with cwd = repo root (the `open` recipe does `cd {{justfile_directory()}}`), so `discussion/...` relative paths resolve correctly.
- Sim command (verbatim): `docker run --rm --platform=linux/amd64 cabt-sim --a <cand> --b <baseline> -n <N>`; build with `docker build --platform=linux/amd64 -t cabt-sim .`.
- Experiment is **3 rounds at n=100, 200, 300**, early-abort OFF (always run all 3).
- Gate is **propose-then-confirm**: Claude writes `@claude PLAN:` and waits for `@claude GO` before running.
- Pi launch recipes follow the existing `local`-family pattern: `{{PI_REPO}}/extensions/{coms,minimal,theme-cycler}.ts`, `--project "{{PROJECT}}"`. `PI_REPO := "/Users/makimakiver/pi-vs-claude-code"`, `PROJECT := "strategy-lab"`.
- Markers (line-prefixed, verbatim): `@claude EXPERIMENT:`, `@claude PLAN:`, `@claude GO`, `@discussion RESULT:`, `@discussion ERROR:`.

---

## File Structure

- `discussion/PROTOCOL.md` — the shared contract (markers, rules, experiment procedure). Read by both pi agents and Claude Code. **Committed.**
- `discussion/discussion.log` — append-only runtime channel. **Gitignored** (runtime data).
- `discussion/loop-prompt.md` — the exact prompt to feed `/loop`, plus usage notes. **Committed.**
- `.gitignore` — add one line to ignore the runtime log.
- `justfile` — add `loop-agent` + `loop-team` recipes that launch protocol-aware pi agents.

---

## Task 1: Scaffold the `discussion/` channel and contract

**Files:**
- Create: `discussion/PROTOCOL.md`
- Create: `discussion/discussion.log`
- Modify: `.gitignore`

**Interfaces:**
- Produces: the marker vocabulary and the `discussion/discussion.log` path consumed by every other task. Markers: `@claude EXPERIMENT:`, `@claude PLAN:`, `@claude GO`, `@discussion RESULT:`, `@discussion ERROR:`.

- [ ] **Step 1: Create `discussion/PROTOCOL.md`**

```markdown
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
```

- [ ] **Step 2: Create the seeded `discussion/discussion.log`**

```
# discussion.log — shared pi <-> Claude Code channel. See discussion/PROTOCOL.md.
# Append-only. Control markers: @claude EXPERIMENT/PLAN/GO, @discussion RESULT/ERROR.
```

- [ ] **Step 3: Gitignore the runtime log**

Add this line to `.gitignore`:

```
discussion/discussion.log
```

- [ ] **Step 4: Verify scaffolding**

Run:
```bash
ls discussion/ && echo "---" && grep -c "@claude EXPERIMENT" discussion/PROTOCOL.md && git check-ignore discussion/discussion.log
```
Expected: lists `PROTOCOL.md` and `discussion.log`; grep count ≥ 1; `git check-ignore` prints `discussion/discussion.log` (confirming it's ignored).

- [ ] **Step 5: Commit**

```bash
git add discussion/PROTOCOL.md .gitignore
git commit -m "feat(loop): add discussion channel protocol + gitignore runtime log

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Add protocol-aware pi launch recipes

**Files:**
- Modify: `justfile` (add `loop-agent` and `loop-team` recipes after the `local-color` recipe, ~line 168)

**Interfaces:**
- Consumes: `discussion/PROTOCOL.md` (Task 1) — agents are told to read it. Existing vars `PI_REPO`, `PROJECT`.
- Produces: `just loop-agent <name> "<role>"` launches one protocol-aware agent; `just loop-team` opens the four roles in macOS Terminals.

- [ ] **Step 1: Add the `loop-agent` recipe**

Insert into `justfile` immediately after the `local-color` recipe (before `local-planner`):

```
# Protocol-aware coms agent wired into the discussion<->Claude loop.
# Reads discussion/PROTOCOL.md and uses discussion/discussion.log as the channel.
loop-agent name role:
    pi \
      -e {{PI_REPO}}/extensions/coms.ts \
      -e {{PI_REPO}}/extensions/minimal.ts \
      -e {{PI_REPO}}/extensions/theme-cycler.ts \
      --name "{{name}}" \
      --cname "{{name}}" \
      --purpose "{{role}}. At session start, read ./discussion/PROTOCOL.md and follow it exactly. Append every turn to ./discussion/discussion.log. When the team agrees on a testable hypothesis, write an '@claude EXPERIMENT:' block naming candidate and baseline agent modules, then read discussion.log until '@discussion RESULT:' or '@discussion ERROR:' appears and continue." \
      --project "{{PROJECT}}"

# Open the full protocol-aware team in macOS Terminal windows.
loop-team:
    just open "loop-agent planner 'high-level strategy planner; decomposes game plans into testable hypotheses'"
    just open "loop-agent strategist 'creates new candidate strategies, heuristics, and policy rules'"
    just open "loop-agent critic 'checks strategic claims against rules, math, legality, and counterplay'"
```

- [ ] **Step 2: Verify the recipes parse**

Run:
```bash
just --list 2>&1 | grep -E "loop-agent|loop-team"
```
Expected: both `loop-agent` and `loop-team` appear in the recipe list (proves the justfile still parses).

- [ ] **Step 3: Dry-check the launch command expands (no pi process)**

Run:
```bash
just --dry-run loop-agent strategist "test role" 2>&1 | grep -E "PROTOCOL.md|discussion.log"
```
Expected: the printed (un-run) command contains both `./discussion/PROTOCOL.md` and `./discussion/discussion.log`, confirming the purpose string is wired correctly. (`--dry-run` prints without executing.)

- [ ] **Step 4: Commit**

```bash
git add justfile
git commit -m "feat(loop): add loop-agent/loop-team recipes (protocol-aware pi agents)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Author the `/loop` prompt for Claude Code

**Files:**
- Create: `discussion/loop-prompt.md`

**Interfaces:**
- Consumes: `discussion/PROTOCOL.md` and `discussion/discussion.log` (Task 1); the Docker sim command (Global Constraints).
- Produces: a copy-paste `/loop` invocation that drives the Claude Code side of the loop.

- [ ] **Step 1: Create `discussion/loop-prompt.md`**

```markdown
# Claude Code loop prompt

Start the watcher in an interactive Claude Code session in the repo root:

    /loop 30s <paste the PROMPT block below>

(`30s` = poll cadence. Omit the interval to let Claude self-pace.)

## PROMPT

Read ./discussion/PROTOCOL.md, then read ./discussion/discussion.log.
Find the newest `@claude EXPERIMENT:` block that has NO `@claude PLAN:`,
`@discussion RESULT:`, or `@discussion ERROR:` after it.

- If there is none: do nothing this tick, just report "no pending trigger".
- If there is a new trigger and you have NOT yet planned it: append a concise
  `@claude PLAN:` block to ./discussion/discussion.log naming the candidate
  module, baseline module, and the 3 runs (n=100/200/300). Then stop for this
  tick (do NOT run anything — wait for `@claude GO`).
- If a trigger already has your `@claude PLAN:` AND a later `@claude GO`:
  run the experiment now —
    1. Ensure the image exists: `docker image inspect cabt-sim >/dev/null 2>&1 ||
       docker build --platform=linux/amd64 -t cabt-sim .`
    2. Run 3 rounds (always all 3, no early abort):
       `docker run --rm --platform=linux/amd64 cabt-sim --a <cand> --b <baseline> -n 100`
       then `-n 200`, then `-n 300`.
    3. Parse the candidate (side A) win-rate from each run's output.
    4. Append a `@discussion RESULT:` block: each round's %, the pooled %
       (n=600), and a VERDICT (IMPROVES / NEUTRAL / REGRESSES vs baseline,
       with a one-line stability note).
- On ANY failure (image build fails, `battle_start failed`, unknown module,
  non-zero exit): append a `@discussion ERROR:` block with the cause instead of
  a RESULT, so the discussion never hangs.

Only ever append to ./discussion/discussion.log. Never rewrite earlier lines.
```

- [ ] **Step 2: Verify the prompt is self-contained**

Run:
```bash
grep -E "PROTOCOL.md|cabt-sim|@discussion RESULT|@discussion ERROR|@claude GO" discussion/loop-prompt.md | wc -l
```
Expected: ≥ 5 (the prompt references the contract, the image, both result markers, and the GO gate).

- [ ] **Step 3: Commit**

```bash
git add discussion/loop-prompt.md
git commit -m "feat(loop): add /loop prompt that drives the Claude Code side

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: End-to-end smoke test of the loop

This task verifies the full handshake using a manual stand-in for the pi agents and a single manual execution of the loop prompt (instead of waiting on `/loop` ticks). Uses a tiny n to keep it fast. No code is committed; this is a behavioral gate.

**Files:**
- Touches (runtime only, gitignored): `discussion/discussion.log`

**Interfaces:**
- Consumes: everything from Tasks 1–3.

- [ ] **Step 1: Inject a fake EXPERIMENT trigger**

Append to the log (simulating the pi agents). Use two real modules from `agents/`:
```bash
cat >> discussion/discussion.log <<'EOF'

[10:00:00] strategist: I think honchkrow_v8 still trails main_v3_pure.
@claude EXPERIMENT: does honchkrow_v8 beat main_v3_pure?
candidate: agents/honchkrow_v8
baseline:  agents/main_v3_pure
EOF
echo "trigger written"
```

- [ ] **Step 2: Run the PLAN half of the loop prompt once, manually**

Read `discussion/PROTOCOL.md` and the loop-prompt's PLAN rule, then append a `@claude PLAN:` block for the pending trigger. Confirm:
```bash
grep -A2 "@claude PLAN" discussion/discussion.log
```
Expected: a PLAN block referencing `honchkrow_v8`, `main_v3_pure`, and `n=100/200/300`.

- [ ] **Step 3: Confirm the gate holds (no GO ⇒ no run)**

Verify NO result was written before approval:
```bash
grep -c "@discussion RESULT" discussion/discussion.log
```
Expected: `0` (Claude must wait for GO).

- [ ] **Step 4: Approve and run the experiment (tiny n smoke)**

Append the GO, then run the experiment with **n=2 per round** (smoke speed, not the real 100/200/300):
```bash
printf '\n@claude GO\n' >> discussion/discussion.log
docker image inspect cabt-sim >/dev/null 2>&1 || docker build --platform=linux/amd64 -t cabt-sim .
docker run --rm --platform=linux/amd64 cabt-sim --a agents.honchkrow_v8 --b agents.main_v3_pure -n 2
```
Expected: the runner prints `A = ... vs B = ...` and two win-count lines with percentages (proves the sim path works end-to-end). If module import names differ, adjust `--a/--b` to the form the image expects and note it.

- [ ] **Step 5: Write the RESULT and confirm once-only semantics**

Append a `@discussion RESULT:` block (from the n=2 run), then re-read the log per the loop prompt's trigger rule and confirm the SAME experiment is no longer "pending" (it now has a RESULT after it):
```bash
grep -c "@discussion RESULT" discussion/discussion.log   # expect 1
```
Expected: `1`. Re-running the loop's "find newest unanswered trigger" logic now yields "no pending trigger" — the trigger fired exactly once.

- [ ] **Step 6: Verify the ERROR path (no hang on bad input)**

Inject a trigger with a nonexistent module and confirm the correct response is an ERROR, not a silent hang:
```bash
docker run --rm --platform=linux/amd64 cabt-sim --a agents.does_not_exist --b agents.main_v3_pure -n 1; echo "exit=$?"
```
Expected: a non-zero exit / import traceback — i.e. the condition under which the loop prompt instructs Claude to write `@discussion ERROR:`. (No commit; this confirms the failure mode is detectable.)

- [ ] **Step 7: Clean up the smoke transcript (optional)**

The log is gitignored, so nothing to commit. Optionally reset it to the seed:
```bash
printf '# discussion.log — shared pi <-> Claude Code channel. See discussion/PROTOCOL.md.\n# Append-only. Control markers: @claude EXPERIMENT/PLAN/GO, @discussion RESULT/ERROR.\n' > discussion/discussion.log
echo "log reset"
```

---

## Self-Review

**Spec coverage:**
- Single shared file channel → Task 1 (`discussion.log`). ✓
- `/loop` activation in this session → Task 3 (loop-prompt). ✓
- Agent-driven logging via instructions → Task 2 (`loop-agent` purpose). ✓
- Marker protocol (EXPERIMENT/PLAN/GO/RESULT/ERROR) → Task 1 (PROTOCOL.md), used in Tasks 3–4. ✓
- Propose-then-confirm gate → Task 3 prompt + Task 4 Step 3. ✓
- 3 rounds n=100/200/300, early-abort OFF → Global Constraints + Tasks 1/3. ✓
- Error handling (no hang) → PROTOCOL.md rule + Task 3 prompt + Task 4 Step 6. ✓
- Once-only trigger semantics → PROTOCOL.md rule + Task 4 Step 5. ✓
- Testing (fake trigger → PLAN → GO → RESULT → once-only → ERROR) → Task 4. ✓
- No `coms.ts` changes → honored (only justfile + markdown). ✓

**Placeholder scan:** No TBD/TODO; every file has full content; `<cand>`/`<baseline>`/`<pct>` are intentional protocol template fields, defined in PROTOCOL.md.

**Consistency:** Marker spellings, the Docker command, `n=100/200/300`, and the `discussion/` paths match across Global Constraints, PROTOCOL.md, loop-prompt.md, and the smoke test.

## Execution Handoff

See the offer in the chat after this plan is saved.
