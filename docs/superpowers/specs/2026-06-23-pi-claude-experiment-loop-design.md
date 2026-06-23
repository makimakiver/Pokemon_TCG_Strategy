# pi ⇄ Claude Code experiment loop — design

**Date:** 2026-06-23
**Status:** approved (design); pending implementation plan

## Purpose

Close the loop between the pi coms discussion agents (planner / strategist / critic /
sim) and Claude Code. The agents discuss strategy and produce a testable hypothesis;
Claude Code runs the hypothesis as a gauntlet experiment and writes the empirical result
back, which re-activates the discussion. A single shared file is the entire channel.

## Architecture

```
pi planner    ─┐
pi strategist ─┤  append turns ──► discussion/discussion.log ◄── poll ── claude /loop
pi critic     ─┘                          ▲                              │  (this session)
                                          └─────── writes back ──────────┘
```

- The pi agents already talk peer-to-peer over unix sockets (`coms.ts`). That is
  unchanged. The loop is layered on top via one append-only file.
- Claude Code is activated by the **`/loop`** skill in an interactive session that polls
  the file (~30s) and acts on new triggers.
- Both sides are LLMs reading a human-readable log; parsing is fuzzy, only the trigger
  tokens must be reliable.

## Components

1. **`discussion/discussion.log`** — append-only, line-oriented, human-readable. Single
   source of truth for the whole loop.
2. **`discussion/PROTOCOL.md`** — the shared contract both sides read (markers + rules).
   Lets pi purpose strings stay short ("at start, read PROTOCOL.md and follow it").
3. **The `/loop` prompt** — the instruction this Claude session runs on each tick.

No changes to `coms.ts` or any pi extension. No new pi code.

## File protocol

Discussion turns: `[HH:MM:SS] <name>: <text>`.
Control lines are line-prefixed (greppable); a marker's body may span lines until the
next marker.

| Marker | Written by | Meaning |
|---|---|---|
| `@claude EXPERIMENT:` | pi agents (on consensus) | hypothesis to test; should name candidate agent module + baseline module |
| `@claude PLAN:` | Claude Code | short plan of exactly what it will run; then it waits |
| `@claude GO` | pi agent / user | approval to execute (propose-then-confirm gate) |
| `@discussion RESULT:` | Claude Code | 3-round win-rates + verdict; wakes the discussion |
| `@discussion ERROR:` | Claude Code | sim/build failure, so the discussion never hangs |

## Claude Code `/loop` behavior

- Poll `discussion/discussion.log` roughly every 30s.
- Act on the **newest `@claude EXPERIMENT` that has no Claude response (`@claude PLAN` /
  `@discussion RESULT` / `@discussion ERROR`) after it** — so each trigger fires exactly
  once.
- On a new trigger → write `@claude PLAN:` describing the candidate module, baseline, and
  the 3 runs, then wait.
- When a `@claude GO` appears **after** that plan → run the experiment.
- Otherwise keep looping. A pending plan with no GO just stays pending.

## The 3-round experiment (noise removal)

On GO, run the gauntlet **three times at increasing n** and report all three so stability
is visible:

```
docker run --rm --platform=linux/amd64 cabt-sim --a <cand> --b <baseline> -n 100   # R1
docker run --rm --platform=linux/amd64 cabt-sim --a <cand> --b <baseline> -n 200   # R2
docker run --rm --platform=linux/amd64 cabt-sim --a <cand> --b <baseline> -n 300   # R3
```

- If the `cabt-sim` image is missing, build it first:
  `docker build --platform=linux/amd64 -t cabt-sim .`
- `runner.py` swaps seats each game (removes seat bias); module names follow the harness
  convention (e.g. `agents.cand` / `main` depending on how the image resolves imports).
- **Early-abort: OFF.** Always run all 3 rounds — washing out noise is the point.
- Report each round's win-rate, a pooled rate (n=600), and a verdict relative to the
  baseline. Example:

```
@discussion RESULT: cand=agents/cand vs baseline=agents/main_v3_pure
  R1 n=100: 47.0%   R2 n=200: 45.5%   R3 n=300: 46.3%   (pooled 46.1%, n=600)
  VERDICT: REGRESSES baseline (stable across rounds, low noise)
```

## Pi agent instructions

Each agent's purpose gets a short addition (or they read `PROTOCOL.md` at start):

> "At session start read `discussion/PROTOCOL.md`. Append every turn to
> `discussion/discussion.log`. When the team agrees on a testable hypothesis, write
> `@claude EXPERIMENT: …` (name the candidate agent module and the baseline), then read
> the log until `@discussion RESULT:` / `@discussion ERROR:` appears, and continue."

Delivered via the existing justfile recipes (`local` / `local-color` / `agent`) purpose
strings, or a thin wrapper that points each agent at `PROTOCOL.md`.

## Error handling

- Sim crash / `battle_start` failure / unknown module name → Claude writes
  `@discussion ERROR:` with the cause, so the loop never deadlocks waiting for a RESULT.
- Concurrent appends are low-frequency and append-only; interleaving risk is negligible.
- If a trigger is ambiguous (missing baseline or candidate), Claude's `@claude PLAN`
  states its assumption or asks for the missing piece rather than guessing silently.

## Testing

1. Append a fake `@claude EXPERIMENT` line → confirm the loop emits a `@claude PLAN`.
2. Append `@claude GO` → confirm it runs (use a tiny n for the smoke test) and writes
   `@discussion RESULT`.
3. Confirm a second identical trigger does not re-fire the first (once-only semantics).
4. Force a bad module name → confirm `@discussion ERROR` is written, not a hang.

## Out of scope (YAGNI)

- No new pi extension / `coms.ts` changes.
- No headless `claude -p` watcher (chosen: interactive `/loop`).
- No automatic pi-poke via `coms_send` (pi resumes by reading the file).
