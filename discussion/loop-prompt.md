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
