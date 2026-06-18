# ptcg-logger

Video → **semantic game-log** extractor for Pokémon TCG match footage.
Pulls a transcript (YouTube subs or Whisper), has an LLM parse it into typed
game events, and assembles them into a JSON episode shaped like a
`kaggle_environments` `cabt` replay — **logs only**.

## What it does / doesn't do

It fills the `observation.logs` portion of an episode with grounded events
(draw, play_basic, evolve, attach_energy, attack, knockout, take_prize, …).

It does **not** — and structurally **cannot** — fill `action`,
`observation.select`, or `search_begin_input`. Those are engine-internal option
indices; a video has no engine behind it, so they're left as honest `null`s.
This is therefore a **game record**, not a trainable `(state, option, index)`
RL episode. It is also a *different game* (Standard paper TCG) from the `cabt`
engine, so the numeric `type`/`area` codes in the cabt projection are
**provisional** — they were reverse-engineered from a single replay
(`lost_1.json`) and live in one table in `schema.py` for easy calibration.

The trustworthy layer is `observation.semanticLogs` (named events). The cabt
numeric projection is a convenience; pass `--no-cabt` to drop it.

## Install

```bash
pip install -r requirements.txt        # pydantic, anthropic
pip install yt-dlp                      # transcript download (or apt/brew)
pip install faster-whisper             # optional, only for --asr whisper
export ANTHROPIC_API_KEY=sk-...
```

## Run

```bash
python -m ptcg_logger.cli \
    "https://www.youtube.com/watch?v=m8np08cT-TQ" \
    --players "Jack Pitcher,Andrew Hedrick" \
    --winner 0 \
    --out match.json
```

`--players` order defines player indices 0 and 1 — match it to the name map the
casters use. `--winner` is optional and only sets `rewards`.

Useful flags: `--asr whisper` (better than auto-subs when captions are bad),
`--win/--overlap` (transcript window seconds; widen if turns get split),
`--model` (defaults to `claude-sonnet-4-6`), `--dump-transcript path.json`.

## Where accuracy comes from / breaks

- **Best signal = commentary.** Casters narrate most plays; ASR of the audio is
  the high-recall channel. Board OCR is not implemented here — add it if you
  need state between narrated beats.
- **Failure modes:** silent plays, fast sequences, ambiguous "he/they"
  attribution, and casters discussing hypotheticals. Events carry a
  `confidence` field; filter on it.
- **Tuning knobs:** the extraction prompt (`prompts.py`) and window size are the
  two things worth iterating on first.

## Layout

```
ptcg_logger/
  schema.py    # SemanticEvent vocab, cabt-log adapter, Episode builder
  prompts.py   # extraction system + user prompt
  acquire.py   # yt-dlp subtitles / faster-whisper transcript
  extract.py   # chunk + Anthropic call + validate + dedupe
  cli.py       # acquire -> extract -> assemble
```
