# pokemon-tcg — cabt self-play harness

Local self-play for the cabt "Limited Card Battle" engine. The engine binary
(`cg/libcg.so`) is **Linux x86-64 only**, so sims run in Docker under emulation.

## Layout

```
.
├── cg/                     # engine package (libcg.so / cg.dll loaded via its own __file__) — imported as `cg`
├── runner.py               # match runner (Docker ENTRYPOINT); imports agents by dotted name
├── Dockerfile              # linux/amd64 python:3.11-slim; ENTRYPOINT = runner.py
├── agents/                 # agent modules (each exposes agent(obs)->list[int] and my_deck)
│   ├── main.py             #   live tuned Crustle/"Palace" agent
│   ├── main_bench.py       #   frozen snapshot used as the benchmark opponent
│   ├── bare_agent.py       #   generic auto-piloting agent (roles derived from BARE_DECK)
│   ├── main_v1.py          #   earlier Palace iteration
│   └── main_megalucario_backup.py
├── tools/                  # offline tooling (run with python; cg-importing ones need Docker)
│   ├── _paths.py           #   path bootstrap: puts repo root on sys.path, exposes DATA/DECKS_DIR
│   ├── decklists.py        #   the 10 meta lists (names/counts)
│   ├── build_decks.py      #   resolve lists -> cabt ids -> data/decks/*.json  (host, no engine)
│   ├── export_cards.py     #   dump engine pool -> data/cards.json            (Docker)
│   ├── dump_cards.py        #   human-readable pool dump                       (Docker)
│   ├── inspect_cards.py    #   inspect a specific deck's cards                 (Docker)
│   └── trace.py            #   single-game step tracer                        (Docker)
├── data/                   # all generated/static data (gitignored from the image via *.json)
│   ├── cards.json          #   engine card pool (1267 cards / 1556 attacks, with names)
│   ├── decks.json          #   all 10 resolved decks
│   ├── decks/              #   per-deck 60-card id lists (deck_<slug>.json) — consumed by BARE_DECK
│   ├── deck_slugs.json     #   archetype name -> slug
│   ├── palace/             #   replay JSON the Palace deck was ported from
│   ├── losers_log.json     #   replay log
│   └── card_reference/     #   official card-id PDFs / CSVs
├── results/                # per-deck gauntlet logs
├── main.py                 # standalone competition submission entrypoint (root)
├── docs/                   # writeups + design docs (RESULTS.md, SGS_RL_PLAN.md, *_TUNING.md, ...)
│   └── diagrams/           #   *.excalidraw strategy / code-flow diagrams
├── submissions/            # packaged competition bundles (*.tar.gz, submission_rebel.py)
├── AGENTS.md               # repo agent/instructions file
└── README.md
```

## Dependency notes (why files sit where they do)

- `cg/` loads its native lib relative to its own `__file__`, so it is relocatable but
  must remain importable as `cg` — every agent does `from cg.api import ...`.
- `runner.py` stays at the repo root so its `sys.path[0]` is the root: that makes both
  `cg` and `agents.*` importable, and keeps the Dockerfile `ENTRYPOINT python runner.py`
  unchanged.
- Tools import `tools/_paths.py` first; it inserts the repo root on `sys.path` (so `cg`
  and `agents` resolve from any CWD) and exposes `DATA` / `DECKS_DIR` so scripts no longer
  depend on the current working directory.
- `.dockerignore` excludes `*.json`, so data reaches the container through the
  `-v "$PWD":/app` mount, not `COPY`. Always run with that mount + `-w /app`.

## Common commands

Build the image (once):
```
docker build --platform=linux/amd64 -t cabt-sim .
```

Regenerate decks from the meta lists (host, no engine needed):
```
python3 tools/build_decks.py
```

Run a match (A vs B, seats swapped each game):
```
docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
  -e BARE_DECK="data/decks/deck_rocket_s_honchkrow.json" \
  cabt-sim --a agents.bare_agent --b agents.main_bench -n 30
```

Run the full 10-deck gauntlet vs the benchmark agent:
```
./scripts/gauntlet.sh            # writes results/<slug>.txt
```

Dump / inspect the card pool (Docker, engine needed):
```
docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
  --entrypoint python cabt-sim tools/export_cards.py
```
# Pokemon_TCG_Strategy
