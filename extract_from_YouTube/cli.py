"""CLI: video URL -> logs-compatible episode JSON.

Example:
  python -m ptcg_logger.cli \
      "https://www.youtube.com/watch?v=m8np08cT-TQ" \
      --players "Jack Pitcher,Andrew Hedrick" \
      --winner 0 \
      --out match.json
"""
from __future__ import annotations
import argparse
import json
import sys

from .acquire import get_transcript
from .extract import extract_events, DEFAULT_MODEL
from .schema import Episode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pokémon TCG video -> semantic log episode")
    ap.add_argument("url")
    ap.add_argument("--players", required=True,
                    help="comma-separated, index order: 'P0,P1'")
    ap.add_argument("--winner", type=int, choices=[0, 1], default=None)
    ap.add_argument("--out", default="episode.json")
    ap.add_argument("--asr", choices=["subs", "whisper"], default="subs")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--win", type=float, default=90.0, help="window seconds")
    ap.add_argument("--overlap", type=float, default=15.0)
    ap.add_argument("--no-cabt", action="store_true",
                    help="emit only semantic logs, skip cabt-code projection")
    ap.add_argument("--dump-transcript", default=None,
                    help="optional path to also save the raw transcript")
    args = ap.parse_args(argv)

    players = [p.strip() for p in args.players.split(",")]
    if len(players) != 2:
        ap.error("--players must be exactly two names")

    print(f"[1/3] transcript via {args.asr} ...")
    segments = get_transcript(args.url, asr=args.asr, lang=args.lang)
    print(f"      {len(segments)} segments")
    if args.dump_transcript:
        with open(args.dump_transcript, "w") as f:
            json.dump(segments, f, indent=2)

    print("[2/3] extracting events ...")
    events = extract_events(segments, players, model=args.model,
                            win_s=args.win, overlap_s=args.overlap)
    print(f"      {len(events)} events")

    print("[3/3] assembling episode ...")
    episode = Episode(players=players, events=events, winner=args.winner,
                      source_url=args.url, emit_cabt=not args.no_cabt).build()
    with open(args.out, "w") as f:
        json.dump(episode, f, indent=2, ensure_ascii=False)
    print(f"      wrote {args.out}  ({len(episode['steps'])} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
