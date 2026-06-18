#!/usr/bin/env bash
# Run the 10 meta decks (bare pilot) vs the benchmark agent; writes results/<slug>.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
GAMES="${1:-30}"
OPP="${2:-agents.main_bench}"
mkdir -p results
DECKS=(dragapult_ex n_s_zoroark_ex crustle slowking hydrapple_ex alakazam
       raging_bolt_ex ogerpon_box lillie_s_clefairy_ex rocket_s_honchkrow)
printf "deck                   wins/%-3s  winner:reason\n" "$GAMES"
for s in "${DECKS[@]}"; do
  docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
    -e BARE_DECK="data/decks/deck_$s.json" \
    cabt-sim --a agents.bare_agent --b "$OPP" -n "$GAMES" > "results/$s.txt" 2>&1
  aw=$(grep -E "^  A \(" "results/$s.txt" | sed -E 's/.*: ([0-9]+) .*/\1/')
  r=$(grep "winner:reason" "results/$s.txt" | sed 's/.*winner:reason  : //')
  printf "%-22s %2s/%-3s   %s\n" "$s" "$aw" "$GAMES" "$r"
done
