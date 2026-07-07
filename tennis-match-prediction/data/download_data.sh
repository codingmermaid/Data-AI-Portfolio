#!/usr/bin/env bash
# Download ATP match results (2000-present) from the TML-Database project:
# https://github.com/Tennismylife/TML-Database  (one CSV per season)
set -euo pipefail
cd "$(dirname "$0")"
for y in $(seq 2000 "$(date +%Y)"); do
  echo "fetching ${y}.csv"
  curl -sf -o "${y}.csv" \
    "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/${y}.csv" \
    || echo "  (no file for ${y} yet)"
done
