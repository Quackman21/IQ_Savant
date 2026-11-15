#!/bin/bash
# Build + open a page with ONLY the swings Claude asked you to re-review (ids in rereview_ids.txt).
#   bash prototypes/swing_map/rereview.sh [game.csv] [season.csv.gz]
cd "$(dirname "$0")/../.."
GAME="${1:-$HOME/Desktop/swing_data/game_849799.csv}"
SEASON="${2:-$HOME/Desktop/swing_data/season_849799.csv.gz}"
IDS="$(tr -d ' \n' < prototypes/swing_map/rereview_ids.txt)"
STANCE="${3:-$HOME/Desktop/swing_data/batting-stance.csv}"
.venv/bin/python prototypes/swing_map/build.py "$GAME" "$SEASON" prototypes/swing_map/rereview.html --only "$IDS" --stance "$STANCE" --league "$HOME/Desktop/swing_data/league_2026.csv.gz" \
  && open prototypes/swing_map/rereview.html
