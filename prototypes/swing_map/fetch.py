"""Pull everything the swing map needs for one game, INCLUDING the postseason (hitters move in the box, so the
swings right before the game matter most).

    python prototypes/swing_map/fetch.py GAME_PK [out_dir]
writes out_dir/game_<pk>.csv (every pitch) and out_dir/season_<pk>.csv.gz (all of that season's pitches, regular +
postseason, through the day before, for everyone who batted)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pybaseball as pb

pk = int(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
out.mkdir(parents=True, exist_ok=True)
g = pb.statcast_single_game(pk)
day = str(g["game_date"].iloc[0])[:10]
g.to_csv(out / f"game_{pk}.csv", index=False)
prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
frames = []
for b in sorted(g["batter"].unique()):
    print("batter", b)
    frames.append(pb.statcast_batter(f"{day[:4]}-03-01", prev, int(b)))
s = pd.concat(frames, ignore_index=True)
s.to_csv(out / f"season_{pk}.csv.gz", index=False)
print(f"{len(g)} pitches in the game; {len(s)} season pitches ({s['game_type'].value_counts().to_dict()})")
print(f"python prototypes/swing_map/build.py {out / f'game_{pk}.csv'} {out / f'season_{pk}.csv.gz'}")
