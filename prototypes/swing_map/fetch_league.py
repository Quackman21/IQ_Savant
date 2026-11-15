"""Pull every contact swing in MLB for a season (regular season + playoffs), week by week, keeping only the columns
the swing map needs. Resumable: re-run it and it skips weeks it already has.

    python prototypes/swing_map/fetch_league.py 2026 [out_dir]      (~30-60 min; ~100 MB)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pybaseball as pb

year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
out = Path(sys.argv[2] if len(sys.argv) > 2 else Path.home() / "Desktop" / "swing_data")
parts = out / f"league_{year}_parts"
parts.mkdir(parents=True, exist_ok=True)
KEEP = ["game_date", "game_pk", "game_type", "batter", "stand", "p_throws", "description", "plate_x", "plate_z",
        "vx0", "vy0", "vz0", "ax", "ay", "az", "intercept_ball_minus_batter_pos_x_inches",
        "intercept_ball_minus_batter_pos_y_inches", "attack_direction", "attack_angle", "bat_speed", "swing_length",
        "hc_x", "hc_y", "launch_speed", "launch_angle"]
CONTACT = {"foul", "foul_tip", "hit_into_play"}

day = pd.Timestamp(f"{year}-03-20")
end = pd.Timestamp(f"{year}-11-10")
while day < end:
    stop = min(day + pd.Timedelta(days=6), end)
    f = parts / f"{day:%Y%m%d}.csv.gz"
    if not f.exists():
        print(f"{day:%b %d} - {stop:%b %d} ...", flush=True)
        try:
            d = pb.statcast(f"{day:%Y-%m-%d}", f"{stop:%Y-%m-%d}", verbose=False)
        except Exception as e:                       # a bad week shouldn't kill the run; re-run later to retry
            print("   failed:", type(e).__name__, e)
            day = stop + pd.Timedelta(days=1)
            continue
        if d is not None and len(d):
            d = d[d["description"].isin(CONTACT) & ~d["game_type"].astype(str).isin(["S", "E"])]
            d[[c for c in KEEP if c in d]].to_csv(f, index=False)
        else:
            pd.DataFrame(columns=KEEP).to_csv(f, index=False)
    day = stop + pd.Timedelta(days=1)

all_ = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(parts.glob("*.csv.gz"))], ignore_index=True)
dest = out / f"league_{year}.csv.gz"
all_.to_csv(dest, index=False)
print(f"{len(all_):,} contact swings, {all_['batter'].nunique()} hitters -> {dest}")
