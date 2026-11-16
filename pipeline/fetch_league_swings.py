"""Every MLB contact swing for a season (regular season + playoffs), week by week, only the columns the swing work
needs. Resumable: re-run it and it skips weeks it already has.

    python -m pipeline.fetch_league_swings 2026 [out_dir]      (~30-60 min a season; ~100 MB)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KEEP = ["game_date", "game_pk", "game_type", "batter", "stand", "p_throws", "description", "plate_x", "plate_z",
        "vx0", "vy0", "vz0", "ax", "ay", "az", "release_speed", "intercept_ball_minus_batter_pos_x_inches",
        "intercept_ball_minus_batter_pos_y_inches", "attack_direction", "attack_angle", "bat_speed", "swing_length",
        "swing_path_tilt", "hc_x", "hc_y", "launch_speed", "launch_angle", "launch_speed_angle", "hit_distance_sc"]
CONTACT = {"foul", "foul_tip", "hit_into_play"}
DEFAULT_DIR = Path.home() / "Desktop" / "swing_data"


def fetch(year: int, out: Path = DEFAULT_DIR, verbose: bool = True) -> Path:
    import pybaseball as pb
    parts = out / f"league_{year}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    day, end = pd.Timestamp(f"{year}-03-15"), pd.Timestamp(f"{year}-11-10")
    while day < end:
        stop = min(day + pd.Timedelta(days=6), end)
        f = parts / f"{day:%Y%m%d}.csv.gz"
        if not f.exists():
            if verbose:
                print(f"  {year}: {day:%b %d} - {stop:%b %d} ...", flush=True)
            try:
                d = pb.statcast(f"{day:%Y-%m-%d}", f"{stop:%Y-%m-%d}", verbose=False)
            except Exception as e:                  # a bad week shouldn't kill the run; re-run later to retry
                print(f"     failed ({type(e).__name__}: {e}) -- re-run to retry", flush=True)
                day = stop + pd.Timedelta(days=1)
                continue
            if d is not None and len(d):
                d = d[d["description"].isin(CONTACT) & ~d["game_type"].astype(str).isin(["S", "E"])]
                d[[c for c in KEEP if c in d]].to_csv(f, index=False)
            else:
                pd.DataFrame(columns=KEEP).to_csv(f, index=False)
        day = stop + pd.Timedelta(days=1)
    frames = [pd.read_csv(p, low_memory=False) for p in sorted(parts.glob("*.csv.gz"))]
    all_ = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame(columns=KEEP)
    dest = out / f"league_{year}.csv.gz"
    all_.to_csv(dest, index=False)
    if verbose:
        print(f"  {year}: {len(all_):,} contact swings, {all_['batter'].nunique() if len(all_) else 0} hitters -> {dest}")
    return dest


if __name__ == "__main__":
    a = sys.argv[1:]
    fetch(int(a[0]) if a else 2026, Path(a[1]).expanduser() if len(a) > 1 else DEFAULT_DIR)
