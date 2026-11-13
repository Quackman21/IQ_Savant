"""Save Sports Info Solutions DRS (as published on FanGraphs) for the fielding comparison study.

    python -m pipeline.fetch_drs                 # 2016 -> last full season, skips seasons already saved
    python -m pipeline.fetch_drs --season 2026 --force

Writes data/compare/drs_<year>.csv (Name, Pos, Inn, DRS + the SIS pieces: rPM = range & positioning, rGFP = good
plays / misplays, rARM, rGDP, ...). FanGraphs sometimes blocks scripts; if a season fails, the fallback is the
copy-paste route printed at the end.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "compare"
KEEP = ["Name", "Team", "Pos", "Inn", "DRS", "rPM", "rGFP", "rARM", "rGDP", "rSB", "rSZ", "rCERA", "xMLBAMID"]
FALLBACK = """
If FanGraphs blocked it: open fangraphs.com/leaders/major-league (Stat Type: Fielding, Season: the year,
Position: All, Min Inn: 0), select the table, copy, paste into Excel / Google Sheets, and save as
data/compare/drs_<year>.csv. It just needs Name, Pos and DRS columns (rPM / rGFP are a bonus)."""


def fetch(year: int) -> pd.DataFrame:
    import pybaseball as pb
    d = pb.fielding_stats(year, year, qual=0)
    cols = [c for c in KEEP if c in d.columns]
    if "DRS" not in cols:
        raise ValueError(f"no DRS column (got {list(d.columns)[:20]}...)")
    d = d[cols]
    return d[pd.to_numeric(d["DRS"], errors="coerce").notna()]


def main(argv: list[str]) -> None:
    from pipeline.train_decisions import latest_full_season
    force = "--force" in argv
    years = [int(argv[argv.index("--season") + 1])] if "--season" in argv else \
        list(range(2016, latest_full_season() + 1))
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for y in years:
        p = OUT / f"drs_{y}.csv"
        if p.exists() and not force:
            print(f"[{y}] already saved", flush=True)
            continue
        try:
            d = fetch(y)
            d.to_csv(p, index=False)
            print(f"[{y}] saved {len(d):,} player-positions -> {p.relative_to(ROOT)}", flush=True)
        except Exception as e:
            failed.append(y)
            print(f"[{y}] FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
        time.sleep(2)                                   # be polite to FanGraphs
    if failed:
        print(f"\nCouldn't fetch: {failed}" + FALLBACK)
    else:
        print("\nDone. Now run:  python -m pipeline.fielding_compare_study")


if __name__ == "__main__":
    main(sys.argv[1:])
