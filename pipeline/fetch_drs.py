"""Save Sports Info Solutions DRS (as published on FanGraphs) for the fielding comparison study.

    python -m pipeline.fetch_drs                 # 2016 -> last full season, skips seasons already saved
    python -m pipeline.fetch_drs --season 2026 --force

Writes data/compare/drs_<year>.csv (Name, Pos, Inn, DRS + the SIS pieces: rPM = range & positioning, rGFP = good
plays / misplays, rARM, rGDP, ...). If FanGraphs blocks the request, the fallback is the
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


API = ("https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=fld&lg=all&qual=0&season={y}"
       "&season1={y}&startdate=&enddate=&month=0&hand=&team=0&pageitems=2000000000&pagenum=1&ind=0&rost=0"
       "&players=&type={t}&postseason=&sortdir=default&sortstat=Defense")


def tidy(rows: list[dict]) -> pd.DataFrame:
    """FanGraphs JSON rows -> one clean table (plain names, no HTML)."""
    d = pd.DataFrame(rows)
    if "PlayerName" in d:
        d = d.drop(columns=[c for c in ("Name",) if c in d]).rename(columns={"PlayerName": "Name"})
    for c in ("Name", "Team", "Pos", "TeamName"):
        if c in d and (d[c].dtype == object or pd.api.types.is_string_dtype(d[c])):
            d[c] = d[c].astype(str).str.replace(r"<[^>]+>", "", regex=True).str.strip()
    if "Team" not in d and "TeamName" in d:
        d["Team"] = d["TeamName"]
    return d


def fetch(year: int) -> pd.DataFrame:
    """FanGraphs' own leaderboard feed (pybaseball's reader comes back empty since FanGraphs' redesign).
    type=1 is the advanced fielding view (DRS and its SIS pieces); falls back to the standard view."""
    import requests
    last = None
    for t in (1, 0):
        r = requests.get(API.format(y=year, t=t), timeout=60,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                                  "Accept": "application/json"})
        r.raise_for_status()
        js = r.json()
        d = tidy(js.get("data", js) if isinstance(js, dict) else js)
        if "DRS" in d and {"Name", "Pos"} <= set(d.columns):
            cols = [c for c in KEEP if c in d.columns]
            d = d[cols]
            return d[pd.to_numeric(d["DRS"], errors="coerce").notna()]
        last = list(d.columns)[:25]
    raise ValueError(f"no DRS column (got {last})")


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
