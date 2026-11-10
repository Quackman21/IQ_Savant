"""Build the MLB hitting reference (league average + standard deviation per stat).

Run from the IQ_Savant folder with the .venv active:

    python -m pipeline.build_league_reference 2026          # one season
    python -m pipeline.build_league_reference 2015 2026     # a range (backfill)
    python -m pipeline.build_league_reference 2026 --force  # rebuild even if already saved

Each season pulls every regular-season pitch from Statcast (~700K rows) a week at a
time, retrying when Savant hiccups. Seasons already saved are skipped, so if the run
stops for any reason just run the same command again and it picks up where it left off.
Results go to data/league/hitting_<season>.csv.gz -- commit and push them so the live
site has them. Re-run with --force when a stat definition or the xwOBA model changes.
"""
from __future__ import annotations

import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CHUNK_DAYS = 7
RETRIES = 5


def _pull_chunk(start: str, end: str) -> pd.DataFrame:
    """One week of pitches, retrying with a growing wait if Savant sends back junk."""
    from pybaseball import statcast
    wait = 20
    for attempt in range(1, RETRIES + 1):
        try:
            df = statcast(start_dt=start, end_dt=end, verbose=False, parallel=False)
            return df if df is not None else pd.DataFrame()
        except Exception as e:                       # bad CSV / timeout / connection reset
            if attempt == RETRIES:
                raise RuntimeError(f"Savant kept failing for {start}..{end}: {e}") from e
            print(f"    {start}..{end} failed ({type(e).__name__}); retry {attempt}/{RETRIES - 1} in {wait}s",
                  flush=True)
            time.sleep(wait)
            wait *= 2
    return pd.DataFrame()


RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"      # not pushed to GitHub


def raw_path(season: int) -> Path:
    return RAW_DIR / f"statcast_{season}.csv.gz"


def season_complete(season: int, today: date | None = None) -> bool:
    today = today or date.today()
    return season < today.year or today.month >= 11


def ensure_season(season: int) -> Path | None:
    """Download a finished season once and keep it (every pitch, every column). Later runs -- the MLB
    averages, SAxwOBA, swing decisions -- read the saved file instead of downloading again."""
    p = raw_path(season)
    if p.exists():
        return p
    print(f"[{season}] downloading every pitch a week at a time…", flush=True)
    pull_season(season)
    return p if p.exists() else None


def pull_season(season: int) -> pd.DataFrame:
    p = raw_path(season)
    if p.exists():
        print(f"[{season}] using saved pitches ({p.name})", flush=True)
        return pd.read_csv(p, low_memory=False)
    from pybaseball import cache
    cache.enable()
    d, last = date(season, 3, 1), min(date(season, 11, 30), date.today())
    parts = []
    while d <= last:
        e = min(d + timedelta(days=CHUNK_DAYS - 1), last)
        df = _pull_chunk(d.isoformat(), e.isoformat())
        if not df.empty:
            parts.append(df)
        print(f"    {e.isoformat()}  ({sum(len(p) for p in parts):,} pitches so far)", flush=True)
        d = e + timedelta(days=1)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if "game_type" in df:
        df = df[df["game_type"].astype(str).eq("R")]
    if season_complete(season) and not raw_path(season).exists():   # keep finished seasons
        import os
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        tmp = RAW_DIR / f"statcast_{season}.{os.getpid()}.tmp"       # this job's own temp file
        df.to_csv(tmp, index=False, compression="gzip")
        if raw_path(season).exists():                 # another job saved it first -- keep theirs
            tmp.unlink(missing_ok=True)
        else:
            tmp.replace(raw_path(season))             # only a complete file ever gets the real name
    return df


def main(argv: list[str]) -> None:
    from core import hit_metrics as hm
    from core import league_ref
    from models import xwoba

    force = "--force" in argv
    years = [a for a in argv if not a.startswith("--")]
    if not years:
        print(__doc__)
        return
    first = int(years[0])
    last = int(years[1]) if len(years) > 1 else first
    for season in range(first, last + 1):
        if not force and league_ref.status(season) == "current":
            print(f"[{season}] already up to date — skipping. Use --force to rebuild anyway.", flush=True)
            continue
        t = time.time()
        print(f"[{season}] pulling Statcast a week at a time…", flush=True)
        raw = pull_season(season)
        if raw.empty:
            print(f"[{season}] no regular-season pitches found — skipping.", flush=True)
            continue
        print(f"[{season}] {len(raw):,} pitches; computing…", flush=True)
        ref = league_ref.build(hm.prep(raw), model=xwoba.ACTIVE)
        from models import framing
        league_ref.save_catching(framing.daily_table(framing.framing_pitches(hm.clean(raw))), season)
        last_game = str(pd.to_datetime(raw["game_date"]).max().date()) if "game_date" in raw else None
        path = league_ref.save(ref, season, last_game)
        print(f"[{season}] saved {path} ({len(ref):,} rows) in {time.time() - t:.0f}s", flush=True)
        del raw, ref


if __name__ == "__main__":
    main(sys.argv[1:])
