"""Direct Savant leaderboard downloads for boards pybaseball is missing or can't read."""
from __future__ import annotations

import io

from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://baseballsavant.mlb.com/leaderboard"


def _csv(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        text = r.content.decode("utf-8-sig")
        df = pd.read_csv(io.StringIO(text))
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def fielding_run_value(year: int, pos: int, min_inn: int = 1) -> pd.DataFrame:
    """Fielding Run Value by position (2 = C ... 9 = RF). Catchers include framing / blocking / throwing."""
    return _csv(f"{BASE}/fielding-run-value?gameType=Regular&seasonStart={year}&seasonEnd={year}"
                f"&type=fielder&position={pos}&minInnings={min_inn}&minResults=1&csv=true")


# Statcast search, limited to balls in play (about 1/5 of all pitches) -- used to train xwOBA
FRAMING_URLS = [      # Savant has moved / renamed this board; each is tried and CHECKED against who caught that year
    "https://baseballsavant.mlb.com/catcher_framing?year={y}&team=&min=1&type=catcher&sort=4,1&csv=true",
    "https://baseballsavant.mlb.com/leaderboard/catcher-framing?type=catcher&year={y}&team=&min=1&csv=true",
    "https://baseballsavant.mlb.com/leaderboard/catcher-framing?season={y}&min=1&csv=true",
    "https://baseballsavant.mlb.com/leaderboard/catcher-framing?seasonStart={y}&seasonEnd={y}&min=1&csv=true",
    "https://baseballsavant.mlb.com/leaderboard/catcher-framing?startYear={y}&endYear={y}&min=1&csv=true",
    "https://baseballsavant.mlb.com/leaderboard/catcher-framing?season_start={y}&season_end={y}&min=1&csv=true",
    "https://baseballsavant.mlb.com/leaderboard/catcher-framing?yearStart={y}&yearEnd={y}&min=1&csv=true",
    "https://baseballsavant.mlb.com/leaderboard/catcher-framing?seasons={y}&min=1&csv=true",
    "https://baseballsavant.mlb.com/leaderboard/catcher-framing?season%5B%5D={y}&min=1&csv=true",
    "https://baseballsavant.mlb.com/leaderboard/catcher-framing?year%5B%5D={y}&min=1&csv=true",
]
COMPARE = Path(__file__).resolve().parent.parent / "data" / "compare"


def _get(url: str, tries: int = 3) -> tuple[pd.DataFrame, str]:
    """Download a CSV, retrying when Savant says slow down. Returns (table, what happened)."""
    import time
    msg = ""
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code in (429, 500, 502, 503, 504):
                msg = f"HTTP {r.status_code}"
                time.sleep(15 * (i + 1))
                continue
            if r.status_code != 200:
                return pd.DataFrame(), f"HTTP {r.status_code}"
            text = r.content.decode("utf-8-sig")
            if text.lstrip().startswith("<"):
                return pd.DataFrame(), "got a web page, not a CSV"
            df = pd.read_csv(io.StringIO(text))
            df.columns = df.columns.str.strip()
            return df, f"{len(df)} rows"
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            time.sleep(5 * (i + 1))
    return pd.DataFrame(), msg


def framing_cols(f: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """(id, runs, called pitches) columns -- old board (runs_extra_strikes, n_called_pitches) or new (rv_tot, pitches)."""
    cols = {c.lower(): c for c in f.columns}
    idc = cols.get("player_id") or cols.get("id") or cols.get("catcher_id")
    rc = cols.get("runs_extra_strikes") or cols.get("rv_tot")
    nc = cols.get("n_called_pitches") or cols.get("pitches")
    return idc, rc, nc


def catcher_framing(year: int, our_takes: pd.Series | None = None, log=print) -> pd.DataFrame:
    """Savant catcher framing for ONE season, verified: the board must list the catchers who caught that season
    (ids) with pitch counts that rise and fall with ours -- a board that ignored the year (and sent this season)
    fails the check and is thrown out. Verified seasons are saved in data/compare/savant_framing_<year>.csv."""
    import time
    cache, good = COMPARE / f"savant_framing_{year}.csv", COMPARE / "savant_framing_url.txt"
    if cache.exists():
        return pd.read_csv(cache)
    urls = list(FRAMING_URLS)
    if good.exists() and good.read_text().strip() in urls:
        urls.remove(good.read_text().strip())
        urls.insert(0, good.read_text().strip())
    for tpl in urls:
        f, msg = _get(tpl.format(y=year))
        time.sleep(2)                                             # be polite -- Savant throttles fast requests
        idc, rc, nc = framing_cols(f) if len(f) else (None, None, None)
        if not (idc and rc and nc):
            log(f"    {year}: {tpl.split('?')[0].split('/')[-1]} [{tpl.split('?')[1].split('&')[0]}] -> "
                f"{msg if not len(f) else 'columns ' + str(list(f.columns)[:6])}")
            continue
        if our_takes is not None and len(our_takes):
            ids = pd.to_numeric(f[idc], errors="coerce")
            ours = our_takes[our_takes >= 200]
            overlap = ids.isin(ours.index).mean()
            both = f.assign(_id=ids).dropna(subset=["_id"]).set_index("_id")[nc].astype(float)
            both = both[both.index.isin(our_takes.index)]
            r = both.rank().corr(our_takes.reindex(both.index).rank()) if len(both) >= 15 else np.nan
            if not (overlap >= 0.8 and r >= 0.85):
                log(f"    {year}: {tpl.split('?')[1].split('&')[0]} -> wrong season? (only {overlap:.0%} of its catchers "
                    f"caught in {year}, pitch-count match r = {r:.2f}) -- not used")
                continue
            log(f"    {year}: Savant framing verified ({overlap:.0%} same catchers, pitch-count match r = {r:.2f})")
        COMPARE.mkdir(parents=True, exist_ok=True)
        f.to_csv(cache, index=False)
        good.write_text(tpl)
        return f
    log(f"    {year}: no Savant framing board matched this season")
    return pd.DataFrame()


def catcher_blocking(year: int) -> pd.DataFrame:
    """Savant catcher blocking leaderboard (Blocks Above Average, blocking runs) for one season."""
    return _csv(f"{BASE}/catcher-blocking?type=Cat&year={year}&minPitches=1&csv=true")


_BIP_URL = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT="
            "&hfPR=hit%5C.%5C.into%5C.%5C.play%7C&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfSea=&hfSit="
            "&player_type=batter&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA="
            "&game_date_gt={start}&game_date_lt={end}&team=&position=&hfRO=&home_road=&hfFlag=&metric_1="
            "&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches&player_event_sort=api_p_release_speed"
            "&sort_order=desc&min_abs=0&type=details&")


def batted_balls(start: str, end: str) -> pd.DataFrame:
    """Regular-season balls in play between two dates (Savant caps a request at ~25K rows, so ask for a week)."""
    r = requests.get(_BIP_URL.format(start=start, end=end), timeout=120)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig")
    if not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), low_memory=False)       # raises if Savant sent an error page
    df.columns = df.columns.str.strip()
    if "error" in df.columns:
        raise RuntimeError(str(df["error"].iloc[0]))
    if "type" in df:                                   # in case the filter is ever ignored
        df = df[df["type"].eq("X")]
    return df
