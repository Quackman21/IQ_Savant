"""Direct Savant leaderboard downloads for boards pybaseball is missing or can't read."""
from __future__ import annotations

import io

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
def catcher_framing(year: int) -> pd.DataFrame:
    """Savant catcher framing leaderboard for one season."""
    return _csv(f"https://baseballsavant.mlb.com/catcher_framing?year={year}&team=&min=1&type=catcher&sort=4,1&csv=true")


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
