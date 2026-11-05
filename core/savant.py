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
