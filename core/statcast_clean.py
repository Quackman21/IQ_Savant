"""pybaseball returns Statcast with pandas' nullable types (blanks are <NA>),
which break comparisons like `ev > 95` when a value is missing. Every Statcast
frame goes through clean() first: numbers become plain floats (blank = NaN),
text becomes plain strings (blank = None)."""
from __future__ import annotations

import pandas as pd

TEXT = {"description", "events", "type", "stand", "p_throws", "pitch_type", "pitch_name", "bb_type",
        "game_type", "home_team", "away_team", "inning_topbot", "player_name", "des"}
KEEP = {"game_date"}


def clean(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    for c in out.columns:
        if c in KEEP:
            continue
        s = out[c]
        if c in TEXT or s.dtype == "string" or s.dtype == object:
            out[c] = s.astype(object).where(s.notna(), None)
        elif pd.api.types.is_bool_dtype(s.dtype):
            out[c] = s.astype(object).where(s.notna(), None)
        elif pd.api.types.is_numeric_dtype(s.dtype):
            out[c] = pd.to_numeric(s, errors="coerce").astype("float64")
    return out
