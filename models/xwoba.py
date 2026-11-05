"""xwOBA models. Every page and the league reference ask for xwOBA through
expected_woba(), so swapping models is one line: change ACTIVE.

To add your own model:
    def iq_v1(df):            # df = Statcast pitches (one row per pitch)
        ...                   # return expected wOBA value per row (NaN where it doesn't apply)
    MODELS["iq_v1"] = iq_v1
    ACTIVE = "iq_v1"
Then rebuild the league reference so league averages use the same model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def savant(df: pd.DataFrame) -> pd.Series:
    """Baseball Savant's xwOBA: expected value on batted balls (EV + LA, plus
    sprint speed on some ground balls), actual wOBA value on walks / HBP / Ks."""
    actual = df.get("woba_value", pd.Series(np.nan, index=df.index))
    est = df.get("estimated_woba_using_speedangle", pd.Series(np.nan, index=df.index))
    bip = df.get("type", pd.Series("", index=df.index)).eq("X")
    return pd.Series(np.where(bip & est.notna(), est, actual), index=df.index)


MODELS = {"savant": savant}
ACTIVE = "savant"
LABEL = {"savant": "Savant xwOBA"}


def expected_woba(df: pd.DataFrame, model: str | None = None) -> pd.Series:
    return MODELS[model or ACTIVE](df)
