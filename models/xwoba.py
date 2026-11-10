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

from pathlib import Path

import numpy as np
import pandas as pd


def savant(df: pd.DataFrame) -> pd.Series:
    """Baseball Savant's xwOBA: expected value on batted balls (EV + LA, plus
    sprint speed on some ground balls), actual wOBA value on walks / HBP / Ks."""
    actual = df.get("woba_value", pd.Series(np.nan, index=df.index))
    est = df.get("estimated_woba_using_speedangle", pd.Series(np.nan, index=df.index))
    bip = df.get("type", pd.Series("", index=df.index)).eq("X")
    return pd.Series(np.where(bip & est.notna(), est, actual), index=df.index)


def iq_v1(df: pd.DataFrame) -> pd.Series:
    """Our xwOBA (models/xwoba_iq.py): batter side + EV + LA + spray angle, park-neutral.
    Batted balls get our expected value; walks / HBP / Ks keep their actual (season) wOBA value.
    Falls back to Savant's number for any ball our model can't score."""
    from models import xwoba_iq
    actual = df.get("woba_value", pd.Series(np.nan, index=df.index))
    ours = xwoba_iq.score_batted_balls(df)
    bip = df.get("type", pd.Series("", index=df.index)).eq("X")
    fallback = savant(df)
    return pd.Series(np.where(bip & ours.notna(), ours, np.where(bip, fallback, actual)), index=df.index)


MODELS = {"savant": savant, "iq_v1": iq_v1}
LABEL = {"savant": "Savant xwOBA", "iq_v1": "IQ xwOBA v1"}

# Which model the whole site uses. Set by:  python -m pipeline.train_xwoba --activate
# (or --use-savant to switch back). Stored in a tiny file so no code edits are needed.
_ACTIVE_FILE = Path(__file__).resolve().parent / "artifacts" / "ACTIVE_XWOBA"


def _read_active() -> str:
    try:
        name = _ACTIVE_FILE.read_text().strip()
    except OSError:
        return "savant"
    if name == "iq_v1":
        from models import xwoba_iq
        if not xwoba_iq.ARTIFACT.exists():
            return "savant"
    return name if name in MODELS else "savant"


ACTIVE = _read_active()


def set_active(name: str) -> None:
    _ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ACTIVE_FILE.write_text(name + "\n")


def expected_woba(df: pd.DataFrame, model: str | None = None) -> pd.Series:
    return MODELS[model or ACTIVE](df)
