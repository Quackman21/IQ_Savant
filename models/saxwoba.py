"""SAxwOBA -- spray-adjusted expected wOBA. Cut and dry:

    inputs:  exit velocity, launch angle, spray angle (field direction, from Statcast's hit coordinates)
    model:   P(out, 1B, 2B, 3B, HR) learned straight from what happened on 2023+ batted balls
    value:   those chances x that season's standard wOBA weights

Nothing else: no sprint speed, no batter side, no park adjustment, no blur, no blending, no calibration.
Walks / HBP / strikeouts keep their actual wOBA value (same as Savant's xwOBA).
The site's regular xwOBA stays Savant's; this is shown beside it as SAxwOBA / SAxwOBACON.

Trained by:  python -m pipeline.train_saxwoba  -> models/artifacts/saxwoba.joblib
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "saxwoba.joblib"
SHIFT_BAN = 2023          # two separate models: 2015-2022 (shift era) and 2023+ (shift ban); no carryover
FEATURES = ["ev", "la", "spray"]


def features(df: pd.DataFrame) -> pd.DataFrame:
    """ev, la, spray. Spray = field direction in degrees (- = left-field side, + = right-field side)."""
    c = lambda n: pd.to_numeric(df[n], errors="coerce") if n in df else pd.Series(np.nan, index=df.index)
    spray = np.degrees(np.arctan((c("hc_x") - 125.42) / (198.27 - c("hc_y")))) * 0.75
    return pd.DataFrame({"ev": c("launch_speed"), "la": c("launch_angle"), "spray": spray}, index=df.index)


_cache: dict = {}


def load() -> dict | None:
    if "m" not in _cache:
        if not ARTIFACT.exists():
            return None
        import joblib
        _cache["m"] = joblib.load(ARTIFACT)
    return _cache["m"]


def score_batted_balls(df: pd.DataFrame, bundle: dict | None = None) -> pd.Series:
    """SAxwOBA value of every ball in play (NaN for other rows). Works on one ball or a whole season."""
    from models import xwoba_iq as xq
    b = bundle or load()
    out = pd.Series(np.nan, index=df.index)
    if b is None or df.empty:
        return out
    bip = df["type"].eq("X") if "type" in df else pd.Series(True, index=df.index)
    if not bip.any():
        return out
    d = df[bip]
    if "eras" in b:                                       # each ball scored by its own era's model
        yr = pd.to_datetime(d["game_date"]).dt.year if "game_date" in d else pd.Series(9999, index=d.index)
        for era, part in (("pre", yr < b["split"]), ("post", yr >= b["split"])):
            if part.any() and b["eras"].get(era):
                out[part[part].index] = score_batted_balls(d[part], b["eras"][era])
        return out
    p = pd.DataFrame(b["model"].predict_proba(features(d)[FEATURES]), columns=list(b["model"].classes_),
                     index=d.index).reindex(columns=xq.CLASSES, fill_value=0.0)
    own, trained = xq.season_weights(d), b["weights"]
    years = pd.to_datetime(d["game_date"]).dt.year if "game_date" in d else pd.Series(max(trained), index=d.index)
    val = pd.Series(0.0, index=d.index)
    for y in years.unique():
        w = own.get(int(y)) or trained.get(int(y)) or trained[min(trained, key=lambda t: abs(t - int(y)))]
        m = years.eq(y)
        val[m] = sum(p.loc[m, h] * w[h] for h in xq.HITS)
    out[bip] = val
    return out


def expected_woba(df: pd.DataFrame) -> pd.Series:
    """Per pitch: SAxwOBA value on balls in play, actual wOBA value on BB / HBP / K. NaN if not trained."""
    actual = df.get("woba_value", pd.Series(np.nan, index=df.index))
    if load() is None:
        return pd.Series(np.nan, index=df.index)
    ours = score_batted_balls(df)
    bip = df.get("type", pd.Series("", index=df.index)).eq("X")
    return pd.Series(np.where(bip, ours, actual), index=df.index)
