"""Swing-decision models: was each pitch a GOOD decision (swing or take)?

Every Swing Decisions stat (GD%, Decision grades) asks good_decision(), so
swapping in your real model is one line: change ACTIVE.

    def iq_v1(df):            # df = Statcast pitches, plus df["attack"] (Heart/Shadow/Chase/Waste)
        ...                   # return 1.0 = good, 0.0 = bad, NaN = no grade for this pitch
    MODELS["iq_v1"] = iq_v1
    ACTIVE = "iq_v1"
Then rebuild the league reference (--force) so league averages use it too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAKES_OK_2K = False   # placeholder rule switch: with 2 strikes, is taking a Shadow pitch fine?


def placeholder_v0(df: pd.DataFrame, swing: pd.Series, attack: pd.Series) -> pd.Series:
    """Simple stand-in until the real decision model exists:
       Heart  -> swing is good, take is bad
       Shadow -> either is fine (borderline pitch), except a take with 2 strikes is bad
       Chase / Waste -> take is good, swing is bad
       Unknown location -> no grade."""
    two_k = df["strikes"].fillna(0).ge(2) if "strikes" in df else pd.Series(False, index=df.index)
    good = pd.Series(np.nan, index=df.index)
    good[attack.eq("Heart")] = swing[attack.eq("Heart")].astype(float)
    sh = attack.eq("Shadow")
    good[sh] = np.where(two_k[sh] & ~swing[sh] & (not TAKES_OK_2K), 0.0, 1.0)
    out = attack.isin(["Chase", "Waste"])
    good[out] = (~swing[out]).astype(float)
    return good


MODELS = {"placeholder_v0": placeholder_v0}
ACTIVE = "placeholder_v0"
LABEL = {"placeholder_v0": "Placeholder decision model (zone-based)"}


def good_decision(df, swing, attack, model: str | None = None) -> pd.Series:
    return MODELS[model or ACTIVE](df, swing, attack)
