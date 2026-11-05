"""Catcher framing model (placeholder, built from public pitch data so it works for
any date range). Like xwOBA / decisions: swap the model by changing ACTIVE.

Framing pitches = taken pitches (called strike / ball) in the Shadow zone.
Expected strike = MLB called-strike rate for that location bin and batter side
(same season). Extra strikes = actual - expected. Runs = extra strikes x RUNS_PER_STRIKE.
Locations (catcher's view, plate_x > 0 = first-base side):
  TOZ / BOZ = top / bottom edge, AS / GS = arm side / glove side edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RUNS_PER_STRIKE = 0.125
TAKEN = {"called_strike", "ball"}
BIN = 0.1


def _norm(df):
    half_w = 17 / 24
    mid = (df["sz_top"] + df["sz_bot"]) / 2
    half_h = (df["sz_top"] - df["sz_bot"]) / 2
    return df["plate_x"] / half_w, (df["plate_z"] - mid) / half_h


def placeholder_v0(df: pd.DataFrame) -> pd.DataFrame:
    """Pitch rows -> one row per framing pitch:
    game_date, catcher, region (TOZ/BOZ/SIDE+/SIDE-), strike (0/1), exp (0..1)."""
    need = {"description", "plate_x", "plate_z", "sz_top", "sz_bot", "fielder_2", "game_date"}
    if df is None or df.empty or not need <= set(df.columns):
        return pd.DataFrame(columns=["game_date", "catcher", "region", "strike", "exp"])
    d = df[df["description"].isin(TAKEN)].dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot", "fielder_2"])
    xn, zn = _norm(d)
    r = np.maximum(xn.abs(), zn.abs())
    d = d[(r >= 0.67) & (r < 1.33)].copy()                     # Shadow zone
    xn, zn = xn[d.index], zn[d.index]
    d["region"] = np.where(zn.abs() >= xn.abs(), np.where(zn > 0, "TOZ", "BOZ"),
                           np.where(xn > 0, "SIDE+", "SIDE-"))
    d["strike"] = d["description"].eq("called_strike").astype(float)
    stand = d["stand"] if "stand" in d else pd.Series("R", index=d.index)
    key = pd.DataFrame({"bx": (xn / BIN).round(), "bz": (zn / BIN).round(), "st": stand})
    d["exp"] = d["strike"].groupby([key["bx"], key["bz"], key["st"]]).transform("mean")
    return pd.DataFrame({"game_date": pd.to_datetime(d["game_date"]).dt.date.astype(str),
                         "catcher": d["fielder_2"].astype(int), "region": d["region"],
                         "strike": d["strike"], "exp": d["exp"]})


MODELS = {"placeholder_v0": placeholder_v0}
ACTIVE = "placeholder_v0"


def framing_pitches(df: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    return MODELS[model or ACTIVE](df)


def daily_table(fp: pd.DataFrame) -> pd.DataFrame:
    """Small file to save per season: catcher x date x region -> n, strikes, expected."""
    if fp.empty:
        return pd.DataFrame(columns=["game_date", "catcher", "region", "n", "strikes", "exp"])
    return (fp.groupby(["game_date", "catcher", "region"])
              .agg(n=("strike", "size"), strikes=("strike", "sum"), exp=("exp", "sum")).reset_index())


def summarize(daily: pd.DataFrame, start: str, end: str, throws: str = "R", min_n: int = 10) -> dict:
    """{catcher: {F_OPPS, F_RS, F_AS, F_GS, F_TOZ, F_BOZ}} for a date window.
    Zone values are percentiles (vs catchers in the same window) of strikes above expected per pitch.
    RHT catcher: glove side = third-base side (plate_x < 0) = SIDE-."""
    if daily is None or daily.empty:
        return {}
    w = daily[(daily["game_date"] >= start) & (daily["game_date"] <= end)]
    if w.empty:
        return {}
    tot = w.groupby("catcher")[["n", "strikes", "exp"]].sum()
    out = {c: {"F_OPPS": r.n, "F_RS": (r.strikes - r.exp) * RUNS_PER_STRIKE} for c, r in tot.iterrows()}
    side = {"F_TOZ": "TOZ", "F_BOZ": "BOZ",
            "F_GS": "SIDE-" if throws == "R" else "SIDE+", "F_AS": "SIDE+" if throws == "R" else "SIDE-"}
    for key, region in side.items():
        g = w[w["region"] == region].groupby("catcher")[["n", "strikes", "exp"]].sum()
        g = g[g["n"] >= min_n]
        if g.empty:
            continue
        rate = (g["strikes"] - g["exp"]) / g["n"]
        pct = rate.rank(pct=True) * 100
        for c, v in pct.items():
            out.setdefault(c, {})[key] = float(v)
    return out
