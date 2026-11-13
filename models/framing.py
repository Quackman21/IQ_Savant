"""Catcher framing, built from public pitch data so it works for any date range.
ACTIVE = iq_v1 (every taken pitch, league-umpire strike chance from IQ Swing Decisions, count-valued runs);
placeholder_v0 kept below (shadow pitches only, location-bin strike rate).

Metrics (summarize): Framing Opps, Extra Strikes, Extra Strikes / Game, SLAA (per pitch), SLAA+ (100 = avg),
Framing Runs, and edge percentiles (arm side / glove side / top / bottom).

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


def iq_v1(df: pd.DataFrame) -> pd.DataFrame:
    """IQ framing: EVERY taken pitch (called strike / ball) the catcher received.
    exp = league-average-umpire strike chance from the IQ Swing Decisions called-strike model (location vs
          the hitter's own zone, count, batter side, pitch type -- not the catcher, umpire or pitcher).
    extra strike = (1 if called strike else 0) - exp   (45% pitch called a strike +0.55, a ball -0.45)
    runs = extra strike x what a strike instead of a ball is worth in that count.
    Falls back to the placeholder until the decision model is trained."""
    from models import decisions_iq as dq
    if df is None or df.empty or dq.load() is None or not {"balls", "strikes"} <= set(df.columns):
        return placeholder_v0(df)                       # not trained yet / no count -> old model
    need = {"description", "plate_x", "plate_z", "sz_top", "sz_bot", "fielder_2", "game_date"}
    if not need <= set(df.columns):
        return pd.DataFrame(columns=["game_date", "game_pk", "catcher", "region", "strike", "exp", "runs"])
    from core import abs_calls
    try:
        df = abs_calls.original_calls(df)               # 2026+: the umpire's ORIGINAL call, not the ABS result
    except Exception as e:
        print(f"  (ABS challenges not loaded: {e} -- using the calls as recorded)", flush=True)
    d = df[df["description"].isin(TAKEN)].dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot", "fielder_2"])
    if d.empty:
        return pd.DataFrame(columns=["game_date", "game_pk", "catcher", "region", "strike", "exp", "runs"])
    exp = center(dq.called_strike_chance(d), d["description"].eq("called_strike"),
                 pd.to_datetime(d["game_date"]).dt.year)
    val = dq.strike_value(d)
    try:
        sit = dq.strike_value_situational(d)
    except Exception:
        sit = pd.Series(np.nan, index=d.index)
    d = d.assign(exp=exp, val=val, sit=sit).dropna(subset=["exp"])
    xn, zn = _norm(d)
    r = np.maximum(xn.abs(), zn.abs())
    edge = np.where(zn.abs() >= xn.abs(), np.where(zn > 0, "TOZ", "BOZ"), np.where(xn > 0, "SIDE+", "SIDE-"))
    region = np.select([r < 0.67, r < 1.33], ["IN", "EDGE"], "OUT")
    region = np.where(region == "EDGE", edge, region)               # shadow pitches keep their side
    strike = d["description"].eq("called_strike").astype(float)
    return pd.DataFrame({"game_date": pd.to_datetime(d["game_date"]).dt.date.astype(str),
                         "game_pk": d["game_pk"] if "game_pk" in d else 0,
                         "catcher": d["fielder_2"].astype(int), "region": region, "strike": strike,
                         "exp": d["exp"], "runs": (strike - d["exp"]) * d["val"].fillna(RUNS_PER_STRIKE),
                         # situational (count + runners + outs) -- stored for after-action reports, not the profile
                         "runs_sit": (strike - d["exp"]) * d["sit"].fillna(d["val"]).fillna(RUNS_PER_STRIKE)})


REG_TAKES = {"pre": 720, "post": 1080}      # regressed SLAA: league-average takes added (framing study split-half)


def center(p: pd.Series, strike: pd.Series, year: pd.Series) -> pd.Series:
    """Shift each season's strike chances (on the log-odds scale) so the league adds to 0 extra strikes.
    Only for whole seasons (the league files) -- never on one catcher's pitches."""
    p = p.astype(float).clip(1e-6, 1 - 1e-6)
    out = p.copy()
    for y in year.unique():
        m = (year == y) & p.notna()
        if m.sum() < 1000:
            continue
        lo, hi, x, target = -2.0, 2.0, np.log(p[m] / (1 - p[m])), strike[m].sum()
        for _ in range(50):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(x + mid)))).sum() < target:
                lo = mid
            else:
                hi = mid
        out[m] = 1 / (1 + np.exp(-(x + (lo + hi) / 2)))
    return out


MODELS = {"placeholder_v0": placeholder_v0, "iq_v1": iq_v1}
ACTIVE = "iq_v1"
EDGE_REGIONS = {"TOZ", "BOZ", "SIDE+", "SIDE-"}


def framing_pitches(df: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    return MODELS[model or ACTIVE](df)


def daily_table(fp: pd.DataFrame) -> pd.DataFrame:
    """Small file to save per season: catcher x game x region -> n, strikes, expected, runs."""
    if fp.empty:
        return pd.DataFrame(columns=["game_date", "game_pk", "catcher", "region", "n", "strikes", "exp", "runs",
                                     "runs_sit"])
    keys = ["game_date"] + (["game_pk"] if "game_pk" in fp else []) + ["catcher", "region"]
    agg = dict(n=("strike", "size"), strikes=("strike", "sum"), exp=("exp", "sum"))
    for c in ("runs", "runs_sit"):
        if c in fp:
            agg[c] = (c, "sum")
    return fp.groupby(keys).agg(**agg).reset_index()


def summarize(daily: pd.DataFrame, start: str, end: str, throws: str = "R", min_n: int = 10) -> dict:
    """{catcher: {F_OPPS, F_RS, F_AS, F_GS, F_TOZ, F_BOZ}} for a date window.
    Zone values are percentiles (vs catchers in the same window) of strikes above expected per pitch.
    RHT catcher: glove side = third-base side (plate_x < 0) = SIDE-."""
    if daily is None or daily.empty:
        return {}
    w = daily[(daily["game_date"] >= start) & (daily["game_date"] <= end)]
    if w.empty:
        return {}
    cols = ["n", "strikes", "exp"] + [c for c in ("runs", "runs_sit") if c in w]
    tot = w.groupby("catcher")[cols].sum()
    games = w.groupby("catcher")["game_pk" if "game_pk" in w else "game_date"].nunique()
    out = {}
    for c, r in tot.iterrows():
        xs = r.strikes - r.exp
        out[c] = {"F_OPPS": r.n, "F_XS": xs, "F_XS/G": xs / games[c] if games[c] else np.nan,
                  "F_SLAA": xs / r.n if r.n else np.nan, "F_SLAA+": 100 * r.strikes / r.exp if r.exp else np.nan,
                  "F_rSLAA": xs / (r.n + REG_TAKES["post" if str(start)[:4] >= "2023" else "pre"]),
                  "F_RS": r.runs if "runs" in r and pd.notna(r.runs) else xs * RUNS_PER_STRIKE,
                  "F_RS_SIT": r.runs_sit if "runs_sit" in r and pd.notna(r.runs_sit) else np.nan}
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
