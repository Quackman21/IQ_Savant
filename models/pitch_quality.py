"""IQ Stuff+ and Location+ (pitch quality models), after Driveline's Stuff+ and FanGraphs' Stuff+ / Location+.

Both predict the RUN VALUE of a pitch from the pitcher's side (higher = better for the pitcher). Balls in play
use the expected value of the contact (count + expected xwOBA on contact), not the actual result, so a
bloop single doesn't make a great pitch look bad.

  Stuff+     what the ball does, NOT where it goes: velo, induced vertical break, horizontal break (arm side +),
             spin, extension, release height, release side (arm side +), arm angle, and the velo / IVB / HB
             difference from the pitcher's primary fastball that season (FF, else SI, else FC).
             Scale: 100 = MLB-average pitch, 10 points = 1 standard deviation (pitch level).
  Location+  where it goes, NOT what it does: inside / away from the hitter, height within his zone, count,
             pitch type, same / opposite hand. Scale: 100 = average FOR THAT PITCH TYPE, 10 points = 1 SD.

Two models each (no carryover): 2015-2022 and 2023+ (pitch clock / shift ban). Every pitch is scored by its
own era's model. Trained by  python -m pipeline.train_pitch_quality  -> models/artifacts/pitch_quality.joblib
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "pitch_quality.joblib"
SHIFT_BAN = 2023
STUFF_FEATURES = ["velo", "ivb", "hb", "spin", "ext", "rel_z", "rel_x", "arm",
                  "d_velo", "d_ivb", "d_hb"]
LOC_FEATURES = ["x_in", "z_norm", "balls", "strikes", "ptype", "same_hand"]
PTYPE = {"FF": 0, "FA": 0, "SI": 1, "FT": 1, "FC": 2, "SL": 3, "ST": 3, "SV": 3, "CU": 4, "KC": 4, "CS": 4,
         "CH": 5, "FS": 6, "FO": 6, "SC": 5, "KN": 7, "EP": 7}
PTYPE_NAME = {0: "4-Seam", 1: "Sinker", 2: "Cutter", 3: "Slider / Sweeper", 4: "Curveball", 5: "Changeup",
              6: "Splitter", 7: "Other"}
PRIMARY = ["FF", "SI", "FC"]


def _c(df, n):
    return pd.to_numeric(df[n], errors="coerce") if n in df else pd.Series(np.nan, index=df.index)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Every input for both models. Fastball differences use the pitcher's primary fastball in that season,
    from the pitches in df (a whole season for training; the player's pitches on the site)."""
    arm = np.where(df["p_throws"].astype(str).eq("L") if "p_throws" in df else False, 1.0, -1.0)
    left_bat = df["stand"].astype(str).eq("L") if "stand" in df else pd.Series(False, index=df.index)
    side = np.where(left_bat, 1.0, -1.0)
    top, bot = _c(df, "sz_top"), _c(df, "sz_bot")
    span = (top - bot).where((top - bot) > 0.5)
    pt = df["pitch_type"].astype(str) if "pitch_type" in df else pd.Series("", index=df.index)
    f = pd.DataFrame({
        "velo": _c(df, "release_speed"), "ivb": 12 * _c(df, "pfx_z"),
        # arm side +: a righty's arm side is the catcher's left (negative pfx_x / release_pos_x), so flip righties
        "hb": 12 * _c(df, "pfx_x") * arm, "spin": _c(df, "release_spin_rate"), "ext": _c(df, "release_extension"),
        "rel_z": _c(df, "release_pos_z"), "rel_x": _c(df, "release_pos_x") * arm, "arm": _c(df, "arm_angle"),
        "x_in": _c(df, "plate_x") * side, "z_norm": (_c(df, "plate_z") - bot) / span,
        "balls": _c(df, "balls").clip(0, 3), "strikes": _c(df, "strikes").clip(0, 2),
        "ptype": pt.map(PTYPE).astype(float),
        "same_hand": (df["p_throws"].astype(str).values == np.where(left_bat, "L", "R")).astype(float)
        if "p_throws" in df else 0.0,
    }, index=df.index)
    # primary fastball per pitcher-season
    who = df["pitcher"] if "pitcher" in df else pd.Series(0, index=df.index)
    yr = pd.to_datetime(df["game_date"]).dt.year if "game_date" in df else pd.Series(0, index=df.index)
    key = pd.DataFrame({"p": who.values, "y": yr.values, "pt": pt.values}, index=df.index)
    fb = key["pt"].isin(PRIMARY)
    ref = pd.DataFrame(index=df.index, columns=["velo", "ivb", "hb"], dtype=float)
    if fb.any():
        counts = key[fb].groupby(["p", "y", "pt"]).size().reset_index(name="n")
        counts["rank"] = counts["pt"].map({p: i for i, p in enumerate(PRIMARY)})
        prim = counts.sort_values(["p", "y", "n", "rank"], ascending=[True, True, False, True]) \
                     .drop_duplicates(["p", "y"])[["p", "y", "pt"]]
        means = pd.concat([key, f[["velo", "ivb", "hb"]]], axis=1).merge(prim, on=["p", "y", "pt"]) \
                  .groupby(["p", "y"])[["velo", "ivb", "hb"]].mean()
        m = key.join(means, on=["p", "y"])
        ref = m[["velo", "ivb", "hb"]].astype(float)
    f["d_velo"] = f["velo"] - ref["velo"]
    f["d_ivb"] = f["ivb"] - ref["ivb"]
    f["d_hb"] = f["hb"] - ref["hb"]
    return f


def target(df: pd.DataFrame, bip_table: dict | None = None, slope: float = 0.85, xq_mean: float = 0.37) -> pd.Series:
    """Pitcher-side run value per pitch; balls in play -> expected value of the contact."""
    rv = _c(df, "delta_run_exp")
    balls = df["description"].isin({"ball", "blocked_ball"}) if "description" in df else pd.Series(False, index=df.index)
    if rv[balls].mean() < 0:                     # make sure it's the batting side first
        rv = -rv
    bip = df["type"].astype(str).eq("X") if "type" in df else pd.Series(False, index=df.index)
    xq = _c(df, "estimated_woba_using_speedangle")
    if bip_table:
        cnt = _c(df, "balls").clip(0, 3).astype("Int64").astype(str) + "-" + _c(df, "strikes").clip(0, 2).astype("Int64").astype(str)
        base = cnt.map(bip_table).astype(float)
        exp = base + slope * (xq - xq_mean)
        rv = rv.where(~(bip & exp.notna()), exp)
    return -rv                                       # pitcher side: higher = better


_cache: dict = {}


def load() -> dict | None:
    if "m" not in _cache:
        if not ARTIFACT.exists():
            return None
        import joblib
        _cache["m"] = joblib.load(ARTIFACT)
    return _cache["m"]


def _score_one(f: pd.DataFrame, b: dict) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=f.index, columns=["stuff_plus", "loc_plus"])
    ok = f["velo"].notna()
    if ok.any():
        s = b["stuff"].predict(f.loc[ok, STUFF_FEATURES])
        out.loc[ok, "stuff_plus"] = 100 + 10 * (s - b["stuff_mean"]) / b["stuff_sd"]
    okl = f[["x_in", "z_norm"]].notna().all(axis=1)
    if okl.any():
        l = pd.Series(b["loc"].predict(f.loc[okl, LOC_FEATURES]), index=f.index[okl])
        g = f.loc[okl, "ptype"]
        mean = g.map(b["loc_mean"]).fillna(b["loc_mean_all"])
        sd = g.map(b["loc_sd"]).fillna(b["loc_sd_all"])
        out.loc[okl, "loc_plus"] = 100 + 10 * (l - mean) / sd
    return out


def score(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Per pitch: stuff_plus, loc_plus (NaN if not trained / missing data). Era bundles route by season."""
    b = bundle or load()
    out = pd.DataFrame(np.nan, index=df.index, columns=["stuff_plus", "loc_plus"])
    if b is None or df.empty:
        return out
    f = features(df)
    if "eras" not in b:
        return _score_one(f, b)
    yr = pd.to_datetime(df["game_date"]).dt.year if "game_date" in df else pd.Series(9999, index=df.index)
    for e, m in (("pre", yr < b["split"]), ("post", yr >= b["split"])):
        if m.any() and b["eras"].get(e):
            out.loc[m] = _score_one(f[m], b["eras"][e]).values
    return out
