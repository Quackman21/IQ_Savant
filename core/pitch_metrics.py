"""Pitching tables: one row per pitch type (Statcast labels) per period / split.

Everything here comes straight from public Statcast pitch data. Columns that
need one of our models (grades, expected rates, shape, command, seam-shifted
wake, spin efficiency, zone-adjusted approach angles, clusters) are defined
but left blank -- see MODEL_COLUMNS.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.statcast_clean import clean

SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
          "foul_bunt", "missed_bunt", "bunt_foul_tip", "foul_pitchout"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt", "foul_tip"}
BALLS = {"ball", "blocked_ball", "pitchout", "hit_by_pitch", "intent_ball"}
TAKES = BALLS | {"called_strike"}

# Statcast pitch colors (Savant palette)
PITCH_COLORS = {"FF": "#D22D49", "SI": "#FE9D00", "FC": "#933F2C", "SL": "#EEE716", "ST": "#DDB33A",
                "SV": "#93AFD4", "CU": "#00D1ED", "KC": "#6236CD", "CS": "#0068FF", "CH": "#1DBE3A",
                "FS": "#3BACAC", "FO": "#55CCAB", "SC": "#60DB33", "KN": "#3C44CD", "EP": "#888888",
                "FA": "#888888", "PO": "#888888"}
TEXT_ON = {"SL": "#7a7400", "ST": "#8a6a00", "SI": "#c26f00"}       # darker text for pale colors

B, G = "blank", "grade"
_NP = [("N", "#", "Number of pitches thrown", "int"), ("PCT", "%", "Share of pitches in this row group", "pct0")]

SUMMARY = [*_NP, "|",
           ("ARESULTS", "aResults", "20-80 Grade, Actual Results Produced by Pitch", G),
           ("XRESULTS", "xResults", "20-80 Grade, Results expected out of pitcher", G),
           ("COMMAND", "Command", "The miss distance or expected miss distance of a pitch converted to the 20-80 scale", G),
           ("ACTL", "ActL", "20-80 Action Grade, Versus LHH", G),
           ("ACTR", "ActR", "20-80 Action Grade, Versus RHH", G),
           ("LOCATION", "Location", "20-80 Grade, Location of Pitch", G)]
DETAILS = [*_NP, "|",
           ("VELO", "Velo", "Average Velocity", "num1"), ("MAX", "Max", "Maximum Velocity", "num1"), "|",
           ("HT", "Ht", "Average Release Height (ft)", "num2"), ("SIDE", "Side", "Average Release Side (ft)", "num2"),
           ("EXT", "Ext", "Average Release Extension (ft)", "num2"),
           ("ARM", "Arm Ang.", "Arm angle is calculated at the moment of pitch release as the angle between two "
                               "vectors: (1) a line drawn directly from the pitcher's shoulder to his wrist and (2) a "
                               "line drawn from the pitcher's shoulder directly to the ground (Statcast, 2024+)", "num1"), "|",
           ("IVB", "IVB", "Average Induced Vertical Break (inches, Statcast)", "num2"),
           ("HB", "HB", "Average Horizontal Break (inches, arm side positive, Statcast)", "num2"),
           ("SPIN_EFF", "Spin Eff.", "Percentage of a pitcher's total spin that directly contributes to the movement "
                                     "of the ball (our model — coming)", B),
           ("SPIN_DIR", "Spin Dir", "Direction of spin of the ball (clock face, Statcast spin axis)", "clock"),
           ("OBS_DIR", "Obs. Dir", "Direction of movement, due to spin (clock face, from observed movement)", "clock"),
           ("OMV_VB", "O-Mv VB", "Average Vertical Movement induced by Seams (our model — coming)", B),
           ("OMV_HB", "O-Mv HB", "Average Horizontal Movement induced by Seams (our model — coming)", B),
           ("SPIN", "Spin", "Average Total Spin Rate (rpm)", "int"), "|",
           ("VAA", "VAA", "Vertical approach angle at the front of home plate (degrees)", "num2"),
           ("HAA", "HAA", "Horizontal approach angle at the front of home plate (degrees)", "num2"),
           ("ZAVAA", "ZAVAA", "The estimated vertical approach angle of a pitch of the observed velocity, release "
                              "point, and movement, if thrown to the middle of the zone (our model — coming)", B),
           ("ZAHAA", "ZAHAA", "The estimated horizontal approach angle of a pitch of the observed velocity, release "
                              "point, and movement, if thrown to the middle of the zone (our model — coming)", B), "|",
           ("ACTL", "ActL", "20-80 Action Grade, Versus LHH", G), ("ACTR", "ActR", "20-80 Action Grade, Versus RHH", G)]
PILLARS = [*_NP, "|",
           ("VELO_T8", "Velo Top 8th", "The average of the top 8th velocities for a given pitch type", "num1"), "|",
           ("SHAPE", "Shape", "The action grade of a pitch when substituting the expected velocity based on shape", G),
           ("SHAPE_L", "Shape vL", "The action grade of a pitch when substituting the expected velocity based on "
                                   "shape versus left", G),
           ("SHAPE_R", "Shape vR", "The action grade of a pitch when substituting the expected velocity based on "
                                   "shape versus right", G), "|",
           ("COMMAND", "Command", "The miss distance or expected miss distance of a pitch converted to the 20-80 scale", G),
           ("MISS", "Miss", "The distance between the final plate location and intended location (needs the "
                            "catcher's target — not public)", B)]
RATES = [*_NP, "|",
         ("ASW", "aSw%", "Actual Swing Rate", "pct"), ("ABL", "aBl%", "Actual Balls on Takes Rate", "pct"),
         ("AWF", "aWf%", "Actual Misses on Swings Rate", "pct"),
         ("AIPRV", "aIPRV", "Actual Average Run Value on Balls In Play (batting team's run value; lower = better "
                            "for the pitcher)", "num2"), "|",
         ("XSW", "xSw%", "Expected Swing Rate (our model — coming)", B),
         ("XBL", "xBl%", "Expected Balls on Takes Rate (our model — coming)", B),
         ("XWF", "xWf%", "Expected Misses on Swings Rate (our model — coming)", B),
         ("XIPRV", "xIPRV", "Expected Average Run Value on Balls In Play (our model — coming)", B)]
TABLE_TABS = {"Summary": SUMMARY, "Details": DETAILS, "Pillars": PILLARS, "Rates": RATES}

# Splits (the "Splits" dropdown)
SPLITS = {"Games": "game", "Batter Hand": "hand", "Count": "count", "RISP": "risp", "Cluster": "cluster"}
SPLIT_LABEL = {"game": "DATE", "hand": "HAND", "count": "COUNT", "risp": "RISP", "cluster": "CLUSTER"}
COUNT_ORDER = ["First Pitch", "Hitter's Count", "Pitcher's Count"]
MODEL_SPLITS = {"cluster"}               # blank until our cluster model exists


def count_group(balls: pd.Series, strikes: pd.Series) -> pd.Series:
    """First Pitch = 0-0. Hitter's = balls > strikes (not 3-2). Pitcher's = strikes > balls, plus 2-2.
    Everything else (1-1, 3-2) is in no count group."""
    b, s = balls.fillna(-1), strikes.fillna(-1)
    first = (b == 0) & (s == 0)
    hitter = (b > s) & ~((b == 3) & (s == 2))
    pitcher = (s > b) | ((b == 2) & (s == 2))
    return pd.Series(np.select([first, hitter, pitcher], COUNT_ORDER, None), index=balls.index)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """Statcast pitches -> the columns the pitching tables / charts use."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = clean(df)
    c = lambda n: df[n] if n in df else pd.Series(np.nan, index=df.index)
    d = pd.DataFrame(index=df.index)
    d["pitch"] = c("pitch_type").fillna("UN")
    d["pitch_name"] = c("pitch_name")
    d["game_date"] = pd.to_datetime(c("game_date")).dt.date
    d["game_pk"] = c("game_pk")
    d["game_type"] = c("game_type")
    d["velo"], d["spin"] = c("release_speed"), c("release_spin_rate")
    d["ht"], d["side"], d["ext"] = c("release_pos_z"), c("release_pos_x"), c("release_extension")
    d["arm"] = c("arm_angle")
    hb_cat = 12 * c("pfx_x")                                       # catcher's view (Statcast)
    arm_side = np.where(c("p_throws").eq("R"), -1, 1)
    d["ivb"], d["hb"] = 12 * c("pfx_z"), hb_cat * arm_side         # HB shown arm-side positive
    d["hb_cat"] = hb_cat                                           # for plots (catcher's view)
    d["spin_axis"] = c("spin_axis")
    d["obs_axis"] = (np.degrees(np.arctan2(-hb_cat, d["ivb"])) + 180) % 360   # 180 = 12:00 (pure ride)
    d["plate_x"], d["plate_z"] = c("plate_x"), c("plate_z")
    d["sz_top"], d["sz_bot"] = c("sz_top"), c("sz_bot")
    vy, vz, vx = c("vy0"), c("vz0"), c("vx0")
    ay, az, ax = c("ay"), c("az"), c("ax")
    vyf = -np.sqrt(np.clip(vy ** 2 - 2 * ay * (50 - 17 / 12), 0, None))
    t = (vyf - vy) / ay
    d["vaa"] = -np.degrees(np.arctan((vz + az * t) / vyf))
    d["haa"] = -np.degrees(np.arctan((vx + ax * t) / vyf))
    desc = c("description").fillna("")
    d["swing"] = desc.isin(SWINGS)
    d["whiff"] = desc.isin(WHIFFS)
    d["take"] = desc.isin(TAKES)
    d["ball"] = desc.isin(BALLS)
    d["bip"] = c("type").eq("X")
    d["rv"] = c("delta_run_exp")
    d["stand"] = c("stand")
    d["hand"] = c("stand").map({"L": "vLHH", "R": "vRHH"})
    d["count"] = count_group(c("balls"), c("strikes"))
    d["risp"] = np.where(c("on_2b").notna() | c("on_3b").notna(), "RISP", "No RISP")
    d["game"] = d["game_date"].astype(str)
    d["cluster"] = None
    return d


def _top8(s: pd.Series) -> float:
    s = s.dropna()
    return s.nlargest(max(1, math.ceil(len(s) / 8))).mean() if len(s) else np.nan


def _circ_mean(deg: pd.Series) -> float:
    r = np.radians(deg.dropna())
    if r.empty:
        return np.nan
    return (np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) + 360) % 360


def row_values(g: pd.DataFrame, total: int) -> dict:
    sw, tk, bip = g["swing"].sum(), g["take"].sum(), g["bip"]
    return {
        "N": len(g), "PCT": len(g) / total if total else np.nan,
        "VELO": g["velo"].mean(), "MAX": g["velo"].max(), "VELO_T8": _top8(g["velo"]),
        "HT": g["ht"].mean(), "SIDE": g["side"].mean(), "EXT": g["ext"].mean(), "ARM": g["arm"].mean(),
        "IVB": g["ivb"].mean(), "HB": g["hb"].mean(), "SPIN": g["spin"].mean(),
        "SPIN_DIR": _circ_mean(g["spin_axis"]), "OBS_DIR": _circ_mean(g["obs_axis"]),
        "VAA": g["vaa"].mean(), "HAA": g["haa"].mean(),
        "ASW": sw / len(g) if len(g) else np.nan, "ABL": g["ball"].sum() / tk if tk else np.nan,
        "AWF": g["whiff"].sum() / sw if sw else np.nan,
        "AIPRV": g.loc[bip, "rv"].mean() if bip.any() else np.nan,
    }


def table_rows(d: pd.DataFrame, split_dims: list[str], pitch: str | None = None) -> list[tuple[dict, dict]]:
    """[(labels, values)] -- labels = {split: value, ..., "Pitch": code}. Pitch types sorted by usage.
    % is the share within the same split group."""
    if d is None or d.empty:
        return []
    dims = [x for x in split_dims if x not in MODEL_SPLITS]
    order = d["pitch"].value_counts().index.tolist()
    out = []
    groups = [((), d)] if not dims else d.groupby(dims, dropna=True, sort=True)
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(dims, key))
        total = len(g)
        for pt in order:
            if pitch and pt != pitch:
                continue
            gp = g[g["pitch"] == pt]
            if gp.empty:
                continue
            out.append(({**labels, "Pitch": pt}, row_values(gp, total)))
    if "count" in dims:
        rank = {v: i for i, v in enumerate(COUNT_ORDER)}
        out.sort(key=lambda r: rank.get(r[0].get("count"), 9))
    if "game" in dims:
        out.sort(key=lambda r: r[0].get("game", ""), reverse=True)
    return out


def clock(deg) -> str:
    """Spin axis degrees -> clock face (180 = 12:00, 90 = 9:00 from the pitcher's view, Statcast convention)."""
    if deg is None or (isinstance(deg, float) and math.isnan(deg)):
        return "-"
    minutes = round(((deg + 180) % 360) / 360 * 12 * 60)
    h, m = divmod(minutes, 60)
    h = 12 if h == 0 else h
    return f"{h}:{m:02d}"
