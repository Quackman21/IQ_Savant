"""Core 4: Bat Speed, Bat-To-Ball, Swing Decisions, Ball Flight.

Grades are how many standard deviations a hitter is from MLB average on one
input stat, shown on the scale the user picks:
    20-80       50 = average, 10 points per standard deviation (clipped 20..80)
    Percentile  share of MLB hitters he's better than (from the same z-score)
    Plus        100 = average, 10 points per standard deviation
Each grade's input lives in GRADES below -- change it there to change the grade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core import hit_metrics as hm

WOBA_SCALE = 1.2          # runs per point of wOBA (approx., used for Ball Flight 600)
SCALES = ["20-80", "Percentile", "Plus"]


@dataclass(frozen=True)
class Grade:
    key: str
    label: str
    stat: str             # catalog stat the grade is built from
    desc: str


GRADES = {g.key: g for g in [
    Grade("BAT SPEED GRADE", "Bat Speed Grade", "HHEV",
          "20-80 Grade, Top 8th Exit Velocity\n(bat speed shows up in the hardest-hit balls)"),
    Grade("SC BAT SPEED GRADE", "SC Bat Speed Grade", "TOP 8TH BAT SPEED", "20-80 Grade, Top 8th Statcast bat speed"),
    Grade("B2B GRADE", "Bat-To-Ball Grade", "CONT%", "20-80 Grade, Bat-to-ball\n[from contact rate on swings]"),
    Grade("DECISION GRADE", "Decision Grade", "GD%", "20-80 Grade, Swing Decision\n[from rate of good decisions]"),
    Grade("0-0 DECISION GRADE", "0-0 Decision Grade", "GD% 0-0", "20-80 Grade, Swing Decision on first pitch"),
    Grade("<2K DECISION GRADE", "<2k Decision Grade", "GD% <2K", "20-80 Grade, Swing Decision with less than two strikes"),
    Grade("2K DECISION GRADE", "2k Decision Grade", "GD% 2K", "20-80 Grade, Swing Decision with two strikes"),
    Grade("FB DECISION GRADE", "FB Decision Grade", "GD% FB", "20-80 Grade, Swing Decision on fastballs"),
    Grade("BB DECISION GRADE", "BB Decision Grade", "GD% BB", "20-80 Grade, Swing Decision on breaking balls"),
    Grade("OS DECISION GRADE", "OS Decision Grade", "GD% OS", "20-80 Grade, Swing Decision on offspeed"),
    Grade("BALL FLIGHT GRADE", "Ball Flight Grade", "xwOBACON", "20-80 Grade, Ball Flight\n[from xwOBA on contact]"),
]}

# Columns every Core 4 tab starts with (after Year / Period / Age / Type / Team / vs)
SHARED = ["PA", "BIP", "K%", "BB%", "BARREL%", "wOBA", "xwOBA", "xwOBA MLB"]
TABS = {
    "Core 4": SHARED + ["BAT SPEED GRADE", "B2B GRADE", "DECISION GRADE", "BALL FLIGHT GRADE"],
    "Bat Speed": SHARED + ["# BAT SPEEDS", "BAT SPEED GRADE", "SC BAT SPEED GRADE", "AVG BAT SPEED",
                           "TOP 8TH BAT SPEED", "SWING LENGTH", "EV>95", "EV", "90th EV", "HHEV", "MAX EV",
                           "EV (FLY)", "90th EV (FLY)", "MAX EV (FLY)"],
    "Bat-To-Ball": SHARED + ["B2B GRADE", "CONT%", "Z-CONT%", "Z-CONT% (HARD)", "Z-CONT% (SOFT)",
                             "H-CONT%", "S-CONT%", "C-CONT%", "W-CONT%"],
    "Swing Decisions": SHARED + ["DECISION GRADE", "DECISIONS", "0-0 DECISIONS", "<2K DECISIONS", "2K DECISIONS",
                                 "0-0 DECISION GRADE", "<2K DECISION GRADE", "2K DECISION GRADE",
                                 "FB DECISIONS", "BB DECISIONS", "OS DECISIONS",
                                 "FB DECISION GRADE", "BB DECISION GRADE", "OS DECISION GRADE",
                                 "GD%", "SWING%", "CHASE%", "PRE2K O-SWING%", "2K O-SWING%",
                                 "H-SWING%", "S-SWING%", "C-SWING%", "W-SWING%"],
    "Ball Flight": SHARED + ["BALL FLIGHT GRADE", "BALL FLIGHT 600", "LA", "LAHHB", "ATTACK", "xwOBACON",
                             "xwOBACON MLB", "WEAK GB%", "GB%", "FB%", "WEAK FB%", "LD%", "LD+FB%", "PULL%",
                             "MIDDLE%", "OPPO%", "AIRPULL%", "FB-PULL%", "PULL-AIR%"],
}

# Stats calculated from other stats + the league reference (not in the pitch catalog)
DERIVED = {
    "xwOBA MLB": ("xwOBA on the MLB scale, i.e. using MLB adjustment (league wOBA - league xwOBA) for all levels",
                  "rate3"),
    "xwOBACON MLB": ("xwOBACON on the MLB scale, i.e. using MLB adjustment (league wOBACON - league xwOBACON) "
                     "for all levels", "rate3"),
    "BALL FLIGHT 600": ("Expected number of runs above average via Ball Flight per 600 PA\n"
                        "[(xwOBACON - MLB xwOBACON) / 1.2 x BIP per PA x 600]", "num1"),
}


# ----------------------------------------------------------------------------
def z_score(value, ref_row: dict, stat: str) -> float:
    avg, sd = (ref_row or {}).get(stat, (math.nan, math.nan))
    if value is None or any(isinstance(x, float) and math.isnan(x) for x in (value, avg, sd)) or not sd:
        return math.nan
    return (value - avg) / sd


def to_scale(z: float, scale: str) -> float:
    if math.isnan(z):
        return math.nan
    if scale == "Percentile":
        return 100 * 0.5 * (1 + math.erf(z / math.sqrt(2)))
    if scale == "Plus":
        return 100 + 10 * z
    return max(20.0, min(80.0, 50 + 10 * z))


def fmt_grade(v: float) -> str:
    return "-" if v is None or math.isnan(v) else f"{v:.0f}"


def grade(key: str, vals, ref_row: dict, scale: str) -> tuple[float, float]:
    """(grade on the chosen scale, z)."""
    g = GRADES[key]
    z = z_score(vals.get(g.stat), ref_row, g.stat)
    return to_scale(z, scale), z


def derived(key: str, vals, ref_row: dict) -> float:
    lg = lambda s: (ref_row or {}).get(s, (math.nan,))[0]
    if key == "xwOBA MLB":
        return vals.get("xwOBA", math.nan) + (lg("wOBA") - lg("xwOBA"))
    if key == "xwOBACON MLB":
        lg_wobacon = lg("wOBACON") if "wOBACON" in (ref_row or {}) else lg("xwOBACON")
        return vals.get("xwOBACON", math.nan) + (lg_wobacon - lg("xwOBACON"))
    if key == "BALL FLIGHT 600":
        pa, bip = vals.get("PA", 0), vals.get("BIP", 0)
        if not pa:
            return math.nan
        return (vals.get("xwOBACON", math.nan) - lg("xwOBACON")) / WOBA_SCALE * (bip / pa) * 600
    return math.nan


def kind_of(key: str) -> str:
    if key in GRADES:
        return "grade"
    if key in DERIVED:
        return "derived"
    return "stat"


def label_of(key: str) -> str:
    if key in GRADES:
        return GRADES[key].label
    if key in DERIVED:
        return key.replace("BALL FLIGHT", "Ball Flight")
    return hm.CATALOG[key].label


def desc_of(key: str, scale: str = "20-80") -> str:
    if key in GRADES:
        d = GRADES[key].desc
        return d.replace("20-80 Grade", {"20-80": "20-80 Grade", "Percentile": "Percentile",
                                         "Plus": "Plus (100 = average)"}[scale])
    if key in DERIVED:
        return DERIVED[key][0]
    return hm.CATALOG[key].desc
