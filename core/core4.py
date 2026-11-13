"""Core 4: Bat Speed, Bat-To-Ball, Swing Decisions, Ball Flight.

Grade columns are in place but BLANK until our own grade models exist.
When a model is ready, set its `stat` (or plug a model in) in GRADES below and
the column fills in on the Evaluation Method the user picks:
    20-80       50 = average, 10 points per standard deviation (NOT capped: a 90 month shows 90)
    Percentile  share of MLB hitters he's better than
    Plus        100 = average, 10 points per standard deviation
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
    stat: str | None      # input for the grade -- None = blank until our model is built
    desc: str


GRADES = {g.key: g for g in [
    Grade("BAT SPEED GRADE", "Bat Speed Grade", "_POWER_Z",
          "20-80 Grade, Bat speed + how hard he hits it\n[Bat speed 50% (avg 25, top 8th 25) | EV on fly balls 30% "
          "(avg 15, 90th 15) | 90th EV + HHEV 10% | avg EV + EV>95 6% | max EVs 4%]\nNo bat tracking: the bat "
          "speed half goes to 90th EV + HHEV. Regressed for sample size; hover a grade for each ingredient."),
    Grade("SC BAT SPEED GRADE", "SC Bat Speed Grade", "_SCBS_Z",
          "20-80 Grade, Statcast bat speed only\n[avg bat speed 50% + top 8th bat speed 50%, regressed]"),
    Grade("B2B GRADE", "Bat-To-Ball Grade", "_B2B_Z",
          "20-80 Grade, Bat-to-ball: contact WHEN HE SWINGS (takes never count)\n[Cont% over expected 45% | "
          "Z-Cont% over expected 20% | Cont% 15% | Heart Cont% 10% | Chase Cont% 10%]\nOver expected = each swing vs "
          "how often an average hitter makes contact on that pitch, so a whiff over the heart costs the most. "
          "Regressed for sample size; hover a grade for each ingredient."),
    Grade("DECISION GRADE", "Decision Grade", None, "20-80 Grade, Swing Decision\n[from rate of good decisions]"),
    Grade("0-0 DECISION GRADE", "0-0 Decision Grade", None, "20-80 Grade, Swing Decision on first pitch"),
    Grade("<2K DECISION GRADE", "<2k Decision Grade", None, "20-80 Grade, Swing Decision with less than two strikes"),
    Grade("2K DECISION GRADE", "2k Decision Grade", None, "20-80 Grade, Swing Decision with two strikes"),
    Grade("FB DECISION GRADE", "FB Decision Grade", None, "20-80 Grade, Swing Decision on fastballs"),
    Grade("BB DECISION GRADE", "BB Decision Grade", None, "20-80 Grade, Swing Decision on breaking balls"),
    Grade("OS DECISION GRADE", "OS Decision Grade", None, "20-80 Grade, Swing Decision on offspeed"),
    Grade("BALL FLIGHT GRADE", "Ball Flight Grade", None, "20-80 Grade, Ball Flight\n[from xwOBA on contact]"),
]}

# Columns every Core 4 tab starts with (after Year / Period / Age / Type / Team / vs)
SHARED = ["PA", "BIP", "K%", "BB%", "BARREL%", "wOBA", "xwOBA", "xwOBA MLB", "SAxwOBA"]
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
                                 "GD%", "DEC QUAL", "DEC RUNS", "DEC RUNS/100", "SWING%", "CHASE%", "PRE2K O-SWING%", "2K O-SWING%",
                                 "H-SWING%", "S-SWING%", "C-SWING%", "W-SWING%"],
    "Ball Flight": SHARED + ["BALL FLIGHT GRADE", "BALL FLIGHT 600", "LA", "LAHHB", "ATTACK", "xwOBACON",
                             "xwOBACON MLB", "SAxwOBACON", "WEAK GB%", "GB%", "FB%", "WEAK FB%", "LD%", "LD+FB%", "PULL%",
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
    return 50 + 10 * z              # no 20-80 cap on purpose: an extreme month should show as extreme


def fmt_grade(v: float) -> str:
    return "-" if v is None or math.isnan(v) else f"{v:.0f}"


def grade(key: str, vals, ref_row: dict, scale: str) -> tuple[float, float]:
    """(grade on the chosen scale, z)."""
    g = GRADES[key]
    if not g.stat:                    # model not built yet -> blank
        return math.nan, math.nan
    if g.stat.startswith("_"):        # model grades: the z comes with the row (models/power.py etc.)
        z = vals.get(g.stat, math.nan) if hasattr(vals, "get") else math.nan
        z = math.nan if z is None else float(z)
        return to_scale(z, scale), z
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
        d = GRADES[key].desc + ("" if GRADES[key].stat else "\n(blank until our grade model is built)")
        return d.replace("20-80 Grade", {"20-80": "20-80 Grade", "Percentile": "Percentile",
                                         "Plus": "Plus (100 = average)"}[scale])
    if key in DERIVED:
        return DERIVED[key][0]
    return hm.CATALOG[key].desc
