"""Hitting stat catalog: ONE place that defines every hitting stat on the site.

Each stat has a label, a hover definition, how it's calculated, how it's shown
and which direction is "good". Pages only list stat keys; the league reference
builder (pipeline/build_league_reference.py) runs the same catalog over every
MLB hitter, so player numbers, league averages and standard deviations always
come from identical formulas -- and from the same xwOBA model.

To change a definition: edit the stat here (and the mask in prep() if needed),
then rebuild the league reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.statcast_clean import clean
from models.decisions import decision_detail
from models.xwoba import expected_woba

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SD_MIN_PA = 100              # hitters with this many PA define the league spread (STDEVS)
QUALIFIED_PA_PER_GAME = 3.1  # MLB qualified hitter: 3.1 PA per team game (502 in a full season)
QUALIFIED_AVG = {"AVG BAT SPEED", "TOP 8TH BAT SPEED", "SWING LENGTH"}   # MLB average = qualified hitters (Statcast)
SPRAY_SCALE = 0.75           # Statcast hit coordinates -> spray angle (degrees)

SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
          "foul_bunt", "missed_bunt", "bunt_foul_tip", "foul_pitchout"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FOULS = {"foul", "foul_tip", "foul_bunt", "bunt_foul_tip", "foul_pitchout"}
FASTBALLS = {"FF", "SI", "FC", "FA"}
BREAKING = {"SL", "ST", "CU", "KC", "SV", "CS", "SC"}
OFFSPEED = {"CH", "FS", "FO", "KN", "EP"}

# Row groupings shown down the side of the tables
ZONE_GROUPINGS = {
    "In / Out of Zone": "zone_io",
    "Attack Zones": "attack",
}
SPLITS = {"Count Group": "count_grp", "Platoon": "platoon", "Pitch Group": "pitch_grp"}
ZONE_ORDER = {"zone_io": ["In Zone", "Out of Zone"], "attack": ["Heart", "Shadow", "Chase", "Waste"]}
SPLIT_ORDER = {"count_grp": ["First Pitch", "Ahead", "Even", "Behind", "Two Strikes"],
               "platoon": ["vs LHP", "vs RHP"], "pitch_grp": ["Fastball", "Breaking", "Offspeed"]}


# ----------------------------------------------------------------------------
# Catalog
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Stat:
    key: str
    label: str
    desc: str          # hover definition: "What it is\n[exact rule]"
    kind: str          # count | ratio | mean | max | q90 | top8
    a: str = ""        # numerator / column
    b: str = ""        # denominator / mask
    fmt: str = "pct"   # pct | rate3 | num1 | int
    good: int = 1      # +1 higher is better, -1 lower is better, 0 neutral (style, not skill)


S = Stat
CATALOG: dict[str, Stat] = {s.key: s for s in [
    # counts
    S("PIT", "PIT", "Pitches seen", "count", "pit", fmt="int", good=0),
    S("K%", "K%", "Strikeout rate\n[Strikeouts / PA]", "ratio", "k", "pa", good=-1),
    S("BB%", "BB%", "Walk rate\n[Walks (incl. intentional) / PA]", "ratio", "bb", "pa", good=1),
    S("SWG", "SWG", "Swings", "count", "swing", fmt="int", good=0),
    S("PA", "PA", "Plate appearances", "count", "pa", fmt="int", good=0),
    S("BIP", "BIP", "Balls in play", "count", "bip", fmt="int", good=0),
    S("TRK", "TRK", "Tracked balls in play\n[BIP with exit velocity and launch angle measured]",
      "count", "trk", fmt="int", good=0),
    # swing decisions
    S("SWING%", "SWING%", "Rate of pitches swung at\n[Swings / Pitches]", "ratio", "swing", "pit", good=0),
    S("Z-SWING%", "Z-SWING%", "Rate of in-zone pitches swung at\n[Swings in zone / Pitches in zone]",
      "ratio", "z_swing", "z_pit", good=1),
    S("O-SWING%", "O-SWING%", "Rate of out-of-zone pitches swung at (chase rate)\n[Swings out of zone / Pitches out of zone]",
      "ratio", "o_swing", "o_pit", good=-1),
    S("WHIFF%", "WHIFF%", "Rate of swings that miss\n[Whiffs / Swings]", "ratio", "whiff", "swing", good=-1),
    S("Z-WHIFF%", "Z-WHIFF%", "Rate of in-zone swings that miss\n[Whiffs in zone / Swings in zone]",
      "ratio", "z_whiff", "z_swing", good=-1),
    S("O-WHIFF%", "O-WHIFF%", "Rate of out-of-zone swings that miss\n[Whiffs out of zone / Swings out of zone]",
      "ratio", "o_whiff", "o_swing", good=-1),
    S("FOUL%", "FOUL%", "Rate of swings fouled off\n[Fouls / Swings]", "ratio", "foul", "swing", good=0),
    # outcomes
    S("xwOBA", "xwOBA", "Expected wOBA (Savant)\n[Exit velo + launch angle (+ sprint speed on some weak balls) on "
      "batted balls; actual value on BB / HBP / K. Same xwOBA whatever the direction]",
      "ratio", "xw", "wden", fmt="rate3", good=1),
    S("xwOBACON", "xwOBACON", "Expected wOBA on contact (Savant)\n[Expected value per ball in play]",
      "ratio", "xw_bip", "bip_w", fmt="rate3", good=1),
    S("SAxwOBA", "SAxwOBA", "Spray-adjusted expected wOBA (ours)\n[Exit velo + launch angle + spray angle on batted "
      "balls -- nothing else (no sprint speed, side, park or calibration); actual value on BB / HBP / K]",
      "ratio", "sxw", "sx_wden", fmt="rate3", good=1),
    S("SAxwOBACON", "SAxwOBACON", "Spray-adjusted expected wOBA on contact (ours)\n[Exit velo + launch angle + "
      "spray angle, per ball in play]", "ratio", "sxw_bip", "sx_bip", fmt="rate3", good=1),
    S("wOBACON", "wOBACON", "wOBA on contact\n[Actual wOBA value per ball in play]",
      "ratio", "wv_bip", "bip_w", fmt="rate3", good=1),
    S("wOBA", "wOBA", "Weighted on-base average\n[Actual wOBA value / wOBA denominator]",
      "ratio", "wv", "wden", fmt="rate3", good=1),
    # contact quality
    S("LA", "LA", "Average launch angle\n[Tracked BIP]", "mean", "la", fmt="num1", good=0),
    S("EV", "EV", "Average exit velocity\n[Tracked BIP]", "mean", "ev", fmt="num1", good=1),
    S("90th EV", "90th EV", "90th percentile exit velocity\n[Tracked BIP]", "q90", "ev", fmt="num1", good=1),
    S("HHEV", "HHEV", "Hard-hit exit velocity: average of the player's top 1/8 of exit velocities\n[Tracked BIP]",
      "top8", "ev", fmt="num1", good=1),
    S("MAX EV", "MAX EV", "Hardest-hit ball\n[Tracked BIP]", "max", "ev", fmt="num1", good=1),
    S("EV<80", "EV<80", "Rate of tracked BIP under 80 mph\n[ExitVelo < 80]", "ratio", "ev_lt80", "trk", good=-1),
    S("EV>90", "EV>90", "Rate of tracked BIP over 90 mph\n[ExitVelo > 90]", "ratio", "ev_gt90", "trk", good=1),
    S("EV>95", "EV>95", "Rate of tracked BIP over 95 mph\n[ExitVelo > 95]", "ratio", "ev_gt95", "trk", good=1),
    S("EV>100", "EV>100", "Rate of tracked BIP over 100 mph\n[ExitVelo > 100]", "ratio", "ev_gt100", "trk", good=1),
    S("EV>105", "EV>105", "Rate of tracked BIP over 105 mph\n[ExitVelo > 105]", "ratio", "ev_gt105", "trk", good=1),
    S("BARREL%", "BARREL%", "Rate of BIP where xBA ≥ .500 and xSLG ≥ 1.500\n(Statcast barrel flag if xBA / xSLG missing)",
      "ratio", "barrel", "trk", good=1),
    # batted-ball types (they add to 100%)
    S("WEAK GB%", "WEAK GB%", "Rate of weak ground balls\n[LA ≥ -90° and LA < -30°] or [ExitSpeed ≤ 50 and LA ≥ -30° and LA < 10°]",
      "ratio", "weak_gb", "trk", good=-1),
    S("GB%", "GB%", "Rate of ground balls\n[ExitVelo > 50 and LA ≥ -30° and LA < 10°]", "ratio", "gb", "trk", good=0),
    S("FB%", "FB%", "Rate of fly balls\n[ExitVelo > 70 and LA ≥ 10° and LA < 51°]", "ratio", "fb", "trk", good=0),
    S("WEAK FB%", "WEAK FB%", "Rate of weak fly balls\n[LA ≥ 51° and LA < 90°] or [ExitSpeed ≤ 70 and LA ≥ 10° and LA < 51°]",
      "ratio", "weak_fb", "trk", good=-1),
    S("LD%", "LD%", "Rate of BIP where 10° ≤ LA ≤ 30°", "ratio", "ld", "trk", good=0),
    S("LD+FB%", "LD+FB%", "Line Drive + Fly Ball Rate\n[BIP that are line drives or fly balls]", "ratio", "ldfb", "trk", good=0),
    S("LAHHB", "LAHHB", "Launch Angle Hard Hit Balls\n[Average LA on BIP ≥ 95 mph]", "mean", "la_hh", fmt="num1", good=0),
    S("ATTACK", "Attack ∠", "Attack Angle\n[Average bat attack angle at contact, Statcast bat tracking]",
      "mean", "attack_angle", fmt="num1", good=0),
    S("EV (FLY)", "EV (Fly)", "Average Exit Velocity on fly balls", "mean", "ev_fb", fmt="num1", good=1),
    S("90th EV (FLY)", "90th EV (Fly)", "90th Percentile Exit Velocity on fly balls", "q90", "ev_fb", fmt="num1", good=1),
    S("MAX EV (FLY)", "MAX EV (Fly)", "Max Fly Ball Exit Velocity", "max", "ev_fb", fmt="num1", good=1),
    # bat tracking (Statcast, 2023+)
    S("# BAT SPEEDS", "# Bat Speeds", "Number of swings with a Statcast bat speed reading", "count", "bs_n", fmt="int", good=0),
    S("AVG BAT SPEED", "Avg Bat Speed", "Average Statcast bat speed\n[Competitive swings: no bunts, each hitter's "
      "slowest 10% dropped -- Statcast's definition. MLB average = qualified hitters]", "mean", "bat_speed",
      fmt="num1", good=1),
    S("TOP 8TH BAT SPEED", "Top 8th Bat Speed", "Average of top 8th Statcast bat speeds", "top8", "bat_speed", fmt="num1", good=1),
    S("SWING LENGTH", "Swing Length", "Average swing length (feet), Statcast bat tracking",
      "mean", "swing_length", fmt="num1", good=0),
    # bat-to-ball
    S("CONT%", "Cont%", "Rate of contact (BIP/foul) given swing", "ratio", "contact", "swing", good=1),
    S("Z-CONT%", "Z-Cont%", "Rate of contact (BIP/foul) given swing on pitch in zone", "ratio", "z_contact", "z_swing", good=1),
    S("Z-CONT% (HARD)", "Z-Cont% (Hard)", "Rate of contact (BIP/foul) given swing on hard pitch in zone\n[Hard = fastballs: FF, SI, FC]",
      "ratio", "zh_contact", "zh_swing", good=1),
    S("Z-CONT% (SOFT)", "Z-Cont% (Soft)", "Rate of contact (BIP/foul) given swing on soft pitch in zone\n[Soft = breaking + offspeed]",
      "ratio", "zs_contact", "zs_swing", good=1),
    S("H-CONT%", "H-Cont%", "Rate of contact (BIP/foul) given swing on pitch in heart", "ratio", "h_contact", "h_swing", good=1),
    S("S-CONT%", "S-Cont%", "Rate of contact (BIP/foul) given swing on pitch in shadow", "ratio", "s_contact", "s_swing", good=1),
    S("C-CONT%", "C-Cont%", "Rate of contact (BIP/foul) given swing on pitch in chase", "ratio", "c_contact", "c_swing", good=1),
    S("W-CONT%", "W-Cont%", "Rate of contact (BIP/foul) given swing on pitch in waste", "ratio", "w_contact", "w_swing", good=1),
    # swing decisions (decision model in models/decisions.py)
    S("DECISIONS", "Decisions", "Number of pitches with a corresponding Decision Grade", "count", "dec", fmt="int", good=0),
    S("0-0 DECISIONS", "0-0 Decisions", "Number of first pitches with a corresponding Decision Grade",
      "count", "dec_00", fmt="int", good=0),
    S("<2K DECISIONS", "<2k Decisions", "Number of pitches in less than 2-strike counts with a corresponding Decision Grade",
      "count", "dec_lt2", fmt="int", good=0),
    S("2K DECISIONS", "2k Decisions", "Number of pitches in 2-strike counts with a corresponding Decision Grade",
      "count", "dec_2k", fmt="int", good=0),
    S("FB DECISIONS", "FB Decisions", "Number of fastballs with a corresponding Decision Grade", "count", "dec_fb", fmt="int", good=0),
    S("BB DECISIONS", "BB Decisions", "Number of breaking balls with a corresponding Decision Grade", "count", "dec_bb", fmt="int", good=0),
    S("OS DECISIONS", "OS Decisions", "Number of offspeed pitches with a corresponding Decision Grade", "count", "dec_os", fmt="int", good=0),
    S("GD%", "GD%", "Rate of good decisions: how OFTEN he made the right swing / take call\n"
      "[Good / (Good + Bad), coin-flip pitches left out; placeholder zone model until IQ Swing Decisions is on]",
      "ratio", "gd", "dec", good=1),
    S("DEC QUAL", "DEC QUAL", "Decision Quality: HOW right or wrong his swing / take calls were, not just which\n"
      "[average decision score per pitch: (chosen option - the other) / what was at stake in that count, x100; "
      "0 = coin flip, each pitch capped at +-150. Blank until IQ Swing Decisions is on]",
      "ratio", "dq_sum", "dq_n", fmt="num1", good=1),
    S("DEC RUNS", "DEC RUNS", "Decision Runs: runs gained (+) or lost (-) by his swing / take choices vs what a "
      "league-average hitter would choose on the same pitches\n[sum over graded pitches; not count-adjusted, so "
      "3-2 mistakes cost more. Blank until IQ Swing Decisions is on]", "count", "druns", "dq_n", fmt="num1", good=1),
    S("DEC RUNS/100", "DEC RUNS/100", "Decision Runs per 100 graded pitches\n[Decision Runs / graded pitches x 100]",
      "ratio", "druns100", "dq_n", fmt="num2", good=1),
    S("GD% 0-0", "GD% 0-0", "Rate of good decisions on first pitch", "ratio", "gd_00", "dec_00", good=1),
    S("GD% <2K", "GD% <2k", "Rate of good decisions with less than two strikes", "ratio", "gd_lt2", "dec_lt2", good=1),
    S("GD% 2K", "GD% 2k", "Rate of good decisions with two strikes", "ratio", "gd_2k", "dec_2k", good=1),
    S("GD% FB", "GD% FB", "Rate of good decisions on fastballs", "ratio", "gd_fb", "dec_fb", good=1),
    S("GD% BB", "GD% BB", "Rate of good decisions on breaking balls", "ratio", "gd_bb", "dec_bb", good=1),
    S("GD% OS", "GD% OS", "Rate of good decisions on offspeed", "ratio", "gd_os", "dec_os", good=1),
    S("CHASE%", "Chase%", "Rate of batter swing given out-of-zone pitch", "ratio", "o_swing", "o_pit", good=-1),
    S("PRE2K O-SWING%", "Pre2k O-Swing%", "Rate of chase (before 2 strikes)", "ratio", "o_swing_lt2", "o_pit_lt2", good=-1),
    S("2K O-SWING%", "2k O-Swing%", "Rate of chase (with 2 strikes)", "ratio", "o_swing_2k", "o_pit_2k", good=-1),
    S("H-SWING%", "H-Swing%", "Rate of swing per heart pitch", "ratio", "h_swing", "h_pit", good=1),
    S("S-SWING%", "S-Swing%", "Rate of swing per shadow pitch", "ratio", "s_swing", "s_pit", good=0),
    S("C-SWING%", "C-Swing%", "Rate of swing per chase pitch", "ratio", "c_swing", "c_pit", good=-1),
    S("W-SWING%", "W-Swing%", "Rate of swing per waste pitch", "ratio", "w_swing", "w_pit", good=-1),
    # direction
    S("PULL%", "PULL%", "Rate of BIP to pull side\n[RHB & SprayAngle ≤ -15° or LHB & SprayAngle ≥ 15°]",
      "ratio", "pull", "spr", good=0),
    S("MIDDLE%", "MIDDLE%", "Rate of BIP to middle field\n[SprayAngle > -15° and SprayAngle < 15°]",
      "ratio", "mid", "spr", good=0),
    S("OPPO%", "OPPO%", "Rate of BIP to opposite field\n[RHB & SprayAngle ≥ 15° or LHB & SprayAngle ≤ -15°]",
      "ratio", "oppo", "spr", good=0),
    S("AIRPULL%", "AIRPULL%", "Rate of all BIP that are pulled in the air above 80 EV\n"
      "[EV > 80, 10° ≤ LA ≤ 45°, RHB & SprayAngle ≤ -10° or LHB & SprayAngle ≥ 10°]", "ratio", "airpull", "spr", good=1),
    S("FB-PULL%", "FB-PULL%", "Rate of fly balls that are pulled\n(Formerly known as Airpull%)\n"
      "[RHB & SprayAngle ≤ -15° or LHB & SprayAngle ≥ 15°]", "ratio", "fb_pull", "fb_spr", good=1),
    S("PULL-AIR%", "PULL-AIR%", "Rate of pulled balls that are fly balls\n[ExitVelo > 70 and LA ≥ 10° and LA < 51°]",
      "ratio", "fb_pull", "pull", good=1),
]}

# What each tab shows. Overview is two stacked lines per group, like the reference.
OVERVIEW = [
    ("COUNTS", ["PIT", "SWG"], ["BIP", "TRK"]),
    ("RATES", ["SWING%", "Z-SWING%", "O-SWING%"], ["WHIFF%", "Z-WHIFF%", "O-WHIFF%"]),
    ("CONTACT", ["LA", "EV", "90th EV", "HHEV", "MAX EV"], ["EV<80", "EV>95", "EV>100", "EV>105", "BARREL%"]),
    ("BIP", ["WEAK GB%", "GB%", "FB%", "WEAK FB%"], ["PULL%", "MIDDLE%", "OPPO%", "AIRPULL%"]),
]
TABS = {
    "Rates": ["PIT", "BIP", "SWING%", "WHIFF%", "FOUL%", "PULL%", "MIDDLE%", "OPPO%",
              "AIRPULL%", "FB-PULL%", "PULL-AIR%", "BARREL%"],
    "Contact": ["PIT", "BIP", "xwOBA", "SAxwOBA", "xwOBACON", "SAxwOBACON", "WEAK GB%", "GB%", "FB%", "WEAK FB%"],
    "Exit": ["PIT", "BIP", "LA", "EV", "90th EV", "MAX EV", "EV<80", "EV>90", "EV>95", "EV>100", "EV>105"],
}
SAMPLE = ["PA", "BIP"]          # sample columns in the click-popup
ALL_KEYS = list(CATALOG)


# ----------------------------------------------------------------------------
# Pitch-level flags
# ----------------------------------------------------------------------------
def _col(df, name, default=np.nan):
    return df[name] if name in df else pd.Series(default, index=df.index)


def spray_angle(df: pd.DataFrame) -> pd.Series:
    x, y = _col(df, "hc_x"), _col(df, "hc_y")
    return np.degrees(np.arctan((x - 125.42) / (198.27 - y))) * SPRAY_SCALE


def attack_zone(df: pd.DataFrame) -> pd.Series:
    """Heart / Shadow / Chase / Waste from pitch location relative to the batter's zone
    (in units of half the zone: Heart < 0.67, Shadow < 1.33, Chase < 2, else Waste)."""
    px, pz = _col(df, "plate_x"), _col(df, "plate_z")
    top, bot = _col(df, "sz_top"), _col(df, "sz_bot")
    nx = px.abs() / (17 / 24)
    nz = (pz - (top + bot) / 2).abs() / ((top - bot) / 2)
    r = np.maximum(nx, nz)
    out = pd.Series(np.select([r < 0.67, r < 1.33, r < 2.0], ["Heart", "Shadow", "Chase"], "Waste"), index=df.index)
    return out.where(r.notna())


def count_group(df: pd.DataFrame) -> pd.Series:
    b, s = _col(df, "balls", 0), _col(df, "strikes", 0)
    return pd.Series(np.select([s >= 2, (b == 0) & (s == 0), b > s, b == s],
                               ["Two Strikes", "First Pitch", "Ahead", "Even"], "Behind"), index=df.index)


def pitch_group(df: pd.DataFrame) -> pd.Series:
    pt = _col(df, "pitch_type", "")
    return pd.Series(np.select([pt.isin(FASTBALLS), pt.isin(BREAKING), pt.isin(OFFSPEED)],
                               ["Fastball", "Breaking", "Offspeed"], None), index=df.index)


def prep(df: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """Add one flag/value column per ingredient the catalog needs."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = clean(df)
    d = pd.DataFrame(index=df.index)
    for keep in ("batter", "game_date", "game_pk", "game_type", "stand", "p_throws"):
        if keep in df:
            d[keep] = df[keep]
    desc = _col(df, "description", "")
    ev_ = _col(df, "events")
    zone = _col(df, "zone")
    is_z, is_o = zone.between(1, 9), zone.between(11, 14)

    d["pit"] = 1
    d["swing"] = desc.isin(SWINGS)
    d["whiff"] = desc.isin(WHIFFS)
    d["foul"] = desc.isin(FOULS)
    d["z_pit"], d["o_pit"] = is_z, is_o
    d["z_swing"], d["o_swing"] = d["swing"] & is_z, d["swing"] & is_o
    d["z_whiff"], d["o_whiff"] = d["whiff"] & is_z, d["whiff"] & is_o
    d["pa"] = ev_.notna() & (ev_ != "truncated_pa")

    bip = _col(df, "type", "").eq("X")
    ev, la = _col(df, "launch_speed"), _col(df, "launch_angle")
    trk = bip & ev.notna() & la.notna()
    d["bip"], d["trk"] = bip, trk
    d["ev"] = ev.where(trk)
    d["la"] = la.where(trk)
    for t, col in ((80, "ev_lt80"),):
        d[col] = trk & (ev < t)
    for t in (90, 95, 100, 105):
        d[f"ev_gt{t}"] = trk & (ev > t)
    xba, xslg = _col(df, "estimated_ba_using_speedangle"), _col(df, "estimated_slg_using_speedangle")
    has_x = xba.notna() & xslg.notna()
    d["barrel"] = trk & np.where(has_x, (xba >= 0.5) & (xslg >= 1.5), _col(df, "launch_speed_angle").eq(6))
    d["ld"] = trk & (la >= 10) & (la <= 30)

    d["weak_gb"] = trk & (((la >= -90) & (la < -30)) | ((ev <= 50) & (la >= -30) & (la < 10)))
    d["gb"] = trk & (ev > 50) & (la >= -30) & (la < 10)
    d["fb"] = trk & (ev > 70) & (la >= 10) & (la < 51)
    d["weak_fb"] = trk & (((la >= 51) & (la < 90)) | ((ev <= 70) & (la >= 10) & (la < 51)))

    spray = spray_angle(df)
    rhb = _col(df, "stand", "").eq("R")
    spr = trk & spray.notna()
    d["spr"] = spr
    d["pull"] = spr & np.where(rhb, spray <= -15, spray >= 15)
    d["oppo"] = spr & np.where(rhb, spray >= 15, spray <= -15)
    d["mid"] = spr & (spray > -15) & (spray < 15)
    d["airpull"] = spr & (ev > 80) & (la >= 10) & (la <= 45) & np.where(rhb, spray <= -10, spray >= 10)
    d["ldfb"] = d["ld"] | d["fb"]
    d["la_hh"] = la.where(trk & (ev >= 95))
    d["ev_fb"] = ev.where(d["fb"])
    d["fb_spr"] = d["fb"] & spr
    d["fb_pull"] = d["fb"] & d["pull"]

    wden = _col(df, "woba_denom", 0).fillna(0)
    xw = expected_woba(df, model)
    d["wden"] = wden
    d["wv"] = _col(df, "woba_value", 0).fillna(0).where(wden > 0, 0)
    d["xw"] = xw.fillna(0).where(wden > 0, 0)
    d["bip_w"] = bip & (wden > 0)
    d["xw_bip"] = xw.fillna(0).where(d["bip_w"], 0)
    from models import saxwoba as _sa                      # ours, shown beside Savant's
    sxw = _sa.expected_woba(df)
    on = sxw.notna()                                        # blank (not .000) until SAxwOBA is trained
    d["sxw"] = sxw.fillna(0).where(wden > 0, 0)
    d["sxw_bip"] = sxw.fillna(0).where(d["bip_w"], 0)
    d["sx_wden"] = pd.Series(wden, index=df.index).where(on, 0)
    d["sx_bip"] = d["bip_w"] & on
    d["wv_bip"] = d["wv"].where(d["bip_w"], 0)

    # strikeouts / walks
    d["k"] = ev_.isin(["strikeout", "strikeout_double_play"])
    d["bb"] = ev_.isin(["walk", "intent_walk"])

    # bat tracking (Statcast 2023+) -- Statcast's COMPETITIVE swings: no bunts, and each hitter's slowest 10% of
    # swings dropped (check / defensive swings), except any 60+ mph swing hit 90+ mph
    bs = _col(df, "bat_speed")
    bunt = desc.astype(str).str.contains("bunt") | ev_.astype(str).str.contains("bunt") | \
        _col(df, "des", "").astype(str).str.contains("bunt")
    tracked = bs.where(d["swing"] & bs.notna() & ~bunt)
    who = d["batter"] if "batter" in d else pd.Series(0, index=d.index)
    g = tracked.groupby(who)
    slow = g.rank(method="first") <= np.floor(g.transform("count") * 0.10)        # exactly the slowest tenth
    keep = tracked.notna() & (~slow | ((tracked >= 60) & (_col(df, "launch_speed") >= 90)))
    d["bat_speed"] = tracked.where(keep)
    d["bs_n"] = d["bat_speed"].notna()
    d["swing_length"] = _col(df, "swing_length").where(d["bs_n"])
    d["attack_angle"] = _col(df, "attack_angle").where(d["swing"])

    # contact by zone / pitch hardness / attack zone
    contact = d["swing"] & ~d["whiff"]
    att = attack_zone(df)
    hard = _col(df, "pitch_type", "").isin(FASTBALLS)
    d["contact"] = contact
    d["z_contact"] = contact & is_z
    d["zh_swing"], d["zh_contact"] = d["swing"] & is_z & hard, contact & is_z & hard
    d["zs_swing"], d["zs_contact"] = d["swing"] & is_z & ~hard, contact & is_z & ~hard
    for code, name in (("h", "Heart"), ("s", "Shadow"), ("c", "Chase"), ("w", "Waste")):
        z_ = att.eq(name)
        d[f"{code}_pit"], d[f"{code}_swing"], d[f"{code}_contact"] = z_, d["swing"] & z_, contact & z_

    # chase by count
    strikes = _col(df, "strikes", 0).fillna(0)
    balls = _col(df, "balls", 0).fillna(0)
    two_k = strikes >= 2
    d["o_pit_lt2"], d["o_swing_lt2"] = is_o & ~two_k, d["o_swing"] & ~two_k
    d["o_pit_2k"], d["o_swing_2k"] = is_o & two_k, d["o_swing"] & two_k

    # swing decisions (model in models/decisions.py)
    detail = decision_detail(df, d["swing"], att)
    good = detail["good"]
    dec = good.notna()
    q = detail["quality"]
    d["dq_n"] = q.notna()                                  # every graded decision, neutral included
    d["dq_sum"] = q.fillna(0)
    d["druns"] = detail["runs"].fillna(0)
    d["druns100"] = 100 * d["druns"]
    g1 = good.fillna(0).astype(float)
    pg = pitch_group(df)
    first = (balls == 0) & (strikes == 0)
    for suffix, mask in (("", dec), ("_00", dec & first), ("_lt2", dec & ~two_k), ("_2k", dec & two_k),
                         ("_fb", dec & pg.eq("Fastball")), ("_bb", dec & pg.eq("Breaking")),
                         ("_os", dec & pg.eq("Offspeed"))):
        d[f"dec{suffix}"] = mask
        d[f"gd{suffix}"] = g1.where(mask, 0)

    # batting team (for Split By Team)
    top = _col(df, "inning_topbot", "").eq("Top")
    d["bat_team"] = _col(df, "away_team", "").where(top, _col(df, "home_team", ""))

    # groupings
    d["zone_io"] = pd.Series(np.select([is_z, is_o], ["In Zone", "Out of Zone"], None), index=df.index)
    d["attack"] = att
    d["count_grp"] = count_group(df)
    d["platoon"] = _col(df, "p_throws", "").map({"L": "vs LHP", "R": "vs RHP"})
    d["pitch_grp"] = pitch_group(df)
    return d


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def _needs():
    sums, shapes = set(), set()
    for st_ in CATALOG.values():
        if st_.kind == "count":
            sums.add(st_.a)
            if st_.b:
                sums.add(st_.b)
        elif st_.kind == "ratio":
            sums.update((st_.a, st_.b))
        else:
            shapes.add((st_.kind, st_.a))
    return sorted(sums), sorted(shapes)


SUM_COLS, SHAPES = _needs()


def _top8(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    return s.nlargest(max(1, math.ceil(len(s) / 8))).mean()


def _shape(x, kind, grouped: bool):
    if kind == "mean":
        return x.mean()
    if kind == "max":
        return x.max()
    if kind == "q90":
        return x.quantile(0.9)
    return x.apply(_top8) if grouped else _top8(x)


def compute(d: pd.DataFrame, by: list[str] | None = None, keys: list[str] | None = None) -> pd.DataFrame:
    """Stat values (raw numbers) per group. by=None -> one row, index 'All'."""
    keys = keys or ALL_KEYS
    if d is None or d.empty:
        return pd.DataFrame(columns=keys)
    missing = {c: 0 for c in SUM_COLS if c not in d} | {c: np.nan for _, c in SHAPES if c not in d}
    if missing:
        d = d.assign(**missing)
    if by:
        g = d.groupby(by, dropna=True, observed=True)
        sums = g[SUM_COLS].sum()
        extra = pd.DataFrame({f"{k}:{c}": _shape(g[c], k, True) for k, c in SHAPES})
    else:
        sums = d[SUM_COLS].sum().to_frame("All").T
        extra = pd.DataFrame({f"{k}:{c}": [_shape(d[c], k, False)] for k, c in SHAPES}, index=["All"])
    sums = sums.astype(float)
    out = pd.DataFrame(index=sums.index)
    for k in keys:
        s = CATALOG[k]
        if s.kind == "count":
            out[k] = sums[s.a].where(sums[s.b] > 0) if s.b else sums[s.a]     # b = "only if any of these"
        elif s.kind == "ratio":
            den = sums[s.b]
            out[k] = (sums[s.a] / den).where(den > 0)
        else:
            out[k] = extra[f"{s.kind}:{s.a}"]
    return out


def fmt(key: str, v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    f = CATALOG[key].fmt
    if f == "pct":
        return f"{100 * v:.1f}%"
    if f == "rate3":
        return f"{v:.3f}".replace("0.", ".", 1) if abs(v) < 1 else f"{v:.3f}"
    if f == "int":
        n = int(round(v))
        return f"{n / 1000:.1f}K" if n >= 100_000 else f"{n}"
    if f == "num2":
        return f"{v:.2f}"
    return f"{v:.1f}"


def fmt_delta(key: str, v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    f = CATALOG[key].fmt
    if f == "pct":
        return f"{100 * v:+.1f}%"
    if f == "rate3":
        return f"{v:+.3f}".replace("0.", ".", 1)
    if f == "num2":
        return f"{v:+.2f}"
    return f"{v:+.1f}"


# ----------------------------------------------------------------------------
# Rows for a player table: zone rows x optional splits
# ----------------------------------------------------------------------------
def player_rows(d: pd.DataFrame, zone_dim: str | None, split_dims: list[str]) -> list[tuple[dict, pd.Series]]:
    """[(labels, values)] in display order. labels = {"ZONE": "In Zone", "Platoon": "vs LHP", ...}."""
    out: list[tuple[dict, pd.Series]] = []
    if d is None or d.empty:
        return out
    zones = ["All"] + (ZONE_ORDER[zone_dim] if zone_dim else [])
    split_dims = [s for s in split_dims if s]
    combos = [{}]
    for sd in split_dims:
        combos = [dict(c, **{sd: v}) for c in combos for v in SPLIT_ORDER[sd]]
    for z in zones:
        dz = d if z == "All" else d[d[zone_dim] == z]
        for c in combos:
            dd = dz
            for sd, v in c.items():
                dd = dd[dd[sd] == v]
            if dd.empty:
                continue
            labels = {"ZONE": z, **c}
            out.append((labels, compute(dd).iloc[0]))
    return out


def ref_key(labels: dict, zone_dim: str | None) -> tuple[str, str]:
    """League-reference lookup key for a row: (dims, values)."""
    dims, vals = [], []
    if labels.get("ZONE", "All") != "All" and zone_dim:
        dims.append(zone_dim)
        vals.append(labels["ZONE"])
    for k, v in labels.items():
        if k != "ZONE":
            dims.append(k)
            vals.append(v)
    return ("+".join(dims) or "all", "|".join(vals) or "All")
