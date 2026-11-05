"""Hitting numbers for the Player Profile.

Two sources:
  * MLB stat lines (G, PA, AVG, OPS ...) from MLB's API -> stat_line(), rates()
  * Pitch-level Statcast from pybaseball.statcast_batter -> contact(), exit_velo(),
    spray(), bat_speed(), core4(), rolling()
All functions return {label: value} dicts (values already formatted) so pages
can show them as tables.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
          "foul_bunt", "missed_bunt", "bunt_foul_tip", "foul_pitchout"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CALLED = {"called_strike"}
FAST_SWING = 75.0


def _pct(n, d) -> str:
    return f"{100 * n / d:.1f}" if d else "-"


def _num(x, nd=1) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "-"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def _rate3(x) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}".replace("0.", ".", 1)


# ----------------------------------------------------------------------------
# MLB stat lines
# ----------------------------------------------------------------------------
LINE = [("G", "gamesPlayed"), ("PA", "plateAppearances"), ("AB", "atBats"), ("R", "runs"),
        ("H", "hits"), ("2B", "doubles"), ("3B", "triples"), ("HR", "homeRuns"), ("RBI", "rbi"),
        ("BB", "baseOnBalls"), ("SO", "strikeOuts"), ("SB", "stolenBases"), ("CS", "caughtStealing")]


def _slash(s: dict) -> tuple[float, float, float]:
    ab, h, bb = s.get("atBats", 0), s.get("hits", 0), s.get("baseOnBalls", 0)
    hbp, sf = s.get("hitByPitch", 0), s.get("sacFlies", 0)
    tb = s.get("totalBases", h + s.get("doubles", 0) + 2 * s.get("triples", 0) + 3 * s.get("homeRuns", 0))
    avg = h / ab if ab else float("nan")
    obp_d = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_d if obp_d else float("nan")
    slg = tb / ab if ab else float("nan")
    return avg, obp, slg


def stat_line(s: dict) -> dict:
    out = {k: s.get(key, 0) for k, key in LINE}
    avg, obp, slg = _slash(s)
    out.update({"AVG": _rate3(avg), "OBP": _rate3(obp), "SLG": _rate3(slg), "OPS": _rate3(obp + slg)})
    return out


def rates(s: dict) -> dict:
    pa, ab = s.get("plateAppearances", 0), s.get("atBats", 0)
    so, bb, hr = s.get("strikeOuts", 0), s.get("baseOnBalls", 0), s.get("homeRuns", 0)
    h = s.get("hits", 0)
    xbh = s.get("doubles", 0) + s.get("triples", 0) + hr
    sb, cs = s.get("stolenBases", 0), s.get("caughtStealing", 0)
    sf = s.get("sacFlies", 0)
    avg, _, slg = _slash(s)
    babip_d = ab - so - hr + sf
    return {
        "PA": pa, "K%": _pct(so, pa), "BB%": _pct(bb, pa),
        "BB/K": _num(bb / so, 2) if so else "-",
        "ISO": _rate3(slg - avg), "BABIP": _rate3((h - hr) / babip_d) if babip_d else "-",
        "HR%": _pct(hr, pa), "XBH%": _pct(xbh, pa), "SB%": _pct(sb, sb + cs),
    }


# ----------------------------------------------------------------------------
# Statcast (one row per pitch)
# ----------------------------------------------------------------------------
def _cols(df, *names) -> bool:
    return all(c in df.columns for c in names)


def contact(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    swing = df["description"].isin(SWINGS)
    whiff = df["description"].isin(WHIFFS)
    zone = df["zone"].between(1, 9)
    pa = df["events"].notna().sum()
    k = df["events"].isin({"strikeout", "strikeout_double_play"}).sum()
    return {
        "Pitches": len(df),
        "Swing%": _pct(swing.sum(), len(df)),
        "Z-Swing%": _pct((swing & zone).sum(), zone.sum()),
        "Chase%": _pct((swing & ~zone).sum(), (~zone).sum()),
        "Contact%": _pct((swing & ~whiff).sum(), swing.sum()),
        "Z-Contact%": _pct((swing & ~whiff & zone).sum(), (swing & zone).sum()),
        "O-Contact%": _pct((swing & ~whiff & ~zone).sum(), (swing & ~zone).sum()),
        "Whiff%": _pct(whiff.sum(), swing.sum()),
        "CSW%": _pct((whiff | df["description"].isin(CALLED)).sum(), len(df)),
        "K%": _pct(k, pa),
    }


def _bip(df):
    return df[df["type"].eq("X")]


def _xwoba(df) -> tuple[float, float]:
    """(wOBA, xwOBA) the way Savant does it: xwOBA uses expected values on
    batted balls and actual values for walks / HBP / strikeouts."""
    d = df[df["woba_denom"].fillna(0) > 0] if "woba_denom" in df else df.iloc[0:0]
    if d.empty:
        return float("nan"), float("nan")
    woba = d["woba_value"].fillna(0).sum() / d["woba_denom"].sum()
    est = d["estimated_woba_using_speedangle"] if "estimated_woba_using_speedangle" in d else d["woba_value"]
    x = np.where(d["type"].eq("X") & est.notna(), est, d["woba_value"].fillna(0)).sum() / d["woba_denom"].sum()
    return woba, x


def exit_velo(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    bip = _bip(df)
    ev, la = bip["launch_speed"].dropna(), bip["launch_angle"].dropna()
    brl = (bip["launch_speed_angle"] == 6).sum() if "launch_speed_angle" in bip else 0
    woba, xwoba = _xwoba(df)
    return {
        "BBE": len(ev),
        "Avg EV": _num(ev.mean()), "Max EV": _num(ev.max()), "EV90": _num(ev.quantile(.9)) if len(ev) else "-",
        "HardHit%": _pct((ev >= 95).sum(), len(ev)), "Barrel%": _pct(brl, len(ev)),
        "Avg LA": _num(la.mean()), "Sweet Spot%": _pct(la.between(8, 32).sum(), len(la)),
        "wOBA": _rate3(woba), "xwOBA": _rate3(xwoba),
    }


def spray(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    bip = _bip(df)
    out = {"BIP": len(bip)}
    if _cols(bip, "hc_x", "hc_y", "stand"):
        b = bip.dropna(subset=["hc_x", "hc_y"])
        ang = np.degrees(np.arctan((b["hc_x"] - 125.42) / (198.27 - b["hc_y"])))
        pull = np.where(b["stand"].eq("R"), ang < -15, ang > 15)
        oppo = np.where(b["stand"].eq("R"), ang > 15, ang < -15)
        n = len(b)
        air = b["bb_type"].isin({"line_drive", "fly_ball"}) if "bb_type" in b else pd.Series(False, b.index)
        out.update({"Pull%": _pct(pull.sum(), n), "Cent%": _pct(n - pull.sum() - oppo.sum(), n),
                    "Oppo%": _pct(oppo.sum(), n), "Pull Air%": _pct((pull & air.to_numpy()).sum(), n)})
    if "bb_type" in bip:
        t = bip["bb_type"]
        n = t.notna().sum()
        out.update({"GB%": _pct(t.eq("ground_ball").sum(), n), "LD%": _pct(t.eq("line_drive").sum(), n),
                    "FB%": _pct(t.eq("fly_ball").sum(), n), "PU%": _pct(t.eq("popup").sum(), n)})
    return out


def bat_speed(df: pd.DataFrame) -> dict:
    """Bat tracking (2024+). Earlier seasons return dashes."""
    if df.empty or "bat_speed" not in df:
        return {"Bat Speed": "-", "Fast Swing%": "-", "Swing Length": "-"}
    bs = df["bat_speed"].dropna()
    sl = df["swing_length"].dropna() if "swing_length" in df else pd.Series(dtype=float)
    return {"Swings Tracked": len(bs), "Bat Speed": _num(bs.mean()),
            "Fast Swing%": _pct((bs >= FAST_SWING).sum(), len(bs)), "Swing Length": _num(sl.mean())}


def core4(df: pd.DataFrame) -> dict[str, dict]:
    c, e, s, b = contact(df), exit_velo(df), spray(df), bat_speed(df)
    g = lambda d, *ks: {k: d.get(k, "-") for k in ks}
    groups = {
        "Bat Speed": g(b, "Bat Speed", "Fast Swing%", "Swing Length"),
        "Bat-To-Ball": g(c, "Contact%", "Z-Contact%", "O-Contact%", "Whiff%"),
        "Swing Decisions": g(c, "Swing%", "Z-Swing%", "Chase%", "CSW%"),
        "Ball Flight": {**g(e, "Avg EV", "Avg LA", "Sweet Spot%", "Barrel%", "HardHit%"), **g(s, "Pull Air%")},
    }
    groups["Core 4"] = {"Bat Speed": b.get("Bat Speed", "-"), "Contact%": c.get("Contact%", "-"),
                        "Chase%": c.get("Chase%", "-"), "Barrel%": e.get("Barrel%", "-")}
    return groups


def rolling(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Rolling wOBA / xwOBA over the last `window` plate appearances."""
    if df.empty or "woba_denom" not in df:
        return pd.DataFrame()
    pa = df[df["woba_denom"].fillna(0) > 0].sort_values(["game_date", "at_bat_number"])
    if len(pa) < 5:
        return pd.DataFrame()
    est = pa.get("estimated_woba_using_speedangle", pa["woba_value"])
    x = np.where(pa["type"].eq("X") & est.notna(), est, pa["woba_value"].fillna(0))
    w = min(window, len(pa))
    out = pd.DataFrame({
        "PA": range(1, len(pa) + 1),
        "wOBA": pa["woba_value"].fillna(0).rolling(w).sum() / pa["woba_denom"].rolling(w).sum(),
        "xwOBA": pd.Series(x, index=pa.index).rolling(w).sum() / pa["woba_denom"].rolling(w).sum(),
    })
    return out.dropna().set_index("PA")


def by_platoon(df: pd.DataFrame, fn) -> dict[str, dict]:
    """Run a metric function for All / vs LHP / vs RHP."""
    out = {"All": fn(df)}
    if "p_throws" in df:
        out["vs LHP"] = fn(df[df["p_throws"].eq("L")])
        out["vs RHP"] = fn(df[df["p_throws"].eq("R")])
    return out
