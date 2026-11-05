"""Pitching numbers for the Player Profile, from pybaseball.statcast_pitcher
(one row per pitch thrown). Values come back formatted, ready for tables."""
from __future__ import annotations

import pandas as pd

from core.hitting import CALLED, SWINGS, WHIFFS, _num, _pct, _rate3, _xwoba

HITS = {"single", "double", "triple", "home_run"}
NOT_AB = {"walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play",
          "sac_bunt_double_play", "catcher_interf", "truncated_pa"}
K = {"strikeout", "strikeout_double_play"}
BB = {"walk", "intent_walk"}

LINE = [("G", "gamesPlayed"), ("GS", "gamesStarted"), ("W", "wins"), ("L", "losses"), ("SV", "saves"),
        ("IP", "inningsPitched"), ("H", "hits"), ("R", "runs"), ("ER", "earnedRuns"), ("HR", "homeRuns"),
        ("BB", "baseOnBalls"), ("SO", "strikeOuts"), ("ERA", "era"), ("WHIP", "whip"), ("AVG", "avg")]


def stat_line(s: dict) -> dict:
    """MLB pitching line from the stats API."""
    return {k: s.get(key, "-") for k, key in LINE}


def _ip_outs(ip) -> int:
    """'123.2' -> 371 outs (MLB writes thirds after the dot)."""
    try:
        whole, _, part = str(ip).partition(".")
        return int(whole) * 3 + int(part or 0)
    except ValueError:
        return 0


def league_rates(tot: dict) -> dict:
    """ERA / WHIP / K% / BB% for an MLB-average row (league totals from the API)."""
    outs = sum(_ip_outs(v) for k, v in tot.items() if k == "inningsPitched") or tot.get("outs", 0)
    ip = outs / 3 if outs else 0
    bf = tot.get("battersFaced", 0)
    return {"ERA": _num(9 * tot.get("earnedRuns", 0) / ip, 2) if ip else "-",
            "WHIP": _num((tot.get("hits", 0) + tot.get("baseOnBalls", 0)) / ip, 2) if ip else "-",
            "K%": _pct(tot.get("strikeOuts", 0), bf), "BB%": _pct(tot.get("baseOnBalls", 0), bf)}


def _swing_masks(df):
    swing = df["description"].isin(SWINGS)
    whiff = df["description"].isin(WHIFFS)
    zone = df["zone"].between(1, 9) if "zone" in df else pd.Series(False, df.index)
    return swing, whiff, zone


def overview(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    ev = df["events"].dropna()
    pa = len(ev[ev != "truncated_pa"])
    swing, whiff, zone = _swing_masks(df)
    bip = df[df["type"].eq("X")]
    la = bip["launch_speed"].dropna()
    brl = (bip["launch_speed_angle"] == 6).sum() if "launch_speed_angle" in bip else 0
    gb = bip["bb_type"].eq("ground_ball").sum() if "bb_type" in bip else 0
    woba, xwoba = _xwoba(df)
    ff = df[df["pitch_type"].isin({"FF", "SI"})]["release_speed"].dropna() if "pitch_type" in df else pd.Series(dtype=float)
    return {
        "Pitches": len(df), "BF": pa,
        "K%": _pct(ev.isin(K).sum(), pa), "BB%": _pct(ev.isin(BB).sum(), pa),
        "Whiff%": _pct(whiff.sum(), swing.sum()),
        "CSW%": _pct((whiff | df["description"].isin(CALLED)).sum(), len(df)),
        "Zone%": _pct(zone.sum(), len(df)),
        "Chase%": _pct((swing & ~zone).sum(), (~zone).sum()),
        "FB Velo": _num(ff.mean()),
        "HardHit%": _pct((la >= 95).sum(), len(la)), "Barrel%": _pct(brl, len(la)),
        "GB%": _pct(gb, len(bip)),
        "wOBA": _rate3(woba), "xwOBA": _rate3(xwoba),
    }


def _by_pitch(df: pd.DataFrame):
    col = "pitch_name" if "pitch_name" in df else "pitch_type"
    d = df[df[col].notna()]
    return d.groupby(col), len(d)


def arsenal(df: pd.DataFrame) -> dict[str, dict]:
    """One row per pitch: usage, velo, spin, movement, whiff, CSW."""
    if df.empty or "pitch_type" not in df:
        return {}
    groups, total = _by_pitch(df)
    rows = {}
    for name, g in sorted(groups, key=lambda kv: -len(kv[1])):
        swing, whiff, zone = _swing_masks(g)
        rows[name] = {
            "#": len(g), "Usage%": _pct(len(g), total),
            "Velo": _num(g["release_speed"].mean()), "Max": _num(g["release_speed"].max()),
            "Spin": _num(g["release_spin_rate"].mean(), 0) if "release_spin_rate" in g else "-",
            "IVB": _num(12 * g["pfx_z"].mean()) if "pfx_z" in g else "-",
            "HB": _num(12 * g["pfx_x"].mean()) if "pfx_x" in g else "-",
            "Ext": _num(g["release_extension"].mean()) if "release_extension" in g else "-",
            "Zone%": _pct(zone.sum(), len(g)),
            "Whiff%": _pct(whiff.sum(), swing.sum()),
            "CSW%": _pct((whiff | g["description"].isin(CALLED)).sum(), len(g)),
        }
    return rows


def results(df: pd.DataFrame) -> dict[str, dict]:
    """One row per pitch: what hitters did against it."""
    if df.empty or "pitch_type" not in df:
        return {}
    groups, _ = _by_pitch(df)
    rows = {}
    for name, g in sorted(groups, key=lambda kv: -len(kv[1])):
        ev = g["events"].dropna()
        ab = ev[~ev.isin(NOT_AB)]
        hits = ev[ev.isin(HITS)]
        tb = (ev == "single").sum() + 2 * (ev == "double").sum() + 3 * (ev == "triple").sum() + 4 * (ev == "home_run").sum()
        bip = g[g["type"].eq("X")]
        la = bip["launch_speed"].dropna()
        woba, xwoba = _xwoba(g)
        rows[name] = {
            "PA": len(ev[ev != "truncated_pa"]), "AVG": _rate3(len(hits) / len(ab)) if len(ab) else "-",
            "SLG": _rate3(tb / len(ab)) if len(ab) else "-",
            "wOBA": _rate3(woba), "xwOBA": _rate3(xwoba),
            "K": int(ev.isin(K).sum()), "HR": int((ev == "home_run").sum()),
            "Avg EV": _num(la.mean()), "HardHit%": _pct((la >= 95).sum(), len(la)),
        }
    return rows


def by_platoon(df: pd.DataFrame, fn) -> dict[str, dict]:
    """All / vs LHB / vs RHB."""
    out = {"All": fn(df)}
    if "stand" in df:
        out["vs LHB"] = fn(df[df["stand"].eq("L")])
        out["vs RHB"] = fn(df[df["stand"].eq("R")])
    return out
