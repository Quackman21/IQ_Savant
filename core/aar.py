"""After-Action Reports (AARs): one player's game, pitch by pitch, grouped into plate appearances.

Hitter AAR v1 (MLB). Pitcher / Catcher / Fielder AARs plug in later with the same building blocks:
    pitches(sc, feed)       -> one row per pitch with everything the page shows (Statcast + our models + video id)
    plate_appearances(p)    -> the rows grouped into PAs (summary line for each)
    box_line(feed, pid)     -> AB R H 2B 3B HR RBI BB K for that game
    contact_orientation(s)  -> which way Statcast's contact-point axes point (checked on his season, not assumed)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SAVANT_VIDEO = "https://baseballsavant.mlb.com/sporty-videos?playId={}"
CONTACT_X = "intercept_ball_minus_batter_pos_x_inches"
CONTACT_Y = "intercept_ball_minus_batter_pos_y_inches"
RESULT = {"ball": "Ball", "called_strike": "Called strike", "swinging_strike": "Swinging strike",
          "swinging_strike_blocked": "Swinging strike", "foul": "Foul", "foul_tip": "Foul tip",
          "hit_into_play": "In play", "blocked_ball": "Ball (blocked)", "hit_by_pitch": "Hit by pitch",
          "foul_bunt": "Foul bunt", "missed_bunt": "Missed bunt", "bunt_foul_tip": "Bunt foul tip",
          "pitchout": "Pitchout", "intent_ball": "Intent ball"}
BB = {"ground_ball": "GB", "line_drive": "LD", "fly_ball": "FB", "popup": "PU"}
PLAY = {"field_out": "Out", "force_out": "Force out", "grounded_into_double_play": "GIDP", "double_play": "Double play",
        "triple_play": "Triple play", "sac_fly": "Sac fly", "sac_bunt": "Sac bunt", "field_error": "Error",
        "fielders_choice": "Fielder's choice", "fielders_choice_out": "Fielder's choice",
        "strikeout_double_play": "Strikeout (DP)", "hit_by_pitch": "Hit by pitch", "catcher_interf": "Catcher int.",
        "sac_fly_double_play": "Sac fly (DP)", "intent_walk": "Intent walk"}


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)


def _txt(df: pd.DataFrame, c: str) -> pd.Series:
    return df[c].astype(str).where(df[c].notna(), "") if c in df else pd.Series("", index=df.index)


def play_ids(feed: dict) -> dict[tuple[int, int], str]:
    """(at_bat_number, pitch_number) -> Savant video id, from MLB's game feed."""
    out = {}
    for p in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        ab = int(p.get("about", {}).get("atBatIndex", -1)) + 1           # feed is 0-based, Statcast 1-based
        for e in p.get("playEvents", []):
            if e.get("isPitch") and e.get("playId") and e.get("pitchNumber") is not None:
                out[(ab, int(e["pitchNumber"]))] = e["playId"]
    return out


def names(feed: dict) -> dict[int, str]:
    people = feed.get("gameData", {}).get("players", {})
    return {int(v["id"]): v.get("fullName", "") for v in people.values() if "id" in v}


def decision_label(z: float) -> str:
    """Our decision grade for one pitch (z = how clearly right the choice was, from the decision model)."""
    from models.decisions_iq import CLEAR, NEUTRAL
    if z != z:
        return ""
    if z >= CLEAR:
        return "Great"
    if z >= NEUTRAL:
        return "Good"
    if z > -NEUTRAL:
        return "Neutral"
    return "Bad" if z > -CLEAR else "Poor"


def pitches(sc: pd.DataFrame, feed: dict | None = None) -> pd.DataFrame:
    """One row per pitch of the hitter's game, in order, with every field the AAR shows."""
    if sc is None or sc.empty:
        return pd.DataFrame()
    from core import hit_metrics as hm
    d = sc.copy()
    d["_ab"], d["_pn"] = _num(d, "at_bat_number"), _num(d, "pitch_number")
    d = d.sort_values(["_ab", "_pn"]).reset_index(drop=True)
    desc = _txt(d, "description")
    rhp = _txt(d, "p_throws").eq("R")
    swing = desc.isin(hm.SWINGS)
    out = pd.DataFrame({
        "ab": d["_ab"].astype("Int64"), "pn": d["_pn"].astype("Int64"), "n": np.arange(1, len(d) + 1),
        "inning": _num(d, "inning").astype("Int64"), "half": np.where(_txt(d, "inning_topbot").eq("Top"), "▲", "▼"),
        "count": _num(d, "balls").fillna(0).astype(int).astype(str) + "-" +
                 _num(d, "strikes").fillna(0).astype(int).astype(str),
        "pitcher_id": _num(d, "pitcher").astype("Int64"), "p_throws": _txt(d, "p_throws"),
        "stand": _txt(d, "stand"),
        "result": desc.map(RESULT).fillna(desc.str.replace("_", " ").str.capitalize()),
        "play": _txt(d, "events").map(PLAY).fillna(_txt(d, "events").str.replace("_", " ").str.capitalize()),
        "ptype": _txt(d, "pitch_type"), "pgroup": hm.pitch_group(d),
        "velo": _num(d, "release_speed"),
        "hb": np.where(rhp, -1, 1) * _num(d, "pfx_x") * 12,            # inches, arm side +
        "ivb": _num(d, "pfx_z") * 12,
        "bb": _txt(d, "bb_type").map(BB).fillna(""),
        "ev": _num(d, "launch_speed"), "la": _num(d, "launch_angle"), "dist": _num(d, "hit_distance_sc"),
        "bat_speed": _num(d, "bat_speed").where(swing),
        "zone": hm.attack_zone(d), "swing": swing, "bip": _txt(d, "type").eq("X"),
        "px": _num(d, "plate_x"), "pz": _num(d, "plate_z"), "sz_top": _num(d, "sz_top"),
        "sz_bot": _num(d, "sz_bot"), "hc_x": _num(d, "hc_x"), "hc_y": _num(d, "hc_y"),
        "cx": _num(d, CONTACT_X).where(swing), "cy": _num(d, CONTACT_Y).where(swing),
    })
    # Savant's 2026 swing-timing / miss-distance fields, whatever they're called in the download
    for c in d.columns:
        lc = c.lower()
        if "miss_dist" in lc:
            out["miss"] = _num(d, c)
        elif "timing" in lc and "timing" not in out:
            out["timing"] = _txt(d, c).str.replace("_", " ").str.title()
    try:
        from models.xwoba import expected_woba
        out["xwoba"] = expected_woba(d).reindex(d.index).where(out["bip"]).values
    except Exception:
        out["xwoba"] = np.nan
    try:
        from models import decisions_iq as dq
        s = dq.score(d)
        out["dec"] = s["z"].map(decision_label).values
        out["dec_runs"] = s["runs_vs_avg"].values
    except Exception:
        out["dec"], out["dec_runs"] = "", np.nan
    if feed:
        vid, who = play_ids(feed), names(feed)
        out["video"] = [vid.get((int(a), int(p)), "") if pd.notna(a) and pd.notna(p) else ""
                        for a, p in zip(out["ab"], out["pn"])]
        out["pitcher"] = out["pitcher_id"].map(lambda x: who.get(int(x), "") if pd.notna(x) else "")
    else:
        out["video"], out["pitcher"] = "", ""
    return out


def plate_appearances(p: pd.DataFrame) -> list[dict]:
    """Rows grouped by at-bat, with the summary line each collapsed PA shows."""
    pas = []
    if p is None or p.empty:
        return pas
    for i, (ab, g) in enumerate(p.groupby("ab", sort=True), 1):
        last = g.iloc[-1]
        res = last["play"] or last["result"]
        bip = g[g["bip"]]
        extra = ""
        if not bip.empty:
            b = bip.iloc[-1]
            bits = [b["bb"], f"{b['ev']:.1f} mph" if b["ev"] == b["ev"] else "",
                    f"xwOBA {b['xwoba']:.3f}".replace("0.", ".") if b["xwoba"] == b["xwoba"] else ""]
            extra = " · ".join(x for x in bits if x)
        pas.append({"i": i, "ab": int(ab), "inning": f"{last['half']}{last['inning']}", "pitcher": last["pitcher"],
                    "p_throws": last["p_throws"], "n_pitches": len(g), "final_count": g.iloc[-1]["count"],
                    "result": res, "extra": extra, "rows": g,
                    "dec_runs": float(g["dec_runs"].sum(min_count=1)) if g["dec_runs"].notna().any() else math.nan})
    return pas


def box_line(feed: dict, pid: int) -> dict:
    """The hitter's line for the game, from MLB's box score."""
    for side in ("away", "home"):
        players = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {}).get("players", {})
        p = players.get(f"ID{pid}")
        if p:
            b = p.get("stats", {}).get("batting", {})
            return {"AB": b.get("atBats", 0), "R": b.get("runs", 0), "H": b.get("hits", 0),
                    "2B": b.get("doubles", 0), "3B": b.get("triples", 0), "HR": b.get("homeRuns", 0),
                    "RBI": b.get("rbi", 0), "BB": b.get("baseOnBalls", 0), "K": b.get("strikeOuts", 0),
                    "side": side, "team_id": feed.get("gameData", {}).get("teams", {}).get(side, {}).get("id")}
    return {}


def contact_orientation(season: pd.DataFrame) -> tuple[int, int] | None:
    """Statcast doesn't document which way the contact-point axes point, so check it on his season instead of
    assuming: pulled balls are met farther OUT FRONT, and outside pitches are met farther FROM HIS BODY.
    Returns (sign for depth, sign for distance from body) so + = out front / + = away from body."""
    if season is None or season.empty or CONTACT_Y not in season or CONTACT_X not in season:
        return None
    from core import hit_metrics as hm
    s = season[pd.to_numeric(season[CONTACT_Y], errors="coerce").notna()]
    if len(s) < 25:
        return None
    rhb = s["stand"].astype(str).eq("R") if "stand" in s else pd.Series(True, index=s.index)
    spray = hm.spray_angle(s)
    pull = np.where(rhb, -spray, spray)                              # + = toward his pull side
    out_x = np.where(rhb, 1, -1) * pd.to_numeric(s["plate_x"], errors="coerce")   # + = away from him
    y, x = pd.to_numeric(s[CONTACT_Y], errors="coerce"), pd.to_numeric(s[CONTACT_X], errors="coerce")
    ry = pd.Series(pull, index=s.index).corr(y)
    rx = pd.Series(out_x, index=s.index).corr(x * np.where(rhb, 1, -1))
    if ry != ry or rx != rx:
        return None
    return (1 if ry >= 0 else -1, (1 if rx >= 0 else -1))


GAME_METRICS = {"O-SWING%": "Chase% (O-Swing%)", "Z-SWING%": "Z-Swing%", "WHIFF%": "Whiff%",
                "DEC RUNS/100": "Decision runs / 100", "GD%": "Good decision %", "xwOBA": "xwOBA",
                "EV": "Avg EV", "AVG BAT SPEED": "Avg bat speed"}


def metric_tiles(game_sc: pd.DataFrame, season_sc: pd.DataFrame, league: dict, key: str) -> list[tuple[str, str]]:
    """(label, value) for Game / Season / MLB average of one catalog stat."""
    from core import hit_metrics as hm
    out = []
    for lab, df in (("Game", game_sc), ("Season", season_sc)):
        v = hm.compute(hm.prep(df), keys=[key]).iloc[0][key] if df is not None and not df.empty else math.nan
        out.append((lab, hm.fmt(key, v)))
    lg = league.get(("all", "All"), {}).get(key)
    out.append(("MLB avg", hm.fmt(key, lg[0]) if lg else "-"))
    return out
