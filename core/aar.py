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


def made_contact(df: pd.DataFrame) -> pd.Series:
    """Fouls and balls in play -- the swings with a real bat-ball contact point."""
    from core import hit_metrics as hm
    desc = _txt(df, "description")
    return desc.isin(hm.FOULS) | desc.eq("hit_into_play")


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
        # point of contact: only swings that actually hit the ball (fouls + balls in play) -- whiffs have no contact
        "cx": _num(d, CONTACT_X).where(made_contact(d)), "cy": _num(d, CONTACT_Y).where(made_contact(d)),
        "slope": plate_slope(d),
    })
    # Savant's 2026 swing-timing / miss-distance fields, whatever they're called in the download
    for c in d.columns:
        lc = c.lower()
        if "miss_dist" in lc:
            out["miss"] = _num(d, c)
        elif "timing" in lc and "timing" not in out:
            out["timing"] = _txt(d, c).str.replace("_", " ").str.title()
    out["pa"] = out["ab"].rank(method="dense").astype(int)
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


PLATE_HALF = 8.5             # inches: plate is 17" wide
FRONT_Y = 17 / 12            # Statcast y of the FRONT edge of the plate (ft; y = 0 is the back point)


def plate_slope(df: pd.DataFrame) -> pd.Series:
    """How fast the ball is drifting sideways as it crosses the front of the plate (inches sideways per inch of
    depth, catcher's view), from Statcast's pitch kinematics (velocity / acceleration measured at y = 50 ft)."""
    vx0, vy0, ax, ay = (_num(df, c) for c in ("vx0", "vy0", "ax", "ay"))
    a, b, c = 0.5 * ay, vy0, 50 - FRONT_Y
    disc = (b * b - 4 * a * c).clip(lower=0)
    t = (-b - np.sqrt(disc)) / (2 * a)
    return (vx0 + ax * t) / (vy0 + ay * t)


def plate_calibration(season: pd.DataFrame, orient: tuple[int, int] | None = None) -> dict | None:
    """Where his contact point sits on the plate, learned from his own swings -- no stance download.

    Statcast measures contact from his center of mass (hips). Contact happens where the BALL is, and Statcast
    tracks the ball: at f inches in front of the plate it's at plate_x + slope x f sideways. Pitches come in at
    different angles (slopes), so across his swings
        plate_x + slope x (depth from hips) - (side-to-side from hips)  =  X0 - slope x F0
    is a straight line in slope: its tilt gives F0 (where his hips are, inches in front of the plate's front edge --
    negative = behind it), its height X0 (how far off the plate). Fitted per batting side; the axis directions are
    picked by which way fits his swings and puts him in a real batter's box."""
    if season is None or season.empty or CONTACT_Y not in season or CONTACT_X not in season:
        return None
    s = season[made_contact(season)].copy() if "description" in season else season.copy()   # contact swings only
    s["_cx"], s["_cy"] = _num(s, CONTACT_X), _num(s, CONTACT_Y)
    s["_s"], s["_px"] = plate_slope(s), _num(s, "plate_x") * 12
    s = s.dropna(subset=["_cx", "_cy", "_s", "_px"])
    out = {}
    for side, g in s.groupby(s["stand"].astype(str) if "stand" in s else pd.Series("R", index=s.index)):
        if len(g) < 40 or g["_s"].std() < 0.01:
            continue
        best = None
        for sx in (1, -1):
            for sy in ((orient[0],) if orient else (1, -1)):
                z = g["_px"] + g["_s"] * sy * g["_cy"] - sx * g["_cx"]
                slope, x0 = np.polyfit(g["_s"], z, 1)                  # fit, drop bad tracks by residual, refit
                res = z - (x0 + slope * g["_s"])
                keep = (res - res.median()).abs() <= max(3 * (res - res.median()).abs().median(), 1.5)
                slope, x0 = np.polyfit(g.loc[keep, "_s"], z[keep], 1)
                f0 = -slope
                resid = float((z[keep] - (x0 + slope * g.loc[keep, "_s"])).std())
                box_ok = -60 <= f0 <= 10 and 12 <= abs(x0) <= 60 and np.sign(x0) == (-1 if side == "R" else 1)
                if box_ok and (best is None or resid < best["resid"]):
                    se = resid / (float(g.loc[keep, "_s"].std()) * max(int(keep.sum()), 1) ** 0.5)
                    best = {"sx": sx, "sy": sy, "F0": float(f0), "X0": float(x0), "n": int(keep.sum()), "resid": resid,
                            "se": float(se)}
        if best:
            out[side] = best
    return out or None


def on_plate(p: pd.DataFrame, cal: dict | None) -> pd.DataFrame:
    """Contact point on the plate for each measured swing: pf = inches in front of the plate's front edge
    (0 = front of the plate), pxc = inches from the middle of the plate, catcher's view -- where the ball was."""
    out = pd.DataFrame({"pf": np.nan, "pxc": np.nan}, index=p.index)
    if not cal or p.empty:
        return out
    for side, c in cal.items():
        m = p["stand"].astype(str).eq(side) & p["cy"].notna()
        if m.any():
            pf = c["F0"] + c["sy"] * p.loc[m, "cy"]
            out.loc[m, "pf"] = pf
            out.loc[m, "pxc"] = p.loc[m, "px"] * 12 + p.loc[m, "slope"] * pf
    return out


TEAM_KEY_ALIAS = {"guardians": "indians", "d-backs": "diamondbacks", "dbacks": "diamondbacks"}


def team_key(team_name: str) -> str:
    """'Boston Red Sox' -> 'red_sox' (the park key in the stadium outlines file)."""
    n = (team_name or "").lower().replace(".", "")
    for two in ("red sox", "white sox", "blue jays"):
        if n.endswith(two):
            return two.replace(" ", "_")
    last = n.split()[-1] if n.split() else ""
    return TEAM_KEY_ALIAS.get(last, last)


def stadium_outlines() -> pd.DataFrame:
    """Traced outlines of every MLB park (fence, infield dirt / grass, foul lines, plate) in Statcast's own hit
    coordinates -- the file that ships with pybaseball (from the GeomMLBStadiums project). Columns team, x, y,
    segment. A copy is kept in data/ in case the package moves it."""
    from pathlib import Path
    here = Path(__file__).resolve().parent.parent / "data" / "mlbstadiums.csv"
    cands = [here]
    try:
        import pybaseball
        cands.insert(0, Path(pybaseball.__file__).resolve().parent / "data" / "mlbstadiums.csv")
    except Exception:
        pass
    for c in cands:
        if c.exists():
            d = pd.read_csv(c)
            if {"team", "x", "y", "segment"} <= set(d.columns):
                if c != here:
                    try:
                        here.parent.mkdir(parents=True, exist_ok=True)
                        d[["team", "x", "y", "segment"]].to_csv(here, index=False)
                    except Exception:
                        pass
                return d[["team", "x", "y", "segment"]]
    return pd.DataFrame(columns=["team", "x", "y", "segment"])


def to_feet(x, y):
    """Statcast hit coordinates -> feet from home plate (x toward 1B side, y toward center)."""
    return (np.asarray(x, float) - 125.42) * 2.5, (198.27 - np.asarray(y, float)) * 2.5


def fence_distance(fx: np.ndarray, fy: np.ndarray, angle_deg: float) -> float:
    """Distance (ft) of a fence outline at a spray angle (0 = center, -45 = left-field line)."""
    ang = np.degrees(np.arctan2(fx, fy))
    i = int(np.argmin(np.abs(ang - angle_deg)))
    return float(np.hypot(fx[i], fy[i]))


HOME_HC = (125.42, 198.27)        # where Statcast's hit coordinates put home plate
WALL_MIN = 250                    # ft: anything closer than this can't be an MLB outfield wall


def wall_only(fx, fy, window: float = 1.0, min_dist: float = 0.0):
    """Just the wall, pole to pole, in order. The traced 'outfield_outer' path also runs down both foul lines to
    home plate, so many points share the same angle; keep only the outermost point near each angle (the wall),
    then order them from the left-field line to the right-field line."""
    fx, fy = np.asarray(fx, float), np.asarray(fy, float)
    ok = np.isfinite(fx) & np.isfinite(fy) & (np.hypot(fx, fy) >= min_dist)
    fx, fy = fx[ok], fy[ok]
    if len(fx) == 0:
        return fx, fy
    ang, dist = np.degrees(np.arctan2(fx, fy)), np.hypot(fx, fy)
    keep = np.array([dist[i] >= 0.97 * dist[np.abs(ang - ang[i]) <= window].max() for i in range(len(fx))],
                    dtype=bool)
    o = np.argsort(ang[keep])
    return fx[keep][o], fy[keep][o]


def park_outline(home_team_name: str, dims: dict | None, outlines: pd.DataFrame | None = None) -> dict:
    """The park to draw: {'segments': {name: (fx, fy) in feet}, 'source': 'outline' | 'distances', 'note': str,
    'poles': (lf, rf)}.

    Each traced park is first slid so ITS home plate sits where Statcast puts home plate (the file's plates aren't
    all at the same spot -- measuring from the wrong spot made foul lines look ~18 ft long). Then the foul-pole
    distances (the two ends of the fence) are checked against MLB's published lines for THIS venue. Center isn't
    checked: MLB's 'center' is the deepest point (Fenway's 420 is the triangle), not straightaway center."""
    d = stadium_outlines() if outlines is None else outlines
    key = team_key(home_team_name)
    t = d[d["team"].astype(str).eq(key)]
    if t.empty:
        return {"segments": {}, "source": "distances", "note": "no traced outline for this park -- fence from "
                                                               "published distances"}
    hp = t[t["segment"].astype(str).eq("home_plate")]
    dx, dy = ((HOME_HC[0] - hp["x"].mean(), HOME_HC[1] - hp["y"].max()) if not hp.empty else (0.0, 0.0))
    segs = {str(seg): to_feet(g["x"].values + dx, g["y"].values + dy)
            for seg, g in t.groupby("segment", sort=False)}
    fence = segs.get("outfield_outer")
    if fence is None:
        return {"segments": {}, "source": "distances", "note": "no traced fence -- fence from published distances"}
    fx, fy = wall_only(*fence, min_dist=WALL_MIN)          # no MLB wall is inside ~300 ft: drop foul-line points
    if len(fx) < 5:
        return {"segments": {}, "source": "distances", "note": "traced fence unusable -- fence from published "
                                                               "distances"}
    segs["outfield_outer"] = (fx, fy)
    if "infield_outer" in segs:
        segs["infield_outer"] = wall_only(*segs["infield_outer"], min_dist=60)
    ang, dist = np.degrees(np.arctan2(fx, fy)), np.hypot(fx, fy)
    poles = (float(dist[ang <= ang.min() + 1.5].max()), float(dist[ang >= ang.max() - 1.5].max()))
    if dims and {"leftLine", "rightLine"} <= set(dims):
        off = {"leftLine": abs(poles[0] - dims["leftLine"]), "rightLine": abs(poles[1] - dims["rightLine"])}
        if max(off.values()) > 15:
            why = f"poles {poles[0]:.0f} / {poles[1]:.0f} vs listed {dims['leftLine']:.0f} / {dims['rightLine']:.0f}"
            return {"segments": {}, "source": "distances", "poles": poles,
                    "note": f"traced outline doesn't match this venue ({why}) -- fence from published distances"}
    return {"segments": segs, "source": "outline", "poles": poles,
            "note": f"traced park outline (poles {poles[0]:.0f} / {poles[1]:.0f} ft)"}


def field_dims(feed: dict) -> dict:
    """Fence distances (ft) of the park this game was played in, from MLB's game feed."""
    fi = feed.get("gameData", {}).get("venue", {}).get("fieldInfo", {}) if feed else {}
    keys = ("leftLine", "left", "leftCenter", "center", "rightCenter", "right", "rightLine")
    d = {k: float(fi[k]) for k in keys if isinstance(fi.get(k), (int, float))}
    return d if {"leftLine", "center", "rightLine"} <= set(d) else {}


def last_pas(season: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """His last n plate appearances (through this game)."""
    if season is None or season.empty or "events" not in season:
        return pd.DataFrame()
    s = season.copy()
    s["_d"] = pd.to_datetime(s["game_date"], errors="coerce")
    s["_k"] = list(zip(s["game_pk"], s["at_bat_number"]))
    ends = s[s["events"].notna() & s["events"].astype(str).ne("truncated_pa")].sort_values(["_d", "at_bat_number"])
    keep = set(ends["_k"].tail(n))
    return s[s["_k"].isin(keep)].drop(columns=["_d", "_k"])


GAME_METRICS = {"GD%": "Good decision %", "O-SWING%": "Chase% (O-Swing%)", "Z-SWING%": "Z-Swing%", "WHIFF%": "Whiff%",
                "DEC RUNS/100": "Decision runs / 100", "xwOBA": "xwOBA",
                "EV": "Avg EV", "AVG BAT SPEED": "Avg bat speed"}


def metric_values(last30: pd.DataFrame, season_sc: pd.DataFrame, league: dict) -> dict:
    """{metric: [(label, value text), ...]} for Last 30 PA / Season / MLB season, every metric at once (the page
    switches between them without reloading)."""
    from core import hit_metrics as hm
    keys = list(GAME_METRICS)
    parts = []
    for lab, df in (("Last 30 PA", last30), ("Season", season_sc)):
        if df is not None and not df.empty:
            parts.append((lab, hm.compute(hm.prep(df), keys=keys).iloc[0]))
        else:
            parts.append((lab, None))
    lg = league.get(("all", "All"), {})
    out = {}
    for k in keys:
        row = [(lab, hm.fmt(k, v[k]) if v is not None else "-") for lab, v in parts]
        row.append(("MLB season", hm.fmt(k, lg[k][0]) if k in lg else "-"))
        out[k] = row
    return out
