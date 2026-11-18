"""Catching AAR: every pitch a catcher received in one game (balls in play left out -- he didn't do anything on
those), with what the IQ catcher models say about each one.

Per pitch:
  Strike prob     league-average umpire's chance it's called a strike if taken (IQ Swing Decisions called-strike
                  model -- location vs the hitter's zone, count, sides, pitch type; never the catcher or umpire).
  Framing runs    takes only: (called strike 1 / ball 0 - strike prob) x what a strike instead of a ball is worth in
                  that count. 2026+: the umpire's ORIGINAL call (an ABS challenge doesn't change his framing).
  WP/PB chance    blocking opportunities only (runner on, not in play, not a HBP): league-average chance it gets by.
  Blocking runs   (chance - what happened) x what a WP / PB costs in that base / out state.
  Throwing runs   steal attempts only: (caught 1 / safe 0 - caught-stealing chance) x the run swing for that base.
  Gameday zone    Statcast's 1-14 zone, named from the CATCHER's side (right-handed catcher, like the blocking
                  model: arm side = his right / 1B side, glove side = his left / 3B side) -- 1-9 inside the zone,
                  11-14 the four corners outside it.
  Region          Savant's attack region (Heart / Shadow / Chase / Waste) + which edge, from the catcher's side:
                  e.g. "Arm-side shadow", "Low glove-side chase". Shadow straddles the edge of the zone.
  Ground          where the ball first reaches the dirt (Statcast's measured flight, carried past the plate),
                  if it does before 3 ft past the middle of the plate.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.aar import PLAY, RESULT, SAVANT_VIDEO, _num, _txt, names, play_ids  # noqa: F401

FRONT_FT = 17 / 12                    # Statcast y of the plate's front edge (y = 0 is the back point)
MID_FT = 17 / 24                      # middle of the plate, front to back
END_FT = MID_FT - 3.0                 # the 3D stops 3 ft past the middle of the plate ...
BALL_R_FT = 2.9 / 24                  # ... or when the ball's bottom reaches the dirt (radius 1.45")
INFIELD = {"1B", "2B", "3B", "SS"}
OUTFIELD = {"LF", "CF", "RF"}
EVENT_NAME = {"wild_pitch": "Wild pitch", "passed_ball": "Passed ball", "stolen_base_2b": "SB 2B",
              "stolen_base_3b": "SB 3B", "stolen_base_home": "SB home", "caught_stealing_2b": "CS 2B",
              "caught_stealing_3b": "CS 3B", "caught_stealing_home": "CS home"}
# Statcast's Gameday zones (catcher's view: 1 = up and to the catcher's left ... 9 = down-right; 11-14 outside)
_ZROW = {1: "Up", 2: "Up", 3: "Up", 4: "Mid", 5: "Mid", 6: "Mid", 7: "Down", 8: "Down", 9: "Down"}
_ZCOL = {1: -1, 4: -1, 7: -1, 2: 0, 5: 0, 8: 0, 3: 1, 6: 1, 9: 1}          # -1 = catcher's left (3B side)
_ZOUT = {11: ("Up", -1), 12: ("Up", 1), 13: ("Down", -1), 14: ("Down", 1)}
REPORTS = ["Hitting", "Catching", "Infield", "Outfield"]


# ----------------------------------------------------------------------------- which reports this player gets
def positions(feed: dict, pid: int) -> dict:
    """{'Hitting': batted, 'Catching': caught, 'Infield': played the infield, 'Outfield': played the outfield}."""
    out = dict.fromkeys(REPORTS, False)
    for side in ("away", "home"):
        p = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {}).get("players", {}).get(f"ID{pid}")
        if not p:
            continue
        b = p.get("stats", {}).get("batting", {}) or {}
        out["Hitting"] = bool(b.get("plateAppearances") or b.get("atBats") or b.get("baseOnBalls")
                              or b.get("hitByPitch") or b.get("sacFlies") or b.get("sacBunts"))
        pos = {str(x.get("abbreviation", "")) for x in (p.get("allPositions") or [])}
        if not pos and p.get("position"):
            pos = {str(p["position"].get("abbreviation", ""))}
        out["Catching"] = "C" in pos
        out["Infield"] = bool(pos & INFIELD)
        out["Outfield"] = bool(pos & OUTFIELD)
    return out


def available(feed: dict, pid: int, caught: bool | None = None) -> list[str]:
    """The report types to offer, in order. `caught` = Statcast shows him behind the plate (fielder_2) -- covers a
    box score that hasn't listed him at C yet."""
    pos = positions(feed, pid)
    if caught:
        pos["Catching"] = True
    return [k for k in REPORTS if pos[k]]


# ----------------------------------------------------------------------------- zones, named from the catcher's side
def _side(col: int, p_throws: str | None = None) -> str:
    """-1 / 0 / +1 across the plate (catcher's view) -> 'Glove' / 'Mid' / 'Arm' for a right-handed catcher
    (his glove side = his left / 3B side, arm side = his right / 1B side). The pitcher's hand doesn't change it."""
    return "Mid" if col == 0 else ("Arm" if col > 0 else "Glove")


def gameday_name(zone, p_throws: str | None = None) -> str:
    """Statcast zone number -> its real name from the catcher's side, e.g. 14 = 'Down arm side (out)'."""
    try:
        z = int(zone)
    except (TypeError, ValueError):
        return ""
    if z == 5:
        return "5 · Middle-middle"
    if z in _ZROW:
        row, side = _ZROW[z], _side(_ZCOL[z], p_throws)
        if side == "Mid":
            return f"{z} · {row} middle"
        return f"{z} · {row}-{side.lower()}" if row != "Mid" else f"{z} · Middle {side.lower()} side"
    if z in _ZOUT:
        row, col = _ZOUT[z]
        return f"{z} · {row} {_side(col, p_throws).lower()} side (out)"
    return str(z)


def region(df: pd.DataFrame) -> pd.Series:
    """Savant's attack region + which edge, from the catcher's side (right-handed catcher: arm side = his right):
    'Heart', 'Arm-side shadow', 'Low glove-side chase', 'High waste' ..."""
    px, pz = _num(df, "plate_x"), _num(df, "plate_z")
    top, bot = _num(df, "sz_top"), _num(df, "sz_bot")
    nx = px / (17 / 24)                                       # + = catcher's right
    nz = (pz - (top + bot) / 2) / ((top - bot) / 2)
    r = np.maximum(nx.abs(), nz.abs())
    reg = np.select([r < 0.67, r < 1.33, r < 2.0], ["heart", "shadow", "chase"], "waste")
    out = []
    for x, z, rr, g in zip(nx, nz, r, reg):
        if not (rr == rr):
            out.append("")
            continue
        if g == "heart":
            out.append("Heart")
            continue
        # an edge counts when the ball is past 0.67 (the heart's edge) that way; both = a corner
        v = "High" if z >= 0.67 else "Low" if z <= -0.67 else ""
        h = ""
        if abs(x) >= 0.67:
            h = "arm-side" if x > 0 else "glove-side"
        lab = " ".join(s for s in (v, h, g) if s)
        out.append(lab[0].upper() + lab[1:])
    return pd.Series(out, index=df.index)


FRAME_ZONES = ["Heart", "Shadow - Top Left", "Shadow - Top", "Shadow - Top Right", "Shadow - Left", "Shadow - Right",
               "Shadow - Bottom Left", "Shadow - Bottom", "Shadow - Bottom Right", "Chase / Waste"]


def framing_zone(df: pd.DataFrame) -> pd.Series:
    """Savant's catcher-framing split (catcher's view), with the Shadow band in eight pieces: the four edges and the
    four corners (past the heart's edge both ways). Heart inside, Chase / Waste outside.
    Left = the catcher's left (his glove side), Right = his right (arm side)."""
    px, pz = _num(df, "plate_x"), _num(df, "plate_z")
    top, bot = _num(df, "sz_top"), _num(df, "sz_bot")
    xn = px / (17 / 24)
    zn = (pz - (top + bot) / 2) / ((top - bot) / 2)
    r = np.maximum(xn.abs(), zn.abs())
    v = np.where(zn >= 0.67, "Top", np.where(zn <= -0.67, "Bottom", ""))
    h = np.where(xn >= 0.67, "Right", np.where(xn <= -0.67, "Left", ""))
    edge = ["Shadow - " + " ".join(p for p in (a, b) if p) for a, b in zip(v, h)]
    out = np.select([r < 0.67, r < 1.33, r >= 1.33], ["Heart", np.array(edge, dtype=object), "Chase / Waste"], "")
    return pd.Series(out, index=df.index)


# ----------------------------------------------------------------------------- the ball's flight
def _t_at(vy0, ay, y_ft):
    a, b, c = 0.5 * ay, vy0, 50 - y_ft
    return (-b - np.sqrt(np.clip(b * b - 4 * a * c, 0, None))) / (2 * a)


def _at50(df: pd.DataFrame):
    """(x, z) at y = 50 ft, anchored so the measured flight crosses the front of the plate at (plate_x, plate_z)."""
    n = lambda c: _num(df, c)
    tf = _t_at(n("vy0"), n("ay"), FRONT_FT)
    return (n("plate_x") - (n("vx0") * tf + 0.5 * n("ax") * tf ** 2),
            n("plate_z") - (n("vz0") * tf + 0.5 * n("az") * tf ** 2))


def ground(df: pd.DataFrame) -> pd.DataFrame:
    """Where each pitch's measured flight first reaches the dirt: gx (ft, catcher's right +), gy (ft in front of
    the plate's FRONT edge; negative = over / behind the plate), dirt = it lands before 3 ft past the middle of the
    plate. Also t_end: when the 3D view stops (that landing, or 3 ft past the middle)."""
    n = lambda c: _num(df, c)
    x50, z50 = _at50(df)
    vz0, az, vy0, ay, vx0, ax = n("vz0"), n("az"), n("vy0"), n("ay"), n("vx0"), n("ax")
    # z(t) = z50 + vz0 t + az t^2 / 2 = ball radius: the later root (the ball is coming down)
    a, b, c = 0.5 * az, vz0, z50 - BALL_R_FT
    disc = b * b - 4 * a * c
    tg = (-b - np.sqrt(np.clip(disc, 0, None))) / (2 * a)
    tg = tg.where(disc >= 0)
    yg = 50 + vy0 * tg + 0.5 * ay * tg ** 2
    xg = x50 + vx0 * tg + 0.5 * ax * tg ** 2
    te = _t_at(vy0, ay, END_FT)
    dirt = (yg >= END_FT) & tg.notna()
    return pd.DataFrame({"gx": xg, "gy": yg - FRONT_FT, "dirt": dirt.fillna(False),
                         "t_end": np.where(dirt, tg, te)}, index=df.index)


def flight(row: dict | pd.Series, step: float = 0.004) -> list[list[float]] | None:
    """One pitch, release -> the dirt (or 3 ft past the middle of the plate), in the 3D frame (inches: x = catcher's
    right, y = toward the pitcher from the front of the plate, z = up) + t (s from the plate's front edge)."""
    g = lambda k: float(row[k]) if k in row and row[k] is not None and row[k] == row[k] else None
    need = ("vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z")
    if any(g(k) is None for k in need):
        return None
    d = pd.DataFrame([{k: g(k) for k in need}])
    x50, z50 = _at50(d)
    x50, z50 = float(x50.iloc[0]), float(z50.iloc[0])
    vx0, vy0, vz0, ax, ay, az = (g(k) for k in ("vx0", "vy0", "vz0", "ax", "ay", "az"))
    ext = g("release_extension") or 6.2
    t_rel = float(_t_at(np.array(vy0), np.array(ay), 60.5 - ext))
    t_front = float(_t_at(np.array(vy0), np.array(ay), FRONT_FT))
    t_end = float(ground(d)["t_end"].iloc[0])
    if not (math.isfinite(t_rel) and math.isfinite(t_end)) or t_end <= t_rel:
        return None
    ts = list(np.arange(t_rel, t_end, step)) + [t_end]
    pts = []
    for t in ts:
        x = x50 + vx0 * t + 0.5 * ax * t * t
        y = 50 + vy0 * t + 0.5 * ay * t * t
        z = z50 + vz0 * t + 0.5 * az * t * t
        pts.append([round(x * 12, 2), round((y - FRONT_FT) * 12, 2), round(max(z, BALL_R_FT) * 12, 2),
                    round(t - t_front, 4)])
    return pts


# ----------------------------------------------------------------------------- the models, per pitch
def challenges(feed: dict | None, pk: int) -> pd.DataFrame:
    """This game's ABS challenges, straight from the game feed (no season file needed)."""
    from core import abs_calls
    rows = abs_calls.parse((feed or {}).get("liveData", {}).get("plays", {}) or {}, pk)
    return pd.DataFrame(rows, columns=abs_calls.COLS)


def _abs(d: pd.DataFrame, ch: pd.DataFrame | None, pid: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pitches with the umpire's ORIGINAL call on challenged takes, per pitch: abs = Successful / Failed on HIS
    challenges, abs_by = who challenged, orig_call = the call before an overturn)."""
    out = pd.DataFrame({"abs": "", "abs_by": "", "orig_call": "", "abs_runs": np.nan, "abs_cost": np.nan,
                        "abs_net": np.nan}, index=d.index)
    dd = d
    try:
        from core import abs_calls
        dd = abs_calls.original_calls(d, challenges=ch) if ch is not None else abs_calls.original_calls(d)
    except Exception:
        return d, out
    if "abs_challenged" not in dd:
        return dd, out
    c = dd["abs_challenged"].fillna(False).astype(bool)
    over = dd["abs_overturned"].fillna(False).astype(bool)
    who = pd.to_numeric(dd["abs_challenger"], errors="coerce")
    bat, pit = _num(d, "batter"), _num(d, "pitcher")
    out["abs_by"] = np.select([c & who.eq(pid), c & who.eq(bat), c & who.eq(pit), c], ["Catcher", "Batter", "Pitcher",
                                                                                         "Challenged"], "")
    out["abs"] = np.where(c & who.eq(pid), np.where(over, "Successful", "Failed"), "")
    try:                                    # what his overturns were worth + what his lost ones cost
        from core import abs_stats
        season = int(pd.to_datetime(d["game_date"]).dt.year.max())
        meta = abs_stats.load(season)[2] if abs_stats.paths(season)["meta"].exists() else {}
        val = abs_stats.call_values(d, meta.get("re288"), meta.get("values")).values      # full game state
        out["abs_runs"] = np.where(out["abs"].eq("Successful"), val, np.where(out["abs"].eq("Failed"), 0.0, np.nan))
        # a lost challenge costs its option value: outs left in the game x the season's per-out rate for the
        # challenges his team had left (2 or 1)
        left = pd.Series(2, index=d.index)
        if ch is not None and not ch.empty and "team" in ch:
            key = list(zip(_num(d, "at_bat_number"), _num(d, "pitch_number")))
            chk = {(float(a), float(p)): (t, o) for a, p, t, o in zip(ch["at_bat_number"], ch["pitch_number"],
                                                                         ch["team"], ch["overturned"])}
            cf = c & (who.eq(pid) | who.eq(pit) | who.eq(_num(d, "fielder_2")))     # challenged by his side
            fld_ids = {chk[k][0] for k, isc in zip(key, cf) if isc and k in chk}
            if fld_ids:
                tid = next(iter(fld_ids))
                lost = sorted((float(a), float(p)) for a, p, t, o in zip(ch["at_bat_number"], ch["pitch_number"],
                                                                          ch["team"], ch["overturned"])
                              if t == tid and str(o).lower() not in ("true", "1", "1.0"))
                inn = _num(d, "inning")
                left = pd.Series([max(2 - sum(1 for x in lost if x < k), 0) for k in key], index=d.index)
                left = left.where(~(inn.ge(10) & left.eq(0)), 1)
        rates = {int(k): v for k, v in (meta.get("per_out") or {}).items() if str(k).isdigit()} or abs_stats.DEFAULT_PER_OUT
        per = left.clip(1, 2).map(rates).fillna(abs_stats.DEFAULT_PER_OUT[2])
        ol = abs_stats.outs_left(d)
        out["abs_cost"] = np.where(out["abs"].eq("Failed"), ol * per, np.where(out["abs"].ne(""), 0.0, np.nan))
        out["abs_net"] = out["abs_runs"] - out["abs_cost"]
    except Exception:
        out["abs_runs"] = np.nan
    out["orig_call"] = np.where(c & over, dd["description"].map(RESULT).fillna(""), "")
    return dd, out


def _framing(d: pd.DataFrame, dd: pd.DataFrame | None = None) -> pd.DataFrame:
    """strike_prob (every pitch) + framing runs / extra strike (takes) -- NaN until the decision model is trained.
    dd = the same pitches on the umpire's original calls (ABS)."""
    out = pd.DataFrame({"strike_prob": np.nan, "extra": np.nan, "frame_runs": np.nan}, index=d.index)
    dd = d if dd is None else dd
    try:
        from models import decisions_iq as dq
        if dq.load() is None:
            return out
        out["strike_prob"] = dq.called_strike_chance(dd).values
        take = dd["description"].astype(str).isin({"called_strike", "ball"})
        val = dq.strike_value(dd).fillna(0.125)
        strike = dd["description"].astype(str).eq("called_strike").astype(float)
        out["extra"] = (strike - out["strike_prob"]).where(take)
        out["frame_runs"] = (out["extra"] * val).where(take)
    except Exception:
        pass
    return out


def _catching(d: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    """Blocking (opportunities) + throwing (steal attempts), per pitch."""
    from models import catching as cm
    out = pd.DataFrame({"opp": cm.block_rows(d) if {"on_1b", "on_2b", "on_3b", "type"} <= set(d) else False,
                        "wp_chance": np.nan, "got_by": 0.0, "block_runs": np.nan, "tier": "",
                        "steal": "", "cs_chance": np.nan, "cs": np.nan, "throw_runs": np.nan}, index=d.index)
    ev = events if events is not None else pd.DataFrame(columns=["game_pk", "at_bat_number", "pitch_number", "event"])
    out["got_by"] = cm.wp_pb_flag(d, ev).values
    att = cm.steal_attempts(d, ev)
    if not att.empty:                                      # mark the pitch even without a model
        key = cm.keyed(d)
        for _, a in att.iterrows():
            m = key.eq(a["key"])
            out.loc[m, "steal"] = a["base"]
            out.loc[m, "cs"] = a["cs"]
    b = cm.load()
    if b is None:
        return out
    for m, sub in cm._bundles(d, b):
        dm = d[m & out["opp"]]
        if not dm.empty and sub.get("block") is not None:
            t = cm.block_table(dm, ev, sub)
            out.loc[dm.index, "wp_chance"] = t["p"]
            out.loc[dm.index, "block_runs"] = t["runs"]
            out.loc[dm.index, "tier"] = t["tier"].map({"easy": "Easy", "medium": "Medium", "tough": "Tough"})
        a = cm.steal_attempts(d[m], ev)
        if not a.empty and sub.get("throw") is not None:
            p = sub["throw"].predict_proba(cm.throw_features(a)[cm.THROW_FEATURES])[:, 1]
            val = a["base"].map(cm.CS_RUNS).fillna(0.65).values
            key = cm.keyed(d)
            for k, pp, cs, v in zip(a["key"], p, a["cs"].values, val):
                mm = key.eq(k)
                out.loc[mm, "cs_chance"] = pp
                out.loc[mm, "throw_runs"] = (cs - pp) * v
    return out


def runner_events(feed: dict, pk: int) -> pd.DataFrame:
    """WP / PB / SB / CS for this game, tied to the pitch each happened on (MLB play-by-play in the game feed)."""
    from pipeline.fetch_events import parse
    rows = parse(feed.get("liveData", {}).get("plays", {}) or {}, pk)
    return pd.DataFrame(rows, columns=["game_pk", "at_bat_number", "pitch_number", "event", "runner", "start", "end",
                                       "is_out"])


def caught(sc: pd.DataFrame, pid: int) -> pd.DataFrame:
    """His pitches: everything thrown while he was behind the plate (Statcast fielder_2)."""
    if sc is None or sc.empty or "fielder_2" not in sc:
        return pd.DataFrame()
    return sc[_num(sc, "fielder_2").eq(pid)]


def pitches(sc: pd.DataFrame, feed: dict | None, pid: int, events: pd.DataFrame | None = None,
            abs_rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per pitch he received, in order, balls in play left out -- every field the Catching AAR shows."""
    d = caught(sc, pid)
    if d.empty:
        return pd.DataFrame()
    d = d[~_txt(d, "type").eq("X") & ~_txt(d, "description").eq("hit_into_play")].copy()
    if d.empty:
        return pd.DataFrame()
    from core import hit_metrics as hm
    d["_ab"], d["_pn"] = _num(d, "at_bat_number"), _num(d, "pitch_number")
    d = d.sort_values(["_ab", "_pn"]).reset_index(drop=True)
    desc = _txt(d, "description")
    rhp = _txt(d, "p_throws").ne("L")
    on = [", ".join(b for b, c in (("1B", "on_1b"), ("2B", "on_2b"), ("3B", "on_3b"))
                    if c in d and pd.notna(v.get(c))) or "Empty" for v in d.to_dict("records")]
    gd = ground(d)
    out = pd.DataFrame({
        "ab": d["_ab"].astype("Int64"), "pn": d["_pn"].astype("Int64"), "n": np.arange(1, len(d) + 1),
        "inning": _num(d, "inning").astype("Int64"), "half": np.where(_txt(d, "inning_topbot").eq("Top"), "▲", "▼"),
        "count": _num(d, "balls").fillna(0).astype(int).astype(str) + "-" +
                 _num(d, "strikes").fillna(0).astype(int).astype(str),
        "outs": _num(d, "outs_when_up").fillna(0).astype(int),
        "pitcher_id": _num(d, "pitcher").astype("Int64"), "batter_id": _num(d, "batter").astype("Int64"),
        "p_throws": _txt(d, "p_throws"), "stand": _txt(d, "stand"),
        "result": desc.map(RESULT).fillna(desc.str.replace("_", " ").str.capitalize()),
        "ptype": _txt(d, "pitch_type"), "pgroup": hm.pitch_group(d).fillna(""),
        "velo": _num(d, "release_speed"),
        "hb": np.where(rhp, -1, 1) * _num(d, "pfx_x") * 12, "ivb": _num(d, "pfx_z") * 12,
        "px": _num(d, "plate_x"), "pz": _num(d, "plate_z"), "sz_top": _num(d, "sz_top"), "sz_bot": _num(d, "sz_bot"),
        "gz": _num(d, "zone").astype("Int64"),
        "zone": [gameday_name(z, h) if pd.notna(z) else "" for z, h in zip(_num(d, "zone"), _txt(d, "p_throws"))],
        "region": region(d), "fzone": framing_zone(d), "take": desc.isin({"called_strike", "ball"}), "strike": desc.eq("called_strike"),
        "swing": desc.isin(hm.SWINGS),
        "on": on, "gx": gd["gx"], "gy": gd["gy"], "dirt": gd["dirt"].astype(bool),
    })
    if abs_rows is None and feed:
        abs_rows = challenges(feed, int(_num(d, "game_pk").dropna().iloc[0])) if _num(d, "game_pk").notna().any() else None
    dd, ab = _abs(d, abs_rows, pid)
    out = out.join(_framing(d, dd)).join(_catching(d, events)).join(ab)
    # what happened on the play: the PA's result on its last pitch + any runner event on this pitch
    ev_txt = pd.Series("", index=d.index)
    if events is not None and not events.empty:
        from models import catching as cm
        e = events.dropna(subset=["pitch_number"]).copy()
        e["key"] = cm.keyed(e)
        lab = e.groupby("key")["event"].apply(lambda s: ", ".join(dict.fromkeys(EVENT_NAME.get(x, x) for x in s)))
        ev_txt = cm.keyed(d).map(lab).fillna("")
    pa_ev = _txt(d, "events")
    pa_txt = pa_ev.map(PLAY).fillna(pa_ev.str.replace("_", " ").str.capitalize())
    out["play"] = [" · ".join(x for x in (a, b) if x) for a, b in zip(pa_txt, ev_txt)]
    if feed:
        vid, who = play_ids(feed), names(feed)
        out["video"] = [vid.get((int(a), int(p)), "") if pd.notna(a) and pd.notna(p) else ""
                        for a, p in zip(out["ab"], out["pn"])]
        out["pitcher"] = out["pitcher_id"].map(lambda x: who.get(int(x), "") if pd.notna(x) else "")
        out["batter"] = out["batter_id"].map(lambda x: who.get(int(x), "") if pd.notna(x) else "")
    else:
        out["video"], out["pitcher"], out["batter"] = "", "", ""
    out["catch_runs"] = out[["frame_runs", "block_runs", "throw_runs"]].sum(axis=1, min_count=1)
    # the raw flight, for the 3D (kept off the table)
    for c in ("vx0", "vy0", "vz0", "ax", "ay", "az", "release_extension"):
        out["_" + c] = _num(d, c)
    return out


def scenes(p: pd.DataFrame) -> dict:
    """n -> what the 3D draws for that pitch: the flight + its readouts."""
    out = {}
    for r in p.to_dict("records"):
        row = {k[1:]: r.get(k) for k in ("_vx0", "_vy0", "_vz0", "_ax", "_ay", "_az", "_release_extension")}
        row.update(plate_x=r.get("px"), plate_z=r.get("pz"))
        pts = flight(row)
        if not pts:
            continue
        sz = [r.get("sz_bot"), r.get("sz_top")]
        out[int(r["n"])] = {"pitch": pts, "sz": [round(12 * (v if v == v and v is not None else d), 2)
                                                 for v, d in zip(sz, (1.6, 3.4))],
                            "dirt": bool(r.get("dirt")), "read": _read(r)}
    return out


def _clean(v, nd=3):
    try:
        f = float(v)
        return round(f, nd) if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _read(r: dict) -> dict:
    return {"ptype": r.get("ptype") or "", "velo": _clean(r.get("velo"), 1), "count": r.get("count"),
            "result": r.get("result") or "", "play": r.get("play") or "", "pitcher": r.get("pitcher") or "",
            "batter": r.get("batter") or "", "zone": r.get("zone") or "", "region": r.get("fzone") or "",
            "hb": _clean(r.get("hb"), 1), "ivb": _clean(r.get("ivb"), 1), "strike_prob": _clean(r.get("strike_prob")),
            "frame_runs": _clean(r.get("frame_runs")), "wp_chance": _clean(r.get("wp_chance")),
            "block_runs": _clean(r.get("block_runs")), "throw_runs": _clean(r.get("throw_runs")),
            "cs_chance": _clean(r.get("cs_chance")), "steal": r.get("steal") or "", "opp": bool(r.get("opp")),
            "got_by": bool(r.get("got_by")), "on": r.get("on") or "", "gx": _clean(r.get("gx"), 2),
            "gy": _clean(r.get("gy"), 2), "dirt": bool(r.get("dirt")), "video": r.get("video") or "",
            "inning": f'{r.get("half")}{r.get("inning")}', "abs": r.get("abs") or "",
            "abs_by": r.get("abs_by") or ""}


def line(p: pd.DataFrame, events: pd.DataFrame | None, pid_pitches: pd.DataFrame | None = None) -> dict:
    """His game behind the plate, for the top-left card (every pitch he caught, balls in play included)."""
    d = pid_pitches if pid_pitches is not None else pd.DataFrame()
    inn = set()
    if not d.empty:
        inn = set(zip(_num(d, "inning"), _txt(d, "inning_topbot")))
    ev = p["play"].astype(str) if "play" in p else pd.Series(dtype=str)
    cnt = lambda s: int(ev.str.contains(s, regex=False).sum())
    return {"Pitches": int(len(d)) if not d.empty else int(len(p)), "Innings": len(inn),
            "Pitchers": int(p["pitcher_id"].nunique()) if "pitcher_id" in p else 0,
            "SB": cnt("SB "), "CS": cnt("CS "), "WP": cnt("Wild pitch"), "PB": cnt("Passed ball"),
            "ABS": (f'{int(p["abs"].eq("Successful").sum())}/{int(p["abs"].ne("").sum())}'
                    if "abs" in p and p["abs"].ne("").any() else "0")}


# ----------------------------------------------------------------------------- hover comps for the game-stat tiles
def _per(tot: dict, g: float, k: str):
    return tot.get(k, np.nan) / g if g else np.nan


def _rate(a, b):
    return a / b if b else np.nan


def comp_values(fr: pd.DataFrame, fl: pd.DataFrame) -> dict:
    """Season-file totals -> the same numbers the tiles show (counts PER GAME caught; rates pooled)."""
    out = {}
    if fr is not None and not fr.empty:
        g = fr[["game_pk", "catcher"]].drop_duplicates().shape[0] if "game_pk" in fr else fr["game_date"].nunique()
        t = fr[[c for c in ("n", "strikes", "exp", "runs") if c in fr]].sum().to_dict()
        from models.framing import EDGE_REGIONS
        e = fr[fr["region"].isin(EDGE_REGIONS)][[c for c in ("n", "strikes", "exp") if c in fr]].sum().to_dict()
        out["framing"] = {"Takes": _per(t, g, "n"), "Called strikes": _per(t, g, "strikes"),
                          "Expected strikes": _per(t, g, "exp"),
                          "Extra strikes": (t.get("strikes", 0) - t.get("exp", 0)) / g if g else np.nan,
                          "Framing runs": _per(t, g, "runs"), "Shadow takes": _per(e, g, "n"),
                          "Shadow strike %": _rate(e.get("strikes", 0), e.get("n", 0)),
                          "Expected": _rate(e.get("exp", 0), e.get("n", 0)), "_games": g}
    if fl is not None and not fl.empty and "c_n" in fl:
        w = fl[fl["pos"].eq(2)] if "pos" in fl else fl
        g = w[["game_pk", "player"]].drop_duplicates().shape[0] if "game_pk" in w else w["game_date"].nunique()
        t = w.select_dtypes("number").sum().to_dict()
        x = lambda k: t.get(k, 0.0)
        out["blocking"] = {"Opportunities": _per(t, g, "c_n"), "Genuine (5%+)": _per(t, g, "g_n"),
                           "WP + PB": _per(t, g, "c_act"), "Expected WP + PB": _per(t, g, "c_exp"),
                           "Blocks above avg": (x("c_exp") - x("c_act")) / g if g else np.nan,
                           "Blocking runs": _per(t, g, "c_runs"),
                           "Tough blocks": _rate(x("tough_blk"), x("tough_n")), "_games": g}
        out["throwing"] = {"Steal attempts": _per(t, g, "t_n"), "Caught": _per(t, g, "t_cs"),
                           "Stolen": (x("t_n") - x("t_cs")) / g if g else np.nan,
                           "Expected CS": _per(t, g, "t_exp"),
                           "CS above exp": (x("t_cs") - x("t_exp")) / g if g else np.nan,
                           "Throwing runs": _per(t, g, "t_runs"), "CS %": _rate(x("t_cs"), x("t_n")),
                           "Expected CS %": _rate(x("t_exp"), x("t_n")), "_games": g}
    f_, b_, t_ = (out.get(k, {}) for k in ("framing", "blocking", "throwing"))
    if f_ or b_ or t_:
        fr_, br_, tr_ = f_.get("Framing runs", np.nan), b_.get("Blocking runs", np.nan), t_.get("Throwing runs", np.nan)
        out["total"] = {"Framing runs": fr_, "Blocking runs": br_, "Throwing runs": tr_,
                        "Catcher runs": np.nansum([fr_, br_, tr_]) if any(v == v for v in (fr_, br_, tr_)) else np.nan,
                        "_games": f_.get("_games") or b_.get("_games")}
    return out


# how each tile's comp reads: count per game, signed per game, or a rate
_PCT = {"Success %", "Shadow strike %", "Expected", "Tough blocks", "CS %", "Expected CS %"}
_SIGNED = {"ABS runs", "Challenge cost", "Net ABS runs", "Extra strikes", "Framing runs", "Blocks above avg", "Blocking runs", "CS above exp", "Throwing runs",
           "Catcher runs"}


def _show(label: str, v) -> str:
    if v is None or v != v:
        return "–"
    if label in _PCT:
        return f"{100 * v:.0f}%"
    if label in _SIGNED:
        s = f"{v:+.2f}"
        return "0.00" if s in ("+0.00", "-0.00") else s
    return f"{v:.1f}"


def _abs_file(season: int) -> pd.DataFrame:
    """The season's challenges: the built ABS table (has runs + what counts) if it's there, else the raw file."""
    from core import abs_calls, abs_stats
    t = abs_stats.load(season)[0]
    if not t.empty:
        return t[t["counted"]]
    f = abs_calls.RAW / f"abs_{season}.csv.gz"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def abs_values(fl: pd.DataFrame | None, ab: pd.DataFrame | None, pid: int | None) -> dict:
    """ABS challenges made BY the catcher, per game caught (pid None = every MLB catcher; games from the fielding
    file, pos 2)."""
    if fl is None or fl.empty or ab is None or ab.empty or "challenger" not in ab:
        return {}
    w = fl[fl["pos"].eq(2)] if "pos" in fl else fl
    games = w[["game_pk", "player"]].drop_duplicates()
    if pid is not None:
        games = games[pd.to_numeric(games["player"], errors="coerce").eq(pid)]
    if games.empty:
        return {}
    a = ab.assign(challenger=pd.to_numeric(ab["challenger"], errors="coerce"))
    if "role" in a:
        a = a[a["role"].eq("Catching")]
    a = a.merge(games, left_on=["game_pk", "challenger"], right_on=["game_pk", "player"], how="inner")
    if pid is not None:
        a = a[a["challenger"].eq(pid)]
    g, n = len(games), len(a)
    s = int(a["overturned"].astype(bool).sum()) if n else 0
    tot = lambda c: float(pd.to_numeric(a[c], errors="coerce").fillna(0).sum()) / g if c in a else np.nan
    runs, cost = tot("runs"), tot("cost")
    return {"abs": {"Challenges": n / g, "Successful": s / g, "Failed": (n - s) / g,
                    "Success %": s / n if n else np.nan, "ABS runs": runs, "Challenge cost": -cost,
                    "Net ABS runs": runs - cost, "_games": g}}


def comps(season: int, pid: int, day: str, fr: pd.DataFrame | None = None, fl: pd.DataFrame | None = None,
          ab: pd.DataFrame | None = None) -> dict:
    """metric -> tile label -> [[who, value], ...] for Season / Last 30 days / MLB (per game caught; rates pooled).
    Through this game's date; MLB = every catcher, whole season file."""
    from core import league_ref
    fr = league_ref.load_catching(season) if fr is None else fr
    fl = league_ref.load_fielding(season) if fl is None else fl
    if (fr is None or fr.empty) and (fl is None or fl.empty):
        return {}
    d30 = (pd.Timestamp(day) - pd.Timedelta(days=29)).date().isoformat()

    def cut(df, who, lo):
        if df is None or df.empty:
            return df
        m = df["game_date"].astype(str).str[:10].between(lo, day)
        if who is not None and who in df:
            m &= pd.to_numeric(df[who], errors="coerce").eq(pid)
        return df[m]
    ab = _abs_file(season) if ab is None else ab
    sets = [("Season", {**comp_values(cut(fr, "catcher", "0000"), cut(fl, "player", "0000")),
                        **abs_values(cut(fl, "player", "0000"), ab, pid)}),
            ("Last 30 days", {**comp_values(cut(fr, "catcher", d30), cut(fl, "player", d30)),
                              **abs_values(cut(fl, "player", d30), ab, pid)}),
            ("MLB avg", {**comp_values(fr, fl), **abs_values(fl, ab, None)})]
    out = {}
    for metric in ("framing", "blocking", "throwing", "total", "abs"):
        labels = {k for _, s in sets for k in s.get(metric, {}) if not k.startswith("_")}
        out[metric] = {lab: [[who, _show(lab, s.get(metric, {}).get(lab))] for who, s in sets] for lab in labels}
        out[metric]["_games"] = [[who, int(s.get(metric, {}).get("_games") or 0)] for who, s in sets[:2]]
    return out
