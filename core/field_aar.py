"""Infield / Outfield AARs: every ball in play he was responsible for in one game, from the IQ fielding model
(models/fielding.py) -- the same per-play credits the profile's Defense tab adds up, so a game's numbers sum to his
season.

No starting spots, no routes: MLB doesn't publish where a fielder stood on each pitch. Each ball is judged by the ball
itself -- exit velo, launch angle, direction, flight distance, batter side:
  out chance    how often an average defense turns a ball like this into an out
  his share     on hits / errors: how often HIS position makes the out on balls like this (learned from the league,
                not from where he stood)
  credit (OPA)  out made by him: +(1 - out chance); not made: -(out chance x his share)
  runs          credit x what an out is worth vs a hit on that kind of ball
  + double plays (infield: turned vs expected) and arm (outfield: the play's run value vs expected, with runners on)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models import fielding as fm

GROUPS = {"Infield": [3, 4, 5, 6], "Outfield": [7, 8, 9]}
EVENT = {"single": "Single", "double": "Double", "triple": "Triple", "field_out": "Out", "force_out": "Force out",
         "grounded_into_double_play": "Double play", "double_play": "Double play", "fielders_choice": "Fielder's choice",
         "fielders_choice_out": "Fielder's choice", "sac_fly": "Sac fly", "sac_fly_double_play": "Sac fly DP",
         "field_error": "Error", "triple_play": "Triple play", "other_out": "Out"}
BB = {"ground_ball": "Grounder", "line_drive": "Liner", "fly_ball": "Fly ball", "popup": "Popup"}
MIN_SHARE = 0.10          # a hit / error is listed as his play when his position carries >= 10% of it


def _n(d: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)


def _t(d: pd.DataFrame, c: str) -> pd.Series:
    return d[c].astype(str).where(d[c].notna(), "") if c in d else pd.Series("", index=d.index)


def direction(spray: float) -> str:
    """Field direction from the spray angle (degrees, + = right field), as fans say it."""
    if not np.isfinite(spray):
        return ""
    for lo, name in ((-99, "LF line"), (-34, "LF"), (-20, "LCF"), (-7, "CF"), (7, "RCF"), (20, "RF"), (34, "RF line")):
        nxt = {-99: -34, -34: -20, -20: -7, -7: 7, 7: 20, 20: 34, 34: 99}[lo]
        if lo <= spray < nxt:
            return name
    return ""


def stars(p_out: float) -> int:
    for k, lo, hi in fm.STARS:
        if lo <= p_out < hi:
            return k
    return 1


def credits(sc: pd.DataFrame, pid: int, view: str, bundle: dict | None = None) -> dict:
    """{'plays', 'dp', 'arm'} DataFrames for one player in one game at the view's positions (empty without the
    fielding model)."""
    b = bundle or fm.load()
    empty = {"plays": pd.DataFrame(), "dp": pd.DataFrame(), "arm": pd.DataFrame(), "tot": {}}
    if b is None or sc is None or sc.empty:
        return empty
    pos = GROUPS[view]
    d = sc.reset_index(drop=True)
    pc = fm.play_credits(d, b)
    if pc.empty or "row" not in pc:
        return empty
    pc = pc[pc["player"].eq(int(pid)) & pc["pos"].isin(pos)]
    opa = pc[pc["opa"].notna()] if "opa" in pc else pc.iloc[0:0]
    # his plays: outs he made, or balls not turned into outs where his position carried at least MIN_SHARE of the
    # responsibility; skip the rest (outs someone else made, balls that were barely his)
    tot = {k: float(opa[k].sum()) for k in ("opa", "runs", "rng", "conv") if k in opa}   # every credit, listed or not
    share = -opa["opa"] / opa["p_out"].clip(lower=1e-6)
    opa = opa[(opa["made"] == 1) | ((opa["opa"] < 0) & (share >= MIN_SHARE))].copy()
    dp = pc[pc.get("dp_opps", pd.Series(0, index=pc.index)).fillna(0) > 0].copy() if "dp_opps" in pc else pc.iloc[0:0]
    arm = pc[pc.get("arm_opps", pd.Series(0, index=pc.index)).fillna(0) > 0].copy() if "arm_opps" in pc else pc.iloc[0:0]
    return {"plays": _rows(d, opa, "play"), "dp": _rows(d, dp, "dp"), "arm": _rows(d, arm, "arm"), "tot": tot}


def _rows(d: pd.DataFrame, pc: pd.DataFrame, kind: str) -> pd.DataFrame:
    if pc.empty:
        return pd.DataFrame()
    r = d.loc[pc["row"].astype(int).values]
    f = fm.features(r)
    ev = _t(r, "events")
    on = [", ".join(b for b, c in (("1B", "on_1b"), ("2B", "on_2b"), ("3B", "on_3b")) if pd.notna(x.get(c))) or "Empty"
          for x in r.to_dict("records")]
    loc = _n(r, "hit_location")
    out = pd.DataFrame({
        "ab": _n(r, "at_bat_number").astype("Int64").values, "pn": _n(r, "pitch_number").astype("Int64").values,
        "inning": _n(r, "inning").astype("Int64").values,
        "half": np.where(_t(r, "inning_topbot").eq("Top"), "▲", "▼"),
        "outs": _n(r, "outs_when_up").fillna(0).astype(int).values, "on": on,
        "batter_id": _n(r, "batter").astype("Int64").values, "pitcher_id": _n(r, "pitcher").astype("Int64").values,
        "stand": _t(r, "stand").values,
        "ev": _n(r, "launch_speed").round(1).values, "la": _n(r, "launch_angle").round(0).values,
        "bb": _t(r, "bb_type").map(BB).fillna("").values,
        "spray": f["spray"].round(1).values, "dir": [direction(s) for s in f["spray"]],
        "dist": _n(r, "hit_distance_sc").round(0).values,
        "hx": ((_n(r, "hc_x") - 125.42) * fm.FT_PER_UNIT).round(1).values,
        "hy": ((198.27 - _n(r, "hc_y")) * fm.FT_PER_UNIT).round(1).values,
        "result": [EVENT.get(e, e.replace("_", " ").capitalize()) +
                   (f" ({fm.POS.get(int(h), '')})" if pd.notna(h) and int(h) in fm.POS else "")
                   for e, h in zip(ev, loc)],
        "des": _t(r, "des").values, "pos": pc["pos"].map(fm.POS).values,
    })
    if kind == "play":
        p_out = pc["p_out"].to_numpy(float)
        made = pc["made"].to_numpy(float) == 1
        share = np.where(made, 1.0, np.where(p_out > 1e-6, -pc["opa"].to_numpy(float) / np.maximum(p_out, 1e-6), 0.0))
        out = out.assign(p_out=np.round(p_out, 3), share=np.round(np.clip(share, 0, 1), 2), made=made,
                         opa=pc["opa"].round(3).values, runs=pc["runs"].round(3).values,
                         rng=pc["rng"].round(3).values, conv=pc["conv"].round(3).values,
                         stars=[stars(x) for x in p_out],
                         kind=np.where(made, "Out", np.where(ev.str.contains("error").values, "Error", "Hit")))
    elif kind == "dp":
        out = out.assign(p_dp=pc["p_dp"].round(3).values, dp=pc["dp"].astype(bool).values,
                         runs=pc["dp_runs"].round(3).values)
    else:
        out = out.assign(exp=pc["arm_exp"].round(3).values, act=pc["arm_act"].round(3).values,
                         runs=pc["arm_runs"].round(3).values)
    return out.sort_values(["ab"]).reset_index(drop=True).assign(n=lambda x: np.arange(1, len(x) + 1))


def attach(c: dict, feed: dict | None) -> dict:
    """Names + video ids from the game feed."""
    if not feed:
        return c
    from core.aar import names, play_ids
    vid, who = play_ids(feed), names(feed)
    for k, df in c.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        df["video"] = [vid.get((int(a), int(p)), "") if pd.notna(a) and pd.notna(p) else "" for a, p in zip(df["ab"], df["pn"])]
        df["batter"] = df["batter_id"].map(lambda x: who.get(int(x), "") if pd.notna(x) else "")
        df["pitcher"] = df["pitcher_id"].map(lambda x: who.get(int(x), "") if pd.notna(x) else "")
    return c


def line(c: dict, view: str) -> dict:
    """The game's totals for the tiles."""
    p, dp, arm = c["plays"], c["dp"], c["arm"]
    has = p is not None and not p.empty
    tot = c.get("tot") or {}                     # every credit, incl. plays too small a share to list
    out = {"plays": int(len(p)) if has else 0, "outs": int(p["made"].sum()) if has else 0,
           **{k: float(tot.get(k, p[k].sum() if has else 0.0)) for k in ("opa", "runs", "rng", "conv")},
           "dp_n": int(len(dp)) if dp is not None and not dp.empty else 0,
           "dp_made": int(dp["dp"].sum()) if dp is not None and not dp.empty else 0,
           "dp_exp": float(dp["p_dp"].sum()) if dp is not None and not dp.empty else 0.0,
           "dp_runs": float(dp["runs"].sum()) if dp is not None and not dp.empty else 0.0,
           "arm_n": int(len(arm)) if arm is not None and not arm.empty else 0,
           "arm_runs": float(arm["runs"].sum()) if arm is not None and not arm.empty else 0.0}
    out["total"] = out["runs"] + (out["dp_runs"] if view == "Infield" else out["arm_runs"])
    if has:
        made = p[p["made"]]
        best = made.loc[made["p_out"].idxmin()] if len(made) else None
        out["best"] = {"p_out": float(best["p_out"]), "inning": int(best["inning"]), "n": int(best["n"])} if best is not None else None
        miss = p[~p["made"] & (p["p_out"] * p["share"] >= 0.5)]
        out["miss"] = int(len(miss))
    else:
        out["best"], out["miss"] = None, 0
    return out


# ----------------------------------------------------------------------------- hover comparisons
KEYS = {"Infield": [("OPA", "Outs vs expected"), ("RS_FIELD", "Fielding runs"), ("DP_RS", "Double-play runs"),
                    ("DRS", "Total runs saved"), ("_N", "Plays")],
        "Outfield": [("OPA", "Outs vs expected"), ("RS_RANGE", "Range runs"), ("RS_ARM", "Arm runs"),
                     ("DRS", "Total runs saved"), ("_N", "Plays")]}


def comps(season: int, pid: int, day: str, view: str, daily: pd.DataFrame | None = None) -> dict:
    """label -> [[Season, v], [Last 30 days, v], [MLB avg, v]] per game at these positions, through this game's
    date (MLB = every player at them, whole season file)."""
    from core import league_ref
    daily = league_ref.load_fielding(season) if daily is None else daily
    if daily is None or daily.empty:
        return {}
    pos = GROUPS[view]
    day = str(day)[:10]
    d30 = (pd.Timestamp(day) - pd.Timedelta(days=29)).date().isoformat()
    dd = daily[daily["pos"].isin(pos)]

    def per_game(w: pd.DataFrame, who: int | None) -> dict:
        if who is not None:
            w = w[w["player"].eq(who)]
        w = w[w["outs"] > 0] if "outs" in w else w
        if w.empty:
            return {}
        g = w.groupby(["player", "game_pk"]).ngroups
        s = fm.summarize(w, "0000", "9999", pos)
        tot = {k: sum(v.get(k, 0) or 0 for v in s.values()) for k, _ in KEYS[view]}
        return {k: v / g for k, v in tot.items()} | {"_games": g}
    sets = [("Season", per_game(dd[dd["game_date"].between("0000", day)], int(pid))),
            ("Last 30 days", per_game(dd[dd["game_date"].between(d30, day)], int(pid))),
            ("MLB avg", per_game(dd, None))]
    out = {}
    for k, lab in KEYS[view]:
        out[lab] = [[who, None if not s or s.get(k) is None else round(float(s[k]), 2)] for who, s in sets]
    out["_games"] = [[who, int(s.get("_games") or 0)] for who, s in sets[:2]]
    return out
