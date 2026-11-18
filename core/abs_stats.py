"""ABS challenges by player (2026+), valued the way Tango lays out (opportunity space + the option value of a
challenge), with run values from the full game state.

Run values      RE288: runs to the end of the inning from every base / out / count state, from the season's own
                pitches. A ball vs a strike in a state = (what a ball leads to) - (what a strike leads to): ball 4 forces
                runners (a run in with the bases loaded), strike 3 is an out (the inning over with two out).
Missed calls    every called pitch the umpire got wrong by the ABS zone (his ORIGINAL call): a ball that touched the zone
                (against the pitcher + catcher) or a called strike outside it (against the hitter). Each = an
                opportunity worth its run value -- unless his team had no challenges left.
Overturns +/-   his overturn runs - what an AVERAGE player in his role would have overturned from his missed calls:
                each missed call x the league's capture rate (runs overturned / runs available) for that role, by how
                far off the call was (<0.5", 0.5-1", 1-2", 2"+) and challenges left (1 / 2).
Lost cost       a lost challenge costs its option value: outs left in the game x a per-out rate for the challenges his
                team had left (2 or 1), measured from the season (future overturn runs teams actually got with 2 left
                vs 1 vs 0). Losing one in the 9th is nearly free; burning the last one in the 2nd isn't.
Cost +/-        what an average player in his role would have lost over his called pitches - his lost cost.
ABS +/-         Overturns +/- + Cost +/-  (runs above average). Graded on this.
Not counted     the last pitch of the game, the call stood, and it was right by more than 3" -- left out of the won /
                lost record (costs ~nothing anyway: no outs left).
Margin          inches the ball was outside the ABS zone (+) or inside it (-): the plate's edge / the hitter's zone top
                + bottom, plus the ball's radius (any part of the ball touching = strike).

    python -m core.abs_stats 2026        # reads data/raw/statcast_2026.csv.gz + the ABS challenge file
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEAGUE = ROOT / "data" / "league"
ROLES = {"Hitting": "batter", "Catching": "fielder_2", "Pitching": "pitcher"}
FINAL_MARGIN_IN = 3.0
BALL_R_IN = 1.45
MIN_GRADE = 5                     # counted challenges before a player's rates go into the MLB spread
MIN_GAMES = 20                    # games in the role before his runs / game go into the spread
ABS_VERSION = 2                   # bump when the valuation changes -> the data update rebuilds the tables
MARGIN_BINS = [0.5, 1.0, 2.0]     # how far off a missed call was (in), for the capture rates
DEFAULT_PER_OUT = {2: 0.001, 1: 0.003}   # lost-challenge cost per out left (Tango) until the season sets its own
# fallback count values (runs, ball minus called strike) -- replaced by the season's own when it's built
DEFAULT_VALUES = {"0-0": .087, "1-0": .108, "2-0": .145, "3-0": .175, "0-1": .072, "1-1": .092, "2-1": .132,
                  "3-1": .245, "0-2": .118, "1-2": .162, "2-2": .261, "3-2": .555}
COLS = ["game_date", "game_pk", "at_bat_number", "pitch_number", "challenger", "role", "overturned", "result",
        "counted", "final_pitch", "margin_in", "from_call", "to_call", "value", "runs", "balls", "strikes", "inning",
        "inning_topbot", "outs_when_up", "batter", "pitcher", "fielder_2", "stand", "p_throws", "pitch_type",
        "release_speed", "plate_x", "plate_z", "sz_top", "sz_bot", "home_team", "away_team", "play_id", "team",
        "outs_left", "left", "cost"]
MISS_COLS = ["game_date", "game_pk", "at_bat_number", "pitch_number", "against", "batter", "pitcher", "fielder_2",
             "team", "orig_call", "margin_in", "value", "left", "bucket", "challenged_role", "overturned", "balls",
             "strikes", "inning", "inning_topbot", "pitch_type", "release_speed", "plate_x", "plate_z", "sz_top",
             "sz_bot", "home_team", "away_team"]


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)


def margin_in(df: pd.DataFrame) -> pd.Series:
    """Inches the ball's centre is outside the ABS zone's touching edge (+), or inside its nearest edge (-)."""
    px, pz = _n(df, "plate_x") * 12, _n(df, "plate_z") * 12
    top, bot = _n(df, "sz_top") * 12, _n(df, "sz_bot") * 12
    x = px.abs() - (8.5 + BALL_R_IN)
    up, down = pz - (top + BALL_R_IN), (bot - BALL_R_IN) - pz
    dz = np.maximum(up, down)
    outside = np.hypot(np.clip(x, 0, None), np.clip(dz, 0, None))
    inside = np.maximum(x, dz)
    return pd.Series(np.where(outside > 0, outside, inside), index=df.index).where(px.notna() & pz.notna())


def count_values(raw: pd.DataFrame) -> dict:
    """'b-s' -> runs a ball is worth over a called strike in that count (batting side), from the season itself."""
    if raw is None or raw.empty or "delta_run_exp" not in raw:
        return dict(DEFAULT_VALUES)
    d = raw[raw["description"].astype(str).isin(["ball", "blocked_ball", "called_strike"])]
    cnt = _n(d, "balls").astype("Int64").astype(str) + "-" + _n(d, "strikes").astype("Int64").astype(str)
    rv = _n(d, "delta_run_exp")
    strike = d["description"].astype(str).eq("called_strike")
    out = {}
    for c in DEFAULT_VALUES:
        m = cnt.eq(c)
        b, s = rv[m & ~strike].mean(), rv[m & strike].mean()
        out[c] = round(abs(float(b - s)), 4) if (m & ~strike).sum() > 50 and (m & strike).sum() > 50 else DEFAULT_VALUES[c]
    return out


def value_of(balls, strikes, values: dict | None = None) -> pd.Series:
    v = values or DEFAULT_VALUES
    key = pd.Series(balls).astype("Int64").astype(str) + "-" + pd.Series(strikes).astype("Int64").astype(str)
    return key.map(v).astype(float)


def _final_pitch(raw: pd.DataFrame) -> pd.Series:
    """True on each game's last pitch."""
    ab, pn = _n(raw, "at_bat_number"), _n(raw, "pitch_number")
    order = ab * 1000 + pn
    return order.eq(order.groupby(raw["game_pk"]).transform("max"))


def _bases(df: pd.DataFrame) -> pd.Series:
    on = lambda c: df[c].notna() if c in df else pd.Series(False, index=df.index)
    return on("on_1b").astype(int) + 2 * on("on_2b").astype(int) + 4 * on("on_3b").astype(int)


# ----------------------------------------------------------------------------- run values by game state
def re288(raw: pd.DataFrame) -> dict:
    """'base|outs|balls|strikes' -> average runs to the end of the inning, from every pitch of the season."""
    need = {"post_bat_score", "bat_score", "game_pk", "inning", "inning_topbot", "outs_when_up", "balls", "strikes"}
    if raw is None or raw.empty or not need <= set(raw):
        return {}
    half = [raw["game_pk"], raw["inning"], raw["inning_topbot"].astype(str)]
    end = _n(raw, "post_bat_score").groupby(half).transform("max")
    to_end = end - _n(raw, "bat_score")
    key = (_bases(raw).astype(str) + "|" + _n(raw, "outs_when_up").astype("Int64").astype(str) + "|"
           + _n(raw, "balls").clip(upper=3).astype("Int64").astype(str) + "|"
           + _n(raw, "strikes").clip(upper=2).astype("Int64").astype(str))
    g = to_end.groupby(key).agg(["mean", "count"])
    return {k: round(float(r["mean"]), 4) for k, r in g.iterrows() if r["count"] >= 30}


def _walk(b: int) -> tuple[int, int]:
    """(bases after a walk, runs in)."""
    if not b & 1:
        return b | 1, 0
    if not b & 2:
        return b | 3, 0
    if not b & 4:
        return 7, 0
    return 7, 1


def state_value(b: int, o: int, balls: int, strikes: int, re: dict) -> float | None:
    """Runs a ball is worth over a strike (batting side) in this base / out / count state."""
    R = lambda bb, oo, ba, st: 0.0 if oo >= 3 else re.get(f"{bb}|{oo}|{ba}|{st}")
    if balls < 3:
        vb = R(b, o, balls + 1, strikes)
    else:
        nb, runs = _walk(b)
        r = R(nb, o, 0, 0)
        vb = None if r is None else runs + r
    vs = R(b, o, balls, strikes + 1) if strikes < 2 else R(b, o + 1, 0, 0)
    if vb is None or vs is None:
        return None
    return float(vb - vs)


def call_values(df: pd.DataFrame, re: dict | None, values: dict | None = None) -> pd.Series:
    """Per pitch: a ball minus a strike in its full game state (RE288); the count value where a state is thin."""
    cnt = value_of(_n(df, "balls"), _n(df, "strikes"), values)
    cnt.index = df.index
    if not re:
        return cnt
    b, o = _bases(df), _n(df, "outs_when_up")
    ba, st = _n(df, "balls").clip(upper=3), _n(df, "strikes").clip(upper=2)
    cache, out = {}, []
    for k in zip(b, o, ba, st):
        if any(pd.isna(x) for x in k):
            out.append(np.nan)
            continue
        k = tuple(int(x) for x in k)
        if k not in cache:
            cache[k] = state_value(*k, re)
        out.append(np.nan if cache[k] is None else cache[k])
    v = pd.Series(out, index=df.index, dtype=float)
    return v.fillna(cnt).clip(lower=0)


# ----------------------------------------------------------------------------- challenges left / outs left
def outs_left(df: pd.DataFrame) -> pd.Series:
    """Outs left in the game at this pitch (54 at the first pitch; extra innings: what's left of that inning)."""
    inn, o = _n(df, "inning"), _n(df, "outs_when_up").fillna(0)
    top = df["inning_topbot"].astype(str).eq("Top") if "inning_topbot" in df else pd.Series(True, index=df.index)
    later = np.clip(9 - inn, 0, None) * 6
    return ((3 - o) + np.where(top, 3, 0) + later).clip(lower=0)


def _teams(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(batting team, fielding team) per pitch."""
    col = lambda c: df[c].astype(str) if c in df else pd.Series("", index=df.index)
    top = col("inning_topbot").eq("Top")
    return (pd.Series(np.where(top, col("away_team"), col("home_team")), index=df.index),
            pd.Series(np.where(top, col("home_team"), col("away_team")), index=df.index))


def challenges_left(df: pd.DataFrame, lost: pd.DataFrame) -> pd.DataFrame:
    """Per pitch: challenges each side had left BEFORE it (bat_left, fld_left). Two per game, kept when overturned;
    a team out of challenges gets one at the start of each extra inning. lost = game_pk / at_bat_number /
    pitch_number / team of every LOST challenge."""
    bat, fld = _teams(df)
    out = pd.DataFrame({"bat_left": 2, "fld_left": 2}, index=df.index)
    if lost is None or lost.empty:
        return out
    games = set(pd.to_numeric(lost["game_pk"], errors="coerce").dropna().astype(int))
    gpk = _n(df, "game_pk").to_numpy()
    ab, pn, inn = _n(df, "at_bat_number").to_numpy(), _n(df, "pitch_number").to_numpy(), _n(df, "inning").to_numpy()
    bat_a, fld_a = bat.to_numpy(), fld.to_numpy()
    bl, fl = out["bat_left"].to_numpy().copy(), out["fld_left"].to_numpy().copy()
    lk = {}
    for g, a, p, t in zip(lost["game_pk"], lost["at_bat_number"], lost["pitch_number"], lost["team"]):
        lk.setdefault(int(g), {})[(int(a), int(p))] = str(t)
    rows = np.flatnonzero(np.isin(gpk, list(games)))
    if len(rows):
        order = np.lexsort((pn[rows], ab[rows], gpk[rows]))
        rows = rows[order]
        cur_g, left, last_inn, mine = None, {}, None, {}
        for i in rows:
            g = gpk[i]
            if g != cur_g:
                cur_g, left, last_inn, mine = g, {}, None, lk.get(int(g), {})
            it = inn[i]
            if it != last_inn and it == it and it >= 10:          # extra inning: a team that's out gets one
                for t in list(left):
                    left[t] = max(left[t], 1)
            last_inn = it
            bl[i] = left.get(bat_a[i], 2)
            fl[i] = left.get(fld_a[i], 2)
            if ab[i] == ab[i] and pn[i] == pn[i]:
                t = mine.get((int(ab[i]), int(pn[i])))
                if t is not None:
                    left[t] = max(left.get(t, 2) - 1, 0)
    out["bat_left"], out["fld_left"] = bl, fl
    return out


# ----------------------------------------------------------------------------- the challenges
def table(raw: pd.DataFrame, ch: pd.DataFrame, re: dict | None = None, values: dict | None = None) -> pd.DataFrame:
    """One row per ABS challenge: role, result, counted, margin, run value (RE288), team, outs + challenges left.
    (cost is added by add_cost once the season's per-out rates are known.)"""
    from core import abs_calls
    if raw is None or raw.empty or ch is None or ch.empty:
        return pd.DataFrame(columns=COLS)
    key = ["game_pk", "at_bat_number", "pitch_number"]
    games = set(pd.to_numeric(ch["game_pk"], errors="coerce").dropna().astype(int))
    r = raw[pd.to_numeric(raw["game_pk"], errors="coerce").isin(games)].copy()
    if r.empty:
        return pd.DataFrame(columns=COLS)
    for k in key:
        r[k] = pd.to_numeric(r[k], errors="coerce")
    r = r.dropna(subset=key)
    r[key] = r[key].astype(int)
    r["final_pitch"] = _final_pitch(r)
    stood = abs_calls.call_that_stood(r)
    desc = r["description"].astype(str)
    r["_stood"] = stood.fillna(desc.map({"called_strike": "S", "ball": "B", "blocked_ball": "B"}))
    c = ch.copy()
    for k in key:
        c[k] = pd.to_numeric(c[k], errors="coerce")
    c = c.dropna(subset=key)
    c[key] = c[key].astype(int)
    if "play_id" not in c:
        c["play_id"] = None
    m = c[key + ["overturned", "challenger", "play_id"]].merge(r, on=key, how="inner")
    if m.empty:
        return pd.DataFrame(columns=COLS)
    who = pd.to_numeric(m["challenger"], errors="coerce")
    m["role"] = np.select([who.eq(_n(m, "batter")), who.eq(_n(m, "fielder_2")), who.eq(_n(m, "pitcher"))],
                          ["Hitting", "Catching", "Pitching"], "Other")
    over = m["overturned"].astype(str).str.lower().isin(["true", "1", "1.0"])
    m["overturned"] = over
    m["result"] = np.where(over, "Successful", "Failed")
    m["to_call"] = m["_stood"].map({"S": "Strike", "B": "Ball"}).fillna("")
    flip = {"Strike": "Ball", "Ball": "Strike"}
    m["from_call"] = np.where(over, m["to_call"].map(flip).fillna(""), m["to_call"])
    m["margin_in"] = margin_in(m).round(1)
    right_by = np.where(m["to_call"].eq("Ball"), m["margin_in"], -m["margin_in"])     # how clearly the call was right
    m["counted"] = over | ~(m["final_pitch"].astype(bool) & (pd.Series(right_by, index=m.index) > FINAL_MARGIN_IN))
    m["value"] = call_values(m, re, values).values
    m["runs"] = np.where(over, m["value"], 0.0)
    m["challenger"] = who
    bat, fld = _teams(m)
    m["team"] = np.where(m["role"].eq("Hitting"), bat, fld)
    m["outs_left"] = outs_left(m).values
    m["game_date"] = m["game_date"].astype(str).str[:10]
    for col in COLS:
        if col not in m:
            m[col] = np.nan
    return m[COLS].sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def attach_left(tab: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Challenges left for his team at each challenge, + the per-pitch frame (for the missed calls / rates)."""
    key = ["game_pk", "at_bat_number", "pitch_number"]
    lost = tab[~tab["overturned"].astype(bool)][key + ["team"]] if not tab.empty else pd.DataFrame(columns=key + ["team"])
    games = set(pd.to_numeric(raw["game_pk"], errors="coerce").dropna().astype(int))
    st = challenges_left(raw, lost[lost["game_pk"].isin(games)] if not lost.empty else lost)
    k = raw[key].apply(pd.to_numeric, errors="coerce")
    st = pd.concat([k, st, pd.DataFrame({"outs_left": outs_left(raw)}, index=raw.index)], axis=1)
    if not tab.empty:
        t = tab.drop(columns=["left"], errors="ignore").merge(st.drop_duplicates(key), on=key, how="left",
                                                              suffixes=("", "_st"))
        t["left"] = np.where(t["role"].eq("Hitting"), t["bat_left"], t["fld_left"])
        tab = t.drop(columns=[c for c in ("bat_left", "fld_left", "outs_left_st") if c in t])
    return tab, st


def loss_rates(st: pd.DataFrame, raw: pd.DataFrame, tab: pd.DataFrame) -> dict:
    """Per-out cost of losing a challenge with 2 / 1 left, measured: at the start of every half inning, each team's
    future overturn runs vs outs left, by challenges left (a line through zero for each: runs per out). Losing one
    with 2 left costs (rate2 - rate1) per out; with 1 left, (rate1 - rate0)."""
    if tab.empty or st.empty:
        return dict(DEFAULT_PER_OUT)
    key = ["game_pk", "at_bat_number", "pitch_number"]
    bat, fld = _teams(raw)
    d = st.assign(bat=bat, fld=fld, inning=_n(raw, "inning"), half=raw["inning_topbot"].astype(str),
                  order=_n(raw, "at_bat_number") * 1000 + _n(raw, "pitch_number"))
    first = d.sort_values("order").groupby(["game_pk", "inning", "half"]).head(1)
    ov = tab[tab["overturned"].astype(bool)]
    ov = ov.assign(order=_n(ov, "at_bat_number") * 1000 + _n(ov, "pitch_number"))
    rows = []
    fut_of = {}                                   # (game, team) -> (overturn orders, runs from there to the end)
    for (g, t), v in ov.groupby(["game_pk", "team"]):
        v = v.sort_values("order")
        fut_of[(g, t)] = (v["order"].to_numpy(), v["runs"].to_numpy()[::-1].cumsum()[::-1])
    for r in first.itertuples(index=False):
        for team, left in ((r.bat, r.bat_left), (r.fld, r.fld_left)):
            f = fut_of.get((r.game_pk, team))
            fut = 0.0
            if f is not None:
                i = int(np.searchsorted(f[0], r.order))
                fut = float(f[1][i]) if i < len(f[1]) else 0.0
            rows.append((min(int(left), 2), float(r.outs_left), fut))
    df = pd.DataFrame(rows, columns=["k", "x", "y"])
    slope = {}
    for k in (0, 1, 2):
        s = df[df["k"] == k]
        slope[k] = float((s["x"] * s["y"]).sum() / (s["x"] ** 2).sum()) if len(s) >= 200 and (s["x"] ** 2).sum() else None
    if slope.get(2) is None or slope.get(1) is None:
        return dict(DEFAULT_PER_OUT)
    s0 = slope.get(0) or 0.0
    r2, r1 = slope[2] - slope[1], slope[1] - s0
    if r2 <= 0 or r1 <= 0:                       # thin / noisy: keep Tango's shape, scaled to the season
        tot = max(slope[2] - s0, 1e-4)
        r2, r1 = 0.25 * tot, 0.75 * tot
    return {2: round(r2, 7), 1: round(r1, 7), "slopes": {k: v for k, v in slope.items() if v is not None}}


def add_cost(tab: pd.DataFrame, rates: dict) -> pd.DataFrame:
    """Each LOST (counted) challenge: outs left in the game x the per-out rate for the challenges his team had left."""
    if tab.empty:
        return tab
    t = tab.copy()
    left = pd.to_numeric(t["left"], errors="coerce").fillna(2).clip(1, 2).astype(int)
    rate = left.map({2: rates.get(2, DEFAULT_PER_OUT[2]), 1: rates.get(1, DEFAULT_PER_OUT[1])})
    lost = ~t["overturned"].astype(bool) & t["counted"].astype(bool)
    t["cost"] = np.where(lost, pd.to_numeric(t["outs_left"], errors="coerce").fillna(0) * rate, 0.0)
    return t


# ----------------------------------------------------------------------------- the opportunities (missed calls)
def missed_calls(orig: pd.DataFrame, re: dict | None, st: pd.DataFrame | None, tab: pd.DataFrame,
                 values: dict | None = None) -> pd.DataFrame:
    """Every called pitch the umpire missed by the ABS zone (orig = the umpire's ORIGINAL calls): who it went
    against, its run value, how far off, the challenges that side had left, and whether / by whom it was challenged."""
    d = orig[orig["description"].astype(str).isin(["called_strike", "ball", "blocked_ball"])]
    if d.empty:
        return pd.DataFrame(columns=MISS_COLS)
    m = margin_in(d)
    strike = d["description"].astype(str).eq("called_strike")
    miss = ((strike & (m > 0)) | (~strike & (m <= 0))).fillna(False)
    d, strike, m = d[miss], strike[miss], m[miss]
    bat, fld = _teams(d)
    out = d[[c for c in MISS_COLS if c in d]].copy()
    out["against"] = np.where(strike, "Hitting", "Defense")         # a bad strike hurts the hitter
    out["team"] = np.where(strike, bat, fld)
    out["orig_call"] = np.where(strike, "Strike", "Ball")
    out["margin_in"] = m.abs().round(2)
    out["value"] = call_values(d, re, values).values
    out["bucket"] = np.digitize(out["margin_in"], MARGIN_BINS)
    if st is not None and not st.empty:
        left = np.where(strike, st.loc[d.index, "bat_left"], st.loc[d.index, "fld_left"])
        out["left"] = pd.Series(left, index=d.index).clip(upper=2)
    else:
        out["left"] = 2
    key = ["game_pk", "at_bat_number", "pitch_number"]
    for k in key:
        out[k] = pd.to_numeric(out[k], errors="coerce")
    if not tab.empty:
        cc = tab[key + ["role", "overturned"]].rename(columns={"role": "challenged_role"}).drop_duplicates(key)
        out = out.drop(columns=["challenged_role", "overturned"], errors="ignore").merge(cc, on=key, how="left")
    out["overturned"] = out.get("overturned", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["challenged_role"] = out.get("challenged_role", pd.Series("", index=out.index)).fillna("")
    out["game_date"] = out["game_date"].astype(str).str[:10] if "game_date" in out else ""
    for c in MISS_COLS:
        if c not in out:
            out[c] = np.nan
    return out[MISS_COLS].reset_index(drop=True)


def _won_by(mc: pd.DataFrame, role: str) -> pd.Series:
    return mc["overturned"].astype(bool) & mc["challenged_role"].eq(role)


def capture_rates(mc: pd.DataFrame) -> dict:
    """role -> 'bucket|left' -> share of the missed-call runs that role overturned (league), + 'all'."""
    out = {}
    for role, against in (("Hitting", "Hitting"), ("Catching", "Defense"), ("Pitching", "Defense")):
        m = mc[mc["against"].eq(against) & pd.to_numeric(mc["left"], errors="coerce").fillna(2).ge(1)]
        won = _won_by(m, role)
        v = pd.to_numeric(m["value"], errors="coerce").fillna(0)
        r = {"all": float(v[won].sum() / v.sum()) if v.sum() else 0.0}
        for (b, lf), g in m.groupby([m["bucket"], pd.to_numeric(m["left"], errors="coerce").fillna(2).clip(1, 2)]):
            vv = v.loc[g.index]
            r[f"{int(b)}|{int(lf)}"] = float(vv[won.loc[g.index]].sum() / vv.sum()) if vv.sum() else 0.0
        out[role] = r
    return out


def expected_overturns(mc: pd.DataFrame, role: str, cap: dict) -> pd.Series:
    """Per missed call: what an average player in this role would have won back from it (0 with no challenges left)."""
    r = cap.get(role, {})
    left = pd.to_numeric(mc["left"], errors="coerce").fillna(2)
    k = mc["bucket"].astype(int).astype(str) + "|" + left.clip(1, 2).astype(int).astype(str)
    rate = k.map(r).fillna(r.get("all", 0.0))
    return (pd.to_numeric(mc["value"], errors="coerce").fillna(0) * rate).where(left >= 1, 0.0)


def _role_misses(mc: pd.DataFrame, role: str) -> pd.DataFrame:
    return mc[mc["against"].eq("Hitting" if role == "Hitting" else "Defense")]


# ----------------------------------------------------------------------------- players
PL_COLS = ["player", "role", "games", "called", "challenges", "overturned", "not_counted", "runs", "cost", "misses",
           "avail", "x_runs", "x_cost"]


def players(raw: pd.DataFrame, tab: pd.DataFrame, mc: pd.DataFrame | None = None, cap: dict | None = None) -> pd.DataFrame:
    """Every player x role: games + called pitches in that role, his challenges, his missed calls (available runs)
    and what an average player would have done with them (x_runs, x_cost)."""
    mc = mc if mc is not None else pd.DataFrame(columns=MISS_COLS)
    cap = cap or {}
    called = raw[raw["description"].astype(str).isin(["called_strike", "ball", "blocked_ball"])] if "description" in raw else raw
    rows = []
    for role, col in ROLES.items():
        if col not in raw:
            continue
        agg = pd.DataFrame({"games": raw[["game_pk", col]].dropna().drop_duplicates().groupby(col).size()})
        agg["called"] = called.groupby(col).size() if col in called else 0
        t = tab[tab["role"].eq(role)] if not tab.empty else tab
        if not t.empty:
            cc = t[t["counted"].astype(bool)]
            agg["challenges"] = cc.groupby("challenger").size()
            agg["overturned"] = cc.groupby("challenger")["overturned"].sum()
            agg["not_counted"] = t[~t["counted"].astype(bool)].groupby("challenger").size()
            agg["runs"] = cc.groupby("challenger")["runs"].sum()
            agg["cost"] = cc.groupby("challenger")["cost"].sum() if "cost" in cc else 0.0
        rm = _role_misses(mc, role)
        if not rm.empty and col in rm:
            usable = pd.to_numeric(rm["left"], errors="coerce").fillna(2).ge(1)
            v = pd.to_numeric(rm["value"], errors="coerce").fillna(0)
            agg["misses"] = rm[usable].groupby(col).size()
            agg["avail"] = v[usable].groupby(rm.loc[usable, col]).sum()
            agg["x_runs"] = expected_overturns(rm, role, cap).groupby(rm[col]).sum()
        for c in ("called", "challenges", "overturned", "not_counted", "runs", "cost", "misses", "avail", "x_runs"):
            agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0.0) if c in agg else 0.0
        agg = agg.reset_index().rename(columns={col: "player", "index": "player"})
        agg["role"] = role
        rows.append(agg)
    if not rows:
        return pd.DataFrame(columns=PL_COLS)
    out = pd.concat(rows, ignore_index=True)
    out["player"] = pd.to_numeric(out["player"], errors="coerce").astype("Int64")
    # an average player's lost-challenge cost over the same called pitches (league cost per called pitch, by role)
    out["x_cost"] = 0.0
    for role, g in out.groupby("role"):
        per = g["cost"].sum() / g["called"].sum() if g["called"].sum() else 0.0
        out.loc[g.index, "x_cost"] = g["called"] * per
    return out[PL_COLS]


# ----------------------------------------------------------------------------- the numbers the profile shows
STATS = ["G", "Challenges", "Overturned", "Failed", "Overturn %", "Missed calls", "Runs available", "Capture %",
         "Overturn runs", "Overturns +/-", "Lost cost", "Cost +/-", "ABS +/-", "Not counted"]
GRADED = {"Overturn %": 1, "Capture %": 1, "Overturns +/-": 1, "Cost +/-": 1, "ABS +/-": 1}


def summarize(games: float, rows: pd.DataFrame, pl_row: dict | None = None) -> dict:
    """rows = his challenges in one role; pl_row = his players() row (missed calls + expectations)."""
    c = rows[rows["counted"].astype(bool)] if not rows.empty else rows
    n = len(c)
    o = int(c["overturned"].astype(bool).sum()) if n else 0
    runs = float(c["runs"].sum()) if n else 0.0
    cost = float(pd.to_numeric(c["cost"], errors="coerce").fillna(0).sum()) if n and "cost" in c else 0.0
    p = pl_row or {}
    avail, xr, xc = float(p.get("avail") or 0), float(p.get("x_runs") or 0), float(p.get("x_cost") or 0)
    has = bool(pl_row)
    over_pm = runs - xr if has else math.nan
    cost_pm = xc - cost if has else math.nan
    return {"G": games, "Challenges": n, "Overturned": o, "Failed": n - o, "Overturn %": o / n if n else math.nan,
            "Missed calls": int(p.get("misses") or 0) if has else math.nan, "Runs available": avail if has else math.nan,
            "Capture %": runs / avail if avail else math.nan, "Overturn runs": runs, "Overturns +/-": over_pm,
            "Lost cost": -cost, "Cost +/-": cost_pm,
            "ABS +/-": over_pm + cost_pm if has else math.nan, "Not counted": int(len(rows) - n)}


def league(pl: pd.DataFrame) -> dict:
    """role -> stat -> (MLB average per player, spread across players) -- the colour grades."""
    out = {}
    if pl is None or pl.empty:
        return out
    for role, g in pl.groupby("role"):
        g = g.copy()
        for c in PL_COLS[2:]:
            g[c] = pd.to_numeric(g.get(c, 0), errors="coerce").fillna(0) if c in g else 0.0
        g["opm"] = g["runs"] - g["x_runs"]
        g["cpm"] = g["x_cost"] - g["cost"]
        g["tot"] = g["opm"] + g["cpm"]
        per = len(g) or 1
        q = g[g["challenges"] >= MIN_GRADE]
        qa = g[g["avail"] >= 1.0]
        qg = g[g["games"] >= MIN_GAMES]
        sd = lambda s_, ok: float(s_.std()) if ok else math.nan
        n, o = g["challenges"].sum(), g["overturned"].sum()
        tot = lambda c: g[c].sum() / per
        out[role] = {
            "G": (tot("games"), math.nan), "Challenges": (tot("challenges"), math.nan),
            "Overturned": (tot("overturned"), math.nan), "Failed": ((n - o) / per, math.nan),
            "Overturn %": (o / n if n else math.nan, sd(q["overturned"] / q["challenges"], len(q) > 2)),
            "Missed calls": (tot("misses"), math.nan), "Runs available": (tot("avail"), math.nan),
            "Capture %": (g["runs"].sum() / g["avail"].sum() if g["avail"].sum() else math.nan,
                          sd(qa["runs"] / qa["avail"], len(qa) > 2)),
            "Overturn runs": (tot("runs"), math.nan),
            "Overturns +/-": (0.0, sd(qg["opm"], len(qg) > 2)), "Lost cost": (-tot("cost"), math.nan),
            "Cost +/-": (0.0, sd(qg["cpm"], len(qg) > 2)), "ABS +/-": (0.0, sd(qg["tot"], len(qg) > 2)),
            "Not counted": (tot("not_counted"), math.nan),
        }
    return out


# ----------------------------------------------------------------------------- files
def paths(season: int) -> dict:
    return {"tab": LEAGUE / f"abs_{season}.csv.gz", "players": LEAGUE / f"abs_players_{season}.csv.gz",
            "misses": LEAGUE / f"abs_misses_{season}.csv.gz", "meta": LEAGUE / f"abs_meta_{season}.json"}


def seasons() -> list[int]:
    return sorted(int(p.stem.split("_")[1].split(".")[0]) for p in LEAGUE.glob("abs_[0-9]*.csv.gz"))


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def load(season: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    p = paths(season)
    tab = pd.read_csv(p["tab"], dtype={"game_date": str, "play_id": str}) if p["tab"].exists() else pd.DataFrame(columns=COLS)
    pl = pd.read_csv(p["players"]) if p["players"].exists() else pd.DataFrame()
    meta = json.loads(p["meta"].read_text()) if p["meta"].exists() else {}
    if not tab.empty:
        tab["counted"] = _bool(tab["counted"])
        tab["overturned"] = _bool(tab["overturned"])
    return tab, pl, meta


def stale(season: int) -> bool:
    """True if the season's tables are missing or were built by an older valuation."""
    p = paths(season)
    if not (p["tab"].exists() and p["meta"].exists()):
        return True
    try:
        return json.loads(p["meta"].read_text()).get("version") != ABS_VERSION
    except Exception:
        return True


def load_misses(season: int) -> pd.DataFrame:
    f = paths(season)["misses"]
    if not f.exists():
        return pd.DataFrame(columns=MISS_COLS)
    mc = pd.read_csv(f, dtype={"game_date": str})
    mc["overturned"] = _bool(mc["overturned"])
    mc["challenged_role"] = mc["challenged_role"].fillna("")
    return mc


def save(raw: pd.DataFrame, season: int, ch: pd.DataFrame | None = None) -> dict:
    """Build + save the season's challenges, missed calls, player totals and the valuation tables."""
    from core import abs_calls
    if ch is None:
        games = pd.to_numeric(raw["game_pk"], errors="coerce").dropna().astype(int).unique().tolist()
        ch = abs_calls.season_challenges(season, games)
    values = count_values(raw)
    re = re288(raw)
    try:
        orig = abs_calls.original_calls(raw, challenges=ch)          # the umpire's calls, before any overturn
    except Exception:
        orig = raw
    tab = table(raw, ch, re, values)
    tab, st = attach_left(tab, raw)
    rates = loss_rates(st, raw, tab)
    tab = add_cost(tab, rates)
    mc = missed_calls(orig, re, st, tab, values)
    cap = capture_rates(mc)
    pl = players(raw, tab, mc, cap)
    p = paths(season)
    LEAGUE.mkdir(parents=True, exist_ok=True)
    tab.to_csv(p["tab"], index=False)
    pl.to_csv(p["players"], index=False)
    mc.to_csv(p["misses"], index=False)
    p["meta"].write_text(json.dumps({"version": ABS_VERSION, "values": values, "re288": re, "per_out": {str(k): v for k, v in rates.items()},
                                     "capture": cap, "final_margin_in": FINAL_MARGIN_IN,
                                     "missed_calls": int(len(mc)),
                                     "last_game": str(tab["game_date"].max()) if len(tab) else None}))
    by = tab["role"].value_counts().to_dict() if len(tab) else {}
    print(f"[{season}] ABS: {len(tab):,} challenges ({by}); {len(mc):,} missed calls; lost-challenge cost per out "
          f"left: {rates.get(2)} (2 left) / {rates.get(1)} (1 left)", flush=True)
    return {"challenges": len(tab), "by_role": by, "rates": rates}


def fill_play_ids(rows: pd.DataFrame, get_pbp) -> pd.DataFrame:
    """Savant video ids for challenges saved before the challenge file kept them: from each game's play-by-play."""
    if rows.empty:
        return rows
    rows = rows.copy()
    if "play_id" not in rows:
        rows["play_id"] = None
    miss = rows["play_id"].isna() | rows["play_id"].astype(str).isin(["", "nan", "None"])
    if not miss.any():
        return rows
    from concurrent.futures import ThreadPoolExecutor
    games = sorted(set(pd.to_numeric(rows.loc[miss, "game_pk"], errors="coerce").dropna().astype(int)))

    def ids(pk):
        try:
            pbp = get_pbp(pk)
        except Exception:
            return pk, {}
        out = {}
        for p in pbp.get("allPlays", []):
            ab = int(p.get("about", {}).get("atBatIndex", -1)) + 1
            for e in p.get("playEvents", []):
                if e.get("isPitch") and e.get("playId") and e.get("pitchNumber") is not None:
                    out[(ab, int(e["pitchNumber"]))] = e["playId"]
        return pk, out
    with ThreadPoolExecutor(6) as ex:
        found = dict(ex.map(ids, games))
    rows.loc[miss, "play_id"] = [found.get(int(g), {}).get((int(a), int(n)), None)
                                 for g, a, n in zip(rows.loc[miss, "game_pk"], rows.loc[miss, "at_bat_number"],
                                                    rows.loc[miss, "pitch_number"])]
    return rows


def main(argv: list[str]) -> None:
    from pipeline import build_league_reference as bl
    for a in argv or ["2026"]:
        season = int(a)
        f = bl.RAW_DIR / f"statcast_{season}.csv.gz"
        if not f.exists():
            print(f"[{season}] no raw Statcast file ({f.name}) -- run the league build first.", flush=True)
            continue
        raw = pd.read_csv(f, low_memory=False)
        save(raw, season)


if __name__ == "__main__":
    main(sys.argv[1:])
