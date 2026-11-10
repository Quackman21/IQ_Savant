"""IQ fielding: DRS and its parts from public Statcast pitch data (any period).

Per ball in play (fielders known from Statcast's fielder_2..fielder_9 columns):
  out chance   how often an average defense turns a ball like this into an out (exit velo, launch angle,
               field direction, distance from home, batter side). Where the ball went is a FAIR input here --
               it's exactly what the fielder is judged on (not like xwOBA, where it leaked luck to the hitter).
  responsible  outs: the fielder who made the play. Hits / errors: shared among positions by WHO USUALLY
               MAKES THE OUT on balls like this (a grounder through the 5.5 hole is on the SS / 3B, not the LF
               who picked it up).
  OPA / CPA    out made: +(1 - out chance) to that fielder; not made: -(out chance x his share)
  RS Field / RS Range   OPA x run value of an out vs a hit on that kind of ball (learned from the data)
  FPA          fielded by him (1 / 0) - chance he'd be the one to field it
  DP runs      ground balls, runner on 1st, < 2 outs, fielded by an infielder: (DP - DP chance) x DP run value
  RS Arm       balls fielded by an outfielder with runners on: expected run value of the play - actual
               (Statcast's run value already includes runners held / extra bases / outfield assists)
  Stars        plays bucketed by out (catch) chance: 5 star = under 25%, 1 star = 90%+

Totals: infield DRS = RS Field + DP runs; outfield DRS = RS Range + RS Arm; catcher DRS = framing + blocking
+ throwing. DRS / Season = per SEASON_INN innings.
Batter speed is NOT used (USE_SPEED = False, your xwOBA rule) -- one line to change if you ever want it.

Era models (2015-2022 shift era, 2023+ shift ban). Trained by  python -m pipeline.train_fielding
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "fielding.joblib"
SHIFT_BAN = 2023
SEASON_INN = 1200
USE_SPEED = False
FT_PER_UNIT = 2.5
POS = {3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}
INFIELD, OUTFIELD = {3, 4, 5, 6}, {7, 8, 9}
FEATURES = ["ev", "la", "spray", "dist", "is_left"]
ARM_FEATURES = FEATURES + ["fielder", "ev_type", "on1", "on2", "on3", "outs"]
OUTS = {"field_out", "force_out", "grounded_into_double_play", "double_play", "fielders_choice_out",
        "sac_fly", "sac_fly_double_play", "triple_play", "other_out", "fielders_choice"}
NOT_FIELDABLE = {"home_run", "sac_bunt", "sac_bunt_double_play", "catcher_interf"}
DP = {"grounded_into_double_play", "double_play"}
EV_TYPE = {"single": 0, "double": 1, "triple": 2, "field_out": 3, "sac_fly": 4, "force_out": 5,
           "fielders_choice": 6, "fielders_choice_out": 6, "field_error": 7, "double_play": 8,
           "sac_fly_double_play": 8, "grounded_into_double_play": 8}
STARS = [(5, 0.0, 0.25), (4, 0.25, 0.5), (3, 0.5, 0.75), (2, 0.75, 0.9), (1, 0.9, 1.01)]


def _c(df, n):
    return pd.to_numeric(df[n], errors="coerce") if n in df else pd.Series(np.nan, index=df.index)


def features(df: pd.DataFrame) -> pd.DataFrame:
    x, y = _c(df, "hc_x") - 125.42, 198.27 - _c(df, "hc_y")
    ev = df["events"].astype(str) if "events" in df else pd.Series("", index=df.index)
    return pd.DataFrame({
        "ev": _c(df, "launch_speed"), "la": _c(df, "launch_angle"),
        "spray": np.degrees(np.arctan2(x, y)) * 0.75, "dist": np.hypot(x, y) * FT_PER_UNIT,
        "is_left": (df["stand"].astype(str) == "L").astype(float) if "stand" in df else 0.0,
        "fielder": _c(df, "hit_location"), "ev_type": ev.map(EV_TYPE).astype(float),
        "on1": df["on_1b"].notna().astype(float) if "on_1b" in df else 0.0,
        "on2": df["on_2b"].notna().astype(float) if "on_2b" in df else 0.0,
        "on3": df["on_3b"].notna().astype(float) if "on_3b" in df else 0.0,
        "outs": _c(df, "outs_when_up"),
    }, index=df.index)


def balls_in_play(df: pd.DataFrame) -> pd.Series:
    """Fieldable balls in play with a location (no homers, bunts, catcher interference)."""
    ev = df["events"].astype(str)
    bunt = df["des"].astype(str).str.contains("bunt", case=False, na=False) if "des" in df else False
    return (df["type"].astype(str).eq("X") & ~ev.isin(NOT_FIELDABLE) & ~bunt
            & _c(df, "hc_x").notna() & _c(df, "launch_speed").notna())


def player_at(d: pd.DataFrame, pos: pd.Series) -> pd.Series:
    """Player id at the given position number on each row (fielder_3 .. fielder_9)."""
    out = pd.Series(np.nan, index=d.index)
    for k in range(2, 10):
        col = f"fielder_{k}"
        if col in d:
            m = pos.eq(k)
            out[m] = _c(d, col)[m]
    return out


def run_bucket(la: pd.Series) -> pd.Series:
    return pd.cut(la, [-1e9, 10, 25, 50, 1e9], labels=["GB", "LD", "FB", "PU"]).astype(str)


_cache: dict = {}


def load() -> dict | None:
    if "m" not in _cache:
        if not ARTIFACT.exists():
            return None
        import joblib
        _cache["m"] = joblib.load(ARTIFACT)
    return _cache["m"]


def _bundles(df: pd.DataFrame, b: dict):
    if "eras" not in b:
        return [(pd.Series(True, index=df.index), b)]
    yr = pd.to_datetime(df["game_date"]).dt.year
    return [(m, b["eras"][e]) for e, m in (("pre", yr < b["split"]), ("post", yr >= b["split"])) if b["eras"].get(e)]


# ----------------------------------------------------------------------------
# Per-play credits
# ----------------------------------------------------------------------------
def play_credits(df: pd.DataFrame, b: dict) -> pd.DataFrame:
    """One row per (ball in play, responsible position): player, pos, opa, runs, fpa, star bucket, ..."""
    rows = []
    for m, sub in _bundles(df, b):
        d = df[m & balls_in_play(df)]
        if d.empty:
            continue
        f = features(d)
        ev = d["events"].astype(str)
        out = ev.isin(OUTS).astype(float)
        p_out = pd.Series(sub["out"].predict_proba(f[FEATURES])[:, 1], index=d.index)
        who = pd.DataFrame(sub["who"].predict_proba(f[FEATURES]), index=d.index, columns=sub["who"].classes_)
        fld = pd.DataFrame(sub["fielded"].predict_proba(f[FEATURES]), index=d.index, columns=sub["fielded"].classes_)
        rv = run_bucket(f["la"]).map(sub["run_value"]).fillna(np.mean(list(sub["run_value"].values())))
        made_by = f["fielder"]
        for k in sorted(POS):
            share = who.get(k, pd.Series(0.0, index=d.index))
            mine = (out == 1) & (made_by == k)
            opa = np.where(mine, 1 - p_out, np.where(out == 0, -p_out * share, 0.0))
            fpa = (made_by == k).astype(float) - fld.get(k, pd.Series(0.0, index=d.index))
            player = _c(d, f"fielder_{k}")
            primary = who.idxmax(axis=1).eq(k) | mine                    # this play is "his" for stars
            keep = (np.abs(opa) > 1e-9) | (np.abs(fpa) > 1e-3) | primary
            if not keep.any():
                continue
            r = pd.DataFrame({"game_date": d["game_date"].astype(str).str[:10], "game_pk": d.get("game_pk", 0),
                              "player": player, "pos": k, "opa": opa, "runs": opa * rv, "fpa": fpa,
                              "primary": primary.astype(float), "p_out": p_out, "out": out,
                              "made": mine.astype(float)}, index=d.index)[keep]
            rows.append(r)
        # double plays: grounders, runner on 1st, < 2 outs, fielded by an infielder
        dp_opp = (f["la"] < 10) & f["on1"].eq(1) & (f["outs"] < 2) & made_by.isin(INFIELD)
        if dp_opp.any() and sub.get("dp") is not None:
            p_dp = sub["dp"].predict_proba(f.loc[dp_opp, FEATURES + ["fielder"]])[:, 1]
            got = ev[dp_opp].isin(DP).astype(float)
            pl = player_at(d, made_by)[dp_opp]
            rows.append(pd.DataFrame({"game_date": d.loc[dp_opp, "game_date"].astype(str).str[:10],
                                      "game_pk": d.loc[dp_opp, "game_pk"] if "game_pk" in d else 0, "player": pl,
                                      "pos": made_by[dp_opp], "dp_opps": 1.0,
                                      "dp_runs": (got - p_dp) * sub["dp_value"]}))
        # arm: balls fielded by an outfielder with runners on
        arm_opp = made_by.isin(OUTFIELD) & (f["on1"] + f["on2"] + f["on3"] > 0)
        if arm_opp.any() and sub.get("arm") is not None:
            rvp = _c(d, "delta_run_exp")[arm_opp] * sub["rv_sign"]           # batting side
            exp = sub["arm"].predict(f.loc[arm_opp, ARM_FEATURES])
            pl = player_at(d, made_by)[arm_opp]
            rows.append(pd.DataFrame({"game_date": d.loc[arm_opp, "game_date"].astype(str).str[:10],
                                      "game_pk": d.loc[arm_opp, "game_pk"] if "game_pk" in d else 0, "player": pl,
                                      "pos": made_by[arm_opp], "arm_opps": 1.0,
                                      "arm_runs": (exp - rvp).values}))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["player"]).assign(player=lambda x: x["player"].astype(int), pos=lambda x: x["pos"].astype(int))


def innings(df: pd.DataFrame) -> pd.DataFrame:
    """Outs recorded while each player was at each position (from each plate appearance's last pitch)."""
    pa = df[df["events"].notna()].copy() if "events" in df else df.iloc[0:0]
    if pa.empty:
        return pd.DataFrame(columns=["game_date", "game_pk", "player", "pos", "outs"])
    pa = pa.sort_values(["game_pk", "at_bat_number"])
    half = [pa["game_pk"], pa["inning"], pa["inning_topbot"].astype(str)]
    nxt = pa.groupby(half)["outs_when_up"].shift(-1)
    pa["outs_made"] = (nxt.fillna(3) - _c(pa, "outs_when_up")).clip(0, 3)
    rows = []
    for k in [2] + sorted(POS):
        col = f"fielder_{k}"
        if col in pa:
            rows.append(pd.DataFrame({"game_date": pa["game_date"].astype(str).str[:10], "game_pk": pa["game_pk"],
                                      "player": pa[col], "pos": k, "outs": pa["outs_made"]}))
    out = pd.concat(rows).dropna(subset=["player"])
    return out.groupby(["game_date", "game_pk", "player", "pos"], as_index=False)["outs"].sum()


def blocking_opps(df: pd.DataFrame) -> pd.DataFrame:
    """Pitches in the dirt (blocked / below the knees by a foot+) with runners on, per catcher per game."""
    if "fielder_2" not in df:
        return pd.DataFrame(columns=["game_date", "game_pk", "player", "pos", "b_opps"])
    on = df["on_1b"].notna() | df["on_2b"].notna() | df["on_3b"].notna()
    dirt = df["description"].astype(str).eq("blocked_ball") | (_c(df, "plate_z") < _c(df, "sz_bot") - 1.0)
    d = df[on & dirt]
    return (pd.DataFrame({"game_date": d["game_date"].astype(str).str[:10], "game_pk": d["game_pk"],
                          "player": d["fielder_2"], "pos": 2, "b_opps": 1.0})
            .dropna(subset=["player"]).groupby(["game_date", "game_pk", "player", "pos"], as_index=False).sum())


SUM_COLS = ["outs", "n", "opa", "runs", "fpa", "dp_opps", "dp_runs", "arm_opps", "arm_runs", "b_opps"] + \
           [f"s{k}_{x}" for k in range(1, 6) for x in ("n", "opa", "out")]


def daily_table(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Small file saved per season: game x player x position -> innings (outs), plays, OPA, runs, FPA, DP, arm,
    stars, blocking opps. Pages add it up for any period."""
    b = bundle or load()
    keys = ["game_date", "game_pk", "player", "pos"]
    parts = [innings(df), blocking_opps(df)]
    if b is not None:
        pc = play_credits(df, b)
        if not pc.empty:
            pc["n"] = pc.get("primary", 0)
            for k, lo, hi in STARS:
                inb = (pc.get("primary", 0) == 1) & pc["p_out"].between(lo, hi, inclusive="left")
                pc[f"s{k}_n"] = inb.astype(float)
                pc[f"s{k}_opa"] = np.where(inb, pc["opa"], 0.0)
                pc[f"s{k}_out"] = np.where(inb, pc["made"], 0.0)
            parts.append(pc[[c for c in keys + SUM_COLS if c in pc]])
    parts = [p.assign(player=p["player"].astype(int), pos=p["pos"].astype(int), game_pk=pd.to_numeric(p["game_pk"]))
             for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=keys + SUM_COLS)
    t = pd.concat(parts, ignore_index=True)
    for c in SUM_COLS:                                   # every column, even if this data had none of it
        if c not in t:
            t[c] = 0.0
    t[SUM_COLS] = t[SUM_COLS].fillna(0.0)
    return t.groupby(keys, as_index=False)[SUM_COLS].sum()


def summarize(daily: pd.DataFrame, start: str, end: str, positions: list[int]) -> dict:
    """player -> the Defense table values for a date window and set of positions."""
    if daily is None or daily.empty:
        return {}
    w = daily[(daily["game_date"] >= start) & (daily["game_date"] <= end) & daily["pos"].isin(positions)]
    if w.empty:
        return {}
    cols = [c for c in SUM_COLS if c in w]
    tot = w.groupby("player")[cols].sum()
    out = {}
    for pid, r in tot.iterrows():
        inn = r.get("outs", 0) / 3
        v = {"OPA": r.get("opa", 0), "CPA": r.get("opa", 0), "FPA": r.get("fpa", 0),
             "RS_FIELD": r.get("runs", 0), "RS_RANGE": r.get("runs", 0), "DP_RS": r.get("dp_runs", 0),
             "RS_ARM": r.get("arm_runs", 0), "B_OPPS": r.get("b_opps", 0), "IQ_INN": inn}
        infield = set(positions) <= INFIELD
        v["DRS"] = v["RS_FIELD"] + v["DP_RS"] if infield else v["RS_RANGE"] + v["RS_ARM"]
        for k in range(1, 6):
            n = r.get(f"s{k}_n", 0)
            v[f"S{k}_N"] = n
            v[f"S{k}"] = (r.get(f"s{k}_opa", 0) if infield else (r.get(f"s{k}_out", 0) / n if n else np.nan))
        out[int(pid)] = v
    return out


def season_rates(daily: pd.DataFrame, positions: list[int], min_inn: float = 300) -> pd.Series:
    """DRS / Season for every MLB player at these positions over the whole file (for percentiles)."""
    s = summarize(daily, "0000", "9999", positions)
    rates = {p: v["DRS"] / v["IQ_INN"] * SEASON_INN for p, v in s.items() if v["IQ_INN"] >= min_inn}
    return pd.Series(rates, dtype=float)
