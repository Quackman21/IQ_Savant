"""IQ fielding: DRS and its parts from public Statcast pitch data (any period).

Per ball in play (fielders known from Statcast's fielder_2..fielder_9 columns):
  out chance   how often an average defense turns a ball like this into an out (exit velo, launch angle,
               field direction, Statcast's projected distance for balls in the air, batter side). Direction is a fair
               input -- it's what the fielder is judged on. Distance comes from the ball's flight, NOT the hit
               coordinates (those mark where a hit was picked up after rolling, which leaks the result).
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
    la = _c(df, "launch_angle")
    # distance: Statcast's PROJECTED distance from the ball's flight, balls in the air only. The hit coordinates
    # mark where a ball was caught -- or where a hit was picked up after rolling -- so their distance partly
    # records the result (it made the model near-certain and shrank every fielder's credit).
    dist = _c(df, "hit_distance_sc").where(la >= 10, -1.0)          # ground balls: -1 (no flight distance)
    return pd.DataFrame({
        "ev": _c(df, "launch_speed"), "la": la,
        "spray": np.degrees(np.arctan2(x, y)) * 0.75, "dist": dist,
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


def _des(d: pd.DataFrame) -> pd.Series:
    return d["des"].astype(str).str.lower() if "des" in d else pd.Series("", index=d.index)


def converted(d: pd.DataFrame) -> pd.Series:
    """Did the fielder who got to the ball finish the play? An out -- or a 'missed catch' error, where the RECEIVER
    dropped a good throw (that's on the first baseman, not the fielder)."""
    return d["events"].astype(str).isin(OUTS) | _des(d).str.contains("missed catch")


def fielding_error(d: pd.DataFrame) -> pd.Series:
    """Bobbles / boots: a FIELDING error (not a throwing error) on the play."""
    return d["events"].astype(str).eq("field_error") & _des(d).str.contains("fielding error")


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
        Xc = f[FEATURES].assign(fielder=made_by)
        got_if = made_by.isin(INFIELD)
        p_conv = pd.Series(np.nan, index=d.index)
        p_ferr = pd.Series(np.nan, index=d.index)
        if sub.get("conv") is not None and got_if.any():
            p_conv[got_if] = sub["conv"].predict_proba(Xc.loc[got_if, FEATURES + ["fielder"]])[:, 1]
        if sub.get("ferr") is not None and got_if.any():
            p_ferr[got_if] = sub["ferr"].predict_proba(Xc.loc[got_if, FEATURES + ["fielder"]])[:, 1]
        did_conv = converted(d).astype(float)
        did_ferr = fielding_error(d).astype(float)
        for k in sorted(POS):
            share = who.get(k, pd.Series(0.0, index=d.index))
            mine = (out == 1) & (made_by == k)
            opa = np.where(mine, 1 - p_out, np.where(out == 0, -p_out * share, 0.0))
            fpa = (made_by == k).astype(float) - fld.get(k, pd.Series(0.0, index=d.index))
            player = _c(d, f"fielder_{k}")
            primary = who.idxmax(axis=1).eq(k) | mine                    # this play is "his" for stars
            # Range + Conversion = OPA. Conversion: balls he fielded (infield), finished vs expected once fielded.
            # Range: everything else -- getting to the ball. Ball Security: fielding errors vs expected (part of
            # Conversion, shown on its own).
            fielded_me = (made_by == k) & p_conv.notna()
            conv = np.where(fielded_me, did_conv - p_conv.fillna(0), 0.0)
            sec = np.where(fielded_me & p_ferr.notna(), p_ferr.fillna(0) - did_ferr, 0.0)
            keep = (np.abs(opa) > 1e-9) | (np.abs(fpa) > 1e-3) | primary
            if not keep.any():
                continue
            r = pd.DataFrame({"game_date": d["game_date"].astype(str).str[:10], "game_pk": d.get("game_pk", 0),
                              "player": player, "pos": k, "opa": opa, "runs": opa * rv, "fpa": fpa,
                              "primary": primary.astype(float), "p_out": p_out, "out": out,
                              "made": mine.astype(float), "rng": opa - conv, "conv": conv, "sec": sec,
                              "conv_n": fielded_me.astype(float)}, index=d.index)[keep]
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


from models.catching import CATCH_COLS  # noqa: E402

SUM_COLS = ["outs", "n", "opa", "runs", "fpa", "dp_opps", "dp_runs", "arm_opps", "arm_runs", "b_opps",
            "rng", "conv", "sec", "conv_n"] + \
           [f"s{k}_{x}" for k in range(1, 6) for x in ("n", "opa", "out")] + CATCH_COLS


def daily_table(df: pd.DataFrame, bundle: dict | None = None, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Small file saved per season: game x player x position -> innings (outs), plays, OPA, runs, FPA, DP, arm,
    stars, blocking opps, and (catchers, with play-by-play events) blocking / throwing. Pages add it up for any
    period."""
    b = bundle or load()
    keys = ["game_date", "game_pk", "player", "pos"]
    parts = [innings(df), blocking_opps(df)]
    if events is not None and not events.empty:
        from models import catching
        parts.append(catching.daily_rows(df, events))
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
             "RS_ARM": r.get("arm_runs", 0), "B_OPPS": r.get("b_opps", 0), "IQ_INN": inn,
             "_N": r.get("n", 0), "_RNG": r.get("rng", 0), "_CONV": r.get("conv", 0), "_SEC": r.get("sec", 0),
             "_CONV_N": r.get("conv_n", 0)}
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


# ----------------------------------------------------------------------------
# Tool grades (20-80) -- only what public data can MEASURE (no stand-ins)
# ----------------------------------------------------------------------------
GRADE_FILE = Path(__file__).resolve().parent.parent / "data" / "league" / "fielding_grades.json"
# chances before each rate is 70% skill. Defaults until python -m pipeline.defense_study measures them
# (it writes data/league/fielding_grades.json, which overrides these).
FOR_70 = {"range": 250, "conv": 300, "sec": 600}
MIN_GRADE_N = {"range": 25, "conv": 20, "sec": 20}
GRADE_SCALE = "talent"         # see scale_factor; the fielding comparison study prints all three to choose from
# (stat, numerator, denominator) -- Range per responsible play, Conversion / Ball Security per ball he fielded
GRADE_STATS = {"range": ("_RNG", "_N"), "conv": ("_CONV", "_CONV_N"), "sec": ("_SEC", "_CONV_N")}
GRADE_KEYS = {"IF": {"range": "RANGE_GRADE", "conv": "CONVERSION_GRADE", "sec": "BALL_SECURITY_GRADE"},
              "OF": {"range": "RANGE"}}


def _for70() -> dict:
    try:
        import json
        return {**FOR_70, **json.loads(GRADE_FILE.read_text()).get("for_70", {})}
    except Exception:
        return dict(FOR_70)


def _rate(v: dict, stat: str) -> tuple[float, float]:
    num, den = GRADE_STATS[stat]
    n = float(v.get(den, 0) or 0)
    return (float(v.get(num, 0) or 0) / n if n else np.nan), n


def grade_spread(daily: pd.DataFrame, positions: list[int], min_inn: float = 300) -> dict:
    """League mean / spread of each graded rate among MLB regulars at these positions (whole season file)."""
    s = summarize(daily, "0000", "9999", positions)
    out = {}
    for stat in GRADE_STATS:
        xs = [_rate(v, stat)[0] for v in s.values() if v["IQ_INN"] >= min_inn]
        xs = pd.Series(xs, dtype=float).dropna()
        ns = pd.Series([_rate(v, stat)[1] for v in s.values() if v["IQ_INN"] >= min_inn], dtype=float)
        if len(xs) >= 15 and xs.std() > 0:
            out[stat] = (float(xs.mean()), float(xs.std()), float(ns.median()))
    return out


def scale_factor(n_typical: float, k: float, scale: str) -> float:
    """What one grade spread (10 points) means. 'raw' (v20.0): the spread of raw season rates -- includes the noise
    the regression already removed, so grades get squeezed twice (one infield season barely leaves 41-59).
    'talent': the spread of TRUE talent (raw spread x sqrt(reliability of a regular's season)) -- a real +2-spread
    defender grades 70 once his sample earns it. 'rank': the spread of a regular's regressed season -- regulars land
    on the same 20-80 spread as the hitting grades; part-timers are still pulled toward 50."""
    rel = n_typical / (n_typical + k) if n_typical else 1.0
    return {"raw": 1.0, "talent": rel ** 0.5, "rank": rel}.get(scale, 1.0)


def grades(v: dict, spread: dict, group: str, scale: str | None = None) -> dict:
    """Range / Conversion / Ball Security grades (infield) or Range (outfield) for one player's summary row,
    regressed by his own sample: 50 = average MLB regular at the position, 10 = one spread."""
    out, k70 = {}, _for70()
    scale = scale or GRADE_SCALE
    for stat, key in GRADE_KEYS[group].items():
        x, n = _rate(v, stat)
        if stat not in spread or x != x or n < MIN_GRADE_N[stat]:
            continue
        mean, sd, *rest = spread[stat]
        k = k70.get(f"{stat}_{group}", k70[stat]) * 0.3 / 0.7
        shrink = n / (n + k)
        out[key] = 50 + 10 * (x - mean) * shrink / (sd * scale_factor(rest[0] if rest else n, k, scale))
        unit = "plays" if stat == "range" else "balls fielded"
        out[key + "_PARTS"] = f"{100 * x:+.1f} outs per 100 {unit} vs average | {int(n)} {unit}, " \
                              f"{100 * shrink:.0f}% trusted"
    return out


# ----------------------------------------------------------------------------
# v20.2 Range = 90% Statcast OAA + 10% ours (full seasons). From the fielding comparison study: OAA knows where the
# fielder STARTED, ours doesn't -- in the infield ours mostly measures the team's positioning (r .06 with OAA at
# SS), so partial-period infield Range is left blank. OF ours agrees with OAA (~.7), so it stays for partial periods.
# ----------------------------------------------------------------------------
RANGE_W_OAA = 0.9
OAA_REL = {"IF": 0.425, "OF": 0.606}       # OAA year-to-year r for regulars = reliability of a full season (study)
TYPICAL_INN = 1100


def oaa_spread(oaa_by_pos: dict, inn_by_pos: dict, min_inn: float = 300) -> tuple[float, float] | None:
    """Mean / spread of OAA per full season (1,200 innings) among regulars at these positions.
    oaa_by_pos: {pos: {player: OAA}}, inn_by_pos: {pos: {player: our innings there}}."""
    xs = [o * SEASON_INN / inn_by_pos[k][p] for k, b in oaa_by_pos.items() for p, o in b.items()
          if inn_by_pos.get(k, {}).get(p, 0) >= min_inn and o == o]
    xs = pd.Series(xs, dtype=float)
    return (float(xs.mean()), float(xs.std())) if len(xs) >= 15 and xs.std() > 0 else None


def range_grade_oaa(oaa: float, inn: float, ospread: tuple | None, v: dict, spread: dict, group: str) -> dict:
    """Full-season Range grade: 90% OAA + 10% our Range (each vs regulars at the position), regressed by innings
    with OAA's reliability, on the talent scale (a typical regular: 50 + 10 x z x sqrt(reliability))."""
    key = GRADE_KEYS[group]["range"]
    if ospread is None or oaa != oaa or not inn:
        return {}
    z_oaa = (oaa * SEASON_INN / inn - ospread[0]) / ospread[1]
    x, n = _rate(v, "range")
    z_ours = (x - spread["range"][0]) / spread["range"][1] if "range" in spread and x == x and n else 0.0
    z = RANGE_W_OAA * z_oaa + (1 - RANGE_W_OAA) * z_ours
    rel = OAA_REL[group]
    k = TYPICAL_INN * (1 - rel) / rel
    shrink = inn / (inn + k)
    return {key: 50 + 10 * z * shrink / rel ** 0.5,
            key + "_PARTS": f"OAA {oaa:+.0f} in {inn:,.0f} inn ({z_oaa:+.1f} spreads, 90%) | ours "
                            f"{z_ours:+.1f} spreads (10%) | {100 * shrink:.0f}% trusted"}
