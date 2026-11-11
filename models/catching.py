"""IQ catcher blocking and throwing (any period), from Statcast pitches + MLB play-by-play runner events.

BLOCKING -- built the way Statcast builds Blocks Above Average:
  Opportunities  every pitch with a runner on base, not put in play, not a hit-by-pitch (pitchouts / intentional
                 balls / pitch-clock calls excluded).
  Chance         WP + PB chance for a league-average catcher (wild pitches and passed balls judged the same,
                 whatever the scorer called it): where it crossed (side to side, height, height vs the zone),
                 speed, movement, pitch type, batter AND pitcher hand, count, swing or not.
                 Statcast also uses where the catcher actually set up (Hawk-Eye, not public) -- the one input we
                 can't match.
  Blocks Above Average  sum of (chance - what happened): a 10% pitch blocked = +0.10, a 2% pitch that gets by
                 = -0.98. An average catcher nets 0.
  Blocking Runs (Statcast)    BAA x 0.25 (their flat value, no game situation).
  Blocking Runs (Situational) each pitch's (chance - what happened) x what a WP / PB would actually cost in that
                 base / out situation (runners each move up a base; a runner on 3rd scores), from the run
                 expectancy of the era. THIS one goes into catcher DRS -- so difficulty and situation both count.
  Tiers (Statcast's lines)    Easy = under 5% WP/PB chance (95%+ caught), Medium = 5-15%, Tough = 15%+.
  Genuine opportunities       Medium + Tough (5%+ chance). Blocks / Block % count only these.
  Big Stop Runs               situational runs from TOUGH pitches he kept in front.
  Zones (catcher's view, every catcher right-handed): Air = not in the dirt; Dirt = crossing under 6 inches,
                 split Glove side (catcher's left, left of the plate), Middle (over the plate), Arm side
                 (catcher's right, right of the plate).
THROWING: every steal attempt (not pickoffs) gets a caught-stealing chance (base, pitch type, velo, location,
  pitcher hand, batter side, outs, count, swing). No runner speed. Throwing runs = (caught - chance) x run swing.
Era models (2015-2022, 2023+). Trained by  python -m pipeline.train_catching
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "catching.joblib"
SHIFT_BAN = 2023
USE_RUNNER_SPEED = False
STATCAST_BLOCK_RUNS = 0.25                           # Statcast: runs per block, no game situation
EASY, TOUGH = 0.05, 0.15                             # WP/PB chance lines (Statcast: 95% / 85% block probability)
DIRT_FT = 0.5                                        # crossing under 6 inches = in the dirt
PLATE_EDGE = 17 / 24                                 # half the plate width (ft)
CS_RUNS = {"2B": 0.63, "3B": 0.80, "home": 1.10}     # caught stealing vs stolen base, by base
BLOCK_FEATURES = ["px", "pz", "pz_rel", "velo", "ivb", "hb", "ptype", "swing", "p_left", "b_left", "balls", "strikes"]
THROW_FEATURES = ["to3", "tohome", "velo", "ptype", "px", "pz_rel", "p_left", "b_left", "outs", "balls",
                  "strikes", "swing"]
SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt",
          "missed_bunt", "bunt_foul_tip"}
NOT_OPPS = {"hit_by_pitch", "pitchout", "intent_ball", "automatic_ball", "automatic_strike", "foul_pitchout"}
ZONES = ["air", "dirt_glove", "dirt_mid", "dirt_arm"]
ZONE_KEY = {"air": "B_AIR", "dirt_glove": "B_DGS", "dirt_mid": "B_DMID", "dirt_arm": "B_DAS"}
CATCH_COLS = (["c_n", "c_exp", "c_act", "c_runs", "c_runs_sc",
               "g_n", "g_blk", "g_exp", "tough_n", "tough_blk", "tough_exp", "med_n", "med_blk", "med_exp",
               "big_runs", "t_n", "t_exp", "t_cs", "t_runs"]
              + [f"{z}_{x}" for z in ZONES for x in ("n", "exp", "act")])


def _c(df, n):
    return pd.to_numeric(df[n], errors="coerce") if n in df else pd.Series(np.nan, index=df.index)


def features(df: pd.DataFrame) -> pd.DataFrame:
    from models.pitch_quality import PTYPE
    desc = df["description"].astype(str) if "description" in df else pd.Series("", index=df.index)
    return pd.DataFrame({
        "px": _c(df, "plate_x"), "pz": _c(df, "plate_z"), "pz_rel": _c(df, "plate_z") - _c(df, "sz_bot"),
        "velo": _c(df, "release_speed"), "ivb": 12 * _c(df, "pfx_z"), "hb": 12 * _c(df, "pfx_x"),
        "ptype": df["pitch_type"].astype(str).map(PTYPE).astype(float) if "pitch_type" in df else np.nan,
        "swing": desc.isin(SWINGS).astype(float),
        "p_left": (df["p_throws"].astype(str) == "L").astype(float) if "p_throws" in df else 0.0,
        "b_left": (df["stand"].astype(str) == "L").astype(float) if "stand" in df else 0.0,
        "balls": _c(df, "balls"), "strikes": _c(df, "strikes"), "outs": _c(df, "outs_when_up"),
    }, index=df.index)


def zone(df: pd.DataFrame) -> pd.Series:
    """By LOCATION only (catcher's view, + plate_x = catcher's right = arm side for a right-handed catcher)."""
    height, x = _c(df, "plate_z"), _c(df, "plate_x")
    side = np.select([x < -PLATE_EDGE, x > PLATE_EDGE], ["dirt_glove", "dirt_arm"], "dirt_mid")
    return pd.Series(np.where(height < DIRT_FT, side, "air"), index=df.index)


def block_rows(df: pd.DataFrame) -> pd.Series:
    """Statcast's blocking opportunities: a runner on, not put in play, not a HBP, with a location."""
    on = df["on_1b"].notna() | df["on_2b"].notna() | df["on_3b"].notna()
    return (on & ~df["type"].astype(str).eq("X") & ~df["description"].astype(str).isin(NOT_OPPS)
            & _c(df, "plate_x").notna() & _c(df, "plate_z").notna())


def base_code(df: pd.DataFrame) -> pd.Series:
    return (df["on_1b"].notna().astype(int) + 2 * df["on_2b"].notna().astype(int)
            + 4 * df["on_3b"].notna().astype(int))


def run_expectancy(df: pd.DataFrame) -> dict:
    """RE by (base code, outs): average runs scored from the start of a plate appearance to the end of the inning."""
    first = df[pd.to_numeric(df["pitch_number"], errors="coerce").eq(1)].copy()
    half = [df["game_pk"], df["inning"], df["inning_topbot"].astype(str)]
    end = pd.to_numeric(df["post_bat_score"], errors="coerce").groupby(half).transform("max")
    first["to_end"] = end[first.index] - pd.to_numeric(first["bat_score"], errors="coerce")
    first["state"] = list(zip(base_code(first), pd.to_numeric(first["outs_when_up"], errors="coerce").astype(int)))
    return first.groupby("state")["to_end"].mean().to_dict()


def wp_values(re: dict) -> dict:
    """What a WP / PB costs in each base / out state: every runner moves up a base, a runner on 3rd scores."""
    out = {}
    for b in range(1, 8):
        for o in range(3):
            on1, on2, on3 = b & 1, b & 2, b & 4
            new = (2 if on1 else 0) + (4 if on2 else 0)
            before, after = re.get((b, o)), re.get((new, o))
            if before is not None and after is not None:
                out[(b, o)] = float((1 if on3 else 0) + after - before)
    return out


def keyed(df: pd.DataFrame) -> pd.Series:
    return (df["game_pk"].astype("int64").astype(str) + "|" + df["at_bat_number"].astype("int64").astype(str) + "|"
            + pd.to_numeric(df["pitch_number"], errors="coerce").fillna(-1).astype("int64").astype(str))


def wp_pb_flag(df: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """1 if a wild pitch or passed ball happened on that pitch (dropped third strikes included)."""
    if events is None or events.empty:
        return pd.Series(0.0, index=df.index)
    e = events[events["event"].isin(["wild_pitch", "passed_ball"])].dropna(subset=["pitch_number"])
    return keyed(df).isin(set(keyed(e))).astype(float)


def steal_attempts(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per stolen-base attempt with the pitch it happened on (pitch columns + base + caught)."""
    if events is None or events.empty:
        return pd.DataFrame()
    e = events[events["event"].astype(str).str.startswith(("stolen_base", "caught_stealing"))].dropna(
        subset=["pitch_number"]).copy()
    if e.empty:
        return pd.DataFrame()
    e["key"] = keyed(e)
    m = e[["key", "event"]].merge(df.assign(key=keyed(df)), on="key", how="inner")
    m["base"] = m["event"].str.extract(r"_(2b|3b|home)$")[0].map({"2b": "2B", "3b": "3B", "home": "home"})
    m["cs"] = m["event"].str.startswith("caught_stealing").astype(float)
    return m.dropna(subset=["base"])


def throw_features(att: pd.DataFrame) -> pd.DataFrame:
    f = features(att)
    f["to3"] = att["base"].eq("3B").astype(float)
    f["tohome"] = att["base"].eq("home").astype(float)
    return f


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


def block_table(d: pd.DataFrame, events: pd.DataFrame, sub: dict) -> pd.DataFrame:
    """Per blocking opportunity: chance, what happened, situation value, credits, tier, zone."""
    p = pd.Series(sub["block"].predict_proba(features(d)[BLOCK_FEATURES])[:, 1], index=d.index)
    y = wp_pb_flag(d, events)
    state = pd.Series(list(zip(base_code(d), _c(d, "outs_when_up").fillna(0).astype(int))), index=d.index)
    val = state.map(sub.get("wp_values", {})).astype(float).fillna(sub.get("wp_value_avg", 0.28))
    return pd.DataFrame({"p": p, "y": y, "val": val, "credit": p - y, "runs": (p - y) * val,
                         "runs_sc": (p - y) * STATCAST_BLOCK_RUNS,
                         "tier": np.select([p >= TOUGH, p >= EASY], ["tough", "medium"], "easy"),
                         "zone": zone(d)}, index=d.index)


def daily_rows(df: pd.DataFrame, events: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Per game x catcher (pos 2 in the fielding file): blocking (overall, tiers, zones, situational + Statcast runs)
    and throwing."""
    b = bundle or load()
    keys = ["game_date", "game_pk", "player", "pos"]
    if b is None or "fielder_2" not in df or events is None:
        return pd.DataFrame(columns=keys + CATCH_COLS)
    parts = []
    for m, sub in _bundles(df, b):
        d = df[m & block_rows(df)]
        if not d.empty and sub.get("block") is not None:
            t = block_table(d, events, sub)
            r = pd.DataFrame({"game_date": d["game_date"].astype(str).str[:10], "game_pk": d["game_pk"],
                              "player": d["fielder_2"], "pos": 2, "c_n": 1.0, "c_exp": t["p"], "c_act": t["y"],
                              "c_runs": t["runs"], "c_runs_sc": t["runs_sc"]}, index=d.index)
            g, tough, med = t["tier"] != "easy", t["tier"] == "tough", t["tier"] == "medium"
            blk = 1 - t["y"]
            r["g_n"], r["g_blk"], r["g_exp"] = g.astype(float), (g * blk), (g * (1 - t["p"]))
            r["tough_n"], r["tough_blk"], r["tough_exp"] = tough.astype(float), tough * blk, tough * (1 - t["p"])
            r["med_n"], r["med_blk"], r["med_exp"] = med.astype(float), med * blk, med * (1 - t["p"])
            r["big_runs"] = np.where(tough & (t["y"] == 0), t["p"] * t["val"], 0.0)
            for z in ZONES:
                inz = (t["zone"] == z).astype(float)
                r[f"{z}_n"], r[f"{z}_exp"], r[f"{z}_act"] = inz, inz * t["p"], inz * t["y"]
            parts.append(r)
        att = steal_attempts(df[m], events)
        if not att.empty and sub.get("throw") is not None:
            p = sub["throw"].predict_proba(throw_features(att)[THROW_FEATURES])[:, 1]
            val = att["base"].map(CS_RUNS).fillna(0.65).values
            parts.append(pd.DataFrame({"game_date": att["game_date"].astype(str).str[:10], "game_pk": att["game_pk"],
                                       "player": att["fielder_2"], "pos": 2, "t_n": 1.0, "t_exp": p,
                                       "t_cs": att["cs"].values, "t_runs": (att["cs"].values - p) * val}))
    if not parts:
        return pd.DataFrame(columns=keys + CATCH_COLS)
    t = pd.concat(parts, ignore_index=True).dropna(subset=["player"])
    for c in CATCH_COLS:
        if c not in t:
            t[c] = 0.0
    t[CATCH_COLS] = t[CATCH_COLS].fillna(0.0)
    t["player"] = t["player"].astype(int)
    return t.groupby(keys, as_index=False)[CATCH_COLS].sum()


def summarize(daily: pd.DataFrame, start: str, end: str, throws: str = "R") -> dict:
    """catcher -> every blocking / throwing column. (Every catcher is treated as right-handed.)"""
    if daily is None or daily.empty or "c_n" not in daily:
        return {}
    w = daily[(daily["game_date"] >= start) & (daily["game_date"] <= end) & daily["pos"].eq(2)]
    cols = [c for c in CATCH_COLS if c in w]
    tot = w.groupby("player")[cols].sum()
    out = {}
    for pid, r in tot.iterrows():
        g = r.get
        v = {"B_RS": g("c_runs") if g("c_n") else np.nan, "B_RS_SC": g("c_runs_sc") if g("c_n") else np.nan,
             "B_BAA": g("c_exp") - g("c_act") if g("c_n") else np.nan,
             "B_OPPS": g("g_n"), "B_BLOCKS": g("g_blk"),
             "B_BLK%": g("g_blk") / g("g_n") if g("g_n") else np.nan,
             "B_XBLK%": g("g_exp") / g("g_n") if g("g_n") else np.nan,
             "B_TOUGH_N": g("tough_n"), "B_TOUGH%": g("tough_blk") / g("tough_n") if g("tough_n") else np.nan,
             "B_MED_N": g("med_n"), "B_MED%": g("med_blk") / g("med_n") if g("med_n") else np.nan,
             "B_BIG": g("big_runs"),
             "T_RS": g("t_runs") if g("t_n") else np.nan, "T_CSAA": g("t_cs") - g("t_exp")}
        for z, key in ZONE_KEY.items():
            n = g(f"{z}_n", 0)
            v[f"_{key}_rate"] = (g(f"{z}_exp") - g(f"{z}_act")) / n if n >= 20 else np.nan
        out[int(pid)] = v
    return out


def bucket_percentiles(daily: pd.DataFrame, value: dict, throws: str = "R") -> dict:
    """Percentile of each zone's blocks-above-average rate vs every MLB catcher that season."""
    season = summarize(daily, "0000", "9999")
    out = {}
    for key in ZONE_KEY.values():
        mine = value.get(f"_{key}_rate", np.nan)
        board = pd.Series([v.get(f"_{key}_rate", np.nan) for v in season.values()]).dropna()
        if mine == mine and len(board) >= 10:
            out[key] = 100 * (board < mine).mean()
    return out
