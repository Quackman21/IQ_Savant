"""The big crunch: every MLB competitive swing, precomputed, so any hitter x any pitch combo is a quick lookup.

For each bat-tracking season, from the saved full download (data/raw/statcast_<season>.csv.gz -- nothing new is
downloaded), every COMPETITIVE swing (Statcast's definition: no bunts; each hitter's slowest 10% dropped unless 60+
mph and hit 90+) with everything the Bat Path tab derives:

  who / what     batter, pitcher, stand, p_throws, pitch type, pitch group, shape cluster, zone, count group, game, date
  measured       bat speed, swing length, attack angle, attack direction (+ = oppo), swing-path tilt, contact
                 intercepts, plate location, exit velo, launch angle, spray, result
  contact depth  pf = inches in front of the plate (MLB's definition: intercept - his stance depth vs that hand)
  plane match    match_v / match_h = the attack angle / direction that would match that pitch (its measured approach
                 at contact); gap_v / gap_h = measured - match (0 = on the pitch's plane / line)
  bat angles     vba / hba at contact -- INFERRED from that swing's measured attack angle, direction + tilt
  quality        squared-up %, blast, smash factor, contact type (Optimal / Backspin / ...), fast (75+)

  data/bat_cube/swings_<season>.parquet (or .csv.gz without pyarrow)     one row per swing
  data/bat_cube/sums_<season>.csv.gz                                       n / sum / sum of squares per metric for
      batter x stand x pitcher hand x pitch type x zone x count group x fast -- additive, so ANY coarser combo
      (all pitches, a pitch group, a few zones, both hands, ...) rolls up exactly: mean and SD for every player and
      every combination of pitch shapes.

    python -m pipeline.bat_cube 2023 2024 2025 2026
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bat_cube"
FAST = 75.0
KEYS = ["batter", "stand", "p_throws", "pitch_type", "zone", "count_grp", "fast"]
METRICS = ["bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt", "pf", "gap_v",
           "gap_h", "match_v", "match_h", "vba", "hba", "launch_speed", "launch_angle", "squared_up_pct", "smash"]
COLS = ["game_date", "game_pk", "game_type", "at_bat_number", "pitch_number", "batter", "pitcher", "stand",
        "p_throws", "pitch_type", "zone", "balls", "strikes", "description", "events", "des", "bb_type", "bat_speed",
        "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
        "intercept_ball_minus_batter_pos_y_inches", "intercept_ball_minus_batter_pos_x_inches", "launch_speed",
        "launch_angle", "launch_speed_angle", "hc_x", "hc_y", "plate_x", "plate_z", "sz_top", "sz_bot",
        "release_speed", "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z", "pfx_x", "pfx_z",
        "arm_angle", "vx0", "vy0", "vz0", "ax", "ay", "az", "estimated_woba_using_speedangle", "delta_run_exp",
        "home_team", "away_team", "inning_topbot"]
SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt",
          "missed_bunt", "bunt_foul_tip"}


def competitive(d: pd.DataFrame) -> pd.Series:
    """Statcast's competitive swings, per hitter-season (same rule as the site's hitting tables)."""
    desc, ev_ = d["description"].astype(str), d["events"].astype(str)
    bunt = desc.str.contains("bunt") | ev_.str.contains("bunt") | d.get("des", pd.Series("", index=d.index)).astype(str).str.contains("bunt")
    bs = pd.to_numeric(d["bat_speed"], errors="coerce")
    tracked = bs.where(desc.isin(SWINGS) & bs.notna() & ~bunt)
    g = tracked.groupby(d["batter"])
    slow = g.rank(method="first") <= np.floor(g.transform("count") * 0.10)
    return tracked.notna() & (~slow | ((tracked >= 60) & (pd.to_numeric(d["launch_speed"], errors="coerce") >= 90)))


def build(season: int, raw: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    from core import aar, batted_ball, hit_metrics as hm
    from core.bat_path import bat_angles
    from pipeline.bat_norms import _stance_x, plane
    t0 = time.time()
    say = (lambda m: print(f"  [{season}] {m} ({time.time() - t0:.0f}s)", flush=True)) if verbose else (lambda m: None)
    if raw is None:
        f = ROOT / "data" / "raw" / f"statcast_{season}.csv.gz"
        head = pd.read_csv(f, nrows=0).columns
        raw = pd.read_csv(f, usecols=[c for c in COLS if c in head], low_memory=False)
        say(f"{len(raw):,} pitches read")
    for c in COLS:
        if c not in raw:
            raw[c] = np.nan
    # pitch-shape clusters need every pitch (fastball differences are per pitcher), so score before filtering
    cluster = pd.Series(None, index=raw.index, dtype=object)
    try:
        from models import pitch_models, pitch_quality as pq
        b = pitch_models._era_bundle(pitch_models.load(), [season])
        if b:
            cluster = pitch_models.cluster_of(pq.features(raw), b)
            say("shape clusters scored")
    except Exception as e:
        say(f"shape clusters skipped ({type(e).__name__})")
    d = raw[competitive(raw)].copy()
    d["cluster"] = cluster.reindex(d.index)
    say(f"{len(d):,} competitive swings")
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    for c in ("bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt", "launch_speed",
              "launch_angle", "release_speed", "plate_x", "plate_z", "zone"):
        d[c] = num(c)
    d["group"] = hm.pitch_group(d)
    d["count_grp"] = hm.count_group(d)
    d["fast"] = d["bat_speed"] >= FAST
    d["made"] = d["description"].astype(str).isin(["foul", "foul_tip", "hit_into_play"])
    d["year"] = season
    # contact depth vs the plate (MLB's definition): his stance depth vs that pitcher hand, else his season
    season_v = aar.stance_depths(season)
    try:
        from core import stance_feed
        views = stance_feed.views(season)
    except Exception:
        views = {}
    keys = zip(d["batter"].astype(int), d["stand"].astype(str), d["p_throws"].astype(str))
    d["depth"] = pd.to_numeric(pd.Series([(views.get(f"vs_{h}", {}) or {}).get((b, s)) or season_v.get((b, s))
                                          for b, s, h in keys], index=d.index), errors="coerce")
    cy = num("intercept_ball_minus_batter_pos_y_inches").where(d["made"])
    d["pf"] = cy - d["depth"]
    say(f"contact depth for {d['pf'].notna().sum():,} contacts")
    # plane match (measured) + bat angles (inferred), contacts with a depth
    for c in ("match_v", "match_h", "gap_v", "gap_h", "vba", "hba"):
        d[c] = np.nan
    have = d["pf"].notna()
    if have.any():
        pl = plane(d.loc[have], d.loc[have, "pf"])
        for c in ("match_v", "match_h", "gap_v", "gap_h"):
            d.loc[have, c] = np.asarray(pl[c], float)
        sx = _stance_x(season)
        h = d[have]
        xs = np.asarray([sx.get((int(b), str(s)), 27.0) for b, s in zip(h["batter"], h["stand"])], float).round(0)
        for x in np.unique(xs):
            m = xs == x
            hh = h[m]
            v, hb = bat_angles(hh["attack_angle"], hh["attack_direction"], hh["swing_path_tilt"], hh["stand"], hh["pf"],
                               hh["depth"], 12 * hh["plate_x"], 12 * hh["plate_z"],
                               {"R": x, "L": x})
            idx = hh.index
            d.loc[idx, "vba"], d.loc[idx, "hba"] = v, hb
        say("plane match + bat angles")
    # contact quality (MLB / Savant definitions) + contact type
    ev, bs, vp = d["launch_speed"].where(d["made"]), d["bat_speed"], d["release_speed"]
    d["squared_up_pct"] = 100 * ev / (1.23 * bs + 0.23 * vp)
    d["squared_up"] = d["squared_up_pct"] >= 80
    d["blast"] = (d["squared_up_pct"] + bs) >= 164
    d["smash"] = 1 + (ev - bs) / (vp + bs)
    try:
        d["ctype"] = batted_ball.spin_labels(d.assign(game_year=season))
    except Exception:
        d["ctype"] = None
    say("contact quality")
    keep = ["year", "game_date", "game_pk", "game_type", "at_bat_number", "pitch_number", "batter", "pitcher", "stand",
            "p_throws", "pitch_type", "group", "cluster", "zone", "balls", "strikes", "count_grp", "description",
            "events", "bb_type", "fast", "made", "depth", "plate_x", "plate_z", "sz_top", "sz_bot", "release_speed",
            "hc_x", "hc_y", "launch_speed_angle", "estimated_woba_using_speedangle", "delta_run_exp",
            "intercept_ball_minus_batter_pos_y_inches", "intercept_ball_minus_batter_pos_x_inches", "squared_up",
            "blast", "ctype"] + [m for m in METRICS if m not in ("launch_speed", "launch_angle")] + \
        ["launch_speed", "launch_angle"]
    out = d[[c for c in dict.fromkeys(keep) if c in d]].reset_index(drop=True)
    return out


def sums(cube: pd.DataFrame) -> pd.DataFrame:
    """Additive sufficient statistics (n, sum, sum of squares per metric) at the finest combo -- roll up anything."""
    c = cube.copy()
    c["zone"] = c["zone"].fillna(0).astype(int)
    c["pitch_type"] = c["pitch_type"].fillna("UN")
    parts = {}
    for m in METRICS:
        v = pd.to_numeric(c[m], errors="coerce")
        parts[f"n_{m}"], parts[f"s_{m}"], parts[f"q_{m}"] = v.notna().astype(int), v.fillna(0), (v * v).fillna(0)
    g = pd.concat([c[KEYS], pd.DataFrame(parts, index=c.index)], axis=1).groupby(KEYS, dropna=False).sum()
    g.insert(0, "swings", c.groupby(KEYS, dropna=False).size())
    return g.reset_index()


def rollup(s: pd.DataFrame, batter: int | None = None, stand: str | None = None, hand: str | None = None,
           types=None, zones=None, counts=None, fast: bool | None = None) -> dict:
    """Mean / SD / n of every metric for any combo (None = all). e.g. rollup(s, batter=547180, types=['FF','SI'],
    zones=[5], fast=True)."""
    m = pd.Series(True, index=s.index)
    if batter is not None:
        m &= s["batter"] == batter
    if stand:
        m &= s["stand"] == stand
    if hand:
        m &= s["p_throws"] == hand
    if types:
        m &= s["pitch_type"].isin(list(types))
    if zones:
        m &= s["zone"].isin(list(zones))
    if counts:
        m &= s["count_grp"].isin(list(counts))
    if fast is not None:
        m &= s["fast"] == fast
    g = s[m]
    out = {"swings": int(g["swings"].sum())}
    for k in METRICS:
        n, sm, q = g[f"n_{k}"].sum(), g[f"s_{k}"].sum(), g[f"q_{k}"].sum()
        mean = sm / n if n else np.nan
        sd = np.sqrt(max(0.0, (q - n * mean * mean) / (n - 1))) if n > 1 else np.nan
        out[k] = {"n": int(n), "mean": None if n == 0 else round(float(mean), 3),
                  "sd": None if n < 2 else round(float(sd), 3)}
    return out


def save(season: int, cube: pd.DataFrame) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        f = OUT / f"swings_{season}.parquet"
        cube.to_parquet(f, index=False)
    except ImportError:
        f = OUT / f"swings_{season}.csv.gz"
        cube.to_csv(f, index=False)
    fs = OUT / f"sums_{season}.csv.gz"
    sums(cube).to_csv(fs, index=False)
    return [f, fs]


def load(season: int) -> pd.DataFrame:
    for f in (OUT / f"swings_{season}.parquet", OUT / f"swings_{season}.csv.gz"):
        if f.exists():
            return pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, low_memory=False)
    return pd.DataFrame()


def load_sums(season: int) -> pd.DataFrame:
    f = OUT / f"sums_{season}.csv.gz"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def run(season: int) -> None:
    t0 = time.time()
    cube = build(season)
    paths = save(season, cube)
    print(f"bat cube {season}: {len(cube):,} competitive swings ({int(cube['fast'].sum()):,} fast, "
          f"{int(cube['pf'].notna().sum()):,} with contact depth, {cube['batter'].nunique():,} hitters) -> "
          f"{', '.join(p.name for p in paths)} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:] or ["2023", "2024", "2025", "2026"]:
        run(int(a))
