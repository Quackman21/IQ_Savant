"""MLB normative ranges for the Bat Path graphs: FAST swings only (bat speed 75+ mph), measured values only.

For every MLB contact swing (fouls + balls in play) with bat tracking:
  depth (pf)        = Statcast's contact in front of his center of mass - his stance depth (Savant's board, vs that
                      pitcher hand when it's there) -> inches in front of the plate (MLB's definition)
  bat speed, attack angle, attack direction (+ = oppo), swing path tilt  -- as measured
  vertical plane gap   = attack angle + the pitch's vertical approach angle at contact (0 = on the pitch's plane)
  horizontal plane gap = attack direction vs the pitch's horizontal approach angle at contact (0 = straight back
                         down the line the ball came in on; + = barrel aimed oppo of it)
Then, for every pitch group (All / Fastball / Breaking / Offspeed) x zone (All / 1-9): mean, SD and n in 2" depth
bins (-24" .. +32"), plus the overall mean / SD at contact. Bins with fewer than MIN_N swings are left empty (the
site falls back to the all-pitches range there).

    python -m pipeline.bat_norms 2026          # reads data/raw/statcast_2026.csv.gz -> data/calibration/bat_norms_2026.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "calibration"
FAST = 75.0
MIN_N = 15
EDGES = np.arange(-24, 34, 2.0)
METRICS = ["bat_speed", "attack_angle", "attack_direction", "swing_path_tilt", "gap_v", "gap_h", "vba", "hba"]
COLS = ["game_type", "batter", "stand", "p_throws", "pitch_type", "zone", "description", "bat_speed", "plate_x", "plate_z",
        "attack_angle", "attack_direction", "swing_path_tilt", "intercept_ball_minus_batter_pos_y_inches",
        "intercept_ball_minus_batter_pos_x_inches", "swing_length", "vx0", "vy0", "vz0", "ax", "ay", "az"]


def approach(df: pd.DataFrame, pf: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The pitch's vertical / horizontal approach angles (deg) where it was hit (pf inches in front of the plate):
    measured trajectory (Statcast's velocity + acceleration at y = 50 ft)."""
    n = lambda c: pd.to_numeric(df[c], errors="coerce")
    ay, vy0 = n("ay"), n("vy0")
    y = 17 / 12 + pf / 12
    a, b, c = 0.5 * ay, vy0, 50 - y
    t = (-b - np.sqrt(np.clip(b * b - 4 * a * c, 0, None))) / (2 * a)
    vx, vy, vz = n("vx0") + n("ax") * t, vy0 + ay * t, n("vz0") + n("az") * t
    return np.degrees(np.arctan2(vz, -vy)), np.degrees(np.arctan2(vx, -vy))


def plane(df: pd.DataFrame, pf: pd.Series) -> dict:
    """Per swing: the attack angle / direction that would MATCH the pitch (match_v = -VAA, match_h = the line straight
    back at the pitch, + = oppo) and the gaps to them (gap = measured - match), all measured."""
    vaa, haa = approach(df, pf)
    aa = pd.to_numeric(df["attack_angle"], errors="coerce")
    ad = pd.to_numeric(df["attack_direction"], errors="coerce")
    right = df["stand"].astype(str).eq("R")
    mv, mh = -vaa, -np.where(right, haa, -haa)
    return {"match_v": mv, "match_h": mh, "gap_v": aa - mv, "gap_h": ad - mh}


def gaps(df: pd.DataFrame, pf: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(vertical, horizontal) plane gaps per swing, measured: bat angles vs the pitch's angles at contact."""
    vaa, haa = approach(df, pf)
    aa = pd.to_numeric(df["attack_angle"], errors="coerce")
    ad = pd.to_numeric(df["attack_direction"], errors="coerce")
    right = df["stand"].astype(str).eq("R")
    return aa + vaa, ad + np.where(right, haa, -haa)


def swings(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    """Contact swings with depth + measured metrics (all speeds; `fast` flags 75+)."""
    from core import aar, hit_metrics as hm
    d = raw[raw["description"].astype(str).isin(["foul", "foul_tip", "hit_into_play"])].copy()
    d = d[pd.to_numeric(d["bat_speed"], errors="coerce").notna()
          & pd.to_numeric(d["intercept_ball_minus_batter_pos_y_inches"], errors="coerce").notna()]
    if d.empty:
        return d
    season_v = aar.stance_depths(season)
    try:
        from core import stance_feed
        views = stance_feed.views(season)
    except Exception:
        views = {}
    keys = list(zip(d["batter"].astype(int), d["stand"].astype(str), d["p_throws"].astype(str)))
    depth = [(views.get(f"vs_{h}", {}) or {}).get((b, s)) or season_v.get((b, s)) for b, s, h in keys]
    d["depth"] = pd.to_numeric(pd.Series(depth, index=d.index), errors="coerce")
    d = d[d["depth"].notna()]
    d["pf"] = pd.to_numeric(d["intercept_ball_minus_batter_pos_y_inches"], errors="coerce") - d["depth"]
    d["gap_v"], d["gap_h"] = gaps(d, d["pf"])
    # bat angles at contact (inferred from each swing's measured angles + tilt; his stance x off the plate)
    from core.bat_path import bat_angles
    sx = _stance_x(season)
    out_v, out_h = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for sd in ("R", "L"):
        m = (d["stand"].astype(str) == sd).to_numpy()
        if m.any():
            g = d[m]
            xs = g["batter"].astype(int).map(lambda b: sx.get((b, sd), 27.0)).to_numpy(float)
            v, h = [], []
            for x in np.unique(xs.round(0)):                      # batch by stance offset (rounded to an inch)
                mm = xs.round(0) == x
                gg = g[mm]
                vv, hh = bat_angles(gg["attack_angle"], gg["attack_direction"], gg["swing_path_tilt"], gg["stand"],
                                    gg["pf"], gg["depth"],
                                    12 * pd.to_numeric(gg.get("plate_x", pd.Series(0.0, index=gg.index)), errors="coerce"),
                                    12 * pd.to_numeric(gg.get("plate_z", pd.Series(2.5, index=gg.index)), errors="coerce"),
                                    {sd: x})
                idx = np.flatnonzero(m)[mm]
                out_v[idx], out_h[idx] = vv, hh
    d["vba"], d["hba"] = out_v, out_h
    d["group"] = hm.pitch_group(d).fillna("Other")
    d["fast"] = pd.to_numeric(d["bat_speed"], errors="coerce") >= FAST
    d["year"] = season
    for m in METRICS[:4]:
        d[m] = pd.to_numeric(d[m], errors="coerce")
    return d


def _stance_x(season: int) -> dict:
    """{(batter, side): inches his center of mass stands off the plate} from Savant's stance board."""
    f = ROOT / "data" / f"batting_stance_{season}.csv"
    try:
        df = pd.read_csv(f)
        side = df["side"] if "side" in df else df.get("bat_side", "R")
        return {(int(b), str(s)): float(x) for b, s, x in zip(df["id"], side, df["avg_batter_x_position"]) if pd.notna(x)}
    except Exception:
        return {}


def _stats(v: pd.Series) -> dict:
    v = v.dropna()
    if len(v) < MIN_N:
        return {"m": None, "s": None, "n": int(len(v))}
    return {"m": round(float(v.mean()), 2), "s": round(float(v.std()), 2), "n": int(len(v))}


def build(d: pd.DataFrame) -> dict:
    """Norms from FAST swings only."""
    from core.bat_path import TGRID, time_band, time_curves
    f = d[d["fast"]]
    centers = ((EDGES[:-1] + EDGES[1:]) / 2).tolist()
    out = {"fast_mph": FAST, "bins": centers, "tgrid": TGRID.tolist(), "keys": {}}
    # every fast swing's angles along its own drawn path (the time view), computed once
    curves = {}
    if len(f):
        n_ = lambda c: pd.to_numeric(f[c], errors="coerce") if c in f else pd.Series(np.nan, index=f.index)
        sx = _stance_x(int(f["year"].iloc[0])) if "year" in f else {}
        xs = np.asarray([sx.get((int(b), str(s_)), 27.0) for b, s_ in zip(f["batter"], f["stand"])], float).round(0)
        curves = {k: np.full((len(f), len(TGRID)), np.nan) for k in ("attack_angle", "attack_direction", "vba", "hba")}
        for x in np.unique(xs):
            mm = xs == x
            g_ = f[mm]
            cv = time_curves(g_["attack_angle"], g_["attack_direction"], g_["swing_path_tilt"], g_["bat_speed"],
                             n_("swing_length")[mm], g_["stand"], g_["pf"], g_["depth"],
                             n_("intercept_ball_minus_batter_pos_x_inches")[mm], 12 * n_("plate_x")[mm].fillna(0),
                             12 * n_("plate_z")[mm].fillna(2.5), {"R": x, "L": x})
            for k in curves:
                curves[k][mm] = cv[k]
    zone = pd.to_numeric(f["zone"], errors="coerce")
    for g in ("All", "Fastball", "Breaking", "Offspeed"):
        for z in ["All"] + list(range(1, 10)):
            m = pd.Series(True, index=f.index)
            if g != "All":
                m &= f["group"].eq(g)
            if z != "All":
                m &= zone.eq(z)
            s = f[m]
            if len(s) < MIN_N:
                continue
            b = pd.cut(s["pf"], EDGES, labels=False)
            k = {"n": int(len(s)), "contact": {mt: _stats(s[mt]) for mt in METRICS + ["pf"]}, "depth": {}}
            for mt in METRICS:
                grp = s.groupby(b)[mt]
                mean, sd, n = grp.mean(), grp.std(), grp.count()
                k["depth"][mt] = {
                    "m": [round(float(mean[i]), 2) if i in mean and n[i] >= MIN_N else None for i in range(len(centers))],
                    "s": [round(float(sd[i]), 2) if i in sd and n[i] >= MIN_N else None for i in range(len(centers))],
                    "n": [int(n[i]) if i in n else 0 for i in range(len(centers))]}
            if curves:
                mk = m.to_numpy()
                k["time"] = time_band({mt: v[mk] for mt, v in curves.items()}, MIN_N)
            out["keys"][f"{g}|{z}"] = k
    return out


def run(season: int, raw: pd.DataFrame | None = None) -> Path:
    if raw is None:
        f = ROOT / "data" / "raw" / f"statcast_{season}.csv.gz"
        head = pd.read_csv(f, nrows=0).columns
        raw = pd.read_csv(f, usecols=[c for c in COLS if c in head], low_memory=False)
    d = swings(raw, season)
    norms = build(d)
    norms["season"] = season
    norms["swings"] = {"contact": int(len(d)), "fast": int(d["fast"].sum()) if len(d) else 0}
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"bat_norms_{season}.json"
    dest.write_text(json.dumps(norms, separators=(",", ":")))
    print(f"bat norms {season}: {norms['swings']['fast']:,} fast swings of {norms['swings']['contact']:,} contact "
          f"swings -> {dest.name} ({len(norms['keys'])} pitch-group x zone ranges)")
    return dest


if __name__ == "__main__":
    for a in sys.argv[1:] or ["2026"]:
        run(int(a))
