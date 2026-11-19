"""Calibrate + validate the swing simulator's commit model against real MLB swings.

For a sample of hitters (most swings first): build each one's Bat Path scene exactly as the page does (his average
swing, his measured path), then throw him the REAL pitches he swung at (their own release, speed, movement and
location from the season file). The commit model decides each one: at his commit point he reads the pitch and aims +
times his swing where a fastball with that look would arrive; then late adjustment, his spread, contact / foul tip /
whiff, and for contact the collision -> exit velo, launch, spray -> fair or foul.

  calibrate   the model's few numbers (speed read, late timing + barrel adjustment, aim / timing spread, foul-tip
              graze) are fit so whiff and foul rates BY PITCH TYPE match what these hitters actually did
  validate    then, hitter by hitter: predicted vs actual whiff %, foul %, in-play %, exit velo on contact and pull %
              -> how well the model tells hitters apart (correlations, average misses), written to the report

    python -m core.swing_cal 2026            # ~5-10 min; reads the bat cube + data/raw/statcast_2026.csv.gz
    python -m core.swing_cal 2026 60         # quicker: 60 hitters

Writes data/calibration/swing_cal_<season>.json (params + report) -- the page picks it up automatically.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import collision as co
from core import swing_sim as ss

ROOT = Path(__file__).resolve().parent.parent
KIN = ["game_pk", "at_bat_number", "pitch_number", "vx0", "vy0", "vz0", "ax", "ay", "az", "pfx_x", "pfx_z",
       "release_extension", "release_speed", "release_pos_x", "release_pos_z", "plate_x", "plate_z"]
GRIDS = {"w_speed": [0.0, 0.25, 0.5, 0.75, 1.0],
         "adj_ms": [0.0, 6.0, 12.0, 18.0, 24.0, 30.0],
         "reach_max": [0.0, 0.75, 1.5, 2.25, 3.0, 4.0],
         "reach_frac": [0.25, 0.5, 0.75],
         "r_tip": [1.8, 2.0, 2.2, 2.4, 2.6],
         "r_foul": [0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 9.0],
         "aim_sd": [0.4, 0.8, 1.2, 1.6, 2.2, 3.0],
         "time_sd_ms": [3.0, 6.0, 9.0, 12.0, 16.0]}


def _say(*a):
    print(*a, flush=True)


def load(season: int) -> pd.DataFrame:
    """Every competitive swing (the bat cube) + its pitch's kinematics (the raw season file)."""
    from pipeline import bat_cube
    cube = bat_cube.load(season)
    if cube.empty:
        raise SystemExit(f"No bat cube for {season} -- run: python -m pipeline.bat_cube {season}")
    raw_f = ROOT / "data" / "raw" / f"statcast_{season}.csv.gz"
    if not raw_f.exists():
        raise SystemExit(f"Missing {raw_f}")
    head = pd.read_csv(raw_f, nrows=0).columns
    raw = pd.read_csv(raw_f, usecols=[c for c in KIN if c in head], low_memory=False)
    keys = ["game_pk", "at_bat_number", "pitch_number"]
    raw = raw.drop_duplicates(keys)
    drop = [c for c in KIN if c in cube.columns and c not in keys]
    d = cube.drop(columns=drop).merge(raw, on=keys, how="inner")
    return d.dropna(subset=["vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z", "bat_speed"])


def hitter_scene(rows: pd.DataFrame):
    """His Bat Path scene, built the way the page builds it (all pitches, all counts, his main side)."""
    from core import bat_path as bp
    stand = rows["stand"].mode().iloc[0]
    r = rows[rows["stand"] == stand]
    sw = bp.swings(r, pd.DataFrame({"pf": r["pf"]}, index=r.index))
    if sw.empty:
        return None, None, stand
    depth = float(pd.to_numeric(r["depth"], errors="coerce").median()) if "depth" in r else 27.0
    if not np.isfinite(depth):
        depth = 27.0
    sc = bp.model_scene(sw, None, "All", "All", None, stand, depth, None, sw["ptype"].unique())
    return sc, sw, stand


def _kin(row) -> dict:
    ext = float(row["release_extension"]) if pd.notna(row.get("release_extension")) else 6.2
    return {k: float(row[k]) for k in ("vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z")} | \
        {"release_pos_y": 60.5 - ext}


def real_raws(sc: dict, rows: pd.DataFrame, w: float, p: dict) -> dict:
    """meet_raw for each real pitch he swung at: he aims + times off its commit-point look (speed read w)."""
    from core.bat_path import synth_pitch
    bat = ss.Bat(sc)
    cs = ss.commit_s(sc, p)
    C = np.asarray(sc["contact"], float)
    yC = float(C[1])
    fbs = {h: synth_pitch(*ss.FB_LOOK[h], h, 0.0, 2.5) for h in ("R", "L")}
    out = []
    for _, row in rows.iterrows():
        d = _kin(row)
        fb = fbs.get(str(row["p_throws"]), fbs["R"])
        try:
            tc = ss._t_front(d, 0.0) - cs
            A, th = ss.aim(d, tc, yC, fb, w)
            shift = (A - C) * np.array([1.0, 0.0, 1.0])
            ts, xyz, v = ss._pitch_samples(d, tc + th - 0.06, tc + th + 0.06, 480)
            out.append(ss.meet_raw(bat, xyz - shift, ts - tc - th, v))
        except Exception:
            out.append(None)
    return out


def _stack(raws: list) -> dict:
    ok = np.array([r is not None for r in raws])
    K = len(ss.D_SHIFTS)
    n = len(raws)
    D = np.full((n, K), np.nan)
    lam = np.zeros((n, K))
    bv, vb, a = (np.zeros((n, K, 3)) for _ in range(3))
    for i, r in enumerate(raws):
        if r is None:
            continue
        D[i], lam[i], bv[i], vb[i], a[i] = r["D"], r["lam"], r["bv"], r["vb"], r["a"]
    return {"D": D, "lam": lam, "bv": bv, "vb": vb, "a": a, "ok": ok}


def outcome(R: dict, p: dict, z_aim, z_time, cons, stand, q: float) -> pd.DataFrame:
    """Vectorised commit-model outcome for stacked raws: result (whiff / foul / fair), ev, la, spray."""
    S = ss.D_SHIFTS.astype(float)
    tau = (z_time * p["time_sd_ms"] * cons)[:, None]
    D = R["D"] + (z_aim * p["aim_sd"] * cons)[:, None]
    win = np.abs(S[None, :] - tau) <= p["adj_ms"] + 1e-9
    near = np.abs(S[None, :] - tau) == np.min(np.abs(S[None, :] - tau), axis=1, keepdims=True)
    win = np.where(win.any(1, keepdims=True), win, near)
    cand = np.where(win & np.isfinite(D), np.abs(D), np.inf)
    m = np.argmin(cand, axis=1)
    i = np.arange(len(D))
    miss = cand[i, m]
    left = miss - np.minimum(p["reach_max"], p["reach_frac"] * miss)
    res = np.where(~np.isfinite(miss) | (left >= ss.R_MEET), "whiff",
                   np.where(left > p["r_tip"], "whiff", np.where(left > p["r_foul"], "foul", "hit")))
    out = pd.DataFrame({"result": res, "ev": np.nan, "la": np.nan, "spray": np.nan})
    h = np.where(res == "hit")[0]
    if len(h):
        Dh = np.sign(D[h, m[h]]) * left[h]
        cr = co.collide(R["bv"][h, m[h]], R["vb"][h, m[h]], R["a"][h, m[h]], Dh, q)
        ev = cr["ev"] * ss.eff_along(R["lam"][h, m[h]])
        sp = ss.bearing(ev, cr["la"], cr["spray"], np.asarray(stand)[h])
        fair = (np.abs(sp) <= 45) & (cr["la"] > -20) & (cr["la"] < 80)
        out.loc[h, "ev"], out.loc[h, "la"], out.loc[h, "spray"] = ev, cr["la"], sp
        out.loc[h, "result"] = np.where(fair, "fair", "foul")
    return out


def _actual_spray(r: pd.DataFrame) -> pd.Series:
    x = pd.to_numeric(r.get("hc_x"), errors="coerce") - 125.42
    y = 198.27 - pd.to_numeric(r.get("hc_y"), errors="coerce")
    return np.degrees(np.arctan2(x, y))


def _pull(spray, stand) -> np.ndarray:
    s = np.asarray(spray, float)
    return np.where(np.asarray(stand) == "L", s, -s) > 15


def score(sim: pd.DataFrame, act: pd.DataFrame) -> float:
    """Squared misses of whiff % and foul % per swing, by pitch type, weighted by swings."""
    e = 0.0
    for k, g in act.groupby("ptype"):
        s = sim.loc[g.index]
        w = len(g)
        for c in ("whiff", "foul"):
            e += w * ((s["result"] == c).mean() - (g["result"] == c).mean()) ** 2
    return e / max(len(act), 1)


def run(season: int, n_hitters: int = 120, per_hitter: int = 250, seed: int = 7) -> dict:
    t0 = time.time()
    q = (co.load(season).get("measured") or {}).get("q", co.Q0)
    d = load(season)
    _say(f"[{season}] {len(d):,} swings with pitch kinematics, {d['batter'].nunique():,} hitters "
         f"({(time.time() - t0) / 60:.1f} min)")
    top = d["batter"].value_counts()
    top = top[top >= 300].index[:n_hitters]
    rng = np.random.default_rng(seed)
    pf_sd = d.groupby("batter")["pf"].std().median()
    p0 = dict(ss.CAL_DEFAULT) | {"league_pf_sd": float(pf_sd) if np.isfinite(pf_sd) else 7.0}
    # build each hitter's scene + the real pitches he swung at
    H = []
    for b in top:
        rows = d[d["batter"] == b]
        try:
            sc, sw, stand = hitter_scene(rows)
        except Exception as e:
            _say(f"  skip {b}: {type(e).__name__}: {e}")
            continue
        if sc is None:
            continue
        r = rows[(rows["stand"] == stand)]
        r = r.sample(min(per_hitter, len(r)), random_state=seed)
        from core.bat_path import swing_result
        r = r.assign(result=swing_result(r), ptype=r["pitch_type"].astype(str))
        r = r[r["result"].isin(["whiff", "foul", "fair"])]
        cons = ss.consistency(sw, p0)
        H.append({"batter": int(b), "stand": stand, "sc": sc, "rows": r, "cons": cons})
        if len(H) % 20 == 0:
            _say(f"  {len(H)} hitters built ({(time.time() - t0) / 60:.1f} min)")
    act = pd.concat([h["rows"].assign(batter=h["batter"], stand=h["stand"]) for h in H], ignore_index=True)
    cons = np.concatenate([np.full(len(h["rows"]), h["cons"]) for h in H])
    stand = act["stand"].to_numpy()
    za, zt = rng.standard_normal(len(act)), rng.standard_normal(len(act))
    _say(f"  {len(H)} hitters, {len(act):,} real swings to simulate")
    # geometry for each speed-read value (the only parameter that changes where he aims)
    stacks = {}
    for w in GRIDS["w_speed"]:
        raws = []
        for h in H:
            raws += real_raws(h["sc"], h["rows"], w, p0)
        stacks[w] = _stack(raws)
        _say(f"  geometry for speed read {w:.2f} ({(time.time() - t0) / 60:.1f} min)")
    # coordinate descent over the grids
    p = dict(p0)
    best = score(outcome(stacks[p["w_speed"]] if p["w_speed"] in stacks else stacks[0.5], p, za, zt, cons, stand, q),
                 act)
    for it in range(3):
        moved = False
        for k, grid in GRIDS.items():
            for v in grid:
                trial = dict(p) | {k: v}
                e = score(outcome(stacks[trial["w_speed"]], trial, za, zt, cons, stand, q), act)
                if e < best - 1e-7:
                    best, p, moved = e, trial, True
        _say(f"  pass {it + 1}: error {best:.5f}  " + ", ".join(f"{k}={p[k]}" for k in GRIDS))
        if not moved:
            break
    sim = outcome(stacks[p["w_speed"]], p, za, zt, cons, stand, q)
    rep = report(sim, act)
    params = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in p.items() if k != "fitted"}
    out = {"season": season, "params": params, "hitters": len(H), "swings": int(len(act)), "error": round(best, 6),
           "report": rep}
    ss.OUT.mkdir(parents=True, exist_ok=True)
    (ss.OUT / f"swing_cal_{season}.json").write_text(json.dumps(out, indent=1))
    _say(text(out))
    _say(f"\nSaved data/calibration/swing_cal_{season}.json ({(time.time() - t0) / 60:.1f} min)")
    return out


def report(sim: pd.DataFrame, act: pd.DataFrame) -> dict:
    """By pitch type (pooled) + hitter by hitter: predicted vs actual."""
    s = sim.set_index(act.index)
    a = act.assign(p_result=s["result"], p_ev=s["ev"], p_spray=s["spray"])
    by_type = []
    for k, g in a.groupby("ptype"):
        if len(g) < 100:
            continue
        by_type.append({"type": k, "n": int(len(g)),
                        **{f"act_{c}": round(float((g["result"] == c).mean()), 3) for c in ("whiff", "foul", "fair")},
                        **{f"sim_{c}": round(float((g["p_result"] == c).mean()), 3) for c in ("whiff", "foul", "fair")}})
    hit = []
    for b, g in a.groupby("batter"):
        st_ = g["stand"].iloc[0]
        fair_a, fair_p = g[g["result"] == "fair"], g[g["p_result"] == "fair"]
        hit.append({"batter": int(b), "n": int(len(g)),
                    "act_whiff": (g["result"] == "whiff").mean(), "sim_whiff": (g["p_result"] == "whiff").mean(),
                    "act_foul": (g["result"] == "foul").mean(), "sim_foul": (g["p_result"] == "foul").mean(),
                    "act_fair": (g["result"] == "fair").mean(), "sim_fair": (g["p_result"] == "fair").mean(),
                    "act_ev": pd.to_numeric(fair_a.get("launch_speed"), errors="coerce").mean(),
                    "sim_ev": fair_p["p_ev"].mean(),
                    "act_pull": float(np.nanmean(_pull(_actual_spray(fair_a), [st_] * len(fair_a))))
                    if len(fair_a) else np.nan,
                    "sim_pull": float(np.nanmean(_pull(fair_p["p_spray"], [st_] * len(fair_p)))) if len(fair_p) else np.nan})
    h = pd.DataFrame(hit)
    val = {}
    for c in ("whiff", "foul", "fair", "ev", "pull"):
        x, y = h[f"sim_{c}"], h[f"act_{c}"]
        ok = x.notna() & y.notna()
        if ok.sum() >= 10:
            val[c] = {"corr": round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 3),
                      "mae": round(float((x[ok] - y[ok]).abs().mean()), 3),
                      "bias": round(float((x[ok] - y[ok]).mean()), 3)}
    return {"by_type": by_type, "hitters": val}


def text(out: dict) -> str:
    p, r = out["params"], out["report"]
    lines = [f"\nCommit model, {out['season']}: {out['hitters']} hitters, {out['swings']:,} real swings",
             f"  speed read {p['w_speed']:.2f} | late timing +/-{p['adj_ms']:.0f} ms | barrel reach "
             f"{p['reach_frac']:.2f} of the miss up to {p['reach_max']:.2f}\" | glancing foul past {p['r_foul']:.1f}\", "
             f"foul tip past {p['r_tip']:.1f}\" | "
             f"spread: aim {p['aim_sd']:.1f}\", timing {p['time_sd_ms']:.0f} ms | commit = swing time + "
             f"{p['decide_s'] * 1000:.0f} ms",
             "\n  By pitch type (per swing)      whiff  act / sim     foul  act / sim     in play  act / sim"]
    for t in r["by_type"]:
        lines.append(f"    {t['type']:<4} n={t['n']:<7,}             {t['act_whiff']:.3f} / {t['sim_whiff']:.3f}"
                     f"       {t['act_foul']:.3f} / {t['sim_foul']:.3f}       {t['act_fair']:.3f} / {t['sim_fair']:.3f}")
    lines.append("\n  Hitter by hitter (predicted vs actual): correlation / average miss / bias")
    for c, v in r["hitters"].items():
        lines.append(f"    {c:<6} r = {v['corr']:+.2f}   miss {v['mae']:.3f}   bias {v['bias']:+.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    a = sys.argv[1:]
    run(int(a[0]) if a else 2026, int(a[1]) if len(a) > 1 else 120)
