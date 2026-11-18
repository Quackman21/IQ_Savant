"""Swing simulator: every possible contact, priced in runs (the contact layer), on top of core.collision.

  bearing    where the ball actually goes. The physics gives the launch direction; Statcast's spray comes from where
             the ball was fielded / landed (hooks, slices, how grounders get fielded). A correction learned from MLB
             balls in play (exit velo, launch angle, physics spray, bat side) -- like Driveline's bearing model.
  price      runs for a batted ball: our spray-aware xwOBA (models/xwoba_iq: EV + LA + spray + side) on the run
             scale: runs = xwOBA / wOBA scale - out value (an out ~ -0.27, a homer ~ +1.4).
  contact layer   for one swing vs one pitch: every offset around the barrel (undercut / topped) x every point along
             it (barrel efficiency: 100% = the sweet spot), simulated and priced -> a heat map, + where HIS contacts
             actually land on it (his balls in play, offset solved from launch angle, efficiency from exit velo).

    python -m core.swing_sim 2026        # trains the bearing correction on the bat cube + reports it
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core import collision as co

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "models" / "artifacts" / "bearing.joblib"
OUT = ROOT / "data" / "calibration"
WOBA_SCALE = 1.24
OUT_RUNS = 0.27
D_GRID = np.round(np.arange(-1.5, 2.01, 0.1), 2)            # inches: - = topped, + = undercut
EFF_GRID = np.round(np.arange(0.70, 1.001, 0.02), 2)        # share of the sweet spot's exit velo
_cache: dict = {}


# ----------------------------------------------------------------------------- bearing
def _feats(ev, la, spray_phys, stand) -> np.ndarray:
    left = (np.asarray(stand) == "L").astype(float)
    sp = np.asarray(spray_phys, float)
    pull = np.where(left == 1, sp, -sp)                     # + = pulled, both sides
    return np.column_stack([np.asarray(ev, float), np.asarray(la, float), pull, left])


def train_bearing(d: pd.DataFrame, q: float) -> dict:
    """Fit observed spray (pull-adjusted) from the physics' spray + exit velo + launch angle + side, on balls in
    play; held-out median miss vs the raw physics."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    vb = co.ball_velocity(d["release_speed"], d["vaa"], d["haa"])
    vB = co.bat_velocity(d["bat_speed"], d["attack_angle"], d["attack_direction"], d["stand"])
    ax = d[["ax_m", "ay_m", "az_m"]].to_numpy()
    D = co.solve_offset(vb, vB, ax, d["launch_angle"].to_numpy(), q)
    r = co.collide(vb, vB, ax, np.nan_to_num(D), q)
    X = _feats(d["launch_speed"], d["launch_angle"], r["spray"], d["stand"])
    left = d["stand"].astype(str).eq("L").to_numpy()
    y = np.where(left, d["spray_obs"], -d["spray_obs"])
    ok = np.isfinite(X).all(1) & np.isfinite(y) & np.isfinite(D)
    X, y, phys = X[ok], y[ok], X[ok, 2]
    rng = np.random.default_rng(1)
    te = rng.random(len(y)) < 0.2
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31)
    m.fit(X[~te], y[~te])
    pred = m.predict(X[te])
    out = {"n": int(len(y)), "physics_mae": round(float(np.median(np.abs(phys[te] - y[te]))), 2),
           "model_mae": round(float(np.median(np.abs(pred - y[te]))), 2),
           "physics_corr": round(float(np.corrcoef(phys[te], y[te])[0, 1]), 3),
           "model_corr": round(float(np.corrcoef(pred, y[te])[0, 1]), 3)}
    m.fit(X, y)
    import joblib
    ART.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m, "q": q}, ART)
    _cache.pop("b", None)
    return out


def bearing(ev, la, spray_phys, stand) -> np.ndarray:
    """Spray (deg, + = 1B side) the ball ends up with; the raw physics until the correction is trained."""
    if "b" not in _cache:
        _cache["b"] = None
        if ART.exists():
            try:
                import joblib
                _cache["b"] = joblib.load(ART)
            except Exception:
                pass
    b = _cache["b"]
    sp = np.asarray(spray_phys, float)
    if b is None:
        return sp
    X = _feats(ev, la, sp, stand)
    ok = np.isfinite(X).all(1)
    pull = np.full(len(sp), np.nan)
    if ok.any():
        pull[ok] = b["model"].predict(X[ok])
    left = np.asarray(stand) == "L"
    return np.where(ok, np.where(left, pull, -pull), sp)


# ----------------------------------------------------------------------------- price
def price(ev, la, spray, stand) -> np.ndarray:
    """Runs per batted ball from our spray-aware xwOBA (Savant-style EV + LA curve if it isn't trained)."""
    ev, la, spray = (np.asarray(x, float) for x in (ev, la, spray))
    left = np.asarray(stand) == "L"
    df = pd.DataFrame({"launch_speed": ev, "launch_angle": la, "stand": np.where(left, "L", "R"),
                       "hc_x": 125.42 + 200 * np.sin(np.radians(spray)), "hc_y": 198.27 - 200 * np.cos(np.radians(spray)),
                       "type": "X", "bb_type": None, "game_date": "2026-06-01"})
    x = None
    try:
        from models import xwoba_iq
        x = xwoba_iq.score_batted_balls(df).to_numpy(float)
    except Exception:
        x = None
    if x is None or not np.isfinite(x).any():
        x = _fallback_xwoba(ev, la)
    else:
        x = np.where(np.isfinite(x), x, _fallback_xwoba(ev, la))
    return x / WOBA_SCALE - OUT_RUNS


def _fallback_xwoba(ev, la) -> np.ndarray:
    """A smooth stand-in for EV + LA xwOBA (barrel zone high, weak / topped low) -- only until xwoba_iq is there."""
    ev, la = np.asarray(ev, float), np.asarray(la, float)
    hard = 1 / (1 + np.exp(-(ev - 98) / 4))
    air = np.exp(-0.5 * ((la - 26) / 9) ** 2)
    liner = np.exp(-0.5 * ((la - 12) / 8) ** 2)
    return np.clip(0.08 + 1.6 * hard * air + 0.55 * liner * (0.4 + 0.6 * hard), 0, 2.0)


# ----------------------------------------------------------------------------- the contact layer
def contact_layer(v_ball, v_bat, axis, stand: str, q: float | None = None, d_grid=D_GRID, eff_grid=EFF_GRID) -> dict:
    """One swing vs one pitch: every offset x barrel efficiency, simulated and priced.
    v_ball, v_bat: (3,) mph; axis: (3,) along the bat. -> grids of ev, la, spray, runs [eff x D]."""
    q = q if q is not None else (co.load(2026).get("measured") or {}).get("q", co.Q0)
    nD = len(d_grid)
    r = co.collide(np.tile(v_ball, (nD, 1)), np.tile(v_bat, (nD, 1)), np.tile(axis, (nD, 1)), d_grid, q)
    ev = np.outer(eff_grid, r["ev"])
    la = np.tile(r["la"], (len(eff_grid), 1))
    sp_phys = np.tile(r["spray"], (len(eff_grid), 1))
    st = np.full(ev.size, stand)
    spray = bearing(ev.ravel(), la.ravel(), sp_phys.ravel(), st).reshape(ev.shape)
    runs = price(ev.ravel(), la.ravel(), spray.ravel(), st).reshape(ev.shape)
    return {"D": list(map(float, d_grid)), "eff": list(map(float, eff_grid)), "ev": ev.round(1).tolist(),
            "la": la.round(1).tolist(), "spray": spray.round(1).tolist(), "runs": runs.round(3).tolist(),
            "spin": np.round(r["spin"]).tolist()}


def his_contacts(rows: pd.DataFrame, q: float | None = None) -> pd.DataFrame:
    """Where his balls in play landed on the barrel: offset (solved from launch angle) + efficiency (exit velo vs
    the sweet spot's at that offset). rows = collision.frame() rows for one hitter."""
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["D", "eff"])
    q = q if q is not None else (co.load(2026).get("measured") or {}).get("q", co.Q0)
    vb = co.ball_velocity(rows["release_speed"], rows["vaa"], rows["haa"])
    vB = co.bat_velocity(rows["bat_speed"], rows["attack_angle"], rows["attack_direction"], rows["stand"])
    ax = rows[["ax_m", "ay_m", "az_m"]].to_numpy()
    D = co.solve_offset(vb, vB, ax, rows["launch_angle"].to_numpy(), q)
    r = co.collide(vb, vB, ax, np.nan_to_num(D), q)
    return pd.DataFrame({"D": D, "eff": rows["launch_speed"].to_numpy() / r["ev"]}, index=rows.index)


def run(season: int, sample: int = 120000) -> dict:
    from core import path_fit
    from pipeline import bat_cube
    lr = (path_fit.load(season) or {}).get("league") or {}
    q = (co.load(season).get("measured") or {}).get("q", co.Q0)
    d = co.frame(bat_cube.load(season), lr)
    d = d[d["spray_obs"].notna()]
    if len(d) > sample:
        d = d.sample(sample, random_state=5)
    b = train_bearing(d, q)
    # what the contact layer looks like for a typical swing (sanity: the best cells should be ~barrels)
    vb = co.ball_velocity([93.0], [-6.0], [0.0])[0]
    vB = co.bat_velocity([72.0], [10.0], [0.0], ["R"])[0]
    ax = co.axis_measured([10.0], [0.0], ["R"], [34.0], lr)[0]
    cl = contact_layer(vb, vB, ax, "R", q)
    R = np.array(cl["runs"])
    i, j = np.unravel_index(np.nanargmax(R), R.shape)
    his = his_contacts(d, q)
    out = {"season": season, "q": q, "bearing": b,
           "typical_best": {"D": cl["D"][j], "eff": cl["eff"][i], "ev": cl["ev"][i][j], "la": cl["la"][i][j],
                            "runs": cl["runs"][i][j]},
           "league_contacts": {"D_p25_p50_p75": [round(float(x), 2) for x in np.nanpercentile(his["D"], [25, 50, 75])],
                               "eff_p25_p50_p75": [round(float(x), 3) for x in np.nanpercentile(his["eff"], [25, 50, 75])]}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"swing_sim_{season}.json").write_text(json.dumps(out, indent=1))
    print(f"\n[{season}] Bearing (spray) correction on {b['n']:,} balls in play, held-out median miss: physics "
          f"{b['physics_mae']:.1f} deg -> corrected {b['model_mae']:.1f} deg (corr {b['physics_corr']:.2f} -> "
          f"{b['model_corr']:.2f})")
    t = out["typical_best"]
    print(f"  typical swing (72 mph, +10 deg) vs a 93 mph fastball: best contact {t['D']:+.1f} in undercut at "
          f"{100 * t['eff']:.0f}% efficiency -> {t['ev']:.0f} mph at {t['la']:.0f} deg = {t['runs']:+.2f} runs")
    lc = out["league_contacts"]
    print(f"  MLB balls in play: offset middle half {lc['D_p25_p50_p75'][0]:+.2f} / {lc['D_p25_p50_p75'][1]:+.2f} / "
          f"{lc['D_p25_p50_p75'][2]:+.2f} in, efficiency {100 * lc['eff_p25_p50_p75'][0]:.0f}% / "
          f"{100 * lc['eff_p25_p50_p75'][1]:.0f}% / {100 * lc['eff_p25_p50_p75'][2]:.0f}%")
    return out


if __name__ == "__main__":
    for a in sys.argv[1:] or ["2026"]:
        run(int(a))
