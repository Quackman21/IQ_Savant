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
def _model_xwoba(ev, la, spray, stand) -> np.ndarray | None:
    """Our spray-aware xwOBA for a batch of batted balls (None if it isn't trained). Slow: one call runs the
    model over every park x spray blur, so it is only used to build the price grid."""
    ev, la, spray = (np.asarray(x, float) for x in (ev, la, spray))
    left = np.asarray(stand) == "L"
    df = pd.DataFrame({"launch_speed": ev, "launch_angle": la, "stand": np.where(left, "L", "R"),
                       "hc_x": 125.42 + 200 * np.sin(np.radians(spray)), "hc_y": 198.27 - 200 * np.cos(np.radians(spray)),
                       "type": "X", "bb_type": None, "game_date": "2026-06-01"})
    try:
        from models import xwoba_iq
        if xwoba_iq.load() is None:
            return None
        x = xwoba_iq.score_batted_balls(df).to_numpy(float)
    except Exception:
        return None
    return x if np.isfinite(x).any() else None


# the price grid: the model scored once over EV x LA x spray x side, then read by interpolation
G_EV = np.arange(40.0, 122.0, 3.0)
G_LA = np.arange(-30.0, 82.0, 4.0)
G_SP = np.arange(-48.0, 49.0, 8.0)
_grid: dict = {}


def _grid_file() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "calibration" / "price_grid.npz"


def price_grid(build: bool = True):
    """{'R': interpolator, 'L': interpolator} over (ev, la, spray) -> xwOBA, or None without the model.
    Rebuilt when the xwOBA model changes (about a minute, once; keyed to its contents)."""
    if "g" in _grid:
        return _grid["g"]
    from scipy.interpolate import RegularGridInterpolator
    try:
        from models import xwoba_iq
        art = xwoba_iq.ARTIFACT
        import hashlib                             # keyed to the model's contents (same on every machine)
        stamp = float(int(hashlib.md5(art.read_bytes()).hexdigest()[:12], 16)) if art.exists() else None
    except Exception:
        stamp = None
    if stamp is None:
        _grid["g"] = None
        return None
    f = _grid_file()
    vals = None
    try:
        z = np.load(f)
        if float(z["stamp"]) == stamp and z["vals"].shape == (2, len(G_EV), len(G_LA), len(G_SP)):
            vals = z["vals"]
    except Exception:
        vals = None
    if vals is None:
        if not build:
            return None
        E, A, S = np.meshgrid(G_EV, G_LA, G_SP, indexing="ij")
        vals = np.zeros((2,) + E.shape)
        for k, side in enumerate("RL"):
            x = _model_xwoba(E.ravel(), A.ravel(), S.ravel(), np.full(E.size, side))
            if x is None:
                _grid["g"] = None
                return None
            x = np.where(np.isfinite(x), x, _fallback_xwoba(E.ravel(), A.ravel()))
            vals[k] = x.reshape(E.shape)
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, vals=vals, stamp=stamp)
        except Exception:
            pass
    _grid["g"] = {s: RegularGridInterpolator((G_EV, G_LA, G_SP), vals[k], bounds_error=False, fill_value=None)
                  for k, s in enumerate("RL")}
    return _grid["g"]


def price(ev, la, spray, stand) -> np.ndarray:
    """Runs per batted ball from our spray-aware xwOBA (read off the price grid), or a Savant-style EV + LA curve
    if the model isn't trained."""
    ev, la, spray = (np.atleast_1d(np.asarray(x, float)) for x in (ev, la, spray))
    st = np.broadcast_to(np.asarray(stand), ev.shape)
    g = price_grid()
    if g is None:
        x = _fallback_xwoba(ev, la)
    else:
        pts = np.column_stack([np.clip(ev, G_EV[0], G_EV[-1]), np.clip(la, G_LA[0], G_LA[-1]),
                               np.clip(spray, G_SP[0], G_SP[-1])])
        x = np.where(st == "L", g["L"](pts), g["R"](pts))
        x = np.where(np.isfinite(x), np.clip(x, 0, 2.1), _fallback_xwoba(ev, la))
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


# ============================================================================= timing + arsenal layers (3D sim)
R_MEET = co.R_BALL + co.R_BAT                   # ball centre within this of the bat's centre line = contact
BARREL = (-6.0, 22.0)                           # inches along the bat from the sweet spot (- = toward the tip)
COMMIT_S = 0.175                                # a typical commit point (his own: commit_s())
WHIFF_RUNS, FOUL_RUNS = -0.06, -0.03            # a swing-and-miss / a foul, on the run scale (average count)
FRONT_FT = 17 / 12
# a typical MLB pitch mix by pitcher hand (usage, velo mean/sd, IVB, HB arm side; inches) -- public Savant averages
MIX = {"FF": (0.33, 94.5, 2.0, 15.5, 8.0), "SI": (0.14, 93.8, 2.0, 8.0, 15.0), "FC": (0.08, 89.0, 2.5, 9.0, -2.0),
       "SL": (0.14, 85.5, 2.5, 1.5, -4.0), "ST": (0.08, 82.0, 2.5, 1.0, -14.0), "CH": (0.12, 86.0, 2.5, 6.0, 14.0),
       "CU": (0.08, 79.0, 3.0, -10.0, -8.0), "FS": (0.03, 86.5, 2.0, 3.0, 10.0)}
PT_COLOR = {"FF": "#d22d49", "SI": "#fe9d00", "FC": "#933f2c", "SL": "#c3bd0e", "ST": "#ddb33a", "CH": "#1dbe3a",
            "CU": "#00d1ed", "FS": "#3bacac"}


def _sl_in(sc: dict):
    """His Statcast swing length in inches (None if missing)."""
    sl = (sc.get("read") or {}).get("swing_length")
    try:
        sl = float(sl)
    except (TypeError, ValueError):
        return None
    return sl * 12 if np.isfinite(sl) and sl > 0 else None


def eff_along(lam) -> np.ndarray:
    """Share of the sweet spot's exit velo at lam inches along the bat from it (a smooth fall-off on both sides;
    ~90% at 3" -- matches MLB's median efficiency). To be refit once contact spots can be measured."""
    lam = np.asarray(lam, float)
    return np.clip(1 - 0.011 * lam ** 2, 0.5, 1.0)


def _traj(d: dict, ts) -> np.ndarray:
    """Pitch centre (scene inches: x, y in front of the plate's front edge, z) at times ts (s, 0 = at y = 50 ft)."""
    ts = np.asarray(ts, float)
    a, b, c = 0.5 * d["ay"], d["vy0"], 50 - FRONT_FT
    tf = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    x = d["plate_x"] + d["vx0"] * (ts - tf) + 0.5 * d["ax"] * (ts ** 2 - tf ** 2)
    z = d["plate_z"] + d["vz0"] * (ts - tf) + 0.5 * d["az"] * (ts ** 2 - tf ** 2)
    y = 50 + d["vy0"] * ts + 0.5 * d["ay"] * ts ** 2
    return np.stack([x * 12, (y - FRONT_FT) * 12, z * 12], -1)


def _t_front(d: dict, y_in: float = 0.0) -> float:
    """Time (from y = 50 ft) the pitch reaches y_in inches in front of the plate's front edge."""
    a, b, c = 0.5 * d["ay"], d["vy0"], 50 - (FRONT_FT + y_in / 12)
    return float((-b - np.sqrt(b * b - 4 * a * c)) / (2 * a))


class Bat:
    """The swing as a function of time (scene arc + bat direction), interpolated."""
    def __init__(self, sc: dict):
        a = np.asarray(sc["arc"], float)
        self.t, self.P = a[:, -1], a[:, :3]
        H = np.asarray(sc.get("hands") or [], float)
        if len(H) != len(a):                                         # the circle: toward its centre
            O = np.asarray(sc["O"], float)
            H = O - self.P
        with np.errstate(invalid="ignore", divide="ignore"):          # a zero-length point stays NaN (skipped)
            self.H = H / np.linalg.norm(H, axis=1, keepdims=True)
        dt = np.gradient(self.t)
        self.V = np.gradient(self.P, axis=0) / np.where(dt == 0, 1e-6, dt)[:, None] / 17.6     # mph

    def at(self, t):
        t = np.asarray(t, float)
        ok = (t >= self.t[0]) & (t <= self.t[-1])
        f = lambda A: np.stack([np.interp(t, self.t, A[:, k]) for k in range(3)], -1)
        return f(self.P), f(self.H), f(self.V), ok


def meet(bat: Bat, ball_xyz: np.ndarray, ball_t: np.ndarray, ball_v: np.ndarray, stand: str, q: float,
         defer: bool = False) -> dict:
    """Where (and whether) one pitch meets the swing: closest approach of the ball's centre to the bat's centre
    line (barrel only) over time; contact if within the two radii -> the collision at that spot, priced."""
    S, H, Vb, ok = bat.at(ball_t)
    lam = np.clip(((ball_xyz - S) * H).sum(-1), *BARREL)
    Q = S + H * lam[:, None]
    dist = np.where(ok, np.linalg.norm(ball_xyz - Q, axis=1), np.inf)
    i = int(np.argmin(dist))
    out = {"i": i, "dist": float(dist[i]), "lam": float(lam[i]), "at": ball_xyz[i].round(2).tolist(),
           "t": float(ball_t[i])}
    if not np.isfinite(dist[i]) or dist[i] >= R_MEET:
        out.update(result="whiff", runs=WHIFF_RUNS)
        return out
    # signed offset: + when the ball is above the bat's centre (perpendicular to the bat + the relative velocity)
    a = H[i]
    u = ball_v[i] - Vb[i]
    up = u - (u @ a) * a
    uh = up / (np.linalg.norm(up) or 1)
    s = np.cross(a, uh)
    s = s / (np.linalg.norm(s) or 1)
    if s[2] < 0:
        s = -s
    D = float((ball_xyz[i] - Q[i]) @ s)
    r = co.collide(ball_v[i:i + 1], Vb[i:i + 1], a[None], [D], q)
    ev = float(r["ev"][0] * eff_along(lam[i]))
    la, sp_phys = float(r["la"][0]), float(r["spray"][0])
    out.update(D=round(D, 2), eff=round(float(eff_along(lam[i])), 3), _ev=ev, _la=la, _sp=sp_phys)
    return out if defer else finish([out], stand)[0]


def finish(rows: list, stand: str) -> list:
    """Spray model + fair / foul + runs for every contact in one batch (the models are slow one ball at a time)."""
    hit = [r for r in rows if "_ev" in r]
    if hit:
        ev = np.array([r["_ev"] for r in hit])
        la = np.array([r["_la"] for r in hit])
        spray = bearing(ev, la, np.array([r["_sp"] for r in hit]), np.full(len(hit), stand))
        runs = price(ev, la, spray, np.full(len(hit), stand))
        for r, e, l_, s, v in zip(hit, ev, la, spray, runs):
            fair = abs(float(s)) <= 45 and -20 < float(l_) < 80
            r.update(result="fair" if fair else "foul", ev=round(float(e), 1), la=round(float(l_), 1),
                     spray=round(float(s), 1), runs=round(float(v) if fair else FOUL_RUNS, 3))
            for k in ("_ev", "_la", "_sp"):
                r.pop(k)
    return rows


# ============================================================================= the commit model (calibrated)
# At his commit point (his own swing time + a decision delay before the ball reaches the plate) he reads where the
# ball is and how fast it's coming, and aims his swing where a FASTBALL with that look would arrive (speed read only
# partly: w_speed). After that he can still adjust a little: up to adj_ms of timing and reach_* inches of barrel.
# Each swing also carries his own spread in timing and aim. A contact grazing the bat (|offset| > r_tip) is a foul tip;
# one well off-centre (> r_foul) glances off foul (back / straight up).
# The numbers are fit on the bat cube so whiff / foul rates by pitch type match MLB swings (calibrate()); until then,
# CAL_DEFAULT.
# defaults: fit (core.swing_cal) to public MLB per-swing whiff / foul rates by pitch type on test hitters -- the Mac
# calibration on real hitters + their real pitches replaces them
CAL_DEFAULT = {"decide_s": 0.040, "w_speed": 0.75, "reach_max": 0.75, "reach_frac": 0.25, "adj_ms": 0.0,
               "r_tip": 2.6, "r_foul": 1.3, "aim_sd": 2.2, "time_sd_ms": 16.0, "league_pf_sd": 7.0, "fitted": False}
D_SHIFTS = np.arange(-36, 37, 3)                                 # ms of timing tried (+ = bat earlier)
FB_LOOK = {"R": (94.5, 15.5, 8.0), "L": (93.0, 15.5, 8.0)}       # the fastball he's reading for (velo, IVB, HB arm)


def cal(season: int = 2026) -> dict:
    """The commit model's parameters: calibrated (data/calibration/swing_cal_<season>.json) or the defaults."""
    key = ("cal", season)
    if key not in _cache:
        p = dict(CAL_DEFAULT)
        from core.path_fit import nearest
        f = nearest("swing_cal", season, OUT) or OUT / f"swing_cal_{season}.json"
        try:
            p.update(json.loads(f.read_text()).get("params", {}))
            p["fitted"] = True
        except Exception:
            pass
        _cache[key] = p
    return _cache[key]


def commit_s(sc: dict, p: dict | None = None) -> float:
    """His commit point, seconds before the ball reaches the plate: his swing time (Statcast swing length at even
    acceleration to his bat speed) + the decision delay."""
    from core.path_fit import swing_time
    p = p or CAL_DEFAULT
    r = sc.get("read") or {}
    sl = _sl_in(sc) or 7.2 * 12
    try:
        bs = float(r.get("bat_speed") or 70.0)
    except (TypeError, ValueError):
        bs = 70.0
    return float(swing_time(sl, bs) + p["decide_s"])


def _state(d: dict, t: float):
    """Ball position (scene in) and velocity (in/s) at time t (s from y = 50 ft)."""
    P = _traj(d, [t])[0]
    V = np.array([d["vx0"] + d["ax"] * t, d["vy0"] + d["ay"] * t, d["vz0"] + d["az"] * t]) * 12
    return P, V


def aim(d: dict, t_commit: float, y_contact: float, fb: dict, w: float):
    """Where and when (s after commit) he expects it at his contact depth: the commit-point state, its speed read as
    fb speed + w x (true - fb), carried on with the fastball's accelerations."""
    P, V = _state(d, t_commit)
    _, Vf = _state(fb, _t_front(fb, 0.0) - 0.15)
    s, sf = np.linalg.norm(V), np.linalg.norm(Vf)
    Vp = V * (sf + w * (s - sf)) / max(s, 1e-6)
    A = np.array([fb["ax"], fb["ay"], fb["az"]]) * 12
    a, b, c = 0.5 * A[1], Vp[1], P[1] - y_contact
    disc = b * b - 4 * a * c
    th = (-b - np.sqrt(disc)) / (2 * a) if abs(a) > 1e-9 and disc >= 0 else -c / b
    if not np.isfinite(th) or th <= 0:
        th = -c / b
    X = P + Vp * th + 0.5 * A * th * th
    return np.array([X[0], y_contact, X[2]]), float(th)


def meet_raw(bat: Bat, ball_xyz: np.ndarray, ball_t: np.ndarray, ball_v: np.ndarray, shifts_ms=D_SHIFTS) -> dict:
    """Closest approach of the ball to the barrel's centre line for every timing shift at once (no contact rule
    yet): signed offset D (in; + = ball above the bat's centre), where along the bat, the velocities and bat axis."""
    sh = np.asarray(shifts_ms, float) / 1000.0
    T = ball_t[None, :] + sh[:, None]                                # (k, n) bat clock
    S, H, Vb, ok = bat.at(T.ravel())
    k, n = T.shape
    S, H, Vb, ok = S.reshape(k, n, 3), H.reshape(k, n, 3), Vb.reshape(k, n, 3), ok.reshape(k, n)
    B = np.broadcast_to(ball_xyz[None], (k, n, 3))
    lam = np.clip(((B - S) * H).sum(-1), *BARREL)
    Q = S + H * lam[..., None]
    dist = np.where(ok, np.linalg.norm(B - Q, axis=-1), np.inf)
    i = np.argmin(dist, axis=1)
    r = np.arange(k)
    a, vb, bv = H[r, i], Vb[r, i], ball_v[i]
    u = bv - vb
    up = u - (u * a).sum(-1, keepdims=True) * a
    uh = up / np.maximum(np.linalg.norm(up, axis=-1, keepdims=True), 1e-9)
    s = np.cross(a, uh)
    s = s / np.maximum(np.linalg.norm(s, axis=-1, keepdims=True), 1e-9)
    s = np.where(s[:, 2:3] < 0, -s, s)
    D = ((B[r, i] - Q[r, i]) * s).sum(-1)
    dmin = dist[r, i]
    D = np.where(np.isfinite(dmin), np.sign(D + 1e-12) * dmin, np.nan)       # closest approach is across the bat
    return {"D": D, "lam": lam[r, i], "a": a, "vb": vb, "bv": bv, "at": B[r, i], "t": T[r, i] - sh,
            "shifts": np.asarray(shifts_ms, float)}


def resolve(raws: list, p: dict, stand: str, q: float, z_aim=None, z_time=None, cons: float = 1.0) -> list:
    """The commit model's outcome for each pitch's meet_raw: his timing error (time_sd x cons), corrected by up to
    adj_ms; his aim error (aim_sd x cons) on the offset; then reach (pull the barrel toward the ball: reach_frac of
    the miss, up to reach_max); contact if the ball still meets the barrel, a foul tip if only grazing; else the
    collision -> bearing -> fair / foul -> runs (all in one batch)."""
    n = len(raws)
    z_aim = np.zeros(n) if z_aim is None else np.asarray(z_aim, float)
    z_time = np.zeros(n) if z_time is None else np.asarray(z_time, float)
    out, hit = [], []
    for j, r in enumerate(raws):
        tau = z_time[j] * p["time_sd_ms"] * cons                   # his timing error (ms; + = early)
        D = r["D"] + z_aim[j] * p["aim_sd"] * cons
        ok = np.isfinite(D) & (np.abs(r["shifts"] - tau) <= p["adj_ms"] + 1e-9)
        if not ok.any():
            ok = np.isfinite(D) & (np.abs(r["shifts"] - tau) == np.min(np.abs(r["shifts"] - tau)))
        if not ok.any():
            out.append({"result": "whiff", "runs": WHIFF_RUNS, "at": r["at"][len(r["shifts"]) // 2].round(2).tolist(),
                        "t": float(r["t"][len(r["shifts"]) // 2]), "dist": None, "lam": None})
            continue
        cand = np.where(ok, np.abs(D), np.inf)
        m = int(np.argmin(cand))
        miss = float(abs(D[m]))
        left = miss - min(p["reach_max"], p["reach_frac"] * miss)
        o = {"at": r["at"][m].round(2).tolist(), "t": float(r["t"][m]), "dist": round(miss, 2),
             "lam": round(float(r["lam"][m]), 2), "shift": float(r["shifts"][m])}
        if left >= R_MEET:
            o.update(result="whiff", runs=WHIFF_RUNS)
        elif left > p["r_tip"]:                                   # only grazes it: a foul tip (Savant: a whiff)
            o.update(result="whiff", runs=WHIFF_RUNS, tip=True, D=round(float(np.sign(D[m]) * left), 2))
        elif left > p.get("r_foul", 9.0):                         # well off-centre: glances off, foul
            o.update(result="foul", runs=FOUL_RUNS, glance=True, D=round(float(np.sign(D[m]) * left), 2))
        else:
            o.update(D=round(float(np.sign(D[m]) * left), 2), eff=round(float(eff_along(r["lam"][m])), 3))
            hit.append((len(out), r["bv"][m], r["vb"][m], r["a"][m], float(np.sign(D[m]) * left),
                        float(eff_along(r["lam"][m]))))
        out.append(o)
    if hit:
        idx = [h[0] for h in hit]
        cr = co.collide(np.array([h[1] for h in hit]), np.array([h[2] for h in hit]), np.array([h[3] for h in hit]),
                        np.array([h[4] for h in hit]), q)
        for k, j in enumerate(idx):
            out[j].update(_ev=float(cr["ev"][k] * hit[k][5]), _la=float(cr["la"][k]), _sp=float(cr["spray"][k]))
        finish([out[j] for j in idx], stand)
    return out


def consistency(sw: pd.DataFrame | None, p: dict) -> float:
    """His swing-to-swing spread vs the league's: SD of his contact depth / the league's (0.6 .. 1.6)."""
    try:
        v = pd.to_numeric(sw["pf"], errors="coerce").dropna()
        if len(v) >= 25:
            return float(np.clip(v.std() / p["league_pf_sd"], 0.6, 1.6))
    except Exception:
        pass
    return 1.0


def _pitch_samples(d: dict, t_from: float, t_to: float, n: int = 1600):
    ts = np.linspace(t_from, t_to, n)
    xyz = _traj(d, ts)
    v = np.gradient(xyz, axis=0) / np.gradient(ts)[:, None] / 17.6
    return ts, xyz, v


def timing_layer(sc: dict, pitch: dict, stand: str, q: float, shifts=np.arange(-30, 31, 3)) -> dict:
    """The pitch he saw vs his swing started earlier / later (ms; + = early): every crossing, priced."""
    bat = Bat(sc)
    t_c = _t_front(pitch, sc["contact"][1])                     # when the ball reaches his contact depth
    ts, xyz, v = _pitch_samples(pitch, t_c - 0.045, t_c + 0.045)
    rows = []
    for sh in shifts:
        m = meet(bat, xyz, ts - t_c + sh / 1000.0, v, stand, q, defer=True)   # bat clock: early = further along
        m["shift"] = int(sh)
        rows.append(m)
    rows = finish(rows, stand)
    pos = [r for r in rows if r["runs"] > 0]
    return {"rows": rows, "window_ms": int(len(pos) * (shifts[1] - shifts[0])) if pos else 0}


def his_mix(sw: pd.DataFrame | None, hand: str, min_n: int = 40) -> dict | None:
    """The pitches he actually swings at from that pitcher hand, as MIX: usage, velo mean / sd, IVB, HB arm side
    (types he's seen < 5 of are dropped). None if too few swings -> the MLB mix."""
    if sw is None or not len(sw) or not {"ptype", "hand", "velo", "ivb", "hb"} <= set(sw):
        return None
    r = sw[sw["hand"].astype(str).eq(hand) & sw["ptype"].isin(list(MIX))].dropna(subset=["velo", "ivb", "hb"])
    if len(r) < min_n:
        return None
    out = {}
    for k, g in r.groupby("ptype"):
        if len(g) >= 5:
            out[k] = (len(g) / len(r), float(g["velo"].mean()), float(max(g["velo"].std(), 1.0)),
                      float(g["ivb"].mean()), float(g["hb"].mean()))
    return out or None


def his_rates(sw: pd.DataFrame | None, hand: str) -> dict:
    """What actually happened on his swings vs that hand, by pitch type: n, whiff / foul / fair per swing."""
    if sw is None or not len(sw) or "result" not in sw:
        return {}
    r = sw[sw["hand"].astype(str).eq(hand) & sw["result"].isin(["whiff", "foul", "fair"])]
    return {k: {"n": int(len(g)), **{c: round(float((g["result"] == c).mean()), 3) for c in ("whiff", "foul", "fair")}}
            for k, g in r.groupby("ptype") if len(g) >= 5}


def _mix_sample(hand: str, n: int, rng, mix: dict | None = None) -> list:
    mix = mix or MIX
    keys = list(mix)
    p = np.array([mix[k][0] for k in keys])
    picks = rng.choice(keys, n, p=p / p.sum())
    out = []
    for k in picks:
        _, v, vs, ivb, hb = mix[k]
        out.append((k, rng.normal(v, vs), rng.normal(ivb, 2.5), rng.normal(hb, 2.5)))
    return out


def arsenal_layer(sc: dict, ref: dict, stand: str, hand: str, q: float, n: int = 160, seed: int = 11,
                  bat_sc: dict | None = None, p: dict | None = None, mix: dict | None = None, cons: float = 1.0,
                  actual: dict | None = None, robust: bool = True) -> dict:
    """The same swing vs pitches that look like the modeled one at HIS commit point: each sampled pitch is lined up
    through the modeled pitch's commit-point position (+ a little scatter), then breaks on its own.
    robust (the page): path robustness -- one fixed swing, timed for the modeled pitch, vs the same MLB mix for every
        hitter, strict contact (the ball must meet the barrel) -> how forgiving his path is, comparable across hitters.
    robust=False: the commit model (he aims + times each off what he sees at commit, with adjustment + spread) -- kept
        for prediction work; its calibration (core.swing_cal) doesn't separate hitters yet."""
    from core.bat_path import synth_pitch
    p = p or CAL_DEFAULT
    rng = np.random.default_rng(seed)
    bat = Bat(bat_sc or sc)
    cs = commit_s(sc, p)
    fb = synth_pitch(*FB_LOOK.get(hand, FB_LOOK["R"]), hand, ref["plate_x"], ref["plate_z"])
    yC = float(sc["contact"][1])
    C = np.asarray(sc["contact"], float)
    t_ref_c = _t_front(ref, yC)
    t_ref_commit = _t_front(ref, 0.0) - cs
    commit = _traj(ref, [t_ref_commit])[0]
    tau = t_ref_c - t_ref_commit                                     # commit -> his contact, for the modeled pitch
    # his aim for the modeled pitch itself: the swing as drawn is aimed there
    aim0, th0 = aim(ref, t_ref_commit, yC, fb, p["w_speed"])
    raws, meta = [], []
    for k, velo, ivb, hb in _mix_sample(hand, n, rng, mix):
        d = synth_pitch(velo, ivb, hb, hand, ref["plate_x"], ref["plate_z"])
        tc = _t_front(d, 0.0) - cs
        off = (commit - _traj(d, [tc])[0]) / 12 + np.array([rng.normal(0, 1.5), 0, rng.normal(0, 1.5)]) / 12
        d = synth_pitch(velo, ivb, hb, hand, ref["plate_x"] + off[0], ref["plate_z"] + off[2])
        tc = _t_front(d, 0.0) - cs
        if robust:                                                    # one fixed swing, timed for the modeled pitch
            shift, dt = np.zeros(3), 0.0
            ts, xyz, v = _pitch_samples(d, tc, _t_front(d, -40.0), 1400)
            raws.append(meet(bat, xyz, ts - tc - tau, v, stand, q, defer=True))
        else:
            A, th = aim(d, tc, yC, fb, p["w_speed"])
            shift = (A - aim0) * np.array([1.0, 0.0, 1.0])            # he moves the swing to where he expects it
            dt = th - th0                                             # ... and times it for when
            ts, xyz, v = _pitch_samples(d, tc + th - 0.06, tc + th + 0.06, 480)
            raws.append(meet_raw(bat, xyz - shift, ts - tc - tau - dt, v))
        tf = np.linspace(_t_front(d, (d["release_pos_y"] - FRONT_FT) * 12), _t_front(d, -12.0), 32)
        fly = np.column_stack([_traj(d, tf), tf - tc - tau]).round(4).tolist()     # release -> plate, scene clock
        line = _traj(d, np.linspace(tc, _t_front(d, -12.0), 24)).round(1).tolist()
        meta.append({"type": k, "velo": round(float(velo), 1), "line": line, "fly": fly, "_shift": shift, "_dt": dt})
    res = finish(raws, stand) if robust else \
        resolve(raws, p, stand, q, rng.standard_normal(n), rng.standard_normal(n), cons)
    rows = []
    for m_, r in zip(meta, res):
        sh, dt = m_.pop("_shift"), m_.pop("_dt")
        r["at"] = (np.asarray(r["at"]) + sh).round(2).tolist()       # back to where the ball really was
        r["t"] = round(float(r["t"]) + dt, 4)                         # on the scene clock
        rows.append({**m_, **r})
    df = pd.DataFrame(rows)
    by = []
    for k, g in df.groupby("type"):
        a_ = (actual or {}).get(k) or {}
        by.append({"type": k, "n": int(len(g)), "velo": round(float(g["velo"].mean()), 1),
                   "whiff": round(float((g["result"] == "whiff").mean()), 3),
                   "foul": round(float((g["result"] == "foul").mean()), 3),
                   "fair": round(float((g["result"] == "fair").mean()), 3), "runs": round(float(g["runs"].mean()), 3),
                   "act_n": a_.get("n"), "act_whiff": a_.get("whiff"), "act_foul": a_.get("foul"),
                   "act_fair": a_.get("fair")})
    by.sort(key=lambda r: -r["n"])
    eye = _traj(ref, [t_ref_commit - 0.09])[0] + np.array([0.0, 0.0, 3.0])   # behind the commit point, along the pitch
    return {"rows": rows, "by_type": by, "commit": commit.round(1).tolist(), "eye": eye.round(1).tolist(),
            "t_commit": round(float(-tau), 4), "commit_ms": int(round(cs * 1000)), "cons": round(float(cons), 2),
            "mix_src": "his" if mix else "mlb", "fitted": bool(p.get("fitted")), "mode": "robust" if robust else "commit",
            "counts": {k: int(v) for k, v in df["result"].value_counts().items()},
            "runs100": round(float(df["runs"].mean() * 100), 2)}


HEAT_D = np.round(np.arange(-2.6, 2.61, 0.2), 2)                 # offset across the barrel (in; + = under it)
HEAT_LAM = np.arange(-6.0, 10.01, 1.0)                           # along it from the sweet spot (- = toward the tip)


def bat_heat(sc: dict, ref: dict, stand: str, q: float, t_from: float = -0.045, t_to: float = 0.035,
             step: float = 0.0025) -> dict:
    """The barrel painted, moment by moment: at each instant of the swing, every spot on the barrel (across x
    along) priced in runs if the modeled pitch met it right then -- the bat's own angle, direction and speed at that
    instant. Also the directions to paint by: f = the face toward the incoming ball, s = 'above' across the bat."""
    bat = Bat(sc)
    ts = np.arange(t_from, t_to + 1e-9, step)
    S, H, Vb, ok = bat.at(ts)
    t_c = _t_front(ref, sc["contact"][1])
    _, vball = _state(ref, t_c)
    vball = vball / 17.6                                             # in/s -> mph
    nD, nL, nT = len(HEAT_D), len(HEAT_LAM), len(ts)
    runs = np.full((nT, nL, nD), np.nan)
    F, Sv = np.zeros((nT, 3)), np.zeros((nT, 3))
    good = [i for i in range(nT) if ok[i] and np.isfinite(H[i]).all() and np.linalg.norm(Vb[i]) > 20]
    for i in good:
        u = vball - Vb[i]
        a = H[i] / np.linalg.norm(H[i])
        up = u - (u @ a) * a
        uh = up / (np.linalg.norm(up) or 1)
        s = np.cross(a, uh)
        s = s / (np.linalg.norm(s) or 1)
        if s[2] < 0:
            s = -s
        F[i], Sv[i] = -uh, s
    if good:
        G = len(good)
        Dg = np.tile(np.repeat(HEAT_D[None, :], nL, 0).ravel(), G)
        Lg = np.tile(np.repeat(HEAT_LAM[:, None], nD, 1).ravel(), G)
        idx = np.repeat(np.array(good), nL * nD)
        r = co.collide(np.repeat(vball[None], len(idx), 0), Vb[idx], H[idx], Dg, q)
        ev = r["ev"] * eff_along(Lg)
        sp = bearing(ev, r["la"], r["spray"], np.full(len(idx), stand))
        fair = (np.abs(sp) <= 45) & (r["la"] > -20) & (r["la"] < 80)
        v = np.where(fair, price(ev, r["la"], sp, np.full(len(idx), stand)), FOUL_RUNS)
        runs[good] = v.reshape(G, nL, nD)
    rnd = lambda A: np.where(np.isfinite(A), np.round(A, 2), None).tolist()
    return {"t": np.round(ts, 4).tolist(), "D": HEAT_D.tolist(), "lam": HEAT_LAM.tolist(), "runs": rnd(runs),
            "f": np.round(F, 3).tolist(), "s": np.round(Sv, 3).tolist()}


def layers(sc: dict, sw: pd.DataFrame | None, stand: str, season: int | None = None) -> dict:
    """Contact + timing + arsenal for a Bat Path scene (its modeled pitch + his measured path). season: the scene's
    (its swings' year) -> that season's calibration files, or the nearest built."""
    season = season or sc.get("season")
    from core import path_fit
    from core.bat_path import synth_pitch
    m = sc.get("model") or {}
    q = (co.load(season).get("measured") or {}).get("q", co.Q0)
    hand = m.get("hand", "R")
    ref = synth_pitch(m["velo"], m["ivb"], m["hb"], hand, m["px"], m["pz"])
    bat = Bat(sc)
    t_c = _t_front(ref, sc["contact"][1])
    ts, xyz, v = _pitch_samples(ref, t_c - 0.002, t_c + 0.002, 9)
    S, H, Vb, _ = bat.at(np.zeros(1))
    vb = v[4]
    cl = contact_layer(vb, Vb[0], H[0], stand, q)
    # his balls in play on the map
    dots = []
    if sw is not None and len(sw) and {"ev", "la", "match_v", "match_h"} <= set(sw):
        r = sw[(sw["stand"] == stand) & sw["ev"].notna() & sw["la"].notna() & sw["match_v"].notna()]
        if len(r):
            right = stand != "L"
            rows = pd.DataFrame({"release_speed": r["velo"], "vaa": -r["match_v"],
                                 "haa": np.where(right, -r["match_h"], r["match_h"]), "bat_speed": r["bat_speed"],
                                 "attack_angle": r["attack_angle"], "attack_direction": r["attack_direction"],
                                 "stand": stand, "launch_angle": r["la"], "launch_speed": r["ev"]}).dropna()
            if len(rows):
                ax = co.axis_measured(rows["attack_angle"], rows["attack_direction"], rows["stand"],
                                      r.loc[rows.index, "swing_path_tilt"], path_fit.load(season).get("league"))
                rows["ax_m"], rows["ay_m"], rows["az_m"] = ax[:, 0], ax[:, 1], ax[:, 2]
                hc = his_contacts(rows, q)
                dots = [[round(float(a), 2), round(float(b), 3)] for a, b in zip(hc["D"], hc["eff"])
                        if a == a and b == b][:400]
    cl["dots"] = dots
    tl = timing_layer(sc, ref, stand, q)
    p = cal(season)
    cons, mix = 1.0, None                     # path robustness: the same test (MLB mix, fixed swing) for every hitter
    ar = arsenal_layer(sc, ref, stand, hand, q, p=p, mix=mix, cons=cons)
    # the same swing with an MLB-average path shape (league turning rates), for comparison
    try:
        lr = path_fit.load(season).get("league") or {}
        L = path_fit._league_for(lr, float(sc["read"].get("tilt") or 34))
        cur = {"center": sc["contact"][1], "lo": sc["contact"][1] - 15, "hi": sc["contact"][1] + 15,
               **{k: {"slope": (L.get(k) or {}).get("slope", 0.0), "curve": (L.get(k) or {}).get("curve", 0.0)}
                  for k in ("attack_angle", "attack_direction", "bat_speed")}}
        v = float(sc["read"]["bat_speed"]) * 17.6
        rb = path_fit.rebuild(sc["contact"], float(sc["read"]["attack_angle"]), float(sc["read"]["attack_direction"]),
                              float(sc["read"]["bat_speed"]), stand, cur, -sc["arc"][0][4] * v, 400.0, center=sc.get("O"),
                              swing_len=_sl_in(sc))
        avg = arsenal_layer(sc, ref, stand, hand, q, bat_sc={"arc": rb["arc"], "hands": rb["hands"], "O": sc["O"]},
                            p=p, mix=mix, cons=cons)
        ar["mlb_runs100"] = avg["runs100"]
    except Exception:
        ar["mlb_runs100"] = None
    try:
        heat = bat_heat(sc, ref, stand, q)
    except Exception:
        heat = None
    return {"contact": cl, "timing": tl, "arsenal": ar, "heat": heat, "pt_color": PT_COLOR, "q": q}
