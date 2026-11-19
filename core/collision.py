"""Bat-ball collision: exit velo, launch angle, spray and spin from the bat's speed / direction / orientation, the
pitch, and WHERE on the barrel the ball was hit.

Physics (published, wood bat):
  * normal direction (along the line of centres): the collision efficiency q (Nathan, "Characterizing the performance
    of baseball bats", AJP 2003): v_out,n = q * v_ball,n(in) + (1 + q) * v_bat,n. q ~ 0.20 at a wood bat's sweet spot
    (MLB's own "max exit velo" for squared-up uses 0.23). Calibrated here on MLB balls in play.
  * tangential direction + spin: Kensrud, Nathan & Smith (AJP 2017, "Oblique collisions of baseballs and softballs
    with a bat"): the contact point's relative tangential velocity reverses by e_x (0.46 wood); with the ball's
    moment of inertia 0.4 m r^2 and the bat's recoil r_x:
        ball tangential change = alpha/(1+alpha) * (1+e_x)/(1+r_x) * (bat - ball tangential velocity)
        spin                   = 1/(1+alpha)     * (1+e_x)/(1+r_x) * (same) / r
  * geometry: offset D (inches) between the ball's centre and the bat's centre line, perpendicular to the relative
    velocity; sin(psi) = D / (r_ball + r_bat). D > 0 = ball above the bat's centre (undercut -> backspin, higher).

What Statcast gives per swing: bat speed + direction (attack angle / direction) at the sweet spot, the pitch's
speed + approach angles, and the result (exit velo, launch angle, spray). The bat's ORIENTATION (where the hands are)
is inferred (core.bat_path / core.path_fit). The offset D isn't measured: it's solved from the launch angle; then
exit velo predicted at that D vs the actual = how close to the sweet spot ALONG the barrel he hit it (efficiency).

    python -m core.collision 2026      # calibrates q on the bat cube + validates exit velo / spray -> report
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "calibration"
R_BALL = 1.43                    # in (Kensrud et al.: 1.42)
R_BAT = 1.25                     # in, barrel radius (2.5" barrel)
ALPHA = 0.4                      # ball moment of inertia / (m r^2)
E_X = 0.46                       # tangential COR, wood (Kensrud et al. 2017)
R_X = 0.29                       # tangential bat recoil, from their bat's mass / inertia (Eq. 7)
Q0 = 0.20                        # collision efficiency at the sweet spot (Nathan 2003); calibrated by run()
PLATE = 0.915                    # pitch speed at the plate / release speed (drag), typical MLB


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1)


def collide(v_ball, v_bat, axis, D, q=Q0, e_x=E_X, r_x=R_X) -> dict:
    """Vectorised. v_ball, v_bat: (n,3) mph at contact (plate frame: x = 1B side, y = toward the pitcher, z = up);
    axis: (n,3) along the bat (either way); D: (n,) inches. -> ev (mph), la (deg), spray (deg, + = 1B side),
    spin (rpm, + = backspin)."""
    v_ball, v_bat, a = (np.asarray(x, float) for x in (v_ball, v_bat, axis))
    a = _unit(a)
    D = np.broadcast_to(np.asarray(D, float), (len(v_ball),))
    u = v_ball - v_bat                                               # ball relative to the bat
    uperp = u - (u * a).sum(-1, keepdims=True) * a
    uh = _unit(uperp)
    s = _unit(np.cross(a, uh))
    s = np.where(s[:, 2:3] < 0, -s, s)                               # "above" the bat's centre line
    sp = np.clip(D / (R_BALL + R_BAT), -0.99, 0.99)
    cp = np.sqrt(1 - sp ** 2)
    n = -cp[:, None] * uh + sp[:, None] * s                          # bat -> ball, along the line of centres
    t = sp[:, None] * uh + cp[:, None] * s
    bn, Bn = (v_ball * n).sum(-1), (v_bat * n).sum(-1)
    bt, Bt = (v_ball * t).sum(-1), (v_bat * t).sum(-1)
    ba = (v_ball * a).sum(-1)
    k = (1 + e_x) / (1 + r_x)
    q = np.broadcast_to(np.asarray(q, float), (len(v_ball),))
    vn = -q * bn + (1 + q) * Bn
    vt = bt + ALPHA / (1 + ALPHA) * k * (Bt - bt)
    out = vn[:, None] * n + vt[:, None] * t + ba[:, None] * a
    ev = np.linalg.norm(out, axis=-1)
    la = np.degrees(np.arcsin(np.clip(out[:, 2] / np.where(ev > 0, ev, 1), -1, 1)))
    spray = np.degrees(np.arctan2(out[:, 0], out[:, 1]))
    w = 1 / (1 + ALPHA) * k * (Bt - bt) * 17.6 / R_BALL               # rad/s
    spin = w * 60 / (2 * math.pi) * np.sign(D + 1e-9)
    return {"ev": ev, "la": la, "spray": spray, "spin": np.abs(spin) * np.sign(sp + 1e-12)}


def solve_offset(v_ball, v_bat, axis, la_target, q=Q0, lo=-2.6, hi=2.6, iters=40) -> np.ndarray:
    """The offset D (in) that gives the observed launch angle (bisection; launch angle rises with D)."""
    la_target = np.asarray(la_target, float)
    a, b = np.full(len(la_target), lo), np.full(len(la_target), hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        la = collide(v_ball, v_bat, axis, m, q)["la"]
        up = la < la_target
        a, b = np.where(up, m, a), np.where(up, b, m)
    D = 0.5 * (a + b)
    edge = (np.abs(D - lo) < 1e-3) | (np.abs(D - hi) < 1e-3)
    return np.where(edge, np.nan, D)


# ----------------------------------------------------------------------------- from a swing's Statcast numbers
def ball_velocity(release_speed, vaa, haa) -> np.ndarray:
    """Pitch velocity at the plate (mph): ~91.5% of release speed, along its approach angles (vaa < 0 = coming
    down, haa + = moving toward the 1B side)."""
    s = np.asarray(release_speed, float) * PLATE
    d = _unit(np.stack([np.tan(np.radians(haa)), -np.ones(len(s)), np.tan(np.radians(vaa))], -1))
    return s[:, None] * d


def bat_velocity(bat_speed, aa, ad, stand) -> np.ndarray:
    h = np.radians(np.where(np.asarray(stand) == "L", -np.asarray(ad, float), np.asarray(ad, float)))
    a = np.radians(np.asarray(aa, float))
    d = np.stack([np.sin(h) * np.cos(a), np.cos(h) * np.cos(a), np.sin(a)], -1)
    return np.asarray(bat_speed, float)[:, None] * d


def axis_measured(aa, ad, stand, tilt, lr: dict | None = None) -> np.ndarray:
    """Where the hands are (sweet spot -> hands), from the MEASURED turning of the path (core.path_fit): toward the
    centre of the path's curve at contact. aa rises and ad turns pull at the league's measured rates for his tilt."""
    from core import path_fit
    aa, ad, tilt = (np.asarray(x, float) for x in (aa, ad, tilt))
    right = np.asarray(stand) != "L"
    sa, sd = np.empty(len(aa)), np.empty(len(aa))
    for i, t in enumerate(tilt):
        L = path_fit._league_for(lr or {}, float(t) if t == t else 34.0)
        sa[i] = (L.get("attack_angle") or {}).get("slope", 0.9)
        sd[i] = (L.get("attack_direction") or {}).get("slope", -1.55)
    h = np.radians(np.where(right, ad, -ad))
    a = np.radians(aa)
    ty = np.cos(h) * np.cos(a)                                      # inches of depth per inch travelled
    da = np.radians(sa) * ty                                        # per inch travelled
    dh = np.radians(np.where(right, sd, -sd)) * ty
    dTa = np.stack([-np.sin(h) * np.sin(a), -np.cos(h) * np.sin(a), np.cos(a)], -1)
    dTh = np.stack([np.cos(h) * np.cos(a), -np.sin(h) * np.cos(a), np.zeros(len(a))], -1)
    return _unit(dTa * da[:, None] + dTh * dh[:, None])


def frame(cube: pd.DataFrame, lr: dict | None = None) -> pd.DataFrame:
    """Balls in play with everything the collision needs (+ both bat-orientation estimates)."""
    from core.bat_path import _plane
    n = lambda c: pd.to_numeric(cube[c], errors="coerce") if c in cube else pd.Series(np.nan, index=cube.index)
    d = cube[(n("launch_speed") > 0) & n("launch_angle").notna() & n("bat_speed").ge(50) & n("pf").notna()
             & n("match_v").notna() & n("match_h").notna() & n("release_speed").notna()].copy()
    if "description" in d:
        d = d[d["description"].astype(str).eq("hit_into_play")]
    if d.empty:
        return d
    right = d["stand"].astype(str).ne("L").to_numpy()
    d["vaa"] = -n("match_v").loc[d.index]
    mh = n("match_h").loc[d.index].to_numpy()
    d["haa"] = np.where(right, -mh, mh)
    dv, w, bad = _plane(d["attack_angle"], d["attack_direction"], d["swing_path_tilt"], d["stand"], d["pf"],
                        n("depth").loc[d.index].fillna(30.0), 12 * n("plate_x").loc[d.index], 12 * n("plate_z").loc[d.index])
    d["ax_c"], d["ay_c"], d["az_c"] = w[:, 0], w[:, 1], w[:, 2]
    m = axis_measured(d["attack_angle"], d["attack_direction"], d["stand"], d["swing_path_tilt"], lr)
    d["ax_m"], d["ay_m"], d["az_m"] = m[:, 0], m[:, 1], m[:, 2]
    if {"hc_x", "hc_y"} <= set(d):
        d["spray_obs"] = np.degrees(np.arctan2(n("hc_x").loc[d.index] - 125.42, 198.27 - n("hc_y").loc[d.index]))
    return d[~bad]


def evaluate(d: pd.DataFrame, q: float, axis: str = "m") -> dict:
    vb = ball_velocity(d["release_speed"], d["vaa"], d["haa"])
    vB = bat_velocity(d["bat_speed"], d["attack_angle"], d["attack_direction"], d["stand"])
    ax = d[[f"ax_{axis}", f"ay_{axis}", f"az_{axis}"]].to_numpy()
    D = solve_offset(vb, vB, ax, d["launch_angle"].to_numpy(), q)
    r = collide(vb, vB, ax, np.nan_to_num(D), q)
    ev = d["launch_speed"].to_numpy()
    eff = ev / r["ev"]
    ok = np.isfinite(D)
    out = {"solved": round(float(ok.mean()), 3), "D_median": round(float(np.nanmedian(D)), 2),
           "D_iqr": [round(float(np.nanpercentile(D, 25)), 2), round(float(np.nanpercentile(D, 75)), 2)],
           "eff_p50": round(float(np.nanmedian(eff[ok])), 3), "eff_p90": round(float(np.nanpercentile(eff[ok], 90)), 3),
           "eff_over_105": round(float(np.mean(eff[ok] > 1.05)), 3),
           "ev_corr": round(float(np.corrcoef(ev[ok], r["ev"][ok])[0, 1]), 3)}
    # D's effect on launch angle (deg per 0.1 in), around a square hit
    r0, r1 = collide(vb, vB, ax, 0.0, q), collide(vb, vB, ax, 0.1, q)
    out["la_per_0p1in"] = round(float(np.median(r1["la"] - r0["la"])), 2)
    out["ev_square_median"] = round(float(np.median(r0["ev"])), 1)
    if "spray_obs" in d:
        so = d["spray_obs"].to_numpy()
        e = (r["spray"] - so + 180) % 360 - 180
        ok2 = ok & np.isfinite(so) & (d["launch_angle"].to_numpy() < 50)
        out["spray_mae"] = round(float(np.median(np.abs(e[ok2]))), 2)
        out["spray_corr"] = round(float(np.corrcoef(r["spray"][ok2], so[ok2])[0, 1]), 3)
    return out


def calibrate_q(d: pd.DataFrame, axis: str = "m") -> float:
    """q so the best-struck balls sit at full efficiency: the 90th percentile of actual / predicted exit velo = 1."""
    best, bq = None, Q0
    for q in np.arange(0.12, 0.30, 0.01):
        p90 = evaluate(d, q, axis)["eff_p90"]
        if best is None or abs(p90 - 1) < best:
            best, bq = abs(p90 - 1), float(q)
    return round(bq, 3)


def run(season: int, sample: int = 60000) -> dict:
    from core import path_fit
    from pipeline import bat_cube
    lr = (path_fit.load(season) or {}).get("league") or {}
    cube = bat_cube.load(season)
    d = frame(cube, lr)
    if len(d) > sample:
        d = d.sample(sample, random_state=3)
    out = {"season": season, "balls": int(len(d))}
    for axis, name in (("m", "measured"), ("c", "circle")):
        q = calibrate_q(d, axis)
        out[name] = {"q": q, **evaluate(d, q, axis)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"collision_{season}.json").write_text(json.dumps(out, indent=1))
    report(out)
    return out


def load(season: int | None) -> dict:
    """This season's collision calibration -- or the nearest season's when it isn't built."""
    from core.path_fit import nearest
    f = nearest("collision", season, OUT)
    return json.loads(f.read_text()) if f is not None else {}


def report(o: dict) -> None:
    print(f"\n[{o['season']}] Bat-ball collision check: {o['balls']:,} balls in play")
    for k in ("measured", "circle"):
        r = o.get(k)
        if not r:
            continue
        print(f"  bat orientation from the {k} path:")
        print(f"    collision efficiency q (calibrated) {r['q']:.3f}   (Nathan: ~0.20 wood; MLB's max-EV uses 0.23)")
        print(f"    offset solved for {100 * r['solved']:.0f}% of balls; median {r['D_median']:+.2f} in "
              f"(middle half {r['D_iqr'][0]:+.2f} to {r['D_iqr'][1]:+.2f}); launch angle +{r['la_per_0p1in']:.1f} deg "
              f"per 0.1 in of undercut")
        print(f"    exit velo vs predicted at that offset: median {100 * r['eff_p50']:.0f}%, "
              f"{100 * r['eff_over_105']:.1f}% above 105% (impossible-ish), corr {r['ev_corr']:.2f}")
        if "spray_mae" in r:
            print(f"    spray direction: median miss {r['spray_mae']:.1f} deg, corr {r['spray_corr']:.2f}")


if __name__ == "__main__":
    for a in sys.argv[1:] or ["2026"]:
        run(int(a))
