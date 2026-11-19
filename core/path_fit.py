"""The bat's path through the zone, MEASURED from a hitter's own swings instead of assumed to be a circle.

Idea: a hitter's swings with the same plane are snapshots of one path caught at different moments -- contact deep in
the zone catches it early, contact out front catches it later. So within a hitter (and a family of similar swings:
same tilt, bat speed, pitch height / side), how his MEASURED attack angle, attack direction and bat speed change with
contact depth IS the rate his path turns, rises and speeds up through the zone.

  league_rates(cube)   within-hitter slopes (+ curvature) of attack angle / direction / bat speed vs contact depth,
                       holding pitch height + side + type fixed -- overall and by swing-path tilt; next to the rates
                       our circle ASSUMES for the same swings (how wrong the circle is, before anything else)
  evaluate(cube)       held-out test, per hitter (5 folds): predict a swing's attack angle / direction at its depth
                       with (a) his measured rates, (b) the circle's rates, (c) no change -- same anchor for all three,
                       so only the rate differs. RMSE overall and deep / middle / out front.
  hitter_curve(...)    one hitter's rates for a family of swings (kernel-weighted by tilt, bat speed, pitch height /
                       side), shrunk toward the league's by how many swings back it.
  rebuild(...)         the 3D path from those rates: step along from contact, pointing each step the way his swings
                       point at that depth; outside the depths his swings reach, keep turning at the last measured rate.

    python -m core.path_fit 2026          # reads the bat cube -> data/calibration/path_fit_2026.json + a report
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
TARGETS = ["attack_angle", "attack_direction", "bat_speed"]
TILT_BINS = [0, 28, 32, 36, 40, 90]
MIN_SWINGS = 120                 # contact swings for a hitter to enter the held-out test
SHRINK_K = 60.0                  # swings' worth of the league rate blended into a hitter's
KERNEL = {"swing_path_tilt": 5.0, "bat_speed": 4.0, "plate_z": 0.40, "xh": 0.40}


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)


def contacts(cube: pd.DataFrame) -> pd.DataFrame:
    """Contact swings with everything the fits need (xh = plate side, + = away from a righty / lefty alike)."""
    d = cube.copy()
    if "made" in d:
        d = d[d["made"].map(lambda v: str(v).lower() in ("true", "1", "1.0"))]
    for c in ["pf", "plate_x", "plate_z", "swing_path_tilt", "bat_speed", "attack_angle", "attack_direction", "depth",
              "intercept_ball_minus_batter_pos_x_inches"]:
        d[c] = _n(d, c)
    d["xh"] = np.where(d["stand"].astype(str).eq("L"), -d["plate_x"], d["plate_x"])
    d["key"] = (d["batter"].astype(str) + "|" + d["stand"].astype(str) + "|" +
                (d["year"].astype(str) if "year" in d else "0"))
    d = d.dropna(subset=["pf", "plate_z", "xh", "swing_path_tilt", "bat_speed", "attack_angle", "attack_direction"])
    return d[(d["pf"] > -30) & (d["pf"] < 45)]


# ----------------------------------------------------------------------------- what the circle assumes
def circle_rates(d: pd.DataFrame, eps: float = 0.04) -> pd.DataFrame:
    """Per swing: the rate (deg per inch of contact depth) its attack angle / direction would change if the barrel
    really moved on our circle (the 3D view's geometry), at its own contact point."""
    from core.bat_path import _plane
    out = pd.DataFrame({"c_aa": np.nan, "c_ad": np.nan}, index=d.index)
    for i in range(0, len(d), 6000):
        g = d.iloc[i:i + 6000]
        dv, w, bad = _plane(g["attack_angle"], g["attack_direction"], g["swing_path_tilt"], g["stand"], g["pf"],
                            g["depth"].fillna(30.0), 12 * g["plate_x"], 12 * g["plate_z"])
        cx = g["intercept_ball_minus_batter_pos_x_inches"].fillna(30.0).to_numpy()
        R = np.clip(np.hypot(cx, g["pf"].to_numpy() + g["depth"].fillna(30.0).to_numpy()), 24, 48)
        side = np.where(g["stand"].astype(str).eq("L"), -1.0, 1.0)

        def ang(phi):
            t = w * np.sin(phi)[:, None] + dv * np.cos(phi)[:, None]
            y = R * (w[:, 1] * (1 - np.cos(phi)) + dv[:, 1] * np.sin(phi))
            aa = np.degrees(np.arctan2(t[:, 2], np.hypot(t[:, 0], t[:, 1])))
            ad = side * np.degrees(np.arctan2(t[:, 0], t[:, 1]))
            return y, aa, ad
        e = np.full(len(g), eps)
        y1, a1, d1 = ang(e)
        y0, a0, d0 = ang(-e)
        dy = np.where(np.abs(y1 - y0) > 1e-6, y1 - y0, np.nan)
        out.iloc[i:i + 6000, 0] = np.where(bad, np.nan, (a1 - a0) / dy)
        out.iloc[i:i + 6000, 1] = np.where(bad, np.nan, (d1 - d0) / dy)
    return out


# ----------------------------------------------------------------------------- the measured rates
def _design(d: pd.DataFrame, center: pd.Series | float, quad: bool = True) -> np.ndarray:
    u = (d["pf"] - center).to_numpy()
    cols = [u, u * u] if quad else [u]
    cols += [d["plate_z"].to_numpy(), d["xh"].to_numpy(), d["swing_path_tilt"].to_numpy()]
    if "group" in d:
        for gname in ("Breaking", "Offspeed"):
            cols.append(d["group"].astype(str).eq(gname).astype(float).to_numpy())
    return np.column_stack(cols)


def _demean(X: np.ndarray, keys: pd.Series) -> np.ndarray:
    df = pd.DataFrame(X)
    return (df - df.groupby(keys.to_numpy()).transform("mean")).to_numpy()


def league_rates(d: pd.DataFrame) -> dict:
    """Within-hitter rates vs contact depth (deg or mph per inch, + per inch^2), pitch height / side / tilt / type held
    fixed -- overall and by tilt bin, next to the circle's average rate for the same swings."""
    if d.empty:
        return {}
    d = d.copy()
    d["center"] = d.groupby("key")["pf"].transform("mean")
    cr = circle_rates(d)
    d["c_aa"], d["c_ad"] = cr["c_aa"], cr["c_ad"]
    out = {"n": int(len(d)), "hitters": int(d["key"].nunique()), "by_tilt": {}}

    def fit(s):
        X = _demean(_design(s, s["center"]), s["key"])
        r = {}
        for tcol in TARGETS:
            y = s[tcol].to_numpy() - s.groupby("key")[tcol].transform("mean").to_numpy()
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            r[tcol] = {"slope": round(float(b[0]), 4), "curve": round(float(b[1]), 5)}
        r["circle"] = {"attack_angle": round(float(s["c_aa"].median()), 4),
                       "attack_direction": round(float(s["c_ad"].median()), 4)}
        r["n"] = int(len(s))
        return r
    out["all"] = fit(d)
    tb = pd.cut(d["swing_path_tilt"], TILT_BINS)
    for b, s in d.groupby(tb, observed=True):
        if len(s) >= 500:
            out["by_tilt"][f"{int(b.left)}-{int(b.right)}"] = fit(s)
    return out


def _league_for(lr: dict, tilt: float) -> dict:
    for k, v in (lr.get("by_tilt") or {}).items():
        lo, hi = (float(x) for x in k.split("-"))
        if lo < tilt <= hi:
            return v
    return lr.get("all") or {}


def hitter_curve(rows: pd.DataFrame, target: dict, lr: dict | None = None, k: float = SHRINK_K) -> dict | None:
    """His path's rates for one family of swings (target: swing_path_tilt, bat_speed, plate_z, xh): kernel-weighted
    fit of attack angle / direction / bat speed vs contact depth (quadratic), pitch height / side held fixed, shrunk
    toward the league's rates by how many swings back it. Covers lo..hi (the depths his similar swings reach)."""
    if rows is None or len(rows) < 25:
        return None
    w = np.ones(len(rows))
    for c, sc in KERNEL.items():
        if c in target and target[c] is not None and c in rows:
            w *= np.exp(-0.5 * ((rows[c].to_numpy() - float(target[c])) / sc) ** 2)
    if w.sum() <= 0:
        return None
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    order = np.argsort(rows["pf"].to_numpy())
    cw = np.cumsum(w[order]) / w.sum()
    pfs = rows["pf"].to_numpy()[order]
    center = float(pfs[np.searchsorted(cw, 0.5)])
    lo, hi = float(pfs[np.searchsorted(cw, 0.05)]), float(pfs[min(np.searchsorted(cw, 0.95), len(pfs) - 1)])
    u = rows["pf"].to_numpy() - center
    X = np.column_stack([np.ones(len(rows)), u, u * u, rows["plate_z"].to_numpy() - float(target.get("plate_z", 0)),
                         rows["xh"].to_numpy() - float(target.get("xh", 0))])
    sw = np.sqrt(w)
    lg = _league_for(lr or {}, float(target.get("swing_path_tilt") or 34))
    out = {"center": round(center, 2), "lo": round(lo, 1), "hi": round(hi, 1), "n_eff": round(n_eff, 1)}
    for tcol in TARGETS:
        b, *_ = np.linalg.lstsq(X * sw[:, None], rows[tcol].to_numpy() * sw, rcond=None)
        L = lg.get(tcol) or {"slope": 0.0, "curve": 0.0}
        f = n_eff / (n_eff + k)
        out[tcol] = {"at": round(float(b[0]), 3), "slope": round(float(f * b[1] + (1 - f) * L["slope"]), 5),
                     "curve": round(float(f * b[2] + (1 - f) * L["curve"]), 6)}
    return out


def value_at(curve: dict, tcol: str, pf: float) -> float:
    """The change from his centre depth: slope * u + curve * u^2 (u = pf - centre)."""
    c = curve[tcol]
    u = pf - curve["center"]
    return c["slope"] * u + c["curve"] * u * u


# ----------------------------------------------------------------------------- held-out test
def evaluate(d: pd.DataFrame, folds: int = 5, seed: int = 7, max_hitters: int | None = None) -> dict:
    """Per hitter (MIN_SWINGS+ contact swings): predict held-out swings' attack angle / direction at their depth.
    measured = his fitted rates; circle = his swings' average circle rate; flat = no change with depth. All three
    share the same anchor (fit with the rate fixed), so only the rate differs."""
    rng = np.random.default_rng(seed)
    keys = d["key"].value_counts()
    keys = keys[keys >= MIN_SWINGS].index.tolist()
    if max_hitters:
        keys = keys[:max_hitters]
    rows = []
    for key in keys:
        s = d[d["key"] == key]
        cr = circle_rates(s)
        s = s.assign(c_aa=cr["c_aa"].to_numpy(), c_ad=cr["c_ad"].to_numpy())
        fold = rng.integers(0, folds, len(s))
        for f in range(folds):
            tr, te = s[fold != f], s[fold == f]
            if len(te) == 0 or len(tr) < 50:
                continue
            c = float(tr["pf"].mean())
            ut, ue = (tr["pf"] - c).to_numpy(), (te["pf"] - c).to_numpy()
            Ct = np.column_stack([np.ones(len(tr)), tr["plate_z"], tr["xh"], tr["swing_path_tilt"]])
            Ce = np.column_stack([np.ones(len(te)), te["plate_z"], te["xh"], te["swing_path_tilt"]])
            for tcol, ccol in (("attack_angle", "c_aa"), ("attack_direction", "c_ad")):
                y = tr[tcol].to_numpy()
                # measured: rate + curve fitted with the anchor
                b, *_ = np.linalg.lstsq(np.column_stack([Ct, ut, ut * ut]), y, rcond=None)
                pm = np.column_stack([Ce, ue, ue * ue]) @ b
                # circle: his swings' circle rate, anchor fitted with that rate fixed
                rc = float(np.nanmedian(tr[ccol]))
                bc, *_ = np.linalg.lstsq(Ct, y - rc * ut, rcond=None)
                pc = Ce @ bc + rc * ue
                # flat: no change with depth
                bf, *_ = np.linalg.lstsq(Ct, y, rcond=None)
                pfl = Ce @ bf
                yt = te[tcol].to_numpy()
                for u_, a, m, ci, fl in zip(ue, yt, pm, pc, pfl):
                    rows.append((key, tcol, u_, a - m, a - ci, a - fl))
    if not rows:
        return {}
    r = pd.DataFrame(rows, columns=["key", "target", "u", "measured", "circle", "flat"])
    rmse = lambda e: round(float(np.sqrt(np.mean(np.square(e)))), 3)
    out = {"hitters": len(keys), "swings": int(len(r) // 2)}
    for tcol, g in r.groupby("target"):
        res = {"all": {m: rmse(g[m]) for m in ("measured", "circle", "flat")}}
        for name, m in (("deep (6+ in behind his avg)", g["u"] <= -6), ("middle", g["u"].abs() < 6),
                        ("out front (6+ in ahead)", g["u"] >= 6)):
            if m.sum() >= 30:
                res[name] = {mm: rmse(g.loc[m, mm]) for mm in ("measured", "circle", "flat")}
                res[name]["n"] = int(m.sum())
        # share of hitters the measured rates beat the circle for
        per = g.groupby("key").apply(lambda x: np.mean(x["measured"] ** 2) < np.mean(x["circle"] ** 2),
                                     include_groups=False)
        res["hitters_measured_beats_circle"] = round(float(per.mean()), 3)
        out[tcol] = res
    return out


# ----------------------------------------------------------------------------- the rebuilt path
def _dir(aa: float, ad: float, side: str) -> np.ndarray:
    h = math.radians(ad if side == "R" else -ad)
    a = math.radians(aa)
    return np.array([math.sin(h) * math.cos(a), math.cos(h) * math.cos(a), math.sin(a)])


R_MIN, R_MAX = 24.0, 60.0          # sweet-spot turning radius allowed beyond the measured span (in)
MAX_TURN_PRE, MAX_TURN_POST = 180.0, 90.0      # degrees of arc drawn before (safety cap) / after (the finish) the zone


def _extend(P: np.ndarray, T: np.ndarray, kvec: np.ndarray, sign: int, total: float, step: float,
            center=None) -> list:
    """Past what his swings cover: carry on round a circle in the swing's plane -- bending toward `center` (the
    swing's centre of rotation, from his tilt) at the measured path's own curvature at the edge of the span (radius
    kept to R_MIN..R_MAX), so the path turns smoothly instead of spiralling. No centre: the measured curve's own
    plane; no curvature either: straight."""
    T = T / np.linalg.norm(T)
    k = float(np.linalg.norm(kvec))
    N = None
    if center is not None:                     # the swing's own plane: bend toward its centre of rotation
        N = np.asarray(center, float) - P
        N = N - (N @ T) * T
        if k <= 1e-4:
            k = 1.0 / max(float(np.linalg.norm(N)), 1e-6)
    elif k > 1e-4:
        N = kvec - (kvec @ T) * T
    if N is not None and np.linalg.norm(N) > 1e-9:
        N = N / np.linalg.norm(N)
        R = float(np.clip(1.0 / max(k, 1e-9), R_MIN, R_MAX))
        total = min(total, math.radians(MAX_TURN_PRE if sign < 0 else MAX_TURN_POST) * R)
    else:
        N, R = None, None
    out, s = [], 0.0
    while s + step <= total + 1e-9:
        s += step
        if N is None:
            Q = P + sign * T * s
        else:
            th = s / R
            Q = P + R * N * (1 - math.cos(th)) + sign * R * T * math.sin(th)
        out.append((Q, s))
    return out


def swing_time(swing_len_in: float, bat_speed_mph: float) -> float:
    """Seconds from the start of the swing to contact: the sweet spot travels his Statcast swing length, speeding up
    evenly from rest to his bat speed (uniform acceleration -> t = 2 L / v). A 7.3 ft swing at 72 mph -> ~0.14 s."""
    v = max(float(bat_speed_mph), 30.0) * 17.6
    return 2.0 * max(float(swing_len_in), 1.0) / v


def rebuild(C, aa0: float, ad0: float, bs0: float, side: str, curve: dict, pre_len: float, post_len: float,
            step: float = 0.25, center=None, swing_len: float | None = None) -> dict:
    """The sweet spot's path from his measured rates. C = contact point (in; y = inches in front of the plate),
    aa0 / ad0 / bs0 = the swing's angles + speed at contact. Steps along by depth while inside lo..hi (each step
    pointing the way his swings point at that depth); past that, carries on round the measured curve (a circle in its
    own plane) out to pre_len / post_len inches of path. Returns points [x, y, z, fitted?, t], and the bat's
    direction (sweet spot -> hands) at each: toward the centre of the path's curve. center = the swing's centre of
    rotation. swing_len = his Statcast swing length (in): when given, the lead-in is exactly that long and is timed
    by even acceleration from rest (so the swing starts swing_time() before contact); the follow-through is a quarter
    turn at most (MAX_TURN_POST) -- nothing after contact is measured."""
    C = np.asarray(C, float)
    if swing_len and swing_len > 6:
        pre_len = float(swing_len)
    pf0 = float(C[1])
    lo, hi = min(curve["lo"], pf0), max(curve["hi"], pf0)
    v0 = max(bs0, 30.0) * 17.6

    def angles(pf):
        return (aa0 + value_at(curve, "attack_angle", pf) - value_at(curve, "attack_angle", pf0),
                ad0 + value_at(curve, "attack_direction", pf) - value_at(curve, "attack_direction", pf0),
                bs0 + value_at(curve, "bat_speed", pf) - value_at(curve, "bat_speed", pf0))

    def walk(sign: int, total: float):
        """sign -1 = back toward the start of the swing, +1 = on past contact."""
        P, s, pts = C.copy(), 0.0, []
        aa, ad, bs = aa0, ad0, bs0
        rate = (0.0, 0.0)                     # last measured turning rate (deg per inch travelled, in walk order)
        measured = True
        while s < total:
            if measured:
                t = _dir(aa, ad, side)
                if t[1] < 0.25:                                        # barrel turning across: depth stops working
                    measured = False
                    continue
                ds = step / t[1]
                Pn = P + sign * t * ds
                if not (lo <= Pn[1] <= hi):
                    measured = False
                    continue
                a2, d2, b2 = angles(Pn[1])
                rate = ((a2 - aa) / ds, (d2 - ad) / ds)
                P, s, aa, ad, bs = Pn, s + ds, a2, d2, b2
                pts.append((P.copy(), s, aa, ad, bs, True))
                continue
            # past what his swings cover: a smooth arc continuing the measured curve (see _extend)
            T = _dir(aa, ad, side)
            kvec = sign * (_dir(aa + rate[0], ad + rate[1], side) - T)       # forward curvature, per inch
            for Q, e in _extend(P, T, kvec, sign, total - s, step, center):
                pts.append((Q, s + e, aa, ad, bs, False))
            break
        return pts

    back, fwd = walk(-1, pre_len), walk(1, post_len)
    seq = [(p, -s, a, d, b, f) for p, s, a, d, b, f in reversed(back)] + [(C.copy(), 0.0, aa0, ad0, bs0, True)] + \
          [(p, s, a, d, b, f) for p, s, a, d, b, f in fwd]
    # time: integrate ds / speed out from contact
    ss = np.array([q[1] for q in seq])
    vv = np.array([max(q[4], 30.0) * 17.6 for q in seq])
    i0 = len(back)
    t = np.zeros(len(seq))
    for i in range(i0 + 1, len(seq)):
        t[i] = t[i - 1] + (ss[i] - ss[i - 1]) / (0.5 * (vv[i] + vv[i - 1]))
    if swing_len and swing_len > 6:                  # even acceleration from rest over the swing length
        L = max(-ss[0], 1.0)
        for i in range(i0 - 1, -1, -1):
            u = min(-ss[i] / L, 1.0)
            t[i] = -(2 * L / v0) * (1 - math.sqrt(1 - u))
    else:
        for i in range(i0 - 1, -1, -1):
            t[i] = t[i + 1] - (ss[i + 1] - ss[i]) / (0.5 * (vv[i] + vv[i + 1]))
    P = np.array([q[0] for q in seq])
    # the bat's direction: toward the centre of the path's curve (the circle's rule, on the measured path)
    T = np.gradient(P, axis=0)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    N = np.gradient(T, axis=0)
    nn = np.linalg.norm(N, axis=1)
    good = nn > 1e-4
    N[good] = N[good] / nn[good, None]
    if (~good).any():
        N[~good] = np.nan
        N = pd.DataFrame(N).ffill().bfill().fillna(0).to_numpy()
    if center is not None:          # the bat lies in his swing plane, pointing at its centre of rotation
        Nc = np.asarray(center, float)[None, :] - P
        Nc = Nc - (Nc * T).sum(1, keepdims=True) * T
        nc = np.linalg.norm(Nc, axis=1)
        okc = nc > 1e-6
        N[okc] = Nc[okc] / nc[okc, None]
    pts = [[round(float(x), 2) for x in P[i]] + [3 if seq[i][5] else 2, round(float(t[i]), 5)] for i in range(len(seq))]
    return {"arc": pts, "hands": [[round(float(x), 3) for x in N[i]] for i in range(len(seq))],
            "speed0": round(v0, 1), "range": [round(lo, 1), round(hi, 1)]}


# ----------------------------------------------------------------------------- run
def run(season: int, max_hitters: int | None = None) -> dict:
    from pipeline import bat_cube
    cube = bat_cube.load(season)
    d = contacts(cube)
    lr = league_rates(d)
    ev = evaluate(d, max_hitters=max_hitters)
    out = {"season": season, "league": lr, "test": ev}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"path_fit_{season}.json").write_text(json.dumps(out, indent=1))
    report(out)
    return out


def load(season: int) -> dict:
    f = OUT / f"path_fit_{season}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def report(out: dict) -> None:
    lr, ev = out.get("league", {}), out.get("test", {})
    a = lr.get("all", {})
    print(f"\n[{out.get('season')}] Bat path through the zone: {lr.get('n', 0):,} contact swings, "
          f"{lr.get('hitters', 0)} hitter-seasons")
    if a:
        print("  per inch of contact depth (further out front), within each hitter, pitch location held fixed:")
        print(f"    attack angle      measured {a['attack_angle']['slope']:+.3f} deg/in   circle assumes "
              f"{a['circle']['attack_angle']:+.3f}")
        print(f"    attack direction  measured {a['attack_direction']['slope']:+.3f} deg/in   circle assumes "
              f"{a['circle']['attack_direction']:+.3f}   (+ = oppo)")
        print(f"    bat speed         measured {a['bat_speed']['slope']:+.3f} mph/in   circle assumes +0.000")
        for k, v in (lr.get("by_tilt") or {}).items():
            print(f"    tilt {k:>5} deg  AA {v['attack_angle']['slope']:+.3f} (circle {v['circle']['attack_angle']:+.3f})"
                  f"  AD {v['attack_direction']['slope']:+.3f} (circle {v['circle']['attack_direction']:+.3f})"
                  f"  n={v['n']:,}")
    for tcol in ("attack_angle", "attack_direction"):
        r = ev.get(tcol)
        if not r:
            continue
        print(f"  held-out {tcol.replace('_', ' ')} error (deg, RMSE): measured / circle / no-change")
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"    {k:<30} {v['measured']:.2f} / {v['circle']:.2f} / {v['flat']:.2f}"
                      + (f"   n={v['n']:,}" if "n" in v else ""))
        print(f"    measured beats the circle for {100 * r['hitters_measured_beats_circle']:.0f}% of hitters")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mh = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--max=")), None)
    for a in args or ["2026"]:
        run(int(a), mh)
