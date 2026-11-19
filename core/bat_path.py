"""Bat Path tab: point of contact vs the plate for a hitter's swings (same method as the AAR -- MLB's definition).

  front (pf)  = Statcast's contact in front of his center of mass - his stance depth (Savant's board: playoff ->
                vs this pitcher's hand -> season), inches in front of the plate's front edge
  side  (pxc) = where the ball was at that depth, inches from the middle of the plate, catcher's view
Hitters not on the stance board fall back to the AAR's calibrated estimate (flagged).
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd

from core import aar

POST = {"F", "D", "L", "W"}


def poc(sc: pd.DataFrame, pid: int, season: int, views: dict | None = None, stance: dict | None = None) -> pd.DataFrame:
    """Per row of sc (same index): pf, pxc, depth_src, estimated (bool)."""
    out = pd.DataFrame({"pf": np.nan, "pxc": np.nan, "depth_src": "", "estimated": False}, index=sc.index)
    if sc is None or sc.empty or aar.CONTACT_Y not in sc:
        return out
    num = lambda c: pd.to_numeric(sc[c], errors="coerce") if c in sc else pd.Series(np.nan, index=sc.index)
    made = aar.made_contact(sc)
    p = pd.DataFrame({"cy": num(aar.CONTACT_Y).where(made), "cx": num(aar.CONTACT_X).where(made),
                      "stand": sc["stand"].astype(str), "p_throws": sc["p_throws"].astype(str),
                      "px": num("plate_x"), "slope": aar.plate_slope(sc)}, index=sc.index)
    post = sc["game_type"].astype(str).isin(POST) if "game_type" in sc else pd.Series(False, index=sc.index)
    st = pd.concat([aar.pitch_stance(p[~post], pid, views or {}, False),
                    aar.pitch_stance(p[post], pid, views or {}, True)]).reindex(sc.index)
    p = p.join(st)
    cal = None
    if p["depth"].isna().any() and p["cy"].notna().any():
        try:
            cal = aar.plate_calibration(sc, group=aar.hip_group(season), stance=stance or {})
        except Exception:
            cal = None
    o = aar.on_plate(p, cal)
    out["pf"], out["pxc"] = o["pf"], o["pxc"]
    out["depth_src"] = p["depth_src"].fillna("")
    out["estimated"] = o["pf"].notna() & p["depth"].isna()
    return out


# ----------------------------------------------------------------------------- the 3D "his path" view
# A modeled pitch (not a real one) + his AVERAGE measured swing on that kind of pitch in that zone.
#   pitch: shape per group below (catcher's-view release; HB arm side +), velo = average of that group from that
#          pitcher hand he actually saw this season (falls back to the defaults below)
#   swing: means of his competitive swings (Statcast's definition) for pitch group x zone x pitcher hand, widening
#          when there are too few: group+zone+hand -> group+zone -> zone -> group -> all
FT = 1.4667                                   # mph -> ft/s
G = 32.174
GROUPS = {   # name: (pitch type, default velo by hand, IVB in, HB in arm side, drag ay ft/s^2 per mph)
    "Fastball": ("FF", {"R": 94.3, "L": 93.4}, 14.0, 9.0, "dead-zone four-seam: average ride, average run"),
    "Breaking": ("SL", {"R": 85.5, "L": 84.6}, 1.0, -5.0, "slider: little ride, a few inches glove side"),
    "Offspeed": ("CH", {"R": 86.0, "L": 84.5}, 6.0, 14.0, "changeup: sinks + runs arm side"),
}
RELEASE = {"R": (-1.9, 5.8, 6.4), "L": (1.9, 5.8, 6.4)}      # release x, z, extension (ft, catcher's view)
ZONES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
MIN_SWINGS, MIN_CONTACT = 5, 3


def zone_name(zone: int, stand: str) -> str:
    """Statcast zone -> hitter's-eye name (catcher's left is inside to a righty, away to a lefty)."""
    left, right = ("In", "Away") if stand == "R" else ("Away", "In")
    if zone <= 9:
        r, c = divmod(zone - 1, 3)
        return f'{("Up", "Middle", "Down")[r]}-{(left, "Middle", right)[c]}'
    return f'Chase {"Up" if zone in (11, 12) else "Down"}-{left if zone in (11, 13) else right}'


def zone_target(zone: int, sz_bot: float, sz_top: float) -> tuple[float, float]:
    """(plate_x, plate_z) ft at the centre of a Statcast zone (catcher's view; 1-9 inside his zone, 11-14 corners
    outside it)."""
    third_x, third_z = 17 / 36, (sz_top - sz_bot) / 3
    if zone <= 9:
        r, c = divmod(zone - 1, 3)
        return (c - 1) * third_x, sz_top - (r + 0.5) * third_z
    x = -1.15 if zone in (11, 13) else 1.15
    z = sz_top + 0.35 if zone in (11, 12) else sz_bot - 0.35
    return x, z


def synth_pitch(velo: float, ivb: float, hb_arm: float, hand: str, plate_x: float, plate_z: float) -> dict:
    """Statcast-style kinematics (velocity + acceleration at y = 50 ft) for a pitch with this velo / movement from a
    typical release, arriving at (plate_x, plate_z) at the front of the plate."""
    rx, rz, ext = RELEASE[hand]
    y_r, y_f = 60.5 - ext, 17 / 12
    v = velo * FT
    ay = 0.32 * velo                                            # drag, ft/s^2 (typical Statcast ay at that velo)
    t40 = (50 - y_f) / (v * 0.93)                               # Statcast's movement window (y = 50 ft -> plate)
    pfx_x = (hb_arm if hand == "L" else -hb_arm) / 12           # catcher's view: RHP arm side = -x
    ax, az = 2 * pfx_x / t40 ** 2, 2 * (ivb / 12) / t40 ** 2 - G
    vy_r = -v * 0.995
    a, b, c = 0.5 * ay, vy_r, y_r - y_f
    T = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    vx_r = (plate_x - rx - 0.5 * ax * T ** 2) / T
    vz_r = (plate_z - rz - 0.5 * az * T ** 2) / T
    c50 = y_r - 50
    t50 = (-b - np.sqrt(b * b - 4 * a * c50)) / (2 * a) if y_r > 50 else 0.0
    return {"vx0": vx_r + ax * t50, "vy0": vy_r + ay * t50, "vz0": vz_r + az * t50, "ax": ax, "ay": ay, "az": az,
            "plate_x": plate_x, "plate_z": plate_z, "release_pos_y": y_r, "release_pos_x": rx, "release_pos_z": rz,
            "release_speed": velo, "pfx_x": pfx_x, "pfx_z": ivb / 12, "release_extension": ext}


HANDS = {"All": None, "RHP": "R", "LHP": "L"}
PITCH_NAMES = {"FF": "4-Seam", "FA": "Fastball", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper",
               "SV": "Slurve", "CU": "Curveball", "KC": "Knuckle Curve", "CS": "Slow Curve", "CH": "Changeup",
               "FS": "Splitter", "FO": "Forkball", "SC": "Screwball", "KN": "Knuckleball", "EP": "Eephus",
               "PO": "Pitchout"}
COUNTS = ["All", "First Pitch", "Ahead", "Even", "Behind", "Two Strikes"]


def zones_of(zone) -> list[int] | None:
    """zone = one zone, a list of zones, or None (every location) -> list or None. All nine = every location."""
    if zone is None:
        return None
    zs = sorted({int(z) for z in (zone if isinstance(zone, (list, tuple, set)) else [zone])})
    return None if not zs or set(zs) >= set(range(1, 10)) else zs


def zones_label(zone, stand: str) -> str:
    zs = zones_of(zone)
    if zs is None:
        return "All locations"
    if len(zs) <= 2:
        return " + ".join(zone_name(z, stand) for z in zs)
    return f"{len(zs)} zones"


def pick(df: pd.DataFrame, types=None, hand: str = "All", count: str = "All", zone: int | None = None,
         stand: str | None = None) -> pd.Series:
    """Rows matching the filters (types = None / empty -> every pitch type; zone None -> every location)."""
    m = pd.Series(True, index=df.index)
    if types:
        m &= df["ptype"].isin(list(types))
    if HANDS.get(hand):
        m &= df["hand"].eq(HANDS[hand])
    if count and count != "All":
        m &= df["count"].eq(count)
    zs = zones_of(zone)
    if zs is not None:
        m &= df["zone"].isin(zs)
    if stand:
        m &= df["stand"].eq(stand)
    return m


def cell_balls(sc: pd.DataFrame, poc: pd.DataFrame, types=None, hand: str = "All", count: str = "All",
               zone: int | None = None, stand: str | None = None) -> pd.DataFrame:
    """Every contact (fouls + balls in play) matching the filters, for the spray chart + point of contact."""
    c = contact_rows(sc, poc)
    return c if c.empty else c[pick(c, types, hand, count, zone, stand)]


def demo_group(types, all_types) -> str:
    """Which modeled pitch to throw: a dead-zone fastball unless every pitch type picked is one breaking or
    offspeed group."""
    from core import hit_metrics as hm
    if not types or set(types) >= set(all_types):
        return "Fastball"
    gs = set(hm.pitch_group(pd.DataFrame({"pitch_type": list(types)})).dropna())
    return gs.pop() if len(gs) == 1 else "Fastball"


def contact_rows(sc: pd.DataFrame, poc: pd.DataFrame) -> pd.DataFrame:
    """Every contact (fouls + balls in play) with what the spray / point-of-contact charts colour by."""
    from core import batted_ball
    from core import hit_metrics as hm
    if sc is None or sc.empty:
        return pd.DataFrame()
    num = lambda c: pd.to_numeric(sc[c], errors="coerce") if c in sc else pd.Series(np.nan, index=sc.index)
    desc = sc["description"].astype(str)
    m = desc.isin(["foul", "foul_tip", "hit_into_play"])
    try:
        ctype = batted_ball.spin_labels(sc)
    except Exception:
        ctype = pd.Series(None, index=sc.index, dtype=object)
    out = pd.DataFrame({
        "bb": (sc["bb_type"] if "bb_type" in sc else pd.Series(None, index=sc.index)).map(aar.BB),
        "ctype": ctype,
        "pf": poc["pf"] if "pf" in poc else np.nan, "pxc": poc["pxc"] if "pxc" in poc else np.nan,
        "bip": desc.eq("hit_into_play"), "ev": num("launch_speed"), "la": num("launch_angle"),
        "hc_x": num("hc_x").where(desc.eq("hit_into_play")), "hc_y": num("hc_y").where(desc.eq("hit_into_play")),
        "pgroup": hm.pitch_group(sc), "ptype": sc["pitch_type"].astype(str) if "pitch_type" in sc else "",
        "p_throws": sc["p_throws"].astype(str), "zone_io": np.where(num("zone").le(9), "In Zone", "Out of Zone"),
        "hand": sc["p_throws"].astype(str), "zone": num("zone"), "count": hm.count_group(sc),
        "stand": sc["stand"].astype(str),
        "date": sc["game_date"].astype(str).str[:10],
        "result": (sc["events"].fillna(sc["description"]) if "events" in sc else desc).astype(str)
                  .str.replace("_", " ").str.capitalize()}, index=sc.index)
    return out[m]


def _plane(aa, ad, tilt, stand, pf, depth, px_in, pz_in, stance_x=None):
    """Per swing: d (sweet-spot direction at contact, from the measured attack angle + direction) and w (unit vector
    from the ball toward the hands' centre, in the plane tilted at the measured swing-path tilt) -- the 3D view's
    geometry, vectorised. Plate frame: x = 1B side, y = toward the pitcher, z = up."""
    aa, ad, tilt = (np.asarray(x, float) for x in (aa, ad, tilt))
    pf, depth, px_in, pz_in = (np.asarray(x, float) for x in (pf, depth, px_in, pz_in))
    right = np.asarray(stand) == "R"
    sx = np.asarray([((stance_x or {}).get("R" if r else "L") or 27.0) for r in right], float)
    h = np.radians(np.where(right, ad, -ad))
    a = np.radians(aa)
    d = np.stack([np.sin(h) * np.cos(a), np.cos(h) * np.cos(a), np.sin(a)], -1)                    # n x 3
    C = np.stack([px_in, pf, pz_in], -1)
    H = np.stack([np.where(right, -(sx + 8.5), sx + 8.5), -depth, pz_in], -1)
    tw = H - C
    tw = tw - (tw * d).sum(-1, keepdims=True) * d
    flat = np.stack([tw[:, 0], tw[:, 1], np.zeros(len(tw))], -1)
    flat = flat / np.linalg.norm(flat, axis=-1, keepdims=True)
    th = np.radians(np.linspace(0, 89.9, 180))[None, :, None]
    w = np.cos(th) * flat[:, None, :] + np.sin(th) * np.array([0, 0, 1.0])[None, None, :]            # n x k x 3
    w = w - (w * d[:, None, :]).sum(-1, keepdims=True) * d[:, None, :]
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    nrm = np.cross(np.broadcast_to(d[:, None, :], w.shape), w)
    err = np.abs(np.arccos(np.clip(np.abs(nrm[..., 2]) / np.linalg.norm(nrm, axis=-1), 0, 1)) - np.radians(tilt)[:, None])
    best = w[np.arange(len(w)), np.nanargmin(np.where(np.isnan(err), np.inf, err), axis=1)]
    bad = ~np.isfinite(aa + ad + tilt + pf + depth + px_in + pz_in)
    return d, best, bad


def bat_angles(aa, ad, tilt, stand, pf, depth, px_in, pz_in, stance_x=None) -> tuple[np.ndarray, np.ndarray]:
    """Vertical + horizontal bat angle at contact for each swing (degrees), from that swing's own measured attack
    angle, attack direction and swing-path tilt -- the same swing-plane geometry the 3D view uses (INFERRED: the bat
    lies across the measured path, in the measured plane). vba < 0 = barrel below the hands; hba > 0 = barrel ahead
    of the hands (toward the pitcher), 0 = square."""
    aa = np.asarray(aa, float)
    lr = _league_rates()
    if lr:                                                   # the measured path's curve (beat the circle on spray)
        from core.collision import axis_measured
        w = axis_measured(aa, ad, stand, tilt, lr)
        bad = ~np.isfinite(aa + np.asarray(ad, float) + np.asarray(tilt, float))
        vba = -np.degrees(np.arcsin(np.clip(w[:, 2], -1, 1)))
        hba = -np.degrees(np.arctan2(w[:, 1], np.abs(w[:, 0])))
        return np.where(bad, np.nan, vba), np.where(bad, np.nan, hba)
    if len(aa) > 8000:                                       # in chunks: n x 180 x 3 arrays
        args = [np.asarray(x) for x in (aa, ad, tilt, stand, pf, depth, px_in, pz_in)]
        parts = [bat_angles(*[x[i:i + 8000] for x in args], stance_x) for i in range(0, len(aa), 8000)]
        return np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])
    d, w, bad = _plane(aa, ad, tilt, stand, pf, depth, px_in, pz_in, stance_x)
    vba = -np.degrees(np.arcsin(np.clip(w[:, 2], -1, 1)))
    hba = -np.degrees(np.arctan2(w[:, 1], np.abs(w[:, 0])))
    return np.where(bad, np.nan, vba), np.where(bad, np.nan, hba)


TGRID = np.arange(-80, 72, 2.0)                               # ms from contact, for the time view's ranges
TIME_METRICS = ["attack_angle", "attack_direction", "vba", "hba"]


def time_curves(aa, ad, tilt, bs, swing_len, stand, pf, depth, cx, px_in, pz_in, stance_x=None,
                grid=TGRID, lr: dict | None = None) -> dict:
    """Each swing's four angles along its path over time, every 2 ms. With the league's MEASURED path rates
    (core.path_fit) when they're built -- the same rule the 3D view's measured path uses -- else the circle."""
    lr = _league_rates() if lr is None else lr
    if lr:
        return time_curves_measured(aa, ad, tilt, bs, swing_len, stand, pf, lr, grid)
    return time_curves_circle(aa, ad, tilt, bs, swing_len, stand, pf, depth, cx, px_in, pz_in, stance_x, grid)


MEASURED_SPAN = 15.0          # inches of contact depth either side of contact the measured rates are trusted over
POST_IN = 40.0                # inches of path drawn past contact (follow-through)


def time_curves_measured(aa, ad, tilt, bs, swing_len, stand, pf, lr: dict, grid=TGRID, dt_ms: float = 0.5) -> dict:
    """Each swing's path stepped out from contact at the league's MEASURED rates for its tilt: attack angle and
    direction change per inch of contact depth (+ how that rate changes), bat speed too. Past MEASURED_SPAN inches
    of depth the path keeps turning at the rate it had there. Drawn back to the start of his measured swing length
    and POST_IN inches past contact. Bat orientation (vba / hba): toward the centre of the path's curve."""
    from core import path_fit
    aa0, ad0, tilt, bs0 = (np.asarray(x, float) for x in (aa, ad, tilt, bs))
    swing_len = np.asarray(swing_len, float) * 12
    n = len(aa0)
    right = np.asarray(stand) != "L"
    side = np.where(right, 1.0, -1.0)
    coef = {}
    for k in ("attack_angle", "attack_direction", "bat_speed"):
        coef[k] = np.array([[(path_fit._league_for(lr, float(t) if t == t else 34.0).get(k) or {}).get(c, 0.0)
                             for c in ("slope", "curve")] for t in tilt]).reshape(n, 2)
    grid = np.asarray(grid, float)
    out = {k: np.full((n, len(grid)), np.nan) for k in TIME_METRICS}
    pos = np.full((n, len(grid), 3), np.nan)
    dt = dt_ms / 1000.0
    for sgn in (1.0, -1.0):
        a, d_, u, s_, P = aa0.copy(), ad0.copy(), np.zeros(n), np.zeros(n), np.zeros((n, 3))
        ra, rd = np.zeros(n), np.zeros(n)
        cols = [j for j, g in enumerate(grid) if (g >= 0) == (sgn > 0)]
        cols = sorted(cols, key=lambda j: abs(grid[j]))
        tnow, ci = 0.0, 0
        limit = np.where(sgn > 0, POST_IN, np.where(np.isfinite(swing_len) & (swing_len > 0), swing_len, 40.0))
        while ci < len(cols):
            while ci < len(cols) and abs(grid[cols[ci]]) / 1000.0 <= tnow + 1e-12:
                j = cols[ci]
                out["attack_angle"][:, j] = np.where(s_ <= limit, a, np.nan)
                out["attack_direction"][:, j] = np.where(s_ <= limit, d_, np.nan)
                pos[:, j] = np.where((s_ <= limit)[:, None], P, np.nan)
                ci += 1
            h = np.radians(side * d_)
            ar = np.radians(a)
            T = np.stack([np.sin(h) * np.cos(ar), np.cos(h) * np.cos(ar), np.sin(ar)], -1)
            v = np.maximum(bs0 + coef["bat_speed"][:, 0] * np.clip(u, -MEASURED_SPAN, MEASURED_SPAN), 30.0) * 17.6
            ds = v * dt
            inside = np.abs(u) < MEASURED_SPAN
            ty = np.maximum(T[:, 1], 0.0)
            ra = np.where(inside, (coef["attack_angle"][:, 0] + 2 * coef["attack_angle"][:, 1] * u) * ty, ra)
            rd = np.where(inside, (coef["attack_direction"][:, 0] + 2 * coef["attack_direction"][:, 1] * u) * ty, rd)
            P = P + sgn * T * ds[:, None]
            u = u + sgn * T[:, 1] * ds
            a = a + sgn * ra * ds
            d_ = d_ + sgn * rd * ds
            s_ = s_ + ds
            tnow += dt
    # bat orientation from the path's curve (sweet spot -> hands = toward the centre of curvature)
    Tg = np.gradient(pos, axis=1)
    Tg = Tg / np.linalg.norm(Tg, axis=-1, keepdims=True)
    N = np.gradient(Tg, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        N = N / np.linalg.norm(N, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        out["vba"] = -np.degrees(np.arcsin(np.clip(N[..., 2], -1, 1)))
        out["hba"] = -np.degrees(np.arctan2(N[..., 1], np.abs(N[..., 0])))
    bad = ~np.isfinite(aa0 + ad0 + tilt + bs0)
    return {k: np.where(bad[:, None], np.nan, v) for k, v in out.items()}


def time_curves_circle(aa, ad, tilt, bs, swing_len, stand, pf, depth, cx, px_in, pz_in, stance_x=None,
                       grid=TGRID) -> dict:
    """Each swing's four angles along its own drawn path, sampled every 2 ms (NaN where the path isn't drawn: before
    bat parallel / beyond the swing length, or past the follow-through) -- the circle (before measured rates)."""
    aa = np.asarray(aa, float)
    if len(aa) > 4000:
        args = [np.asarray(x) for x in (aa, ad, tilt, bs, swing_len, stand, pf, depth, cx, px_in, pz_in)]
        parts = [time_curves_circle(*[x[i:i + 4000] for x in args], stance_x, grid) for i in range(0, len(aa), 4000)]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d, w, bad = _plane(aa, ad, tilt, stand, pf, depth, px_in, pz_in, stance_x)
    bs, swing_len = np.asarray(bs, float) * 17.6, np.asarray(swing_len, float) * 12
    cx, pf, depth = (np.asarray(x, float) for x in (cx, pf, depth))
    R = np.clip(np.hypot(np.nan_to_num(cx, nan=30.0), pf + depth), 24, 48)
    measured = np.minimum(bs * 0.040, R * 1.2)
    phi_bp = -np.pi / 2 - np.arctan2(d[:, 2], w[:, 2])
    s_bp = R * phi_bp
    bp_ok = (np.nan_to_num(swing_len) <= 0) | (-s_bp <= swing_len)
    start = np.where(bp_ok, np.minimum(-measured, s_bp), -measured)                 # inches along the path
    end = R * np.pi / 2
    s_ = bs[:, None] * (np.asarray(grid)[None, :] / 1000.0)                         # n x T inches
    phi = s_ / R[:, None]
    c, sn = np.cos(phi)[..., None], np.sin(phi)[..., None]
    tang = w[:, None, :] * sn + d[:, None, :] * c                                   # direction of travel
    hand = w[:, None, :] * c - d[:, None, :] * sn                                   # barrel -> hands (unit)
    side = np.where(np.asarray(stand) == "R", 1.0, -1.0)[:, None]
    wrap = lambda v: (v + 540) % 360 - 180
    out = {"attack_angle": np.degrees(np.arctan2(tang[..., 2], np.hypot(tang[..., 0], tang[..., 1]))),
           "attack_direction": wrap(side * np.degrees(np.arctan2(tang[..., 0], tang[..., 1]))),
           "vba": -np.degrees(np.arcsin(np.clip(hand[..., 2], -1, 1))),
           "hba": -np.degrees(np.arctan2(hand[..., 1], np.abs(hand[..., 0])))}
    drawn = (s_ >= start[:, None]) & (s_ <= end[:, None]) & ~bad[:, None] & np.isfinite(bs)[:, None]
    return {k: np.where(drawn, v, np.nan) for k, v in out.items()}


def time_band(curves: dict, min_n: int = 15) -> dict:
    """Mean / SD / n at each time in the grid, from time_curves()."""
    out = {}
    for k, v in curves.items():
        n = np.isfinite(v).sum(0)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # empty / single-swing bins -> NaN, as intended
            m, s = np.nanmean(v, 0), np.nanstd(v, 0, ddof=1)
        ok = n >= min_n
        out[k] = {"m": [round(float(x), 2) if o else None for x, o in zip(m, ok)],
                  "s": [round(float(x), 2) if o else None for x, o in zip(s, ok)], "n": [int(x) for x in n]}
    return out


def swing_result(sc: pd.DataFrame) -> pd.Series:
    """whiff / foul / fair per pitch (Savant's convention: a foul tip is a whiff)."""
    d = sc["description"].astype(str) if "description" in sc else pd.Series("", index=sc.index)
    return pd.Series(np.select([d.isin(["swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"]),
                                d.isin(["foul", "foul_bunt"]), d.eq("hit_into_play")], ["whiff", "foul", "fair"], ""),
                     index=sc.index)


def swings(sc: pd.DataFrame, poc: pd.DataFrame | None = None, stance_x: dict | None = None) -> pd.DataFrame:
    """His competitive swings with what the view needs: group, zone, hand, the measured swing + contact."""
    from core import hit_metrics as hm
    if sc is None or sc.empty or "bat_speed" not in sc:
        return pd.DataFrame()
    d = hm.prep(sc)
    num = lambda c: pd.to_numeric(sc[c], errors="coerce") if c in sc else pd.Series(np.nan, index=sc.index)
    made = aar.made_contact(sc)
    out = pd.DataFrame({
        "group": hm.pitch_group(sc), "zone": num("zone"), "hand": sc["p_throws"].astype(str),
        "ptype": sc["pitch_type"].astype(str) if "pitch_type" in sc else "", "count": hm.count_group(sc),
        "stand": sc["stand"].astype(str), "bat_speed": d["bat_speed"], "swing_length": num("swing_length"),
        "attack_angle": num("attack_angle"), "attack_direction": num("attack_direction"),
        "swing_path_tilt": num("swing_path_tilt"), "velo": num("release_speed"),
        "cy": num(aar.CONTACT_Y).where(made), "cx": num(aar.CONTACT_X).where(made),
        "sz_top": num("sz_top"), "sz_bot": num("sz_bot"), "px_in": 12 * num("plate_x"),
        "pz_in": 12 * num("plate_z"), "ev": num("launch_speed").where(made), "la": num("launch_angle").where(made),
        # the pitch's shape (IVB, HB arm side; in) + what happened (Savant's whiff counts foul tips)
        "ivb": 12 * num("pfx_z"), "hb": 12 * num("pfx_x") * np.where(sc["p_throws"].astype(str).eq("L"), 1, -1),
        "result": swing_result(sc)},
        index=sc.index)
    # where he met it (MLB's contact point vs the plate, from the page's point-of-contact layer) + measured plane gaps
    out["pf"] = poc["pf"].reindex(sc.index) if poc is not None and "pf" in poc else np.nan
    for k in ("gap_v", "gap_h", "match_v", "match_h", "vba", "hba"):
        out[k] = np.nan
    have = out["pf"].notna()
    if have.any() and {"vx0", "vy0", "vz0", "ax", "ay", "az"} <= set(sc.columns):
        from pipeline.bat_norms import plane
        pl = plane(sc.loc[have], out.loc[have, "pf"])
        for k in ("gap_v", "gap_h", "match_v", "match_h"):
            out.loc[have, k] = np.asarray(pl[k], float)
    if have.any():
        vba, hba = bat_angles(out.loc[have, "attack_angle"], out.loc[have, "attack_direction"],
                              out.loc[have, "swing_path_tilt"], out.loc[have, "stand"], out.loc[have, "pf"],
                              out.loc[have, "cy"] - out.loc[have, "pf"], 12 * num("plate_x")[have],
                              12 * num("plate_z")[have], stance_x)
        out.loc[have, "vba"], out.loc[have, "hba"] = vba, hba
    return out[out["bat_speed"].notna()]


def profile(sw: pd.DataFrame, types=None, hand: str = "All", count: str = "All", zone: int | None = None,
            stand: str | None = None) -> dict:
    """His average swing for the filters, widening (drop the location, then the filters) when there are too few."""
    if sw.empty:
        return {}
    s = sw if stand is None else sw[sw["stand"] == stand]
    here = "here" if zones_of(zone) is not None else "all locations"
    levels = [(f"your filters, {here}", pick(s, types, hand, count, zone)),
              ("your filters, all locations", pick(s, types, hand, count)),
              (f"every pitch, {here}", pick(s, zone=zone)),
              ("all his swings", pd.Series(True, index=s.index))]
    out = {}
    for name, m in levels:
        g = s[m]
        if len(g) >= MIN_SWINGS and not out:
            out = {"n": int(len(g)), "level": name,
                   **{k: float(g[k].mean()) for k in ("bat_speed", "swing_length", "attack_angle",
                                                     "attack_direction", "swing_path_tilt")}}
        c = g[g["cy"].notna()]
        if len(c) >= MIN_CONTACT and out and "cy" not in out:
            out.update({"cy": float(c["cy"].mean()), "cx": float(c["cx"].mean()), "n_contact": int(len(c)),
                        "contact_level": name})
    return out


def velo_for(sw: pd.DataFrame, group: str, hand: str, types=None) -> tuple[float, str]:
    m = (sw.group == group) & (sw.hand == hand) if not sw.empty else None
    if m is not None and types:
        m &= sw["ptype"].isin(list(types))
    v = sw[m]["velo"].dropna() if m is not None else pd.Series(dtype=float)
    if len(v) >= 20:
        return float(v.mean()), f"average {hand}HP {group.lower()} he swung at"
    return GROUPS[group][1][hand], f"typical {hand}HP {group.lower()}"


def model_scene(sw: pd.DataFrame, types=None, hand: str = "All", count: str = "All", zone: int | None = None,
                stand: str = "R", depth: float | None = None, stance_x: float | None = None,
                all_types=None) -> dict | None:
    """The 3D scene (core.swing3d format): his average swing for the filters vs a modeled pitch -- a dead-zone
    fastball unless the pitch types picked are all one other group; RHP unless filtered to LHP; middle-middle
    unless a location is picked."""
    from core import swing3d
    pr = profile(sw, types, hand, count, zone, stand)
    if not pr or "cy" not in pr or depth is None:
        return None
    group = demo_group(types, all_types if all_types is not None else sw["ptype"].unique())
    ph = HANDS.get(hand) or "R"
    ptype, _, ivb, hb, desc = GROUPS[group]
    me = sw[sw["stand"] == stand]
    velo, vsrc = velo_for(me, group, ph, types)
    top = float(me["sz_top"].mean()) if me["sz_top"].notna().any() else 3.4
    bot = float(me["sz_bot"].mean()) if me["sz_bot"].notna().any() else 1.6
    tg = [zone_target(z, bot, top) for z in (zones_of(zone) or [5])]          # middle-middle unless picked
    px, pz = float(np.mean([t[0] for t in tg])), float(np.mean([t[1] for t in tg]))
    row = {**synth_pitch(velo, ivb, hb, ph, px, pz), "description": "foul", "stand": stand, "p_throws": ph,
           "bat_speed": pr["bat_speed"], "swing_length": pr["swing_length"], "attack_angle": pr["attack_angle"],
           "attack_direction": pr["attack_direction"], "swing_path_tilt": pr["swing_path_tilt"],
           aar.CONTACT_Y: pr["cy"], aar.CONTACT_X: pr["cx"], "sz_top": top, "sz_bot": bot}
    sc = swing3d.scene(row, depth, stance_x)
    if sc is None:
        return None
    sc["read"].update({"ptype": ptype, "count": "", "result": "", "video": ""})
    try:
        sc = fit_path(sc, me, stand, pz, px, pr)
    except Exception:
        sc["path_src"] = {"kind": "circle"}
    sc["model"] = {**pr, "velo": velo, "velo_src": vsrc, "ivb": ivb, "hb": hb, "shape": desc, "ptype": ptype,
                   "zone": zone, "hand": ph, "group": group, "depth": depth, "px": px, "pz": pz}
    return sc


_LEAGUE_RATES: dict = {}


def _league_rates() -> dict:
    """The newest season's league path rates (core.path_fit), cached."""
    if "lr" not in _LEAGUE_RATES:
        from core import path_fit
        lr = {}
        for f in sorted(path_fit.OUT.glob("path_fit_*.json"), reverse=True):
            try:
                lr = json.loads(f.read_text()).get("league") or {}
            except Exception:
                lr = {}
            if lr:
                break
        _LEAGUE_RATES["lr"] = lr
    return _LEAGUE_RATES["lr"]


def fit_path(sc: dict, me: pd.DataFrame, stand: str, pz: float, px: float, pr: dict) -> dict:
    """Swap the assumed circle for his MEASURED path (core.path_fit) where his swings back it: his contact swings
    with the same kind of plane / speed / pitch location, lined up by contact depth, give how fast the path turns,
    rises and speeds up through the zone. Inside the depths those swings reach: measured (flag 3; Statcast's own
    window stays flag 1); beyond: carries on round the measured curve (flag 2, drawn see-through)."""
    from core import path_fit
    rows = me[me["pf"].notna()] if "pf" in me else me.iloc[0:0]
    rows = pd.DataFrame({"pf": rows.get("pf"), "plate_z": (rows.get("pz_in") / 12.0).fillna(pz),
                         "xh": ((rows.get("px_in") / 12.0) * (-1 if stand == "L" else 1)).fillna(
                             px * (-1 if stand == "L" else 1)),
                         "swing_path_tilt": rows.get("swing_path_tilt"), "bat_speed": rows.get("bat_speed"),
                         "attack_angle": rows.get("attack_angle"), "attack_direction": rows.get("attack_direction")}
                        ).dropna() if len(rows) else pd.DataFrame()
    target = {"swing_path_tilt": pr.get("swing_path_tilt"), "bat_speed": pr.get("bat_speed"), "plate_z": pz,
              "xh": px * (-1 if stand == "L" else 1)}
    cur = path_fit.hitter_curve(rows, target, _league_rates()) if len(rows) else None
    if not cur:
        sc["path_src"] = {"kind": "circle", "why": "too few contact swings to measure his path"}
        return sc
    arc = sc["arc"]
    v = float(pr.get("bat_speed") or sc["read"]["bat_speed"]) * 17.6
    pre_len = max(-arc[0][4] * v, 6.0)
    sl = sc["read"].get("swing_length")
    sl_in = float(sl) * 12 if sl is not None and np.isfinite(sl) and sl > 0 else None      # Statcast swing length
    r = path_fit.rebuild(sc["contact"], float(sc["read"]["attack_angle"]), float(sc["read"]["attack_direction"]),
                         float(sc["read"]["bat_speed"]), stand, cur, pre_len, 400.0, center=sc.get("O"),
                         swing_len=sl_in)
    pts, hands = r["arc"], r["hands"]
    for p in pts:                                            # Statcast's measured window keeps its own flag
        if p[3] == 3 and -0.040 <= p[4] <= 0:
            p[3] = 1
    i0 = min(range(len(pts)), key=lambda i: abs(pts[i][4]))
    sc["arc"], sc["hands"] = pts, hands
    sc["read"]["vba"] = round(-math.degrees(math.asin(max(-1.0, min(1.0, hands[i0][2])))), 1)
    bp = None                                                # bat parallel: hands level with the barrel, before contact
    for i in range(i0, 0, -1):
        if hands[i][2] * hands[i - 1][2] <= 0:
            bp = {"at": pts[i][:3], "inches_before": round(sum(
                math.dist(pts[j][:3], pts[j + 1][:3]) for j in range(i, i0)), 1), "t": pts[i][4]}
            break
    sc["bp"] = bp
    sc["read"]["bp_in"] = bp["inches_before"] if bp else None
    sc["path_src"] = {"kind": "measured", "n_eff": cur["n_eff"], "range": r["range"],
                      "rates": {k: cur[k]["slope"] for k in ("attack_angle", "attack_direction", "bat_speed")}}
    return sc


def home_team(sc: pd.DataFrame) -> str | None:
    """His team's abbreviation that season: the home team in the games he batted in the bottom half."""
    if sc is None or sc.empty or "inning_topbot" not in sc or "home_team" not in sc:
        return None
    h = sc.loc[sc["inning_topbot"].astype(str).str.startswith("B"), "home_team"].dropna()
    return str(h.mode().iloc[0]) if not h.empty else None


def home_park(abbr: str, season: int) -> dict:
    """{team, venue, dims, outline} for that team's home park that season (MLB stats API + the traced outlines)."""
    import requests

    from core import mlb_api
    teams = mlb_api.get_teams(season)
    alias = {"AZ": "ARI", "ARI": "AZ", "ATH": "OAK", "OAK": "ATH", "CWS": "CHW", "CHW": "CWS", "WSH": "WSN",
             "WSN": "WSH", "KC": "KCR", "SD": "SDP", "SF": "SFG", "TB": "TBR"}
    want = {abbr.upper(), alias.get(abbr.upper(), abbr.upper())}
    tid = next((i for i, t in teams.items() if str(t.get("abbr", "")).upper() in want), None)
    if tid is None:
        return {}
    name = teams[tid].get("name", "")
    venue = (mlb_api.get_team(tid).get("venue") or {})
    dims = {}
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/venues/{venue.get('id')}?hydrate=fieldInfo", timeout=15)
        v = (r.json().get("venues") or [{}])[0]
        dims = aar.field_dims({"gameData": {"venue": v}})
    except Exception:
        pass
    return {"team": name, "venue": venue.get("name", ""), "dims": dims,
            "outline": aar.park_outline(name, dims or None, venue=venue.get("name", ""))}



# ----------------------------------------------------------------------------- swing groups (compare several swings)
GROUP_COLORS = ["#2f5fb3", "#e07b24", "#2e9d5b", "#8e44ad"]
MAX_GROUPS = len(GROUP_COLORS)
DEFAULT_GROUP = {"types": None, "hand": "All", "count": "All", "zone": [5]}


def group_label(g: dict, stand: str) -> str:
    bits = [zones_label(g.get("zone"), stand),
            ", ".join(g["types"]) if g.get("types") else "All pitches",
            g["count"] if g.get("count", "All") != "All" else "All counts",
            f'vs {g["hand"]}' if g.get("hand", "All") != "All" else "Both hands"]
    return " · ".join(bits)


def group_scenes(sw: pd.DataFrame, groups: list[dict], stand: str, depth_for, stance_x, all_types,
                 nm: dict | None = None) -> list:
    """One 3D scene per swing group (None where there isn't enough), tagged with its name + colour, with his
    measured swings for the graphs (series) and the matching MLB fast-swing range (norm).
    depth_for(hand, group) -> his stance depth for that pitcher-hand filter."""
    out = []
    for k, g in enumerate(groups):
        sc = model_scene(sw, g.get("types"), g.get("hand", "All"), g.get("count", "All"), g.get("zone"), stand,
                         depth_for(g.get("hand", "All"), g), stance_x, all_types)
        if sc is not None:
            sc["name"], sc["color"], sc["gid"] = group_label(g, stand), GROUP_COLORS[k % MAX_GROUPS], k
            sc["series"] = series(sw, g, stand)
            sc["norm"] = norm_for(nm or {}, g.get("types"), g.get("zone"), all_types)
            if not any(o is not None for o in out):          # the simulator layers, for the first group shown
                try:
                    from core import swing_sim
                    sc["layers"] = swing_sim.layers(sc, sw, stand)
                except Exception:
                    sc["layers"] = None
        out.append(sc)
    return out



# ----------------------------------------------------------------------------- graphs: his swings + MLB norms
SERIES = ["pf", "bat_speed", "attack_angle", "attack_direction", "swing_path_tilt", "gap_v", "gap_h", "vba", "hba",
          "match_v", "match_h"]
_norms: dict = {}


def norms(season: int) -> dict:
    """MLB fast-swing (75+) ranges from pipeline.bat_norms ({} until it's been run)."""
    from pathlib import Path
    import json
    if season not in _norms:
        f = Path(__file__).resolve().parents[1] / "data" / "calibration" / f"bat_norms_{season}.json"
        try:
            _norms[season] = json.loads(f.read_text())
        except Exception:
            _norms[season] = {}
    return _norms[season]


def norm_for(nm: dict, types, zone, all_types=None) -> dict:
    """The matching MLB range: the group's pitch group (when every type picked is one group) x its zone (when it's
    one zone); falls back to all pitches / all zones. Returns {key, depth, contact, bins, fallback}."""
    if not nm or not nm.get("keys"):
        return {}
    g = "All"
    if types:
        from core import hit_metrics as hm
        gs = set(hm.pitch_group(pd.DataFrame({"pitch_type": list(types)})).dropna())
        g = gs.pop() if len(gs) == 1 else "All"
    zs = zones_of(zone)
    z = str(zs[0]) if zs and len(zs) == 1 else "All"
    for key in (f"{g}|{z}", f"{g}|All", f"All|{z}", "All|All"):
        if key in nm["keys"]:
            k = nm["keys"][key]
            return {"key": key, "bins": nm["bins"], "depth": k["depth"], "contact": k["contact"],
                    "fast": nm.get("fast_mph", 75), "all": nm["keys"].get("All|All", {}).get("depth"),
                    "tgrid": nm.get("tgrid"), "time": k.get("time"),
                    "time_all": nm["keys"].get("All|All", {}).get("time")}
    return {}


def series(sw: pd.DataFrame, g: dict, stand: str) -> dict:
    """His measured contact swings for a swing group: one row per swing [pf, bat speed, AA, AD, tilt, gap_v, gap_h]."""
    s = sw[pick(sw, g.get("types"), g.get("hand", "All"), g.get("count", "All"), g.get("zone"), stand)]
    s = s[s["pf"].notna()]
    rows = s[SERIES].round(2).to_numpy()
    return {"cols": SERIES, "pts": [[None if pd.isna(v) else float(v) for v in r] for r in rows],
            "n_fast": int((s["bat_speed"] >= 75).sum())}



def path_angles(sc: dict) -> pd.DataFrame:
    """Along the drawn path (sweet spot): t (ms), measured/inferred flag, depth y (in front of the plate), the path's
    vertical + horizontal angles vs the pitch (gap_v / gap_h, batter-oriented: + = steeper / oppo), and the bat's
    horizontal angle (hba, + = barrel ahead of the hands) -- the same numbers the 3D view colours by."""
    a = np.asarray([q[:5] for q in sc["arc"]], float)
    O = np.asarray(sc["O"], float)
    n = len(a)
    prv, nxt = a[np.maximum(np.arange(n) - 1, 0)], a[np.minimum(np.arange(n) + 1, n - 1)]
    d = nxt[:, :3] - prv[:, :3]
    pv = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
    ph = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
    side = -1.0 if sc.get("side") == "L" else 1.0
    wrap = lambda v: (v + 540) % 360 - 180
    h = O - a[:, :3]
    return pd.DataFrame({"t": a[:, 4] * 1000, "flag": a[:, 3], "y": a[:, 1],
                         "gap_v": pv + sc["read"]["vaa"], "gap_h": wrap(side * (ph + sc["read"]["haa"])),
                         "hba": -np.degrees(np.arctan2(h[:, 1], np.abs(h[:, 0])))})


def on_plane(sc: dict, mode: str = "gap_v", tol: float = 5.0) -> dict:
    """The stretch of path around contact within `tol` degrees of the pitch: ms, inches of path, where it starts /
    ends vs the plate, and how far into it he makes contact (0 = start, 1 = end). After contact = inferred."""
    p = path_angles(sc)
    i = int((p["t"] >= 0).idxmax()) if (p["t"] >= 0).any() else len(p) - 1
    if abs(p[mode].iloc[i]) > tol:
        return {"on": False, "gap": float(p[mode].iloc[i])}
    s_, e = i, i
    while s_ > 0 and abs(p[mode].iloc[s_ - 1]) <= tol:
        s_ -= 1
    while e < len(p) - 1 and abs(p[mode].iloc[e + 1]) <= tol:
        e += 1
    pts = np.asarray([q[:3] for q in sc["arc"]], float)[s_:e + 1]
    t0, t1 = float(p["t"].iloc[s_]), float(p["t"].iloc[e])
    return {"on": True, "ms": t1 - t0, "inches": float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()),
            "y0": float(p["y"].iloc[s_]), "y1": float(p["y"].iloc[e]), "before": -t0, "after": t1,
            "pct": -t0 / (t1 - t0) if t1 > t0 else 0.5}



def self_time(sw: pd.DataFrame, stand: str, stance_x: dict | None = None) -> dict:
    """His own time ranges (all his swings with a contact depth), for 'vs Himself' in the time view:
    {'fast': band, 'comp': band, 'tgrid': [...]} -- fast = his 75+ swings (empty if he has none)."""
    s = sw[(sw["stand"] == stand) & sw["pf"].notna()]
    if s.empty:
        return {}
    cv = time_curves(s["attack_angle"], s["attack_direction"], s["swing_path_tilt"], s["bat_speed"], s["swing_length"],
                     s["stand"], s["pf"], s["cy"] - s["pf"], s["cx"], s["px_in"], s["pz_in"], stance_x)
    fast = (s["bat_speed"] >= 75).to_numpy()
    return {"tgrid": TGRID.tolist(), "comp": time_band(cv, 5),
            "fast": time_band({k: v[fast] for k, v in cv.items()}, 5) if fast.sum() >= 5 else None}
