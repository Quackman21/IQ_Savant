"""3D swing scene for the AAR: the measured pitch + the bat's sweet-spot path, for swings that made contact.

Frame (inches): x = catcher's right (toward the 1B side), y = toward the pitcher measured from the FRONT edge of the
plate (0 = front edge, -17 = back point), z = up. Times in seconds from contact (negative = before).

  * PITCH (measured): Statcast's release velocity + acceleration (given at y = 50 ft), anchored to its plate crossing,
    from release to the contact point.
  * CONTACT: MLB's definition -- Statcast's contact (inches in front of his center of mass) minus his stance depth
    (Savant's board; p['depth']) -> how far in front of the plate; side to side + height = where the ball was there.
  * BAT: the last 40 ms before contact is what Statcast measures -- the sweet spot arrives at the ball at the measured
    attack angle + attack direction, in a plane tilted at the measured swing-path tilt (leaning toward his measured
    stance position, centre above the ball). The bat lies across that path in the plane, so its angle is known
    everywhere on it: BAT PARALLEL (barrel level with the hands) is inferred from those measured angles and the path
    is drawn back to it, styled as inferred; the follow-through continues the plane a quarter turn past contact,
    also inferred. No body.
  * BALL OFF THE BAT (balls in play only): measured launch angle + exit velo, aimed at where it was fielded. Statcast
    doesn't track where fouls go, so fouls get no flight.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FRONT_FT = 17 / 12
CONTACT_DESC = {"foul", "foul_tip", "hit_into_play"}


def _num(row, k):
    try:
        v = float(row.get(k))
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _t_at(d, y_ft):
    """Time after the y = 50 ft reference when the ball reaches depth y_ft (Statcast: y from the back point)."""
    a, b, c = 0.5 * d["ay"], d["vy0"], 50 - y_ft
    return (-b - math.sqrt(max(b * b - 4 * a * c, 0))) / (2 * a)


def ball_at(d, y_ft):
    """Ball (x, z) in feet at depth y_ft, anchored so it crosses the front of the plate at (plate_x, plate_z)."""
    tf = _t_at(d, FRONT_FT)
    x50 = d["plate_x"] - (d["vx0"] * tf + 0.5 * d["ax"] * tf ** 2)
    z50 = d["plate_z"] - (d["vz0"] * tf + 0.5 * d["az"] * tf ** 2)
    t = _t_at(d, y_ft)
    return x50 + d["vx0"] * t + 0.5 * d["ax"] * t ** 2, z50 + d["vz0"] * t + 0.5 * d["az"] * t ** 2


def _unit(v):
    return v / np.linalg.norm(v)


def scene(row, depth: float | None, stance_x: float | None = None) -> dict | None:
    """Everything the 3D view draws for one contact swing, or None if anything measured is missing."""
    need = ("vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z", "bat_speed", "attack_angle",
            "attack_direction", "swing_path_tilt", "intercept_ball_minus_batter_pos_y_inches")
    d = {k: _num(row, k) for k in need}
    if any(v is None for v in d.values()) or depth is None or str(row.get("description")) not in CONTACT_DESC:
        return None
    side = str(row.get("stand", "R"))
    pf = d["intercept_ball_minus_batter_pos_y_inches"] - depth            # inches in front of the plate (MLB)
    yc_ft = FRONT_FT + pf / 12
    bx, bz = ball_at(d, yc_ft)
    C = np.array([bx * 12, pf, bz * 12])
    t0 = _t_at(d, yc_ft)

    # pitch: release -> contact
    y_rel = _num(row, "release_pos_y") or 54.0
    pitch = []
    for y in np.linspace(y_rel, yc_ft, 90):
        x, z = ball_at(d, y)
        pitch.append([round(x * 12, 2), round((y - FRONT_FT) * 12, 2), round(z * 12, 2),
                      round(_t_at(d, y) - t0, 5)])
    # the ball's angles arriving at contact (degrees; vertical negative = descending)
    t = t0
    vx, vy, vz = d["vx0"] + d["ax"] * t, d["vy0"] + d["ay"] * t, d["vz0"] + d["az"] * t
    vaa = math.degrees(math.atan2(vz, -vy))
    haa = math.degrees(math.atan2(vx, -vy))

    # bat: sweet-spot arc through C
    ad, aa = d["attack_direction"], d["attack_angle"]
    h = math.radians(ad if side == "R" else -ad)                            # + toward the 1B side
    a = math.radians(aa)
    dvec = np.array([math.sin(h) * math.cos(a), math.cos(h) * math.cos(a), math.sin(a)])
    off = (stance_x if stance_x is not None else 27.0) + 8.5               # his CoM off the plate's middle (inches)
    H = np.array([-off if side == "R" else off, -depth, C[2]])             # where he stands (measured), for the lean
    toward = H - C
    toward = toward - toward.dot(dvec) * dvec
    flat = _unit(np.array([toward[0], toward[1], 0.0]))
    T = math.radians(d["swing_path_tilt"])
    best = None
    for th in np.radians(np.linspace(0, 89.9, 900)):                       # centre above the ball (hands)
        w = math.cos(th) * flat + math.sin(th) * np.array([0, 0, 1.0])
        w = _unit(w - w.dot(dvec) * dvec)
        n = np.cross(dvec, w)
        err = abs(math.acos(min(1.0, abs(n[2]) / np.linalg.norm(n))) - T)
        if best is None or err < best[0]:
            best = (err, w)
    w = best[1]
    cx, cy = _num(row, "intercept_ball_minus_batter_pos_x_inches") or 30.0, pf + depth
    R = float(np.clip(math.hypot(cx, cy), 24, 48))                         # sweet spot's radius about the hands
    O = C + R * w
    v = d["bat_speed"] * 17.6                                              # mph -> in/s
    # The bat lies across the sweet spot's path, in the swing plane, pointing back to the hands (the arc's centre):
    #   bat direction at angle phi along the arc = w*cos(phi) - d*sin(phi)   (phi = 0 at contact, negative = before)
    # -> its vertical bat angle anywhere on the path, and BAT PARALLEL = where that direction is level (barrel even
    # with the hands): w_z*cos(phi) - d_z*sin(phi) = 0  ->  phi = -pi/2 - atan2(d_z, w_z).
    vba = -math.degrees(math.asin(max(-1.0, min(1.0, float(w[2])))))   # at contact; negative = barrel below hands
    phi_bp = -math.pi / 2 - math.atan2(float(dvec[2]), float(w[2]))
    s_bp = R * phi_bp                                                    # inches along the path before contact
    swing_len = (_num(row, "swing_length") or 0) * 12
    bp_ok = swing_len <= 0 or -s_bp <= swing_len                         # within his measured swing length
    measured = min(v * 0.040, R * 1.2)
    start = min(-measured, s_bp) if bp_ok else -measured
    follow = R * math.pi / 2                                             # follow-through: a quarter turn, inferred
    arc = []
    for s_ in sorted(set(np.linspace(start, 0, 90).tolist() + [-measured] + np.linspace(0, follow, 30)[1:].tolist())):
        phi = s_ / R
        P = O + R * (-w * math.cos(phi) + dvec * math.sin(phi))
        # 1 = Statcast's measured last 40 ms; 2 = inferred from the measured plane + angles (bat parallel -> there)
        arc.append([round(float(P[0]), 2), round(float(P[1]), 2), round(float(P[2]), 2),
                    1 if -measured - 1e-9 <= s_ <= 1e-9 else 2, round(s_ / v, 5)])
    bp = None
    if bp_ok:
        P = O + R * (-w * math.cos(phi_bp) + dvec * math.sin(phi_bp))
        bp = {"at": [round(float(x), 2) for x in P], "inches_before": round(-s_bp, 1), "t": round(s_bp / v, 5)}

    # ball off the bat (balls in play, measured)
    exit_ = None
    ev, la = _num(row, "launch_speed"), _num(row, "launch_angle")
    hx, hy = _num(row, "hc_x"), _num(row, "hc_y")
    if str(row.get("description")) == "hit_into_play" and None not in (ev, la, hx, hy):
        X, Y = (hx - 125.42) * 30.0, (198.27 - hy) * 30.0 - 17.0
        ph = math.atan2(X - C[0], Y - C[1])
        u = np.array([math.sin(ph) * math.cos(math.radians(la)), math.cos(ph) * math.cos(math.radians(la)),
                      math.sin(math.radians(la))])
        exit_ = []
        for tt in np.linspace(0, 0.25, 50):
            P = C + u * ev * 17.6 * tt + np.array([0, 0, -0.5 * 386.1 * tt * tt])
            if P[2] < 0:
                break
            exit_.append([round(float(P[0]), 2), round(float(P[1]), 2), round(float(P[2]), 2), round(float(tt), 4)])
    from core import batted_ball
    lab = batted_ball.spin_label(row) or {}
    q = batted_ball.contact_quality(row)
    return {
        "side": side, "contact": [round(float(c), 2) for c in C], "O": [round(float(o), 2) for o in O],
        "pitch": pitch, "arc": arc, "exit": exit_, "bp": bp,
        "sz": [round((_num(row, "sz_bot") or 1.6) * 12, 1), round((_num(row, "sz_top") or 3.4) * 12, 1)],
        "aim": [round(float(x), 3) for x in dvec],
        "read": {"bat_speed": d["bat_speed"], "swing_length": _num(row, "swing_length"), "attack_angle": aa,
                 "attack_direction": ad, "tilt": d["swing_path_tilt"], "front": round(pf, 1),
                 "vba": round(vba, 1), "bp_in": bp["inches_before"] if bp else None,
                 "side_in": round(float(C[0]), 1), "height": round(float(C[2]), 1), "vaa": round(vaa, 1),
                 "haa": round(haa, 1), "plane_gap": round(aa + vaa, 1), "ev": ev, "la": la,
                 "velo": _num(row, "release_speed"), "spin": lab.get("label"), "spin_why": lab.get("why"),
                 "squared_up_pct": q.get("squared_up_pct"), "smash": q.get("smash")},
    }


def scenes(sc: pd.DataFrame, p: pd.DataFrame, stance_x: dict | None = None) -> dict:
    """{pitch number n (the AAR's row #): scene} for every contact swing; sc = the game's Statcast rows, p = the
    AAR's pitch table (same order -- core.aar.pitches sorts by at-bat, pitch), stance_x = {'R'/'L': inches off the
    plate} from Savant's board."""
    if sc is None or sc.empty or p is None or p.empty or "depth" not in p:
        return {}
    d = sc.copy()
    d["_ab"] = pd.to_numeric(d["at_bat_number"], errors="coerce")
    d["_pn"] = pd.to_numeric(d["pitch_number"], errors="coerce")
    d = d.sort_values(["_ab", "_pn"]).reset_index(drop=True)
    out = {}
    for (i, r), (_, q) in zip(d.iterrows(), p.iterrows()):
        dep = q.get("depth")
        try:
            s = scene(r, float(dep) if pd.notna(dep) else None, (stance_x or {}).get(str(r.get("stand"))))
        except Exception:
            s = None
        if s:
            s["read"].update({"ptype": str(q.get("ptype", "")), "count": str(q.get("count", "")),
                              "result": str(q.get("play") or q.get("result") or ""), "video": str(q.get("video", ""))})
            out[int(q["n"])] = s
    return out


def stance_offsets(season: int, pid: int) -> dict:
    """{'R'/'L': inches his center of mass stands off the plate's inside edge} from Savant's board (measured)."""
    from pathlib import Path
    f = Path(__file__).resolve().parents[1] / "data" / f"batting_stance_{season}.csv"
    if not f.exists():
        return {}
    try:
        df = pd.read_csv(f)
        df = df[df["id"] == pid]
        side = df["side"] if "side" in df else df.get("bat_side", "R")
        return {str(sd): float(x) for sd, x in zip(side, df["avg_batter_x_position"]) if pd.notna(x)}
    except Exception:
        return {}
