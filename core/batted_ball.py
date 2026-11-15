"""What kind of contact it was -- a label, not an rpm -- from measured numbers only.

Two gaps between where the ball went and where the bat was going (both measured by Statcast):
  vertical:   launch angle - attack angle   (ball leaving above the bat's path = hit under it = backspin;
                                              below = over the top = topspin)
  horizontal: spray direction - attack direction, balls in the air (ball off the barrel's line = sidespin:
              pull side of the aim = hook, opposite side = slice)
Both are centred on how SQUARED-UP balls behave (EV >= 85% of the most that bat speed + pitch speed allow; 2026,
World Series hitters' seasons: launch angle sits 3.8 deg above the attack angle, spray 9.7 deg to the pull side of it).
Balls in play only -- fouls and whiffs get no contact grade.
Squared-up = MLB's definition (EV vs 1.23 x bat speed + 0.23 x pitch speed); smash factor = Savant's
(1 + (EV - bat speed) / (pitch speed + bat speed)).
"""
from __future__ import annotations

import math

V_CENTER, H_CENTER = 3.8, -9.7      # degrees; recalibrate from a full league season when it's pulled
V_CUT, H_CUT = 10.0, 20.0

LABELS = {"Optimal": "squared up and in the air at a good angle -- the ball traveled",
          "Backspin": "hit under the ball -- it left well above the bat's path",
          "Topspin": "hit over the top -- it left well below the bat's path",
          "Hook": "off the barrel's line toward the pull side -- sidespin hooking it",
          "Slice": "off the barrel's line toward the opposite field -- sidespin slicing it",
          "Knuckle": "near the middle of the ball but not squared up -- little spin"}


def _f(row, k):
    try:
        v = float(row.get(k))
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def contact_quality(row) -> dict:
    ev, bs, vp = _f(row, "launch_speed"), _f(row, "bat_speed"), _f(row, "release_speed")
    out = {}
    if None not in (ev, bs, vp) and bs > 0:
        mx = 1.23 * bs + 0.23 * vp
        out["squared_up_pct"] = round(100 * ev / mx, 1)
        out["squared_up"] = ev >= 0.8 * mx
        out["smash"] = round(1 + (ev - bs) / (vp + bs), 2)
    return out


def spin_label(row) -> dict | None:
    """{label, why, v_gap, h_gap} for a BALL IN PLAY with exit data, else None (fouls / whiffs get no grade)."""
    if "description" in row and str(row.get("description")) != "hit_into_play":
        return None
    la, aa, ad = _f(row, "launch_angle"), _f(row, "attack_angle"), _f(row, "attack_direction")
    if la is None or aa is None or _f(row, "launch_speed") is None:
        return None
    q = contact_quality(row)
    v_gap = la - aa - V_CENTER
    h_gap = None
    hx, hy = _f(row, "hc_x"), _f(row, "hc_y")
    if ad is not None and hx is not None and hy is not None and la >= 10:
        ang = math.degrees(math.atan2(hx - 125.42, 198.27 - hy))
        spray = ang if str(row.get("stand", "R")) == "R" else -ang       # negative = pull side
        h_gap = spray - ad - H_CENTER
    barrel = _f(row, "launch_speed_angle") == 6
    if barrel or (q.get("squared_up_pct", 0) >= 90 and 8 <= la <= 32):
        lab = "Optimal"
    elif h_gap is not None and abs(h_gap) >= H_CUT:
        lab = "Hook" if h_gap < 0 else "Slice"
    elif v_gap >= V_CUT:
        lab = "Backspin"
    elif v_gap <= -V_CUT:
        lab = "Topspin"
    elif abs(v_gap) < 4 and (h_gap is None or abs(h_gap) < 10):
        lab = "Knuckle"
    else:
        lab = "Backspin" if v_gap > 0 else "Topspin"
    return {"label": lab, "why": LABELS[lab], "v_gap": round(v_gap, 1),
            "h_gap": None if h_gap is None else round(h_gap, 1), **q}
