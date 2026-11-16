"""Bat Path tab: point of contact vs the plate for a hitter's swings (same method as the AAR -- MLB's definition).

  front (pf)  = Statcast's contact in front of his center of mass - his stance depth (Savant's board: playoff ->
                vs this pitcher's hand -> season), inches in front of the plate's front edge
  side  (pxc) = where the ball was at that depth, inches from the middle of the plate, catcher's view
Hitters not on the stance board fall back to the AAR's calibrated estimate (flagged).
"""
from __future__ import annotations

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


def swings(sc: pd.DataFrame) -> pd.DataFrame:
    """His competitive swings with what the view needs: group, zone, hand, the measured swing + contact."""
    from core import hit_metrics as hm
    if sc is None or sc.empty or "bat_speed" not in sc:
        return pd.DataFrame()
    d = hm.prep(sc)
    num = lambda c: pd.to_numeric(sc[c], errors="coerce") if c in sc else pd.Series(np.nan, index=sc.index)
    made = aar.made_contact(sc)
    out = pd.DataFrame({
        "group": hm.pitch_group(sc), "zone": num("zone"), "hand": sc["p_throws"].astype(str),
        "stand": sc["stand"].astype(str), "bat_speed": d["bat_speed"], "swing_length": num("swing_length"),
        "attack_angle": num("attack_angle"), "attack_direction": num("attack_direction"),
        "swing_path_tilt": num("swing_path_tilt"), "velo": num("release_speed"),
        "cy": num(aar.CONTACT_Y).where(made), "cx": num(aar.CONTACT_X).where(made),
        "sz_top": num("sz_top"), "sz_bot": num("sz_bot")}, index=sc.index)
    return out[out["bat_speed"].notna()]


def profile(sw: pd.DataFrame, group: str, zone: int, hand: str, stand: str | None = None) -> dict:
    """His average swing for group x zone x hand, widening the sample until there are enough swings."""
    if sw.empty:
        return {}
    s = sw if stand is None else sw[sw["stand"] == stand]
    levels = [("this pitch group, zone + pitcher hand", (s.group == group) & (s.zone == zone) & (s.hand == hand)),
              ("this pitch group + zone (both hands)", (s.group == group) & (s.zone == zone)),
              ("this zone (all pitches)", s.zone == zone),
              ("this pitch group (all zones)", s.group == group),
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


def velo_for(sw: pd.DataFrame, group: str, hand: str) -> tuple[float, str]:
    v = sw[(sw.group == group) & (sw.hand == hand)]["velo"].dropna() if not sw.empty else pd.Series(dtype=float)
    if len(v) >= 20:
        return float(v.mean()), f"average {hand}HP {group.lower()} he swung at"
    return GROUPS[group][1][hand], f"typical {hand}HP {group.lower()}"


def model_scene(sw: pd.DataFrame, group: str, zone: int, hand: str, depth: float | None,
                stance_x: float | None, stand: str) -> dict | None:
    """The 3D scene (core.swing3d format) for his average swing vs the modeled pitch, plus what went into it."""
    from core import swing3d
    pr = profile(sw, group, zone, hand, stand)
    if not pr or "cy" not in pr or depth is None:
        return None
    ptype, _, ivb, hb, desc = GROUPS[group]
    velo, vsrc = velo_for(sw[sw["stand"] == stand], group, hand)
    me = sw[sw["stand"] == stand]
    top = float(me["sz_top"].mean()) if me["sz_top"].notna().any() else 3.4
    bot = float(me["sz_bot"].mean()) if me["sz_bot"].notna().any() else 1.6
    px, pz = zone_target(zone, bot, top)
    row = {**synth_pitch(velo, ivb, hb, hand, px, pz), "description": "foul", "stand": stand, "p_throws": hand,
           "bat_speed": pr["bat_speed"], "swing_length": pr["swing_length"], "attack_angle": pr["attack_angle"],
           "attack_direction": pr["attack_direction"], "swing_path_tilt": pr["swing_path_tilt"],
           aar.CONTACT_Y: pr["cy"], aar.CONTACT_X: pr["cx"], "sz_top": top, "sz_bot": bot}
    sc = swing3d.scene(row, depth, stance_x)
    if sc is None:
        return None
    sc["read"].update({"ptype": ptype, "count": "", "result": "", "video": ""})
    sc["model"] = {**pr, "velo": velo, "velo_src": vsrc, "ivb": ivb, "hb": hb, "shape": desc, "ptype": ptype,
                   "zone": zone, "hand": hand, "group": group, "depth": depth}
    return sc
