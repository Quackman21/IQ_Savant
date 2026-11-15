"""Swing map prototype (standalone -- NOT part of the site yet).

    python prototypes/swing_map/build.py GAME.csv SEASON.csv.gz [out.html]

GAME.csv     every pitch of one game (pybaseball statcast_single_game)
SEASON.csv   the season's swings of everyone who batted in it (for each hitter's hip position)

For every swing it draws, around home plate (front-center of the plate = 0, 0, 0; inches; x = toward the 1B side,
y = toward the pitcher, z = up):
  * the PITCH's flight (measured: Statcast's release velocity + acceleration, through its plate crossing)
  * the CONTACT POINT: Statcast's intercept (from his hips) + his hip position at contact (fit from his season's
    swings: contact happens where the ball is -- the ball's sideways drift vs where he met it pins his hips down)
  * the SWEET-SPOT PATH: an arc in his swing plane (tilt measured), moving at the measured attack angle +
    attack direction at contact. The last 40 ms (what Statcast measures the plane over) is solid = measured
    direction; the rest of the arc back to the start of the swing (swing length) and the follow-through are dashed
    = modeled shape.
Conventions (checked on the data, not assumed): intercept Y = out in front of the hips (avg ~30"), intercept X =
toward the plate (r .93 / -.91 with plate_x for R / L), attack direction negative = pull side.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FRONT_FT = 17 / 12
CY, CX = "intercept_ball_minus_batter_pos_y_inches", "intercept_ball_minus_batter_pos_x_inches"
CONTACT = {"foul", "foul_tip", "hit_into_play"}


# ------------------------------------------------------------------ pitch physics (Statcast: ft, ft/s, ft/s^2)
def _t_at(d, y_ft):
    """Time after the y = 50 ft reference when the ball reaches depth y_ft."""
    a, b, c = 0.5 * d["ay"], d["vy0"], 50 - y_ft
    return (-b - np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * a)


def pitch_at(d, y_ft):
    """Ball (x, z) in feet at depth y_ft (Statcast frame: y from the back point of the plate)."""
    t_front = _t_at(d, FRONT_FT)
    x50 = d["plate_x"] - (d["vx0"] * t_front + 0.5 * d["ax"] * t_front ** 2)
    z50 = d["plate_z"] - (d["vz0"] * t_front + 0.5 * d["az"] * t_front ** 2)
    t = _t_at(d, y_ft)
    return x50 + d["vx0"] * t + 0.5 * d["ax"] * t ** 2, z50 + d["vz0"] * t + 0.5 * d["az"] * t ** 2


def drift_slope(d):
    t = _t_at(d, FRONT_FT)
    vy = d["vy0"] + d["ay"] * t
    return (d["vx0"] + d["ax"] * t) / vy


# ------------------------------------------------------------------ hip position at contact (per hitter, per side)
HALF_LIFE = 90.0     # 30 over-reacted (Frelick / Hicks pushed too deep in your re-review); 90 still tracks real moves


def _line(g, weighted=True):
    """Z = X0 + dL*[pitcher is a lefty] + slope * s, trimmed and (optionally) recency-weighted.
    F0 = -slope = hips vs the front of the plate.
    Why the lefty term: his hips sit in a slightly different side-to-side spot vs LHP than vs RHP, and the ball's
    sideways drift mostly flips with the pitcher's hand -- without it that difference leaked into the depth (it put
    Hicks / Palacios behind their own stance; Savant's leaderboard caught it)."""
    w = g["w"].to_numpy() if weighted else np.ones(len(g))
    sv, zv, Lv = g["s"].to_numpy(), g["Z"].to_numpy(), g["L"].to_numpy()
    use_L = min(Lv.sum(), (1 - Lv).sum()) >= 15
    def solve(k):
        cols = [np.ones(k.sum()), sv[k]] + ([Lv[k]] if use_L else [])
        X, z, wk = np.column_stack(cols), zv[k], w[k]
        ws = wk * ((wk.sum() ** 2 / (wk ** 2).sum()) / wk.sum())          # weights scaled to the effective n
        A = X.T @ (X * ws[:, None])
        b = np.linalg.solve(A, X.T @ (ws * z))
        r = z - X @ b
        n_eff = ws.sum()
        sig2 = float((ws * r * r).sum() / max(n_eff - X.shape[1], 1))
        return b, r, float(math.sqrt(sig2 * np.linalg.inv(A)[1, 1])), n_eff
    k = np.ones(len(g), bool)
    b, r, _, _ = solve(k)
    full_r = zv - (b[0] + b[1] * sv + (b[2] * Lv if use_L else 0))
    k = np.abs(full_r - np.median(full_r)) <= max(3 * np.median(np.abs(full_r - np.median(full_r))), 1.5)
    if k.sum() < 30:
        return None
    b, r, se, n_eff = solve(k)
    sR = float(sv[k][Lv[k] == 0].mean()) if (Lv[k] == 0).any() else 0.0
    sL = float(sv[k][Lv[k] == 1].mean()) if (Lv[k] == 1).any() else sR
    return {"F0": float(-b[1]), "X0": float(b[0]), "dL": float(b[2]) if use_L else 0.0, "sl": float(b[1]),
            "sR": sR, "sL": sL,
            "se": se, "n": int(k.sum()), "n_eff": float(n_eff), "rsd": float(r.std())}


# Savant's batting-stance leaderboard is a CHECK, never an input: where he STANDS (depth in the box, inches behind
# the front of the plate). His hips should end up a little forward of that by contact (stride + weight shift):
# behind the stance is impossible, a huge move forward is suspicious.
def stance_check(fits: dict, stance_csv: str | None) -> list:
    if not stance_csv or not Path(stance_csv).exists():
        return []
    st = pd.read_csv(stance_csv)
    rows = []
    for (b, side), f in fits.items():
        m = st[(st["id"] == b) & (st["side"] == side)]
        if m.empty:
            continue
        depth = float(m["avg_batter_y_position"].iloc[0])
        fwd = depth - (-f["F0"])                          # inches his hips moved forward from stance to contact
        f["stance"], f["fwd"] = depth, fwd
        f["stance_grade"] = "r" if fwd < -2 else ("y" if fwd < 0 or fwd > 12 else "g")
        rows.append((b, side, -f["F0"], depth, fwd, f["stance_grade"]))
    return rows


def shrink_to_group(fits: dict, group: dict | None = None) -> None:
    """Pull each hitter's depth toward the group average, more when his own fit is noisy. Savant's at-contact foot
    positions (9 hitters) put everyone's hips in a narrow 13-19" band; our raw fits spread far wider than that, so
    most of the hitter-to-hitter spread was noise. Organic: the average + spread come from our own fits, not Savant."""
    ref = group or fits                                              # league-wide when available
    if len(ref) < 5:
        return
    F = np.array([-f["F0"] for f in ref.values()])
    se = np.array([f["se"] for f in ref.values()])
    mu = float(F.mean())
    tau2 = max(float(F.var(ddof=1) - (se ** 2).mean()), 1.0)      # real hitter-to-hitter spread
    for f in fits.values():
        raw = -f["F0"]
        w = tau2 / (tau2 + f["se"] ** 2)
        est = mu + w * (raw - mu)
        # keep the side-to-side hip spot consistent with the new depth (re-centre at his average drift per hand)
        d = f["sl"] - est
        f["X0"] += d * f["sR"]
        f["dL"] += d * (f["sL"] - f["sR"])
        f.update({"raw": raw, "F0": -est, "sl": est, "se": float(math.sqrt(1 / (1 / tau2 + 1 / f["se"] ** 2))),
                  "group": mu})


def hip_fits(season: pd.DataFrame, asof: str | None = None) -> dict:
    c = season[season["description"].isin(CONTACT)].dropna(subset=[CX, CY, "vx0", "plate_x"]).copy()
    c["s"] = drift_slope(c)
    c["side"] = np.where(c["stand"].eq("R"), 1, -1)
    c["Z"] = c["plate_x"] * 12 + c["s"] * c[CY] - c["side"] * c[CX]
    c["L"] = c["p_throws"].eq("L").astype(float)
    # Hitters MOVE in the box during a season (your review flagged Chourio; his monthly fits run 7.5" -> 17.4" -> 9.8"),
    # so recent swings count more: weight halves every HALF_LIFE days before the game.
    c["days"] = (pd.Timestamp(asof) - pd.to_datetime(c["game_date"])).dt.days.clip(lower=0) if asof else 0
    c["w"] = 0.5 ** (c["days"] / HALF_LIFE)
    out = {}
    for (b, st), g in c.groupby(["batter", "stand"]):
        if len(g) < 40:
            continue
        f = _line(g)
        if f is None:
            continue
        # how much his spot moves month to month beyond what noise explains -> added to the uncertainty
        mon = [m for _, x in g.groupby(pd.to_datetime(g["game_date"]).dt.to_period("M")) if len(x) >= 40
               for m in [_line(x, weighted=False)] if m]
        if len(mon) >= 3:
            extra = max(np.var([m["F0"] for m in mon], ddof=1) - np.mean([m["se"] ** 2 for m in mon]), 0.0)
            f["drift"] = float(math.sqrt(extra))
        else:
            f["drift"] = 0.0
        f["se_fit"] = f["se"]
        f["se"] = float(math.sqrt(f["se_fit"] ** 2 + (f["drift"] ** 2) * 0.5))   # recency weighting removes ~half
        # cross-check 1: bat direction vs depth (deeper contact = barrel still pointing oppo). Uses attack direction,
        # which the placement never touches.
        h = g.dropna(subset=["attack_direction"])
        if len(h) >= 30:
            bb, aa = np.polyfit(h["attack_direction"], h[CY], 1)
            f["ad"] = {"a": float(aa), "b": float(bb), "sd": float((h[CY] - (aa + bb * h["attack_direction"])).std())}
        # cross-check 2: where the ball went vs where the bat was pointing (balls in play)
        q = h.dropna(subset=["hc_x", "hc_y"])
        q = q[q["description"].eq("hit_into_play") & (spray_ad(q, st).abs() <= 50)]
        if len(q) >= 20:
            off = spray_ad(q, st) - q["attack_direction"]
            f["spray"] = {"mean": float(off.mean()), "sd": float(off.std())}
        out[(int(b), st)] = f
    return out


def spray_ad(d, stand):
    """Spray angle in attack-direction terms: degrees, negative = pull side."""
    ang = np.degrees(np.arctan2(d["hc_x"] - 125.42, 198.27 - d["hc_y"]))      # + = toward right field
    return ang if stand == "R" else -ang


def hip_x(hip, row):
    """His hips' side-to-side spot for this pitcher's hand."""
    return hip["X0"] + (hip["dL"] if row.get("p_throws") == "L" else 0.0)


def grade(z, ok=1.0, warn=2.0):
    a = abs(z)
    return "g" if a <= ok else ("y" if a <= warn else "r")


def fit_checks(row, hip):
    """How much to trust this swing's placement. Each check returns (label, shown value, grade g/y/r, note)."""
    out = []
    se = hip["se"]
    # shared by every swing he takes -> shown, but NOT part of this swing's dot (it painted all of Bauers yellow)
    out.append(("Hip position (him, all swings)", f'{-hip["F0"]:.1f}" ± {se:.1f}" ({hip["n"]} swings; his swings alone '
                f'{hip.get("raw", -hip["F0"]):.1f}", group avg {hip.get("group", -hip["F0"]):.1f}")',
                "i", "his swings, pulled toward the group average when they're noisy"))
    if "stance" in hip:
        note = {"g": "model agrees: hips end up a little forward of where he stands",
                "y": "check: hips barely move or move a lot from his stance",
                "r": "model disagrees: hips can't end up behind where he stands"}[hip["stance_grade"]]
        out.append(("Savant check (stance)", f'stands {hip["stance"]:.1f}" · model hips {-hip["F0"]:.1f}" '
                    f'· moved {hip["fwd"]:+.1f}"', hip["stance_grade"], note))
    s_ = float(drift_slope(row))
    side = 1 if row["stand"] == "R" else -1
    Z = row["plate_x"] * 12 + s_ * row[CY] - side * row[CX]
    res = Z - (hip_x(hip, row) + hip["sl"] * s_)
    out.append(("This swing vs his fit", f'{res:+.1f}"', grade(res, 2.0, 4.0),
                "where the ball was vs where his hips + Statcast's intercept say he met it"))
    if "ad" in hip:
        pred = hip["ad"]["a"] + hip["ad"]["b"] * row["attack_direction"]
        diff = row[CY] - pred
        z = diff / hip["ad"]["sd"]
        out.append(("Depth vs bat direction", f'{diff:+.1f}" ({z:+.1f} sd)', "s" if abs(z) > 2 else "i",
                    "unusual for him (late / early / check swing) -- not a placement problem" if abs(z) > 2
                    else "normal for him"))
    sp = None if pd.isna(row.get("hc_x")) else float(spray_ad(row, row["stand"]))
    if "spray" in hip and row["description"] == "hit_into_play" and sp is not None and abs(sp) <= 50:
        off = sp - row["attack_direction"] - hip["spray"]["mean"]
        z = off / hip["spray"]["sd"]
        out.append(("Ball direction vs bat direction", f'{off:+.0f}° ({z:+.1f} sd)', "s" if abs(z) > 2 else "i",
                    "ball went well off the barrel's line (hook / slice / off the end), or was fielded far from where it "
                    "was hit" if abs(z) > 2 else "normal for him"))
    if row["description"] not in CONTACT and not pd.isna(row.get("miss_distance")):
        out.append(("Whiff", f'missed by {row["miss_distance"]:.1f}"', "i",
                    "drawn through the ball's spot; the real barrel passed this far away"))
    return out


# ------------------------------------------------------------------ swing plane + sweet-spot arc
def unit(v):
    return v / np.linalg.norm(v)


def swing_geometry(row, hip, side):
    """Contact point, bat direction and the sweet-spot arc in plate inches (origin = front-center of the plate)."""
    yc = hip["F0"] + row[CY]                                         # inches in front of the front edge
    bx, bz = pitch_at(row, FRONT_FT + yc / 12)
    C = np.array([bx * 12, yc, bz * 12])                            # where the ball is = where he met it
    H = np.array([hip_x(hip, row), hip["F0"], C[2]])                      # his hips, at contact height for the arc center
    ad = row["attack_direction"]
    h = math.radians(ad if side == "R" else -ad)                    # field angle, + toward the 1B side
    a = math.radians(row["attack_angle"])
    d = np.array([math.sin(h) * math.cos(a), math.cos(h) * math.cos(a), math.sin(a)])
    # plane through C containing d, tilted T to the ground, leaning so the arc's center sits toward his body
    T = math.radians(row["swing_path_tilt"])
    toward = H - C
    toward = toward - toward.dot(d) * d
    # Two planes fit any tilt (center tipped up or down) -- the swing pivots around his hands / lead shoulder, which
    # sit ABOVE the ball, so the arc's center is always raised: the barrel drops in from above and behind him
    flat = unit(np.array([toward[0], toward[1], 0.0]))
    best = None
    for th in np.radians(np.linspace(0, 89.9, 1800)):
        w = math.cos(th) * flat + math.sin(th) * np.array([0, 0, 1.0])
        w = unit(w - w.dot(d) * d)
        n = np.cross(d, w)
        tilt = math.acos(min(1.0, abs(n[2]) / np.linalg.norm(n)))
        err = abs(tilt - T)
        if best is None or err < best[0]:
            best = (err, w)
    w = best[1]
    R = float(np.hypot(row[CX], row[CY]))                           # sweet spot circles roughly about his body
    O = C + R * w
    v = row["bat_speed"] * 17.6                                      # mph -> inches per second
    measured = min(v * 0.040, R * 1.2)                               # last 40 ms = Statcast's plane window
    # Statcast's plane only describes the last 40 ms -- earlier the bat is in a different plane (coming down from
    # the hands), so the modeled extension stays short instead of looping the whole swing length through one plane
    back = min(row["swing_length"] * 12, measured + 30.0)
    fwd = min(v * 0.03, 50.0)                                        # ~30 ms of follow-through (modeled)
    pts = []
    for s_ in np.linspace(-back, fwd, 90):
        phi = s_ / R
        P = O + R * (-w * math.cos(phi) + d * math.sin(phi))
        pts.append([round(float(P[0]), 2), round(float(P[1]), 2), round(float(P[2]), 2),
                    "m" if -measured <= s_ <= 0 else "x", round(float(s_ / v), 5)])     # last = seconds from contact
    return C, d, pts, [round(float(x), 2) for x in O], h


def pitch_path(row, yc_in, stop_at_contact):
    """Ball flight in plate inches with seconds from contact (negative = before). Stops at the bat if he hit it."""
    t0 = _t_at(row, FRONT_FT + yc_in / 12)
    end = yc_in / 12 + FRONT_FT if stop_at_contact else FRONT_FT - 2.2
    out = []
    for y in np.linspace(FRONT_FT + 6, end, 60):                      # 6 ft in front of the plate -> contact / catcher
        x, z = pitch_at(row, y)
        out.append([round(float(x * 12), 2), round(float((y - FRONT_FT) * 12), 2), round(float(z * 12), 2),
                    round(float(_t_at(row, y) - t0), 5)])
    return out


def exit_path(row, C, feed_xy=None):
    """Ball off the bat: launch angle (measured) + direction from where it was fielded / landed. Statcast's file has no
    direction for fouls; MLB's game feed sometimes does. With neither, no flight is drawn -- your review showed the
    bat's direction is a bad guess (most fouls go straight back)."""
    if pd.isna(row.get("launch_speed")) or pd.isna(row.get("launch_angle")):
        return None, None
    # aim from the CONTACT POINT at the spot it was fielded (hit coordinates: home = back point of the plate,
    # 2.5 ft per unit) -- matters for balls down the line, where a few inches decide fair / foul on the drawing
    if not pd.isna(row.get("hc_x")):
        hx, hy, how = row["hc_x"], row["hc_y"], "measured"
    elif feed_xy:
        hx, hy, how = feed_xy[0], feed_xy[1], "measured (MLB feed)"
    else:
        return None, "untracked"
    X, Y = (hx - 125.42) * 30.0, (198.27 - hy) * 30.0 - 17.0                 # inches, plate frame
    phi = math.atan2(X - C[0], Y - C[1])
    la = math.radians(row["launch_angle"])
    u = np.array([math.sin(phi) * math.cos(la), math.cos(phi) * math.cos(la), math.sin(la)])
    v = row["launch_speed"] * 17.6
    pts = []
    for t in np.linspace(0, 0.08, 40):
        P = np.array(C) + u * v * t + np.array([0, 0, -0.5 * 386.1 * t * t])      # gravity, inches/s^2
        pts.append([round(float(P[0]), 2), round(float(P[1]), 2), round(float(P[2]), 2), round(float(t), 5)])
    return pts, how


# ------------------------------------------------------------------ build
def build(game_csv: str, season_csv: str, out_html: str, only: set | None = None,
          stance_csv: str | None = None, league_csv: str | None = None) -> None:
    g = pd.read_csv(game_csv, low_memory=False)
    s = pd.read_csv(season_csv, low_memory=False)
    gpk, gday = int(g["game_pk"].iloc[0]), str(g["game_date"].iloc[0])[:10]
    s = s[pd.to_numeric(s["game_pk"], errors="coerce").ne(gpk) & (s["game_date"].astype(str).str[:10] <= gday)]
    if "game_type" in s:                         # spring training tracking isn't reliable -- regular season + playoffs only
        s = s[~s["game_type"].astype(str).isin(["S", "E"])]
    fits = hip_fits(s, asof=gday)
    group = None
    if league_csv and Path(league_csv).exists():                    # every MLB hitter -> the group average + spread
        lg = pd.read_csv(league_csv, low_memory=False)
        lg = lg[lg["game_date"].astype(str).str[:10] <= gday]
        group = hip_fits(lg, asof=gday)
        print(f"group average from {len(group)} MLB hitters")
    shrink_to_group(fits, group)
    check = stance_check(fits, stance_csv)
    if check:
        print("Model vs Savant stance (hips at contact should be 0-12\" forward of where he stands):")
        for b, side, hips, depth, fwd, gr in sorted(check, key=lambda r: r[4]):
            flag = {"g": "ok", "y": "CHECK", "r": "WRONG (behind his stance)"}[gr]
            print(f"  {b} {side}: model {hips:5.1f}\"  stance {depth:5.1f}\"  moved forward {fwd:+5.1f}\"  {flag}")
        fw = [r[4] for r in check]
        print(f"  {sum(r[5] == 'g' for r in check)}/{len(check)} agree; average move forward {np.mean(fw):+.1f}\"")
    names, play = {}, {}                         # Statcast's player_name is the PITCHER -- look hitters up by id
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from core import mlb_api
        names = {int(k): v for k, v in mlb_api.get_people(sorted(int(x) for x in g["batter"].unique())).items()}
    except Exception:
        pass
    hits = {}
    try:                                         # each pitch's Savant video id (and any foul coordinates) from MLB's feed
        from core import aar, mlb_api
        feed = mlb_api.get_game_feed(int(g["game_pk"].iloc[0]))
        play = aar.play_ids(feed)
        for pl in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
            ab = int(pl.get("about", {}).get("atBatIndex", -1)) + 1
            for e in pl.get("playEvents", []):
                c = (e.get("hitData") or {}).get("coordinates") or {}
                if e.get("pitchNumber") is not None and c.get("coordX") is not None and c.get("coordY") is not None:
                    hits[(ab, int(e["pitchNumber"]))] = (float(c["coordX"]), float(c["coordY"]))
    except Exception:
        pass
    g = g.sort_values(["at_bat_number", "pitch_number"])
    swings = []
    for _, r in g.iterrows():
        if only and f'{int(r["at_bat_number"])}-{int(r["pitch_number"])}' not in only:
            continue
        if pd.isna(r.get("bat_speed")) or pd.isna(r.get(CY)) or pd.isna(r.get("swing_path_tilt")):
            continue
        hip = fits.get((int(r["batter"]), r["stand"]))
        if hip is None:
            continue
        C, d, arc, O, bat_h = swing_geometry(r, hip, r["stand"])
        hit = r["description"] in ("foul", "hit_into_play")
        out_pts, out_how = exit_path(r, C, hits.get((int(r["at_bat_number"]), int(r["pitch_number"])))) if hit else (None, None)
        vid = play.get((int(r["at_bat_number"]), int(r["pitch_number"])))
        swings.append({
            "id": f'{int(r["at_bat_number"])}-{int(r["pitch_number"])}',
            "game_pk": int(r["game_pk"]), "ab": int(r["at_bat_number"]), "pn": int(r["pitch_number"]),
            "bid": int(r["batter"]),
            "batter": names.get(int(r["batter"]), f'#{int(r["batter"])}'), "stand": r["stand"],
            "inning": f'{"T" if r["inning_topbot"] == "Top" else "B"}{int(r["inning"])}',
            "count": f'{int(r["balls"])}-{int(r["strikes"])}', "pitch": r["pitch_type"],
            "velo": r["release_speed"], "result": str(r["description"]).replace("_", " "),
            "event": "" if pd.isna(r["events"]) else str(r["events"]).replace("_", " "),
            "ev": None if pd.isna(r["launch_speed"]) else r["launch_speed"],
            "la": None if pd.isna(r["launch_angle"]) else r["launch_angle"],
            "bat_speed": r["bat_speed"], "swing_length": r["swing_length"], "tilt": r["swing_path_tilt"],
            "attack_angle": r["attack_angle"], "attack_dir": r["attack_direction"],
            "miss": None if pd.isna(r.get("miss_distance")) else r["miss_distance"],
            "contact": [round(float(v), 2) for v in C], "hip": [round(hip_x(hip, r), 1), round(hip["F0"], 1)],
            "hip_se": round(hip["se"], 1), "hip_n": hip["n"], "arc": arc, "ball": pitch_path(r, C[1], hit), "O": O,
            "exit": out_pts, "exit_how": out_how,
            "video": f"https://baseballsavant.mlb.com/sporty-videos?playId={vid}" if vid
                     else f"https://baseballsavant.mlb.com/gamefeed?gamePk={int(r['game_pk'])}",
            "video_exact": bool(vid),
            "made_contact": r["description"] in CONTACT,
            "sz": [round(float(r["sz_bot"]) * 12, 1), round(float(r["sz_top"]) * 12, 1)],
            "checks": (ck := fit_checks(r, hip)),
            # trust in the PLACEMENT only (hip fit + this swing's own residual); swing-pattern flags don't count
            "overall": next((c[2] for c in ck if c[0] == "This swing vs his fit"), "g"),
            "savant": hip.get("stance_grade", ""),
            "unusual": any(c[2] == "s" for c in ck),
        })
    html = (Path(__file__).parent / "template.html").read_text()
    Path(out_html).write_text(html.replace("/*DATA*/", json.dumps(swings)))
    print(f"{len(swings)} swings ({sum(w['video_exact'] for w in swings)} with their own video, "
          f"{sum(w['exit_how'] == 'measured (MLB feed)' for w in swings)} fouls with a direction from MLB's feed) -> {out_html}")


if __name__ == "__main__":
    # optional: --only 35-3,50-6,...  (at_bat-pitch ids) to build a page with just those swings
    # --stance batting-stance.csv  (Savant's Batting Stance leaderboard download) anchors each hitter's depth
    # --league league_2026.csv.gz  (fetch_league.py) -> group average from every MLB hitter
    a, opt = sys.argv[1:], {}
    for flag in ("--only", "--stance", "--league"):
        if flag in a:
            i = a.index(flag)
            opt[flag] = a[i + 1]
            a = a[:i] + a[i + 2:]
    only = set(opt["--only"].split(",")) if "--only" in opt else None
    build(a[0], a[1], a[2] if len(a) > 2 else str(Path(__file__).parent / "swing_map.html"), only,
          opt.get("--stance"), opt.get("--league"))
