import math

import numpy as np
import pandas as pd

from core import path_fit as pfit


def _hitters(n_hit=4, n=400, seed=3, aa_slope=0.30, ad_slope=-1.2):
    """Synthetic hitters whose TRUE path turns at known rates: attack angle +aa_slope deg / in out front,
    attack direction ad_slope deg / in (toward the pull side), + he adapts to pitch height (+4 deg / ft)."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_hit):
        c = rng.normal(5, 3)
        for _ in range(n):
            pf = rng.normal(c, 7)
            z, x = rng.normal(2.5, 0.5), rng.normal(0, 0.6)
            u = pf - c
            rows.append({"batter": 100 + h, "stand": "R", "year": 2026, "made": True, "pf": pf, "depth": 30.0,
                         "plate_x": x, "plate_z": z, "swing_path_tilt": rng.normal(34, 3),
                         "bat_speed": 72 + 0.05 * u + rng.normal(0, 2),
                         "attack_angle": 8 + aa_slope * u + 4 * (z - 2.5) + rng.normal(0, 2),
                         "attack_direction": -2 + ad_slope * u + 6 * x + rng.normal(0, 2),
                         "intercept_ball_minus_batter_pos_x_inches": 30.0, "group": "Fastball"})
    return pfit.contacts(pd.DataFrame(rows))


def test_league_rates_recover_the_true_turn():
    d = _hitters()
    lr = pfit.league_rates(d)
    assert abs(lr["all"]["attack_angle"]["slope"] - 0.30) < 0.05
    assert abs(lr["all"]["attack_direction"]["slope"] + 1.2) < 0.06
    assert lr["all"]["circle"]["attack_direction"] == lr["all"]["circle"]["attack_direction"]   # computed (not NaN)


def test_measured_rates_beat_no_change_held_out():
    ev = pfit.evaluate(_hitters())
    for t in ("attack_angle", "attack_direction"):
        assert ev[t]["all"]["measured"] < ev[t]["all"]["flat"]
        assert ev[t]["all"]["measured"] < 2.3                       # ~ the noise we put in (2 deg)


def test_hitter_curve_and_rebuilt_path_follow_the_rates():
    d = _hitters(n_hit=1, n=600)
    lr = pfit.league_rates(_hitters(seed=9))                       # shrinks toward the league, so give it one
    cur = pfit.hitter_curve(d, {"swing_path_tilt": 34, "bat_speed": 72, "plate_z": 2.5, "xh": 0.0}, lr)
    assert abs(cur["attack_direction"]["slope"] + 1.2) < 0.15 and cur["lo"] < cur["center"] < cur["hi"]
    C = [0.0, cur["center"], 30.0]
    r = pfit.rebuild(C, 8.0, -2.0, 72.0, "R", cur, pre_len=30, post_len=30)
    arc = np.array([p[:3] for p in r["arc"]])
    t = np.array([p[4] for p in r["arc"]])
    assert np.all(np.diff(t) > 0)                                 # time runs forward
    i0 = int(np.argmin(np.abs(t)))
    assert np.allclose(arc[i0], C)
    # 6 inches further out front (inside his range), the path points ~7 deg further to the pull side
    k = int(np.argmin(np.abs(arc[:, 1] - (cur["center"] + 6))))
    tang = arc[k + 1] - arc[k - 1]
    ad = math.degrees(math.atan2(tang[0], tang[1]))
    assert abs(ad - (-2.0 + 6 * cur["attack_direction"]["slope"] + 36 * cur["attack_direction"]["curve"])) < 1.0
    assert {p[3] for p in r["arc"]} == {2, 3}                     # measured (3) inside, extended (2) beyond
    assert len(r["hands"]) == len(r["arc"])


def test_time_curves_follow_the_measured_rates():
    from core.bat_path import TGRID, time_curves
    lr = {"all": {"attack_angle": {"slope": 0.9, "curve": 0.0}, "attack_direction": {"slope": -1.55, "curve": 0.0},
                  "bat_speed": {"slope": 0.23, "curve": 0.0}}}
    n = 3
    cv = time_curves([8.0] * n, [0.0] * n, [34.0] * n, [72.0] * n, [7.0] * n, ["R"] * n, [5.0] * n, [30.0] * n,
                     [30.0] * n, [0.0] * n, [30.0] * n, lr=lr)
    j0 = int(np.argmin(np.abs(TGRID)))
    assert abs(cv["attack_angle"][0, j0] - 8.0) < 1e-6
    later = j0 + 3                                                    # 6 ms after contact
    assert cv["attack_angle"][0, later] > 8.0 + 3 and cv["attack_direction"][0, later] < -5   # rising, turning pull
    assert -60 < cv["vba"][0, j0] < -10                               # barrel below the hands
    assert np.isnan(cv["attack_angle"][0, 0]) or TGRID[0] > -70       # drawn back only to his swing length
