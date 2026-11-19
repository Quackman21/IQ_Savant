import math

import numpy as np
import pandas as pd

from core import aar, swing3d
from ui import aar as aar_ui


def _row(**kw):
    r = {"description": "hit_into_play", "stand": "R", "p_throws": "R", "vx0": 6.0, "vy0": -130.0, "vz0": -5.0,
         "ax": -10.0, "ay": 28.0, "az": -16.0, "plate_x": 0.2, "plate_z": 2.5, "release_pos_y": 54.0,
         "bat_speed": 72.0, "swing_length": 7.2, "attack_angle": 10.0, "attack_direction": -5.0,
         "swing_path_tilt": 30.0, "intercept_ball_minus_batter_pos_y_inches": 31.0,
         "intercept_ball_minus_batter_pos_x_inches": 30.0, "launch_speed": 100.0, "launch_angle": 20.0,
         "hc_x": 100.0, "hc_y": 80.0, "sz_top": 3.4, "sz_bot": 1.6, "release_speed": 92.0}
    r.update(kw)
    return pd.Series(r)


def test_contact_is_mlbs_definition_and_on_the_pitch():
    s = swing3d.scene(_row(), depth=28.0, stance_x=27.0)
    assert s["read"]["front"] == 3.0                                    # 31 in front of hips - 28 stance depth
    last = s["pitch"][-1]
    assert abs(last[1] - 3.0) < 0.01 and abs(last[3]) < 1e-6           # the pitch ends at contact, t = 0
    assert np.allclose(last[:3], s["contact"], atol=0.02)
    assert s["pitch"][0][3] < -0.35                                     # starts at release, ~0.4 s before


def test_bat_moves_through_contact_at_the_measured_angles():
    s = swing3d.scene(_row(), depth=28.0, stance_x=27.0)
    arc = np.array([a[:3] for a in s["arc"]])
    t = np.array([a[4] for a in s["arc"]])
    i0 = int(np.argmin(np.abs(t)))
    assert abs(t[i0]) < 1e-9 and np.allclose(arc[i0], s["contact"], atol=0.01)  # at the ball at contact
    assert min(a[4] for a in s["arc"] if a[3] == 1) >= -0.0405                  # measured part = last 40 ms
    v = np.array(s["aim"])                                               # direction of travel at contact
    assert abs(math.degrees(math.atan2(v[2], math.hypot(v[0], v[1]))) - 10.0) < 1.0     # attack angle
    assert abs(math.degrees(math.atan2(v[0], v[1])) - (-5.0)) < 1.0                       # RHB: -5 = pull = 3B side
    assert O_above(s)
    assert all(a[3] == 2 for a in s["arc"] if a[4] > 0)                 # follow-through: inferred, never "measured"


def O_above(s):
    return s["O"][2] > s["contact"][2]                                  # arc centre (hands) above the ball


def test_only_contact_swings_and_fouls_get_no_flight():
    assert swing3d.scene(_row(description="swinging_strike"), depth=28.0) is None
    assert swing3d.scene(_row(), depth=None) is None
    foul = swing3d.scene(_row(description="foul", hc_x=None, hc_y=None), depth=28.0)
    assert foul is not None and foul["exit"] is None
    assert swing3d.scene(_row(), depth=28.0)["exit"]


def test_aar_puts_a_3d_button_on_contact_swings():
    from tests.test_aar import game
    g = game()
    for k, v in {"vx0": 6.0, "vy0": -130.0, "vz0": -5.0, "ax": -10.0, "ay": 28.0, "az": -16.0, "attack_angle": 8.0,
                 "attack_direction": 2.0, "swing_path_tilt": 30.0, "swing_length": 7.0}.items():
        g[k] = v
    p = aar.pitches(g)
    p = p.join(aar.pitch_stance(p, 1, {"season": {(1, "R"): 28.0}})).join(aar.on_plate(aar.pitches(g).assign(
        depth=28.0), None))
    s3 = swing3d.scenes(g, p)
    assert set(s3) == set(p.loc[p["cy"].notna(), "n"])                  # fouls + balls in play, nothing else
    html = aar_ui.component_html(p, aar.plate_appearances(p), {"AB": 4}, {k: [] for k in aar.GAME_METRICS},
                                 aar.GAME_METRICS, "", scenes3d=s3, header="<div id='hdr'>Hitter</div>")
    assert html.count('class="s3d"') == len(s3) and 'id="ovl"' in html and "id='hdr'" in html
    # the profile's path display, per swing: plane-match colouring, on-plane read-out, arrow, legend boxes
    assert 'id="tv"' in html and 'id="s3trail"' in html and 'id="s3arrow"' in html and 'id="tl"' in html
    assert "function onPlane" in html and "function planeRows" in html and "function onBuilt" in html


def test_bat_angle_and_bat_parallel_are_inferred_from_the_measured_plane():
    s = swing3d.scene(_row(), depth=28.0, stance_x=27.0)
    assert -45 < s["read"]["vba"] < 0                                   # barrel below the hands at contact
    bp = s["bp"]
    assert bp and abs(bp["at"][2] - s["O"][2]) < 0.05                    # bat parallel: barrel level with the hands
    assert bp["inches_before"] <= 7.2 * 12 and bp["t"] < -0.04           # inside his swing, before the 40 ms window
    assert {a[3] for a in s["arc"]} == {1, 2}                            # measured window + inferred lead-in
    assert all(a[3] == 2 for a in s["arc"] if a[4] > 0)                 # follow-through: inferred, never "measured"
    short = swing3d.scene(_row(swing_length=3.0), depth=28.0)            # a swing too short to reach bat parallel
    assert short["bp"] is None and not any(a[3] == 2 and a[4] < 0 for a in short["arc"])


def test_contact_labels():
    from core import batted_ball as bb
    base = dict(_row())
    lab = lambda **kw: bb.spin_label(pd.Series({**base, **kw}))["label"]
    assert lab(launch_speed=108.0, launch_angle=24.0, attack_angle=18.0, launch_speed_angle=6) == "Optimal"
    assert lab(launch_speed=80.0, launch_angle=55.0, attack_angle=10.0, hc_x=None, hc_y=None) == "Backspin"
    assert lab(launch_speed=85.0, launch_angle=-20.0, attack_angle=10.0) == "Topspin"
    # RHB, barrel aimed at -5 (pull) but the ball went way to right field -> slice
    assert lab(launch_speed=85.0, launch_angle=30.0, attack_angle=20.0, attack_direction=-5.0, hc_x=200.0, hc_y=120.0) == "Slice"
    q = bb.contact_quality(pd.Series({**base, "launch_speed": 100.0, "bat_speed": 72.0, "release_speed": 92.0}))
    assert q["smash"] == round(1 + (100 - 72) / (92 + 72), 2) and q["squared_up"]


def test_no_contact_grade_unless_in_play():
    from core import batted_ball as bb
    foul = _row(description="foul")
    assert bb.spin_label(foul) is None
    s = swing3d.scene(foul, depth=28.0)
    assert s["read"]["spin"] is None and s["read"]["squared_up_pct"] is None and s["read"]["smash"] is None
