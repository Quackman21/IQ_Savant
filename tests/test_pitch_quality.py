import numpy as np
import pandas as pd

from models import pitch_quality as pq


def test_arm_side_mirrors_and_fastball_differences():
    df = pd.DataFrame({"p_throws": ["R", "L", "R"], "pfx_x": [-1.2, 1.2, 0.5], "release_pos_x": [-2.0, 2.0, -2.0],
                       "pfx_z": [1.4, 1.4, 0.1], "release_speed": [95, 95, 85], "pitch_type": ["FF", "FF", "SL"],
                       "pitcher": [1, 2, 1], "game_date": "2025-05-01", "stand": "R"})
    f = pq.features(df)
    assert f.loc[0, "hb"] == f.loc[1, "hb"] > 0 and f.loc[0, "rel_x"] == f.loc[1, "rel_x"] > 0
    assert f.loc[2, "d_velo"] == -10 and round(f.loc[2, "d_ivb"], 6) == round(12 * (0.1 - 1.4), 6)


def test_train_and_score_scale():
    from pipeline import train_pitch_quality as t
    rng = np.random.default_rng(1)
    n = 20000
    talent = rng.random(n)
    df = pd.DataFrame({"game_date": "2025-06-01", "game_pk": rng.integers(1, 300, n), "pitcher": rng.integers(1, 50, n),
                       "p_throws": "R", "stand": rng.choice(["L", "R"], n), "pitch_type": "FF",
                       "release_speed": 92 + 5 * talent, "pfx_x": -0.7, "pfx_z": 1.3, "release_spin_rate": 2300,
                       "release_extension": 6.3, "release_pos_x": -2, "release_pos_z": 5.8, "arm_angle": 40.0,
                       "plate_x": rng.normal(0, 0.8, n), "plate_z": rng.normal(2.5, 0.8, n), "sz_top": 3.4, "sz_bot": 1.6,
                       "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n), "type": "S",
                       "description": rng.choice(["ball", "called_strike"], n), "delta_run_exp": 0.0,
                       "estimated_woba_using_speedangle": np.nan})
    df["delta_run_exp"] = np.where(df["description"] == "ball", 0.05, -0.05) - 0.04 * talent + rng.normal(0, 0.05, n)
    b = t.fit(df)
    s = pq._score_one(pq.features(df), b)
    assert abs(s["stuff_plus"].mean() - 100) < 1 and abs(s["loc_plus"].mean() - 100) < 1
    assert np.corrcoef(s["stuff_plus"], talent)[0, 1] > 0.5        # more velo -> more Stuff+
