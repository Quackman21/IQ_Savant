import numpy as np
import pandas as pd

from core import batted_ball, hit_metrics as hm


def _swings(n=300, seed=4):
    rng = np.random.default_rng(seed)
    desc = rng.choice(["hit_into_play", "foul", "swinging_strike", "ball"], n, p=[.3, .3, .2, .2])
    df = pd.DataFrame({
        "batter": 1, "game_date": "2025-06-01", "game_pk": 1, "game_type": "R", "stand": rng.choice(["L", "R"], n),
        "p_throws": rng.choice(["L", "R"], n), "pitch_type": rng.choice(["FF", "SL", "CH"], n),
        "description": desc, "events": np.where(desc == "hit_into_play", "single", None),
        "type": np.where(desc == "hit_into_play", "X", "S"), "zone": rng.integers(1, 15, n),
        "bat_speed": np.where(desc != "ball", rng.normal(72, 5, n), np.nan),
        "swing_length": 7.2, "attack_angle": rng.normal(10, 8, n), "attack_direction": rng.normal(-2, 10, n),
        "swing_path_tilt": rng.normal(32, 4, n), "release_speed": rng.normal(91, 4, n),
        "launch_speed": np.where(np.isin(desc, ["hit_into_play", "foul"]), rng.normal(88, 12, n), np.nan),
        "launch_angle": rng.normal(12, 25, n), "hc_x": rng.normal(125, 40, n), "hc_y": rng.normal(120, 30, n),
        "intercept_ball_minus_batter_pos_y_inches": rng.normal(30, 5, n),
        "intercept_ball_minus_batter_pos_x_inches": rng.normal(28, 3, n), "launch_speed_angle": rng.integers(1, 7, n),
    })
    return df


def test_vector_labels_match_row_labels():
    df = _swings()
    vec = batted_ball.spin_labels(df)
    for i, r in df.iterrows():
        one = batted_ball.spin_label(r)
        assert (one["label"] if one else None) == vec[i]


def test_bat_path_metrics():
    df = _swings()
    df["poc_front"] = df["intercept_ball_minus_batter_pos_y_inches"] - 28
    v = hm.compute(hm.prep(df)).iloc[0]
    made = df["description"].isin(["foul", "hit_into_play"])
    assert abs(v["POC DEPTH"] - df.loc[made, "intercept_ball_minus_batter_pos_y_inches"].mean()) < 1e-9
    assert abs(v["POC FRONT"] - (v["POC DEPTH"] - 28)) < 1e-9
    assert 0 <= v["SQ UP%/CON"] <= 1 and v["SQ UP%/SW"] <= v["SQ UP%/CON"]
    assert v["BLAST%/CON"] <= v["SQ UP%/CON"] + 1e-9
    types = ["OPTIMAL%", "BACKSPIN%", "TOPSPIN%", "HOOK%", "SLICE%", "KNUCKLE%"]
    assert abs(sum(v[t] for t in types) - 1) < 1e-9
    for keys in hm.BAT_PATH.values():
        assert all(k in hm.CATALOG for k in keys)


def test_poc_map_renders():
    from ui.bat_path import poc_map_html
    m = pd.DataFrame({"pf": [5.0, -3.0, 10.0], "pxc": [1.0, -4.0, 2.0], "bip": [True, False, True],
                      "ev": [101.0, np.nan, 80.0], "la": [20.0, np.nan, -5.0], "pgroup": ["Fastball"] * 3,
                      "ptype": ["FF"] * 3, "p_throws": ["R"] * 3, "zone_io": ["In Zone"] * 3,
                      "date": ["2025-06-01"] * 3, "result": ["Single", "Foul", "Field out"]})
    h = poc_map_html(m)
    assert h.count("<circle") == 3 and "Where he meets it" in h
    assert "No measured" in poc_map_html(m.assign(pf=np.nan))


def test_poc_uses_stance_depth():
    from core import bat_path
    df = _swings(40).assign(plate_x=0.0, vx0=0.0, vy0=-130.0, ax=0.0, ay=25.0)
    views = {"season": {(1, "R"): 27.0, (1, "L"): 29.0}}
    out = bat_path.poc(df, 1, 2025, views, {})
    made = df["description"].isin(["foul", "hit_into_play"])
    exp = df["intercept_ball_minus_batter_pos_y_inches"] - np.where(df["stand"] == "R", 27.0, 29.0)
    assert np.allclose(out.loc[made, "pf"], exp[made]) and out.loc[~made, "pf"].isna().all()
    assert not out["estimated"].any()
