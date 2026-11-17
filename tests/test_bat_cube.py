import numpy as np
import pandas as pd


def _raw(n=5000, seed=5):
    rng = np.random.default_rng(seed)
    desc = rng.choice(["foul", "hit_into_play", "swinging_strike", "ball", "called_strike", "foul_bunt"], n,
                      p=[.25, .25, .15, .2, .1, .05])
    return pd.DataFrame({
        "game_date": "2026-06-01", "game_pk": rng.integers(1, 50, n), "game_type": "R", "batter": rng.integers(1, 20, n),
        "pitcher": rng.integers(100, 140, n), "stand": rng.choice(["L", "R"], n), "p_throws": rng.choice(["L", "R"], n),
        "pitch_type": rng.choice(["FF", "SL", "CH", "SI"], n), "zone": rng.integers(1, 15, n),
        "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n), "description": desc,
        "events": np.where(desc == "hit_into_play", "single", None), "bat_speed": rng.normal(71, 6, n),
        "swing_length": rng.normal(7.2, .5, n), "attack_angle": rng.normal(10, 7, n),
        "attack_direction": rng.normal(0, 10, n), "swing_path_tilt": rng.normal(32, 4, n),
        "intercept_ball_minus_batter_pos_y_inches": rng.normal(30, 6, n),
        "intercept_ball_minus_batter_pos_x_inches": rng.normal(28, 3, n), "launch_speed": rng.normal(88, 12, n),
        "launch_angle": rng.normal(12, 25, n), "hc_x": rng.normal(125, 40, n), "hc_y": rng.normal(120, 30, n),
        "plate_x": rng.normal(0, .6, n), "plate_z": rng.normal(2.5, .6, n), "sz_top": 3.4, "sz_bot": 1.6,
        "release_speed": rng.normal(90, 5, n), "vx0": 5.0, "vy0": -135.0, "vz0": -5.0, "ax": -8.0, "ay": 28.0,
        "az": -15.0})


def test_cube_and_exact_rollups(monkeypatch):
    from core import aar
    from pipeline import bat_cube
    monkeypatch.setattr(aar, "stance_depths", lambda season: {(b, s): 28.0 for b in range(1, 20) for s in "LR"})
    raw = _raw()
    cube = bat_cube.build(2026, raw, verbose=False)
    assert len(cube) and not cube["description"].astype(str).str.contains("bunt").any()
    assert set(cube["description"]) <= bat_cube.SWINGS
    made = cube["made"]
    assert cube.loc[made, "pf"].notna().all() and cube.loc[~made, "pf"].isna().all()
    assert np.allclose(cube.loc[made, "gap_v"], cube.loc[made, "attack_angle"] - cube.loc[made, "match_v"])
    assert cube.loc[made, "vba"].notna().mean() > 0.95
    s = bat_cube.sums(cube)
    assert s["swings"].sum() == len(cube)
    # any combo rolls up exactly
    for kw, m in [({}, pd.Series(True, index=cube.index)),
                  ({"batter": 3, "types": ["FF", "SI"], "zones": [5, 6], "fast": True},
                   (cube.batter == 3) & cube.pitch_type.isin(["FF", "SI"]) & cube.zone.isin([5, 6]) & cube.fast),
                  ({"hand": "L", "counts": ["Two Strikes"]}, (cube.p_throws == "L") & (cube.count_grp == "Two Strikes"))]:
        r = bat_cube.rollup(s, **kw)
        g = cube[m]
        assert r["swings"] == len(g)
        for k in ("attack_angle", "gap_v", "vba"):
            v = g[k].dropna()
            assert r[k]["n"] == len(v)
            if len(v) > 1:
                assert abs(r[k]["mean"] - v.mean()) < 1e-3 and abs(r[k]["sd"] - v.std()) < 1e-3
