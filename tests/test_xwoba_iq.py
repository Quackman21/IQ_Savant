import numpy as np
import pandas as pd

from models import xwoba_iq as xq


def test_features_pull_adjusted_and_labels():
    df = pd.DataFrame({"stand": ["R", "L"], "launch_speed": [100, 100], "launch_angle": [25, 25],
                       "hc_x": [60, 60], "hc_y": [120, 120]})          # both hit to left field
    f = xq.features(df)
    assert f.loc[0, "spray_pull"] > 0 > f.loc[1, "spray_pull"]          # pulled for RHB, oppo for LHB
    lab = xq.label(pd.Series(["single", "field_error", "sac_fly", "sac_bunt", "home_run"]))
    assert lab.tolist()[:3] == ["single", "out", "out"] and pd.isna(lab[3]) and lab[4] == "home_run"


def test_season_weights_from_savant_values():
    df = pd.DataFrame({"game_date": ["2025-05-01"] * 4 + ["2024-05-01"] * 4,
                       "events": ["single", "double", "triple", "home_run"] * 2,
                       "woba_value": [0.88, 1.25, 1.59, 2.05, 0.89, 1.27, 1.61, 2.07]})
    w = xq.season_weights(df)
    assert w[2025]["single"] == 0.88 and w[2024]["home_run"] == 2.07


def test_iq_model_falls_back_to_savant_when_untrained(monkeypatch, tmp_path):
    monkeypatch.setattr(xq, "ARTIFACT", tmp_path / "none.joblib")
    xq._model_cache.clear()
    from models import xwoba
    df = pd.DataFrame({"type": ["X", "B"], "woba_value": [0.0, 0.7], "estimated_woba_using_speedangle": [0.4, np.nan],
                       "events": ["field_out", "walk"]})
    out = xwoba.iq_v1(df)
    assert out.tolist() == [0.4, 0.7]                    # no model yet -> Savant on BIP, actual on walk


def test_air_spray_only_on_air_balls_and_direction_buckets():
    df = pd.DataFrame({"stand": ["R", "R", "L"], "launch_speed": [100, 95, 90], "launch_angle": [25, -5, 5],
                       "hc_x": [60, 60, 60], "hc_y": [120, 120, 120], "venue": [3, 3, 7]})
    f = xq.features(df)
    assert f.loc[0, "spray_air"] == f.loc[0, "spray_pull"]          # fly ball keeps the exact angle
    assert np.isnan(f.loc[1, "spray_air"])                           # ground ball: no exact angle
    assert f["spray_dir"].tolist() == [1.0, 1.0, -1.0]                # pulled, pulled, oppo (LHB to left field)
    assert f["park"].tolist() == [3, 3, 7]


def test_park_neutral_scoring_and_calibration():
    from pipeline import train_xwoba as t
    rng = np.random.default_rng(0)
    n = 6000
    ev, la = rng.normal(90, 12, n), rng.normal(12, 25, n)
    venue = rng.integers(1, 4, n)
    hit = rng.random(n) < np.clip(0.05 + 0.01 * (ev - 85) + 0.1 * (venue == 1), 0, 0.9)
    kind = np.where(hit, rng.choice(["single", "double", "triple", "home_run"], n, p=[.6, .25, .03, .12]), "field_out")
    df = pd.DataFrame({"game_date": "2025-06-01", "stand": "R", "type": "X", "events": kind,
                       "launch_speed": ev, "launch_angle": la, "hc_x": 125, "hc_y": 100, "venue": venue,
                       "woba_value": pd.Series(kind).map({"single": .88, "double": 1.25, "triple": 1.59,
                                                          "home_run": 2.05}).fillna(0).values, "woba_denom": 1})
    b = t.fit(df, "air_dir")
    assert b["features"] == xq.VARIANTS["air_dir"][0] and b["parks"]
    one = df.iloc[[0]].assign(venue=np.nan)             # scoring never needs the park
    two = df.iloc[[0]].assign(venue=1)
    assert xq.score_batted_balls(one, b).iloc[0] == xq.score_batted_balls(two, b).iloc[0]
    x = xq.score_batted_balls(df, b)
    assert abs(x.mean() - df["woba_value"].mean()) < 1e-9             # calibrated to league wOBACON


def test_fly_ball_spray_and_blur_scoring():
    df = pd.DataFrame({"stand": ["R"] * 3, "launch_speed": [100, 100, 90], "launch_angle": [30, 15, 22],
                       "hc_x": [60] * 3, "hc_y": [120] * 3})
    f = xq.features(df)
    assert not np.isnan(f.loc[0, "spray_fly"]) and np.isnan(f.loc[1, "spray_fly"])   # 15 deg = liner
    assert np.isnan(f.loc[2, "spray_hr"])                                          # under 95 mph
    from pipeline import train_xwoba as t
    rng = np.random.default_rng(2)
    n = 4000
    kind = rng.choice(["field_out", "single", "double", "triple", "home_run"], n, p=[.66, .2, .07, .01, .06])
    d = pd.DataFrame({"game_date": "2025-06-01", "stand": "R", "type": "X", "events": kind,
                      "launch_speed": rng.normal(90, 12, n), "launch_angle": rng.normal(12, 25, n),
                      "hc_x": rng.uniform(40, 210, n), "hc_y": rng.uniform(60, 180, n), "venue": rng.integers(1, 4, n),
                      "woba_value": pd.Series(kind).map({"single": .88, "double": 1.25, "triple": 1.59,
                                                         "home_run": 2.05}).fillna(0).values, "woba_denom": 1})
    b = t.fit(d, "blur12")
    assert b["blur"] == 12
    x = xq.score_batted_balls(d, b)
    assert x.notna().all() and abs(x.mean() - d["woba_value"].mean()) < 1e-9


def test_blur_by_ball_type_widths():
    X = pd.DataFrame({"la": [-5, 15, 30, np.nan]})
    assert xq.blur_widths(X, "by_type").tolist() == [18.0, 12.0, 0.0, 12.0]
    assert xq.blur_widths(X, 0) is None
