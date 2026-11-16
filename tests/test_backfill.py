import json

import numpy as np
import pandas as pd

from core import batted_ball
from pipeline import backfill_era


def _league(n=400, seed=3):
    rng = np.random.default_rng(seed)
    aa = rng.normal(10, 8, n)
    bs, vp = rng.normal(72, 3, n), rng.normal(92, 3, n)
    ev = (1.23 * bs + 0.23 * vp) * rng.uniform(0.86, 0.98, n)          # all squared up
    return pd.DataFrame({"description": "hit_into_play", "stand": "R", "launch_speed": ev, "bat_speed": bs,
                         "release_speed": vp, "attack_angle": aa, "launch_angle": aa + 5.0 + rng.normal(0, 1, n),
                         "attack_direction": 0.0, "hc_x": 125.42 - 50 * np.tan(np.radians(8)), "hc_y": 198.27 - 50})


def test_calibration_finds_where_squared_up_balls_leave():
    c = batted_ball.calibrate(_league())
    assert abs(c["v_center"] - 5.0) < 0.5 and abs(c["h_center"] - (-8.0)) < 0.5 and c["n_squared_up"] == 400


def test_season_centres_are_read_from_the_backfill(tmp_path, monkeypatch):
    import core.batted_ball as bb
    cal = tmp_path / "data" / "calibration"
    cal.mkdir(parents=True)
    (cal / "batted_ball_2031.json").write_text(json.dumps({"v_center": 6.5, "h_center": -11.0}))
    monkeypatch.setattr(bb, "__file__", str(tmp_path / "core" / "batted_ball.py"))
    bb._CAL.clear()
    assert bb.centers(2031) == (6.5, -11.0) and bb.centers(2030) == (bb.V_CENTER, bb.H_CENTER)
    bb._CAL.clear()


def test_backfill_keeps_going_when_a_season_fails(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(backfill_era, "CAL", tmp_path)

    def step(season, force):
        seen.append(season)
        if season == 2024:
            raise RuntimeError("Savant down")
    monkeypatch.setattr(backfill_era, "season_steps", step)
    import pipeline.update as up
    monkeypatch.setattr(up, "main", lambda argv: None)
    monkeypatch.setattr(up, "_git", lambda *a: 0)
    backfill_era.main(["2023", "2024", "2025"])
    log = (tmp_path / "backfill_log.txt").read_text()
    assert seen == [2023, 2024, 2025] and "2024 FAILED" in log and "backfill done" in log
