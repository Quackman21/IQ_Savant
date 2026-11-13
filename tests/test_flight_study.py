import numpy as np
import pandas as pd

from pipeline import flight_study as fs


def test_flight_runs_take_exit_velo_out(monkeypatch):
    rng = np.random.default_rng(0)
    n = 20000
    ev, la = rng.normal(90, 12, n), rng.normal(12, 25, n)
    raw = pd.DataFrame({"game_pk": rng.integers(0, 500, n), "game_type": "R", "batter": rng.integers(1, 200, n),
                        "description": "hit_into_play", "events": "field_out", "type": "X", "launch_speed": ev,
                        "launch_angle": la, "hc_x": 125 + rng.normal(0, 40, n), "hc_y": 120 + rng.normal(0, 20, n),
                        "stand": "R", "woba_value": 0.3, "woba_denom": 1})
    from models import saxwoba
    from pipeline import build_league_reference as bl
    monkeypatch.setattr(bl, "pull_season", lambda y: raw)
    # expected value rises with exit velo AND with a good launch angle
    monkeypatch.setattr(saxwoba, "expected_woba",
                        lambda df: pd.Series(0.01 * (df["launch_speed"] - 60) + 0.3 * np.exp(-((df["launch_angle"] - 20) / 15) ** 2),
                                             index=df.index))
    d = fs.load(2025)
    assert abs(np.corrcoef(d["ev"], d["flight_x"])[0, 1]) < 0.02          # hitting it hard earns nothing here
    assert d.loc[d.la.between(15, 25), "flight_x"].mean() > d.loc[d.la < -10, "flight_x"].mean()   # good angle does
