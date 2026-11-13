import pandas as pd

from pipeline import defense_study as ds


def test_for70_spearman_brown():
    # r = .5 at 100 chances a half -> k = 100 -> .70 at 233
    assert round(ds.for70(0.5, 100)) == 233
    assert ds.for70(0, 100) != ds.for70(0, 100)          # NaN when nothing repeats


def test_framing_section_compares_count_and_situational(monkeypatch):
    from core import league_ref
    def fake(y):
        rows = []
        for c in range(30):
            rows.append({"catcher": c, "n": 3000, "strikes": 1500 + c * 3 + (y % 2), "exp": 1500,
                         "runs": c * 0.4, "runs_sit": c * 0.45 + (y % 3) * 0.1, "game_date": f"{y}-05-01"})
        return pd.DataFrame(rows)
    monkeypatch.setattr(league_ref, "load_catching", fake)
    L = "\n".join(ds.framing_section([2023, 2024, 2025]))
    assert "r(count runs, situational runs)" in L and "best predicts NEXT season" in L


def test_strike_value_one_era_season_and_str_counts():
    """A 2015 season has no post-2023 rows; that empty era used to crash framing on the PC (str - str)."""
    import numpy as np
    from models import decisions_iq as dq
    t = {"count": {"ball": {"0-0": .05, "3-2": .40}, "strike": {"0-0": -.05, "3-2": -.30}},
         "bo": {"ball": {"3-2|7|2": .9}, "strike": {"3-2|7|2": -.4}}}
    b = {"eras": {"pre": {"tables": t}, "post": {"tables": t}}, "split": 2023}
    df = pd.DataFrame({"game_date": ["2015-05-01"] * 2, "balls": [0, 3], "strikes": [0, 2],
                       "on_1b": [np.nan, 1], "on_2b": [np.nan, 1], "on_3b": [np.nan, 1], "outs_when_up": [0, 2]})
    v = dq.strike_value(df, b)
    assert np.allclose(v, [.10, .70])
    s = dq.strike_value_situational(df, b)
    assert np.isclose(s.iloc[0], .10) and s.iloc[1] > v.iloc[1]     # bases loaded, 2 outs: worth more
