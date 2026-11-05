import math

import numpy as np
import pandas as pd

from core import pitch_metrics as pm


def test_count_groups():
    b = pd.Series([0, 1, 2, 3, 3, 0, 1, 2, 1, 3])
    s = pd.Series([0, 0, 1, 1, 2, 1, 2, 2, 1, 0])
    g = [None if pd.isna(x) else x for x in pm.count_group(b, s).tolist()]
    assert g == ["First Pitch", "Hitter's Count", "Hitter's Count", "Hitter's Count", None,
                 "Pitcher's Count", "Pitcher's Count", "Pitcher's Count", None, "Hitter's Count"]


def test_clock_and_obs_dir():
    assert pm.clock(180) == "12:00" and pm.clock(225) == "1:30" and pm.clock(90) == "9:00"
    df = pd.DataFrame({"pitch_type": ["FF"] * 2, "pfx_z": [1.4, 1.4], "pfx_x": [-0.5, -0.5], "p_throws": "R",
                       "release_speed": [95, 97], "game_date": "2026-06-01", "description": "ball"})
    d = pm.prep(df)
    assert d["hb"].iloc[0] == 6.0                      # RHP arm side run shown positive
    assert 180 < d["obs_axis"].iloc[0] < 225           # ride + arm side ~ 12:30-1:30


def _df():
    r = np.random.default_rng(1)
    n = 600
    return pd.DataFrame({"pitch_type": r.choice(["FF", "SL", "CH"], n, p=[.5, .3, .2]), "p_throws": "R",
                         "release_speed": r.normal(92, 3, n), "pfx_z": r.normal(1, .3, n), "pfx_x": r.normal(-.4, .3, n),
                         "description": r.choice(["ball", "called_strike", "swinging_strike", "foul", "hit_into_play"], n),
                         "type": "S", "stand": r.choice(["L", "R"], n), "balls": r.integers(0, 4, n),
                         "strikes": r.integers(0, 3, n), "game_date": r.choice(["2026-06-01", "2026-06-06"], n),
                         "delta_run_exp": r.normal(0, .1, n)})


def test_rows_and_splits():
    d = pm.prep(_df())
    rows = pm.table_rows(d, [])
    assert [r[0]["Pitch"] for r in rows] == ["FF", "SL", "CH"]
    assert math.isclose(sum(r[1]["PCT"] for r in rows), 1.0)
    rows = pm.table_rows(d, ["hand", "count"])
    assert {r[0]["hand"] for r in rows} == {"vLHH", "vRHH"}
    assert [r[0]["count"] for r in rows][0] == "First Pitch"
    games = pm.table_rows(d, ["game"])
    assert games[0][0]["game"] == "2026-06-06"         # newest game first
    assert pm.table_rows(d, [], pitch="SL")[0][0]["Pitch"] == "SL" and len(pm.table_rows(d, [], pitch="SL")) == 1
    v = rows[0][1]
    assert 0 <= v["ASW"] <= 1 and 0 <= v["ABL"] <= 1


def test_evaluation_method_headers_and_grades():
    from ui.pitching import table_html
    rows = [{"year": 2026, "period": "Season", "labels": {"Pitch": "FF"},
             "vals": {"N": 500, "ACTL_Z": 1.0}, "season_row": True}]
    plus = table_html(pm.SUMMARY, rows, [], "Plus")
    assert "Plus (100 = average)" in plus and ">110<" in plus          # a future model's z = 1 -> 110 on Plus
    assert ">60<" in table_html(pm.SUMMARY, rows, [], "20-80")
    assert ">84<" in table_html(pm.SUMMARY, rows, [], "Percentile")
