import numpy as np
import pandas as pd

from models import decision_grade as dg


def _season(seed=0, n_hitters=80, pitches=2400):
    """Hitters differ in discipline (laying off chase / waste) and in plain aggressiveness (swing rate)."""
    rng = np.random.default_rng(seed)
    disc, aggro = rng.normal(0, .6, n_hitters), rng.normal(0, .6, n_hitters)
    disc[0], aggro[0] = 2.0, 0.0                 # hitter 1000: elite discipline
    disc[1], aggro[1] = 0.0, 2.0                 # hitter 1001: just swings more, no better choices
    h = np.repeat(np.arange(n_hitters), pitches)
    n = len(h)
    px, pz = rng.normal(0, 1.0, n), rng.normal(2.5, 1.0, n)
    r = np.maximum(np.abs(px) / (17 / 24), np.abs(pz - 2.5) / 0.9)
    should = r < 0.9
    p = 1 / (1 + np.exp(-(1.0 - 1.6 * r + aggro[h] + disc[h] * np.where(should, .5, -1))))
    swing = rng.random(n) < p
    gp = rng.integers(0, 2430, n)
    raw = pd.DataFrame({"batter": h + 1000, "description": np.where(swing, "foul", np.where(r < 1, "called_strike", "ball")),
                        "plate_x": px, "plate_z": pz, "sz_top": 3.4, "sz_bot": 1.6, "zone": np.where(r < 1, 5, 13),
                        "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
                        "pitch_type": rng.choice(["FF", "SL", "CH"], n), "game_pk": gp,
                        "events": np.where(rng.random(n) < 1 / 3.9, "field_out", None),
                        "home_team": np.array([f"T{g % 30}" for g in gp]),
                        "away_team": np.array([f"T{(g + 15) % 30}" for g in gp])})
    right = np.where(should, swing, ~swing)
    runs = pd.Series(np.where(right, .01, -.02) * np.where(r > 1.3, 2, 1), index=raw.index)
    return raw, runs, pd.Series(True, index=raw.index)


def test_discipline_grades_high_and_swing_rate_alone_does_not():
    raw, runs, graded = _season()
    ref = dg.build_ref(raw, runs, graded, 2026)
    assert all("sd_blend" in ref["splits"][s] for s in dg.SPLITS)
    m0, m1 = raw.batter == 1000, raw.batter == 1001
    star = dg.score(raw[m0], runs[m0], graded[m0], ref)
    hacker = dg.score(raw[m1], runs[m1], graded[m1], ref)
    assert 50 + 10 * star["_DEC_Z"] > 65
    assert 50 + 10 * hacker["_DEC_Z"] < 50 + 10 * star["_DEC_Z"] - 10      # swinging more isn't rewarded
    assert "DEC RUNS/100" in star["_DEC_PARTS"] and "decisions" in star["_DEC_PARTS"]


def test_splits_use_the_right_pieces_and_small_samples_blank():
    raw, runs, graded = _season(seed=1)
    ref = dg.build_ref(raw, runs, graded, 2026)
    assert "Chase+waste swing% 2k" not in ref["splits"]["00"]["stats"]
    assert "Chase+waste swing% <2k" not in ref["splits"]["2k"]["stats"]
    m = raw.batter == 1000
    few = dg.score(raw[m].head(10), runs[m].head(10), graded[m].head(10), ref)
    assert few["_DEC_Z"] != few["_DEC_Z"]                                    # under the minimum -> blank
    some = dg.score(raw[m].head(300), runs[m].head(300), graded[m].head(300), ref)
    full = dg.score(raw[m], runs[m], graded[m], ref)
    assert abs(some["_DEC_Z"]) < abs(full["_DEC_Z"])                        # regressed toward average
