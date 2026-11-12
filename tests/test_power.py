import numpy as np
import pandas as pd

from models import power


def _season(year, n_hitters=120, pitches=900, tracked=True, seed=0):
    rng = np.random.default_rng(seed)
    bs_true = rng.normal(72, 3, n_hitters)
    bs_true[0] = 80                                            # hitter 1000 = big power
    h = np.repeat(np.arange(n_hitters), pitches)
    n = len(h)
    bs = bs_true[h] + rng.normal(0, 3, n)
    ps = rng.normal(93, 3, n)
    desc = rng.choice(["ball", "called_strike", "swinging_strike", "foul", "hit_into_play"], n,
                      p=[.35, .17, .11, .17, .20])
    bip = desc == "hit_into_play"
    ev = np.clip(0.8 + rng.normal(0, .1, n), .3, 1) * (1.23 * bs + .23 * ps)
    return pd.DataFrame({"game_pk": rng.integers(0, 2000, n), "game_date": f"{year}-06-01", "game_type": "R",
                         "batter": h + 1000, "description": desc,
                         "type": np.where(bip, "X", np.where(desc == "ball", "B", "S")),
                         "launch_speed": np.where(bip, ev, np.nan),
                         "launch_angle": np.where(bip, rng.normal(14, 25, n), np.nan), "release_speed": ps,
                         "bat_speed": np.where(np.isin(desc, ["swinging_strike", "foul", "hit_into_play"]) & tracked,
                                               bs, np.nan), "swing_length": 7.2})


def test_power_grade_finds_the_slugger_and_regresses_small_samples():
    s = _season(2025)
    ref = power.build_ref(s, 2025)
    assert "sd_bat" in ref and "sd_sc" in ref
    big = power.score(s[s.batter == 1000], ref)
    assert 50 + 10 * big["_POWER_Z"] > 65 and 50 + 10 * big["_SCBS_Z"] > 65
    assert big["_POWER_KIND"] == "bat speed + exit velo" and "Avg Bat Speed" in big["_POWER_PARTS"]
    few = s[(s.batter == 1000)].head(25)                       # a few games
    small = power.score(few, ref)
    assert abs(small["_POWER_Z"]) < abs(big["_POWER_Z"])       # pulled toward average


def test_no_bat_tracking_falls_back_to_exit_velo():
    s = _season(2019, tracked=False, seed=1)
    ref = power.build_ref(s, 2019)
    assert "sd_bat" not in ref and "sd_ev" in ref
    g = power.score(s[s.batter == 1000], ref)
    assert g["_POWER_KIND"] == "exit velo only" and g["_POWER_Z"] > 1 and g["_SCBS_Z"] != g["_SCBS_Z"]


def test_ref_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(power, "REF_DIR", tmp_path)
    ref = power.build_ref(_season(2024, n_hitters=60), 2024)
    power.save_ref(ref)
    assert power.load_ref(2024)["stats"].keys() == ref["stats"].keys()
