import numpy as np
import pandas as pd

from models import fielding as fm


def test_innings_from_outs_and_drs_parts():
    df = pd.DataFrame({"game_date": "2025-05-01", "game_pk": 1, "at_bat_number": [1, 2, 3, 4],
                       "inning": 1, "inning_topbot": "Top", "outs_when_up": [0, 1, 1, 2],
                       "events": ["field_out", "single", "double_play", "strikeout"],
                       **{f"fielder_{k}": 100 + k for k in range(2, 10)}})
    inn = fm.innings(df)
    assert inn["outs"].unique().tolist() == [3.0] and set(inn["pos"]) == set(range(2, 10))
    daily = pd.DataFrame({"game_date": ["2025-05-01"] * 2, "game_pk": 1, "player": [7, 6], "pos": [7, 6],
                          "outs": [27, 27], "n": [3, 4], "opa": [0.5, -0.2], "runs": [0.4, -0.16], "fpa": [0.1, 0],
                          "dp_opps": [0, 1], "dp_runs": [0, 0.3], "arm_opps": [1, 0], "arm_runs": [0.2, 0],
                          "b_opps": 0})
    of = fm.summarize(daily, "2025-01-01", "2025-12-31", [7])[7]
    inf = fm.summarize(daily, "2025-01-01", "2025-12-31", [6])[6]
    assert np.isclose(of["DRS"], 0.6) and np.isclose(inf["DRS"], 0.14) and of["IQ_INN"] == 9


def test_player_at_position():
    d = pd.DataFrame({f"fielder_{k}": [10 * k, 20 * k] for k in range(2, 10)})
    assert fm.player_at(d, pd.Series([6, 9])).tolist() == [60, 180]


def test_compare_name_matching_and_stats():
    from pipeline import compare_fielding as cf
    assert cf.norm("Ceddanne Rafaela") == cf.norm("ceddanne  rafaela")
    assert cf.norm("Ronald Acuña Jr.") == "ronald acuna"
    a = pd.Series(np.arange(20.0)); b = a * 2
    r, slope, n = cf._stats(a, b)
    assert round(r, 6) == 1 and round(slope, 6) == 0.5 and n == 20


def test_distance_only_for_balls_in_the_air():
    f = fm.features(pd.DataFrame({"launch_speed": [100, 100], "launch_angle": [5, 25], "hit_distance_sc": [120, 380],
                                  "hc_x": [125, 125], "hc_y": [100, 50], "events": ["single", "field_out"]}))
    assert f.loc[0, "dist"] == -1 and f.loc[1, "dist"] == 380


def test_range_plus_conversion_equals_opa_end_to_end():
    import numpy as np
    import pandas as pd
    from models import fielding as fm
    from pipeline import train_fielding as tf
    rng = np.random.default_rng(0)
    n = 6000
    la = rng.uniform(-30, 60, n)
    hcx = rng.uniform(60, 190, n)
    loc = np.where(la < 10, np.where(hcx < 100, 5, np.where(hcx < 125, 6, np.where(hcx < 150, 4, 3))),
                   np.where(hcx < 105, 7, np.where(hcx < 145, 8, 9)))
    u = rng.random(n)
    ferr = (la < 10) & (u < 0.03)
    out = ~ferr & (u < 0.72)
    ev = np.where(ferr, "field_error", np.where(out, "field_out", "single"))
    des = np.where(ferr, "reaches on a fielding error by shortstop", np.where(out, "groundout", "single"))
    d = pd.DataFrame({"game_date": "2025-06-01", "game_pk": rng.integers(0, 400, n), "at_bat_number": 1,
                      "inning": 1, "inning_topbot": "Top", "outs_when_up": rng.integers(0, 3, n), "events": ev,
                      "des": des, "type": "X", "description": "hit_into_play", "hit_location": loc,
                      "launch_speed": rng.uniform(60, 110, n), "launch_angle": la,
                      "hit_distance_sc": rng.uniform(50, 400, n), "hc_x": hcx, "hc_y": rng.uniform(80, 180, n),
                      "stand": "R", "on_1b": np.where(rng.random(n) < .3, 1, None), "on_2b": None, "on_3b": None,
                      "delta_run_exp": np.where(out, -0.25, 0.4), "game_type": "R",
                      **{f"fielder_{k}": 100 + k for k in range(2, 10)}})
    b = tf.fit(d)
    assert b["conv"] is not None and b["ferr"] is not None
    pc = fm.play_credits(d, b)
    p_ = pc.dropna(subset=["opa"])                                         # DP / arm rows carry no OPA
    assert np.allclose(p_["rng"] + p_["conv"], p_["opa"])
    assert pc.loc[pc.pos.isin([7, 8, 9]), "conv"].abs().sum() == 0          # outfield: no conversion split
    s = fm.summarize(fm.daily_table(d, b), "0000", "9999", [6])[106]
    assert s["_CONV_N"] > 0 and abs(s["_RNG"] + s["_CONV"] - s["OPA"]) < 1e-6
