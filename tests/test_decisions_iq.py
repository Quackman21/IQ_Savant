import numpy as np
import pandas as pd

from models import decisions_iq as dq


def test_actions_fouls_bunts_and_clock():
    df = pd.DataFrame({"description": ["foul_tip", "foul", "hit_into_play", "hit_into_play", "foul_bunt",
                                       "automatic_ball", "ball", "called_strike", "hit_by_pitch"],
                       "des": ["", "", "lines out", "sacrifice bunt", "", "", "", "", ""]})
    assert dq.action(df).tolist() == ["swing", "swing", "swing", None, None, None, "take", "take", "take"]


def _fake(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    balls, strikes = rng.integers(0, 4, n), rng.integers(0, 3, n)
    x, z = rng.normal(0, 0.9, n), rng.normal(2.5, 0.9, n)
    dist = np.maximum(np.abs(x) - 0.83, 0) + np.maximum(1.6 - z, 0) + np.maximum(z - 3.4, 0)
    inz = dist == 0
    sw = rng.random(n) < np.where(inz, 0.65, 0.3)
    u = rng.random(n)
    miss = np.clip(0.1 + 0.4 * dist, 0, 0.9)
    desc = np.where(sw, np.where(u < miss, "swinging_strike", np.where(u < miss + 0.3, "foul", "hit_into_play")),
                    np.where(rng.random(n) < np.where(inz, 0.92, 0.08), "called_strike", "ball"))
    xq = np.clip(0.45 - 0.3 * dist + rng.normal(0, 0.05, n), 0.02, 1.2)
    V = 0.03 * balls - 0.04 * strikes
    rv = np.select([desc == "ball", np.isin(desc, ["called_strike", "swinging_strike"]) | ((desc == "foul") & (strikes < 2)),
                    desc == "foul"],
                   [np.where(balls == 3, 0.33 - V, 0.04), np.where(strikes == 2, -0.28 - V, -0.05), 0.0],
                   (xq - 0.36) * 1.1 - V)
    return pd.DataFrame({"game_date": "2025-06-01", "game_pk": rng.integers(1, 500, n), "batter": 1,
                         "stand": "R", "p_throws": "R", "balls": balls, "strikes": strikes,
                         "outs_when_up": rng.integers(0, 3, n), "on_1b": np.nan, "on_2b": np.nan,
                         "on_3b": np.where(rng.random(n) < 0.2, 1.0, np.nan), "description": desc, "des": "",
                         "type": np.where(desc == "hit_into_play", "X", "S"), "plate_x": x, "plate_z": z,
                         "sz_top": 3.4, "sz_bot": 1.6, "release_speed": 92.0, "pfx_x": 0.0, "pfx_z": 1.0,
                         "pitch_type": "FF", "delta_run_exp": -rv,
                         "estimated_woba_using_speedangle": np.where(desc == "hit_into_play", xq, np.nan),
                         "woba_value": 0.0})


def test_decisions_make_baseball_sense_and_hierarchy_holds():
    from pipeline import train_decisions as t
    df = _fake()
    b = t.fit(df)
    assert b["tables"]["count"]["ball"]["3-2"] > 0.2 and b["tables"]["count"]["strike"]["3-2"] < -0.2
    probe = df.iloc[:4].copy()
    probe[["plate_x", "plate_z", "balls", "strikes"]] = [[0, 2.5, 1, 1], [0, 2.5, 1, 1], [1.8, 0.6, 0, 0], [1.8, 0.6, 0, 0]]
    probe["description"] = ["hit_into_play", "called_strike", "swinging_strike", "ball"]
    g = dq.score(probe, b)["good"].tolist()
    assert g == [1.0, 0.0, 0.0, 1.0]            # swing middle-middle good, take it bad; chase dirt bad, take it good
    sc = dq.score(df, b)
    flipped = sc[sc["flipped"] == 1]
    assert (flipped["z_core"].abs() < dq.CLEAR).all()   # runners / outs never overrule a clear call


def test_two_strike_foul_has_value_but_not_before():
    from pipeline import train_decisions as t
    b = t.fit(_fake(seed=1))
    p = pd.Series([0.0, 0.0]); st = pd.DataFrame({"count": ["1-1", "1-2"], "bo": ["1-1|0|0", "1-2|0|0"],
                                                 "two_k": [False, True]})
    one = pd.Series([1.0, 1.0])
    _, swing, _ = dq._values(st, b, "count", p, p, one, p, p)       # a swing that is always fouled
    assert swing[0] == b["tables"]["count"]["strike"]["1-1"] and swing[1] == 0.0


def test_iq_framing_extra_strikes_slaa_and_runs():
    from models import framing
    from pipeline import train_decisions as t
    b = t.fit(_fake(seed=3))
    dq._cache["m"] = b
    try:
        rng = np.random.default_rng(5)
        n = 30000
        df = _fake(n, seed=6)
        df = df[df["description"].isin(["called_strike", "ball"])].copy()
        df["fielder_2"] = rng.choice([1, 2], len(df))
        df["game_pk"] = rng.integers(1, 60, len(df))
        p = dq.called_strike_chance(df)
        # catcher 1 steals strikes on borderline pitches
        steal = (df["fielder_2"] == 1) & (df["description"] == "ball") & p.between(0.2, 0.8) & (rng.random(len(df)) < 0.4)
        df.loc[steal, "description"] = "called_strike"
        fp = framing.framing_pitches(df, "iq_v1")
        assert {"runs", "exp"} <= set(fp.columns) and fp["exp"].between(0, 1).all()
        one = fp.iloc[0]
        assert abs((one["strike"] - one["exp"]) - ((1.0 if one["strike"] else 0.0) - one["exp"])) < 1e-12
        s = framing.summarize(framing.daily_table(fp), "2025-01-01", "2025-12-31")
        c1, c2 = s[1], s[2]
        assert c1["F_XS"] > c2["F_XS"] and c1["F_SLAA+"] > c2["F_SLAA+"] and c1["F_RS"] > c2["F_RS"]
        assert c1["F_XS/G"] > c2["F_XS/G"] and abs(c1["F_SLAA"] - c1["F_XS"] / c1["F_OPPS"]) < 1e-12
        assert abs(c1["F_SLAA+"] - 100 * (c1["F_XS"] + fp[fp.catcher == 1]["exp"].sum()) / fp[fp.catcher == 1]["exp"].sum()) < 1e-9
        sv = dq.strike_value(df.iloc[:3])
        assert (sv > 0).all()                                   # a strike instead of a ball always helps the defense
    finally:
        dq._cache.clear()
