import pandas as pd

from core import pitching


def _df():
    rows = [  # pitch_name, desc, type, events, zone, velo, stand, ev, woba_value, woba_denom
        ("4-Seam Fastball", "swinging_strike", "S", None, 5, 97.0, "R", None, None, None),
        ("4-Seam Fastball", "called_strike", "S", None, 4, 96.0, "L", None, None, None),
        ("4-Seam Fastball", "swinging_strike", "S", "strikeout", 12, 98.0, "R", None, 0.0, 1),
        ("Slider", "hit_into_play", "X", "home_run", 5, 86.0, "L", 105.0, 2.0, 1),
        ("Slider", "ball", "B", "walk", 13, 85.0, "R", None, 0.7, 1),
    ]
    df = pd.DataFrame(rows, columns=["pitch_name", "description", "type", "events", "zone", "release_speed",
                                     "stand", "launch_speed", "woba_value", "woba_denom"])
    df["pitch_type"] = df["pitch_name"].map({"4-Seam Fastball": "FF", "Slider": "SL"})
    df["pfx_z"], df["pfx_x"] = 1.4, -0.5
    return df


def test_overview():
    o = pitching.overview(_df())
    assert (o["Pitches"], o["BF"], o["K%"], o["BB%"]) == (5, 3, "33.3", "33.3")
    assert o["Whiff%"] == "66.7" and o["FB Velo"] == "97.0"


def test_arsenal_and_results():
    a = pitching.arsenal(_df())
    assert list(a) == ["4-Seam Fastball", "Slider"]
    assert a["4-Seam Fastball"]["Usage%"] == "60.0" and a["4-Seam Fastball"]["IVB"] == "16.8"
    r = pitching.results(_df())
    assert r["Slider"]["HR"] == 1 and r["Slider"]["AVG"] == "1.000"


def test_platoon_and_ip():
    p = pitching.by_platoon(_df(), pitching.overview)
    assert p["vs LHB"]["Pitches"] == 2
    assert pitching._ip_outs("123.2") == 371
