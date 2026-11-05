import pandas as pd

from core import hitting


def test_stat_line_and_rates():
    s = {"gamesPlayed": 10, "plateAppearances": 40, "atBats": 35, "hits": 10, "doubles": 2,
         "triples": 0, "homeRuns": 3, "baseOnBalls": 4, "hitByPitch": 1, "sacFlies": 0,
         "strikeOuts": 8, "stolenBases": 3, "caughtStealing": 1}
    line = hitting.stat_line(s)
    assert line["AVG"] == ".286" and line["OBP"] == ".375" and line["SLG"] == ".600" and line["OPS"] == ".975"
    r = hitting.rates(s)
    assert r["K%"] == "20.0" and r["BB%"] == "10.0" and r["ISO"] == ".314" and r["SB%"] == "75.0"
    assert r["BABIP"] == f"{(10 - 3) / (35 - 8 - 3):.3f}".replace("0.", ".")


def _pitches():
    rows = [
        # desc, zone, type, events, ev, la, lsa, stand, p_throws, hc_x, hc_y, bb_type, bat_speed
        ("swinging_strike", 5, "S", None, None, None, None, "R", "R", None, None, None, 72.0),
        ("hit_into_play", 5, "X", "home_run", 108.0, 28, 6, "R", "R", 60.0, 80.0, "fly_ball", 78.0),
        ("ball", 13, "B", None, None, None, None, "R", "L", None, None, None, None),
        ("foul", 12, "S", None, None, None, None, "R", "L", None, None, None, 70.0),
        ("swinging_strike", 14, "S", "strikeout", None, None, None, "R", "L", None, None, None, 76.0),
        ("hit_into_play", 6, "X", "field_out", 92.0, -5, 1, "R", "R", 190.0, 150.0, "ground_ball", 74.0),
    ]
    cols = ["description", "zone", "type", "events", "launch_speed", "launch_angle", "launch_speed_angle",
            "stand", "p_throws", "hc_x", "hc_y", "bb_type", "bat_speed"]
    df = pd.DataFrame(rows, columns=cols)
    df["swing_length"] = 7.2
    df["woba_value"] = [None, 2.0, None, None, 0.0, 0.0]
    df["woba_denom"] = [None, 1, None, None, 1, 1]
    df["estimated_woba_using_speedangle"] = [None, 1.8, None, None, None, 0.2]
    df["game_date"] = "2026-06-01"
    df["at_bat_number"] = range(len(df))
    return df


def test_contact_exit_spray_core4():
    df = _pitches()
    c = hitting.contact(df)
    assert c["Swing%"] == "83.3" and c["Whiff%"] == "40.0" and c["Chase%"] == "66.7"
    e = hitting.exit_velo(df)
    assert e["BBE"] == 2 and e["Max EV"] == "108.0" and e["Barrel%"] == "50.0" and e["HardHit%"] == "50.0"
    assert e["wOBA"] == ".667" and e["xwOBA"] == ".667"
    s = hitting.spray(df)
    assert s["Pull%"] == "50.0" and s["Oppo%"] == "50.0" and s["GB%"] == "50.0"
    groups = hitting.core4(df)
    assert set(groups) == {"Core 4", "Bat Speed", "Bat-To-Ball", "Swing Decisions", "Ball Flight"}
    assert groups["Bat Speed"]["Fast Swing%"] == "40.0"
    plat = hitting.by_platoon(df, hitting.contact)
    assert plat["vs LHP"]["Pitches"] == 3


def test_empty_and_old_seasons():
    assert hitting.contact(pd.DataFrame()) == {}
    old = _pitches().drop(columns=["bat_speed", "swing_length"])
    assert hitting.bat_speed(old)["Bat Speed"] == "-"
