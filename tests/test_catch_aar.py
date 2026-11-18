import numpy as np
import pandas as pd

from core import catch_aar as ca


def _kin(z50=5.0, vz0=-6.0, az=-26.0, vy0=-130.0, ay=27.0, x50=0.5, vx0=-4.0, ax=-8.0):
    """A pitch defined at y = 50 ft; plate_x / plate_z worked out from it (front of the plate)."""
    a, b, c = 0.5 * ay, vy0, 50 - 17 / 12
    t = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return {"vx0": vx0, "vy0": vy0, "vz0": vz0, "ax": ax, "ay": ay, "az": az,
            "plate_x": x50 + vx0 * t + 0.5 * ax * t * t, "plate_z": z50 + vz0 * t + 0.5 * az * t * t}


def test_ground_point_matches_brute_force_flight():
    k = _kin(z50=4.6, vz0=-8.0)                       # a ball in the dirt
    g = ca.ground(pd.DataFrame([k])).iloc[0]
    ts = np.linspace(0, 0.6, 60001)
    z = 4.6 + k["vz0"] * ts + 0.5 * k["az"] * ts ** 2
    i = np.argmax(z <= ca.BALL_R_FT)
    y = 50 + k["vy0"] * ts[i] + 0.5 * k["ay"] * ts[i] ** 2
    assert g["dirt"] and abs(g["gy"] - (y - 17 / 12)) < 0.02
    hi = ca.ground(pd.DataFrame([_kin(z50=6.5, vz0=-3.0)])).iloc[0]      # up in the zone: the catcher has it first
    assert not hi["dirt"]


def test_flight_ends_in_the_dirt_or_three_feet_past_the_middle():
    low = ca.flight(_kin(z50=4.6, vz0=-8.0))
    assert abs(low[-1][2] - ca.BALL_R_FT * 12) < 0.05 and low[0][1] > 600         # release ~54 ft out
    high = ca.flight(_kin(z50=6.5, vz0=-3.0))
    assert abs(high[-1][1] - (ca.END_FT - 17 / 12) * 12) < 0.1 and high[-1][2] > 6


def test_zone_names_are_from_the_catchers_side():
    assert ca.gameday_name(14, "R") == "14 · Down arm side (out)"        # catcher's right = his arm side
    assert ca.gameday_name(14, "L") == "14 · Down arm side (out)"        # whoever is pitching
    assert ca.gameday_name(1, "R") == "1 · Up-glove" and ca.gameday_name(5, "R") == "5 · Middle-middle"
    df = pd.DataFrame({"plate_x": [0.0, -0.75, 0.75, 0.0, -1.3], "plate_z": [2.5, 2.5, 1.55, 3.95, 0.5],
                       "sz_top": 3.5, "sz_bot": 1.5, "p_throws": ["R", "R", "R", "L", "R"]})
    assert ca.region(df).tolist() == ["Heart", "Glove-side shadow", "Low arm-side shadow", "High chase",
                                      "Low glove-side waste"]


def _feed():
    return {"liveData": {"boxscore": {"teams": {"home": {"players": {
        "ID7": {"stats": {"batting": {"plateAppearances": 4}}, "allPositions": [{"abbreviation": "C"}]},
        "ID8": {"stats": {"batting": {}}, "allPositions": [{"abbreviation": "C"}, {"abbreviation": "1B"}]}}}}},
        "plays": {"allPlays": [{"about": {"atBatIndex": 0}, "playEvents": [
            {"isPitch": True, "pitchNumber": 1, "index": 0, "playId": "v1"},
            {"isPitch": True, "pitchNumber": 2, "index": 1, "playId": "v2"},
            {"isPitch": False, "index": 2}],
            "runners": [{"movement": {"start": "1B", "end": "2B"},
                         "details": {"eventType": "wild_pitch", "playIndex": 2, "runner": {"id": 3}}}]}]}},
        "gameData": {"players": {"ID99": {"id": 99, "fullName": "Arm Guy"}}}}


def test_reports_and_pitches_leave_out_balls_in_play():
    f = _feed()
    assert ca.available(f, 7) == ["Hitting", "Catching"]
    assert ca.available(f, 8) == ["Catching", "Infield"]
    rows = []
    for pn, desc, typ, z50 in ((1, "ball", "B", 6.0), (2, "blocked_ball", "B", 4.4), (3, "hit_into_play", "X", 6.0)):
        k = _kin(z50=z50, vz0=-8.0 if z50 < 5 else -5.0)
        rows.append({**k, "game_pk": 1, "game_date": "2025-06-01", "at_bat_number": 1, "pitch_number": pn,
                     "description": desc, "type": typ, "fielder_2": 7, "pitcher": 99, "batter": 5, "p_throws": "R",
                     "stand": "R", "pitch_type": "SL", "release_speed": 86.0, "pfx_x": 0.3, "pfx_z": 0.1,
                     "sz_top": 3.4, "sz_bot": 1.6, "balls": pn - 1, "strikes": 0, "outs_when_up": 0, "inning": 1,
                     "inning_topbot": "Top", "on_1b": 3.0, "on_2b": np.nan, "on_3b": np.nan, "zone": 14,
                     "events": None})
    sc = pd.DataFrame(rows)
    ev = ca.runner_events(f, 1)
    p = ca.pitches(sc, f, 7, ev)
    assert len(p) == 2 and p["pitcher"].tolist() == ["Arm Guy"] * 2 and p["video"].tolist() == ["v1", "v2"]
    assert p["opp"].all() and p["got_by"].tolist() == [0.0, 1.0] and p["play"].tolist() == ["", "Wild pitch"]
    assert p["on"].tolist() == ["1B", "1B"] and p.loc[1, "dirt"] and not p.loc[0, "dirt"]
    assert p["zone"].iloc[0] == "14 · Down arm side (out)"
    sc3 = ca.scenes(p)
    assert set(sc3) == {1, 2} and sc3[2]["dirt"]
    from ui import catch_aar as ui
    html = ui.component_html(p, ca.line(p, ev, ca.caught(sc, 7)), scenes3d=sc3, has_model=False)
    assert 'id="rows-data"' in html and "Where it hit the dirt" in html and "Blocking opportunities" in html
    assert ca.line(p, ev, ca.caught(sc, 7))["WP"] == 1


def test_hover_comps_are_per_game_and_rates_pooled():
    fr = pd.DataFrame({"game_date": ["2026-05-01", "2026-05-01", "2026-06-20", "2026-06-20", "2026-06-20"],
                       "game_pk": [1, 1, 2, 2, 2], "catcher": [7, 7, 7, 7, 9], "region": ["IN", "TOZ", "IN", "BOZ", "IN"],
                       "n": [30, 10, 40, 20, 50], "strikes": [20, 5, 30, 8, 30], "exp": [21, 4, 29, 9, 31],
                       "runs": [-.1, .1, .1, -.1, -.1]})
    fl = pd.DataFrame({"game_date": ["2026-05-01", "2026-06-20", "2026-06-20"], "game_pk": [1, 2, 2],
                       "player": [7, 7, 9], "pos": 2, "c_n": [10, 6, 8], "c_exp": [.5, .3, .4], "c_act": [0, 1, 0],
                       "c_runs": [.1, -.2, .1], "g_n": [2, 1, 3], "tough_n": [1, 0, 1], "tough_blk": [1, 0, 0],
                       "t_n": [1, 2, 0], "t_exp": [.2, .5, 0], "t_cs": [1, 0, 0], "t_runs": [.4, -.3, 0]})
    c = ca.comps(2026, 7, "2026-06-25", fr, fl)
    f = dict(c["framing"]["Takes"])
    assert f["Season"] == "50.0" and f["Last 30 days"] == "60.0" and f["MLB avg"] == "50.0"
    assert dict(c["framing"]["Shadow strike %"])["Season"] == "43%"            # 13 of 30 edge takes
    assert dict(c["throwing"]["CS %"])["Season"] == "33%" and dict(c["blocking"]["Opportunities"])["Season"] == "8.0"
    assert dict(c["total"]["Catcher runs"])["Season"] == "0.00"
    assert dict(c["framing"]["_games"]) == {"Season": 2, "Last 30 days": 1}


def test_framing_zone_split_matches_savant():
    df = pd.DataFrame({"plate_x": [0.0, 0.0, 0.0, -0.75, 0.75, 0.75, -0.6, 1.6],
                       "plate_z": [2.5, 3.5, 1.5, 2.5, 2.5, 3.6, 1.6, 2.5], "sz_top": 3.5, "sz_bot": 1.5})
    assert ca.framing_zone(df).tolist() == ["Heart", "Shadow - Top", "Shadow - Bottom", "Shadow - Left",
                                            "Shadow - Right", "Shadow - Top Right", "Shadow - Bottom Left",
                                            "Chase / Waste"]


def test_catcher_abs_challenges_successful_or_failed():
    plays = {"allPlays": [{"about": {"atBatIndex": 0}, "playEvents": [
        {"isPitch": True, "pitchNumber": 1, "index": 0, "details": {"call": {"code": "C"}},
         "reviewDetails": {"isOverturned": True, "player": {"id": 7}, "reviewType": "MJ"}},
        {"isPitch": True, "pitchNumber": 2, "index": 1, "details": {"call": {"code": "B"}},
         "reviewDetails": {"isOverturned": False, "player": {"id": 7}}},
        {"isPitch": True, "pitchNumber": 3, "index": 2, "details": {"call": {"code": "B"}},
         "reviewDetails": {"isOverturned": False, "player": {"id": 5}}},
        {"isPitch": True, "pitchNumber": 4, "index": 3}]}]}
    feed = {"liveData": {"plays": plays}, "gameData": {"players": {}}}
    rows = []
    for pn, desc, b, s in ((1, "called_strike", 0, 0), (2, "ball", 0, 1), (3, "ball", 1, 1), (4, "ball", 2, 1)):
        rows.append({**_kin(), "game_pk": 1, "game_date": "2026-06-01", "at_bat_number": 1, "pitch_number": pn,
                     "description": desc, "type": "S" if desc == "called_strike" else "B", "fielder_2": 7,
                     "pitcher": 99, "batter": 5, "p_throws": "R", "stand": "R", "pitch_type": "FF", "balls": b,
                     "strikes": s, "inning": 1, "inning_topbot": "Top", "sz_top": 3.4, "sz_bot": 1.6, "zone": 5,
                     "on_1b": np.nan, "on_2b": np.nan, "on_3b": np.nan, "events": None})
    p = ca.pitches(pd.DataFrame(rows), feed, 7, ca.runner_events(feed, 1))
    assert p["abs"].tolist() == ["Successful", "Failed", "", ""]
    assert p["abs_by"].tolist() == ["Catcher", "Catcher", "Batter", ""]
    assert p["orig_call"].tolist() == ["Ball", "", "", ""]          # overturned: the umpire had called it a ball
    assert ca.line(p, None)["ABS"] == "1/2"
    fl = pd.DataFrame({"game_date": ["2026-06-01", "2026-06-01"], "game_pk": [1, 1], "player": [7, 8], "pos": 2})
    ab = pd.DataFrame({"game_pk": [1, 1, 1], "challenger": [7, 7, 5], "overturned": [True, False, False]})
    v = ca.abs_values(fl, ab, 7)["abs"]
    assert v["Challenges"] == 2 and v["Success %"] == 0.5 and ca.abs_values(fl, ab, None)["abs"]["Challenges"] == 1
