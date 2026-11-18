import numpy as np
import pandas as pd

from core import abs_stats as ab


def _raw():
    """Two games. Game 1: hitter 5 challenges a strike (overturned), catcher 7 a ball (stood).
    Game 2: the LAST pitch -- catcher 7 challenges a ball way outside (stood -> not counted)."""
    rows = []
    def p(g, abn, pn, desc, b, s, px, pz, ev=None, bat=5):
        rows.append({"game_pk": g, "game_date": f"2026-06-0{g}", "at_bat_number": abn, "pitch_number": pn,
                     "description": desc, "balls": b, "strikes": s, "plate_x": px, "plate_z": pz, "sz_top": 3.4,
                     "sz_bot": 1.6, "batter": bat, "pitcher": 99, "fielder_2": 7, "events": ev, "inning": 1,
                     "inning_topbot": "Top", "stand": "R", "p_throws": "R", "pitch_type": "FF", "release_speed": 95.0,
                     "home_team": "BOS", "away_team": "NYY", "delta_run_exp": 0.0, "outs_when_up": 0})
    p(1, 1, 1, "ball", 0, 0, 0.75, 2.5)                  # was a strike, hitter overturned it -> ball
    p(1, 1, 2, "ball", 1, 0, 1.5, 2.5)                  # catcher challenged, ball stood
    p(1, 1, 3, "called_strike", 2, 0, 0.0, 2.5, "")
    p(1, 1, 4, "called_strike", 2, 1, 0.0, 2.5)
    p(2, 1, 1, "called_strike", 0, 0, 0.0, 2.5)
    p(2, 1, 2, "ball", 0, 1, 2.2, 2.5, "")              # last pitch of the game, ~16" outside
    return pd.DataFrame(rows)


def test_roles_results_and_the_last_pitch_rule():
    ch = pd.DataFrame({"game_pk": [1, 1, 2], "at_bat_number": [1, 1, 1], "pitch_number": [1, 2, 2],
                       "overturned": [True, False, False], "challenger": [5, 7, 7]})
    t = ab.table(_raw(), ch, ab.DEFAULT_VALUES)
    assert t["role"].tolist() == ["Hitting", "Catching", "Catching"]
    assert t["result"].tolist() == ["Successful", "Failed", "Failed"]
    assert t[["from_call", "to_call"]].values.tolist() == [["Strike", "Ball"], ["Ball", "Ball"], ["Ball", "Ball"]]
    assert t["counted"].tolist() == [True, True, False] and t["final_pitch"].tolist() == [False, False, True]
    assert abs(t["runs"].iloc[0] - ab.DEFAULT_VALUES["0-0"]) < 1e-9 and t["runs"].iloc[1:].eq(0).all()
    assert t["margin_in"].iloc[2] > 3
    pl = ab.players(_raw(), t)
    c7 = pl[(pl["player"] == 7) & (pl["role"] == "Catching")].iloc[0]
    assert c7["games"] == 2 and c7["challenges"] == 1 and c7["not_counted"] == 1
    s = ab.summarize(2, t[t["role"].eq("Catching")])
    assert s["Challenges"] == 1 and s["Not counted"] == 1 and s["Overturn %"] == 0


def test_margin_inside_and_outside():
    df = pd.DataFrame({"plate_x": [0.0, 1.0, 0.0], "plate_z": [2.5, 2.5, 4.0], "sz_top": 3.4, "sz_bot": 1.6})
    m = ab.margin_in(df)
    assert m.iloc[0] < 0 and abs(m.iloc[1] - (12 - 9.95)) < 0.01 and abs(m.iloc[2] - (48 - 40.8 - 1.45)) < 0.01


def test_count_values_from_the_season():
    rng = np.random.default_rng(0)
    n = 4000
    raw = pd.DataFrame({"description": rng.choice(["ball", "called_strike"], n), "balls": 3, "strikes": 2})
    raw["delta_run_exp"] = np.where(raw["description"].eq("ball"), 0.3, -0.25)
    v = ab.count_values(raw)
    assert abs(v["3-2"] - 0.55) < 1e-9 and v["0-0"] == ab.DEFAULT_VALUES["0-0"]


def test_failed_challenge_costs_that_games_missed_calls_blended_to_the_season():
    misses = pd.DataFrame({"game_pk": [1, 2], "against": ["BOS", "NYY"], "value": [0.2, 0.1]})
    costs, avg = ab.game_costs(misses)
    assert abs(avg - 0.15) < 1e-9 and abs(costs[(1, "BOS")] - (0.2 + 3 * 0.15) / 4) < 1e-9
    ch = pd.DataFrame({"game_pk": [1, 1, 2], "at_bat_number": [1, 1, 1], "pitch_number": [1, 2, 2],
                       "overturned": [True, False, False], "challenger": [5, 7, 7]})
    t = ab.table(_raw(), ch, ab.DEFAULT_VALUES, costs, avg, misses)
    # hitter overturned: no cost; catcher (BOS fielding) failed mid-game: game 1's BOS value; last pitch: free
    assert t["cost"].iloc[0] == 0 and abs(t["cost"].iloc[1] - costs[(1, "BOS")]) < 1e-9 and t["cost"].iloc[2] == 0
    assert abs(t["net"].iloc[1] + costs[(1, "BOS")]) < 1e-9
    s = ab.summarize(2, t[t["role"].eq("Catching")])
    assert abs(s["Net ABS Runs"] + costs[(1, "BOS")]) < 1e-9


def test_missed_calls_by_the_abs_zone():
    raw = pd.DataFrame({"description": ["called_strike", "ball", "ball", "called_strike"],
                        "plate_x": [1.5, 0.0, 1.5, 0.0], "plate_z": [2.5] * 4, "sz_top": 3.4, "sz_bot": 1.6,
                        "balls": 0, "strikes": 0, "game_pk": 1, "inning_topbot": "Top", "home_team": "BOS",
                        "away_team": "NYY"})
    m = ab.missed_calls(raw)
    assert m["against"].tolist() == ["NYY", "BOS"]          # bad strike on the hitters; ball in the zone on the pitcher
