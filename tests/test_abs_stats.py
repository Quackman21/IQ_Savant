import numpy as np
import pandas as pd

from core import abs_stats as ab


def _raw():
    """Two games. Game 1: hitter 5 challenges a strike (overturned), catcher 7 a ball (stood).
    Game 2: the LAST pitch -- catcher 7 challenges a ball way outside (stood -> not counted)."""
    rows = []

    def p(g, abn, pn, desc, b, s, px, pz, ev=None, bat=5, inn=1, outs=0):
        rows.append({"game_pk": g, "game_date": f"2026-06-0{g}", "at_bat_number": abn, "pitch_number": pn,
                     "description": desc, "balls": b, "strikes": s, "plate_x": px, "plate_z": pz, "sz_top": 3.4,
                     "sz_bot": 1.6, "batter": bat, "pitcher": 99, "fielder_2": 7, "events": ev, "inning": inn,
                     "inning_topbot": "Top", "stand": "R", "p_throws": "R", "pitch_type": "FF", "release_speed": 95.0,
                     "home_team": "BOS", "away_team": "NYY", "delta_run_exp": 0.0, "outs_when_up": outs,
                     "on_1b": np.nan, "on_2b": np.nan, "on_3b": np.nan, "bat_score": 0, "post_bat_score": 0})
    p(1, 1, 1, "ball", 0, 0, 0.75, 2.5)                  # was a strike, hitter overturned it -> ball
    p(1, 1, 2, "ball", 1, 0, 1.5, 2.5)                  # catcher challenged, ball stood
    p(1, 1, 3, "called_strike", 2, 0, 0.0, 2.5, "")
    p(1, 1, 4, "ball", 2, 1, 0.0, 2.5)                  # a missed call against BOS's pitcher (ball in the zone)
    p(2, 1, 1, "called_strike", 0, 0, 0.0, 2.5, inn=9, outs=2)
    p(2, 1, 2, "ball", 0, 1, 2.2, 2.5, "", inn=9, outs=2)  # last pitch of the game, ~16" outside
    return pd.DataFrame(rows)


CH = pd.DataFrame({"game_pk": [1, 1, 2], "at_bat_number": [1, 1, 1], "pitch_number": [1, 2, 2],
                   "overturned": [True, False, False], "challenger": [5, 7, 7]})


def test_roles_results_and_the_last_pitch_rule():
    t = ab.table(_raw(), CH)
    assert t["role"].tolist() == ["Hitting", "Catching", "Catching"]
    assert t["result"].tolist() == ["Successful", "Failed", "Failed"]
    assert t[["from_call", "to_call"]].values.tolist() == [["Strike", "Ball"], ["Ball", "Ball"], ["Ball", "Ball"]]
    assert t["counted"].tolist() == [True, True, False] and t["final_pitch"].tolist() == [False, False, True]
    assert t["team"].tolist() == ["NYY", "BOS", "BOS"] and t["margin_in"].iloc[2] > 3


def test_outs_left_in_the_game():
    df = pd.DataFrame({"inning": [1, 1, 9, 9, 10], "inning_topbot": ["Top", "Bot", "Top", "Bot", "Top"],
                       "outs_when_up": [0, 0, 2, 2, 1]})
    assert ab.outs_left(df).tolist() == [54, 51, 4, 1, 5]


def test_run_values_follow_the_game_state():
    re = {f"{b}|{o}|{ba}|{s}": 0.5 for b in range(8) for o in range(3) for ba in range(4) for s in range(3)}
    # 3-2, bases loaded, 2 out: ball = a run in + the same situation; strike = inning over
    assert abs(ab.state_value(7, 2, 3, 2, re) - 1.5) < 1e-9
    # 0-0, bases empty: ball and strike lead to states worth the same here -> 0
    assert ab.state_value(0, 0, 0, 0, re) == 0.0


def test_challenges_left_two_per_game_and_one_back_in_extras():
    df = pd.DataFrame({"game_pk": 1, "at_bat_number": [1, 2, 3, 4, 5], "pitch_number": 1,
                       "inning": [1, 2, 3, 10, 10], "inning_topbot": "Top", "home_team": "BOS", "away_team": "NYY"})
    lost = pd.DataFrame({"game_pk": [1, 1], "at_bat_number": [1, 2], "pitch_number": [1, 1], "team": ["BOS", "BOS"]})
    st = ab.challenges_left(df, lost)
    assert st["fld_left"].tolist() == [2, 1, 0, 1, 1] and st["bat_left"].tolist() == [2, 2, 2, 2, 2]


def test_lost_challenge_costs_outs_left_times_the_rate():
    raw = _raw()
    t, st = ab.attach_left(ab.table(raw, CH), raw)
    t = ab.add_cost(t, {2: 0.001, 1: 0.003})
    assert t["cost"].iloc[0] == 0                                 # overturned: free
    assert abs(t["cost"].iloc[1] - 54 * 0.001) < 1e-9             # lost in the 1st with 2 left
    assert t["cost"].iloc[2] == 0                                 # not counted (last pitch)
    assert t["left"].tolist() == [2, 2, 2]


def test_missed_calls_capture_and_above_average():
    raw = _raw()
    t, st = ab.attach_left(ab.table(raw, CH), raw)
    t = ab.add_cost(t, {2: 0.001, 1: 0.003})
    from core import abs_calls
    mc = ab.missed_calls(abs_calls.original_calls(raw, challenges=CH), None, st, t)
    # the ball in the zone (pitch 4) is a missed call against the defense; pitch 1 isn't -- the umpire's ORIGINAL
    # call there was a strike (the hitter got it overturned to a ball, which the final data shows)
    assert mc["against"].tolist() == ["Defense"] and mc["challenged_role"].tolist() == [""]
    cap = {"Catching": {"all": 0.4}}
    pl = ab.players(raw, t, mc, cap)
    c7 = pl[(pl["player"] == 7) & (pl["role"] == "Catching")].iloc[0]
    v = float(mc["value"].iloc[0])
    assert abs(c7["avail"] - v) < 1e-9 and abs(c7["x_runs"] - 0.4 * v) < 1e-9
    s = ab.summarize(2, t[t["role"].eq("Catching")], c7.to_dict())
    assert abs(s["Overturns +/-"] + 0.4 * v) < 1e-9 and s["Capture %"] == 0


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
