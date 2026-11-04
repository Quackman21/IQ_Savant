import pandas as pd

from core.daily_summary import player_ids, summarize


def _df():
    rows = [
        # pitcher, batter, top/bot, type, desc, events, pitch, velo, ev, dist, lsa
        (10, 1, "Top", "X", "hit_into_play", "home_run", "FF", 99.0, 112.0, 440, 6),
        (10, 2, "Top", "S", "swinging_strike", None, "SL", 88.0, None, None, None),
        (10, 2, "Top", "S", "swinging_strike", "strikeout", "ST", 85.0, None, None, None),
        (20, 3, "Bot", "X", "hit_into_play", "field_out", "SI", 101.5, 90.0, 300, 3),
        (20, 4, "Bot", "B", "ball", "walk", "CH", 84.0, None, None, None),
        (20, 5, "Bot", "X", "hit_into_play", "single", "CU", 79.0, 96.0, 250, 4),
    ]
    cols = ["pitcher", "batter", "inning_topbot", "type", "description", "events", "pitch_type",
            "release_speed", "launch_speed", "hit_distance_sc", "launch_speed_angle"]
    df = pd.DataFrame(rows, columns=cols)
    df["home_team"], df["away_team"] = "BOS", "SEA"
    return df


def test_leaders_and_teams():
    s = summarize(_df())
    assert s["exit_velo"][0] == {"player": 1, "team": "SEA", "value": 112.0}   # top = away batting
    assert s["distance"][0]["value"] == 440
    assert s["pitch_velo"][0] == {"player": 20, "team": "SEA", "value": 101.5}  # bot = away pitching
    assert s["whiffs"] == [{"player": 10, "team": "BOS", "value": 2}]
    assert player_ids(s) == [1, 3, 5, 10, 20]


def test_totals():
    t = summarize(_df())["totals"]
    assert (t["Pitchers"], t["Batters"], t["Pitches"]) == (2, 5, 6)
    assert t["SL%"] == round(100 * 2 / 6, 1)           # SL + sweeper grouped
    assert (t["AB"], t["H"], t["HR"], t["1B"], t["SO"], t["BIP"]) == (4, 2, 1, 1, 1, 3)
    assert t["AVG"] == ".500" and t["HH"] == 2 and t["Brl"] == 1


def test_empty_day():
    assert summarize(pd.DataFrame()) == {}
