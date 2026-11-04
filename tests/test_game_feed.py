from core import game_feed
from core.mlb_api import _location


def test_city_names():
    assert _location({"name": "Tampa Bay Rays", "teamName": "Rays", "locationName": "St. Petersburg"}) == "Tampa Bay"
    assert _location({"name": "Texas Rangers", "teamName": "Rangers", "locationName": "Arlington"}) == "Texas"
    assert _location({"name": "Athletics", "teamName": "Athletics"}) == ""


def _feed():
    def player(pid, name, pos, order=None, bat=None, pit=None):
        p = {"person": {"id": pid, "fullName": name}, "position": {"abbreviation": pos},
             "stats": {"batting": bat or {}, "pitching": pit or {}},
             "seasonStats": {"batting": {"avg": ".250"}, "pitching": {"era": "3.10"}}}
        if order:
            p["battingOrder"] = order
        return p
    away_players = {"ID1": player(1, "Lead Off", "CF", "100", {"atBats": 4, "hits": 2, "runs": 1, "rbi": 1}),
                    "ID2": player(2, "Pinch Hit", "PH", "101", {"atBats": 1}),
                    "ID3": player(3, "Ace Arm", "P", pit={"inningsPitched": "7.0", "strikeOuts": 9, "numberOfPitches": 101})}
    return {
        "gameData": {"game": {"pk": 777, "type": "R"}, "datetime": {"officialDate": "2026-09-01"},
                     "teams": {"away": {"id": 136}, "home": {"id": 111}},
                     "status": {"abstractGameState": "Final", "detailedState": "Final"},
                     "players": {"ID1": {"fullName": "Lead Off", "batSide": {"code": "L"}},
                                 "ID3": {"fullName": "Ace Arm", "pitchHand": {"code": "R"}}}},
        "liveData": {
            "linescore": {"scheduledInnings": 9, "currentInning": 9,
                          "innings": [{"num": i, "away": {"runs": 1 if i == 1 else 0},
                                       "home": ({"runs": 0} if i < 9 else {})} for i in range(1, 10)],
                          "teams": {"away": {"runs": 1, "hits": 5, "errors": 0},
                                    "home": {"runs": 0, "hits": 3, "errors": 1}}},
            "boxscore": {"teams": {"away": {"players": away_players, "pitchers": [3], "battingOrder": [1]},
                                   "home": {"players": {}, "pitchers": [], "battingOrder": []}}},
            "plays": {"allPlays": [
                {"about": {"inning": 1, "halfInning": "top", "isScoringPlay": True},
                 "result": {"event": "Home Run", "description": "Lead Off homers", "awayScore": 1, "homeScore": 0},
                 "matchup": {"batter": {"fullName": "Lead Off"}, "pitcher": {"fullName": "X"}},
                 "count": {"outs": 0},
                 "playEvents": [{"isPitch": True, "pitchNumber": 1, "details": {"description": "In play, run(s)",
                                 "type": {"description": "Four-Seam Fastball"}}, "pitchData": {"startSpeed": 96.1},
                                 "count": {"balls": 0, "strikes": 0}}]},
                {"about": {"inning": 1, "halfInning": "bottom"}, "result": {"event": "Strikeout"},
                 "matchup": {"batter": {"fullName": "Y"}, "pitcher": {"fullName": "Ace Arm"}}, "playEvents": []},
            ]}},
    }


def test_header_and_linescore():
    f = _feed()
    hd = game_feed.header(f)
    assert (hd["away"], hd["home"], hd["away_runs"], hd["state"]) == (136, 111, 1, "Final")
    ls = game_feed.linescore(f)
    assert ls["away"]["innings"][0] == 1 and ls["home"]["innings"][8] == "X" and ls["home"]["E"] == 1


def test_box_lineup_plays():
    f = _feed()
    bat = game_feed.batting(f, "away")
    assert [b["name"] for b in bat] == ["Lead Off", "Pinch Hit"] and bat[1]["sub"] and bat[0]["H"] == 2
    pit = game_feed.pitching(f, "away")
    assert pit[0]["IP"] == "7.0" and pit[0]["SO"] == 9 and pit[0]["P"] == 101
    lu = game_feed.lineup(f, "away")
    assert lu["nine"][0]["bats"] == "L" and lu["pitcher"]["throws"] == "R"
    halves = game_feed.plays(f)
    assert [(h["inning"], h["half"]) for h in halves] == [(1, "top"), (1, "bottom")]
    assert halves[0]["plays"][0]["scoring"] and halves[0]["plays"][0]["pitches"][0]["mph"] == 96.1
