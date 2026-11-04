"""Run with:  python -m pytest -q"""
import random
from datetime import date, datetime, timedelta

from core.scoreboard import (ET, Game, can_clinch_with_win, champion, display_date,
                             pick_day, rank_games, series_odds)

# 30 fake teams: ids 1-15 AL (divs 201-203), 16-30 NL (divs 204-206)
TEAMS = {t: {"abbr": f"T{t:02d}", "name": f"Team {t}", "team_name": f"Team{t}",
             "league": 103 if t <= 15 else 104, "division": 201 + (t - 1) // 5}
         for t in range(1, 31)}


def g(pk, day, home, away, gtype="R", winner=None, state=None, **kw):
    state = state or ("final" if winner else "pre")
    return Game(pk=pk, gtype=gtype, day=day, start_utc=f"{day}T23:00:00Z",
                home=home, away=away, state=state, winner=winner, **kw)


def fake_season(played_through: str, bias: dict, seed=1):
    """162-ish game schedule; teams in `bias` win much more/less often."""
    rng = random.Random(seed)
    games, pk, d0 = [], 1, date(2026, 3, 26)
    for i in range(183):
        day = (d0 + timedelta(days=i)).isoformat()
        ids = list(TEAMS)
        rng.shuffle(ids)
        for h, a in zip(ids[::2], ids[1::2]):
            p = 0.5 + bias.get(h, 0) - bias.get(a, 0)
            w = (h if rng.random() < p else a) if day < played_through else None
            games.append(g(pk, day, h, a, winner=w))
            pk += 1
    return games


def test_switch_time():
    assert display_date(datetime(2026, 7, 4, 10, 29, tzinfo=ET)) == date(2026, 7, 3)
    assert display_date(datetime(2026, 7, 4, 10, 30, tzinfo=ET)) == date(2026, 7, 4)


def test_offseason_falls_back_to_last_game_day():
    games = [g(1, "2026-10-30", 1, 16, "W", winner=1)]
    assert pick_day(games, date(2026, 11, 3)) == "2026-10-30"


def test_series_odds():
    assert series_odds(1, 1) == 0.5
    assert abs(series_odds(1, 4) - 15 / 16) < 1e-9


def test_world_series_like_screenshot():
    MIL, TB = 16, 5
    days = ["2026-10-24", "2026-10-25", "2026-10-27", "2026-10-28", "2026-10-29"]
    winners = [MIL, MIL, MIL, TB, MIL]
    ws = [g(100 + i, d, MIL if i >= 2 else TB, TB if i >= 2 else MIL, "W", winner=w,
            series_game=i + 1, series_len=7) for i, (d, w) in enumerate(zip(days, winners))]
    reg = [g(1, "2026-09-01", MIL, 1, winner=MIL)]
    top = rank_games(reg + ws, "2026-10-29", TEAMS)
    assert len(top) == 1
    game5 = top[0]
    assert set(game5.clinch) == {MIL}           # MIL up 3-1: red star
    assert game5.series_wins[MIL] == 4
    assert champion(reg + ws, "2026-10-29") == MIL
    assert champion(reg + ws, "2026-10-28") is None
    # game 4 (MIL up 3-0) is a bigger swing than game 1 (0-0)? No: 0-0 G1 swing 31%, 3-0 swing 12.5%
    g1 = rank_games(reg + ws, "2026-10-24", TEAMS)[0]
    g4 = rank_games(reg + ws, "2026-10-28", TEAMS)[0]
    assert g1.importance > g4.importance > 0


def test_contenders_outrank_eliminated_teams_in_september():
    # T01 dominant, T02/T03 battling for a wild card, T14/T15 terrible
    bias = {1: .2, 2: .06, 3: .06, 4: .05, 14: -.25, 15: -.25}
    season = fake_season("2026-09-20", bias)
    day = "2026-09-20"
    season += [g(9001, day, 2, 3), g(9002, day, 14, 15)]   # extra marquee + dud games
    top = rank_games(season, day, TEAMS, top_n=20)
    order = [x.pk for x in top]
    assert order.index(9001) < order.index(9002)
    assert len(rank_games(season, day, TEAMS)) == 5


def test_clinch_with_win():
    # T01 has 95 wins, 5 left; division rivals T02-T05 can reach at most 95.
    wins = {t: 70 for t in TEAMS}
    left = {t: 5 for t in TEAMS}
    wins[1] = 95
    wins.update({2: 90})
    # a win puts T01 at 96 > every AL rival's ceiling (T02 = 95)
    assert can_clinch_with_win(1, 2, wins, left, TEAMS) == "z"   # division + best record
    # an ordinary team can't
    assert not can_clinch_with_win(6, 7, wins, left, TEAMS)


def _ws(upto_games=5):
    MIL, TB = 16, 5
    days = ["2026-10-24", "2026-10-25", "2026-10-27", "2026-10-28", "2026-10-29"]
    winners = [MIL, MIL, MIL, TB, MIL]
    lcs = [g(50 + i, f"2026-10-{15 + i}", 16, 20, "L", winner=16, series_game=i + 1, series_len=7)
           for i in range(4)]
    ws = [g(100 + i, d, MIL, TB, "W", winner=w, series_game=i + 1, series_len=7)
          for i, (d, w) in enumerate(zip(days[:upto_games], winners))]
    return [g(1, "2026-09-01", 16, 1, winner=16)] + lcs + ws


def test_postseason_shows_last_five_in_order():
    from core.scoreboard import postseason_strip
    games, off = postseason_strip(_ws(2), "2026-10-25")
    assert [x.pk for x in games] == [51, 52, 53, 100, 101]
    assert [x.day for x in games] == sorted(x.day for x in games)
    assert games[-1].pk == 101 and not off


def test_postseason_off_day_box():
    from core.scoreboard import postseason_strip
    games, off = postseason_strip(_ws(2), "2026-10-26")      # no game on the 26th
    assert off and games[-1].pk == 101


def test_no_off_day_box_after_title():
    from core.scoreboard import postseason_strip
    games, off = postseason_strip(_ws(5), "2026-11-03")
    assert not off and [x.pk for x in games] == [100, 101, 102, 103, 104]


def test_regular_season_unaffected():
    from core.scoreboard import postseason_strip
    assert postseason_strip(_ws(5), "2026-09-01") is None


def test_clinch_labels():
    wins = {t: 70 for t in TEAMS}
    left = {t: 5 for t in TEAMS}
    # T01 leads its division comfortably, but T06 (other division) could still pass it
    wins.update({1: 90, 2: 85, 6: 92})
    assert can_clinch_with_win(1, 2, wins, left, TEAMS) == "y"
    # wild card only: T03 can't catch T01 in the division but a win locks a berth
    wins = {t: 70 for t in TEAMS}
    wins.update({1: 100, 3: 88, 6: 99, 11: 99, 7: 85, 12: 85, 8: 83})
    assert can_clinch_with_win(3, 4, wins, left, TEAMS) == "w"


def test_congrats_cards():
    from core.scoreboard import congrats_for
    games = _ws(5)
    cards = congrats_for(games, "2026-11-03")
    assert cards == [{"round": "W", "winner": 16, "wins": 4, "played": 5, "day": "2026-10-29"}]
    lcs = congrats_for(_ws(0), "2026-10-20")          # LCS swept 4-0 on Oct 18
    assert lcs and lcs[0]["round"] == "L" and lcs[0]["played"] == 4


def test_standings_parse_and_tables():
    from core.standings import parse, division_tables, wildcard_table
    def rec(tid, w, l, dr, wcr, clinch=None):
        r = {"team": {"id": tid}, "wins": w, "losses": l, "winningPercentage": f"{w/(w+l):.3f}"[1:],
             "divisionRank": str(dr), "wildCardRank": str(wcr), "divisionGamesBack": "-",
             "runsScored": 700, "runsAllowed": 650, "runDifferential": 50,
             "streak": {"streakCode": "W2"},
             "records": {"splitRecords": [{"type": "lastTen", "wins": 6, "losses": 4},
                                          {"type": "home", "wins": 50, "losses": 31}],
                         "expectedRecords": [{"type": "xWinLoss", "wins": 90, "losses": 72}]}}
        if clinch:
            r["clinchIndicator"] = clinch
        return r
    payload = {"records": [{"league": {"id": 103}, "division": {"id": 201},
                            "teamRecords": [rec(1, 95, 67, 1, 0, "z"), rec(2, 90, 72, 2, 1, "w"),
                                            rec(3, 80, 82, 3, 4)]}]}
    rows = parse(payload)
    assert rows[0]["clinch"] == "z" and rows[1]["clinch"] == "w" and rows[2]["clinch"] is None
    assert rows[0]["l10"] == "6-4" and rows[0]["xwl"] == "90-72" and rows[0]["diff"] == "+50"
    assert rows[0]["away"] == "-"                      # missing split shows a dash
    name, teams = division_tables(rows, 103)[0]
    assert name == "AL East" and [r["team"] for r in teams] == [1, 2, 3]
    leaders, race = wildcard_table(rows, 103)
    assert [r["team"] for r in leaders] == [1] and [r["team"] for r in race] == [2, 3]


def test_standings_date_resolution():
    from ui.standings import resolve
    season26 = [g(1, "2026-03-26", 1, 2, winner=1), g(2, "2026-09-27", 1, 2, winner=2)]
    by_year = {2026: season26, 2027: [g(3, "2027-03-25", 1, 2)]}
    games = lambda y: by_year.get(y, [])
    assert resolve(games, "2026-06-01") == (2026, "2026-06-01")   # mid-season: as of that day
    assert resolve(games, "2026-11-03") == (2026, None)            # offseason: final standings
    assert resolve(games, "2026-09-27") == (2026, None)            # last day: final standings
    assert resolve(games, "2027-01-10") == (2026, None)            # winter: last season's final
    assert resolve(games, "2026-03-20") == (2025, None)            # before opening day


def test_wild_card_spots_by_year():
    from core.standings import wc_spots
    assert (wc_spots(2019), wc_spots(2020), wc_spots(2021), wc_spots(2024)) == (2, 5, 2, 3)


def test_next_game_after_picked_date():
    from core.standings import next_games
    games = [g(1, "2026-06-01", 1, 2, winner=1), g(2, "2026-06-02", 2, 1), g(3, "2026-06-03", 3, 1)]
    assert next_games(games, "2026-06-01", TEAMS)[1] == "@ T02 · Jun 2"
    assert next_games(games, "2026-06-02", TEAMS)[1] == "@ T02 · Jun 2"   # today's unplayed game
