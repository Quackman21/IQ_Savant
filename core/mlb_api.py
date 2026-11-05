"""Thin wrapper around MLB's free Stats API (statsapi.mlb.com).

pybaseball covers Statcast, FanGraphs and Baseball Reference, but it has no
schedule / live-score feed, so the scoreboard reads MLB's API directly.
No key or account is needed.
"""
from __future__ import annotations

import requests

BASE = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 20

# R = regular season, F = Wild Card, D = Division Series, L = LCS, W = World Series
REGULAR = "R"
POSTSEASON = ("F", "D", "L", "W")
COUNTED_TYPES = (REGULAR, *POSTSEASON)

AL, NL = 103, 104


def _get(path: str, **params) -> dict:
    r = requests.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _flatten(payload: dict) -> list[dict]:
    games = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            if g.get("gameType") in COUNTED_TYPES:
                games.append(g)
    return games


def _location(t: dict) -> str:
    """'Tampa Bay' from 'Tampa Bay Rays' (locationName is the ballpark's city,
    e.g. St. Petersburg / Arlington, so it isn't used)."""
    name, club = t.get("name", ""), t.get("teamName", "")
    if club and name.endswith(club) and name != club:
        return name[: -len(club)].strip()
    return t.get("franchiseName", "") if t.get("franchiseName") != club else ""


def get_teams(season: int) -> dict[int, dict]:
    """team_id -> {abbr, name, team_name, league, division}."""
    data = _get("teams", sportId=1, season=season)
    out = {}
    for t in data.get("teams", []):
        out[t["id"]] = {
            "abbr": t.get("abbreviation", t.get("teamCode", "")).upper(),
            "name": t.get("name", ""),
            "team_name": t.get("teamName", t.get("name", "")),
            "location": _location(t),
            "league": t.get("league", {}).get("id"),
            "division": t.get("division", {}).get("id"),
        }
    return out


def get_season_games(season: int) -> list[dict]:
    """Every regular-season and postseason game for a season (no spring training)."""
    data = _get("schedule", sportId=1,
                startDate=f"{season}-02-01", endDate=f"{season}-11-30")
    return _flatten(data)


def get_standings(season: int, on: str | None = None) -> dict:
    """Regular-season standings for both leagues (records, splits, clinch flags).
    `on` = 'YYYY-MM-DD' for standings as of the end of that day; None = latest."""
    params = dict(leagueId=f"{AL},{NL}", season=season, standingsTypes="regularSeason")
    if on:
        y, m, d = on.split("-")
        params["date"] = f"{m}/{d}/{y}"
    return _get("standings", **params)


def get_people(ids: list[int]) -> dict[int, str]:
    """player id -> full name (Statcast only gives the pitcher's name)."""
    out: dict[int, str] = {}
    ids = [int(i) for i in ids]
    for k in range(0, len(ids), 150):
        data = _get("people", personIds=",".join(map(str, ids[k:k + 150])))
        out.update({p["id"]: p.get("fullName", "") for p in data.get("people", [])})
    return out


def get_boxscore(pk: int) -> dict:
    return _get(f"game/{pk}/boxscore")


def get_linescore(pk: int) -> dict:
    return _get(f"game/{pk}/linescore")


def get_playbyplay(pk: int) -> dict:
    return _get(f"game/{pk}/playByPlay")


def get_game_feed(game_pk: int) -> dict:
    """Everything about one game: box score, lineups, every play and pitch."""
    r = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_day_games(date_str: str) -> list[dict]:
    """One day's games with live linescore (inning, outs) for the ticker."""
    data = _get("schedule", sportId=1, date=date_str, hydrate="linescore")
    return _flatten(data)


# ----------------------------------------------------------------------------
# Players
# ----------------------------------------------------------------------------
LEVELS = {1: "MLB", 11: "AAA", 12: "AA", 13: "A+", 14: "A", 16: "Rookie", 17: "Winter"}
MILB_SPORTS = "11,12,13,14,16"


def search_people(name: str) -> list[dict]:
    """Players matching a name (MLB + minors): [{id, name, pos, team}]."""
    data = _get("people/search", names=name, sportIds=f"1,{MILB_SPORTS}", hydrate="currentTeam")
    out = []
    for p in data.get("people", []):
        out.append({"id": p["id"], "name": p.get("fullName", ""),
                    "pos": p.get("primaryPosition", {}).get("abbreviation", ""),
                    "team": p.get("currentTeam", {}).get("name", ""),
                    "active": p.get("active", False)})
    return out


def get_season_players(season: int) -> list[dict]:
    """Every player on an MLB roster that season (id, fullName, primaryPosition, currentTeam)."""
    return _get("sports/1/players", season=season).get("people", [])


def get_person(pid: int) -> dict:
    data = _get(f"people/{pid}", hydrate="currentTeam")
    return (data.get("people") or [{}])[0]


def get_team(team_id: int) -> dict:
    data = _get(f"teams/{team_id}")
    return (data.get("teams") or [{}])[0]


def get_draft_pick(year: int, pid: int) -> dict | None:
    """This player's pick in that year's draft (round, overall, school, bonus)."""
    data = _get(f"draft/{year}")
    for rnd in data.get("drafts", {}).get("rounds", []):
        for pick in rnd.get("picks", []):
            if pick.get("person", {}).get("id") == pid:
                return pick
    return None


def get_transactions(pid: int) -> list[dict]:
    return _get("transactions", playerId=pid).get("transactions", [])


def get_stats(pid: int, group: str, kind: str, **params) -> list[dict]:
    """Stat splits for a player. kind: season | yearByYear | byDateRange | gameLog | statSplits."""
    data = _get(f"people/{pid}/stats", stats=kind, group=group, **params)
    return (data.get("stats") or [{}])[0].get("splits", [])


def get_league_totals(season: int, group: str = "hitting") -> dict:
    """MLB-wide totals for a season (sum of all 30 teams), for 'Show Avgs'."""
    data = _get("teams/stats", season=season, group=group, stats="season", sportIds=1)
    total: dict = {}
    for s in (data.get("stats") or [{}])[0].get("splits", []):
        for k, v in s.get("stat", {}).items():
            if isinstance(v, (int, float)):
                total[k] = total.get(k, 0) + v
    return total
