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
