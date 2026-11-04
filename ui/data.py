"""Cached data loaders shared by every page. Each one hits MLB's API at most
once per cache window, no matter how many visitors or page switches."""
from __future__ import annotations

from datetime import date

import streamlit as st

from core import mlb_api, standings
from core.scoreboard import Game, dedupe, parse_game, rank_games


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def teams_for(season: int) -> dict:
    return mlb_api.get_teams(season)


@st.cache_data(ttl=900, show_spinner=False)
def season_games(season: int) -> list[Game]:
    return [parse_game(g) for g in mlb_api.get_season_games(season)]


@st.cache_data(ttl=3 * 3600, show_spinner="Ranking games…")
def ranked(season: int, day: str) -> list[Game]:
    return rank_games(season_games(season), day, teams_for(season))


@st.cache_data(ttl=60, show_spinner=False)
def live(day: str) -> dict[int, Game]:
    return {g.pk: g for g in dedupe([parse_game(x) for x in mlb_api.get_day_games(day)])}


@st.cache_data(ttl=900, show_spinner="Loading standings…")
def standings_for(season: int, on: str | None = None) -> list[dict]:
    return standings.parse(mlb_api.get_standings(season, on))


def current_season(today: date) -> int:
    """This year once games exist, otherwise last year (winter / spring)."""
    if any(g.day <= today.isoformat() for g in season_games(today.year)):
        return today.year
    return today.year - 1
