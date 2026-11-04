"""Team Profile (starter)."""
import streamlit as st

from core.scoreboard import display_date
from ui import data

st.title("Team Profile")

season = data.current_season(display_date())
teams = data.teams_for(season)
ids = sorted(teams, key=lambda t: teams[t]["name"])
tid = st.selectbox("Team", ids, format_func=lambda t: teams[t]["name"])
st.image(f"https://www.mlbstatic.com/team-logos/{tid}.svg", width=90)

row = next((r for r in data.standings_for(season, None) if r["team"] == tid), None)
if row:
    cols = st.columns(6)
    for col, (label, key) in zip(cols, [("Record", None), ("PCT", "pct"), ("GB", "gb"),
                                        ("Run diff", "diff"), ("Last 10", "l10"), ("Streak", "strk")]):
        col.metric(label, f'{row["w"]}-{row["l"]}' if key is None else row[key])

games = sorted((g for g in data.season_games(season) if tid in (g.home, g.away) and g.played),
               key=lambda g: g.day, reverse=True)[:10]
st.markdown("**Last 10 games**")
for g in games:
    home = g.home == tid
    opp = teams.get(g.away if home else g.home, {}).get("abbr", "?")
    us, them = (g.home_score, g.away_score) if home else (g.away_score, g.home_score)
    st.write(f'{g.day} · {"vs" if home else "@"} {opp} · {"W" if g.winner == tid else "L"} {us}-{them}')
