"""Calendar: '15 MLB Games - 9/1/2026', a day-of-competition summary from
Statcast, then every game that day. Date lives in the URL (?date=YYYY-MM-DD)."""
from datetime import date, datetime

import streamlit as st
from pybaseball import statcast

from core import mlb_api
from core.daily_summary import player_ids, summarize
from core.statcast_clean import clean
from core.scoreboard import ET, display_date, pick_day
from core.standings import EARLIEST
from ui import data
from ui.calendar import games_html, summary_html
from ui.standings import TITLE_CSS

show = st.html if hasattr(st, "html") else (lambda h: st.markdown(h, unsafe_allow_html=True))


def _summary(day: str) -> tuple[dict, dict]:
    df = clean(statcast(start_dt=day, end_dt=day, verbose=False))
    s = summarize(df)
    names = mlb_api.get_people(player_ids(s)) if s else {}
    return s, names


@st.cache_data(ttl=24 * 3600, show_spinner="Pulling Statcast for the day…")
def summary_past(day: str):
    return _summary(day)


@st.cache_data(ttl=600, show_spinner="Pulling Statcast for the day…")
def summary_recent(day: str):
    return _summary(day)


today = datetime.now(ET).date()
earliest = date.fromisoformat(EARLIEST)

# default: the most recent day with games
target = display_date()
season = data.current_season(target)
last = pick_day(data.season_games(season), target)
default = date.fromisoformat(last) if last else target
try:
    default = min(max(date.fromisoformat(st.query_params.get("date", "")), earliest), today)
except ValueError:
    pass


def _date_changed():
    st.query_params["date"] = st.session_state["calendar_date"].isoformat()


try:
    title_col, date_col = st.columns([5, 1.3], vertical_alignment="bottom")
except TypeError:
    title_col, date_col = st.columns([5, 1.3])
with date_col:
    day = st.date_input("Date", value=default, key="calendar_date", min_value=earliest,
                        max_value=today, format="MM/DD/YYYY", label_visibility="collapsed",
                        on_change=_date_changed)

iso = day.isoformat()
games = [g for g in data.season_games(day.year) if g.day == iso]
teams = data.teams_for(day.year)
if iso >= display_date().isoformat():                      # today: live scores
    live = data.live(iso)
    for g in games:
        cur = live.get(g.pk)
        if cur:
            g.state, g.winner = cur.state, cur.winner
            g.home_score, g.away_score = cur.home_score, cur.away_score
            g.inning, g.inning_state, g.sched_innings = cur.inning, cur.inning_state, cur.sched_innings

n = sum(g.state != "void" for g in games)
with title_col:
    st.markdown(TITLE_CSS + f'<h2 class="iq-title">{n} MLB Game{"" if n == 1 else "s"} - '
                f'{day.month}/{day.day}/{day.year}</h2>', unsafe_allow_html=True)

if not games:
    st.info("No MLB games on this date.")
    st.stop()

# ---- day-of-competition summary -------------------------------------------
if any(g.state in ("live", "final") for g in games):
    try:
        recent = (today - day).days <= 2
        s, names = (summary_recent if recent else summary_past)(iso)
        if s:
            show(summary_html(s, names, teams))
        else:
            st.caption("Statcast data for this day isn't posted yet.")
    except Exception as e:
        st.caption(f"Day summary unavailable right now ({type(e).__name__}).")
else:
    st.caption("The day summary appears once games are underway.")

# ---- games ------------------------------------------------------------------
show(games_html(games, teams))
