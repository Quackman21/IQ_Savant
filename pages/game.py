"""Game page (opens in a new tab from the Calendar): /game?pk=<gamePk>&view=box|lineup|pbp"""
from datetime import date, datetime

import streamlit as st

from core import game_feed, mlb_api
from core.scoreboard import ET
from ui import data
from ui.game import box_html, linescore_html, lineup_html, plays_html
from ui.standings import TITLE_CSS

show = st.html if hasattr(st, "html") else (lambda h: st.markdown(h, unsafe_allow_html=True))
VIEWS = {"box": "Box Score", "lineup": "Lineup", "pbp": "Play by Play"}
LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"


@st.cache_data(ttl=30, show_spinner="Loading game…")
def feed(pk: int) -> dict:
    return mlb_api.get_game_feed(pk)


try:
    pk = int(st.query_params.get("pk", ""))
except ValueError:
    st.info("Open a game from the Calendar page.")
    st.stop()

try:
    f = feed(pk)
except Exception as e:
    st.caption(f"Game unavailable right now ({type(e).__name__}).")
    st.stop()

hd = game_feed.header(f)
season = int((hd["date"] or str(date.today()))[:4])
teams = data.teams_for(season)
away, home = hd["away"], hd["home"]
an, hn = teams.get(away, {}).get("team_name", "Away"), teams.get(home, {}).get("team_name", "Home")

# ---- title: "Mariners 9 @ Red Sox 6" + status ------------------------------
if hd["state"] == "Final":
    status = "Final" + (f"/{hd['inning']}" if hd["inning"] and hd["inning"] != 9 else "")
elif hd["state"] == "Live":
    status = f'{hd["inning_state"] or ""} {hd["inning"] or ""}'.strip()
else:
    try:
        t = datetime.fromisoformat(hd["start"].replace("Z", "+00:00")).astimezone(ET)
        status = t.strftime("%I:%M %p ET").lstrip("0")
    except ValueError:
        status = hd["detail"]
score = hd["state"] in ("Final", "Live")
a_s = f' {hd["away_runs"]}' if score and hd["away_runs"] is not None else ""
h_s = f' {hd["home_runs"]}' if score and hd["home_runs"] is not None else ""
st.markdown(TITLE_CSS + f'<h2 class="iq-title">'
            f'<img src="{LOGO.format(id=away)}" style="height:1em;vertical-align:-0.12em"> {an}{a_s} @ '
            f'<img src="{LOGO.format(id=home)}" style="height:1em;vertical-align:-0.12em"> {hn}{h_s}</h2>',
            unsafe_allow_html=True)
d = date.fromisoformat(hd["date"]) if hd["date"] else None
st.caption(" · ".join(x for x in [status, f"{d.month}/{d.day}/{d.year}" if d else "", hd["venue"]] if x))

# ---- view switcher (also kept in the URL) -----------------------------------
view = st.query_params.get("view", "box")
view = view if view in VIEWS else "box"


def _view_changed():
    label = st.session_state.get("game_view") or VIEWS["box"]
    st.query_params["view"] = next(k for k, v in VIEWS.items() if v == label)


if hasattr(st, "segmented_control"):
    st.segmented_control("View", list(VIEWS.values()), default=VIEWS[view], key="game_view",
                         label_visibility="collapsed", on_change=_view_changed)
else:
    st.radio("View", list(VIEWS.values()), index=list(VIEWS).index(view), horizontal=True,
             key="game_view", label_visibility="collapsed", on_change=_view_changed)
label = st.session_state.get("game_view") or VIEWS[view]

if hd["state"] == "Live":
    if st.button("↻ Refresh", help="Live games update about every 30 seconds"):
        feed.clear()
        st.rerun()

# ---- content ----------------------------------------------------------------
if label == "Box Score":
    show(linescore_html(game_feed.linescore(f), teams, away, home))
    left, right = st.columns(2)
    for col, side, tid in ((left, "away", away), (right, "home", home)):
        with col:
            show(box_html(game_feed.batting(f, side), game_feed.pitching(f, side), teams.get(tid, {}), tid))
elif label == "Lineup":
    left, right = st.columns(2)
    for col, side, tid in ((left, "away", away), (right, "home", home)):
        with col:
            show(lineup_html(game_feed.lineup(f, side), teams.get(tid, {}), tid))
else:
    scoring_only = st.toggle("Scoring plays only", value=False)
    st.caption("Click a play to see every pitch.")
    show(plays_html(game_feed.plays(f), teams, away, home, scoring_only))
