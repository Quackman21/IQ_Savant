"""IQ Savant entry point: run with  streamlit run app.py

The left-hand menu is built here. Each page lives in its own file in pages/.
To add a page: create pages/<name>.py, make an st.Page for it below, and add
a st.page_link for it in the sidebar section.
"""
import streamlit as st

st.set_page_config(page_title="IQ Savant", layout="wide")

standings = st.Page("pages/standings.py", title="Standings",
                    icon=":material/format_list_numbered:", default=True)
calendar = st.Page("pages/calendar.py", title="Calendar", icon=":material/calendar_month:")
leaderboard = st.Page("pages/leaderboard.py", title="Leaderboard", icon=":material/leaderboard:")
player = st.Page("pages/player_profile.py", title="Player Profile", icon=":material/person:")
team = st.Page("pages/team_profile.py", title="Team Profile", icon=":material/stadium:")
game = st.Page("pages/game.py", title="Game", icon=":material/sports_baseball:", url_path="game")

# Hidden built-in menu; we draw our own so Profile can be a dropdown
# (the Game page isn't in the menu; it opens in a new tab from the Calendar)
current = st.navigation([standings, calendar, leaderboard, player, team, game], position="hidden")

with st.sidebar:
    st.page_link(standings)
    st.page_link(calendar)
    st.page_link(leaderboard)
    with st.expander("Profile", icon=":material/badge:",
                     expanded=current.title in ("Player Profile", "Team Profile")):
        st.page_link(player)
        st.page_link(team)

current.run()
