"""Leaderboard: starter version (exit velo & barrels). More boards to come."""
from datetime import date

import streamlit as st
from pybaseball import statcast_batter_exitvelo_barrels


@st.cache_data(ttl=6 * 3600, show_spinner="Loading leaderboard…")
def ev_board(year: int, min_bbe: int):
    return statcast_batter_exitvelo_barrels(year, minBBE=min_bbe)


st.title("Leaderboard")
c1, c2 = st.columns([1, 1])
year = c1.selectbox("Season", list(range(date.today().year, 2014, -1)))
min_bbe = c2.number_input("Min batted balls", 1, 500, 100, step=25)
df = ev_board(year, min_bbe)
if df.empty:
    st.info("No data for that season yet.")
else:
    st.dataframe(df, hide_index=True, use_container_width=True)
