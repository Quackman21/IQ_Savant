"""Player Profile (starter)."""
from datetime import date

import streamlit as st
from pybaseball import playerid_lookup, statcast_batter


@st.cache_data(ttl=6 * 3600, show_spinner="Pulling Statcast…")
def batter(start, end, pid):
    return statcast_batter(start, end, pid)


st.title("Player Profile")
a, b = st.columns(2)
first, last = a.text_input("First name"), b.text_input("Last name")
if first and last:
    found = playerid_lookup(last, first)
    if found.empty:
        st.warning("No player found with that name.")
        st.stop()
    pid = int(found.key_mlbam.iloc[0])
    st.image(f"https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{pid}/headshot/67/current", width=120)
    s, e = st.date_input("Range", [date(2026, 3, 26), date(2026, 9, 30)], format="MM/DD/YYYY")
    st.dataframe(batter(s.isoformat(), e.isoformat(), pid), hide_index=True)
