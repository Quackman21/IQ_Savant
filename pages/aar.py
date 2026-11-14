"""After-Action Report (opens in a new tab from the box score clipboard icon): /aar?pk=<gamePk>&pid=<player id>

Hitter AAR: box line, game metric vs season vs MLB, whole-game zone / spray / point-of-contact charts, then every
plate appearance as its own collapsible section (open several at once) with its pitches, its own three charts and a
video link for every pitch. Pitcher / Catcher / Fielder AARs will be added with the same pieces (core/aar.py)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from core import aar, game_feed, league_ref, mlb_api
from core.statcast_clean import clean
from ui import aar as aar_ui
from ui import data

show = st.html if hasattr(st, "html") else (lambda h: st.markdown(h, unsafe_allow_html=True))


@st.cache_data(ttl=60, show_spinner="Loading game…")
def feed(pk: int) -> dict:
    return mlb_api.get_game_feed(pk)


@st.cache_data(ttl=6 * 3600, show_spinner="Pulling Statcast…")
def statcast(pid: int, start: str, end: str) -> pd.DataFrame:
    import pybaseball as pb
    return clean(pb.statcast_batter(start, end, pid))


@st.cache_data(ttl=3600, show_spinner=False)
def league(season: int) -> dict:
    return league_ref.load(season)


try:
    pk, pid = int(st.query_params.get("pk", "")), int(st.query_params.get("pid", ""))
except ValueError:
    st.info("Open an AAR from a game's box score (the clipboard next to a hitter's name).")
    st.stop()

try:
    f = feed(pk)
except Exception as e:
    st.caption(f"Game unavailable right now ({type(e).__name__}).")
    st.stop()

hd = game_feed.header(f)
gday = hd["date"] or date.today().isoformat()
season = int(gday[:4])
teams = data.teams_for(season)
line = aar.box_line(f, pid)
who = aar.names(f).get(pid, "Hitter")
team = teams.get(line.get("team_id"), {}).get("name", "") if line else ""
gm = {**hd, "abbr_away": teams.get(hd["away"], {}).get("abbr", ""),
      "abbr_home": teams.get(hd["home"], {}).get("abbr", ""),
      "date": f"{int(gday[5:7])}/{int(gday[8:10])}/{gday[:4]}"}

show(aar_ui.header_html(pid, who, team, gm))
if not line:
    st.info("This player didn't bat in this game.")
    st.stop()
show(aar_ui.line_html(line))

sc = statcast(pid, gday, gday)
if not sc.empty and "game_pk" in sc:
    sc = sc[pd.to_numeric(sc["game_pk"], errors="coerce").eq(pk)]
if sc.empty:
    st.info("Statcast hasn't posted this game's pitches yet -- they usually appear within an hour or two of the "
            "final out. The box line above is live.")
    st.stop()

season_sc = statcast(pid, f"{season}-03-01", gday)
if not season_sc.empty and "game_type" in season_sc:
    season_sc = season_sc[season_sc["game_type"].astype(str).eq("R")]
p = aar.pitches(sc, f)
pas = aar.plate_appearances(p)
orient = aar.contact_orientation(season_sc)

# ---- game metric vs season vs MLB ---------------------------------------------------------------------------
c = st.columns([1.3, 4])
key = c[0].selectbox("Metric", list(aar.GAME_METRICS), format_func=aar.GAME_METRICS.get, key="aar_metric")
with c[1]:
    show(aar_ui.tiles_html(aar.metric_tiles(sc, season_sc, league(season), key), aar.GAME_METRICS[key]))

# ---- whole game ------------------------------------------------------------------------------------------------
st.markdown("##### Whole game")
show(aar_ui.charts_html(p, orient))

# ---- plate appearances -----------------------------------------------------------------------------------------
h = st.columns([3, 1.2])
h[0].markdown(f"##### Plate appearances ({len(pas)})")
open_all = h[1].toggle("Open every PA", value=False, key="aar_open_all")
st.caption("Click a plate appearance to open it -- open as many as you want. Each shows its pitches and its own "
           "zone, spray and contact charts. 🎥 plays that pitch.")
show(aar_ui.pa_html(pas, orient, open_all=open_all))

with st.expander("Every pitch in one table"):
    show(aar_ui.CSS + '<div class="aar">' + aar_ui.pitch_table(p) + "</div>")
st.caption("Decision = our swing-decision model on that exact pitch (count, location, pitch): Great / Good = the "
           "right call, Bad / Poor = the wrong one; Dec Runs = runs vs what an average hitter does with that pitch. "
           "Zone = attack zone. Point of contact is Statcast bat tracking -- shown only when measured.")
