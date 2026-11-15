"""After-Action Report (opens in a new tab from a box score or lineup): /aar?pk=<gamePk>&pid=<player id>

Hitter AAR: header, then ONE interactive panel -- box line, Last 30 PA / Season / MLB season tiles (pick the
metric), zone / spray / point-of-contact charts, and a single pitch table grouped by plate appearance (every PA
starts collapsed). Click pitches on any chart or in the table and they light up everywhere; select as many as you
want. Pitcher / Catcher / Fielder AARs will reuse the same pieces (core/aar.py, ui/aar.py)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def venue_dims(venue_id: int) -> dict:
    """Fence distances for a park (MLB stats API), when the game feed doesn't carry them."""
    try:
        import requests
        r = requests.get(f"https://statsapi.mlb.com/api/v1/venues/{venue_id}?hydrate=fieldInfo", timeout=15)
        v = (r.json().get("venues") or [{}])[0]
        return aar.field_dims({"gameData": {"venue": v}})
    except Exception:
        return {}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def park_outline(home_name: str, dims_items: tuple, venue: str) -> dict:
    return aar.park_outline(home_name, dict(dims_items), venue=venue)


try:
    pk, pid = int(st.query_params.get("pk", "")), int(st.query_params.get("pid", ""))
except ValueError:
    st.info("Open an AAR from a game's box score or lineup (the report icon next to a hitter).")
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

sc = statcast(pid, gday, gday)
if not sc.empty and "game_pk" in sc:
    sc = sc[pd.to_numeric(sc["game_pk"], errors="coerce").eq(pk)]
if sc.empty:
    st.info("Statcast hasn't posted this game's pitches yet -- they usually appear within an hour or two of the "
            "final out.")
    st.stop()

season_sc = statcast(pid, f"{season}-03-01", gday)
if not season_sc.empty and "game_type" in season_sc:
    season_sc = season_sc[season_sc["game_type"].astype(str).eq("R")]

p = aar.pitches(sc, f)
pas = aar.plate_appearances(p)

# point of contact on the plate: Statcast measures it from his hips; his own season's swings (where the ball was vs
# where he met it) place his hips in the box, so every contact point lands on the plate -- no stance download
cal = aar.plate_calibration(season_sc)
p = p.join(aar.on_plate(p, cal))
pas = aar.plate_appearances(p)
if p["cx"].notna().any() and not cal:
    note = "Contact measured, but he needs ~40 measured swings this season before it can be placed on the plate."
else:
    note = "No measured contact."

tip = "; ".join(f'{"RHB" if k == "R" else "LHB"}: placed from {c["n"]} of his swings this season (his hips sit '
                f'{-c["F0"]:.0f}" ± {c["se"]:.0f}" behind the front of the plate)' for k, c in (cal or {}).items())

dims = aar.field_dims(f) or venue_dims(int(f.get("gameData", {}).get("venue", {}).get("id", 0) or 0))
home_name = teams.get(hd["home"], {}).get("name", "") or f.get("gameData", {}).get("teams", {}).get("home", {}).get("name", "")
outline = park_outline(home_name, tuple(sorted(dims.items())), hd["venue"] or "")

tiles = aar.metric_values(aar.last_pas(season_sc, 30), season_sc, league(season))
components.html(aar_ui.component_html(p, pas, line, tiles, aar.GAME_METRICS, note, dims, hd["venue"],
                                      contact_tip=tip, outline=outline),
                height=aar_ui.estimate_height(len(pas), len(p)), scrolling=True)
st.caption("Click any pitch on a chart or in the table to light it up everywhere; click more to compare. Opening "
           "plate appearances narrows the charts to those PAs. Decision = our swing-decision model on that exact "
           "pitch; Dec Runs = runs vs what an average hitter does with it. Point of contact: Statcast bat tracking "
           "placed on the plate from his own swings (front-center of the plate = 0, 0; hover the chart for how many). "
           "The spray chart uses this park's real fence distances.")
