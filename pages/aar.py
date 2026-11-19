"""After-Action Report (opens in a new tab from a box score or lineup): /aar?pk=<gamePk>&pid=<player id>

Hitter AAR: header, then ONE interactive panel -- box line, Last 30 PA / Season / MLB season tiles (pick the
metric), zone / spray / point-of-contact charts, and a single pitch table grouped by plate appearance (every PA
starts collapsed). Click pitches on any chart or in the table and they light up everywhere; select as many as you
want. Report picker (top left): Hitting / Catching / Infield / Outfield -- only the ones he played in this game.
Catching AAR: core/catch_aar.py + ui/catch_aar.py (every pitch he received, balls in play left out).
Infield / Outfield AARs: core/field_aar.py + ui/field_aar.py (every ball that was his, from the IQ fielding model)."""
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


@st.cache_data(ttl=6 * 3600, show_spinner="Pulling the game's pitches…")
def game_statcast(pk: int, day: str) -> pd.DataFrame:
    """Every pitch of the game (both teams) -- the catcher's view needs the other team's pitchers' pitches."""
    import pybaseball as pb
    df = None
    try:
        df = pb.statcast_single_game(pk)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        try:
            df = pb.statcast(day, day)
            df = df[pd.to_numeric(df["game_pk"], errors="coerce").eq(pk)] if df is not None and len(df) else df
        except Exception:
            df = None
    return clean(df) if df is not None else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def catch_comps(season: int, pid: int, day: str) -> dict:
    from core import catch_aar
    return catch_aar.comps(season, pid, day)


@st.cache_data(ttl=3600, show_spinner=False)
def field_comps(season: int, pid: int, day: str, view: str) -> dict:
    from core import field_aar
    return field_aar.comps(season, pid, day, view)


@st.cache_data(ttl=3600, show_spinner=False)
def league(season: int) -> dict:
    return league_ref.load(season)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def stance_board(season: int) -> dict:
    return aar.stance_depths(season)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def stance_views(season: int) -> dict:
    from core import stance_feed
    return stance_feed.views(season)


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

# ---- which report (top left): only the ones he played in this game
from core import catch_aar
reports = catch_aar.available(f, pid)
if not reports:
    show(aar_ui.header_html(pid, who, team, gm, "Player"))
    st.info("He didn't bat or play the field in this game.")
    st.stop()
want = st.query_params.get("view", "")
view = want if want in reports else reports[0]


def _view_changed():
    st.query_params["view"] = st.session_state.get("aar_view") or view


pick_col, _ = st.columns([2, 5])
with pick_col:
    if hasattr(st, "segmented_control"):
        st.segmented_control("Report", reports, default=view, key="aar_view", label_visibility="collapsed",
                             on_change=_view_changed)
    else:
        st.radio("Report", reports, index=reports.index(view), horizontal=True, key="aar_view",
                 label_visibility="collapsed", on_change=_view_changed)
view = st.session_state.get("aar_view") or view
if not team:
    for side in ("away", "home"):
        if f"ID{pid}" in f.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {}).get("players", {}):
            team = teams.get(hd[side], {}).get("name", "")

if view == "Catching":
    from ui import catch_aar as catch_ui
    head = aar_ui.header_html(pid, who, team, gm, "Catcher")
    full = game_statcast(pk, gday)
    if full.empty or "fielder_2" not in full:
        show(head)
        st.info("Statcast hasn't posted this game's pitches yet -- they usually appear within an hour or two of the "
                "final out.")
        st.stop()
    events = catch_aar.runner_events(f, pk)
    mine = catch_aar.caught(full, pid)
    cp = catch_aar.pitches(full, f, pid, events)
    if cp.empty:
        show(head)
        st.info("No pitches caught by him in Statcast for this game.")
        st.stop()
    from models import catching as cm
    from models import decisions_iq as dq
    has_model = cm.load() is not None and dq.load() is not None
    sc3 = catch_aar.scenes(cp)
    try:
        cmp = catch_comps(season, pid, gday)
    except Exception:
        cmp = {}
    components.html(catch_ui.component_html(cp, catch_aar.line(cp, events, mine), header=head, scenes3d=sc3,
                                            has_model=has_model, comps=cmp),
                    height=catch_ui.estimate_height(len(cp)), scrolling=True)
    st.caption("Every pitch he received (balls in play left out). Strike % = an average umpire's chance it's called "
               "a strike if taken; Framing Runs = (called strike or not - that chance) x what a strike is worth in "
               "the count (2026+: the umpire's original call, before any ABS challenge). WP/PB % = an average "
               "catcher's chance a blocking chance (runner on) gets by; Blocking Runs = (that chance - what happened) "
               "x what a WP/PB costs in that base/out state. Throwing Runs = (caught - CS chance) x the run swing. "
               "Zone = Statcast's 1-14 zone named from the catcher's side -- arm side = his right (1B side), glove side "
               "= his left (11-14 are outside the zone); Region = Savant's Heart / Shadow / Chase / Waste + which "
               "edge, same sides. The cube opens the pitch's flight in "
               "3D, to the dirt or 3 ft past the middle of the plate.")
    st.stop()

if view in ("Infield", "Outfield"):
    from core import field_aar
    from models import fielding as fm
    from ui import field_aar as field_ui
    head = aar_ui.header_html(pid, who, team, gm, "Infielder" if view == "Infield" else "Outfielder")
    full = game_statcast(pk, gday)
    if full.empty or "fielder_3" not in full:
        show(head)
        st.info("Statcast hasn't posted this game's pitches yet -- they usually appear within an hour or two of the "
                "final out.")
        st.stop()
    has_model = fm.load() is not None
    fc = field_aar.attach(field_aar.credits(full, pid, view), f)
    fl = field_aar.line(fc, view)
    try:
        fcmp = field_comps(season, pid, gday, view)
    except Exception:
        fcmp = {}
    dims = aar.field_dims(f) or venue_dims(int(f.get("gameData", {}).get("venue", {}).get("id") or 0))
    home_name = (teams.get(hd["home"], {}).get("name", "")
                 or f.get("gameData", {}).get("teams", {}).get("home", {}).get("name", ""))
    try:                                           # the park this game was played in (same outline as the spray chart)
        outline = park_outline(home_name, tuple(sorted(dims.items())), hd["venue"] or "")
    except Exception:
        outline = None
    n = 0 if fc["plays"] is None or fc["plays"].empty else len(fc["plays"])
    components.html(field_ui.component_html(fc, fl, view, header=head, comps=fcmp, dims=dims, has_model=has_model,
                                            outline=outline, park=hd["venue"] or ""),
                    height=field_ui.estimate_height(n), scrolling=True)
    st.caption("Every ball in play that was his, judged by the ball alone (exit velo, launch angle, direction, flight "
               "distance, batter side) with the IQ fielding model -- the same one behind the profile's Defense tab. "
               "Out chance = how often an average defense makes the out on a ball like that; his share = how often "
               "his position is the one that makes it (from the league, not from where he stood). Credit: out he made "
               "= 1 - out chance; hit / error = -(out chance x his share). No starting spots or routes -- MLB doesn't "
               "publish them.")
    st.stop()

head = aar_ui.header_html(pid, who, team, gm)
if not line:
    show(head)
    st.info("This player didn't bat in this game.")
    st.stop()

sc = statcast(pid, gday, gday)
if not sc.empty and "game_pk" in sc:
    sc = sc[pd.to_numeric(sc["game_pk"], errors="coerce").eq(pk)]
if sc.empty:
    show(head)
    st.info("Statcast hasn't posted this game's pitches yet -- they usually appear within an hour or two of the "
            "final out.")
    st.stop()

season_all = statcast(pid, f"{season}-03-01", gday)
# regular season + playoffs for where his hips are at contact (hitters move in the box -- the swings right before
# this game matter most); this game itself is left out. Tiles stay regular season.
hip_sc = season_all
if not hip_sc.empty and "game_pk" in hip_sc:
    hip_sc = hip_sc[pd.to_numeric(hip_sc["game_pk"], errors="coerce").ne(pk)]
season_sc = season_all
if not season_sc.empty and "game_type" in season_sc:
    season_sc = season_sc[season_sc["game_type"].astype(str).eq("R")]

p = aar.pitches(sc, f)
pas = aar.plate_appearances(p)

# point of contact on the plate: Statcast measures it from his hips; his own season's swings (where the ball was vs
# where he met it) place his hips in the box, so every contact point lands on the plate -- no stance download
stance = {sd: d for (b, sd), d in stance_board(season).items() if b == pid}
cal = aar.plate_calibration(hip_sc, asof=gday, group=aar.hip_group(season), stance=stance)
playoff = str(f.get("gameData", {}).get("game", {}).get("type", "R")) in ("F", "D", "L", "W")
p = p.join(aar.pitch_stance(p, pid, stance_views(season), playoff))
p = p.join(aar.on_plate(p, cal))
pas = aar.plate_appearances(p)
if p["cx"].notna().any() and not cal:
    note = "Contact measured, but he needs ~40 measured swings this season before it can be placed on the plate."
else:
    note = "No measured contact."

SRC = {"playoff": "his playoff stance", "vs_L": "his stance vs LHP", "vs_R": "his stance vs RHP",
       "season": "his season stance"}


def _tip(k, c):
    hand = "RHB" if k == "R" else "LHB"
    used = p.loc[p["stand"].astype(str).eq(k) & p["depth"].notna(), ["depth_src", "depth"]] if "depth" in p else None
    if used is not None and not used.empty:
        parts = ", ".join(f'{SRC.get(src, src)} {g["depth"].iloc[0]:.1f}"' for src, g in used.groupby("depth_src"))
        return (f'{hand}: MLB\'s contact point vs the plate -- Statcast\'s contact (from his hips) minus his stance '
                f'depth from Savant\'s board, pitch by pitch ({parts}); side to side + height = where the ball was')
    if c["source"] == "mlb":
        return (f'{hand}: MLB\'s contact point vs the plate -- Statcast\'s contact (from his hips) minus his stance '
                f'depth ({c["depth"]:.1f}" from the front of the plate, Savant stance board); side to side + height = '
                f'where the ball was')
    return (f'{hand}: not on Savant\'s stance board yet -- stance depth estimated from {c["n"]} of his swings '
            f'(~{c["depth"]:.0f}" ± {2 * c["se"]:.0f}")')


tip = "; ".join(_tip(k, c) for k, c in (cal or {}).items())
estimated = bool(p["pf"].notna().any() and (p.loc[p["pf"].notna(), "depth"].isna().any() if "depth" in p else True)
                 and any(c["source"] != "mlb" for c in (cal or {}).values()))

dims = aar.field_dims(f) or venue_dims(int(f.get("gameData", {}).get("venue", {}).get("id", 0) or 0))
home_name = teams.get(hd["home"], {}).get("name", "") or f.get("gameData", {}).get("teams", {}).get("home", {}).get("name", "")
outline = park_outline(home_name, tuple(sorted(dims.items())), hd["venue"] or "")

# This game: the same pull the Season tile uses (it works), filtered to this game; the single-game pull is the fallback
game_rows = season_all[pd.to_numeric(season_all["game_pk"], errors="coerce").eq(pk)] \
    if (not season_all.empty and "game_pk" in season_all) else pd.DataFrame()
tiles = aar.metric_values(aar.last_pas(season_sc, 30), season_sc, league(season),
                          game=game_rows if not game_rows.empty else sc)
from core import swing3d, weather
try:
    scenes3d = swing3d.scenes(sc, p, swing3d.stance_offsets(season, pid))   # contact swings: pitch + bat in 3D
except Exception:
    scenes3d = {}
# the air: game conditions in the header; per ball, the wind along its direction + air density in the 3D readouts
try:
    cond0 = weather.conditions(pk, season, feed=f)
except Exception:
    cond0 = {}
if cond0:
    head = head + (f'<div style="margin:-6px 0 10px 4px;color:#6a7681;font:12.5px system-ui">'
                   f'Conditions: {weather.describe(cond0)}</div>')
for n, s3 in scenes3d.items():
    try:
        r = p.loc[p["n"].eq(n)].iloc[0]
        c = weather.conditions(pk, season, int(r["inning"]), str(r["half"]) == "▲", feed=f)
        s3["read"]["air_pct"] = c.get("density_vs_std_pct")
        s3["read"]["temp_f"] = c.get("temp_f")
        if s3.get("exit") and pd.notna(r.get("hc_x")) and c.get("wind_mph") is not None:
            import math
            spray = math.degrees(math.atan2(float(r["hc_x"]) - 125.42, 198.27 - float(r["hc_y"])))
            s3["read"]["tailwind"] = (round(weather.along(c["wind_mph"], c["wind_field_deg"], spray), 1)
                                      if c.get("wind_field_deg") is not None else (0.0 if c["wind_mph"] == 0 else None))
    except Exception:
        pass
components.html(aar_ui.component_html(p, pas, line, tiles, aar.GAME_METRICS, note, dims, hd["venue"],
                                      contact_tip=tip, outline=outline, contact_est=estimated,
                                      header=head, scenes3d=scenes3d),
                height=aar_ui.estimate_height(len(pas), len(p)), scrolling=True)
st.caption("Click any pitch on a chart or in the table to light it up everywhere; click more to compare. The cube "
           "next to a swing's video opens that pitch + swing in 3D (contact swings). Opening "
           "plate appearances narrows the charts to those PAs. Decision = our swing-decision model on that exact "
           "pitch; Dec Runs = runs vs what an average hitter does with it. Point of contact: MLB's contact point vs "
           "the plate (Statcast's contact from his hips minus his stance depth; front-center of the plate = 0, 0). "
           "The spray chart uses this park's real fence distances.")
