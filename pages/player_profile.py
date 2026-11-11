"""Player Profile: search -> white header card -> Summary | Stats | Game Log | Sandbox.
Summary's sub-tabs depend on where the player plays: Hitting | Core 4 |
Catching | Infield | Outfield | Baserunning, and Pitching for pitchers.
The player lives in the URL (?id=<MLBAM id>)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pybaseball as pb
import streamlit as st

from core import hit_metrics as hm
from core import savant
from models import catching as catching_model
from models import fielding as fielding_model
from models import framing as framing_model
from core import hitting, league_ref, mlb_api, pitching
from core.statcast_clean import clean
from core.player import bio, draft_entry, position_games, signing_entry, summary_tabs
from ui import data
from core import core4 as c4
from ui.core4 import table_html as core4_table
import math

from core import defense as dfn
from core import pitch_metrics as pm
from ui import pitching as pitch_ui
from ui.pitching import table_html as pitching_table
from ui.defense import table_html as defense_table
from ui.hitting import overview_html, tab_html
from ui.periods import picker as period_picker
from ui.periods import display_order as period_order
from ui.periods import resolve as resolve_period
from ui.player import empty_html, header_html, percentile_html, table_html

show = st.html if hasattr(st, "html") else (lambda h: st.markdown(h, unsafe_allow_html=True))
TODAY = date.today()
FIRST_SEASON = 2015                       # Statcast era
POSTSEASON = {"F", "D", "L", "W"}


# =============================================================================
# Cached data
# =============================================================================
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def search(name: str) -> list[dict]:
    try:
        return mlb_api.search_people(name)
    except Exception:                                    # fall back to pybaseball's register
        parts = name.split()
        if len(parts) < 2:
            return []
        df = pb.playerid_lookup(parts[-1], " ".join(parts[:-1]))
        return [{"id": int(r.key_mlbam), "name": f"{r.name_first.title()} {r.name_last.title()}",
                 "pos": "", "team": "", "active": True} for r in df.itertuples() if r.key_mlbam > 0]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def person(pid: int) -> dict:
    return mlb_api.get_person(pid)


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def entry_line(pid: int, draft_year, country: str) -> str:
    try:
        if draft_year:
            pick = mlb_api.get_draft_pick(int(draft_year), pid)
            if pick:
                abbr = data.teams_for(int(draft_year)).get(pick.get("team", {}).get("id"), {}).get("abbr", "")
                return draft_entry(pick, int(draft_year), abbr or pick.get("team", {}).get("name", ""))
            return f"{draft_year} Draft"
        teams = data.teams_for(TODAY.year)
        return signing_entry(mlb_api.get_transactions(pid), country,
                             lambda t: teams.get(t, {}).get("abbr")) or "-"
    except Exception:
        return "-"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def level_of(team_id: int) -> str:
    try:
        sport = mlb_api.get_team(team_id).get("sport", {}).get("id")
        return mlb_api.LEVELS.get(sport, "MiLB")
    except Exception:
        return "MLB"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def stats(pid: int, group: str, kind: str, **params) -> list[dict]:
    return mlb_api.get_stats(pid, group, kind, **params)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def league(season: int) -> dict:
    return mlb_api.get_league_totals(season)


@st.cache_data(ttl=3600, show_spinner=False)
def league_ref_for(season: int) -> dict:
    return league_ref.load(season)


@st.cache_data(ttl=3600, show_spinner=False)
def catching_daily(season: int) -> pd.DataFrame:
    return league_ref.load_catching(season)


@st.cache_data(ttl=3600, show_spinner=False)
def fielding_daily(season: int) -> pd.DataFrame:
    return league_ref.load_fielding(season)


@st.cache_data(ttl=3600, show_spinner=False)
def ref_status(season: int) -> str:
    return league_ref.status(season)


def data_note(seasons) -> None:
    """One quiet line when the MLB reference is missing / built with older definitions."""
    bad = sorted({y for y in seasons if ref_status(y) != "current"})
    if bad:
        st.caption(f"MLB comparisons for {', '.join(map(str, bad))} are missing or built with older stat "
                   "definitions — run `python -m pipeline.update` next time you step away.")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def league_p(season: int) -> dict:
    return mlb_api.get_league_totals(season, "pitching")


@st.cache_data(ttl=6 * 3600, show_spinner="Pulling Statcast…")
def statcast(pid: int, start: str, end: str) -> pd.DataFrame:
    return clean(pb.statcast_batter(start, end, pid))


@st.cache_data(ttl=6 * 3600, show_spinner="Pulling Statcast…")
def statcast_p(pid: int, start: str, end: str) -> pd.DataFrame:
    return clean(pb.statcast_pitcher(start, end, pid))


@st.cache_data(ttl=24 * 3600, show_spinner="Loading leaderboard…")
def board(kind: str, year: int) -> pd.DataFrame:
    """Savant leaderboards via pybaseball. 'frv:5' = fielding run value at 3B, etc."""
    if kind.startswith("frv:"):
        fn = lambda: savant.fielding_run_value(year, int(kind[4:]), 1)
    elif kind.startswith("oaa:"):
        fn = lambda: pb.statcast_outs_above_average(year, int(kind[4:]), 1)
    else:
        fn = {"percentiles": lambda: pb.statcast_batter_percentile_ranks(year),
              "p_percentiles": lambda: pb.statcast_pitcher_percentile_ranks(year),
              "framing": lambda: pb.statcast_catcher_framing(year),
              "poptime": lambda: pb.statcast_catcher_poptime(year),
              "sprint": lambda: pb.statcast_sprint_speed(year, 10),
              "oaa_if": lambda: pb.statcast_outs_above_average(year, "IF", 1),
              "oaa_of": lambda: pb.statcast_outs_above_average(year, "OF", 1),
              "of_jump": lambda: pb.statcast_outfielder_jump(year, 1),
              "of_catch": lambda: pb.statcast_outfield_catch_prob(year, 1),
              "of_dir": lambda: pb.statcast_outfield_directional_oaa(year, 1)}[kind]
    try:
        df = fn()
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# =============================================================================
# Small helpers
# =============================================================================
def keep_choice(key: str, options: list, default=None) -> None:
    """Before a tab bar draws: if its saved value is missing / not an option, reset it to the default."""
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]


def hold(key: str) -> None:
    """on_change: clicking the already-selected tab would clear it (nothing highlighted).
    Put the last choice back instead, so one option is always selected."""
    last = f"{key}__last"
    if st.session_state.get(key) in (None, []):
        st.session_state[key] = st.session_state.get(last)
    else:
        st.session_state[last] = st.session_state[key]


def age_years(day) -> str:
    try:
        bd = date.fromisoformat(p.get("birthDate", ""))
        return f"{(day - bd).days / 365.25:.1f}"
    except (ValueError, NameError):
        return "-"


def log_window(group: str, year: int, start, end) -> list[dict]:
    """Game-by-game splits (MLB game log) between two dates -- works for months / weeks / windows,
    which MLB's season totals can't do for fielding."""
    out = []
    for s_ in stats(pid, group, "gameLog", season=year, sportId=1):
        try:
            d = date.fromisoformat(s_.get("date", ""))
        except ValueError:
            continue
        if start <= d <= end:
            out.append(s_)
    return out


def period_window(per):
    """(start, end, label) for a period; Latest Week / Last 30 Days end on the player's last game."""
    if per.kind == "season":
        return date(per.season, 3, 1), date(per.season, 11, 30), "Season"
    if per.kind == "post":
        return date(per.season, 9, 28), date(per.season, 11, 5), "Postseason"
    log = stats(pid, "hitting", "gameLog", season=per.season, sportId=1)
    return resolve_period(per, pd.Series([s_.get("date") for s_ in log]))


def hitting_splits(per, start, end) -> list[dict]:
    if per.kind == "season":
        return stats(pid, "hitting", "season", season=per.season, sportId=1)
    if per.kind == "post":
        return stats(pid, "hitting", "season", season=per.season, sportId=1, gameType="P")
    return log_window("hitting", per.season, start, end)


def seg(label, options, default=None, key=None, disabled=False, help=None, collapsed=False,
        selection_mode="single", on_change=None):
    vis = "collapsed" if collapsed else "visible"
    if key and selection_mode == "single" and default is not None:
        keep_choice(key, options, default)
        if on_change is None:
            on_change = lambda: hold(key)
        st.session_state.setdefault(f"{key}__last", st.session_state.get(key, default))
    if hasattr(st, "segmented_control"):
        v = st.segmented_control(label, options, default=default, key=key, disabled=disabled,
                                 help=help, label_visibility=vis, selection_mode=selection_mode,
                                 on_change=on_change)
    elif selection_mode == "multi":
        v = st.multiselect(label, options, default=default or [], key=key, disabled=disabled, help=help)
    else:
        v = st.radio(label, options, index=options.index(default) if default in options else 0,
                     horizontal=True, key=key, disabled=disabled, help=help, label_visibility=vis)
    return v


ID_COLS = ("player_id", "entity_id", "resp_fielder_id", "id", "catcher_id", "mlbam_id", "batter")
SKIP = {"year", "age", "team_id", "player_id", "entity_id", "resp_fielder_id", "id", "catcher_id", "mlbam_id", "batter",
        "player_name", "last_name", "first_name", "name", "team", "team_name", "position"}
LOWER_IS_BETTER = ("time", "exchange", "pop", "whiff", "k_percent", "chase", "swing_length")


def row_for(df: pd.DataFrame, pid: int) -> pd.Series | None:
    for c in ID_COLS:
        if c in df.columns:
            hit = df[pd.to_numeric(df[c], errors="coerce") == pid]
            if not hit.empty:
                return hit.iloc[0]
    return None


UPPER = {"xwoba": "xwOBA", "xba": "xBA", "xslg": "xSLG", "xiso": "xISO", "oaa": "OAA",
         "woba": "wOBA", "ev": "EV", "la": "LA", "hp": "HP", "2b": "2B", "3b": "3B", "rv": "RV"}


def pretty(col: str) -> str:
    """'brl_percent' -> 'Barrel %', 'pop_2b_sba' -> 'Pop 2B Sba'."""
    words = col.lower().replace("_percent", " %").replace("percent", "%").replace("brl", "barrel").split("_")
    out = " ".join(UPPER.get(w, w.capitalize()) for w in " ".join(words).split())
    return out.replace(" %", "%")


def numeric_cols(df: pd.DataFrame, match: tuple[str, ...] | None = None) -> list[str]:
    cols = [c for c in df.columns if c.lower() not in SKIP and pd.api.types.is_numeric_dtype(df[c])]
    if match:
        cols = [c for c in cols if any(m in c.lower() for m in match)]
    return cols


def board_stats(kind: str, years: list[int], pid: int, match=None) -> dict[str, dict]:
    out = {}
    for y in years:
        df = board(kind, y)
        r = row_for(df, pid) if not df.empty else None
        if r is not None:
            out[str(y)] = {pretty(c): (round(float(r[c]), 2) if pd.notna(r[c]) else "-")
                           for c in numeric_cols(df, match)}
    return out


def board_percentiles(kind: str, years: list[int], pid: int, match=None, already=False) -> dict[str, float]:
    """Percentile of the player within each leaderboard (averaged over seasons).
    `already` = the board is Savant's percentile-rank table (values are ranks)."""
    acc: dict[str, list[float]] = {}
    for y in years:
        df = board(kind, y)
        r = row_for(df, pid) if not df.empty else None
        if r is None:
            continue
        for c in numeric_cols(df, match):
            if pd.isna(r[c]):
                continue
            if already:
                p = float(r[c])
            else:
                col = pd.to_numeric(df[c], errors="coerce").dropna()
                p = 100 * (col < r[c]).mean()
                if any(k in c.lower() for k in LOWER_IS_BETTER):
                    p = 100 - p
            acc.setdefault(pretty(c), []).append(p)
    return {k: sum(v) / len(v) for k, v in acc.items()}


INFIELD_POS = ["1B", "2B", "3B", "SS"]
OUTFIELD_POS = ["LF", "CF", "RF"]
POS_NUM = {"C": 2, "1B": 3, "2B": 4, "3B": 5, "SS": 6, "LF": 7, "CF": 8, "RF": 9}


def spray_chart(df: pd.DataFrame) -> None:
    """Batted balls on a field (Statcast hit coordinates), coloured by result."""
    import altair as alt
    bip = df[df["type"].eq("X")].dropna(subset=["hc_x", "hc_y"]) if "hc_x" in df else pd.DataFrame()
    if bip.empty:
        return
    pts = pd.DataFrame({
        "x": (bip["hc_x"] - 125.42) * 2.5, "y": (198.27 - bip["hc_y"]) * 2.5,
        "Result": bip["events"].fillna("other").str.replace("_", " ").str.title(),
        "EV": bip["launch_speed"].round(1), "LA": bip["launch_angle"].round(0),
        "Date": bip["game_date"].astype(str),
    })
    lines = pd.DataFrame({"x": [-300, 0, 300], "y": [300, 0, 300]})
    field = alt.Chart(lines).mark_line(color="#b9c0c6").encode(x="x:Q", y="y:Q")
    dots = alt.Chart(pts).mark_circle(size=55, opacity=.8).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-330, 330])),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-20, 460])),
        color=alt.Color("Result:N", legend=alt.Legend(orient="right")),
        tooltip=["Date", "Result", "EV", "LA"])
    st.altair_chart((field + dots).properties(height=420), use_container_width=True)


def seasons_played(pid: int, group: str) -> list[int]:
    """Seasons with MLB or MiLB games (Statcast era onward), newest first."""
    yrs: set[int] = set()
    for extra in ({}, {"leagueListId": "milb_all"}):
        try:
            yrs |= {int(s["season"]) for s in stats(pid, group, "yearByYear", **extra)}
        except Exception:
            pass
    return sorted((y for y in yrs if y >= FIRST_SEASON), reverse=True) or [TODAY.year]


def sum_stats(splits: list[dict]) -> dict:
    tot: dict = {}
    for s in splits:
        for k, v in s.get("stat", {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                tot[k] = tot.get(k, 0) + v
    return tot


def level_label(split: dict) -> str:
    sp = split.get("sport", {})
    return mlb_api.LEVELS.get(sp.get("id"), sp.get("abbreviation", ""))


# =============================================================================
# Search
# =============================================================================
pid = None
try:
    pid = int(st.query_params.get("id", ""))
except ValueError:
    pass

@st.cache_data(ttl=24 * 3600, show_spinner="Loading players…")
def all_players() -> list[dict]:
    """Every MLB player from the last three seasons, newest season first (deduped)."""
    out, seen = [], set()
    for yr in range(TODAY.year, TODAY.year - 3, -1):
        try:
            people = mlb_api.get_season_players(yr)
        except Exception:
            continue
        teams = data.teams_for(yr)
        for p_ in people:
            if p_["id"] in seen:
                continue
            seen.add(p_["id"])
            out.append({"id": p_["id"], "name": p_.get("fullName", ""), "season": yr,
                        "pos": p_.get("primaryPosition", {}).get("abbreviation", ""),
                        "team": teams.get(p_.get("currentTeam", {}).get("id"), {}).get("team_name", "")})
    return sorted(out, key=lambda x: x["name"])


players = all_players()
pick = st.selectbox(
    "Search player", players, index=None, key="pp_search",
    placeholder="Search player name…  (type first or last name)",
    format_func=lambda x: f'{x["name"]}  ·  MLB | {x["season"]} | {x["pos"]} | {x["team"]}',
    label_visibility="collapsed")
if pick and pick["id"] != pid:
    st.query_params["id"] = str(pick["id"])
    st.session_state.pop("pp_search", None)
    st.rerun()

with st.expander("Can't find someone? Search all levels (incl. minor leagues)"):
    q = st.text_input("Search all players", placeholder="Full or partial name", label_visibility="collapsed")
    if q and len(q.strip()) >= 3:
        found = search(q.strip())
        if not found:
            st.caption("No players found.")
        else:
            pick2 = st.selectbox("Matches", found, index=None, placeholder=f"{len(found)} match(es) — pick a player",
                                 format_func=lambda p: " · ".join(x for x in (p["name"], p["pos"], p["team"]) if x),
                                 label_visibility="collapsed")
            if pick2 and pick2["id"] != pid:
                st.query_params["id"] = str(pick2["id"])
                st.rerun()

if not pid:
    st.title("Player Profile")
    st.caption("Search for a player to open their profile.")
    st.stop()

# =============================================================================
# Header
# =============================================================================
try:
    p = person(pid)
except Exception as e:
    st.caption(f"Player unavailable right now ({type(e).__name__}).")
    st.stop()
if not p:
    st.warning("Player not found.")
    st.stop()

b = bio(p)
teams_now = data.teams_for(TODAY.year)
level = "MLB" if b["team_id"] in teams_now else (level_of(b["team_id"]) if b["team_id"] else "")
show(header_html(pid, b, teams_now.get(b["team_id"]), level,
                 entry_line(pid, b["draft_year"], b["country"])))

group = "pitching" if b["is_pitcher"] else "hitting"
years = seasons_played(pid, "hitting" if not b["is_pitcher"] else "pitching")
p_years = seasons_played(pid, "pitching")

# which Summary sub-tabs this player gets (by where he has played the last 3 seasons)
try:
    field = stats(pid, "fielding", "yearByYear") + stats(pid, "fielding", "yearByYear", leagueListId="milb_all")
except Exception:
    field = []
try:
    career_pa = sum_stats(stats(pid, "hitting", "yearByYear")).get("plateAppearances", 0)
except Exception:
    career_pa = 0
try:
    career_pitch_g = sum_stats(stats(pid, "pitching", "yearByYear")).get("gamesPlayed", 0)
except Exception:
    career_pitch_g = 0
primary = p.get("primaryPosition", {}).get("abbreviation", "")
games_at = position_games(field, since=TODAY.year - 2)
SUB_TABS = summary_tabs(primary, games_at, career_pa, career_pitch_g) or ["Hitting"]

# tab bars styled like the reference: main = bold uppercase with underline,
# sub = grey "A | B | C" links with the active one green and underlined
ACTIVE = (':is([data-testid$="Active"],[kind$="Active"],[aria-checked="true"],[aria-pressed="true"],'
          '[aria-selected="true"])')
st.markdown(("""<style>
/* every toggle on the page: the selected option is bold and dark */
[data-testid="stButtonGroup"] buttonACTIVE p{
  font-weight:800!important;color:#16202b!important}
.st-key-pp_main [data-testid="stButtonGroup"]{display:flex;justify-content:center;
  border-bottom:1px solid #e3e7ea;width:100%}
.st-key-pp_main button{background:transparent!important;border:none!important;border-radius:0!important;
  box-shadow:none!important;padding:10px 20px!important;border-bottom:3px solid transparent!important}
.st-key-pp_main button p{font-weight:600!important;letter-spacing:.8px;text-transform:uppercase;
  color:#8a949d;font-size:14px!important}
.st-key-pp_main buttonACTIVE{border-bottom-color:#16202b!important}
.st-key-pp_main buttonACTIVE p{color:#16202b!important;font-weight:800!important}
.st-key-pp_sub, .st-key-pp_def{background:#f1f3f2;padding:8px 10px;margin:6px 0 10px}
.st-key-pp_def{background:#f7f8f8;margin-top:-4px}
.st-key-pp_sub [data-testid="stButtonGroup"], .st-key-pp_def [data-testid="stButtonGroup"]{gap:0}
.st-key-pp_sub button, .st-key-pp_def button{background:transparent!important;border:none!important;
  border-radius:0!important;box-shadow:none!important;padding:2px 14px!important}
.st-key-pp_sub button p, .st-key-pp_def button p{font-weight:600!important;letter-spacing:.4px;
  text-transform:uppercase;color:#8a949d;font-size:14px!important}
.st-key-pp_sub button + button, .st-key-pp_def button + button{border-left:1px solid #b9c0c6!important}
.st-key-pp_sub buttonACTIVE p, .st-key-pp_def buttonACTIVE p{color:#3fa37f!important;font-weight:800!important;
  text-decoration:underline;text-underline-offset:4px}
.stTabs [data-baseweb="tab-list"]{background:#f5f6f6;gap:0}
.stTabs [data-baseweb="tab"]{flex:1;justify-content:center;padding:10px 6px}
.stTabs [data-baseweb="tab"] p{font-weight:600;color:#8a949d;text-transform:uppercase;letter-spacing:.4px}
.stTabs [aria-selected="true"]{background:#fff;border:1px solid #dfe3e6;border-bottom:none;border-radius:6px 6px 0 0}
.stTabs [aria-selected="true"] p{color:#16202b;font-weight:800}
</style>""").replace("ACTIVE", ACTIVE), unsafe_allow_html=True)

main = seg("Section", ["Summary", "Stats", "Game Log", "Sandbox"], default="Summary",
           key="pp_main", collapsed=True) or "Summary"

# =============================================================================
# SUMMARY
# =============================================================================
if main == "Summary":
    if st.session_state.get("pp_sub") not in SUB_TABS:          # new player: reset sub-tab
        st.session_state.pop("pp_sub", None)
    sub = seg("Summary view", SUB_TABS, default=SUB_TABS[0], key="pp_sub", collapsed=True) or SUB_TABS[0]

    # ---------------------------------------------------------------- HITTING
    if sub == "Hitting":
        c = st.columns([1.5, 1.1, 1.1, 1.2, 1.9, 0.9])
        with c[0]:
            mode = seg("Mode", ["Game", "Season", "Dates"], default="Season", key="hit_mode", collapsed=True) or "Season"
        with c[1]:
            season = st.selectbox("Seasons", years, key="hit_season")
        with c[2]:
            st.selectbox("League", ["MLB"], key="hit_league", disabled=True, help="MiLB comes later")
        with c[3]:
            cat = st.selectbox("Stat Category", ["Regular", "Postseason"], key="hit_cat")
        with c[5]:
            st.write("")
            avgs = st.toggle("Show Avgs", key="hit_avgs")

        gtype = "R" if cat == "Regular" else "P"
        start, end, label = f"{season}-03-01", f"{season}-11-30", str(season)
        if mode == "Dates":
            rng = st.date_input("Dates", [date(season, 4, 1), min(date(season, 9, 30), TODAY)],
                                format="MM/DD/YYYY", key="hit_dates")
            if len(rng) == 2:
                start, end = rng[0].isoformat(), rng[1].isoformat()
                label = f"{rng[0].month}/{rng[0].day}–{rng[1].month}/{rng[1].day}"
        elif mode == "Game":
            log = stats(pid, "hitting", "gameLog", season=season, gameType=gtype, sportId=1)
            if log:
                g = st.selectbox("Game", list(reversed(log)), key="hit_game",
                                 format_func=lambda s: f'{s.get("date", "")} '
                                 f'{"vs" if s.get("isHome") else "@"} {s.get("opponent", {}).get("name", "")}')
                start = end = g.get("date", start)
                label = g.get("date", "")

        # Statcast pitches for the selection -> catalog-ready flags
        sc = statcast(pid, start, end)
        if not sc.empty and "game_type" in sc:
            sc = sc[sc["game_type"].eq("R")] if gtype == "R" else sc[sc["game_type"].isin(POSTSEASON)]
        d = hm.prep(sc)
        ref = league_ref_for(season)
        ctx = {"player": f'{b["first"]} {b["last"]}', "season_label": str(season)}
        data_note([season])

        # rows down the side: zone grouping + any splits (apply to Rates / Contact / Exit)
        r = st.columns([3.2, 2.2, 1.6])
        with r[0]:
            split_pick = seg("Split", list(hm.SPLITS), key="hit_splits", selection_mode="multi") or []
        with r[2]:
            zg = st.selectbox("Zone Grouping", list(hm.ZONE_GROUPINGS), key="hit_zone")
        zone_dim = hm.ZONE_GROUPINGS[zg]
        split_dims = [hm.SPLITS[x] for x in split_pick]
        rows = hm.player_rows(d, zone_dim, split_dims)

        def ref_for(labels):
            return ref.get(hm.ref_key(labels, zone_dim), {})

        tabs = st.tabs(["Overview", "Rates", "Contact", "Exit", "Rolling", "Spray"])
        with tabs[0]:
            all_vals = hm.compute(d).iloc[0] if not d.empty else None
            show(overview_html(label, all_vals, ref.get(("all", "All"), {}), ctx))
        for i, name in ((1, "Rates"), (2, "Contact"), (3, "Exit")):
            with tabs[i]:
                show(tab_html(hm.TABS[name], rows, label, ref_for, ctx, zone_dim, show_avgs=avgs))
        with tabs[4]:
            roll = hitting.rolling(sc)
            if roll.empty:
                show(empty_html("Not enough plate appearances for a rolling chart."))
            else:
                st.caption("Rolling 50 plate appearances — full Rolling view comes in v13.")
                st.line_chart(roll)
        with tabs[5]:
            if not sc.empty:
                spray_chart(sc)
        st.caption("Hover a column name for its definition. Click a number to compare it with MLB average.")

    # ---------------------------------------------------------------- CORE 4
    elif sub == "Core 4":
        c = st.columns([2.3, 1.2, 1.5, 1.0, 1.7, 2.1])
        with c[0]:
            st.caption("Periods")
            periods = period_picker("c4", years)
        with c[1]:
            agg = st.selectbox("Split", ["Aggregate", "Combined"], key="c4_split",
                               help="Aggregate = one row per period. Combined = all selected periods as one row.")
        with c[2]:
            cat = st.selectbox("Stat Category", ["Regular", "Postseason", "Reg + Post"], key="c4_cat")
        with c[3]:
            by_team = seg("Split By", ["Team"], key="c4_by") == "Team"
        with c[4]:
            disp = seg("Display", ["Stats", "Percentiles"], default="Stats", key="c4_disp") or "Stats"
        with c[5]:
            scale = seg("Evaluation Method", c4.SCALES, default="20-80", key="c4_scale") or "20-80"
        if not periods:
            st.caption("Pick at least one period.")
            st.stop()

        keep_types = {"Regular": {"R"}, "Postseason": POSTSEASON, "Reg + Post": {"R"} | POSTSEASON}[cat]
        seasons_raw, seasons_post = {}, {}
        for y in sorted({p_.season for p_ in periods}, reverse=True):
            df_y = statcast(pid, f"{y}-03-01", f"{y}-11-30")
            if not df_y.empty and "game_type" in df_y:
                seasons_post[y] = df_y[df_y["game_type"].isin(POSTSEASON)]
                df_y = df_y[df_y["game_type"].isin(keep_types)]
            seasons_raw[y] = df_y

        def age_at(day) -> str:
            try:
                bd = date.fromisoformat(p.get("birthDate", ""))
                return f"{(day - bd).days / 365.25:.1f}"
            except ValueError:
                return "-"

        def describe(df_, label_year, period_label, mid):
            types = set(df_["game_type"].astype(str)) if "game_type" in df_ else {"R"}
            typ = ", ".join(t for t, ok in (("Reg", "R" in types), ("Post", bool(types & POSTSEASON))) if ok)
            return {"Year": label_year, "Period": period_label, "Age": age_at(mid), "Type": typ, "vs": "All"}

        rows_c4, chunks = [], []
        for per in sorted(periods, key=period_order):
            raw_y = seasons_raw.get(per.season, pd.DataFrame())
            if per.kind == "post":
                raw_y = seasons_post.get(per.season, pd.DataFrame())
            if raw_y.empty:
                continue
            win = resolve_period(per, raw_y["game_date"])
            if not win:
                continue
            start_d, end_d, plabel = win
            gd = pd.to_datetime(raw_y["game_date"]).dt.date
            part = raw_y[(gd >= start_d) & (gd <= end_d)]
            if part.empty:
                continue
            chunks.append(part)
            if agg == "Combined":
                continue
            dd = hm.prep(part)
            ref_y = league_ref_for(per.season).get(("all", "All"), {})
            mid = start_d + (end_d - start_d) / 2
            groups = [(t, dd[dd["bat_team"] == t]) for t in sorted(dd["bat_team"].dropna().unique())] \
                if by_team else [(", ".join(sorted(dd["bat_team"].dropna().unique())), dd)]
            for team, g_ in groups:
                if g_.empty:
                    continue
                lead = describe(part, per.season, plabel, mid)
                lead["Team"] = team
                rows_c4.append({"lead": lead, "vals": hm.compute(g_).iloc[0], "ref": ref_y,
                                "season_row": per.kind == "season"})
        if agg == "Combined" and chunks:
            allp = pd.concat(chunks)
            key_cols = [k for k in ("game_pk", "at_bat_number", "pitch_number") if k in allp]
            if key_cols:
                allp = allp.drop_duplicates(subset=key_cols)
            dd = hm.prep(allp)
            yrs = sorted({p_.season for p_ in periods})
            label_year = str(yrs[0]) if len(yrs) == 1 else f"{yrs[0]}–{yrs[-1]}"
            gd = pd.to_datetime(allp["game_date"]).dt.date
            mid = gd.min() + (gd.max() - gd.min()) / 2
            ref_y = league_ref_for(yrs[-1]).get(("all", "All"), {})
            groups = [(t, dd[dd["bat_team"] == t]) for t in sorted(dd["bat_team"].dropna().unique())] \
                if by_team else [(", ".join(sorted(dd["bat_team"].dropna().unique())), dd)]
            for team, g_ in groups:
                lead = describe(allp, label_year, "Combined", mid)
                lead["Team"] = team
                rows_c4.append({"lead": lead, "vals": hm.compute(g_).iloc[0], "ref": ref_y})

        ctx = {"player": f'{b["first"]} {b["last"]}', "season_label": str(TODAY.year)}
        data_note({p_.season for p_ in periods})
        tabs = st.tabs(list(c4.TABS))
        for tab, name in zip(tabs, c4.TABS):
            with tab:
                show(core4_table(c4.TABS[name], rows_c4, ctx, scale, disp))
        if any(int(str(r["lead"]["Year"])[:4]) < 2023 for r in rows_c4 if str(r["lead"]["Year"])[:4].isdigit()):
            st.caption("Bat tracking (bat speed, swing length, attack angle) starts in 2023 — earlier seasons show –.")
        st.caption("Decision stats use a placeholder decision model (models/decisions.py) until ours is built.")

    # ---------------------------------------------------------------- BASERUNNING
    elif sub == "Baserunning":
        c = st.columns([2.3, 1.3, 2.4, 1.0, 1.7])
        with c[0]:
            st.caption("Periods")
            periods = period_picker("br", years)
        with c[3]:
            seg("Split By", ["Level", "Org", "Team"], key="br_by", help="Level / Org arrive with MiLB")
        with c[4]:
            seg("Display", ["Stats", "Percentiles"], default="Stats", key="br_disp")
        if not periods:
            st.caption("Pick at least one period.")
            st.stop()

        rows_b = []
        for per in sorted(periods, key=period_order):
            win = period_window(per)
            if not win:
                continue
            start_d, end_d, plabel = win
            hit = sum_stats(hitting_splits(per, start_d, end_d))
            if not hit.get("gamesPlayed") and not hit.get("plateAppearances"):
                continue
            sb, cs = hit.get("stolenBases", 0), hit.get("caughtStealing", 0)
            vals = {"G": hit.get("gamesPlayed", 0), "PA": hit.get("plateAppearances", 0), "SB": sb, "CS": cs,
                    "SB_ATT": sb + cs, "SB%": sb / (sb + cs) if sb + cs else math.nan}
            if per.kind == "season":
                spr = row_for(board("sprint", per.season), pid)
                vals.update({"SPRINT": dfn.col(spr, "sprint_speed"), "HP1B": dfn.col(spr, "hp_to_1b"),
                             "BOLTS": dfn.col(spr, "bolts")})
            mid = start_d + (end_d - start_d) / 2
            lead = {"Year": per.season, "Period": plabel, "Age": age_years(mid),
                    "Type": "Post" if per.kind == "post" else "Reg", "ORG": "", "LVL >": "MLB", "Pos >": ""}
            rows_b.append({"lead": lead, "vals": vals, "pct": {}, "season_row": per.kind == "season"})

        tabs = st.tabs(list(dfn.BR_TABS))
        for tab, name in zip(tabs, dfn.BR_TABS):
            with tab:
                show(defense_table(dfn.BR_TABS[name], rows_b))
        st.caption("RVA (run value on the bases), leads and grades stay blank until our models are built. "
                   "Sprint speed / home-to-first are full seasons only.")

    # ---------------------------------------------------------------- DEFENSE
    elif sub == "Defense":
        groups = dfn.defense_tabs(primary, games_at) or ["Infield"]
        keep_choice("pp_def", groups)
        grp = seg("Defense view", groups, default=groups[0], key="pp_def", collapsed=True,
                  on_change=lambda: hold("pp_def"))
        grp = grp if grp in groups else groups[0]
        here = dfn.GROUP_POS[grp]
        played = [x for x in here if games_at.get(x, 0) > 0] or ([primary] if primary in here else here)

        c = st.columns([2.3, 1.3, 2.4, 1.0, 1.7])
        with c[0]:
            st.caption("Periods")
            periods = period_picker(f"d_{grp}", years)
        with c[1]:
            pos_pick = "Aggregate"
            if grp != "Catching":
                pos_pick = st.selectbox("Pos", ["Aggregate"] + played, key=f"d_{grp}_pos")
        with c[3]:
            by_team = seg("Split By", ["Level", "Org", "Team"], key=f"d_{grp}_by",
                          help="Level / Org arrive with MiLB") == "Team"
        with c[4]:
            seg("Display", ["Stats", "Percentiles"], default="Stats", key=f"d_{grp}_disp",
                on_change=lambda: hold(f"d_{grp}_disp"))
        positions = here if pos_pick == "Aggregate" else [pos_pick]
        if not periods:
            st.caption("Pick at least one period.")
            st.stop()

        teams_by_year = {}

        def abbr(year, tid):
            if year not in teams_by_year:
                teams_by_year[year] = data.teams_for(year)
            return teams_by_year[year].get(tid, {}).get("abbr", "")

        def game_dates(year):
            log = stats(pid, "hitting", "gameLog", season=year, sportId=1)
            return pd.Series([s_.get("date") for s_ in log])

        def season_boards(year):
            """Savant leaderboard rows for this player (full seasons only)."""
            b_ = {}
            if grp == "Infield":
                oaa_key = "oaa_if" if pos_pick == "Aggregate" else f"oaa:{dfn.POS_NUM[pos_pick]}"
                b_["oaa"] = row_for(board(oaa_key, year), pid)
                frv = [row_for(board(f"frv:{dfn.POS_NUM[x]}", year), pid) for x in positions]
                b_["frv"] = sum(dfn.col(r_, "run") for r_ in frv if r_ is not None) if any(
                    r_ is not None for r_ in frv) else math.nan
            elif grp == "Outfield":
                oaa_key = "oaa_of" if pos_pick == "Aggregate" else f"oaa:{dfn.POS_NUM[pos_pick]}"
                b_["oaa"] = row_for(board(oaa_key, year), pid)
                b_["jump"] = row_for(board("of_jump", year), pid)
                b_["catch_board"] = board("of_catch", year)
                b_["catch"] = row_for(b_["catch_board"], pid)
                frv = [row_for(board(f"frv:{dfn.POS_NUM[x]}", year), pid) for x in positions]
                b_["frv"] = sum(dfn.col(r_, "run") for r_ in frv if r_ is not None) if any(
                    r_ is not None for r_ in frv) else math.nan
            else:
                b_["pop"] = row_for(board("poptime", year), pid)
                b_["frv"] = row_for(board("frv:2", year), pid)
            return b_

        rows_d = []
        for per in sorted(periods, key=period_order):
            if per.kind == "season":
                splits = stats(pid, "fielding", "season", season=per.season, sportId=1)
                start_d, end_d, plabel = date(per.season, 3, 1), date(per.season, 11, 30), "Season"
            elif per.kind == "post":
                splits = stats(pid, "fielding", "season", season=per.season, sportId=1, gameType="P")
                start_d, end_d, plabel = date(per.season, 9, 28), date(per.season, 11, 5), "Postseason"
            else:
                win = period_window(per)
                if not win:
                    continue
                start_d, end_d, plabel = win
                splits = log_window("fielding", per.season, start_d, end_d)
            team_sets = [None]
            if by_team:
                team_sets = sorted({s_.get("team", {}).get("id") for s_ in splits if s_.get("team", {}).get("id")})
            sb = season_boards(per.season) if per.kind == "season" else {}
            for team in team_sets:
                sp = [s_ for s_ in splits if team is None or s_.get("team", {}).get("id") == team]
                m = dfn.mlb_counts(sp, positions)
                if not m["G"]:
                    continue
                mid = start_d + (end_d - start_d) / 2
                lead = {"Year": per.season, "Period": plabel, "Age": age_years(mid),
                        "Type": "Post" if per.kind == "post" else "Reg",
                        "ORG": ", ".join(filter(None, (abbr(per.season, t) for t in m["team_ids"]))),
                        "LVL >": "MLB", "Pos >": m["Pos"]}
                vals = {k: m[k] for k in ("N", "INN", "PO", "A", "E", "DP", "SB", "CS", "CS%", "T_OPPS")}
                pct = {}
                if sb and grp in ("Infield", "Outfield"):
                    vals["OAA"] = dfn.col(sb.get("oaa"), "outs_above_average")
                    vals["FRV"] = sb.get("frv")
                if sb and grp == "Outfield":
                    j = sb.get("jump")
                    vals["REACT_FT"] = dfn.col(j, "reaction")
                    vals["BURST_FT"] = dfn.col(j, "burst")
                    vals["ROUTE_FT"] = dfn.col(j, "rout")
                    cr, cb = sb.get("catch"), sb.get("catch_board")
                    for k in range(1, 6):
                        vals[f"S{k}_N"] = dfn.col(cr, f"{k}star", "opp")
                        rate = dfn.col(cr, f"{k}star", "percent")
                        vals[f"S{k}"] = rate / 100 if rate == rate and rate > 1 else rate
                        pct[f"S{k}"] = dfn.pctile(cb, rate, f"{k}star", "percent")
                if grp == "Catching":
                    fr = framing_model.summarize(catching_daily(per.season), start_d.isoformat(),
                                                 end_d.isoformat(), p.get("pitchHand", {}).get("code", "R"),
                                                 min_n=40 if per.kind == "season" else 10).get(pid, {})
                    vals.update(fr)
                inn = m["INN"]
                per_season = lambda x: x / inn * fielding_model.SEASON_INN if inn and x == x else math.nan
                if grp in ("Infield", "Outfield"):           # our DRS model, any period
                    nums = [dfn.POS_NUM[x] for x in positions]
                    fz = fielding_model.summarize(fielding_daily(per.season), start_d.isoformat(),
                                                  end_d.isoformat(), nums).get(pid, {})
                    vals.update({k: v for k, v in fz.items() if k not in ("IQ_INN",)})
                    if fz:
                        for k_ in range(1, 6):
                            pct.pop(f"S{k_}", None)              # our stars replace Savant's
                        vals["DRS/S"] = per_season(fz["DRS"])
                        board_rates = fielding_model.season_rates(fielding_daily(per.season), nums)
                        if len(board_rates) and vals["DRS/S"] == vals["DRS/S"]:
                            vals["PCT_MLB"] = vals["PCT_LVL"] = 100 * (board_rates < vals["DRS/S"]).mean()
                if grp == "Catching":
                    fd = fielding_daily(per.season)
                    throws = p.get("pitchHand", {}).get("code", "R")
                    cz = catching_model.summarize(fd, start_d.isoformat(), end_d.isoformat(), throws).get(pid, {})
                    iq_catch = {k_: v_ for k_, v_ in cz.items() if not k_.startswith("_")}
                    if cz:                                         # bucket percentiles vs MLB catchers
                        pct.update(catching_model.bucket_percentiles(fd, cz, throws))
                if sb and grp == "Catching":
                    pop, frv = sb.get("pop"), sb.get("frv")
                    vals["B_RS"] = dfn.col(frv, "block")             # Statcast, used only if ours is missing
                    vals["T_RS"] = dfn.col(frv, "throw")
                    vals["POP"] = dfn.col(pop, "pop_2b", exclude=("count", "cs", "sb_", "rank"))
                    vals["XFER"] = dfn.col(pop, "exchange")
                    vals["ARM"] = dfn.col(pop, "arm")
                if grp == "Catching":
                    for k_, v_ in iq_catch.items():                 # ours (any period) over Statcast's seasons
                        if v_ == v_:
                            vals[k_] = v_
                    f_rs, b_rs, t_rs = vals.get("F_RS", math.nan), vals.get("B_RS", math.nan), vals.get("T_RS", math.nan)
                    if all(x == x for x in (f_rs, b_rs, t_rs)):
                        vals["DRS"] = f_rs + b_rs + t_rs
                        vals["DRS/S"] = per_season(vals["DRS"])
                    vals["F_RS/S"], vals["B_RS/S"], vals["T_RS/S"] = per_season(f_rs), per_season(b_rs), per_season(t_rs)
                    parts_ = [vals.get(k_, math.nan) for k_ in ("F_OPPS", "B_OPPS", "T_OPPS")]
                    vals["OPPS"] = sum(x for x in parts_ if x == x) if any(x == x for x in parts_) else math.nan
                rows_d.append({"lead": lead, "vals": vals, "pct": pct, "season_row": per.kind == "season"})

        tabs = st.tabs(list(dfn.TABS[grp]))
        for tab, name in zip(tabs, dfn.TABS[grp]):
            with tab:
                show(defense_table(dfn.TABS[grp][name], rows_d))
        st.caption("DRS, RS, OPA / FPA / CPA and stars come from our fielding model (any period; needs "
                   "python -m pipeline.train_fielding + the data update). Tool grades need player-tracking data that "
                   "isn't public. Statcast columns (OAA, FRV, jump, pop time) are full seasons only; catching DRS adds "
                   "Statcast blocking / throwing, so it shows on season rows.")
        if grp == "Catching":
            st.caption("Framing comes from our own framing model (models/framing.py), so it works for any period. "
                       "It needs the data update: python -m pipeline.update")
            data_note({p_.season for p_ in periods})

    # ---------------------------------------------------------------- PITCHING
    elif sub == "Pitching":
        c = st.columns([2.3, 1.3, 2.4, 1.1, 2.1])
        with c[0]:
            st.caption("Periods")
            periods = period_picker("pit", p_years, season_only_default=True)
        with c[4]:
            scale = seg("Evaluation Method", c4.SCALES, default="20-80", key="pit_scale") or "20-80"
        with c[1]:
            cat = st.selectbox("Stat Category", ["Regular", "Postseason", "Reg + Post"], key="pit_cat")
        with c[2]:
            split_pick = st.multiselect("Splits", list(pm.SPLITS), key="pit_splits",
                                        help="Cluster is waiting on our pitch-cluster model")
        if not periods:
            st.caption("Pick at least one period.")
            st.stop()
        split_dims = [pm.SPLITS[x] for x in split_pick if pm.SPLITS[x] not in pm.MODEL_SPLITS]
        if "Cluster" in split_pick:
            st.caption("Cluster split is blank until our pitch-cluster model is built.")

        keep_types = {"Regular": {"R"}, "Postseason": POSTSEASON, "Reg + Post": {"R"} | POSTSEASON}[cat]
        raw_by_year, post_by_year = {}, {}
        for y in sorted({p_.season for p_ in periods}, reverse=True):
            d_y = pm.prep(statcast_p(pid, f"{y}-03-01", f"{y}-11-30"))
            if not d_y.empty:
                post_by_year[y] = d_y[d_y["game_type"].isin(POSTSEASON)]
                d_y = d_y[d_y["game_type"].isin(keep_types)]
            raw_by_year[y] = d_y

        pitch_opts = ["All"]
        for d_y in raw_by_year.values():
            if not d_y.empty:
                pitch_opts += [x for x in d_y["pitch"].value_counts().index if x not in pitch_opts]
        with c[3]:
            pitch_pick = st.selectbox("Pitch", pitch_opts, key="pit_pitch")
        pitch_f = None if pitch_pick == "All" else pitch_pick

        period_data, rows_p = [], []
        for per in sorted(periods, key=period_order):
            d_y = post_by_year.get(per.season) if per.kind == "post" else raw_by_year.get(per.season)
            if d_y is None or d_y.empty:
                continue
            win = resolve_period(per, pd.Series(d_y["game_date"].astype(str)))
            if not win:
                continue
            start_d, end_d, plabel = win
            part = d_y[(d_y["game_date"] >= start_d) & (d_y["game_date"] <= end_d)]
            if part.empty:
                continue
            period_data.append((f"{per.season} {plabel}", part))
            for labels, vals in pm.table_rows(part, split_dims, pitch_f):
                rows_p.append({"year": per.season, "period": plabel, "labels": labels, "vals": vals,
                               "season_row": per.kind == "season"})

        tabs = st.tabs(["Summary", "Details", "Pillars", "Rates", "Plot", "Usage", "Location", "Trends"])
        for tab, name in zip(tabs[:4], ("Summary", "Details", "Pillars", "Rates")):
            with tab:
                show(pitching_table(pm.TABLE_TABS[name], rows_p, split_dims, scale))

        chart_names = [nm for nm, _ in period_data]

        def chart_data(key):
            if not period_data:
                return pd.DataFrame()
            pick = st.selectbox("Chart period", chart_names, key=key) if len(chart_names) > 1 else chart_names[0]
            d_c = dict(period_data)[pick]
            return d_c[d_c["pitch"] == pitch_f] if pitch_f else d_c

        with tabs[4]:
            d_c = chart_data("pit_plot_period")
            charts = pitch_ui.plot_charts(d_c) if not d_c.empty else []
            cols3 = st.columns(3)
            for col_, ch in zip(cols3, charts):
                with col_:
                    st.altair_chart(ch, use_container_width=True)
        with tabs[5]:
            d_c = chart_data("pit_usage_period")
            ch = pitch_ui.usage_chart(d_c, "count" in split_dims) if not d_c.empty else None
            if ch is not None:
                st.altair_chart(ch, use_container_width="count" not in split_dims)
        with tabs[6]:
            d_c = chart_data("pit_loc_period")
            rows_dims = [x for x in ("hand", "count", "risp") if x in split_dims]
            ch = pitch_ui.location_chart(d_c, rows_dims) if not d_c.empty else None
            if ch is not None:
                st.altair_chart(ch)
            st.caption("Catcher's view. Dot colouring by pitch grade comes with our pitch models.")
        with tabs[7]:
            tc = st.columns([1.6, 1.3, 1.2, 1.2])
            with tc[0]:
                mode = seg("Trend", ["By Appearance", "Last 400 Pit"], default="By Appearance", key="pit_tr_mode",
                           collapsed=True) or "By Appearance"
            with tc[1]:
                prop = st.selectbox("Property", list(pitch_ui.TREND_PROPS), key="pit_tr_prop")
            with tc[2]:
                start_y = st.selectbox("Start Season", sorted(p_years), index=max(0, len(p_years) - 3)
                                       if len(p_years) >= 3 else 0, key="pit_tr_start")
            with tc[3]:
                tr_pitch = st.selectbox("Pitch", pitch_opts, key="pit_tr_pitch")
            frames = []
            for y in sorted(x for x in p_years if x >= start_y):
                d_y = pm.prep(statcast_p(pid, f"{y}-03-01", f"{y}-11-30"))
                if not d_y.empty:
                    frames.append(d_y[d_y["game_type"].isin(keep_types)])
            tf = pitch_ui.trend_frame(pd.concat(frames) if frames else pd.DataFrame(), prop, mode,
                                      None if tr_pitch == "All" else tr_pitch) if frames else pd.DataFrame()
            ch = pitch_ui.trend_chart(tf, prop, mode) if not tf.empty else None
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
            else:
                show(empty_html("No pitches in those seasons."))

        st.caption("Grades, expected rates, shape, command, spin efficiency, seam-shifted movement, zone-adjusted "
                   "approach angles and clusters stay blank until our pitch models are built.")

# =============================================================================
# STATS
# =============================================================================
elif main == "Stats":
    c = st.columns([1, 1, 4])
    lg = c[0].selectbox("League", ["MLB", "MiLB"], key="st_league")
    grp = c[1].selectbox("Type", ["Hitting", "Pitching"], index=1 if b["is_pitcher"] else 0, key="st_group").lower()
    params = {} if lg == "MLB" else {"leagueListId": "milb_all"}
    splits = stats(pid, grp, "yearByYear", **params)
    rows = {}
    for s in splits:
        team = s.get("team", {}).get("name", "") or "Total"
        key = f'{s.get("season", "")} {team}' + (f" ({level_label(s)})" if lg == "MiLB" else "")
        st_ = s.get("stat", {})
        if grp == "hitting":
            rows[key] = hitting.stat_line(st_)
        else:
            rows[key] = {k: st_.get(v, "-") for k, v in (
                ("G", "gamesPlayed"), ("GS", "gamesStarted"), ("W", "wins"), ("L", "losses"),
                ("SV", "saves"), ("IP", "inningsPitched"), ("H", "hits"), ("ER", "earnedRuns"),
                ("HR", "homeRuns"), ("BB", "baseOnBalls"), ("SO", "strikeOuts"), ("ERA", "era"),
                ("WHIP", "whip"), ("AVG", "avg"))}
    show(table_html(rows, "Season"))

# =============================================================================
# GAME LOG
# =============================================================================
elif main == "Game Log":
    c = st.columns([1, 1, 4])
    season = c[0].selectbox("Season", years, key="gl_season")
    grp = c[1].selectbox("Type", ["Hitting", "Pitching"], index=1 if b["is_pitcher"] else 0, key="gl_group").lower()
    log = stats(pid, grp, "gameLog", season=season)
    rows = {}
    for s in reversed(log):
        opp = data.teams_for(season).get(s.get("opponent", {}).get("id"), {}).get("abbr", "")
        key = f'{s.get("date", "")} {"vs" if s.get("isHome") else "@"} {opp}'
        st_ = s.get("stat", {})
        if grp == "hitting":
            line = hitting.stat_line(st_)
            rows[key] = {k: line[k] for k in ("PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "SB", "AVG", "OPS")}
        else:
            rows[key] = {k: st_.get(v, "-") for k, v in (
                ("IP", "inningsPitched"), ("H", "hits"), ("R", "runs"), ("ER", "earnedRuns"),
                ("BB", "baseOnBalls"), ("SO", "strikeOuts"), ("HR", "homeRuns"),
                ("P", "numberOfPitches"), ("ERA", "era"))}
    show(table_html(rows, "Game"))

# =============================================================================
# SANDBOX
# =============================================================================
else:
    show(empty_html("Sandbox is your space to build custom views for this player — tell Claude what you want here."))
