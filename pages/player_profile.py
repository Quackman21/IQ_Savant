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
from core import hitting, league_ref, mlb_api, pitching
from core.statcast_clean import clean
from core.player import bio, draft_entry, position_games, signing_entry, summary_tabs
from ui import data
from core import core4 as c4
from ui.core4 import table_html as core4_table
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
        fn = lambda: pb.statcast_fielding_run_value(year, int(kind[4:]), 1)
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
def seg(label, options, default=None, key=None, disabled=False, help=None, collapsed=False,
        selection_mode="single"):
    vis = "collapsed" if collapsed else "visible"
    if hasattr(st, "segmented_control"):
        v = st.segmented_control(label, options, default=default, key=key, disabled=disabled,
                                 help=help, label_visibility=vis, selection_mode=selection_mode)
    elif selection_mode == "multi":
        v = st.multiselect(label, options, default=default or [], key=key, disabled=disabled, help=help)
    else:
        v = st.radio(label, options, index=options.index(default) if default in options else 0,
                     horizontal=True, key=key, disabled=disabled, help=help, label_visibility=vis)
    return v


ID_COLS = ("player_id", "entity_id", "id", "catcher_id", "mlbam_id", "batter")
SKIP = {"year", "age", "team_id", "player_id", "entity_id", "id", "catcher_id", "mlbam_id", "batter",
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
primary = p.get("primaryPosition", {}).get("abbreviation", "")
games_at = position_games(field, since=TODAY.year - 2)
SUB_TABS = summary_tabs(primary, games_at, career_pa)

# tab bars styled like the reference: main = bold uppercase with underline,
# sub = grey "A | B | C" links with the active one green and underlined
st.markdown("""<style>
.st-key-pp_main [data-testid="stButtonGroup"]{display:flex;justify-content:center;
  border-bottom:1px solid #e3e7ea;width:100%}
.st-key-pp_main button{background:transparent!important;border:none!important;border-radius:0!important;
  box-shadow:none!important;padding:10px 20px!important;border-bottom:3px solid transparent!important}
.st-key-pp_main button p{font-weight:700!important;letter-spacing:.8px;text-transform:uppercase;
  color:#8a949d;font-size:14px!important}
.st-key-pp_main button[data-testid$="Active"]{border-bottom-color:#16202b!important}
.st-key-pp_main button[data-testid$="Active"] p{color:#16202b}
.st-key-pp_sub{background:#f1f3f2;padding:8px 10px;margin:6px 0 10px}
.st-key-pp_sub [data-testid="stButtonGroup"]{gap:0}
.st-key-pp_sub button{background:transparent!important;border:none!important;border-radius:0!important;
  box-shadow:none!important;padding:2px 14px!important}
.st-key-pp_sub button p{font-weight:600!important;letter-spacing:.4px;text-transform:uppercase;
  color:#8a949d;font-size:14px!important}
.st-key-pp_sub button + button{border-left:1px solid #b9c0c6!important}
.st-key-pp_sub button[data-testid$="Active"] p{color:#3fa37f;text-decoration:underline;text-underline-offset:4px}
.stTabs [data-baseweb="tab-list"]{background:#f5f6f6;gap:0}
.stTabs [data-baseweb="tab"]{flex:1;justify-content:center;padding:10px 6px}
.stTabs [data-baseweb="tab"] p{font-weight:700;color:#8a949d;text-transform:uppercase;letter-spacing:.4px}
.stTabs [aria-selected="true"]{background:#fff;border:1px solid #dfe3e6;border-bottom:none;border-radius:6px 6px 0 0}
.stTabs [aria-selected="true"] p{color:#16202b}
</style>""", unsafe_allow_html=True)

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
        seasons_raw = {}
        for y in sorted({p_.season for p_ in periods}, reverse=True):
            df_y = statcast(pid, f"{y}-03-01", f"{y}-11-30")
            if not df_y.empty and "game_type" in df_y:
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

    # ---------------------------------------------------------------- CATCHING
    elif sub == "Catching":
        c = st.columns([1.6, 1.6, 2.2, 1.6])
        with c[0]:
            st.caption("Periods")
            periods = sorted({p_.season for p_ in period_picker("ca", years, by_month=False)}, reverse=True)
        with c[2]:
            seg("Split By", ["Level", "Org", "Team"], key="ca_by", disabled=True, help="Coming soon")
        with c[3]:
            disp = seg("Display", ["Stats", "Percentiles"], default="Stats", key="ca_disp") or "Stats"

        def show_board(kind, match=None):
            if disp == "Stats":
                show(table_html(board_stats(kind, periods, pid, match), "Season"))
            else:
                show(percentile_html(board_percentiles(kind, periods, pid, match)))

        tabs = st.tabs(["Overview", "Framing", "Blocking", "Throwing"])
        with tabs[0]:
            show_board("framing", ("runs", "strike_rate", "rv"))
            show_board("poptime", ("pop_2b", "arm", "exchange"))
        with tabs[1]:
            show_board("framing")
        with tabs[2]:
            show(empty_html("Blocking is set up and waiting — Savant's blocking leaderboard isn't in pybaseball yet."))
        with tabs[3]:
            show_board("poptime")
        if not any(row_for(board("framing", y), pid) is not None for y in periods if not board("framing", y).empty):
            st.caption("Catching data only exists for players who caught in those seasons.")

    # ---------------------------------------------------------------- BASERUNNING
    elif sub == "Baserunning":
        c = st.columns([1.6, 1.6, 2.2, 1.6])
        with c[0]:
            st.caption("Periods")
            periods = sorted({p_.season for p_ in period_picker("br", years, by_month=False)}, reverse=True)
        with c[2]:
            seg("Split By", ["Level", "Org", "Team"], key="br_by", disabled=True, help="Coming soon")
        with c[3]:
            disp = seg("Display", ["Stats", "Percentiles"], default="Stats", key="br_disp") or "Stats"

        tabs = st.tabs(["Advancing", "Base Stealing", "Tools"])
        with tabs[0]:
            show(empty_html("Advancing (extra bases taken) is set up and waiting for a data source."))
        with tabs[1]:
            rows = {}
            for y in periods:
                s = sum_stats(stats(pid, "hitting", "season", season=y, sportId=1))
                if s:
                    sb, cs = s.get("stolenBases", 0), s.get("caughtStealing", 0)
                    rows[str(y)] = {"SB": sb, "CS": cs, "Att": sb + cs,
                                    "SB%": f"{100 * sb / (sb + cs):.1f}" if sb + cs else "-",
                                    "G": s.get("gamesPlayed", 0)}
            show(table_html(rows, "Season"))
        with tabs[2]:
            if disp == "Stats":
                show(table_html(board_stats("sprint", periods, pid), "Season"))
            else:
                show(percentile_html(board_percentiles("sprint", periods, pid)))

    # ---------------------------------------------------------------- INFIELD / OUTFIELD
    elif sub in ("Infield", "Outfield"):
        infield = sub == "Infield"
        k = "if" if infield else "of"
        c = st.columns([1.6, 1.6, 2.2, 1.6])
        with c[0]:
            st.caption("Periods")
            periods = sorted({p_.season for p_ in period_picker(k, years, by_month=False)}, reverse=True)
        with c[2]:
            seg("Split By", ["Level", "Org", "Team"], key=f"{k}_by", disabled=True, help="Coming soon")
        with c[3]:
            disp = seg("Display", ["Stats", "Percentiles"], default="Stats", key=f"{k}_disp") or "Stats"
        here = INFIELD_POS if infield else OUTFIELD_POS
        played = [pos for pos in here if games_at.get(pos, 0) > 0]
        if not played and primary in here:
            played = [primary]
        st.caption("Positions played (last 3 seasons): " +
                   (", ".join(f"{pos} ({games_at[pos]} G)" for pos in here if games_at.get(pos)) or primary))

        def show_board(kind, match=None):
            if disp == "Stats":
                show(table_html(board_stats(kind, periods, pid, match), "Season"))
            else:
                show(percentile_html(board_percentiles(kind, periods, pid, match)))

        def run_value():
            rows = {}
            for y in periods:
                for pos in played:
                    num = POS_NUM.get(pos)
                    df = board(f"frv:{num}", y) if num else pd.DataFrame()
                    r = row_for(df, pid) if not df.empty else None
                    if r is not None:
                        rows[f"{y} {pos}"] = {pretty(cn): (round(float(r[cn]), 2) if pd.notna(r[cn]) else "-")
                                              for cn in numeric_cols(df)}
            show(table_html(rows, "Season"))

        names = ["Overview", "Outs Above Average", "Run Value"] if infield else \
            ["Overview", "Outs Above Average", "Jump", "Catch Probability", "Directional"]
        tabs = st.tabs(names)
        with tabs[0]:
            show_board(f"oaa_{k}", ("outs_above_average", "fielding_runs", "success_rate"))
            if not infield:
                show_board("of_jump", ("jump", "reaction", "burst", "route"))
        with tabs[1]:
            show_board(f"oaa_{k}")
        if infield:
            with tabs[2]:
                run_value()
        else:
            with tabs[2]:
                show_board("of_jump")
            with tabs[3]:
                show_board("of_catch")
            with tabs[4]:
                show_board("of_dir")

    # ---------------------------------------------------------------- PITCHING
    elif sub == "Pitching":
        c = st.columns([1.5, 1.1, 1.2, 0.6, 1.2, 0.9])
        with c[0]:
            mode = seg("Mode", ["Game", "Season", "Dates"], default="Season", key="pit_mode", collapsed=True) or "Season"
        with c[1]:
            season = st.selectbox("Seasons", p_years, key="pit_season")
        with c[2]:
            cat = st.selectbox("Stat Category", ["Regular", "Postseason"], key="pit_cat")
        with c[4]:
            split = seg("Split", ["Platoon"], key="pit_split")
        with c[5]:
            st.write("")
            avgs = st.toggle("Show Avgs", key="pit_avgs")

        gtype = "R" if cat == "Regular" else "P"
        start, end, label = f"{season}-03-01", f"{season}-11-30", str(season)
        if mode == "Dates":
            rng = st.date_input("Dates", [date(season, 4, 1), min(date(season, 9, 30), TODAY)],
                                format="MM/DD/YYYY", key="pit_dates")
            if len(rng) == 2:
                start, end = rng[0].isoformat(), rng[1].isoformat()
                label = f"{rng[0].month}/{rng[0].day}–{rng[1].month}/{rng[1].day}"
        elif mode == "Game":
            log = stats(pid, "pitching", "gameLog", season=season, gameType=gtype, sportId=1)
            if log:
                g = st.selectbox("Game", list(reversed(log)), key="pit_game",
                                 format_func=lambda s: f'{s.get("date", "")} '
                                 f'{"vs" if s.get("isHome") else "@"} {s.get("opponent", {}).get("name", "")}')
                start = end = g.get("date", start)
                label = g.get("date", "")

        sc = statcast_p(pid, start, end)
        if not sc.empty and "game_type" in sc:
            sc = sc[sc["game_type"].eq("R")] if gtype == "R" else sc[sc["game_type"].isin(POSTSEASON)]

        def rows_for(fn):
            if split == "Platoon":
                return {f"{label} {k}" if k != "All" else label: v for k, v in pitching.by_platoon(sc, fn).items()}
            return {label: fn(sc)}

        tabs = st.tabs(["Overview", "Arsenal", "Results", "Percentiles"])
        with tabs[0]:
            if mode == "Season":
                splits = stats(pid, "pitching", "season", season=season, gameType=gtype, sportId=1)
            else:
                splits = stats(pid, "pitching", "byDateRange", startDate=start, endDate=end, gameType=gtype, sportId=1)
            line = {label: pitching.stat_line(s.get("stat", {})) for s in splits[-1:]}
            if avgs:
                lr = pitching.league_rates(league_p(season))
                line["MLB Avg"] = {k: lr.get(k, "-") for k in next(iter(line.values()), lr)}
            show(table_html(line, "", {"MLB Avg"}))
            show(table_html(rows_for(pitching.overview)))
        with tabs[1]:
            show(table_html(pitching.arsenal(sc), "Pitch"))
            st.caption("IVB / HB in inches (catcher's view). Velo in mph, spin in rpm, extension in feet.")
        with tabs[2]:
            show(table_html(pitching.results(sc), "Pitch"))
        with tabs[3]:
            show(percentile_html(board_percentiles("p_percentiles", [season], pid, already=True)))
            st.caption(f"Savant percentile ranks vs. MLB pitchers, {season}.")

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
