"""Division and Wild Card standings pages."""
from __future__ import annotations

from html import escape

from core.mlb_api import AL, NL
from core.standings import (CLINCH_TEXT, EARLIEST, LEAGUE_NAME, division_tables,
                            next_games, wc_spots, wildcard_table)

LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"

# (header, row key, tooltip)
COLS = [("W", "w", "Wins"), ("L", "l", "Losses"), ("PCT", "pct", "Winning Percentage"),
        ("GB", "gb", "Games Back"), ("E#", "e", "Elimination Number"),
        ("L10", "l10", "Record in the Last 10 Games"), ("STRK", "strk", "Current Streak"),
        ("RS", "rs", "Runs Scored"), ("RA", "ra", "Runs Allowed"),
        ("DIFF", "diff", "Run Differential (Runs Scored - Runs Allowed)"),
        ("X-W/L", "xwl", "Expected Win-Loss Record based on Runs Scored and Runs Allowed"),
        ("HOME", "home", "Record at Home"), ("AWAY", "away", "Record while Away"),
        ("≥.500", "winners", "Record against ≥.500 teams"), ("NEXT GAME", "next", "Next Game")]

CSS = """
<style>
.iq-st{overflow-x:auto;margin:0 0 18px;-webkit-overflow-scrolling:touch}
.iq-st h4{font:700 15px system-ui,sans-serif;margin:0 0 6px;color:#16202b;letter-spacing:.3px}
.iq-st table{border-collapse:collapse;width:100%;font:13px ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  color:#16202b;white-space:nowrap}
.iq-st th{font:600 11px system-ui,sans-serif;color:#6a7681;text-align:right;padding:6px 8px;
  border-bottom:1.5px solid #4a8b96;background:#fff;cursor:help}
.iq-st td{text-align:right;padding:5px 8px;border-bottom:1px solid #e6eaed}
.iq-st th:first-child,.iq-st td:first-child{text-align:left;position:sticky;left:0;background:#fff;z-index:1}
.iq-st td.tm{display:flex;align-items:center;gap:8px;min-width:96px}
.iq-st td.tm img{width:22px;height:22px;object-fit:contain}
.iq-st .ci{color:#d22e2e;font-weight:700;cursor:help}
.iq-st .pos{color:#1e7d3c}.iq-st .neg{color:#c0392b}
.iq-st tr.cut td{border-bottom:2px dashed #d22e2e}
.iq-legend{font:12px system-ui,sans-serif;color:#4b5560;columns:2;max-width:760px}
.iq-legend div{break-inside:avoid;margin:2px 0}
.iq-legend b{display:inline-block;min-width:78px;color:#16202b}
</style>
"""


def _team_cell(r: dict, teams: dict) -> str:
    abbr = escape(teams.get(r["team"], {}).get("abbr", "?"))
    ci = (f'<span class="ci" title="{CLINCH_TEXT[r["clinch"]]}">{r["clinch"]}-</span>'
          if r.get("clinch") else "")
    return f'<td class="tm"><img src="{LOGO.format(id=r["team"])}" alt="">{ci}{abbr}</td>'


def _table(title: str, rows: list[dict], teams: dict, nxt: dict, wild: bool = False,
           cut_after: int | None = None) -> str:
    head = "".join(f'<th title="{tip}">{"WCGB" if (wild and k == "gb") else h}</th>'
                   for h, k, tip in COLS)
    body = []
    for i, r in enumerate(rows):
        r = dict(r, next=nxt.get(r["team"], "-"))
        if wild:
            r["gb"], r["e"] = r["wcgb"], r["wc_e"]
        cells = []
        for _, k, _ in COLS:
            v = escape(str(r.get(k, "-")))
            cls = ' class="pos"' if k == "diff" and v.startswith("+") else \
                  ' class="neg"' if k == "diff" and v.startswith("-") and v != "-" else ""
            cells.append(f"<td{cls}>{v}</td>")
        cut = ' class="cut"' if cut_after is not None and i == cut_after - 1 and len(rows) > cut_after else ""
        body.append(f"<tr{cut}>{_team_cell(r, teams)}{''.join(cells)}</tr>")
    return (f'<div class="iq-st"><h4>{escape(title)}</h4><table><thead><tr><th>TEAM</th>{head}'
            f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def division_html(rows, teams, league, nxt) -> str:
    return CSS + "".join(_table(name, t, teams, nxt) for name, t in division_tables(rows, league))


def wildcard_html(rows, teams, league, nxt, season: int = 2026) -> str:
    leaders, race = wildcard_table(rows, league)
    short = "AL" if league == AL else "NL"
    return CSS + _table(f"{short} Division Leaders", leaders, teams, nxt) + \
        _table(f"{short} Wild Card", race, teams, nxt, wild=True, cut_after=wc_spots(season))


def legend_html() -> str:
    items = [f"<div><b>{k}</b>{v}</div>" for k, v in CLINCH_TEXT.items()]
    items += [f"<div><b>{h}</b>{tip}</div>" for h, _, tip in COLS]
    return CSS + f'<div class="iq-legend">{"".join(items)}</div>'


# ----------------------------------------------------------------------------
# Streamlit page: "2026 [Division | Wild Card] Standings"  ...  [date]
# ----------------------------------------------------------------------------
def resolve(games, picked: str) -> tuple[int, str | None]:
    """(season, as-of date) to request for a picked day; as-of None = latest.
    Before a season's first game -> last season's final standings.
    After a season's last regular-season game -> that season's final standings."""
    year = int(picked[:4])
    played = [g.day for g in games(year) if g.gtype == "R" and g.played]
    if not played or min(played) > picked:
        return year - 1, None
    if picked >= max(played):
        return year, None
    return year, picked


TITLE_CSS = """
<style>
h2.iq-title{margin:0;padding:0}
h2.iq-title a.iq-tog,h2.iq-title a.iq-tog:hover,h2.iq-title a.iq-tog:visited{
  color:inherit !important;text-decoration:none !important;font:inherit;cursor:pointer}
h2.iq-title a:not(.iq-tog){display:none}          /* hide Streamlit's # anchor icon */
</style>
"""
VIEWS = {"division": "Division", "wildcard": "Wild Card"}


def render_standings() -> None:
    """Title reads '2026 Division Standings'. Clicking 'Division' switches to
    'Wild Card' (and back). The view and date live in the URL (?view=&date=),
    so a view can be bookmarked or shared."""
    from datetime import date, datetime
    from urllib.parse import urlencode

    import streamlit as st

    from core.scoreboard import ET
    from ui import data

    show = st.html if hasattr(st, "html") else (lambda h: st.markdown(h, unsafe_allow_html=True))
    today = datetime.now(ET).date()
    earliest = date.fromisoformat(EARLIEST)

    qp = st.query_params
    view = qp.get("view", "division")
    view = view if view in VIEWS else "division"
    try:
        picked = min(max(date.fromisoformat(qp.get("date", "")), earliest), today)
    except ValueError:
        picked = today

    def _date_changed():
        d = st.session_state["standings_date"]
        if d == today:
            st.query_params.pop("date", None)
        else:
            st.query_params["date"] = d.isoformat()

    try:
        title_col, date_col = st.columns([5, 1.3], vertical_alignment="bottom")
    except TypeError:                                   # older Streamlit
        title_col, date_col = st.columns([5, 1.3])
    with date_col:
        picked = st.date_input("Standings as of", value=picked, key="standings_date",
                               min_value=earliest, max_value=today, format="MM/DD/YYYY",
                               label_visibility="collapsed", on_change=_date_changed)

    try:
        season, on = resolve(data.season_games, picked.isoformat())
        rows = data.standings_for(season, on)
        if not rows and on:                              # feed has nothing for that day
            rows, on = data.standings_for(season, None), None
        teams = data.teams_for(season)
        nxt = next_games(data.season_games(season), on or "9999-12-31", teams)
    except Exception as e:
        st.caption(f"Standings unavailable right now ({type(e).__name__}).")
        return

    other = "wildcard" if view == "division" else "division"
    params = {"view": other}
    if picked != today:
        params["date"] = picked.isoformat()
    with title_col:
        st.markdown(
            TITLE_CSS +
            f'<h2 class="iq-title">{season} '
            f'<a class="iq-tog" href="?{urlencode(params)}" target="_self" '
            f'title="Switch to {VIEWS[other]} standings">{VIEWS[view]}</a> Standings</h2>',
            unsafe_allow_html=True)
    if on is None:
        st.caption(f"Final {season} regular-season standings")
    else:
        st.caption(f"As of the end of {picked:%B} {picked.day}, {picked.year}")

    if not rows:
        st.info("No standings for that date.")
        return
    left, right = st.columns(2)                          # AL left, NL right
    for col, league in ((left, AL), (right, NL)):
        with col:
            st.markdown(f"**{LEAGUE_NAME[league]}**")
            if view == "division":
                show(division_html(rows, teams, league, nxt))
            else:
                show(wildcard_html(rows, teams, league, nxt, season))
    with st.expander("Clinch indicators & column key"):
        show(legend_html())
