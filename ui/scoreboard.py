"""Top-of-page scoreboard strip (home page only).

    from ui.scoreboard import render_scoreboard
    render_scoreboard()
"""
from __future__ import annotations

from datetime import date, datetime
from html import escape

from core.scoreboard import (CLINCH_LABEL, ET, POSTSEASON, ROUND_LABEL, Game, congrats_for,
                             dedupe, display_date, pick_day, postseason_strip)

LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"
ROUND_NAME = {"F": "Wild Card Series", "D": "Division Series", "L": "LCS", "W": "World Series"}

CSS = """
<style>
.iq-sb{display:flex;align-items:stretch;gap:0;margin:0 0 1rem;overflow-x:auto;
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;color:#16202b;
  -webkit-overflow-scrolling:touch}
.iq-sb>div{border:1.5px solid #4a8b96;margin-left:-1.5px;background:#fff;flex:0 0 auto}
.iq-sb .lbl{display:flex;flex-direction:column;justify-content:center;align-items:center;
  min-width:84px;padding:6px 10px}
.iq-sb .lbl b{font-size:22px;letter-spacing:1px}
.iq-sb .lbl span{font-size:11px;color:#6a7681;margin-top:2px;letter-spacing:.5px}
.iq-sb .champ{display:flex;flex-direction:column;justify-content:center;align-items:center;
  min-width:150px;padding:6px 14px;background:#eef6f7;gap:4px}
.iq-sb .champ .t{font-size:20px;display:flex;align-items:center;gap:8px}
.iq-sb .champ .t span{font-size:18px;font-weight:700}
.iq-sb .champ b{font-size:14px;font-family:system-ui,sans-serif;display:flex;align-items:center;gap:6px}
.iq-sb .champ img{width:26px;height:26px;object-fit:contain}
.iq-sb .g{display:grid;grid-template-columns:1fr auto;gap:0 14px;min-width:190px;padding:6px 12px}
.iq-sb .rows{display:flex;flex-direction:column;justify-content:center;gap:4px}
.iq-sb .row{display:grid;grid-template-columns:26px 58px 26px;align-items:center;gap:6px}
.iq-sb .row img{width:24px;height:24px;object-fit:contain}
.iq-sb .ab{font-size:16px;letter-spacing:1px;line-height:1}
.iq-sb .sc{font-size:16px;text-align:right}
.iq-sb .row.lose{color:#8a949d}
.iq-sb .row.win .sc,.iq-sb .row.win .ab{font-weight:700}
.iq-sb .dots{display:block;font-size:9px;letter-spacing:1px;color:#c49a2c;line-height:1;margin-top:2px}
.iq-sb .dots .o{color:#b9c0c6}
.iq-sb .star{color:#d22e2e}
.iq-sb .st{display:flex;flex-direction:column;justify-content:center;align-items:flex-end;
  font-size:13px;white-space:nowrap}
.iq-sb .st .s{color:#c0392b}
.iq-sb .st .live{color:#1e7d3c;font-weight:700}
.iq-sb .st .sub{color:#4b5560;font-size:12px;margin-top:6px}
.iq-sb .cl{grid-column:1/-1;font-size:11px;color:#d22e2e;margin-top:4px;white-space:nowrap;
  font-family:system-ui,sans-serif}
.iq-sb .off{display:flex;flex-direction:column;justify-content:center;align-items:center;
  min-width:190px;padding:6px 12px;background:#f6f7f8}
.iq-sb .off b{font-size:16px;letter-spacing:2px;color:#4b5560}
.iq-sb .off span{font-size:11px;color:#8a949d;margin-top:4px;font-family:system-ui,sans-serif}
.iq-sb .empty{padding:14px 18px;font-family:system-ui,sans-serif;color:#6a7681}
</style>
"""


def _status(g: Game) -> tuple[str, str]:
    if g.state == "final":
        extra = f"/{g.inning}" if g.inning and g.inning != g.sched_innings else ""
        return f"Final{extra}", "s"
    if g.state == "live":
        half = {"Top": "Top", "Bottom": "Bot", "Middle": "Mid", "End": "End"}.get(g.inning_state or "", "")
        return (f"{half} {g.inning}".strip() if g.inning else "Live"), "live"
    try:
        t = datetime.fromisoformat(g.start_utc.replace("Z", "+00:00")).astimezone(ET)
        return t.strftime("%I:%M %p").lstrip("0").replace(" ", " "), "pre"
    except (ValueError, AttributeError):
        return "TBD", "pre"


def _dots(g: Game, team: int) -> str:
    if not g.series_need:
        return ""
    need, wins = g.series_need, g.series_wins.get(team, 0)
    star = team in g.clinch
    filled = min(wins, need - 1) if star else min(wins, need)
    slots = need - 1 if star else need
    out = "●" * filled + '<span class="o">' + "○" * (slots - filled) + "</span>"
    if star:
        out += '<span class="star">★</span>'
    return f'<span class="dots">{out}</span>'


def _row(g: Game, team: int, score, teams: dict) -> str:
    abbr = escape(teams.get(team, {}).get("abbr", "???"))
    cls = ""
    if g.state == "final" and g.winner:
        cls = "win" if g.winner == team else "lose"
    reg_star = '<span class="star"> ★</span>' if (team in g.clinch and not g.series_need) else ""
    sc = "" if score is None or g.state == "pre" else score
    return (f'<div class="row {cls}"><img src="{LOGO.format(id=team)}" alt="">'
            f'<div><span class="ab">{abbr}{reg_star}</span>{_dots(g, team)}</div>'
            f'<span class="sc">{sc}</span></div>')


def _congrats(c: dict) -> str:
    tip = f'Won the {ROUND_NAME[c["round"]]} {c["wins"]}-{c["played"] - c["wins"]}'
    return (f'<div class="champ" title="{tip}">'
            f'<span class="t">🏆<span>{c["wins"]}/{c["played"]}</span></span>'
            f'<b>Congrats <img src="{LOGO.format(id=c["winner"])}" alt="">!</b></div>')


def build_html(games: list[Game], teams: dict, day: str, today: date,
               congrats: list[dict] | None = None, off_day: bool = False) -> str:
    d = date.fromisoformat(day)
    if games and games[-1].gtype in POSTSEASON:      # label = latest round
        big, small = ROUND_LABEL[games[-1].gtype], f"{d:%b} {d.day}".upper()
    else:
        rel = {0: "TODAY", 1: "YESTERDAY"}.get((today - d).days, d.strftime("%a").upper())
        big, small = f"{d:%b} {d.day}".upper(), rel
    parts = [f'<div class="lbl"><b>{big}</b><span>{small}</span></div>']
    parts += [_congrats(c) for c in (congrats or [])]
    if not games and not off_day:
        parts.append('<div class="empty">No games</div>')
    for g in games:
        label, cls = _status(g)
        sub = f"{ROUND_LABEL[g.gtype]} G{g.series_game}" if g.gtype in POSTSEASON and g.series_game else ""
        tip = f"Moves {g.importance * 100:.1f} pts of {'title' if g.series_need else 'playoff'} odds"
        clinch = ""
        if g.clinch and not g.series_need:           # regular season: say what's at stake
            bits = [f'{escape(teams.get(t, {}).get("abbr", ""))}: {CLINCH_LABEL[c]}'
                    for t, c in g.clinch.items()]
            clinch = f'<div class="cl">★ Win clinches {" · ".join(bits)}</div>'
        parts.append(
            f'<div class="g" title="{tip}"><div class="rows">'
            f'{_row(g, g.away, g.away_score, teams)}{_row(g, g.home, g.home_score, teams)}</div>'
            f'<div class="st"><span class="{cls}">{label}</span><span class="sub">{sub}</span></div>'
            f'{clinch}</div>')
    if off_day:
        parts.append('<div class="off"><b>OFF DAY</b><span>No games scheduled</span></div>')
    return CSS + f'<div class="iq-sb">{"".join(parts)}</div>'


# ----------------------------------------------------------------------------
# Streamlit wiring
# ----------------------------------------------------------------------------
def render_scoreboard() -> None:
    import streamlit as st
    from ui import data

    show = st.html if hasattr(st, "html") else (lambda h: st.markdown(h, unsafe_allow_html=True))

    def overlay(games, day):                         # live score / inning
        now = data.live(day)
        for g in games:
            cur = now.get(g.pk)
            if cur:
                g.state, g.winner = cur.state, cur.winner
                g.home_score, g.away_score = cur.home_score, cur.away_score
                g.inning, g.inning_state, g.sched_innings = cur.inning, cur.inning_state, cur.sched_innings

    try:
        target = display_date()
        season = data.current_season(target)
        games_all = data.season_games(season)
        teams = data.teams_for(season)

        # Postseason (and the winter after it): last 5 postseason games
        strip = postseason_strip(games_all, target.isoformat())
        if strip:
            games, off_day = strip
            overlay(games, target.isoformat())
            congrats = congrats_for(dedupe(games_all), target.isoformat())
            show(build_html(games, teams, target.isoformat(), target, congrats, off_day))
            return

        # Regular season: top 5 games of the day by importance
        day = pick_day(games_all, target)
        if day is None:
            return
        top = data.ranked(season, day)
        overlay(top, day)
        show(build_html(top, teams, day, target))
    except Exception as e:                           # never take the whole page down
        st.caption(f"Scoreboard unavailable right now ({type(e).__name__}).")
