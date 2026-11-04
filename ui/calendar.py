"""Calendar page pieces: day summary (leaders + All Games row) and game list."""
from __future__ import annotations

from datetime import datetime
from html import escape

from core.scoreboard import ET, POSTSEASON, ROUND_LABEL, Game

LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"
HEADSHOT = ("https://img.mlbstatic.com/mlb-photos/image/upload/"
            "d_people:generic:headshot:67:current.png/w_60,q_auto:best/v1/people/{id}/headshot/67/current")
# Statcast abbreviations that differ from MLB's team list
ABBR_FIX = {"ARI": "AZ", "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB",
            "WSN": "WSH", "OAK": "ATH"}
BOARDS = [("exit_velo", "Top Exit Velocity", "MPH", "{:.1f}"),
          ("distance", "Top Distances", "Ft.", "{:.0f}"),
          ("pitch_velo", "Top Pitch Velocity", "MPH", "{:.1f}"),
          ("whiffs", "Swing & Misses", "#", "{:d}")]

CSS = """
<style>
.iq-day{font:13px system-ui,sans-serif;color:#16202b}
.iq-day .leads{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:4px 22px}
.iq-day table{border-collapse:collapse;width:100%}
.iq-day .leads th{font:700 13px system-ui,sans-serif;text-align:left;padding:4px 6px;
  border-bottom:1.5px solid #4a8b96}
.iq-day .leads th.v{text-align:right}
.iq-day .leads td{font-size:13px;padding:3px 6px;border-bottom:1px solid #eef1f3;vertical-align:middle}
.iq-day .leads td.v{text-align:right;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.iq-day .who{display:flex;align-items:center;gap:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.iq-day .who .tl{width:16px;height:16px;object-fit:contain;flex:0 0 auto}
.iq-day .who .hs{width:24px;height:24px;border-radius:50%;object-fit:cover;background:#eef1f3;flex:0 0 auto}
.iq-day .tot{overflow-x:auto;margin-top:10px;-webkit-overflow-scrolling:touch}
.iq-day .tot table{white-space:nowrap;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.iq-day .tot th{font:600 11px system-ui,sans-serif;color:#6a7681;text-align:right;padding:4px 8px;
  border-bottom:1.5px solid #4a8b96}
.iq-day .tot td{font-size:13px;text-align:right;padding:4px 8px;border-bottom:1px solid #e6eaed}
.iq-day .tot th:first-child,.iq-day .tot td:first-child{text-align:left;position:sticky;left:0;background:#fff}
.iq-day .note{font-style:italic;color:#4b5560;font-size:12px;margin:10px 0 18px}
.iq-games{font-family:system-ui,sans-serif;color:#16202b;display:flex;flex-direction:column;gap:12px;margin-top:8px}
.iq-games .tag{display:inline-block;font:700 10px system-ui,sans-serif;letter-spacing:.6px;color:#fff;
  background:#d9534f;padding:3px 6px}
.iq-games .tag.live{background:#1e7d3c}
.iq-games .tag.pre,.iq-games .tag.void{background:#6a7681}
.iq-games .card{display:flex;align-items:center;gap:12px;padding:12px 14px;
  border:1.5px solid #4a8b96;background:#eef6f7;flex-wrap:wrap}
.iq-games .card img{width:30px;height:30px;object-fit:contain}
.iq-games .tn{display:flex;flex-direction:column;line-height:1.1;min-width:92px}
.iq-games .tn small{font-size:10px;color:#4b5560}
.iq-games .tn b{font-size:15px;font-weight:600}
.iq-games .tn.r{align-items:flex-end;text-align:right}
.iq-games .sc{font:400 26px ui-monospace,"SF Mono",Menlo,Consolas,monospace;min-width:34px;text-align:center}
.iq-games .sc.win{color:#d22e2e}
.iq-games .at{color:#9aa4ad;font-size:12px}
.iq-games .links{font-size:12px;margin-top:6px;padding:0 14px 0 0;display:flex;gap:6px;color:#9aa4ad}
.iq-games .links a{color:#16202b;text-decoration:none;font-weight:600}
.iq-games .links a:hover{color:#d22e2e;text-decoration:underline}
.iq-games .card{flex-direction:row}
.iq-games .rnd{margin-left:auto;font:12px ui-monospace,"SF Mono",Menlo,Consolas,monospace;color:#4b5560}
</style>
"""


def _team_id(abbr: str, by_abbr: dict) -> int | None:
    return by_abbr.get(ABBR_FIX.get(abbr, abbr))


def summary_html(summary: dict, names: dict, teams: dict) -> str:
    by_abbr = {t["abbr"]: tid for tid, t in teams.items()}
    boards = []
    for key, title, unit, fmt in BOARDS:
        rows = []
        for r in summary.get(key, []):
            tid = _team_id(r["team"], by_abbr)
            logo = f'<img class="tl" src="{LOGO.format(id=tid)}" alt="">' if tid else ""
            name = escape(names.get(r["player"], str(r["player"])))
            val = fmt.format(int(r["value"]) if fmt == "{:d}" else r["value"])
            rows.append(f'<tr><td><div class="who">{logo}'
                        f'<img class="hs" src="{HEADSHOT.format(id=r["player"])}" alt="">{name}</div></td>'
                        f'<td class="v">{val}</td></tr>')
        boards.append(f'<table><thead><tr><th>{title}</th><th class="v">{unit}</th></tr></thead>'
                      f'<tbody>{"".join(rows)}</tbody></table>')
    tot = summary.get("totals", {})
    head = "".join(f"<th>{escape(k)}</th>" for k in tot)
    vals = "".join(f"<td>{escape(str(v))}</td>" for v in tot.values())
    return (CSS + f'<div class="iq-day"><div class="leads">{"".join(boards)}</div>'
            f'<div class="tot"><table><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody><tr><td>All Games</td>{vals}</tr></tbody></table></div>'
            f'<div class="note">Real-time pitch type classifications are automated and may change '
            f'as data gets reviewed in the future.</div></div>')


def _tag(g: Game) -> tuple[str, str]:
    if g.state == "void":
        return "POSTPONED", "void"
    if g.state == "final":
        extra = f"/{g.inning}" if g.inning and g.inning != g.sched_innings else ""
        return f"FINAL{extra}", "final"
    if g.state == "live":
        half = {"Top": "TOP", "Bottom": "BOT", "Middle": "MID", "End": "END"}.get(g.inning_state or "", "")
        return f"LIVE · {half} {g.inning or ''}".strip(), "live"
    try:
        t = datetime.fromisoformat(g.start_utc.replace("Z", "+00:00")).astimezone(ET)
        return t.strftime("%I:%M %p ET").lstrip("0"), "pre"
    except (ValueError, AttributeError):
        return "TBD", "pre"


def _side(team: int, teams: dict, right: bool = False) -> str:
    t = teams.get(team, {})
    name = f'<div class="tn{" r" if right else ""}"><small>{escape(t.get("location", ""))}</small>' \
           f'<b>{escape(t.get("team_name", "?"))}</b></div>'
    logo = f'<img src="{LOGO.format(id=team)}" alt="">'
    return name + logo if right else logo + name


GAME_VIEWS = [("box", "Box Score"), ("lineup", "Lineup"), ("pbp", "Play by Play")]


def _links(pk: int) -> str:
    """Box Score | Lineup | Play by Play -- each opens the game page in a new tab."""
    return " | ".join(f'<a href="/game?pk={pk}&view={v}" target="_blank" rel="noopener">{label}</a>'
                      for v, label in GAME_VIEWS)


def games_html(games: list[Game], teams: dict) -> str:
    cards = []
    for g in sorted(games, key=lambda g: (g.start_utc, g.pk)):
        label, cls = _tag(g)
        show_score = g.state in ("final", "live")
        a = g.away_score if show_score and g.away_score is not None else ""
        h = g.home_score if show_score and g.home_score is not None else ""
        aw = " win" if g.state == "final" and g.winner == g.away else ""
        hw = " win" if g.state == "final" and g.winner == g.home else ""
        rnd = (f'<span class="rnd">{ROUND_LABEL[g.gtype]} · Game {g.series_game}</span>'
               if g.gtype in POSTSEASON and g.series_game else "")
        cards.append(
            f'<div><span class="tag {cls}">{escape(label)}</span><div class="card">'
            f'{_side(g.away, teams)}<span class="sc{aw}">{a}</span><span class="at">@</span>'
            f'<span class="sc{hw}">{h}</span>{_side(g.home, teams, right=True)}{rnd}'
            f'<div class="links" style="flex-basis:100%">{_links(g.pk)}</div></div></div>')
    return CSS + f'<div class="iq-games">{"".join(cards)}</div>'
