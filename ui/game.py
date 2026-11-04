"""Game page views: Box Score, Lineup, Play by Play (HTML, same style as Standings)."""
from __future__ import annotations

from html import escape

ORD = {1: "1st", 2: "2nd", 3: "3rd"}
HEADSHOT = ("https://img.mlbstatic.com/mlb-photos/image/upload/"
            "d_people:generic:headshot:67:current.png/w_60,q_auto:best/v1/people/{id}/headshot/67/current")

CSS = """
<style>
.iq-g{font:13px system-ui,sans-serif;color:#16202b}
.iq-g h4{font:700 15px system-ui,sans-serif;margin:14px 0 6px;display:flex;align-items:center;gap:8px}
.iq-g h4 img{width:24px;height:24px;object-fit:contain}
.iq-g .wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
.iq-g table{border-collapse:collapse;width:100%;white-space:nowrap}
.iq-g th{font:600 11px system-ui,sans-serif;color:#6a7681;text-align:right;padding:5px 8px;
  border-bottom:1.5px solid #4a8b96}
.iq-g td{font:13px ui-monospace,"SF Mono",Menlo,Consolas,monospace;text-align:right;padding:4px 8px;
  border-bottom:1px solid #e6eaed}
.iq-g th:first-child,.iq-g td:first-child{text-align:left;font-family:system-ui,sans-serif}
.iq-g td.sub{padding-left:22px;color:#4b5560}
.iq-g td .pos{color:#6a7681;font-size:11px;margin-left:6px}
.iq-g.ls table{width:auto;min-width:420px}
.iq-g.ls td,.iq-g.ls th{min-width:22px;text-align:center}
.iq-g.ls td.tot,.iq-g.ls th.tot{font-weight:700;border-left:1px solid #e6eaed}
.iq-g.ls td:first-child{display:flex;align-items:center;gap:8px;font-weight:600}
.iq-g.ls img{width:20px;height:20px;object-fit:contain}
.iq-g .lu td,.iq-g .lu th{font-family:system-ui,sans-serif;text-align:left}
.iq-g .lu td.n{width:28px;color:#6a7681;font-family:ui-monospace,monospace}
.iq-g .lu .who{display:flex;align-items:center;gap:8px}
.iq-g .lu .who img{width:28px;height:28px;border-radius:50%;object-fit:cover;background:#eef1f3}
.iq-g .half{font:700 14px system-ui,sans-serif;margin:16px 0 4px;padding-bottom:4px;
  border-bottom:1.5px solid #4a8b96;display:flex;gap:8px;align-items:center}
.iq-g .half img{width:20px;height:20px;object-fit:contain}
.iq-g details{border-bottom:1px solid #e6eaed;padding:6px 2px}
.iq-g summary{cursor:pointer;list-style:none;display:grid;grid-template-columns:130px 1fr auto;gap:10px;align-items:baseline}
.iq-g summary::-webkit-details-marker{display:none}
.iq-g summary .ev{font-weight:700}
.iq-g details.score summary .ev{color:#d22e2e}
.iq-g summary .sc{font:12px ui-monospace,monospace;color:#d22e2e;white-space:nowrap}
.iq-g .pitches{margin:6px 0 2px 140px;font:12px ui-monospace,monospace;color:#4b5560}
.iq-g .pitches div{padding:1px 0}
.iq-g .empty{color:#6a7681;padding:10px 0}
@media (max-width:640px){.iq-g summary{grid-template-columns:1fr}.iq-g .pitches{margin-left:8px}}
</style>
"""
LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"


def _table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(rows_cell for rows_cell in r) + "</tr>" for r in rows)
    return f'<div class="wrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def linescore_html(ls: dict, teams: dict, away: int, home: int) -> str:
    head = [""] + [str(i + 1) for i in range(ls["n"])] + ["R", "H", "E"]
    rows = []
    for side, tid in (("away", away), ("home", home)):
        t = ls[side]
        rows.append([f'<td><img src="{LOGO.format(id=tid)}" alt="">{escape(teams.get(tid, {}).get("abbr", ""))}</td>']
                    + [f"<td>{x}</td>" for x in t["innings"]]
                    + [f'<td class="tot">{t["R"]}</td>', f'<td>{t["H"]}</td>', f'<td>{t["E"]}</td>'])
    return CSS + f'<div class="iq-g ls">{_table(head, rows)}</div>'


def box_html(bat: list[dict], pit: list[dict], team: dict, tid: int) -> str:
    title = f'<h4><img src="{LOGO.format(id=tid)}" alt="">{escape(team.get("name", ""))}</h4>'
    if not bat and not pit:
        return CSS + f'<div class="iq-g">{title}<div class="empty">Box score appears once the game starts.</div></div>'
    bcols = ["AB", "R", "H", "RBI", "BB", "SO", "HR", "AVG"]
    brows = [[f'<td class="{"sub" if b["sub"] else ""}">{escape(b["name"])}<span class="pos">{b["pos"]}</span></td>']
             + [f"<td>{b[c]}</td>" for c in bcols] for b in bat]
    tot = ["<td><b>Totals</b></td>"] + [f"<td><b>{sum(b[c] for b in bat)}</b></td>" for c in bcols[:-1]] + ["<td></td>"]
    pcols = ["IP", "H", "R", "ER", "BB", "SO", "HR", "P", "ERA"]
    prows = [[f"<td>{escape(p['name'])}</td>"] + [f"<td>{p[c]}</td>" for c in pcols] for p in pit]
    return (CSS + f'<div class="iq-g">{title}{_table(["Batters"] + bcols, brows + [tot])}'
            f'<div style="height:10px"></div>{_table(["Pitchers"] + pcols, prows)}</div>')


def lineup_html(lu: dict, team: dict, tid: int) -> str:
    title = f'<h4><img src="{LOGO.format(id=tid)}" alt="">{escape(team.get("name", ""))}</h4>'
    rows = [[f'<td class="n">{b["spot"]}</td>',
             f'<td><div class="who"><img src="{HEADSHOT.format(id=b["id"])}" alt="">{escape(b["name"])}</div></td>',
             f'<td>{b["pos"]}</td>', f'<td>{b["bats"]}</td>'] for b in lu["nine"]]
    sp = lu.get("pitcher")
    if sp:
        tag = "Probable" if sp.get("probable") else "SP"
        rows.append([f'<td class="n">{tag}</td>',
                     f'<td><div class="who"><img src="{HEADSHOT.format(id=sp["id"])}" alt="">{escape(sp["name"])}</div></td>',
                     "<td>P</td>", f'<td>Throws {sp["throws"]}</td>'])
    body = _table(["#", "Player", "Pos", "B/T"], rows, "lu") if rows else ""
    if not lu["nine"]:
        body = '<div class="empty">Lineup not posted yet.</div>' + body
    return CSS + f'<div class="iq-g">{title}{body}</div>'


def _mph(v) -> str:
    return f"{v:.1f} mph" if v else ""


def plays_html(halves: list[dict], teams: dict, away: int, home: int, scoring_only: bool) -> str:
    out = []
    for h in halves:
        plays = [p for p in h["plays"] if p["scoring"] or not scoring_only]
        if not plays:
            continue
        batting = away if h["half"] == "top" else home
        name = teams.get(batting, {}).get("team_name", "")
        inning = ORD.get(h["inning"], f'{h["inning"]}th')
        half = "Top" if h["half"] == "top" else "Bottom"
        out.append(f'<div class="half"><img src="{LOGO.format(id=batting)}" alt="">'
                   f'{half} {inning} · {escape(name)}</div>')
        for p in plays:
            a, hm = p["score"]
            sc = (f'<span class="sc">{teams.get(away, {}).get("abbr", "")} {a} – '
                  f'{teams.get(home, {}).get("abbr", "")} {hm}</span>') if p["scoring"] else "<span></span>"
            pitches = "".join(
                f'<div>{q["n"] or ""}. {escape(q["type"] or "")} {_mph(q["mph"])}'
                f' · {escape(q["call"])} ({q["count"]})</div>' for q in p["pitches"])
            out.append(f'<details class="{"score" if p["scoring"] else ""}"><summary>'
                       f'<span class="ev">{escape(p["event"] or "In progress")}</span>'
                       f'<span>{escape(p["desc"] or p["batter"] + " batting")}</span>{sc}</summary>'
                       f'<div class="pitches">Pitcher: {escape(p["pitcher"])}{pitches}</div></details>')
    if not out:
        out.append('<div class="empty">No plays yet.</div>')
    return CSS + f'<div class="iq-g">{"".join(out)}</div>'
