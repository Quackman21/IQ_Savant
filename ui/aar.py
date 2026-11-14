"""AAR page pieces (HTML + inline SVG, same look as the rest of the site): header, box line, metric tiles,
zone / spray / point-of-contact charts, and collapsible plate appearances (open as many as you want)."""
from __future__ import annotations

import math
from html import escape

import pandas as pd

from core.aar import SAVANT_VIDEO

TEAL, INK, MUTED, LINE = "#4a8b96", "#16202b", "#6a7681", "#e6eaed"
SWING, TAKE = "#d9822b", "#5b6bc8"
RES_COLOR = {"Single": "#2bb3c0", "Double": "#e0b300", "Triple": "#8e5bd0", "Home run": "#3a9a4f"}
OUT_COLOR = "#9aa5ae"
PT_COLOR = {"FF": "#c0392b", "FA": "#c0392b", "SI": "#e67e22", "FC": "#8e5b3a", "SL": "#c9a400", "ST": "#b8860b",
            "SV": "#b8860b", "CU": "#2e6fd1", "KC": "#2e6fd1", "CS": "#2e6fd1", "CH": "#2e9b57", "FS": "#16a2a8",
            "FO": "#16a2a8", "KN": "#666", "EP": "#666"}
DEC_COLOR = {"Great": ("#1f7a3a", "#fff"), "Good": ("#cdebd5", INK), "Neutral": ("#eef1f3", MUTED),
             "Bad": ("#f6d3d0", INK), "Poor": ("#b8322a", "#fff")}

CSS = f"""
<style>
.aar{{font:13px system-ui,sans-serif;color:{INK}}}
.aar .card{{border:1px solid {LINE};border-radius:10px;padding:10px 12px;background:#fff}}
.aar .hdr{{display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
.aar .hdr img.hs{{width:56px;height:56px;border-radius:50%;object-fit:cover;background:#eef1f3}}
.aar .hdr .nm{{font:700 22px system-ui,sans-serif}}
.aar .hdr .sub{{color:{MUTED};font-size:12px}}
.aar .hdr .gm{{margin-left:auto;display:flex;align-items:center;gap:10px;font:700 20px system-ui,sans-serif}}
.aar .hdr .gm img{{width:30px;height:30px;object-fit:contain}}
.aar .hdr .gm .d{{font:12px system-ui,sans-serif;color:{MUTED};text-align:center;line-height:1.3}}
.aar table.line{{border-collapse:collapse;width:100%;margin-top:8px}}
.aar table.line th{{font:600 11px system-ui,sans-serif;color:{MUTED};padding:4px;border-bottom:1.5px solid {TEAL}}}
.aar table.line td{{font:600 16px ui-monospace,monospace;text-align:center;padding:4px}}
.aar .tiles{{display:flex;gap:8px;flex-wrap:wrap}}
.aar .tile{{flex:1;min-width:110px;text-align:center;border:1px solid {LINE};border-radius:8px;padding:6px}}
.aar .tile .l{{font-size:11px;color:{MUTED}}}
.aar .tile .v{{font:700 18px ui-monospace,monospace}}
.aar .charts{{display:flex;gap:10px;flex-wrap:wrap;justify-content:space-between}}
.aar .chart{{flex:1;min-width:230px;max-width:340px;text-align:center}}
.aar .chart .t{{font:600 12px system-ui,sans-serif;color:{MUTED};margin-bottom:2px}}
.aar .chart svg{{width:100%;height:auto}}
.aar .chart .leg{{font-size:10.5px;color:{MUTED}}}
.aar .chart .none{{color:{MUTED};font-size:12px;padding:40px 6px}}
.aar details.pa{{border:1px solid {LINE};border-radius:10px;margin:8px 0;background:#fff}}
.aar details.pa[open]{{border-color:{TEAL};box-shadow:0 1px 6px rgba(74,139,150,.18)}}
.aar details.pa summary{{cursor:pointer;list-style:none;display:grid;align-items:center;gap:10px;padding:9px 12px;
  grid-template-columns:22px 52px minmax(140px,1.2fr) 110px minmax(150px,2fr) 110px}}
.aar details.pa summary::-webkit-details-marker{{display:none}}
.aar details.pa summary .car{{color:{TEAL};font-weight:700;transition:transform .15s}}
.aar details.pa[open] summary .car{{transform:rotate(90deg)}}
.aar summary .inn{{font:700 13px ui-monospace,monospace}}
.aar summary .res{{font-weight:700}}
.aar summary .ex{{color:{MUTED};font-weight:400;margin-left:6px}}
.aar summary .dr{{text-align:right;font:12px ui-monospace,monospace}}
.aar .pa-body{{padding:4px 12px 12px;border-top:1px solid {LINE}}}
.aar .wrap{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
.aar table.p{{border-collapse:collapse;width:100%;white-space:nowrap;margin:6px 0 10px}}
.aar table.p th{{font:600 11px system-ui,sans-serif;color:{MUTED};padding:5px 7px;border-bottom:1.5px solid {TEAL};
  text-align:right}}
.aar table.p td{{font:12.5px ui-monospace,"SF Mono",Menlo,monospace;padding:4px 7px;border-bottom:1px solid {LINE};
  text-align:right}}
.aar table.p td.l,.aar table.p th.l{{text-align:left;font-family:system-ui,sans-serif}}
.aar table.p tr.bip td{{background:#f6fafb}}
.aar .chip{{display:inline-block;padding:1px 7px;border-radius:9px;font:600 11px system-ui,sans-serif}}
.aar a.vid{{color:{TEAL};text-decoration:none;font-size:15px}}
.aar a.vid:hover{{color:{INK}}}
.aar .muted{{color:{MUTED}}}
@media (max-width:700px){{.aar details.pa summary{{grid-template-columns:18px 44px 1fr;row-gap:2px}}
  .aar summary .cnt,.aar summary .dr{{display:none}}}}
</style>
"""

HEADSHOT = ("https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/"
            "w_120,q_auto:best/v1/people/{id}/headshot/67/current")
LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"


def _f(v, fmt="{:.1f}", dash="–"):
    return dash if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA else fmt.format(v)


def _woba(v) -> str:
    return _f(v, "{:.3f}").replace("0.", ".") if v == v else "–"


def _woba_style(v) -> str:
    if v != v:
        return ""
    t = max(0.0, min(1.0, (v - 0.150) / 0.550))
    r, g, b = int(246 - 200 * t), int(246 - 100 * t), int(246 - 190 * t)
    return f"background:rgb({r},{g},{b});color:{'#fff' if t > .6 else INK}"


# ----------------------------------------------------------------------------- header + tiles
def header_html(pid: int, name: str, team: str, gm: dict) -> str:
    a, h = gm["away"], gm["home"]
    return (CSS + f'<div class="aar"><div class="card hdr"><img class="hs" src="{HEADSHOT.format(id=pid)}" alt="">'
            f'<div><div class="sub">{escape(team)} · Hitter AAR</div><div class="nm">{escape(name)}</div></div>'
            f'<div class="gm">{_f(gm["away_runs"], "{}", "")}<img src="{LOGO.format(id=a)}" alt="">'
            f'<div class="d">{escape(gm["abbr_away"])} @ {escape(gm["abbr_home"])}<br>{escape(gm["date"])}</div>'
            f'<img src="{LOGO.format(id=h)}" alt="">{_f(gm["home_runs"], "{}", "")}</div></div></div>')


def line_html(line: dict) -> str:
    cols = ["AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "K"]
    return (CSS + '<div class="aar"><div class="card"><table class="line"><tr>'
            + "".join(f"<th>{c}</th>" for c in cols) + "</tr><tr>"
            + "".join(f"<td>{line.get(c, '–')}</td>" for c in cols) + "</tr></table></div></div>")


def tiles_html(tiles: list[tuple[str, str]], metric_label: str) -> str:
    return (CSS + '<div class="aar"><div class="tiles">'
            + "".join(f'<div class="tile"><div class="l">{escape(lab)} {escape(metric_label)}</div>'
                      f'<div class="v">{escape(v)}</div></div>' for lab, v in tiles) + "</div></div>")


# ----------------------------------------------------------------------------- charts
def _shape(kind: str, x: float, y: float, r: float, fill: str, stroke: str = "#fff") -> str:
    if kind in ("Breaking", "LD"):
        return (f'<polygon points="{x},{y - r * 1.15} {x - r * 1.05},{y + r * .8} {x + r * 1.05},{y + r * .8}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width=".8"/>')
    if kind in ("Offspeed", "PU"):
        return (f'<rect x="{x - r * .85}" y="{y - r * .85}" width="{r * 1.7}" height="{r * 1.7}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width=".8"/>')
    if kind == "GB":
        return (f'<polygon points="{x},{y + r * 1.15} {x - r * 1.05},{y - r * .8} {x + r * 1.05},{y - r * .8}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width=".8"/>')
    if kind == "FB":
        return (f'<polygon points="{x},{y - r * 1.2} {x + r * 1.1},{y} {x},{y + r * 1.2} {x - r * 1.1},{y}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width=".8"/>')
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width=".8"/>'


def _label(n, x, y, size=7.5) -> str:
    return (f'<text x="{x}" y="{y + size * .36}" text-anchor="middle" font-size="{size}" font-weight="700" '
            f'fill="#fff" font-family="system-ui,sans-serif">{n}</text>')


def _result_color(play: str) -> str:
    return RES_COLOR.get(play, OUT_COLOR)


def zone_svg(rows: pd.DataFrame) -> str:
    """Catcher's view. Solid box = strike zone; dashed = heart; dotted = edge of the shadow (attack zones)."""
    W, H = 260, 290
    xs = lambda x: W / 2 + x * 62                              # feet -> px
    top = rows["sz_top"].median() if rows["sz_top"].notna().any() else 3.4
    bot = rows["sz_bot"].median() if rows["sz_bot"].notna().any() else 1.6
    zs = lambda z: 250 - (z - 0.2) * 52
    hw, mid, hh = 17 / 24, (top + bot) / 2, (top - bot) / 2
    box = lambda k, dash, w: (f'<rect x="{xs(-hw * k)}" y="{zs(mid + hh * k)}" width="{xs(hw * k) - xs(-hw * k)}" '
                              f'height="{zs(mid - hh * k) - zs(mid + hh * k)}" fill="none" stroke="{TEAL if k == 1 else "#9fb9be"}" '
                              f'stroke-width="{w}" stroke-dasharray="{dash}"/>')
    g = [box(1.33, "2 3", 1), box(1, "", 1.8), box(0.67, "5 4", 1)]
    g.append(f'<polygon points="{xs(-hw)},{262} {xs(hw)},{262} {xs(hw)},{268} {xs(0)},{276} {xs(-hw)},{268}" '
             f'fill="#f1f4f5" stroke="#9fb9be"/>')
    for r in rows.itertuples():
        if r.px != r.px or r.pz != r.pz:
            continue
        x, y = max(8, min(W - 8, xs(r.px))), max(8, min(H - 20, zs(r.pz)))
        g.append(_shape(r.pgroup, x, y, 8.5, SWING if r.swing else TAKE) + _label(r.n, x, y + (1.5 if r.pgroup == "Breaking" else 0)))
    leg = (f'<span style="color:{SWING}">●</span> Swing &nbsp;<span style="color:{TAKE}">●</span> Take &nbsp;'
           f'● FB &nbsp;▲ BB &nbsp;■ OS')
    return _chart("Pitch location (catcher's view)", f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>', leg)


def spray_svg(rows: pd.DataFrame) -> str:
    W, H = 280, 270
    hx, hy, s = W / 2, 250, 0.52                          # home plate in px; px per foot
    pt = lambda ft_x, ft_y: (hx + ft_x * s, hy - ft_y * s)
    g = []
    l1, r1, c1 = pt(-233, 233), pt(233, 233), pt(0, 405)
    g.append(f'<path d="M{hx},{hy} L{l1[0]},{l1[1]} Q{c1[0]},{c1[1] - 95} {r1[0]},{r1[1]} Z" fill="#eef5ef" '
             f'stroke="#9fb9be"/>')
    b1, b2, b3 = pt(63.6, 63.6), pt(0, 127.3), pt(-63.6, 63.6)
    g.append(f'<polygon points="{hx},{hy} {b1[0]},{b1[1]} {b2[0]},{b2[1]} {b3[0]},{b3[1]}" fill="#f4ede2" '
             f'stroke="#c9b48f"/>')
    for r in rows.itertuples():
        if not r.bip or r.hc_x != r.hc_x or r.hc_y != r.hc_y:
            continue
        fx, fy = (r.hc_x - 125.42) * 2.5, (198.27 - r.hc_y) * 2.5
        x, y = pt(fx, fy)
        x, y = max(8, min(W - 8, x)), max(8, min(H - 8, y))
        g.append(_shape(r.bb or "GB", x, y, 8.5, _result_color(r.play)) + _label(r.n, x, y + (1.5 if r.bb == "LD" else -1.2 if r.bb == "GB" else 0)))
    leg = ("▼ GB ▲ LD ◆ FB ■ PU &nbsp;" + " ".join(f'<span style="color:{c}">●</span> {k.replace("Home run", "HR")}'
                                                 for k, c in RES_COLOR.items()) + f' <span style="color:{OUT_COLOR}">●</span> Out')
    return _chart("Spray", f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>', leg)


def contact_svg(rows: pd.DataFrame, orient: tuple[int, int] | None) -> str:
    """Point of contact from Statcast bat tracking, relative to the hitter's body (measured, not estimated)."""
    have = rows[rows["swing"] & rows["cx"].notna() & rows["cy"].notna()]
    if have.empty or orient is None:
        why = ("No measured contact here." if have.empty else
               "Measured, but not enough of his season yet to confirm the axes.")
        return _chart("Point of contact", f'<div class="none">{why}</div>', "")
    sy, sx = orient
    W, H = 260, 290
    xs = lambda v: 40 + v * 5.0                           # inches from body -> px
    ys = lambda v: 200 - v * 5.2                          # inches out front -> px
    g = [f'<line x1="{xs(0)}" y1="14" x2="{xs(0)}" y2="276" stroke="#cfd8dc"/>',
         f'<line x1="14" y1="{ys(0)}" x2="{W - 10}" y2="{ys(0)}" stroke="#cfd8dc"/>',
         f'<text x="{xs(0) + 4}" y="22" font-size="9" fill="{MUTED}">out front ↑</text>',
         f'<text x="{W - 12}" y="{ys(0) - 4}" font-size="9" fill="{MUTED}" text-anchor="end">away from body →</text>',
         f'<circle cx="{xs(0)}" cy="{ys(0)}" r="5" fill="#cfd8dc"/>',
         f'<text x="{xs(0) - 7}" y="{ys(0) + 14}" font-size="9" fill="{MUTED}" text-anchor="end">him</text>']
    for v in (-10, 10, 20, 30):
        g.append(f'<text x="{xs(0) - 4}" y="{ys(v) + 3}" font-size="8" fill="#9aa5ae" text-anchor="end">{v}</text>')
    for v in (10, 20, 30, 40):
        g.append(f'<text x="{xs(v)}" y="{ys(0) + 12}" font-size="8" fill="#9aa5ae" text-anchor="middle">{v}</text>')
    for r in have.itertuples():
        side = 1 if r.stand == "R" else -1
        x = max(8, min(W - 8, xs(sx * side * r.cx)))
        y = max(8, min(H - 8, ys(sy * r.cy)))
        color = _result_color(r.play) if r.bip else (SWING if "Foul" in r.result else "#b8322a")
        g.append(_shape(r.bb if r.bip else "", x, y, 8, color) + _label(r.n, x, y))
    leg = ("inches from his body · ● whiff (red) / foul (orange) / in play by result")
    return _chart("Point of contact", f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>', leg)


def _chart(title: str, body: str, leg: str) -> str:
    return f'<div class="chart"><div class="t">{title}</div>{body}<div class="leg">{leg}</div></div>'


def charts_html(rows: pd.DataFrame, orient, wrap: bool = True) -> str:
    inner = f'<div class="charts">{zone_svg(rows)}{spray_svg(rows)}{contact_svg(rows, orient)}</div>'
    return (CSS + f'<div class="aar"><div class="card">{inner}</div></div>') if wrap else inner


# ----------------------------------------------------------------------------- pitch table + PAs
def _cols(p: pd.DataFrame) -> list[tuple[str, str, str]]:
    cols = [("n", "#", "r"), ("video", "Video", "c"), ("inn", "Inn", "r"), ("count", "Count", "r"),
            ("pitcher", "Pitcher", "l"), ("result", "Pitch Result", "l"), ("play", "Play Result", "l"),
            ("ptype", "Type", "r"), ("velo", "Velo", "r"), ("hb", "HB", "r"), ("ivb", "IVB", "r"),
            ("bb", "BB Type", "r"), ("ev", "EV", "r"), ("bat_speed", "Bat Speed", "r"), ("la", "LA", "r"),
            ("dist", "Dist", "r"), ("xwoba", "xwOBA", "r"), ("zone", "Zone", "r"), ("dec", "Decision", "r"),
            ("dec_runs", "Dec Runs", "r")]
    if "timing" in p and p["timing"].replace("", pd.NA).notna().any():
        cols.append(("timing", "Timing", "r"))
    if "miss" in p and p["miss"].notna().any():
        cols.append(("miss", "Miss (in)", "r"))
    return cols


def _cell(key: str, r) -> tuple[str, str]:
    v = getattr(r, key, None)
    if key == "video":
        return (f'<a class="vid" href="{SAVANT_VIDEO.format(v)}" target="_blank" rel="noopener" '
                f'title="Play this pitch">&#127909;</a>' if v else "–"), ""
    if key == "inn":
        return f"{r.half}{r.inning}", ""
    if key == "ptype":
        return f'<b style="color:{PT_COLOR.get(v, INK)}">{escape(str(v))}</b>', ""
    if key in ("velo", "hb", "ivb", "ev", "bat_speed", "la", "miss"):
        return _f(v), ""
    if key == "dist":
        return _f(v, "{:.0f}"), ""
    if key == "xwoba":
        return _woba(v), _woba_style(v)
    if key == "dec":
        if not v:
            return "–", ""
        bg, fg = DEC_COLOR[v]
        return f'<span class="chip" style="background:{bg};color:{fg}">{v}</span>', ""
    if key == "dec_runs":
        return _f(v, "{:+.2f}"), ""
    return escape(str(v)) if v not in (None, "") and v == v else "–", ""


def pitch_table(rows: pd.DataFrame) -> str:
    cols = _cols(rows)
    head = "".join(f'<th class="{"l" if a == "l" else ""}">{lab}</th>' for _, lab, a in cols)
    body = ""
    for r in rows.itertuples():
        tds = ""
        for key, _, a in cols:
            html, style = _cell(key, r)
            tds += f'<td class="{"l" if a == "l" else ""}" style="{style}">{html}</td>'
        body += f'<tr class="{"bip" if r.bip else ""}">{tds}</tr>'
    return f'<div class="wrap"><table class="p"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def pa_html(pas: list[dict], orient, open_all: bool = False) -> str:
    """Every plate appearance as its own collapsible section -- open as many as you like. Opening one shows its
    pitches plus its own zone / spray / contact charts."""
    out = []
    for pa in pas:
        ex = f'<span class="ex">{escape(pa["extra"])}</span>' if pa["extra"] else ""
        dr = _f(pa["dec_runs"], "{:+.2f} dec runs", "")
        hand = f' ({pa["p_throws"]}HP)' if pa["p_throws"] else ""
        out.append(
            f'<details class="pa"{" open" if open_all else ""}><summary><span class="car">›</span>'
            f'<span class="inn">{escape(pa["inning"])}</span>'
            f'<span>PA {pa["i"]} · vs {escape(pa["pitcher"] or "—")}<span class="muted">{hand}</span></span>'
            f'<span class="cnt muted">{pa["n_pitches"]} pitch{"es" if pa["n_pitches"] != 1 else ""} · {pa["final_count"]}</span>'
            f'<span class="res">{escape(pa["result"])}{ex}</span><span class="dr">{dr}</span></summary>'
            f'<div class="pa-body">{pitch_table(pa["rows"])}{charts_html(pa["rows"], orient, wrap=False)}</div>'
            f'</details>')
    return CSS + '<div class="aar">' + "".join(out) + "</div>"
