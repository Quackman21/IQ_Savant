"""AAR page: one interactive panel (a Streamlit component, so it can run its own clicks):

    [ box line + Last 30 PA / Season / MLB season tiles (metric picker) ]  [ zone ][ spray ][ point of contact ]
    [ ONE pitch table, grouped by plate appearance -- every PA starts collapsed; open as many as you want     ]

Click any pitch -- a dot on any chart or a row in the table -- and it lights up everywhere at once; click more to
select several; "Clear" resets. Opening PAs filters the charts to those PAs (none open = the whole game). Selecting a
pitch from a chart opens its PA so its row is visible."""
from __future__ import annotations

import json
import math
from html import escape

import numpy as np
import pandas as pd

from core.aar import PLATE_HALF, SAVANT_VIDEO

TEAL, INK, MUTED, LINE = "#4a8b96", "#16202b", "#6a7681", "#e6eaed"
SWING, TAKE = "#d9822b", "#5b6bc8"
RES_COLOR = {"Single": "#2bb3c0", "Double": "#e0b300", "Triple": "#8e5bd0", "Home run": "#3a9a4f"}
OUT_COLOR = "#9aa5ae"
PT_COLOR = {"FF": "#c0392b", "FA": "#c0392b", "SI": "#e67e22", "FC": "#8e5b3a", "SL": "#c9a400", "ST": "#b8860b",
            "SV": "#b8860b", "SW": "#b8860b", "CU": "#2e6fd1", "KC": "#2e6fd1", "CS": "#2e6fd1", "CH": "#2e9b57",
            "FS": "#16a2a8", "FO": "#16a2a8", "KN": "#666", "EP": "#666"}
DEC_COLOR = {"Great": ("#1f7a3a", "#fff"), "Good": ("#cdebd5", INK), "Neutral": ("#eef1f3", MUTED),
             "Bad": ("#f6d3d0", INK), "Poor": ("#b8322a", "#fff")}
HEADSHOT = ("https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/"
            "w_120,q_auto:best/v1/people/{id}/headshot/67/current")
LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"
CAMERA = ('<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.8" '
          'stroke-linecap="round" stroke-linejoin="round"><rect x="2.5" y="6" width="13" height="12" rx="2"/>'
          '<path d="M15.5 10.5l6-3.5v10l-6-3.5z"/></svg>')

HEAD_CSS = f"""<style>
.aarh{{font:13px system-ui,sans-serif;color:{INK};display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  border:1px solid {LINE};border-radius:10px;padding:10px 14px;background:#fff}}
.aarh img.hs{{width:56px;height:56px;border-radius:50%;object-fit:cover;background:#eef1f3}}
.aarh .nm{{font:700 22px system-ui,sans-serif}} .aarh .sub{{color:{MUTED};font-size:12px}}
.aarh .gm{{margin-left:auto;display:flex;align-items:center;gap:10px;font:700 20px system-ui,sans-serif}}
.aarh .gm img{{width:30px;height:30px;object-fit:contain}}
.aarh .gm .d{{font:12px system-ui,sans-serif;color:{MUTED};text-align:center;line-height:1.3}}
</style>"""


def _f(v, fmt="{:.1f}", dash="–"):
    try:
        if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
            return dash
    except TypeError:
        pass
    return fmt.format(v)


def _woba(v) -> str:
    return _f(v, "{:.3f}").replace("0.", ".") if v == v else "–"


def _woba_style(v) -> str:
    if v != v:
        return ""
    t = max(0.0, min(1.0, (v - 0.150) / 0.550))
    r, g, b = int(246 - 200 * t), int(246 - 100 * t), int(246 - 190 * t)
    return f"background:rgb({r},{g},{b});color:{'#fff' if t > .6 else INK}"


def header_html(pid: int, name: str, team: str, gm: dict) -> str:
    return (HEAD_CSS + f'<div class="aarh"><img class="hs" src="{HEADSHOT.format(id=pid)}" alt="">'
            f'<div><div class="sub">{escape(team)} · Hitter AAR</div><div class="nm">{escape(name)}</div></div>'
            f'<div class="gm">{_f(gm["away_runs"], "{}", "")}<img src="{LOGO.format(id=gm["away"])}" alt="">'
            f'<div class="d">{escape(gm["abbr_away"])} @ {escape(gm["abbr_home"])}<br>{escape(gm["date"])}</div>'
            f'<img src="{LOGO.format(id=gm["home"])}" alt="">{_f(gm["home_runs"], "{}", "")}</div></div>')


# ----------------------------------------------------------------------------- chart marks
def _shape(kind: str, x: float, y: float, r: float, fill: str) -> str:
    st = f'fill="{fill}" stroke="#fff" stroke-width=".9"'
    if kind in ("Breaking", "LD"):
        return f'<polygon class="m" points="{x:.1f},{y - r * 1.15:.1f} {x - r * 1.05:.1f},{y + r * .8:.1f} {x + r * 1.05:.1f},{y + r * .8:.1f}" {st}/>'
    if kind in ("Offspeed", "PU"):
        return f'<rect class="m" x="{x - r * .85:.1f}" y="{y - r * .85:.1f}" width="{r * 1.7:.1f}" height="{r * 1.7:.1f}" {st}/>'
    if kind == "GB":
        return f'<polygon class="m" points="{x:.1f},{y + r * 1.15:.1f} {x - r * 1.05:.1f},{y - r * .8:.1f} {x + r * 1.05:.1f},{y - r * .8:.1f}" {st}/>'
    if kind == "FB":
        return f'<polygon class="m" points="{x:.1f},{y - r * 1.2:.1f} {x + r * 1.1:.1f},{y:.1f} {x:.1f},{y + r * 1.2:.1f} {x - r * 1.1:.1f},{y:.1f}" {st}/>'
    return f'<circle class="m" cx="{x:.1f}" cy="{y:.1f}" r="{r}" {st}/>'


def _mark(r, kind: str, x: float, y: float, fill: str, rad: float = 8.5) -> str:
    dy = 1.6 if kind in ("Breaking", "LD") else -1.3 if kind == "GB" else 0
    return (f'<g class="pt" data-n="{r.n}" data-pa="{r.pa}" data-pt="{escape(str(r.ptype))}" '
            f'data-hand="{escape(str(r.p_throws))}">{_shape(kind, x, y, rad, fill)}'
            f'<text x="{x:.1f}" y="{y + dy + 2.7:.1f}" text-anchor="middle">{r.n}</text></g>')


def _res_color(play: str) -> str:
    return RES_COLOR.get(play, OUT_COLOR)


def zone_svg(p: pd.DataFrame) -> str:
    W, H = 270, 300
    xs = lambda x: W / 2 + x * 62
    top = p["sz_top"].median() if p["sz_top"].notna().any() else 3.4
    bot = p["sz_bot"].median() if p["sz_bot"].notna().any() else 1.6
    zs = lambda z: 258 - (z - 0.25) * 52
    hw, mid, hh = 17 / 24, (top + bot) / 2, (top - bot) / 2

    def box(k, dash, w, col):
        return (f'<rect x="{xs(-hw * k):.1f}" y="{zs(mid + hh * k):.1f}" width="{xs(hw * k) - xs(-hw * k):.1f}" '
                f'height="{zs(mid - hh * k) - zs(mid + hh * k):.1f}" fill="none" stroke="{col}" stroke-width="{w}" '
                f'stroke-dasharray="{dash}"/>')
    g = [box(1.33, "2 3", 1, "#9fb9be"), box(1, "", 1.8, TEAL), box(0.67, "5 4", 1, "#9fb9be"),
         f'<text x="{xs(-hw * 1.33) - 4:.1f}" y="{zs(top) + 3:.1f}" text-anchor="end" class="ax">{top:.2f}\'</text>',
         f'<text x="{xs(-hw * 1.33) - 4:.1f}" y="{zs(bot) + 3:.1f}" text-anchor="end" class="ax">{bot:.2f}\'</text>',
         f'<polygon points="{xs(-hw):.1f},270 {xs(hw):.1f},270 {xs(hw):.1f},276 {xs(0):.1f},284 {xs(-hw):.1f},276" '
         f'fill="#f1f4f5" stroke="#9fb9be"/>']
    for r in p.itertuples():
        if r.px == r.px and r.pz == r.pz:
            x, y = max(9, min(W - 9, xs(r.px))), max(9, min(H - 22, zs(r.pz)))
            g.append(_mark(r, r.pgroup, x, y, SWING if r.swing else TAKE))
    leg = (f'<i style="color:{SWING}">●</i> Swing <i style="color:{TAKE}">●</i> Take &nbsp;● FB ▲ BB ■ OS'
           f' &nbsp;<span class="faint">solid = zone · dashed = heart · dotted = shadow</span>')
    return _chart("Pitch location", "catcher's view", f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>', leg)


DEFAULT_DIMS = {"leftLine": 330, "leftCenter": 375, "center": 400, "rightCenter": 375, "rightLine": 330}
LINE_ANGLES = {"leftLine": -45, "leftCenter": -22.5, "center": 0, "rightCenter": 22.5, "rightLine": 45}


def _poly(pts) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def spray_svg(p: pd.DataFrame, dims: dict | None = None, park: str = "", outline: dict | None = None) -> str:
    """Spray chart on the actual park: the traced outline (fence, dirt, grass, lines) when it matches this venue's
    published distances; otherwise straight lines between the published distances (labeled approximate)."""
    W, H = 290, 300
    dims = dims or DEFAULT_DIMS
    segs = (outline or {}).get("segments") or {}
    far = max([max(dims.values())] + [float(np.max(np.hypot(*v))) for v in segs.values()] if segs else
              [max(dims.values())])
    sc = 262 / (far + 12)
    hx, hy = W / 2, 290
    pt = lambda fx, fy: (hx + fx * sc, hy - fy * sc)
    g = []
    if segs:
        fx, fy = segs["outfield_outer"]
        fence = [pt(a, b) for a, b in zip(fx, fy)]
        g.append(f'<polygon points="{hx:.1f},{hy:.1f} {_poly(fence)}" fill="#eef5ef" stroke="#5f8f96" '
                 f'stroke-width="1.6" stroke-linejoin="round"/>')
        for seg, fill, stroke in (("infield_outer", "#f4ede2", "#c9b48f"), ("infield_inner", "#e8f1e8", "#c9b48f")):
            if seg in segs:
                g.append(f'<polygon points="{_poly(pt(a, b) for a, b in zip(*segs[seg]))}" fill="{fill}" '
                         f'stroke="{stroke}" stroke-width=".8"/>')
        if "foul_lines" in segs:
            g.append(f'<polyline points="{_poly(pt(a, b) for a, b in zip(*segs["foul_lines"]))}" fill="none" '
                     f'stroke="#b9c7cb" stroke-width=".8"/>')
        note = ""
    else:
        pts = [pt(dims[k] * math.sin(math.radians(a)), dims[k] * math.cos(math.radians(a)))
               for k, a in LINE_ANGLES.items() if k in dims]
        g.append(f'<polygon points="{hx:.1f},{hy:.1f} {_poly(pts)}" fill="#eef5ef" stroke="#7fa3a9" '
                 f'stroke-width="1.3" stroke-dasharray="5 3"/>')
        b1, b2, b3 = pt(63.6, 63.6), pt(0, 127.3), pt(-63.6, 63.6)
        g.append(f'<polygon points="{hx},{hy} {_poly([b1, b2, b3])}" fill="#f4ede2" stroke="#c9b48f"/>')
        note = " (approximate fence)"
    for k, anchor, dx in (("leftLine", "start", 5), ("center", "middle", 0), ("rightLine", "end", -5)):
        if k in dims:
            a = LINE_ANGLES[k]
            x, y = pt(dims[k] * math.sin(math.radians(a)), dims[k] * math.cos(math.radians(a)))
            g.append(f'<text x="{x + dx:.1f}" y="{y - 5:.1f}" class="ax" text-anchor="{anchor}">{dims[k]:.0f}</text>')
    for r in p.itertuples():
        if r.bip and r.hc_x == r.hc_x and r.hc_y == r.hc_y:
            x, y = pt((r.hc_x - 125.42) * 2.5, (198.27 - r.hc_y) * 2.5)
            g.append(_mark(r, r.bb or "GB", max(9, min(W - 9, x)), max(9, min(H - 9, y)), _res_color(r.play)))
    leg = ("▼ GB ▲ LD ◆ FB ■ PU &nbsp;" + " ".join(f'<i style="color:{c}">●</i> {k.replace("Home run", "HR")}'
                                                 for k, c in RES_COLOR.items()) + f' <i style="color:{OUT_COLOR}">●</i> Out')
    return _chart("Spray", escape(park + note), f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>', leg,
                  tip=(outline or {}).get("note", ""))


def contact_svg(p: pd.DataFrame, note: str, tip: str = "") -> str:
    """Point of contact on the plate (catcher's view): front-center of the plate = 0, 0; up = out in front."""
    have = p[p["swing"] & p["pf"].notna() & p["pxc"].notna()]
    if have.empty:
        return _chart("Point of contact", "", f'<div class="none">{escape(note)}</div>', "")
    W, H = 270, 300
    xs = lambda v: W / 2 + v * 5.2
    ys = lambda v: 165 - v * 5.2
    g = []
    for v in range(-20, 35, 5):
        g.append(f'<line x1="22" y1="{ys(v):.1f}" x2="{W - 6}" y2="{ys(v):.1f}" stroke="#eef1f3" '
                 f'stroke-dasharray="{"" if v == 0 else "3 3"}"/><text x="18" y="{ys(v) + 3:.1f}" class="ax" '
                 f'text-anchor="end">{v}</text>')
    for v in (-10, 0, 10):
        g.append(f'<text x="{xs(v):.1f}" y="{H - 4}" class="ax" text-anchor="middle">{v}</text>')
    h = PLATE_HALF
    g.append(f'<polygon points="{xs(-h):.1f},{ys(0):.1f} {xs(h):.1f},{ys(0):.1f} {xs(h):.1f},{ys(-h):.1f} '
             f'{xs(0):.1f},{ys(-17):.1f} {xs(-h):.1f},{ys(-h):.1f}" fill="#f6f8f9" stroke="{TEAL}" stroke-width="1.6"/>')
    g.append(f'<text x="{W - 8}" y="{ys(30) + 9:.1f}" class="ax" text-anchor="end">out front ↑</text>')
    for r in have.itertuples():
        color = _res_color(r.play) if r.bip else (SWING if "Foul" in r.result else "#b8322a")
        x, y = max(9, min(W - 9, xs(r.pxc))), max(9, min(H - 16, ys(r.pf)))
        g.append(_mark(r, r.bb if r.bip else "", x, y, color))
    leg = (f'<i style="color:#b8322a">●</i> Whiff <i style="color:{SWING}">●</i> Foul &nbsp;in play: shape = BB type, '
           f'color = result &nbsp;<span class="faint">inches · catcher\'s view</span>')
    return _chart("Point of contact", "on the plate", f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>', leg,
                  tip=tip)


def _chart(title: str, sub: str, body: str, leg: str, tip: str = "") -> str:
    return (f'<div class="chart" title="{escape(tip)}"><div class="t">{title}'
            f'{f" <span class=faint>· {sub}</span>" if sub else ""}</div>{body}<div class="leg">{leg}</div></div>')


# ----------------------------------------------------------------------------- the one table
COLS = [("n", "#", "r"), ("video", "Video", "c"), ("inn", "Inn", "r"), ("count", "Count", "r"),
        ("pitcher", "Pitcher", "l"), ("result", "Pitch Result", "l"), ("play", "Play Result", "l"),
        ("ptype", "Type", "c"), ("velo", "Velo", "r"), ("hb", "HB", "r"), ("ivb", "IVB", "r"),
        ("bb", "BB Type", "c"), ("ev", "EV", "r"), ("bat_speed", "Bat Speed", "r"), ("la", "LA", "r"),
        ("dist", "Dist", "r"), ("xwoba", "xwOBA", "r"), ("zone", "Zone", "c"), ("dec", "Decision", "c"),
        ("dec_runs", "Dec Runs", "r")]


def _cols(p: pd.DataFrame) -> list:
    cols = list(COLS)
    if "timing" in p and p["timing"].replace("", pd.NA).notna().any():
        cols.append(("timing", "Timing", "c"))
    if "miss" in p and p["miss"].notna().any():
        cols.append(("miss", "Miss (in)", "r"))
    return cols


def _cell(key: str, r) -> tuple[str, str]:
    v = getattr(r, key, None)
    if key == "video":
        return (f'<a class="vid" href="{SAVANT_VIDEO.format(v)}" target="_blank" rel="noopener" '
                f'title="Play this pitch">{CAMERA}</a>' if v else "–"), ""
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
        return (_f(v, "{:+.2f}").replace("-0.00", "0.00").replace("+0.00", "0.00")), ""
    return (escape(str(v)) if v not in (None, "") and v == v else "–"), ""


def table_html(p: pd.DataFrame, pas: list[dict]) -> str:
    cols = _cols(p)
    head = "".join(f'<th class="{a}">{lab}</th>' for _, lab, a in cols)
    body = ""
    for pa in pas:
        dr = _f(pa["dec_runs"], "{:+.2f} dec runs", "")
        hand = f' ({pa["p_throws"]}HP)' if pa["p_throws"] else ""
        ex = f' <span class="faint">{escape(pa["extra"])}</span>' if pa["extra"] else ""
        body += (f'<tr class="pa-h" data-pa="{pa["i"]}"><td colspan="{len(cols)}"><div class="pah">'
                 f'<span class="car">›</span><b class="inn">{escape(pa["inning"])}</b>'
                 f'<span>PA {pa["i"]} · vs {escape(pa["pitcher"] or "—")}<span class="faint">{hand}</span></span>'
                 f'<span class="faint">{pa["n_pitches"]} pitch{"es" if pa["n_pitches"] != 1 else ""}</span>'
                 f'<b>{escape(pa["result"])}</b>{ex}<span class="dr">{dr}</span></div></td></tr>')
        for r in pa["rows"].itertuples():
            tds = "".join(f'<td class="{a}" style="{st}">{h}</td>' for (k, _, a) in cols for h, st in [_cell(k, r)])
            body += (f'<tr class="row{" bip" if r.bip else ""}" data-n="{r.n}" data-pa="{pa["i"]}" '
                     f'data-pt="{escape(str(r.ptype))}" data-hand="{escape(str(r.p_throws))}">{tds}</tr>')
    return f'<table class="p"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def line_html(line: dict) -> str:
    cols = ["AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "K"]
    return ('<table class="line"><tr>' + "".join(f"<th>{c}</th>" for c in cols) + "</tr><tr>"
            + "".join(f"<td>{line.get(c, '–')}</td>" for c in cols) + "</tr></table>")


# ----------------------------------------------------------------------------- the component
CSS = f"""
*{{box-sizing:border-box}} body{{margin:0;font:13px system-ui,-apple-system,sans-serif;color:{INK};background:transparent}}
.top{{display:grid;grid-template-columns:minmax(300px,1fr) 2.4fr;gap:10px;align-items:stretch}}
.card{{background:#fff;border:1px solid {LINE};border-radius:10px;padding:10px 12px}}
.left{{display:flex;flex-direction:column;gap:10px}}
table.line{{border-collapse:collapse;width:100%}} table.line th{{font:600 11px system-ui;color:{MUTED};padding:3px;
  border-bottom:1.5px solid {TEAL}}} table.line td{{font:600 17px ui-monospace,monospace;text-align:center;padding:5px 2px}}
.tiles .hd{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
.tiles .hd .lab{{font:600 12px system-ui;color:{MUTED}}}
.tiles select{{font:13px system-ui;padding:4px 8px;border:1px solid {LINE};border-radius:7px;background:#fff}}
.tiles .row{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}}
.tile{{text-align:center;border:1px solid {LINE};border-radius:8px;padding:8px 4px}}
.tile .l{{font-size:11px;color:{MUTED}}} .tile .v{{font:700 19px ui-monospace,monospace;margin-top:2px}}
.charts{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
.chart{{text-align:center;min-width:0}} .chart .t{{font:600 12px system-ui;color:{INK};margin-bottom:2px}}
.chart svg{{width:100%;height:auto;display:block}} .chart .leg{{font-size:10.5px;color:{MUTED};line-height:1.5}}
.chart .leg i{{font-style:normal}} .chart .none{{color:{MUTED};font-size:12px;padding:70px 10px}}
.faint{{color:{MUTED};font-weight:400}}
svg text{{font:700 7.5px system-ui;fill:#fff;pointer-events:none}} svg text.ax{{font:500 8.5px system-ui;fill:#8a96a0}}
g.pt{{cursor:pointer;transition:opacity .15s}} g.pt.hide{{display:none}}
.has-sel g.pt:not(.sel){{opacity:.22}}
g.pt.sel .m{{stroke:#111;stroke-width:2.2;filter:drop-shadow(0 0 4px rgba(255,196,0,.95)) drop-shadow(0 0 2px #ffc400)}}
.bar{{display:flex;align-items:center;gap:8px;margin:12px 0 6px;flex-wrap:wrap}}
.bar .ttl{{font:700 15px system-ui;margin-right:auto}}
.bar button{{font:600 12px system-ui;border:1px solid {LINE};background:#fff;border-radius:7px;padding:5px 10px;cursor:pointer}}
.bar button:hover{{border-color:{TEAL};color:{TEAL}}} .bar .sel-n{{font:600 12px system-ui;color:{TEAL}}}
.filters{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:0 0 8px}}
.filters .lab{{font:600 12px system-ui;color:{MUTED};margin-right:2px}} .filters .sep{{width:14px}}
.chip-f{{font:700 12px ui-monospace,monospace;border:1.5px solid var(--c);color:var(--c);background:#fff;
  border-radius:12px;padding:3px 9px;cursor:pointer;opacity:.45}}
.chip-f.on{{background:var(--c);color:#fff;opacity:1}}
.seg{{display:inline-flex;border:1px solid {LINE};border-radius:8px;overflow:hidden}}
.seg button{{font:600 12px system-ui;border:none;background:#fff;padding:4px 10px;cursor:pointer;color:{INK}}}
.seg button.on{{background:{TEAL};color:#fff}} #shown{{margin-left:6px}}
g.pt.filt,tr.row.filt,tr.pa-h.filt{{display:none!important}}
.wrap{{overflow-x:auto;background:#fff;border:1px solid {LINE};border-radius:10px}}
table.p{{border-collapse:collapse;width:100%;white-space:nowrap}}
table.p th{{font:600 11px system-ui;color:{MUTED};padding:7px;border-bottom:1.5px solid {TEAL};position:sticky;top:0;background:#fff}}
table.p td{{font:12.5px ui-monospace,"SF Mono",Menlo,monospace;padding:5px 7px;border-bottom:1px solid {LINE}}}
.r{{text-align:right}} .l{{text-align:left;font-family:system-ui!important}} .c{{text-align:center}}
tr.pa-h td{{background:#f7f9fa;cursor:pointer;padding:8px 10px;border-bottom:1px solid {LINE}}}
tr.pa-h:hover td{{background:#eef4f5}}
.pah{{display:flex;gap:14px;align-items:baseline;font:13px system-ui}}
.pah .car{{color:{TEAL};font-weight:800;display:inline-block;transition:transform .15s;width:10px}}
tr.pa-h.open .car{{transform:rotate(90deg)}} tr.pa-h.open td{{background:#eaf3f4}}
.pah .inn{{font-family:ui-monospace,monospace;min-width:28px}} .pah .dr{{margin-left:auto;font:12px ui-monospace,monospace}}
tr.row{{display:none;cursor:pointer}} tr.row.show{{display:table-row}} tr.row.bip td{{background:#fafcfc}}
tr.row:hover td{{background:#f2f7f8}}
tr.row.sel td{{background:#fff4c7!important;box-shadow:inset 0 1px 0 #ffd34d,inset 0 -1px 0 #ffd34d}}
.has-sel tr.row:not(.sel) td{{opacity:.55}}
.chip{{display:inline-block;padding:1px 8px;border-radius:9px;font:600 11px system-ui}}
a.vid{{color:{TEAL};display:inline-flex}} a.vid:hover{{color:{INK}}}
@media (max-width:900px){{.top{{grid-template-columns:1fr}}.charts{{grid-template-columns:1fr 1fr}}}}
"""

JS = """
const root = document.getElementById('aar');
const open = new Set(), sel = new Set(), types = new Set();
const allPA = [...document.querySelectorAll('tr.pa-h')].map(h => h.dataset.pa);
allPA.forEach(p => open.add(p));                       // every PA open to start
document.querySelectorAll('.chip-f').forEach(c => types.add(c.dataset.f));
let hand = '';
const pass = el => types.has(el.dataset.pt) && (!hand || el.dataset.hand === hand);
function apply() {
  document.querySelectorAll('tr.pa-h').forEach(h => h.classList.toggle('open', open.has(h.dataset.pa)));
  const live = new Set(); let n = 0, tot = 0;
  document.querySelectorAll('tr.row').forEach(r => {
    const ok = pass(r); tot++; if (ok) { n++; live.add(r.dataset.pa); }
    r.classList.toggle('filt', !ok);
    r.classList.toggle('show', open.has(r.dataset.pa));
    r.classList.toggle('sel', sel.has(r.dataset.n));
  });
  document.querySelectorAll('tr.pa-h').forEach(h => h.classList.toggle('filt', !live.has(h.dataset.pa)));
  document.querySelectorAll('g.pt').forEach(g => {
    g.classList.toggle('filt', !pass(g));
    g.classList.toggle('hide', open.size > 0 && !open.has(g.dataset.pa));
    g.classList.toggle('sel', sel.has(g.dataset.n));
  });
  document.getElementById('shown').textContent = n < tot ? `${n} of ${tot} pitches` : '';
  root.classList.toggle('has-sel', sel.size > 0);
  document.getElementById('seln').textContent = sel.size ? sel.size + ' selected' : '';
  document.getElementById('scope').textContent = open.size ? 'charts: open PAs' : 'charts: whole game';
  resize();
}
function toggle(set, k) { set.has(k) ? set.delete(k) : set.add(k); }
document.querySelectorAll('tr.pa-h').forEach(h => h.addEventListener('click', () => { toggle(open, h.dataset.pa); apply(); }));
document.querySelectorAll('tr.row').forEach(r => r.addEventListener('click', e => {
  if (e.target.closest('a')) return; toggle(sel, r.dataset.n); apply(); }));
document.querySelectorAll('g.pt').forEach(g => g.addEventListener('click', () => {
  const was = sel.has(g.dataset.n); toggle(sel, g.dataset.n); if (!was) open.add(g.dataset.pa); apply(); }));
document.getElementById('expand').onclick = () => { allPA.forEach(p => open.add(p)); apply(); };
document.getElementById('collapse').onclick = () => { open.clear(); apply(); };
document.getElementById('clear').onclick = () => { sel.clear(); apply(); };
document.querySelectorAll('.chip-f').forEach(c => c.onclick = () => {
  toggle(types, c.dataset.f); c.classList.toggle('on', types.has(c.dataset.f)); apply(); });
document.querySelectorAll('#hand button').forEach(b => b.onclick = () => {
  hand = b.dataset.h; document.querySelectorAll('#hand button').forEach(x => x.classList.toggle('on', x === b)); apply(); });
const tiles = JSON.parse(document.getElementById('tiles-data').textContent);
const pick = document.getElementById('metric');
function showTiles() {
  document.getElementById('tiles').innerHTML = tiles[pick.value].map(([l, v]) =>
    `<div class="tile"><div class="l">${l}</div><div class="v">${v}</div></div>`).join('');
}
pick.onchange = () => { showTiles(); resize(); };
function resize() {
  const h = document.documentElement.scrollHeight + 4;
  window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setFrameHeight', height: h}, '*');
}
showTiles(); apply(); window.addEventListener('load', resize);
"""


def component_html(p: pd.DataFrame, pas: list[dict], line: dict, tiles: dict, metric_names: dict,
                   contact_note: str, dims: dict | None = None, park: str = "", default_metric: str = "GD%",
                   contact_tip: str = "", outline: dict | None = None) -> str:
    opts = "".join(f'<option value="{escape(k)}"{" selected" if k == default_metric else ""}>{escape(v)}</option>'
                   for k, v in metric_names.items())
    left = (f'<div class="left"><div class="card">{line_html(line)}</div>'
            f'<div class="card tiles"><div class="hd"><span class="lab">Compare</span>'
            f'<select id="metric">{opts}</select></div><div class="row" id="tiles"></div></div></div>')
    charts = f'<div class="card charts">{zone_svg(p)}{spray_svg(p, dims, park, outline)}{contact_svg(p, contact_note, contact_tip)}</div>'
    types = [t for t in pd.unique(p["ptype"]) if t]
    chips = "".join(f'<button class="chip-f on" data-f="{escape(t)}" style="--c:{PT_COLOR.get(t, INK)}">'
                    f'{escape(t)}</button>' for t in types)
    filt = ('<div class="filters"><span class="lab">Pitch types</span>' + chips +
            '<span class="sep"></span><span class="lab">Pitcher</span><div class="seg" id="hand">'
            '<button class="on" data-h="">All</button><button data-h="R">vs RHP</button>'
            '<button data-h="L">vs LHP</button></div><span class="faint" id="shown"></span></div>')
    bar = ('<div class="bar"><span class="ttl">Plate appearances</span><span class="sel-n" id="seln"></span>'
           '<span class="faint" id="scope"></span><button id="expand">Open all</button>'
           '<button id="collapse">Collapse all</button><button id="clear">Clear selection</button></div>' + filt)
    data = json.dumps(tiles).replace("</", "<\\/")
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>'
            f'<div id="aar"><div class="top">{left}{charts}</div>{bar}'
            f'<div class="wrap">{table_html(p, pas)}</div></div>'
            f'<script type="application/json" id="tiles-data">{data}</script><script>{JS}</script></body></html>')


def estimate_height(n_pas: int, n_pitches: int) -> int:
    """Starting height before the panel reports its own (every PA open)."""
    return 560 + 40 * n_pas + 34 * n_pitches + 40
