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

CUBE = ('<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.8l8 4.6v9.2l-8 4.6-8-4.6V7.4z"/>'
        '<path d="M4 7.4l8 4.6 8-4.6M12 12v9.2"/></svg>')
THREE_JS = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"

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


def _by_spray(fx, fy):
    """Outline points in order from the left-field line to the right-field line (the traced file isn't always in
    order, and joining points out of order scrambles the shape)."""
    fx, fy = np.asarray(fx, float), np.asarray(fy, float)
    ok = np.isfinite(fx) & np.isfinite(fy) & (np.hypot(fx, fy) > 1)
    fx, fy = fx[ok], fy[ok]
    o = np.argsort(np.arctan2(fx, fy))
    return fx[o], fy[o]


def _around_center(fx, fy):
    """A closed outline's points in order around its own middle."""
    fx, fy = np.asarray(fx, float), np.asarray(fy, float)
    ok = np.isfinite(fx) & np.isfinite(fy)
    fx, fy = fx[ok], fy[ok]
    o = np.argsort(np.arctan2(fx - fx.mean(), fy - fy.mean()))
    return fx[o], fy[o]


def _poly(pts) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def spray_svg(p: pd.DataFrame, dims: dict | None = None, park: str = "", outline: dict | None = None) -> str:
    """Spray chart on the actual park: the traced outline (fence, dirt, grass, lines) when it matches this venue's
    published distances; otherwise straight lines between the published distances (labeled approximate)."""
    W, H = 290, 300
    dims = dims or DEFAULT_DIMS
    segs = (outline or {}).get("segments") or {}
    if segs:                                  # fit the whole park: its depth AND its width
        wx_, wy_ = segs["outfield_outer"]
        deep, wide = float(np.max(wy_)), float(np.max(np.abs(wx_)))
    else:
        deep = max(dims.values())
        wide = deep * 0.75
    sc = min(262 / (deep + 14), (W / 2 - 10) / (wide + 6))
    hx, hy = W / 2, 290
    pt = lambda fx, fy: (hx + fx * sc, hy - fy * sc)
    g = []
    if segs:
        fx, fy = _by_spray(*segs["outfield_outer"])
        fence = [pt(a, b) for a, b in zip(fx, fy)]
        g.append(f'<polygon points="{hx:.1f},{hy:.1f} {_poly(fence)}" fill="#eef5ef" stroke="#5f8f96" '
                 f'stroke-width="1.6" stroke-linejoin="round"/>')
        if "infield_outer" in segs:                                # dirt: arc from line to line, closed at home
            ix, iy = _by_spray(*segs["infield_outer"])
            g.append(f'<polygon points="{hx:.1f},{hy:.1f} {_poly(pt(a, b) for a, b in zip(ix, iy))}" '
                     f'fill="#f4ede2" stroke="#c9b48f" stroke-width=".8"/>')
        if "infield_inner" in segs:                                # infield grass: a closed shape
            ix, iy = _around_center(*segs["infield_inner"])
            g.append(f'<polygon points="{_poly(pt(a, b) for a, b in zip(ix, iy))}" fill="#e8f1e8" '
                     f'stroke="#c9b48f" stroke-width=".8"/>')
        for a_, b_ in ((fx[0], fy[0]), (fx[-1], fy[-1])):          # foul lines: home -> each pole
            x2, y2 = pt(a_, b_)
            g.append(f'<line x1="{hx:.1f}" y1="{hy:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#b9c7cb" '
                     f'stroke-width=".9"/>')
        note = ""
    else:
        pts = [pt(dims[k] * math.sin(math.radians(a)), dims[k] * math.cos(math.radians(a)))
               for k, a in LINE_ANGLES.items() if k in dims]
        g.append(f'<polygon points="{hx:.1f},{hy:.1f} {_poly(pts)}" fill="#eef5ef" stroke="#7fa3a9" '
                 f'stroke-width="1.3" stroke-dasharray="5 3"/>')
        b1, b2, b3 = pt(63.6, 63.6), pt(0, 127.3), pt(-63.6, 63.6)
        g.append(f'<polygon points="{hx},{hy} {_poly([b1, b2, b3])}" fill="#f4ede2" stroke="#c9b48f"/>')
        note = " (approximate fence)"
    if segs:                         # labels where they belong on the real wall: the poles + the deepest point
        dist = np.hypot(fx, fy)
        spots = {"leftLine": (fx[0], fy[0], "start", 5), "rightLine": (fx[-1], fy[-1], "end", -5),
                 "center": (fx[int(np.argmax(dist))], fy[int(np.argmax(dist))], "middle", 0)}
    else:
        spots = {k: (dims[k] * math.sin(math.radians(LINE_ANGLES[k])), dims[k] * math.cos(math.radians(LINE_ANGLES[k])),
                     anchor, dx) for k, anchor, dx in (("leftLine", "start", 5), ("center", "middle", 0),
                                                       ("rightLine", "end", -5)) if k in dims}
    for k, (ax_, ay_, anchor, dx) in spots.items():
        if k in dims:
            x, y = pt(ax_, ay_)
            g.append(f'<text x="{x + dx:.1f}" y="{y - 5:.1f}" class="ax" text-anchor="{anchor}">{dims[k]:.0f}</text>')
    for r in p.itertuples():
        if r.bip and r.hc_x == r.hc_x and r.hc_y == r.hc_y:
            x, y = pt((r.hc_x - 125.42) * 2.5, (198.27 - r.hc_y) * 2.5)
            g.append(_mark(r, r.bb or "GB", max(9, min(W - 9, x)), max(9, min(H - 9, y)), _res_color(r.play)))
    leg = ("▼ GB ▲ LD ◆ FB ■ PU &nbsp;" + " ".join(f'<i style="color:{c}">●</i> {k.replace("Home run", "HR")}'
                                                 for k, c in RES_COLOR.items()) + f' <i style="color:{OUT_COLOR}">●</i> Out')
    return _chart("Spray", escape(park + note), f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>', leg,
                  tip=(outline or {}).get("note", ""))


def contact_svg(p: pd.DataFrame, note: str, tip: str = "", estimated: bool = False) -> str:
    """Point of contact on the plate (catcher's view): front-center of the plate = 0, 0; up = out in front."""
    have = p[p["pf"].notna() & p["pxc"].notna()]                 # fouls + balls in play only (see core.aar)
    title = "Point of contact*" if estimated else "Point of contact"
    if have.empty:
        return _chart(title, "", f'<div class="none">{escape(note)}</div>', "")
    W, H = 270, 330
    xs = lambda v: W / 2 + v * 4.6
    ys = lambda v: 160 - v * 4.6                                 # 30" out front down to 35" behind the front
    g = []
    for v in range(-35, 35, 5):
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
        color = _res_color(r.play) if r.bip else SWING
        x, y = max(9, min(W - 9, xs(r.pxc))), max(9, min(H - 16, ys(r.pf)))
        g.append(_mark(r, r.bb if r.bip else "", x, y, color))
    how = ('<b>*</b> front to back estimated -- he isn\'t on Savant\'s stance board yet' if estimated else
           'front to back = MLB\'s contact point vs the plate; side to side = where the ball was')
    leg = (f'{how} &nbsp; <i style="color:{SWING}">●</i> Foul &nbsp;in play: shape = BB type, '
           f'color = result &nbsp;<span class="faint">inches · catcher\'s view</span>')
    return _chart(title, "on the plate", f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>', leg,
                  tip=tip)


def _chart(title: str, sub: str, body: str, leg: str, tip: str = "") -> str:
    return (f'<div class="chart" title="{escape(tip)}"><div class="t">{title}'
            f'{f" <span class=faint>· {sub}</span>" if sub else ""}</div>{body}<div class="leg">{leg}</div></div>')


# ----------------------------------------------------------------------------- the one table
COLS = [("n", "#", "r"), ("video", "Video", "c"), ("inn", "Inn", "r"), ("count", "Count", "r"),
        ("pitcher", "Pitcher", "l"), ("result", "Pitch Result", "l"), ("play", "Play Result", "l"),
        ("ptype", "Type", "c"), ("velo", "Velo", "r"), ("hb", "HB", "r"), ("ivb", "IVB", "r"),
        ("bb", "BB Type", "c"), ("spin", "Contact", "c"), ("ev", "EV", "r"), ("bat_speed", "Bat Speed", "r"), ("la", "LA", "r"),
        ("dist", "Dist", "r"), ("xwoba", "xwOBA", "r"), ("zone", "Zone", "c"), ("dec", "Decision", "c"),
        ("dec_runs", "Dec Runs", "r")]


SPIN_COLOR = {"Optimal": ("#cdebd6", "#14532d"), "Backspin": ("#dbe7f6", "#1e3a6b"), "Topspin": ("#f6e3cf", "#7a3e0c"),
              "Hook": ("#efdcf3", "#5b2470"), "Slice": ("#f7dde4", "#7a1f3d"), "Knuckle": ("#eef1f3", "#3d4650")}


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
        cam = (f'<a class="vid" href="{SAVANT_VIDEO.format(v)}" target="_blank" rel="noopener" '
               f'title="Play this pitch">{CAMERA}</a>' if v else "")
        s3 = (f'<button class="s3d" data-n="{r.n}" title="Swing + pitch in 3D">{CUBE}</button>'
              if getattr(r, "has3d", False) else "")
        return (f'<span class="vcell">{cam}{s3}</span>' if (cam or s3) else "–"), ""
    if key == "inn":
        return f"{r.half}{r.inning}", ""
    if key == "spin":
        if not v:
            return "–", ""
        bg, fg = SPIN_COLOR.get(v, ("#eef1f3", INK))
        return (f'<span class="chip" style="background:{bg};color:{fg}" '
                f'title="{escape(getattr(r, "spin_why", ""))}">{v}</span>'), ""
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
.tiles .row{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}}
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
.vcell{{display:inline-flex;gap:6px;align-items:center}}
button.s3d{{border:none;background:none;padding:0;color:{TEAL};cursor:pointer;display:inline-flex}}
button.s3d:hover{{color:{INK}}}
#ovl{{position:absolute;left:0;top:0;width:100%;background:rgba(22,32,43,.55);z-index:50;display:none}}
#ovl.on{{display:block}}
#dlg{{position:absolute;left:50%;transform:translateX(-50%);width:min(1180px,96%);background:#fff;border-radius:12px;
  box-shadow:0 12px 40px rgba(0,0,0,.35);overflow:hidden}}
#dlg .dh{{display:flex;align-items:center;gap:12px;padding:10px 14px;border-bottom:1px solid {LINE}}}
#dlg .dh b{{font:700 15px system-ui}} #dlg .dh .x{{margin-left:auto;border:none;background:none;font:22px system-ui;
  cursor:pointer;color:{MUTED};line-height:1}} #dlg .dh .x:hover{{color:{INK}}}
#dlg .db{{display:grid;grid-template-columns:1fr 280px}}
#stage{{position:relative;height:560px;background:#f4f7f8;cursor:grab}} #stage:active{{cursor:grabbing}}
#stage canvas{{display:block}} #stage .msg{{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:center;color:{MUTED};font:13px system-ui;text-align:center;padding:30px}}
#cams{{position:absolute;left:10px;top:10px;display:flex;gap:4px;flex-wrap:wrap}}
#cams button,#ctl button{{font:600 11.5px system-ui;border:1px solid {LINE};background:#fff;border-radius:6px;
  padding:4px 8px;cursor:pointer}} #cams button.on{{background:{TEAL};color:#fff;border-color:{TEAL}}}
#ctl{{display:flex;align-items:center;gap:8px;padding:8px 12px;border-top:1px solid {LINE}}}
#ctl select{{font:12px system-ui}} #ctl .scw{{flex:1;position:relative;display:flex}} #ctl input[type=range]{{flex:1}}
#ctl .tick{{position:absolute;top:-6px;width:2px;height:8px;background:#c0392b;transform:translateX(-1px);pointer-events:none}}
#ctl .tick.bp{{background:#7b5cc4}} #ctl #s3bp{{color:#5b3fa8}} #ctl .clk{{font:12px ui-monospace,monospace;
  color:{MUTED};min-width:120px;text-align:right}}
#ctl a{{font:600 12px system-ui;color:#fff;background:{INK};border-radius:6px;padding:5px 10px;text-decoration:none;
  display:inline-flex;gap:5px;align-items:center}}
#rd{{border-left:1px solid {LINE};padding:10px 12px;font:12.5px system-ui;overflow:auto;max-height:620px}}
#rd h5{{margin:10px 0 4px;font:700 11px system-ui;color:{MUTED};text-transform:uppercase;letter-spacing:.04em}}
#rd table{{width:100%;border-collapse:collapse}} #rd td{{padding:3px 0;border-bottom:1px solid #f0f2f4}}
#rd td:last-child{{text-align:right;font-family:ui-monospace,monospace}}
#rd .lg{{font-size:11px;color:{MUTED};line-height:1.6;margin-top:8px}} #rd .lg i{{font-style:normal;font-weight:800}}
.tag{{font-size:9.5px;padding:0 5px;border-radius:7px;margin-left:5px;background:#dbeef1;color:#2c6670}}
.tag.x{{background:#eef1f3;color:{MUTED}}} .tag.inf{{background:#ece6f8;color:#5b3fa8}}
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
  if (e.target.closest('a') || e.target.closest('button')) return; toggle(sel, r.dataset.n); apply(); }));
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
  document.getElementById('tiles').innerHTML = tiles[pick.value].map(([l, v, st]) =>
    `<div class="tile" style="${st || ''}"><div class="l">${l}</div><div class="v">${v}</div></div>`).join('');
}
pick.onchange = () => { showTiles(); resize(); };
function resize() {
  const h = document.documentElement.scrollHeight + 4;
  window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setFrameHeight', height: h}, '*');
}
showTiles(); apply(); window.addEventListener('load', resize);
"""


OVERLAY = """<div id="ovl"><div id="dlg"><div class="dh"><b id="s3t"></b><span class="faint" id="s3s"></span>
<button class="x" id="s3x" title="Close (Esc)">&times;</button></div><div class="db"><div><div id="stage">
<div id="cams"></div><div class="msg" id="s3msg">Loading 3D…</div></div><div id="ctl"><button id="s3p">❚❚</button>
<select id="s3sp"><option value="4">1/4 speed</option><option value="10" selected>1/10</option>
<option value="25">1/25</option><option value="60">1/60</option></select><span class="scw"><input id="s3sc" type="range" min="0"
max="1000" value="0"><span id="s3tickbp" class="tick bp" title="Bat parallel (inferred)"></span><span id="s3tick" class="tick" title="Contact"></span></span><button id="s3bp" title="Jump to bat parallel (inferred)">Bat parallel</button><button id="s3k" title="Jump to contact">Contact</button><span class="clk" id="s3c"></span><a id="s3v" target="_blank" rel="noopener">Video</a></div></div>
<div id="rd"></div></div></div></div>"""

S3D_JS = r"""
const S3 = JSON.parse(document.getElementById('s3d-data').textContent);
const ovl = document.getElementById('ovl'), dlg = document.getElementById('dlg'), stage = document.getElementById('stage');
let R3 = null;                     // live renderer state
function loadThree(cb) {
  if (window.THREE) return cb();
  const s = document.createElement('script'); s.src = THREE_SRC; s.onload = cb;
  s.onerror = () => { document.getElementById('s3msg').textContent = "Couldn't load the 3D library (needs an internet connection)."; };
  document.head.appendChild(s);
}
const V = (p) => new THREE.Vector3(p[0], p[2], -p[1]);   // plate frame (x, y toward pitcher, z up) -> three (x, up, -y)
function fmt(v, d = 1) { return v === null || v === undefined ? '–' : (+v).toFixed(d); }
function at(pts, t) {
  const T = q => q[q.length - 1];
  if (!pts || !pts.length || t < T(pts[0])) return null;
  for (let k = 1; k < pts.length; k++) if (T(pts[k]) >= t) {
    const a = pts[k - 1], b = pts[k], f = (t - T(a)) / ((T(b) - T(a)) || 1);
    return [0, 1, 2].map(i => a[i] + (b[i] - a[i]) * f);
  }
  return null;
}
function upto(pts, t) { return pts ? pts.filter(q => q[q.length - 1] <= t) : []; }
function line(pts, mat) {
  const g = new THREE.BufferGeometry().setFromPoints(pts.map(V)); const l = new THREE.Line(g, mat);
  if (mat.isLineDashedMaterial) l.computeLineDistances(); return l;
}
function readouts(sc) {
  const r = sc.read, m = '<span class="tag">measured</span>', i = '<span class="tag inf">inferred</span>';
  const row = (k, v, t = '') => `<tr><td>${k}${t}</td><td>${v}</td></tr>`;
  const smash = r.smash, sqp = r.squared_up_pct;
  document.getElementById('rd').innerHTML =
    '<h5>Swing</h5><table>' + row('Bat speed', fmt(r.bat_speed) + ' mph', m) + row('Swing length', fmt(r.swing_length) + ' ft', m) +
    row('Attack angle', fmt(r.attack_angle) + '°', m) + row('Attack direction', fmt(r.attack_direction) + '° ' +
      (r.attack_direction < 0 ? 'pull' : 'oppo'), m) + row('Swing tilt', fmt(r.tilt) + '°', m) +
    row('Vertical bat angle at contact', fmt(r.vba) + '°', i) +
    row('Bat parallel', r.bp_in ? fmt(r.bp_in, 0) + '″ of path before contact' : 'past his swing length', i) + '</table>' +
    '<h5>Contact (vs the plate)</h5><table>' + row('Out front', fmt(r.front) + '″', m) +
    row('Side (+ = 1B side)', fmt(r.side_in) + '″', m) + row('Height', fmt(r.height) + '″', m) + '</table>' +
    '<h5>Pitch at contact</h5><table>' + row(r.ptype + ' velo', fmt(r.velo) + ' mph', m) +
    row('Coming down at', fmt(-r.vaa) + '°', m) + row('Swing vs pitch plane', (r.plane_gap > 0 ? '+' : '') + fmt(r.plane_gap) + '°',
      '') + '</table>' +
    (r.ev ? '<h5>Ball off the bat</h5><table>' + row('Exit velo', fmt(r.ev) + ' mph', m) + row('Launch angle', fmt(r.la, 0) + '°', m) +
      (sqp ? row('Squared-up', fmt(sqp, 0) + '% of max' + (sqp >= 80 ? ' ✓' : ''), m) : '') +
      (smash ? row('Smash factor', fmt(smash, 2), m) : '') +
      (r.spin ? row('Contact', '<b>' + r.spin + '</b>', i) + `<tr><td colspan="2" style="color:#6a7681;font-size:11px">${r.spin_why}</td></tr>` : '') +
      '</table>' : '') +
    '<div class="lg"><i style="color:#e07b24">━</i> pitch (measured) &nbsp; <i style="color:#2f5fb3">━</i> bat, last 40 ms ' +
    'before contact (measured) &nbsp; <i style="color:#7b5cc4">┅ ●</i> path back to bat parallel -- inferred from the ' +
    'measured tilt, attack angle + direction (the bat lies across its path in that plane), and the follow-through &nbsp; ' +
    '<i style="color:#c0392b">●</i> contact &nbsp; <i style="color:#2e9d5b">━</i> ball off the bat' +
    (sc.exit ? '' : ' (fouls: direction not tracked)') + '<br>Swing vs pitch plane = attack angle minus how steeply ' +
    'the pitch is coming down (0 = matched).</div>';
}
function build(sc) {
  const W = stage.clientWidth, H = stage.clientHeight;
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0xf4f7f8);
  const cam = new THREE.PerspectiveCamera(38, W / H, 1, 20000);
  const ren = new THREE.WebGLRenderer({antialias: true}); ren.setSize(W, H); ren.setPixelRatio(Math.min(2, window.devicePixelRatio));
  stage.appendChild(ren.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x99aabb, 0.9)); const dl = new THREE.DirectionalLight(0xffffff, 0.6);
  dl.position.set(100, 300, 200); scene.add(dl);
  const dark = new THREE.LineBasicMaterial({color: 0x16202b});
  // ground + plate + boxes + foul lines (the field's own measurements)
  const shp = new THREE.Shape([[-8.5, 0], [8.5, 0], [8.5, -8.5], [0, -17], [-8.5, -8.5]].map(([x, y]) => new THREE.Vector2(x, y)));
  const plate = new THREE.Mesh(new THREE.ShapeGeometry(shp), new THREE.MeshBasicMaterial({color: 0xdde4e8, side: THREE.DoubleSide}));
  plate.rotation.x = -Math.PI / 2; plate.position.y = 0.1; scene.add(plate);
  scene.add(line([[-8.5, 0, .2], [8.5, 0, .2], [8.5, -8.5, .2], [0, -17, .2], [-8.5, -8.5, .2], [-8.5, 0, .2]], dark));
  for (const sg of [-1, 1]) {
    scene.add(line([[sg * 14.5, -44.5, .2], [sg * 62.5, -44.5, .2], [sg * 62.5, 27.5, .2], [sg * 14.5, 27.5, .2], [sg * 14.5, -44.5, .2]], dark));
    scene.add(line([[sg * 44.5, 27.5, .2], [sg * 1200, -17 + 1200, .2]], dark));
  }
  // his strike zone at the front of the plate
  scene.add(line([[-8.5, 0, sc.sz[0]], [8.5, 0, sc.sz[0]], [8.5, 0, sc.sz[1]], [-8.5, 0, sc.sz[1]], [-8.5, 0, sc.sz[0]]],
    new THREE.LineDashedMaterial({color: 0x16202b, dashSize: 2, gapSize: 1.5})));
  // ghosts of the whole thing
  scene.add(line(sc.pitch, new THREE.LineBasicMaterial({color: 0xe07b24, transparent: true, opacity: .25})));
  const dashed = new THREE.LineDashedMaterial({color: 0x7b5cc4, dashSize: 2.2, gapSize: 1.6});
  const pre = sc.arc.filter(q => q[3] === 2 && q[4] < 0), post = sc.arc.filter(q => q[3] === 2 && q[4] > 0);
  const m0 = sc.arc.find(q => q[3] === 1), m1 = [...sc.arc].reverse().find(q => q[3] === 1);
  if (pre.length) scene.add(line(pre.concat([m0]), dashed));
  if (post.length) scene.add(line([m1].concat(post), dashed));
  if (sc.bp) { const m = new THREE.Mesh(new THREE.SphereGeometry(1.8, 16, 12), new THREE.MeshBasicMaterial({color: 0x7b5cc4}));
    m.position.copy(V(sc.bp.at)); scene.add(m); }
  const meas = sc.arc.filter(q => q[3] === 1);
  if (meas.length > 1) scene.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(meas.map(V)), 40, 0.6, 8),
    new THREE.MeshLambertMaterial({color: 0x2f5fb3})));
  if (sc.exit) scene.add(line(sc.exit, new THREE.LineBasicMaterial({color: 0x2e9d5b, transparent: true, opacity: .3})));
  // live pieces
  const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: 0xe07b24}));
  const etrail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: 0x2e9d5b}));
  scene.add(trail, etrail);
  const ball = new THREE.Mesh(new THREE.SphereGeometry(1.45, 20, 14), new THREE.MeshLambertMaterial({color: 0xffffff, emissive: 0x332211}));
  scene.add(ball);
  const bat = new THREE.Mesh(new THREE.CylinderGeometry(1.3, 0.55, 33, 16), new THREE.MeshLambertMaterial({color: 0x9a7b52}));
  scene.add(bat);
  const hit = new THREE.Mesh(new THREE.SphereGeometry(2.2, 18, 12), new THREE.MeshBasicMaterial({color: 0xc0392b}));
  hit.position.copy(V(sc.contact)); scene.add(hit);
  // camera: orbit around the contact point
  const tgt = V([sc.contact[0] * 0.5, sc.contact[1] * 0.5 - 4, 30]);
  const st = {th: 0, ph: 1.25, r: 190};
  const openSide = sc.side === 'R' ? 1 : -1;
  const PRESETS = {Catcher: {th: 0, ph: 1.32, r: 170}, 'Open side': {th: openSide * Math.PI / 2, ph: 1.38, r: 190},
                   Overhead: {th: 0.0001, ph: 0.06, r: 260}, Pitcher: {th: Math.PI, ph: 1.4, r: 260}};
  function place() {
    cam.position.set(tgt.x + st.r * Math.sin(st.ph) * Math.sin(st.th), tgt.y + st.r * Math.cos(st.ph),
                     tgt.z + st.r * Math.sin(st.ph) * Math.cos(st.th));
    cam.lookAt(tgt);
  }
  const cams = document.getElementById('cams'); cams.innerHTML = '';
  Object.entries(PRESETS).forEach(([k, v], i) => {
    const b = document.createElement('button'); b.textContent = k; if (!i) b.classList.add('on');
    b.onclick = () => { Object.assign(st, v); place(); cams.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); };
    cams.appendChild(b);
  });
  let drag = null;
  ren.domElement.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; ren.domElement.setPointerCapture(e.pointerId); });
  ren.domElement.addEventListener('pointermove', e => {
    if (!drag) return; st.th -= (e.clientX - drag[0]) * 0.008; st.ph = Math.max(0.05, Math.min(1.55, st.ph - (e.clientY - drag[1]) * 0.008));
    drag = [e.clientX, e.clientY]; place(); cams.querySelectorAll('button').forEach(x => x.classList.remove('on'));
  });
  ren.domElement.addEventListener('pointerup', () => { drag = null; });
  ren.domElement.addEventListener('wheel', e => { e.preventDefault(); st.r = Math.max(40, Math.min(1500, st.r * (1 + e.deltaY * 0.001))); place(); }, {passive: false});
  place();
  // time
  const T = q => q[q.length - 1];
  const t0 = T(sc.pitch[0]), t1 = Math.max(sc.exit ? 0.12 : 0.02, T(sc.arc[sc.arc.length - 1]));
  const S = {t: t0, playing: true, last: null, raf: 0};
  function frame(t) {
    const bt = upto(sc.pitch, t).map(V);
    trail.geometry.dispose(); trail.geometry = new THREE.BufferGeometry().setFromPoints(bt.length > 1 ? bt : [V(sc.pitch[0]), V(sc.pitch[0])]);
    const et = t > 0 && sc.exit ? upto(sc.exit, t).map(V) : [];
    etrail.geometry.dispose(); etrail.geometry = new THREE.BufferGeometry().setFromPoints(et.length > 1 ? et : [V(sc.contact), V(sc.contact)]);
    const B = t <= 0 ? at(sc.pitch, t) : (sc.exit ? at(sc.exit, t) : null);
    ball.visible = !!B; if (B) ball.position.copy(V(B));
      const tEnd = T(sc.arc[sc.arc.length - 1]);
    const P = t < T(sc.arc[0]) ? null : at(sc.arc, Math.min(t, tEnd));        // holds the finish of the follow-through
    bat.visible = !!P;
    if (P) {
      const u = new THREE.Vector3(sc.O[0] - P[0], sc.O[2] - P[2], -(sc.O[1] - P[1])).normalize();   // sweet spot -> hands
      const mid = V(P).add(u.clone().multiplyScalar(10.5));                                         // 33" bat, sweet spot 6" from the end
      bat.position.copy(mid); bat.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), u);
    }
    hit.visible = t >= 0;
    const pre = sc.arc.find(q => q[3] === 1), winStart = pre ? T(pre) : 0;
    document.getElementById('s3c').textContent = ((sc.bp && t < winStart && t >= T(sc.arc[0])) || (t > 0 && !sc.exit))
      ? 'inferred path · time not measured' : `${t >= 0 ? '+' : ''}${(t * 1000).toFixed(0)} ms from contact`;
    ren.render(scene, cam);
  }
  function tick(now) {
    if (S.playing && S.last !== null) {
      S.t += (now - S.last) / 1000 / +document.getElementById('s3sp').value;
      if (S.t > t1 + 0.02) S.t = t0;
      document.getElementById('s3sc').value = Math.round((Math.min(S.t, t1) - t0) / (t1 - t0) * 1000);
    }
    S.last = now; frame(Math.min(S.t, t1)); S.raf = requestAnimationFrame(tick);
  }
  S.raf = requestAnimationFrame(tick);
  document.getElementById('s3p').onclick = e => { S.playing = !S.playing; e.target.textContent = S.playing ? '❚❚' : '▶'; };
  const toContact = () => { S.playing = false; document.getElementById('s3p').textContent = '▶'; S.t = 0;
    document.getElementById('s3sc').value = Math.round((0 - t0) / (t1 - t0) * 1000); };
  document.getElementById('s3k').onclick = toContact;
  document.getElementById('s3tick').style.left = (100 * (0 - t0) / (t1 - t0)) + '%';
  const bpb = document.getElementById('s3bp'), bpt = document.getElementById('s3tickbp');
  if (sc.bp) {
    bpb.style.display = bpt.style.display = '';
    bpt.style.left = (100 * (sc.bp.t - t0) / (t1 - t0)) + '%';
    bpb.onclick = () => { S.playing = false; document.getElementById('s3p').textContent = '▶'; S.t = sc.bp.t;
      document.getElementById('s3sc').value = Math.round((sc.bp.t - t0) / (t1 - t0) * 1000); };
  } else { bpb.style.display = bpt.style.display = 'none'; }
  document.getElementById('s3sc').oninput = e => {
    S.playing = false; document.getElementById('s3p').textContent = '▶'; S.t = t0 + (t1 - t0) * e.target.value / 1000; };
  return {stop() { cancelAnimationFrame(S.raf); ren.dispose(); ren.domElement.remove(); }};
}
function open3d(n, btn) {
  const sc = S3[n]; if (!sc) return;
  const r = sc.read;
  document.getElementById('s3t').textContent = `${r.count} ${r.ptype} ${fmt(r.velo)} · ${r.result}`;
  document.getElementById('s3s').textContent = 'drag to orbit · scroll to zoom';
  const v = document.getElementById('s3v');
  if (r.video) { v.href = 'https://baseballsavant.mlb.com/sporty-videos?playId=' + r.video; v.style.display = ''; } else v.style.display = 'none';
  readouts(sc);
  const docH = Math.max(document.documentElement.scrollHeight, 760);
  ovl.style.height = docH + 'px'; ovl.classList.add('on');
  const rowTop = btn.getBoundingClientRect().top + window.scrollY;
  dlg.style.top = Math.max(10, Math.min(rowTop - 160, docH - dlg.offsetHeight - 10)) + 'px';
  if (document.documentElement.scrollHeight < docH) document.body.style.minHeight = docH + 'px';
  resize();
  const msg = document.getElementById('s3msg'); msg.style.display = '';
  loadThree(() => { msg.style.display = 'none'; if (R3) R3.stop(); R3 = build(sc); });
}
function close3d() { ovl.classList.remove('on'); if (R3) { R3.stop(); R3 = null; } document.body.style.minHeight = ''; resize(); }
document.querySelectorAll('button.s3d').forEach(b => b.addEventListener('click', e => { e.stopPropagation(); open3d(b.dataset.n, b); }));
document.getElementById('s3x').onclick = close3d;
ovl.addEventListener('click', e => { if (e.target === ovl) close3d(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape' && ovl.classList.contains('on')) close3d(); });
"""


def component_html(p: pd.DataFrame, pas: list[dict], line: dict, tiles: dict, metric_names: dict,
                   contact_note: str, dims: dict | None = None, park: str = "", default_metric: str = "GD%",
                   contact_tip: str = "", outline: dict | None = None, contact_est: bool = False,
                   header: str = "", scenes3d: dict | None = None) -> str:
    scenes3d = scenes3d or {}
    if "n" in p:
        p = p.assign(has3d=p["n"].isin(list(scenes3d)))
        pas = [{**pa, "rows": pa["rows"].assign(has3d=pa["rows"]["n"].isin(list(scenes3d)))} for pa in pas]
    opts = "".join(f'<option value="{escape(k)}"{" selected" if k == default_metric else ""}>{escape(v)}</option>'
                   for k, v in metric_names.items())
    left = (f'<div class="left"><div class="card">{line_html(line)}</div>'
            f'<div class="card tiles"><div class="hd"><span class="lab">Compare</span>'
            f'<select id="metric">{opts}</select></div><div class="row" id="tiles"></div></div></div>')
    charts = f'<div class="card charts">{zone_svg(p)}{spray_svg(p, dims, park, outline)}{contact_svg(p, contact_note, contact_tip, contact_est)}</div>'
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
    s3 = json.dumps({str(k): v for k, v in scenes3d.items()}).replace("</", "<\\/")
    head = header.replace("<style>", "<style>", 1) if header else ""
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>'
            f'<div id="aar">{head}<div class="top">{left}{charts}</div>{bar}'
            f'<div class="wrap">{table_html(p, pas)}</div></div>{OVERLAY}'
            f'<script type="application/json" id="tiles-data">{data}</script>'
            f'<script type="application/json" id="s3d-data">{s3}</script><script>{JS}</script>'
            f'<script>const THREE_SRC = "{THREE_JS}";{S3D_JS}</script></body></html>')


def estimate_height(n_pas: int, n_pitches: int) -> int:
    """Starting height before the panel reports its own (every PA open)."""
    return 560 + 40 * n_pas + 34 * n_pitches + 40
