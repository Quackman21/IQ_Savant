"""Point-of-contact map for the Bat Path tab (top-down on the plate, catcher's view)."""
from __future__ import annotations

import math
from html import escape

import numpy as np
import pandas as pd

PLATE_HALF = 8.5
GROUP_COLOR = {"Fastball": "#D22D49", "Breaking": "#3C7DD9", "Offspeed": "#1DBE3A"}
CSS = """<style>
.bp{font-family:system-ui,sans-serif;color:#16202b;display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start}
.bp .card{border:1px solid #d9dee2;border-radius:8px;padding:10px 12px;background:#fff}
.bp .t{font:700 15px system-ui,sans-serif;margin-bottom:4px}
.bp .faint{color:#6a7681;font-weight:400}
.bp .ax{font:10px system-ui,sans-serif;fill:#6a7681}
.bp .leg{font-size:12px;color:#33404c;margin-top:6px;max-width:460px;line-height:1.45}
.bp table{border-collapse:collapse;font-size:13px}
.bp th{font-weight:700;padding:5px 9px;border-bottom:1.5px solid #9aa4ad;text-align:right}
.bp th:first-child,.bp td:first-child{text-align:left;font-family:system-ui,sans-serif}
.bp td{padding:5px 9px;border-bottom:1px solid #e6eaed;text-align:right;font-family:ui-monospace,Menlo,monospace}
</style>"""


# ----------------------------------------------------------------------------- colour coding + heat maps
COLOR_BY = ["Swing Group", "Exit Velo", "Launch Angle", "Hit Type", "Result", "Contact Type", "Pitch Type",
            "Pitch Group"]
DISPLAY = ["Points", "Heat Map"]
NUMERIC = {"Exit Velo": ("ev", 70.0, 106.0, " mph", "soft", "hard"),
           "Launch Angle": ("la", -20.0, 50.0, "°", "on the ground", "in the air")}
HIT_TYPE = {"GB": "#a0703c", "LD": "#e4572e", "FB": "#2e86de", "PU": "#9b59b6"}
HIT_NAME = {"GB": "Ground ball", "LD": "Line drive", "FB": "Fly ball", "PU": "Pop-up"}
CONTACT_COLOR = {"Optimal": "#1f9d55", "Backspin": "#2e6fd1", "Topspin": "#d9822b", "Hook": "#8e44ad",
                 "Slice": "#d6336c", "Knuckle": "#7f8c8d"}
GRAY = "#9aa4ad"


def _scale(v, lo: float, hi: float) -> str:
    """Blue (low) -> white (middle) -> red (high)."""
    if v is None or pd.isna(v):
        return GRAY
    t = float(np.clip((v - (lo + hi) / 2) / ((hi - lo) / 2), -1, 1))
    if t >= 0:
        r, g, b = 255, int(255 - 200 * t), int(255 - 210 * t)
    else:
        r, g, b = int(255 + 205 * t), int(255 + 130 * t), 255
    return f"#{r:02x}{g:02x}{b:02x}"


def _ev_color(ev) -> str:
    return _scale(ev, 70.0, 106.0)


def _cat(r, mode: str) -> str | None:
    """The category a point falls in for a categorical colour mode (None = none)."""
    v = {"Hit Type": getattr(r, "bb", None), "Result": getattr(r, "result", None),
         "Contact Type": getattr(r, "ctype", None), "Pitch Type": getattr(r, "ptype", None),
         "Pitch Group": getattr(r, "pgroup", None)}.get(mode)
    if mode == "Result" and not getattr(r, "bip", True):
        return "Foul"
    if mode == "Result" and v not in RES_COLOR_():
        return "Out"
    return v if isinstance(v, str) and v and v != "nan" else None


def RES_COLOR_() -> dict:
    from ui.aar import OUT_COLOR, RES_COLOR
    return {**RES_COLOR, "Out": OUT_COLOR, "Foul": "#c9cfd4"}


def _cat_colors(mode: str) -> dict:
    from ui.aar import PT_COLOR
    return {"Hit Type": HIT_TYPE, "Result": RES_COLOR_(), "Contact Type": CONTACT_COLOR, "Pitch Type": PT_COLOR,
            "Pitch Group": GROUP_COLOR}.get(mode, {})


def point_color(r, mode: str) -> str:
    if mode == "Swing Group":
        return getattr(r, "gcolor", "#2f5fb3") or "#2f5fb3"
    if mode in NUMERIC:
        col, lo, hi = NUMERIC[mode][:3]
        return _scale(getattr(r, col, None), lo, hi)
    return _cat_colors(mode).get(_cat(r, mode), GRAY)


def legend_html(d: pd.DataFrame, mode: str, display: str) -> str:
    sg = _groups_of(d)
    if mode == "Swing Group" or (display == "Heat Map" and mode not in NUMERIC and sg):
        if not sg:
            return "one swing group" + (" · heat = how often" if display == "Heat Map" else "")
        return ((("heat = how often, in each group's colour · " if display == "Heat Map" else "")) +
                " ".join(f'<span style="white-space:nowrap" title="{escape(n)}"><i style="color:{c};font-style:normal">'
                         f'{ {"circle": "●", "square": "■", "triangle": "▲", "diamond": "◆"}[SHAPES[k % 4]] }</i> '
                         f'{chr(65 + k)}</span>' for k, (n, c) in enumerate(sg)))
    if display == "Heat Map" and mode not in NUMERIC:
        bar = "linear-gradient(90deg,rgba(214,51,40,.08),rgba(214,51,40,.9))"
        return (f'<span style="display:inline-flex;align-items:center;gap:6px">heat = how often · rare '
                f'<i style="display:inline-block;width:90px;height:9px;border-radius:4px;background:{bar}"></i> often</span>')
    if mode in NUMERIC:
        col, lo, hi, unit, lw, hw = NUMERIC[mode]
        bar = f"linear-gradient(90deg,{_scale(lo, lo, hi)},#ffffff,{_scale(hi, lo, hi)})"
        what = "average " if display == "Heat Map" else ""
        return (f'<span style="display:inline-flex;align-items:center;gap:6px">{what}{mode.lower()}: {lo:.0f}{unit} '
                f'({lw}) <i style="display:inline-block;width:90px;height:9px;border-radius:4px;border:1px solid #d9dee2;'
                f'background:{bar}"></i> {hi:.0f}{unit} ({hw})</span>')
    cols = _cat_colors(mode)
    seen = {_cat(r, mode) for r in d.itertuples()} - {None}
    names = HIT_NAME if mode == "Hit Type" else {}
    return " ".join(f'<span style="white-space:nowrap"><i style="color:{c};font-style:normal">●</i> '
                    f'{escape(names.get(k, k))}</span>' for k, c in cols.items() if k in seen)


def heat_layer(px: np.ndarray, py: np.ndarray, vals: np.ndarray | None, W: float, H: float, mode: str,
               cell: float = 8.0, sigma: float = 14.0, unit: str = "", tint: str | None = None) -> list[str]:
    """Smoothed heat cells (SVG rects). vals=None -> how often (density); else the smoothed average of vals."""
    if len(px) == 0:
        return []
    nx, ny = int(np.ceil(W / cell)), int(np.ceil(H / cell))
    cx, cy = (np.arange(nx) + 0.5) * cell, (np.arange(ny) + 0.5) * cell
    wx = np.exp(-0.5 * ((cx[None, :] - px[:, None]) / sigma) ** 2)          # points x cols
    wy = np.exp(-0.5 * ((cy[None, :] - py[:, None]) / sigma) ** 2)          # points x rows
    dens = wy.T @ wx                                                        # rows x cols, ~ points nearby
    out = []
    if vals is None:
        top = dens.max()
        for j, i in zip(*np.nonzero(dens > 0.06 * top)):
            a = 0.08 + 0.82 * dens[j, i] / top
            out.append(f'<rect x="{i * cell:.1f}" y="{j * cell:.1f}" width="{cell + .3:.1f}" height="{cell + .3:.1f}" '
                       f'fill="{tint or "#d63328"}" fill-opacity="{a:.2f}"/>')
        return out
    ok = ~np.isnan(vals)
    if not ok.any():
        return []
    v = np.where(ok, vals, 0.0)
    num = (wy[ok].T * v[ok]) @ wx[ok]
    den = wy[ok].T @ wx[ok]
    col, lo, hi = NUMERIC[mode][:3]
    for j, i in zip(*np.nonzero(den >= 0.9)):                               # need about a ball's worth nearby
        m = num[j, i] / den[j, i]
        a = min(0.9, 0.35 + 0.15 * den[j, i])
        out.append(f'<rect x="{i * cell:.1f}" y="{j * cell:.1f}" width="{cell + .3:.1f}" height="{cell + .3:.1f}" '
                   f'fill="{_scale(m, lo, hi)}" fill-opacity="{a:.2f}"><title>{m:.0f}{unit} avg</title></rect>')
    return out


SHAPES = ["circle", "square", "triangle", "diamond"]


def _marker(k: int, x: float, y: float, r: float, style: str, tip: str) -> str:
    """Swing group k's marker shape (A circle, B square, C triangle, D diamond)."""
    t = f"<title>{escape(tip)}</title>"
    shape = SHAPES[k % 4]
    if shape == "square":
        return f'<rect class="g{k}" x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" {style}>{t}</rect>'
    if shape == "triangle":
        return (f'<polygon class="g{k}" points="{x:.1f},{y - 1.2 * r:.1f} {x + 1.1 * r:.1f},{y + .8 * r:.1f} '
                f'{x - 1.1 * r:.1f},{y + .8 * r:.1f}" {style}>{t}</polygon>')
    if shape == "diamond":
        return (f'<polygon class="g{k}" points="{x:.1f},{y - 1.3 * r:.1f} {x + 1.3 * r:.1f},{y:.1f} {x:.1f},'
                f'{y + 1.3 * r:.1f} {x - 1.3 * r:.1f},{y:.1f}" {style}>{t}</polygon>')
    return f'<circle class="g{k}" cx="{x:.1f}" cy="{y:.1f}" r="{r}" {style}>{t}</circle>'


def _chips(cid: str, groups: list[tuple]) -> str:
    """This chart's own on/off chips, one per swing group (groups = [(name, colour), ...])."""
    if len(groups) < 2:
        return ""
    glyph = {"circle": "●", "square": "■", "triangle": "▲", "diamond": "◆"}
    btn = "".join(f'<button class="on" data-k="{k}" title="{escape(n)}" style="--c:{c}">'
                  f'<span style="color:{c}">{glyph[SHAPES[k % 4]]}</span> {chr(65 + k)}</button>'
                  for k, (n, c) in enumerate(groups))
    js = (f"<script>document.querySelectorAll('#{cid} .chips button').forEach(b=>b.onclick=()=>{{"
          f"b.classList.toggle('on');const on=b.classList.contains('on');"
          f"document.querySelectorAll('#{cid} .g'+b.dataset.k).forEach(e=>e.style.display=on?'':'none');}});</script>")
    css = ("<style>.chips{display:flex;gap:5px;margin:2px 0 4px}.chips button{font:700 12px system-ui;border:1px solid "
           "#d9dee2;background:#fff;border-radius:6px;padding:3px 9px;cursor:pointer;color:#9aa4ad}"
           ".chips button.on{color:#16202b;border-color:var(--c)}.chips button:not(.on) span{opacity:.3}</style>")
    return css + f'<div class="chips">{btn}</div>' + js


def _groups_of(d: pd.DataFrame) -> list[tuple]:
    if "grp" not in d or d.empty:
        return []
    g = d.drop_duplicates("grp").sort_values("grp")
    return list(zip(g["gname"], g["gcolor"]))


def poc_map_html(d: pd.DataFrame, color_by: str = "Exit Velo", estimated: bool = False, summary: bool = True,
                 title: str = "Point of contact", display: str = "Points") -> str:
    """d: one row per contact with pf, pxc, ev, la, result, pgroup, bip, date, ptype."""
    have = d[d["pf"].notna() & d["pxc"].notna()] if {"pf", "pxc"} <= set(d.columns) else d.iloc[0:0]
    if have.empty:
        return CSS + '<div class="bp"><div class="card">No measured contact points for this selection.</div></div>'
    if summary:                                                   # whole season: fixed frame, +30" .. -40"
        top, bot, half = 30, -40, 25
    else:                                                         # one cell: frame the plate + his contacts
        top = int(np.ceil(max(15, have["pf"].max() + 5) / 5) * 5)
        bot = int(np.floor(min(-20, have["pf"].min() - 5) / 5) * 5)
        half = int(np.ceil(max(15, have["pxc"].abs().max() + 4) / 5) * 5)
    S = 6.0
    W, H = 2 * half * S + 40, (top - bot) * S + 30
    xs = lambda v: W / 2 + 15 + v * S
    ys = lambda v: 8 + (top - v) * S
    g = []
    for v in range(bot, top + 1, 5):
        g.append(f'<line x1="30" y1="{ys(v):.1f}" x2="{W - 8}" y2="{ys(v):.1f}" stroke="#eef1f3" '
                 f'stroke-dasharray="{"" if v == 0 else "3 3"}"/><text x="25" y="{ys(v) + 3:.1f}" class="ax" '
                 f'text-anchor="end">{v}</text>')
    for v in range(-half, half + 1, 5):
        g.append(f'<text x="{xs(v):.1f}" y="{H - 6}" class="ax" text-anchor="middle">{v}</text>')
    h = PLATE_HALF
    g.append(f'<polygon points="{xs(-h):.1f},{ys(0):.1f} {xs(h):.1f},{ys(0):.1f} {xs(h):.1f},{ys(-h):.1f} '
             f'{xs(0):.1f},{ys(-17):.1f} {xs(-h):.1f},{ys(-h):.1f}" fill="#f6f8f9" stroke="#0e7c86" stroke-width="1.8"/>')
    g.append(f'<text x="{W - 10}" y="{ys(top) + 10:.1f}" class="ax" text-anchor="end">toward the pitcher ↑</text>')
    sg = _groups_of(have)
    gk = have["grp"].to_numpy(int) if sg else np.zeros(len(have), int)
    if display == "Heat Map":
        for k in sorted(set(gk)):
            hk = have[gk == k]
            px, py = xs(hk["pxc"].to_numpy(float)), ys(hk["pf"].to_numpy(float))
            vals = hk[NUMERIC[color_by][0]].to_numpy(float) if color_by in NUMERIC else None
            g.insert(1, f'<g class="g{k}">' + "".join(heat_layer(px, py, vals, W, H, color_by, cell=6.0, sigma=2.2 * S,
                     unit=NUMERIC.get(color_by, ("",) * 4)[3], tint=sg[k][1] if sg and vals is None else None)) + "</g>")
    else:
        order = np.argsort(have["bip"].to_numpy())                     # fouls first, balls in play on top
        for i in order:
            r = next(have.iloc[[i]].itertuples())
            hollow, k = not r.bip, int(gk[i])
            col = sg[k][1] if color_by == "Swing Group" and sg else point_color(r, color_by)
            x, y = float(np.clip(xs(r.pxc), 8, W - 8)), float(np.clip(ys(r.pf), 8, H - 18))
            tip = ((f"{chr(65 + k)}: " if sg else "") + f"{r.date} · {r.ptype} · {r.result}"
                   + (f" · {r.ev:.0f} mph, {r.la:.0f}°" if not pd.isna(r.ev) else "")
                   + f" · {r.pf:+.1f}\" front, {r.pxc:+.1f}\" side")
            style = (f'fill="#fff" fill-opacity=".6" stroke="{col}" stroke-width="1.6"' if hollow
                     else f'fill="{col}" fill-opacity=".88" stroke="#33404c" stroke-width=".6"')
            g.append(_marker(k, x, y, 4 if hollow else 5, style, tip))
    if summary:
        groups = [(grp, gg, GROUP_COLOR.get(grp, "#33404c"), "") for grp, gg in have.groupby("pgroup")]
    elif sg:
        groups = [(k, have[gk == k], sg[k][1], f' class="g{k}"') for k in range(len(sg))]
    else:
        groups = [("All", have, "#ffffff", "")]
    for grp, gg, col, cls in groups:
        if len(gg) < 3:
            continue
        x, y = xs(gg["pxc"].mean()), ys(gg["pf"].mean())
        g.append(f'<g{cls} stroke="#16202b" stroke-width="3"><line x1="{x - 8:.1f}" y1="{y:.1f}" x2="{x + 8:.1f}" '
                 f'y2="{y:.1f}"/><line x1="{x:.1f}" y1="{y - 8:.1f}" x2="{x:.1f}" y2="{y + 8:.1f}"/></g>'
                 f'<g{cls} stroke="{col}" stroke-width="1.6"><line x1="{x - 7:.1f}" y1="{y:.1f}" x2="{x + 7:.1f}" '
                 f'y2="{y:.1f}"/><line x1="{x:.1f}" y1="{y - 7:.1f}" x2="{x:.1f}" y2="{y + 7:.1f}"/></g>')
    how = ("<b>*</b> front to back estimated -- not on Savant's stance board yet. " if estimated else "") + \
        "Front to back = MLB's contact point vs the plate (Statcast's contact from his hips − his stance depth); " \
        "side to side = where the ball was. "
    key = legend_html(have, color_by, display) + ("" if display == "Heat Map" else " · filled = in play, hollow = foul")
    legend = (" ".join(f'<span style="color:{c}">✚</span> {k} avg' for k, c in GROUP_COLOR.items()) if summary
              else " ".join(f'<span style="color:{c}">✚</span> {chr(65 + k)} avg' for k, (n, c) in enumerate(sg))
              if sg else "✚ his average here")
    cid = f"poc{abs(hash((title, len(have), color_by, display))) % 10 ** 8}"
    chart = (f'<div class="card" id="{cid}" style="flex:1 1 380px;max-width:{"560px" if summary else "100%"}"><div class="t">{escape(title)}{"*" if estimated else ""} '
             f'<span class="faint">· {len(have)} contacts · inches · catcher\'s view</span></div>{_chips(cid, sg)}'
             f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;max-height:{"none" if summary else "400px"}">'
             f'{"".join(g)}</svg>'
             f'<div class="leg">{key}<br>{legend}. {"Hover a dot for the pitch. " if display == "Points" else ""}'
             f'<span class="faint">{how}</span></div></div>')
    return CSS + f'<div class="bp">{chart}{_summary(have) if summary else ""}</div>'


def _summary(have: pd.DataFrame) -> str:
    rows = [("All", have)] + [(k, have[have["pgroup"] == k]) for k in ("Fastball", "Breaking", "Offspeed")] + \
        [(k, have[have["zone_io"] == k]) for k in ("In Zone", "Out of Zone")] + \
        [(k, have[have["p_throws"] == h]) for k, h in (("vs RHP", "R"), ("vs LHP", "L"))]
    body = ""
    for name, g in rows:
        if g.empty:
            continue
        ip = g[g["bip"]]
        body += (f'<tr><td>{name}</td><td>{len(g)}</td><td>{g["pf"].mean():+.1f}</td><td>{g["pf"].std():.1f}</td>'
                 f'<td>{g["pxc"].mean():+.1f}</td><td>{"–" if ip.empty else f"{ip.ev.mean():.1f}"}</td></tr>')
    return (f'<div class="card"><div class="t">Where he meets it</div><table><tr><th>Group</th><th>#</th>'
            f'<th title="inches in front of the plate">Front</th><th title="spread of his contact depth -- how '
            f'much his timing point moves">± Depth</th><th title="inches from the middle of the plate, catcher\'s '
            f'view">Side</th><th>EV (in play)</th></tr>{body}</table></div>')


# ----------------------------------------------------------------------------- inline 3D path (same engine as the AAR)
_S3D_CSS = """
*{box-sizing:border-box} body{margin:0;font-family:system-ui,sans-serif;color:#16202b;background:#fff}
#wrap{border:1px solid #e6eaed;border-radius:10px;overflow:hidden}
#hd{display:flex;align-items:baseline;gap:10px;padding:9px 12px;border-bottom:1px solid #e6eaed}
#hd b{font:700 15px system-ui} #hd span{color:#6a7681;font-size:12.5px}
#stage{position:relative;width:100%;aspect-ratio:1/1;background:#f4f7f8;cursor:grab} #stage:active{cursor:grabbing}
#stage canvas{display:block} #stage .msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  color:#6a7681;font:13px system-ui;text-align:center;padding:30px}
#cams{position:absolute;left:10px;top:10px;display:flex;gap:4px;flex-wrap:wrap;z-index:2}
#cams button,#ctl button{font:600 11.5px system-ui;border:1px solid #e6eaed;background:#fff;border-radius:6px;
  padding:4px 8px;cursor:pointer} #cams button.on{background:#4a8b96;color:#fff;border-color:#4a8b96}
#ctl{display:flex;align-items:center;gap:8px;padding:8px 12px;border-top:1px solid #e6eaed;flex-wrap:wrap}
#ctl select{font:12px system-ui} #ctl .scw{flex:1;position:relative;display:flex;min-width:120px}
#ctl input[type=range]{flex:1}
#ctl .tick{position:absolute;top:-6px;width:2px;height:8px;background:#c0392b;transform:translateX(-1px);pointer-events:none}
#ctl .tick.bp{background:#7b5cc4} #ctl #s3bp{color:#5b3fa8}
#ctl .clk{font:12px ui-monospace,monospace;color:#6a7681;min-width:120px;text-align:right}
#lg{font-size:11.5px;color:#6a7681;line-height:1.55;padding:7px 12px;border-top:1px solid #e6eaed}
#grp{position:absolute;right:10px;top:10px;display:flex;gap:4px;z-index:2}
#grp button{font:700 12px system-ui;border:1px solid #d9dee2;background:#fff;border-radius:6px;padding:4px 9px;
  cursor:pointer;display:inline-flex;align-items:center;gap:5px;color:#9aa4ad}
#grp button i{display:inline-block;width:10px;height:10px;border-radius:50%;background:#d9dee2}
#grp button.on{color:#16202b;border-color:var(--c)} #grp button.on i{background:var(--c)}
#lg i{font-style:normal;font-weight:800}
"""


def path3d_html(scene, title: str, sub: str) -> str:
    """Self-contained page: the 3D view, square, filling its frame's width. scene = one scene, or a list of swing-
    group scenes (each with name + color) -> all drawn together, with a toggle chip per group on this chart."""
    import json

    from ui.aar import S3D_CORE, THREE_JS
    if isinstance(scene, list):
        scs = [x for x in scene if x]
        scene = {**scs[0], "extra": [{k: x[k] for k in ("pitch", "arc", "contact", "O", "bp", "color", "name")}
                                     for x in scs[1:]]}
    data = json.dumps(scene).replace("</", "<\\/")
    ctl = ('<div id="ctl"><button id="s3p">❚❚</button><select id="s3sp"><option value="4">1/4 speed</option>'
           '<option value="10" selected>1/10</option><option value="25">1/25</option><option value="60">1/60</option>'
           '</select><span class="scw"><input id="s3sc" type="range" min="0" max="1000" value="0"><span id="s3tickbp" '
           'class="tick bp" title="Bat parallel (inferred)"></span><span id="s3tick" class="tick" title="Contact">'
           '</span></span><button id="s3bp">Bat parallel</button><button id="s3k">Contact</button>'
           '<span class="clk" id="s3c"></span></div>')
    multi = bool(scene.get("name"))
    legend = ('<div id="lg">Each colour = one swing group (A, B, … -- toggle them top right). Solid tube = his average '
              'bat path, last 40 ms before contact (measured); dashed + dot = back to bat parallel + follow-through '
              '(inferred); thin line = that group\'s modeled pitch; big dot = his average contact point · drag to '
              'orbit, scroll to zoom</div>') if multi else \
             ('<div id="lg"><i style="color:#e07b24">━</i> modeled pitch &nbsp; <i style="color:#2f5fb3">━</i> his '
              'average bat path, last 40 ms before contact (from his measured swings) &nbsp; <i style="color:#7b5cc4">┅ ●'
              '</i> back to bat parallel + follow-through, inferred from his tilt, attack angle + direction &nbsp; '
              '<i style="color:#c0392b">●</i> his average contact point · drag to orbit, scroll to zoom</div>')
    js = """
function fit() {
  const h = document.documentElement.scrollHeight + 4;
  window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setFrameHeight', height: h}, '*');
}
const SC = JSON.parse(document.getElementById('s3d-data').textContent);
const GRP = SC.name ? [{name: SC.name, color: SC.color}].concat(SC.extra || []) : [];
const ON = GRP.map(() => true);
const chips = document.getElementById('grp');
GRP.forEach((g, k) => {
  const b = document.createElement('button'); b.className = 'on'; b.style.setProperty('--c', g.color);
  b.innerHTML = `<i></i>${String.fromCharCode(65 + k)}`; b.title = g.name;
  b.onclick = () => { ON[k] = !ON[k]; b.classList.toggle('on', ON[k]); if (R3) R3.setOn(k, ON[k]); };
  chips.appendChild(b);
});
let lastW = 0;
function start() {
  const w = stage.clientWidth; if (Math.abs(w - lastW) < 8) return; lastW = w;
  document.getElementById('s3msg').style.display = 'none';
  if (R3) R3.stop(); R3 = build(SC); ON.forEach((on, k) => R3.setOn(k, on)); fit();
}
loadThree(start);
window.addEventListener('resize', () => { if (window.THREE) start(); });
window.addEventListener('load', fit); fit();
"""
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{_S3D_CSS}</style></head><body>'
            f'<div id="wrap"><div id="hd"><b>{escape(title)}</b><span>{escape(sub)}</span></div>'
            f'<div id="stage"><div id="cams"></div><div id="grp"></div><div class="msg" id="s3msg">Loading 3D…</div></div>'
            f'{ctl}{legend}</div>'
            f'<script type="application/json" id="s3d-data">{data}</script>'
            f'<script>const THREE_SRC = "{THREE_JS}";{S3D_CORE}{js}</script></body></html>')


def model_card_html(sc: dict, hitter_side: str) -> str:
    """What went into the view: the modeled pitch + his averages (measured) + geometry (inferred)."""
    m, r = sc["model"], sc["read"]
    f = lambda v, d=1: "–" if v is None else f"{v:.{d}f}"
    side = "pull" if (m["attack_direction"] < 0) else "oppo"
    rows = [
        ("§Pitch (modeled)", ""),
        (f'{m["hand"]}HP {m["group"].lower()}', f'{m["ptype"]} · {f(m["velo"])} mph'),
        ("Shape", f'{f(m["ivb"], 0)}″ IVB · {f(m["hb"], 0)}″ HB arm side'),
        ("Type", m["shape"]), ("Velo", m["velo_src"]),
        ("Coming down at", f'{f(-r["vaa"])}°'),
        (f'§His average swing ({m["n"]} swings)', m["level"]),
        ("Bat speed", f'{f(m["bat_speed"])} mph'), ("Swing length", f'{f(m["swing_length"])} ft'),
        ("Attack angle", f'{f(m["attack_angle"])}°'), ("Attack direction", f'{f(m["attack_direction"])}° {side}'),
        ("Swing tilt", f'{f(m["swing_path_tilt"])}°'),
        ("Swing vs pitch plane", f'{r["plane_gap"]:+.1f}° (0 = matched)'),
        (f'§His average contact ({m["n_contact"]} contacts)', m["contact_level"]),
        ("Out front of the plate", f'{f(r["front"])}″'),
        ("§Where the ball is there (modeled pitch)", ""),
        ("Side (+ = 1B side)", f'{f(r["side_in"])}″'), ("Height", f'{f(r["height"])}″'),
        ("§Geometry (inferred)", ""),
        ("Vertical bat angle at contact", f'{f(r["vba"])}°'),
        ("Bat parallel", f'{f(r["bp_in"], 0)}″ of path before contact' if r.get("bp_in") else "past his swing length"),
    ]
    body = ""
    for k, v in rows:
        if k.startswith("§"):
            body += f'<tr class="h"><td colspan="2">{escape(k[1:])}{"" if not v else f" <span>· {escape(v)}</span>"}</td></tr>'
        else:
            body += f'<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>'
    css = ("<style>.mc{font:13px system-ui;color:#16202b;border:1px solid #e6eaed;border-radius:10px;padding:6px 12px}"
           ".mc table{width:100%;border-collapse:collapse}.mc td{padding:4px 0;border-bottom:1px solid #f0f2f4}"
           ".mc td:last-child{text-align:right;font-family:ui-monospace,Menlo,monospace}"
           ".mc tr.h td{font:700 11px system-ui;color:#6a7681;text-transform:uppercase;letter-spacing:.04em;"
           "padding-top:11px;text-align:left;border-bottom:1px solid #d9dee2}.mc tr.h span{text-transform:none;"
           "font-weight:400;letter-spacing:0}</style>")
    return css + f'<div class="mc"><table>{body}</table></div>'


def spray_html(d: pd.DataFrame, title: str = "Spray chart", park: dict | None = None, color_by: str = "Exit Velo",
               display: str = "Points") -> str:
    """Balls in play on his home park (traced outline when it matches the venue, else its published distances;
    a generic park if neither is known), coloured by exit velo. d: hc_x, hc_y, ev, la, result, date, ptype."""
    from ui.aar import DEFAULT_DIMS, LINE_ANGLES, _around_center, _by_spray, _poly
    park = park or {}
    b = d.dropna(subset=["hc_x", "hc_y"]) if {"hc_x", "hc_y"} <= set(d.columns) else \
        pd.DataFrame(columns=["hc_x", "hc_y", "ev", "la", "result", "date", "ptype"])
    W, H = 440, 330
    dims = park.get("dims") or DEFAULT_DIMS
    segs = (park.get("outline") or {}).get("segments") or {}
    if segs:
        wx_, wy_ = segs["outfield_outer"]
        deep, wide = float(np.max(wy_)), float(np.max(np.abs(wx_)))
    else:
        deep = max(dims.values())
        wide = deep * 0.75
    S = min((H - 30) / (deep + 14), (W / 2 - 10) / (wide + 6))
    hx, hy = W / 2, H - 12
    pt = lambda fx, fy: (hx + fx * S, hy - fy * S)
    g = []
    if segs:
        fx, fy = _by_spray(*segs["outfield_outer"])
        fence_poly = f'{hx:.1f},{hy:.1f} {_poly(pt(a, c) for a, c in zip(fx, fy))}'
        g.append(f'<polygon points="{hx:.1f},{hy:.1f} {_poly(pt(a, c) for a, c in zip(fx, fy))}" fill="#eef5ef" '
                 f'stroke="#5f8f96" stroke-width="1.6" stroke-linejoin="round"/>')
        if "infield_outer" in segs:
            ix, iy = _by_spray(*segs["infield_outer"])
            g.append(f'<polygon points="{hx:.1f},{hy:.1f} {_poly(pt(a, c) for a, c in zip(ix, iy))}" '
                     f'fill="#f4ede2" stroke="#c9b48f" stroke-width=".8"/>')
        if "infield_inner" in segs:
            ix, iy = _around_center(*segs["infield_inner"])
            g.append(f'<polygon points="{_poly(pt(a, c) for a, c in zip(ix, iy))}" fill="#e8f1e8" '
                     f'stroke="#c9b48f" stroke-width=".8"/>')
        for a_, c_ in ((fx[0], fy[0]), (fx[-1], fy[-1])):
            x2, y2 = pt(a_, c_)
            g.append(f'<line x1="{hx:.1f}" y1="{hy:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#b9c7cb" stroke-width=".9"/>')
        dist = np.hypot(fx, fy)
        spots = {"leftLine": (fx[0], fy[0], "start", 5), "rightLine": (fx[-1], fy[-1], "end", -5),
                 "center": (fx[int(np.argmax(dist))], fy[int(np.argmax(dist))], "middle", 0)}
    else:
        pts = [pt(dims[k] * math.sin(math.radians(a)), dims[k] * math.cos(math.radians(a)))
               for k, a in LINE_ANGLES.items() if k in dims]
        fence_poly = f'{hx:.1f},{hy:.1f} {_poly(pts)}'
        g.append(f'<polygon points="{hx:.1f},{hy:.1f} {_poly(pts)}" fill="#eef5ef" stroke="#7fa3a9" '
                 f'stroke-width="1.3" stroke-dasharray="5 3"/>')
        g.append(f'<polygon points="{hx},{hy} {_poly([pt(63.6, 63.6), pt(0, 127.3), pt(-63.6, 63.6)])}" '
                 f'fill="#f4ede2" stroke="#c9b48f"/>')
        spots = {k: (dims[k] * math.sin(math.radians(LINE_ANGLES[k])), dims[k] * math.cos(math.radians(LINE_ANGLES[k])),
                     an, dx) for k, an, dx in (("leftLine", "start", 5), ("center", "middle", 0),
                                               ("rightLine", "end", -5)) if k in dims}
    for k, (ax_, ay_, anchor, dx) in spots.items():
        if k in dims:
            x, y = pt(ax_, ay_)
            g.append(f'<text x="{x + dx:.1f}" y="{y - 5:.1f}" class="ax" text-anchor="{anchor}">{dims[k]:.0f}</text>')
    fx_all, fy_all = (b["hc_x"].to_numpy(float) - 125.42) * 2.5, (198.27 - b["hc_y"].to_numpy(float)) * 2.5
    sg = _groups_of(b)
    gk = b["grp"].to_numpy(int) if sg else np.zeros(len(b), int)
    cid = f"spr{abs(hash((title, len(b), color_by, display))) % 10 ** 8}"
    if display == "Heat Map" and not b.empty:
        hxp, hyp = hx + fx_all * S, hy - fy_all * S
        g.append(f'<clipPath id="{cid}c"><polygon points="{fence_poly}"/></clipPath>')
        for k in sorted(set(gk)):
            m = gk == k
            vals = b[NUMERIC[color_by][0]].to_numpy(float)[m] if color_by in NUMERIC else None
            g.append(f'<g class="g{k}" clip-path="url(#{cid}c)">'
                     + "".join(heat_layer(hxp[m], hyp[m], vals, W, H, color_by, cell=5.0, sigma=22 * S,
                                          unit=NUMERIC.get(color_by, ("",) * 4)[3],
                                          tint=sg[k][1] if sg and vals is None else None)) + "</g>")
    else:
        for r, x, y, k in zip(b.itertuples(), fx_all, fy_all, gk):
            px, py = pt(x, y)
            col = sg[k][1] if color_by == "Swing Group" and sg else point_color(r, color_by)
            tip = ((f"{chr(65 + k)}: " if sg else "") + f"{r.date} · {r.ptype} · {r.result}"
                   + (f" · {r.ev:.0f} mph, {r.la:.0f}°" if not pd.isna(r.ev) else "") + f" · {math.hypot(x, y):.0f} ft")
            g.append(_marker(int(k), float(np.clip(px, 6, W - 6)), float(np.clip(py, 6, H - 6)), 5,
                             f'fill="{col}" fill-opacity=".9" stroke="#33404c" stroke-width=".6"', tip))
    where = park.get("venue") or ""
    approx = "" if segs or not park.get("dims") else " (approximate fence)"
    sub = f' <span class="faint">· {escape(where)}{approx}</span>' if where else \
        ' <span class="faint">· generic park (home park unknown)</span>'
    note = ("no balls in play here" if b.empty else
            f'{len(b)} balls in play · avg EV {b["ev"].mean():.1f} · hard-hit {(b["ev"] >= 95).mean():.0%}')
    note += " · all games, drawn on his home park" + (" · hover a dot" if display == "Points" else "")
    note = legend_html(b, color_by, display) + "<br>" + note
    return CSS + (f'<div class="bp"><div class="card" id="{cid}" style="flex:1" title="{escape((park.get("outline") or {}).get("note", ""))}">'
                  f'<div class="t">{escape(title)}{sub}</div>{_chips(cid, sg)}'
                  f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;max-height:330px">{"".join(g)}</svg>'
                  f'<div class="leg">{note}</div></div></div>')


def framed(html: str) -> str:
    """A chart as its own small page (Streamlit's st.html strips SVG) that tells Streamlit its height."""
    js = ("<script>function fit(){window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',"
          "height:document.documentElement.scrollHeight+2},'*');}window.addEventListener('load',fit);"
          "window.addEventListener('resize',fit);new ResizeObserver(fit).observe(document.body);fit();</script>")
    return ('<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0;font-family:system-ui,'
            f'sans-serif">{html}{js}</body></html>')


def compare_html(scenes: list) -> str:
    """The swing groups side by side: what was averaged, the measured swing, the inferred geometry."""
    scs = [x for x in scenes if x]
    if not scs:
        return ""
    f = lambda v, d=1: "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"
    rows = [("Swings averaged", lambda s: f'{s["model"]["n"]} <span class="faint">({escape(s["model"]["level"])})</span>'),
            ("Modeled pitch", lambda s: f'{s["model"]["hand"]}HP {s["model"]["ptype"]} {f(s["model"]["velo"])} mph'),
            ("Bat speed", lambda s: f'{f(s["model"]["bat_speed"])} mph'),
            ("Swing length", lambda s: f'{f(s["model"]["swing_length"])} ft'),
            ("Attack angle", lambda s: f'{f(s["model"]["attack_angle"])}°'),
            ("Attack direction", lambda s: f'{f(s["model"]["attack_direction"])}° '
                                           f'{"pull" if s["model"]["attack_direction"] < 0 else "oppo"}'),
            ("Swing tilt", lambda s: f'{f(s["model"]["swing_path_tilt"])}°'),
            ("Swing vs pitch plane", lambda s: f'{s["read"]["plane_gap"]:+.1f}°'),
            ("Contact out front", lambda s: f'{f(s["read"]["front"])}″'),
            ("Contact height", lambda s: f'{f(s["read"]["height"])}″'),
            ("Vertical bat angle (inferred)", lambda s: f'{f(s["read"]["vba"])}°'),
            ("Bat parallel before contact (inferred)", lambda s: f'{f(s["read"].get("bp_in"), 0)}″')]
    head = "".join(f'<th style="border-bottom:3px solid {s["color"]}" title="{escape(s["name"])}">'
                   f'{chr(65 + s["gid"])}<div class="faint" style="font-weight:400;font-size:11px">{escape(s["name"])}</div></th>'
                   for s in scs)
    body = "".join(f'<tr><td>{k}</td>' + "".join(f"<td>{fn(s)}</td>" for s in scs) + "</tr>" for k, fn in rows)
    return CSS + (f'<div class="bp"><div class="card" style="flex:1"><div class="t">Swing groups side by side</div>'
                  f'<table style="width:100%"><tr><th></th>{head}</tr>{body}</table></div></div>')
