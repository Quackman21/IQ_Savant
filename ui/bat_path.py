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
#wrap{border:1px solid #e6eaed;border-radius:10px;overflow:hidden;max-width:720px;margin:0 auto}
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
        scene = {**scs[0], "extra": [{k: x.get(k) for k in ("pitch", "arc", "contact", "O", "bp", "color", "name",
                                                             "series", "norm", "read", "side", "hands")}
                                     for x in scs[1:]]}
    data = json.dumps(scene).replace("</", "<\\/")
    ctl = ('<div id="ctl"><button id="s3p">❚❚</button><select id="s3sp"><option value="4">1/4 speed</option>'
           '<option value="10" selected>1/10</option><option value="25">1/25</option><option value="60">1/60</option>'
           '</select><span class="scw"><input id="s3sc" type="range" min="0" max="1000" value="0"><span id="s3tickbp" '
           'class="tick bp" title="Bat parallel (inferred)"></span><span id="s3tick" class="tick" title="Contact">'
           '</span></span><button id="s3bp">Bat parallel</button><button id="s3k">Contact</button>'
           '<span class="clk" id="s3c"></span>' + graph_parts()[0] + '</div>')
    multi = bool(scene.get("name"))
    legend = ('<div id="lg">Each colour = one swing group (A, B, … -- toggle them top right). Solid tube = his average '
              'bat path, last 40 ms before contact (measured); solid line = how his path turns through the zone, '
              'measured from his own swings at different contact depths; dashed + dot = beyond what his swings cover '
              '(kept turning at his last measured rate, or the circle when there aren\'t enough swings); thin line = that group\'s modeled pitch; big dot = his average contact point · drag to '
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
  if (R3) R3.stop(); R3 = build(SC); ON.forEach((on, k) => R3.setOn(k, on));
  if (typeof paint === 'function' && trailMode !== 'group') paint();
  if (typeof setTab === 'function' && TAB !== 'Swing') setTab(TAB); fit();
}
loadThree(start);
window.addEventListener('resize', () => { if (window.THREE) start(); });
window.addEventListener('load', fit); fit();
"""
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{_S3D_CSS}{GRAPH_CSS}{LAYERS_CSS}</style></head><body>'
            f'<div id="wrap"><div id="hd"><b>{escape(title)}</b><span>{escape(sub)}</span><div id="tabs"></div></div>'
            f'<div id="stage"><div id="cams"></div><div id="grp"></div><div id="tl" hidden></div><div id="al" hidden></div>'
            f'<div class="msg" id="s3msg">Loading 3D…</div></div>'
            f'{ctl}{legend}<div id="lay" hidden></div></div>'
            f'<script type="application/json" id="s3d-data">{data}</script>'
            f'<script>const THREE_SRC = "{THREE_JS}";{S3D_CORE}{js}{COMMON_JS}{TRAIL_JS}{LAYERS_JS}</script></body></html>')


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


def _hba0(s: dict) -> float:
    from core.bat_path import path_angles
    p = path_angles(s)
    return float(p.loc[(p["t"] >= 0).idxmax(), "hba"])


def _window(s: dict, mode: str) -> str:
    from core.bat_path import on_plane
    w = on_plane(s, mode)
    if not w["on"]:
        return f'not at contact ({w["gap"]:+.0f}°)'
    where = "early" if w["pct"] < 0.35 else "late" if w["pct"] > 0.65 else "centered"
    return (f'{w["ms"]:.0f} ms · {w["y0"]:.0f}″→{w["y1"]:.0f}″ · contact {w["before"]:.0f} ms in '
            f'<span class="faint">({where})</span>')


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
            ("Horizontal bat angle at contact (inferred)", lambda s: f'{_hba0(s):+.0f}° '
                                                                     f'{"ahead of" if _hba0(s) > 0 else "behind"} hands'),
            ("On plane, vertical (±5° of the pitch)", lambda s: _window(s, "gap_v")),
            ("On line, horizontal (±5° of the pitch)", lambda s: _window(s, "gap_h")),
            ("Bat parallel before contact (inferred)", lambda s: f'{f(s["read"].get("bp_in"), 0)}″')]
    head = "".join(f'<th style="border-bottom:3px solid {s["color"]}" title="{escape(s["name"])}">'
                   f'{chr(65 + s["gid"])}<div class="faint" style="font-weight:400;font-size:11px">{escape(s["name"])}</div></th>'
                   for s in scs)
    body = "".join(f'<tr><td>{k}</td>' + "".join(f"<td>{fn(s)}</td>" for s in scs) + "</tr>" for k, fn in rows)
    return CSS + (f'<div class="bp"><div class="card" style="flex:1"><div class="t">Swing groups side by side</div>'
                  f'<table style="width:100%"><tr><th></th>{head}</tr>{body}</table></div></div>')


# ----------------------------------------------------------------------------- graphs + trail colours in the 3D window
GRAPH_CSS = """
#tl{position:absolute;left:10px;top:44px;z-index:2;background:rgba(255,255,255,.93);border:1px solid #e6eaed;
  border-radius:8px;padding:6px 9px;font:11px system-ui;color:#33404c;max-width:280px}
#tl[hidden]{display:none} #tl .bar{height:9px;border-radius:4px;margin:4px 0 2px}
#tl .lab{display:flex;justify-content:space-between;color:#6a7681;font-size:10px}
#ctl select#s3trail{max-width:230px}
#tv{display:inline-flex;border:1px solid #d9dee2;border-radius:6px;overflow:hidden}
#s3arrow.on{background:#4a8b96;color:#fff;border-color:#4a8b96}
#al{position:absolute;right:10px;bottom:10px;z-index:2;background:rgba(255,255,255,.93);border:1px solid #e6eaed;
  border-radius:8px;padding:6px 9px;font:11.5px system-ui;color:#33404c;line-height:1.5}
#al[hidden]{display:none}
#tl,#al{cursor:grab;touch-action:none;user-select:none} #tl.fdrag,#al.fdrag{cursor:grabbing;opacity:.85}
.ftb{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:-2px -3px 3px;color:#9aa5ae;font-size:10px}
.fgrip{letter-spacing:.5px} .fmin{border:1px solid #d9dee2;background:#fff;border-radius:5px;width:20px;height:18px;
  line-height:14px;padding:0;font:700 12px system-ui;color:#33404c;cursor:pointer}
.fmin-on .fb{display:none} .fmin-on .ftb{margin-bottom:-2px}
#tv button{border:none!important;border-radius:0!important;padding:4px 9px} #tv button.on{background:#4a8b96;color:#fff}
"""

# shared by the 3D window (trail colours) and the chart under it
COMMON_JS = r"""
// floating boxes over the 3D view: drag anywhere on the box to move it, "–" to tuck it away; remembered per viewer
function fstore(id, v) { try { if (v === undefined) return JSON.parse(localStorage.getItem('bp_float_' + id) || 'null');
  localStorage.setItem('bp_float_' + id, JSON.stringify(v)); } catch (e) { return null; } }
function fbody(el, title) {
  if (el._fb) return el._fb;
  el.innerHTML = '';
  const tb = document.createElement('div'), body = document.createElement('div'), btn = document.createElement('button');
  tb.className = 'ftb'; tb.innerHTML = `<span class="fgrip">⋮⋮ ${title}</span>`; btn.className = 'fmin'; btn.title = 'Minimize';
  tb.appendChild(btn); body.className = 'fb'; el.append(tb, body); el._fb = body;
  const st = fstore(el.id) || {};
  const setMin = m => { el.classList.toggle('fmin-on', m); btn.textContent = m ? '+' : '–'; btn.title = m ? 'Show' : 'Minimize'; };
  setMin(!!st.min);
  if (st.x !== undefined) { el.style.left = st.x + 'px'; el.style.top = st.y + 'px'; el.style.right = 'auto'; el.style.bottom = 'auto'; }
  btn.onclick = e => { e.stopPropagation(); const s = fstore(el.id) || {}; s.min = !el.classList.contains('fmin-on'); setMin(s.min); fstore(el.id, s); };
  el.addEventListener('pointerdown', e => {
    if (e.target === btn) return;
    const host = el.offsetParent || el.parentElement, r0 = el.getBoundingClientRect(), h0 = host.getBoundingClientRect();
    const ox = e.clientX - r0.left, oy = e.clientY - r0.top;
    el.setPointerCapture(e.pointerId); el.classList.add('fdrag'); e.preventDefault(); e.stopPropagation();
    const mv = ev => { const x = Math.max(0, Math.min(h0.width - 40, ev.clientX - h0.left - ox)),
                             y = Math.max(0, Math.min(h0.height - 24, ev.clientY - h0.top - oy));
      el.style.left = x + 'px'; el.style.top = y + 'px'; el.style.right = 'auto'; el.style.bottom = 'auto'; };
    const up = ev => { el.removeEventListener('pointermove', mv); el.removeEventListener('pointerup', up); el.classList.remove('fdrag');
      const s = fstore(el.id) || {}; s.x = parseFloat(el.style.left); s.y = parseFloat(el.style.top); fstore(el.id, s); };
    el.addEventListener('pointermove', mv); el.addEventListener('pointerup', up);
  });
  return body;
}

const STOPS = [[-1, [215, 38, 61]], [-0.66, [244, 96, 54]], [-0.33, [249, 200, 14]], [0, [46, 157, 91]],
               [0.33, [27, 152, 224]], [0.66, [61, 52, 139]], [1, [123, 44, 191]]];
function match(v, lim = 15) {          // green = on the pitch's plane -> yellow -> red = off it (by how far, either way)
  if (v === null || v === undefined || isNaN(v)) return [0.62, 0.64, 0.66];
  const t = Math.min(1, Math.abs(v) / lim), G = [46, 157, 91], Y = [242, 192, 30], R = [214, 51, 40];
  const [a, b, f] = t < 0.5 ? [G, Y, t / 0.5] : [Y, R, (t - 0.5) / 0.5];
  return a.map((c, j) => (c + (b[j] - c) * f) / 255);
}
function rainbow(v, lim) {
  if (v === null || v === undefined || isNaN(v)) return [0.62, 0.64, 0.66];
  const t = Math.max(-1, Math.min(1, v / lim));
  for (let i = 1; i < STOPS.length; i++) if (t <= STOPS[i][0]) {
    const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i], f = (t - t0) / (t1 - t0);
    return c0.map((c, j) => (c + (c1[j] - c) * f) / 255);
  }
  return STOPS[STOPS.length - 1][1].map(c => c / 255);
}
function pathInfo(g) {                 // along the drawn path: angles vs the pitch (+ = steeper / oppo of its line)
  const a = g.arc, out = [], side = g.side === 'L' ? -1 : 1, wrap = v => ((v + 540) % 360) - 180;
  for (let i = 0; i < a.length; i++) {
    const p = a[Math.max(0, i - 1)], n = a[Math.min(a.length - 1, i + 1)];
    const dx = n[0] - p[0], dy = n[1] - p[1], dz = n[2] - p[2];
    const pv = Math.atan2(dz, Math.hypot(dx, dy)) * 180 / Math.PI, ph = Math.atan2(dx, dy) * 180 / Math.PI;
    const hv = g.hands && g.hands.length === a.length ? g.hands[i] : [g.O[0] - a[i][0], g.O[1] - a[i][1], g.O[2] - a[i][2]];
    const hx = hv[0], hy = hv[1], hz = hv[2];
    out.push({t: a[i][4] * 1000, flag: a[i][3], y: a[i][1], gap_v: pv + g.read.vaa, gap_h: wrap(side * (ph + g.read.haa)),
              attack_angle: pv, attack_direction: wrap(side * ph),
              vba: -Math.asin(hz / Math.hypot(hx, hy, hz)) * 180 / Math.PI, hba: -Math.atan2(hy, Math.abs(hx)) * 180 / Math.PI});
  }
  return out;
}
function onPlane(g, mode) {            // the stretch around contact within 5 degrees of the pitch
  const info = pathInfo(g), a = g.arc;
  let i = a.findIndex(q => q[4] >= 0); if (i < 0) i = a.length - 1;
  if (Math.abs(info[i][mode]) > 5) return {on: false, gap: info[i][mode]};
  let s = i, e = i;
  while (s > 0 && Math.abs(info[s - 1][mode]) <= 5) s--;
  while (e < a.length - 1 && Math.abs(info[e + 1][mode]) <= 5) e++;
  let len = 0; for (let j = s + 1; j <= e; j++) len += Math.hypot(a[j][0] - a[j - 1][0], a[j][1] - a[j - 1][1], a[j][2] - a[j - 1][2]);
  const t0 = a[s][4] * 1000, t1 = a[e][4] * 1000;
  return {on: true, len, ms: t1 - t0, y0: a[s][1], y1: a[e][1], before: -t0, pct: -t0 / Math.max(1e-6, t1 - t0)};
}
const PLANE = {gap_v: ['Vertical Plane Match (bat path vs the pitch\'s VAA)', 'flatter than the pitch', 'steeper than the pitch'],
               gap_h: ['Horizontal Plane Match (bat path vs the pitch\'s HAA)', 'pull side of its line', 'oppo side of its line']};
"""

TRAIL_JS = r"""
const GSD = [SC].concat(SC.extra || []);
let trailMode = 'group', rainbowBy = 'gap_v';
function paint() {
  if (!R3 || !R3.GR) return;
  const mode = trailMode;
  R3.GR.forEach((G, k) => {
    G.objs.filter(o => o.userData.paint).forEach(o => { R3.scene.remove(o); o.geometry.dispose(); });
    G.objs = G.objs.filter(o => !o.userData.paint);
    G.objs.forEach(o => {
      const orig = (o.type === 'Mesh' && o.geometry.type === 'TubeGeometry') || (o.type === 'Line' && o.material.type === 'LineDashedMaterial');
      if (orig) { o.userData.off = mode !== 'group'; o.visible = G.on && !o.userData.off; }
    });
    if (mode === 'group') return;
    const g = GSD[k], cols = pathInfo(g).map(q => match(q[mode])), a = g.arc;
    const add = (o) => { o.userData.paint = true; o.visible = G.on; R3.scene.add(o); G.objs.push(o); };
    const meas = a.map((q, i) => [q, cols[i]]).filter(x => x[0][3] === 1);
    if (meas.length > 1) {
      const SEG = 60, RAD = 8, geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(meas.map(x => V(x[0]))), SEG, 0.75, RAD);
      const c = [];
      for (let i = 0; i <= SEG; i++) { const m = meas[Math.round(i / SEG * (meas.length - 1))][1]; for (let r = 0; r <= RAD; r++) c.push(...m); }
      geo.setAttribute('color', new THREE.Float32BufferAttribute(c, 3));
      add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({vertexColors: true})));
    }
    const all = a.map((q, i) => [q, cols[i]]);
    const i0 = all.findIndex(x => x[0][3] === 1), i1 = all.length - 1 - [...all].reverse().findIndex(x => x[0][3] === 1);
    for (const part of [all.slice(0, i0 + 1), all.slice(i1)]) {        // inferred: lead-in + follow-through past contact
      if (part.length < 2) continue;
      const SEG = Math.max(8, part.length * 2), RAD = 8;
      const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(part.map(x => V(x[0]))), SEG, 0.55, RAD);
      const c = [];
      for (let i = 0; i <= SEG; i++) { const m = part[Math.round(i / SEG * (part.length - 1))][1]; for (let r = 0; r <= RAD; r++) c.push(...m); }
      geo.setAttribute('color', new THREE.Float32BufferAttribute(c, 3));
      add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({vertexColors: true, transparent: true, opacity: 0.5})));
    }
  });
  legendTrail();
}
function legendTrail() {
  const el = document.getElementById('tl');
  if (trailMode === 'group') { el.hidden = true; return; }
  const [name, lo, hi] = PLANE[trailMode];
  fbody(el, 'Legend').innerHTML = `<b>${name}</b><div class="bar" style="background:linear-gradient(90deg,#2e9d5b,#f2c01e,#d63328)"></div>
    <div class="lab"><span>on the pitch's ${trailMode === 'gap_v' ? 'plane' : 'line'}</span><span>7.5° off</span><span>15°+ off</span></div>` +
    GSD.map((g, k) => { const w = onPlane(g, trailMode), L = `<b style="color:${g.color || '#2f5fb3'}">${g.name ? String.fromCharCode(65 + k) : ''}</b>`;
      if (!w.on) return `<div style="margin-top:3px">${L} not within 5° at contact (${w.gap > 0 ? '+' : ''}${w.gap.toFixed(0)}°)</div>`;
      const where = w.pct < 0.35 ? 'early in it' : w.pct > 0.65 ? 'late in it' : 'in the middle of it';
      return `<div style="margin-top:3px">${L} within 5° for ${w.ms.toFixed(0)} ms (${w.len.toFixed(0)}″ of path, ${w.y0.toFixed(0)}″ → ${w.y1.toFixed(0)}″ vs the plate); contact ${w.before.toFixed(0)} ms in, ${where}</div>`; }).join('') +
    '<div style="margin-top:3px;color:#6a7681">Solid = measured last 40 ms · see-through = inferred (lead-in from bat parallel, follow-through past contact).</div>';
  el.hidden = false;
}
// ---- arrow on the barrel: where the sweet spot is heading, side to side (horizontal attack angle), live
let arrowOn = false, arrows = [], arrowsFor = null;
function arrowLoop() {
  requestAnimationFrame(arrowLoop);
  const box = document.getElementById('al');
  if (!R3 || !R3.scene) return;
  if (arrowsFor !== R3) {                             // the scene was rebuilt (resize): new arrows
    arrows = GSD.map(g => { const h = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), new THREE.Vector3(), 16, 0x2e9d5b, 5, 3.2);
      h.visible = false; R3.scene.add(h); return h; });
    arrowsFor = R3;
  }
  const t = R3.time(), lines = [];
  GSD.forEach((g, k) => {
    const h = arrows[k], on = arrowOn && (!R3.GR || R3.GR[k].on);
    const P = at(g.arc, t), P0 = at(g.arc, t - 0.002), P1 = at(g.arc, t + 0.002);
    if (!on || !P || !(P0 || P1)) { h.visible = false; return; }
    const A = P0 || P, B = P1 || P, dx = B[0] - A[0], dy = B[1] - A[1];
    if (Math.hypot(dx, dy) < 1e-6) { h.visible = false; return; }
    const side = g.side === 'L' ? -1 : 1, wrap = v => ((v + 540) % 360) - 180;
    const ph = Math.atan2(dx, dy) * 180 / Math.PI, had = wrap(side * ph), gap = wrap(side * (ph + g.read.haa));
    h.position.copy(V(P)); h.setDirection(new THREE.Vector3(dx, 0, -dy).normalize());
    const c = match(gap); h.setColor(new THREE.Color(c[0], c[1], c[2])); h.visible = true;
    lines.push(`<b style="color:${g.color || '#2f5fb3'}">${g.name ? String.fromCharCode(65 + k) : ''}</b> ➜ ${had >= 0 ? '+' : ''}${had.toFixed(0)}° ${had >= 0 ? 'oppo' : 'pull'} · ${Math.abs(gap).toFixed(0)}° off the pitch's line`);
  });
  box.hidden = !arrowOn || !lines.length;
  const html_ = `<b>Horizontal attack angle</b> (arrow on the barrel)<br>` + lines.join('<br>');
  const fb_ = fbody(box, 'Arrow');
  if (fb_._h !== html_) { fb_.innerHTML = html_; fb_._h = html_; }
}
document.getElementById('s3arrow').onclick = e => { arrowOn = !arrowOn; e.target.classList.toggle('on', arrowOn); };
arrowLoop();
function setView(v) {
  document.querySelectorAll('#tv button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  document.getElementById('s3trail').disabled = v === 'regular';
  trailMode = v === 'regular' ? 'group' : rainbowBy; paint();
}
document.querySelectorAll('#tv button').forEach(b => b.onclick = () => setView(b.dataset.v));
document.getElementById('s3trail').onchange = e => { rainbowBy = e.target.value; if (trailMode !== 'group') { trailMode = rainbowBy; paint(); } };
"""


def graph_parts() -> tuple[str, str]:
    """(Regular / Rainbow controls for the 3D window's bar, '')."""
    ctl = ('<span id="tv"><button data-v="regular" class="on">Regular</button><button data-v="rainbow">Rainbow'
           '</button></span><select id="s3trail" disabled title="What the rainbow shows">'
           '<option value="gap_v">Vertical Plane Match</option>'
           '<option value="gap_h">Horizontal Plane Match</option></select>'
           '<button id="s3arrow" title="Arrow on the barrel: which way it is travelling (horizontal attack angle) '
           'through the swing">Arrow</button>')
    return ctl, ""


# ----------------------------------------------------------------------------- the chart under the 3D view
CHART_JS = r"""
const D = JSON.parse(document.getElementById('cd').textContent);
const GS = D.groups, ALL = D.all, ON = GS.map(() => true);
const COLS = (GS[0] && GS[0].series) ? GS[0].series.cols : [];
const C = k => COLS.indexOf(k);
function rows(src, key, fast) {
  const ci = C(key), pi = C('pf'), bi = C('bat_speed');
  let pts = (src ? src.pts : []).filter(p => p[ci] !== null && p[pi] !== null);
  const fb = fast && src && src.n_fast === 0;
  if (fast && !fb) pts = pts.filter(p => p[bi] >= 75);
  return {pts, fb, ci, pi};
}
function kern(pts, ci, pi, x, bw = 3) {                 // smoothed mean + SD at depth x (kernel, bw inches)
  let sw = 0, sv = 0, sq = 0;
  for (const p of pts) { const w = Math.exp(-0.5 * ((x - p[pi]) / bw) ** 2); sw += w; sv += w * p[ci]; sq += w * p[ci] * p[ci]; }
  if (sw < 1.5) return null; const m = sv / sw; return {m, s: Math.sqrt(Math.max(0, sq / sw - m * m)), w: sw};
}
function leagueAt(g, key, x) {
  const N = g.norm || {}; if (!N.bins) return null;
  const f = src => { if (!src || !src[key]) return null; const b = N.bins, m = src[key].m, s = src[key].s;
    let i = b.findIndex(c => c >= x); if (i <= 0 || i >= b.length) return null;
    if (m[i - 1] === null || m[i] === null) return null; const t = (x - b[i - 1]) / (b[i] - b[i - 1]);
    return {m: m[i - 1] + (m[i] - m[i - 1]) * t, s: s[i - 1] + (s[i] - s[i - 1]) * t}; };
  return f(N.depth) || f(N.all);
}
function ticks(lo, hi, n = 5) { const st = Math.pow(10, Math.floor(Math.log10((hi - lo) / n))), e = (hi - lo) / n / st;
  const step = st * (e > 5 ? 10 : e > 2 ? 5 : e > 1 ? 2 : 1), out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6)); return out; }
const NAME = {attack_angle: ['Vertical attack angle', '°', '+ = uphill'], vba: ['Vertical bat angle (inferred)', '°', '− = barrel below hands'],
              hba: ['Horizontal bat angle (inferred)', '°', '+ = barrel ahead of hands, 0 = square'],
              attack_direction: ['Horizontal attack angle', '°', '+ = oppo, − = pull']};
const MATCH = {attack_angle: 'match_v', attack_direction: 'match_h'};
function draw() {
  const key = document.getElementById('cm').value, xmode = document.getElementById('cx').value,
        fast = document.getElementById('cs').value === 'fast', vs = document.getElementById('cv').value;
  const box = document.getElementById('chart'), W = Math.max(320, box.clientWidth), H = 320, L = 46, R = 14, Tp = 16, B = 36;
  const svg = [], note = [], lines = [], dots = [], bands = [], marks = [], refs = [];
  let xs;
  GS.forEach((g, k) => {                       // the angle that would match this group's pitch (attack angles only)
    if (!ON[k] || !MATCH[key]) return;
    const r = rows(g.series, MATCH[key], false), v = r.pts.map(p => p[r.ci]);
    const side = g.side === 'L' ? -1 : 1;
    const m = v.length ? v.reduce((a, b) => a + b, 0) / v.length : (key === 'attack_angle' ? -g.read.vaa : -side * g.read.haa);
    refs.push([m, g.color]);
  });
  if (xmode === 'depth') {
    xs = [-20, 28];
    GS.forEach((g, k) => {
      if (!ON[k]) return;
      const r = rows(g.series, key, fast), col = g.color;
      if (r.fb) note.push(`${String.fromCharCode(65 + k)}: no 75+ swings, using competitive`);
      r.pts.forEach(p => dots.push([p[r.pi], p[r.ci], col]));
      const seg = []; for (let x = xs[0]; x <= xs[1]; x += 0.5) { const q = kern(r.pts, r.ci, r.pi, x); seg.push([x, q ? q.m : null]); }
      lines.push([seg, col, false]);
      if (g.read && g.read.front !== null) marks.push([g.read.front, col]);
      if (vs === 'league') {
        const band = []; for (let x = xs[0]; x <= xs[1]; x += 1) { const n = leagueAt(g, key, x); if (n && n.s !== null) band.push([x, n.m, n.s]); }
        if (band.length) bands.push([band, GS.length > 1 ? col : '#9aa4ad']);
      }
    });
    if (vs === 'self') {
      const r = rows(ALL, key, fast), band = [];
      for (let x = xs[0]; x <= xs[1]; x += 1) { const q = kern(r.pts, r.ci, r.pi, x, 4); if (q) band.push([x, q.m, q.s]); }
      if (band.length) bands.push([band, '#9aa4ad']);
      note.push(`Shaded = all of his ${fast && ALL.n_fast > 0 ? 'fast (75+)' : 'competitive'} swings, ±1 SD at each depth.`);
    } else {
      const k0 = (GS[0].norm || {}).key;
      note.push('Shaded = MLB fast swings (75+ mph) ±1 SD at each depth' + (k0 ? ` (${k0.replace('|', ', zone ')})` : '') + '.');
      if (!(GS[0].norm || {}).bins) note.push('MLB range not built yet: run  python -m pipeline.bat_norms <season>');
    }
    marks.push([0, '#9aa4ad', 'front of plate']);
    note.unshift('Each dot = one of his swings, placed where he met the ball; line = smoothed. Vertical dashed = his average contact.' +
                 (key === 'vba' || key === 'hba' ? ' Bat angles are inferred from each swing\'s measured attack angle, direction + tilt.' : ' Attack angles are measured.'));
  } else {
    let lo = 0, hi = 10;
    GS.forEach((g, k) => {
      if (!ON[k]) return;
      const info = pathInfo(g); lo = Math.min(lo, info[0].t); hi = Math.max(hi, info[info.length - 1].t);
      const i0 = info.findIndex(q => q.flag === 1), i1 = info.length - 1 - [...info].reverse().findIndex(q => q.flag === 1);
      lines.push([info.slice(0, i0 + 1).map(q => [q.t, q[key]]), g.color, true]);
      lines.push([info.slice(i0, i1 + 1).map(q => [q.t, q[key]]), g.color, false]);
      lines.push([info.slice(i1).map(q => [q.t, q[key]]), g.color, true]);
      if (vs === 'league') {
        const N = g.norm || {}, T = N.tgrid, src = (N.time || {})[key], all = (N.time_all || {})[key];
        if (T && (src || all)) {
          const band = T.map((t, i) => { const q = src && src.m[i] !== null ? src : all;
            return q && q.m[i] !== null ? [t, q.m[i], q.s[i]] : null; }).filter(Boolean);
          if (band.length) bands.push([band, GS.length > 1 ? g.color : '#9aa4ad']);
        }
      }
    });
    if (vs === 'self') {
      const st = D.self_time || {}, b = (fast && st.fast) ? st.fast : st.comp;
      if (b && b[key] && st.tgrid) {
        const band = st.tgrid.map((t, i) => b[key].m[i] !== null ? [t, b[key].m[i], b[key].s[i]] : null).filter(Boolean);
        if (band.length) bands.push([band, '#9aa4ad']);
      }
      if (fast && st.comp && !st.fast) note.push('He has under 5 fast (75+) swings with a contact point: his range uses competitive swings.');
    }
    xs = [Math.floor(lo / 10) * 10, Math.ceil(hi / 10) * 10];
    marks.push([0, '#c0392b', 'contact']);
    note.push('Along his average path, through contact and into the follow-through: solid = measured last 40 ms, dashed = inferred. ' +
              (vs === 'self' ? 'Shaded = all of his own swings, each along its own path, ±1 SD at each moment.'
                             : 'Shaded = MLB fast swings (75+ mph), each along its own path, ±1 SD at each moment' +
                               ((GS[0].norm || {}).key ? ` (${GS[0].norm.key.replace('|', ', zone ')})` : '') + '.') +
              ' Every path here (his and the range) comes from the same swing model on measured angles.');
    if (vs === 'league' && !(GS[0].norm || {}).tgrid) note.push('Time ranges not built yet: rerun  python -m pipeline.bat_norms <season>');
  }
  if (refs.length) note.push('Horizontal dashed = the angle that matches that pitch (green = within 5°).');
  const ys = [];
  refs.forEach(r => ys.push(r[0] - 6, r[0] + 6));
  dots.forEach(d => ys.push(d[1])); lines.forEach(l => l[0].forEach(p => p[1] !== null && p[1] !== undefined && ys.push(p[1])));
  bands.forEach(b => b[0].forEach(p => { ys.push(p[1] - p[2]); ys.push(p[1] + p[2]); }));
  let y0 = Math.min(...ys), y1 = Math.max(...ys); if (!isFinite(y0)) { y0 = -10; y1 = 10; }
  const pad = (y1 - y0) * 0.06 || 1; y0 -= pad; y1 += pad;
  const X = x => L + (x - xs[0]) / (xs[1] - xs[0]) * (W - L - R), Y = y => Tp + (y1 - y) / (y1 - y0) * (H - Tp - B);
  refs.forEach(([m, c]) => svg.push(`<rect x="${L}" y="${Y(m + 5)}" width="${W - L - R}" height="${Y(m - 5) - Y(m + 5)}" fill="#2e9d5b" fill-opacity=".06"/>`));
  ticks(y0, y1).forEach(v => svg.push(`<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#eef1f3"/><text x="${L - 5}" y="${Y(v) + 3}" text-anchor="end">${v}</text>`));
  ticks(xs[0], xs[1], 8).forEach(v => svg.push(`<text x="${X(v)}" y="${H - B + 14}" text-anchor="middle">${v}</text>`));
  svg.push(`<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${xmode === 'depth' ? 'inches in front of the plate (deep → out front)' : 'ms from contact'}</text>`);
  svg.push(`<text x="12" y="${(Tp + H - B) / 2}" transform="rotate(-90 12 ${(Tp + H - B) / 2})" text-anchor="middle">${NAME[key][0]} (°) · ${NAME[key][2]}</text>`);
  if (xmode === 'time') { const ms = GS.flatMap((g, k) => ON[k] ? g.arc.filter(q => q[3] === 1).map(q => q[4] * 1000) : []);
    if (ms.length) svg.push(`<rect x="${X(Math.min(...ms))}" y="${Tp}" width="${X(Math.max(...ms)) - X(Math.min(...ms))}" height="${H - Tp - B}" fill="#2f5fb3" fill-opacity=".05"/>`); }
  bands.forEach(([b, c, kind, k]) => {
    if (kind === 'bar') { const [x, m, s] = b[0], xx = X(x) + 7 + 9 * (k || 0);
      svg.push(`<line x1="${xx}" x2="${xx}" y1="${Y(m - s)}" y2="${Y(m + s)}" stroke="${c}" stroke-width="6" stroke-opacity=".35" stroke-linecap="round"/><circle cx="${xx}" cy="${Y(m)}" r="3" fill="#fff" stroke="${c}"/>`); return; }
    const runs = []; let cur = [];
    b.forEach((p, i) => { if (i && p[0] - b[i - 1][0] > 2.5) { runs.push(cur); cur = []; } cur.push(p); }); runs.push(cur);
    runs.filter(r => r.length > 1).forEach(r => {
      const up = r.map(p => `${X(p[0])},${Y(p[1] + p[2])}`), dn = r.slice().reverse().map(p => `${X(p[0])},${Y(p[1] - p[2])}`);
      svg.push(`<polygon points="${up.concat(dn).join(' ')}" fill="${c}" fill-opacity="${c === '#9aa4ad' ? .22 : .1}"/>`);
      svg.push(`<polyline points="${r.map(p => `${X(p[0])},${Y(p[1])}`).join(' ')}" fill="none" stroke="#9aa4ad" stroke-dasharray="3 3"/>`);
    });
  });
  refs.forEach(([m, c]) => svg.push(`<line x1="${L}" x2="${W - R}" y1="${Y(m)}" y2="${Y(m)}" stroke="${c}" stroke-dasharray="7 4" stroke-opacity=".8"/><text x="${W - R - 4}" y="${Y(m) - 4}" text-anchor="end" fill="${c}">matches the pitch</text>`));
  marks.forEach(([x, c, lab]) => { if (x < xs[0] || x > xs[1]) return;
    svg.push(`<line x1="${X(x)}" x2="${X(x)}" y1="${Tp}" y2="${H - B}" stroke="${c}" stroke-dasharray="4 3"/>` + (lab ? `<text x="${X(x) + 3}" y="${Tp + 9}">${lab}</text>` : '')); });
  dots.forEach(([x, y, c]) => { if (x >= xs[0] && x <= xs[1]) svg.push(`<circle cx="${X(x)}" cy="${Y(y)}" r="2.4" fill="${c}" fill-opacity=".3"/>`); });
  lines.forEach(([pts, c, dash]) => { let run = [];
    const flush = () => { if (run.length > 1) svg.push(`<polyline points="${run.join(' ')}" fill="none" stroke="${c}" stroke-width="2.4" ${dash ? 'stroke-dasharray="5 4"' : ''}/>`); run = []; };
    pts.forEach(([x, y]) => { if (y === null || y === undefined || x < xs[0] || x > xs[1]) flush(); else run.push(`${X(x)},${Y(y)}`); }); flush(); });
  box.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${svg.join('')}</svg>`;
  document.getElementById('cn').innerHTML = note.join('<br>');
}
const chips = document.getElementById('cg');
GS.forEach((g, k) => { const b = document.createElement('button'); b.className = 'on'; b.style.setProperty('--c', g.color);
  b.innerHTML = `<i></i>${String.fromCharCode(65 + k)}`; b.title = g.name;
  b.onclick = () => { ON[k] = !ON[k]; b.classList.toggle('on', ON[k]); draw(); }; chips.appendChild(b); });
['cm', 'cx', 'cs', 'cv'].forEach(id => document.getElementById(id).onchange = draw);
window.addEventListener('resize', draw); draw();
"""


def chart_html(scenes: list, all_series: dict, self_time: dict | None = None) -> str:
    """The plane-matching chart under the 3D view: how well each loaded swing matches the pitch's plane (vertical
    vs VAA) or line (horizontal vs HAA), by contact depth (his measured swings) or time (his average path), vs the
    league (MLB fast swings) or himself (all his swings). Drawn at the frame's real width with a fixed height."""
    import json
    scs = [x for x in scenes if x]
    data = {"groups": [{k: x.get(k) for k in ("name", "color", "series", "norm", "read", "arc", "side", "O")} for x in scs],
            "all": all_series, "self_time": self_time or {}}
    css = ("body{margin:0;font:12px system-ui,sans-serif;color:#16202b}"
           ".wrap{border:1px solid #e6eaed;border-radius:10px;padding:8px 12px 8px}"
           ".hd{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:4px}.hd b{font-size:14px;margin-right:6px}"
           "select{font:12px system-ui;padding:3px 5px;border:1px solid #d9dee2;border-radius:6px;background:#fff}"
           "#cg{margin-left:auto;display:flex;gap:4px}#cg button{font:700 12px system-ui;border:1px solid #d9dee2;"
           "background:#fff;border-radius:6px;padding:3px 8px;cursor:pointer;display:inline-flex;gap:5px;"
           "align-items:center;color:#9aa4ad}#cg i{width:9px;height:9px;border-radius:50%;background:#d9dee2;"
           "display:inline-block}#cg button.on{color:#16202b;border-color:var(--c)}#cg button.on i{background:var(--c)}"
           "#chart text{font:10px system-ui;fill:#6a7681}#cn{color:#6a7681;font-size:11px;line-height:1.45}")
    head = ('<div class="hd"><b>Bat angles</b>'
            '<select id="cm"><option value="attack_angle">Vertical attack angle</option>'
            '<option value="vba">Vertical bat angle (inferred)</option>'
            '<option value="hba">Horizontal bat angle (inferred)</option>'
            '<option value="attack_direction">Horizontal attack angle</option></select>'
            '<select id="cx"><option value="depth">By contact depth (his measured swings)</option>'
            '<option value="time">By time (his average path)</option></select>'
            '<select id="cv"><option value="league">vs League (MLB fast swings)</option>'
            '<option value="self">vs Himself (all his swings)</option></select>'
            '<select id="cs"><option value="fast">His fast swings (75+)</option>'
            '<option value="comp">His competitive swings</option></select><span id="cg"></span></div>')
    js = ("<script>function fit(){window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',"
          "height:document.documentElement.scrollHeight+2},'*');}</script>")
    blob = json.dumps(data).replace("</", "<\\/")
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body><div class="wrap">'
            f'{head}<div id="chart"></div><div id="cn"></div></div>'
            f'<script type="application/json" id="cd">{blob}</script>'
            f'{js}<script>{COMMON_JS}{CHART_JS}fit();</script></body></html>')


# ============================================================================= simulator tabs: Contact / Timing / Arsenal
LAYERS_CSS = """
#tabs{display:inline-flex;border:1px solid #d9dee2;border-radius:8px;overflow:hidden;margin-left:auto}
#tabs button{font:600 12px system-ui;border:none;background:#fff;padding:5px 12px;cursor:pointer;color:#16202b}
#tabs button.on{background:#4a8b96;color:#fff}
#hd{display:flex;align-items:center;gap:10px}
#lay{margin-top:10px;font:12.5px system-ui;color:#16202b} #lay[hidden]{display:none}
#lay .h{font-weight:700;font-size:13.5px} #lay .s{color:#6a7681;font-size:11.5px;margin:2px 0 6px;line-height:1.45}
#lay svg{display:block;max-width:100%} #lay .tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}
#lay .tile{border:1px solid #e6eaed;border-radius:8px;padding:5px;text-align:center}
#lay .tile div:first-child{font-size:10.5px;color:#6a7681} #lay .tile div:last-child{font:700 14px ui-monospace,Menlo,monospace}
#lay .leg{display:flex;align-items:center;gap:6px;font-size:10.5px;color:#6a7681;margin-top:4px}
#lay .bar{height:8px;width:180px;border-radius:4px;background:linear-gradient(90deg,#313695,#74add1,#ffffbf,#fdae61,#a50026)}
#lay table{border-collapse:collapse;width:100%;margin-top:6px} #lay td,#lay th{padding:4px 6px;border-bottom:1px solid #eef1f3;
  text-align:right;font-size:12px} #lay td:first-child,#lay th:first-child{text-align:left} #lay th{color:#6a7681;font-size:10.5px}
#lay .act{color:#9aa5ae;font-weight:400} #lay .camb{display:flex;gap:6px;margin:4px 0} #lay .camb button{font:600 11.5px system-ui;border:1px solid #d9dee2;background:#fff;
  border-radius:6px;padding:4px 9px;cursor:pointer} #lay .tip{font-size:11px;color:#6a7681;min-height:15px;margin-top:3px}
"""

LAYERS_JS = r"""
const LY = SC.layers || null;
const HEAT = [[0, [49, 54, 149]], [.35, [116, 173, 209]], [.55, [255, 255, 191]], [.75, [253, 174, 97]], [1, [165, 0, 38]]];
function heat(v, lo = -0.3, hi = 1.2) {
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  for (let i = 1; i < HEAT.length; i++) if (t <= HEAT[i][0]) {
    const [t0, c0] = HEAT[i - 1], [t1, c1] = HEAT[i], f = (t - t0) / (t1 - t0);
    return c0.map((c, k) => Math.round(c + (c1[k] - c) * f)); }
  return HEAT[HEAT.length - 1][1];
}
const rgb = c => `rgb(${c.join(',')})`;
const OUTC = {fair: '#2e9d5b', foul: '#e0a400', whiff: '#9aa5ae'};
const sgn = (v, d = 2) => (v > 0.0005 ? '+' : '') + Number(v).toFixed(d);
let TAB = 'Swing', OVL = null;
function clearOverlay() { if (OVL && R3) { R3.scene.remove(OVL); } OVL = null; }
function overlay() { clearOverlay(); if (!R3) return null; OVL = new THREE.Group(); R3.scene.add(OVL); return OVL; }
function ball(at, color, r = 1.45, op = 1) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 10), new THREE.MeshBasicMaterial({color, transparent: op < 1, opacity: op}));
  m.position.copy(V(at)); return m;
}
function atContact() { const k = document.getElementById('s3k'); if (k) k.click(); }
const leg = '<div class="leg">−0.3 runs (out / whiff)<span class="bar"></span>+1.2 (barrel)</div>';
// ---------- Contact
function grid(cl, W, H, dots) {
  const nE = cl.eff.length, nD = cl.D.length, cw = W / nD, ch = H / nE;
  let s = '';
  for (let i = 0; i < nE; i++) for (let j = 0; j < nD; j++)
    s += `<rect x="${(40 + j * cw).toFixed(1)}" y="${(8 + (nE - 1 - i) * ch).toFixed(1)}" width="${(cw + .6).toFixed(1)}" height="${(ch + .6).toFixed(1)}" fill="${rgb(heat(cl.runs[i][j]))}" data-i="${i}" data-j="${j}"/>`;
  const X = d => 40 + (d - cl.D[0] + 0.05) / (cl.D[nD - 1] - cl.D[0] + 0.1) * W, Y = e => 8 + (1 - (e - cl.eff[0]) / (cl.eff[nE - 1] - cl.eff[0])) * H;
  (dots || []).forEach(([d, e]) => { if (d >= cl.D[0] && d <= cl.D[nD - 1] && e >= cl.eff[0])
    s += `<circle cx="${X(d).toFixed(1)}" cy="${Y(Math.min(e, 1)).toFixed(1)}" r="2.3" fill="#fff" stroke="#16202b" stroke-width=".7" opacity=".85" pointer-events="none"/>`; });
  [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2].forEach(v => s += `<text x="${X(v).toFixed(1)}" y="${H + 22}" font-size="10" fill="#6a7681" text-anchor="middle">${sgn(v, 1)}"</text>`);
  [.7, .8, .9, 1].forEach(e => s += `<text x="34" y="${(Y(e) + 3).toFixed(1)}" font-size="10" fill="#6a7681" text-anchor="end">${Math.round(e * 100)}%</text>`);
  return `<svg viewBox="0 0 ${W + 50} ${H + 44}" width="100%">${s}<text x="${40 + W / 2}" y="${H + 38}" font-size="10.5" text-anchor="middle">← topped · square · undercut →</text><text x="9" y="${8 + H / 2}" font-size="10.5" transform="rotate(-90 9 ${8 + H / 2})" text-anchor="middle">barrel efficiency</text></svg>`;
}
function lookupRuns(cl, d, e) {
  let j = 0, i = 0;
  cl.D.forEach((v, k) => { if (Math.abs(v - d) < Math.abs(cl.D[j] - d)) j = k; });
  cl.eff.forEach((v, k) => { if (Math.abs(v - e) < Math.abs(cl.eff[i] - e)) i = k; });
  return cl.runs[i][j];
}
function showContact() {
  const cl = LY.contact; let bi = 0, bj = 0;
  cl.runs.forEach((r, i) => r.forEach((v, j) => { if (v > cl.runs[bi][bj]) { bi = i; bj = j; } }));
  const his = (cl.dots || []).filter(([d, e]) => d >= cl.D[0] && d <= cl.D[cl.D.length - 1]);
  const val = his.length ? his.reduce((a, [d, e]) => a + lookupRuns(cl, d, Math.min(e, 1)), 0) / his.length : null;
  document.getElementById('lay').innerHTML = `<div class="h">Contact: every possible contact on the barrel, priced</div>
    <div class="camb"><button onclick="closeUp()">Bat close-up</button><button onclick="R3.preset('Catcher')">Catcher</button><button onclick="R3.preset('Open side')">Open side</button></div>
    <div class="s">This swing vs this pitch. <b>The bat in 3D is painted live</b>: at every instant of the swing, each spot on the barrel coloured by what the pitch would be worth if it met that spot right then (the bat's own angle, direction and speed at that moment). Play or scrub the swing to watch the good part of the barrel grow and fade. The chart: the same thing at his contact point -- across: topped ← square → undercut; up: share of the sweet spot's exit velo. White dots = his real balls in play.</div>
    <div class="tip" id="ptip"></div>${grid(cl, 520, 250, cl.dots)}${leg}<div class="tip" id="ltip">Hover a cell.</div>
    <div class="tiles"><div class="tile"><div>Best contact</div><div>${sgn(cl.D[bj], 1)}" · ${Math.round(cl.eff[bi] * 100)}%</div></div>
    <div class="tile"><div>→</div><div>${Math.round(cl.ev[bi][bj])} mph · ${Math.round(cl.la[bi][bj])}°</div></div>
    <div class="tile"><div>Best cell</div><div>${sgn(cl.runs[bi][bj])} runs</div></div>
    <div class="tile"><div>His contacts here (avg)</div><div>${val === null ? '–' : sgn(val)}</div></div></div>`;
  document.querySelectorAll('#lay rect[data-i]').forEach(r => r.onmouseenter = () => { const i = +r.dataset.i, j = +r.dataset.j;
    document.getElementById('ltip').textContent = `${sgn(cl.D[j], 1)}" ${cl.D[j] >= 0 ? 'under' : 'over'} · ${Math.round(cl.eff[i] * 100)}% efficiency → ${cl.ev[i][j]} mph, ${cl.la[i][j]}°, spray ${cl.spray[i][j]}° · ${sgn(cl.runs[i][j])} runs`; });
  overlay(); paintOn('contact');
}
// ---------- Timing
function showTiming() {
  const T = LY.timing, cl = LY.contact, rows = T.rows, W = 520, H = 220;
  const Dg = []; for (let d = -1.5; d <= 2.001; d += 0.25) Dg.push(+d.toFixed(2));
  const cw = W / rows.length, ch = H / Dg.length;
  let s = '';
  rows.forEach((r, a) => Dg.forEach((d, b) => {
    const v = r.result === 'whiff' ? -0.3 : lookupRuns(cl, d, r.eff || 1);
    s += `<rect x="${(40 + a * cw).toFixed(1)}" y="${(8 + (Dg.length - 1 - b) * ch).toFixed(1)}" width="${(cw + .6).toFixed(1)}" height="${(ch + .6).toFixed(1)}" fill="${rgb(heat(v))}" opacity=".55"/>`; }));
  const Y = d => 8 + (1 - (d - Dg[0]) / (Dg[Dg.length - 1] - Dg[0])) * H;
  const pts = rows.filter(r => r.D !== undefined).map(r => `${(40 + (rows.indexOf(r) + .5) * cw).toFixed(1)},${Y(Math.max(-1.5, Math.min(2, r.D))).toFixed(1)}`).join(' ');
  s += `<polyline points="${pts}" fill="none" stroke="#16202b" stroke-width="2" stroke-dasharray="4 3"/>`;
  rows.forEach((r, a) => { const x = 40 + (a + .5) * cw;
    if (r.result === 'whiff') s += `<text x="${x}" y="${8 + H / 2}" font-size="11" text-anchor="middle" fill="#6a7681">×</text>`;
    else s += `<circle cx="${x.toFixed(1)}" cy="${Y(Math.max(-1.5, Math.min(2, r.D))).toFixed(1)}" r="4" fill="${rgb(heat(r.runs))}" stroke="#16202b" stroke-width=".8"/>`;
    if (a % 2 === 0) s += `<text x="${x.toFixed(1)}" y="${H + 22}" font-size="10" fill="#6a7681" text-anchor="middle">${r.shift > 0 ? '+' : ''}${r.shift}</text>`; });
  [-1, 0, 1, 2].forEach(d => s += `<text x="34" y="${(Y(d) + 3).toFixed(1)}" font-size="10" fill="#6a7681" text-anchor="end">${sgn(d, 0)}"</text>`);
  const strip = rows.map(r => `<td data-s="${r.shift}" title="${r.shift} ms: ${r.result}${r.ev ? ', ' + r.ev + ' mph ' + r.la + '°' : ''} (${sgn(r.runs)})" style="background:${rgb(heat(r.runs))};padding:7px 0"></td>`).join('');
  const by = sh => rows.find(r => r.shift === sh) || {runs: NaN};
  document.getElementById('lay').innerHTML = `<div class="h">Timing: every crossing of this pitch</div>
    <div class="camb"><button onclick="closeUp()">Bat close-up</button><button onclick="R3.preset('Catcher')">Catcher</button><button onclick="R3.preset('Open side')">Open side</button></div>
    <div class="s">Same swing, same pitch, started earlier (+ ms) or later (−). <b>In 3D, the bat at any moment is the bat this pitch meets if he's that many ms early</b> (scrub to −15 ms: that's 15 ms late) -- painted with what each barrel spot is worth, ● = where the ball actually lands on it. Chart: where it meets the barrel at each timing (above = undercut), coloured by runs; × = a miss. Early = further along his path: higher and more pull.</div><div class="tip" id="ptip"></div>
    <svg viewBox="0 0 ${W + 50} ${H + 40}" width="100%">${s}<text x="${40 + W / 2}" y="${H + 36}" font-size="10.5" text-anchor="middle">← late · ms vs on time · early →</text></svg>${leg}
    <table id="tstrip"><tr><td style="width:96px">Value by timing</td>${strip}</tr></table>
    <div class="tiles"><div class="tile"><div>On time</div><div>${sgn(by(0).runs)}</div></div><div class="tile"><div>15 ms late</div><div>${sgn(by(-15).runs)}</div></div>
    <div class="tile"><div>15 ms early</div><div>${sgn(by(15).runs)}</div></div><div class="tile"><div>Window above 0 runs</div><div>${T.window_ms} ms</div></div></div>`;
  const g = overlay(); if (!g) return;
  rows.forEach(r => g.add(ball(r.at, r.result === 'whiff' ? 0x9aa5ae : new THREE.Color(...heat(r.runs).map(c => c / 255)), 0.45, r.result === 'whiff' ? .35 : 0.45)));
  paintOn('timing');
}
// ---------- Arsenal
function showArsenal() {
  const A = LY.arsenal, PC = LY.pt_color, W = 260, H = 260;
  const X = x => W / 2 + (x / 12) * 75, Y = z => H - 20 - ((z / 12) - 1.0) * 75;
  let s = `<rect x="${X(-8.5)}" y="${Y(42)}" width="${X(8.5) - X(-8.5)}" height="${Y(18) - Y(42)}" fill="none" stroke="#4a8b96" stroke-width="1.6"/>`;
  A.rows.forEach(r => { const c = PC[r.type] || '#555', x = X(r.at[0]).toFixed(1), y = Y(r.at[2]).toFixed(1);
    s += r.result === 'fair' ? `<circle cx="${x}" cy="${y}" r="3.4" fill="${c}"/>` : r.result === 'whiff' ? `<circle cx="${x}" cy="${y}" r="3.2" fill="none" stroke="${c}" stroke-width="1.5"/>`
      : `<rect x="${x - 3}" y="${y - 3}" width="6" height="6" fill="none" stroke="${c}" stroke-width="1.4"/>`; });
  const pct = v => v === null || v === undefined ? '–' : Math.round(v * 100) + '%';
  const vs = (m, a) => `${pct(m)}<span class="act"> / ${pct(a)}</span>`;
  const rows = A.by_type.map(b => `<tr><td><b style="color:${PC[b.type] || '#555'}">${b.type}</b></td><td>${b.n}</td><td>${b.velo}</td><td>${vs(b.whiff, b.act_whiff)}</td><td>${vs(b.foul, b.act_foul)}</td><td>${vs(b.fair, b.act_fair)}</td><td style="background:${rgb(heat(b.runs, -0.15, 0.3))}">${sgn(b.runs)}</td></tr>`).join('');
  const c = A.counts || {};
  document.getElementById('lay').innerHTML = `<div class="h">Arsenal: the same swing vs every pitch that looks the same at the commit point</div>
    <div class="s">${A.mix_src === 'his' ? 'The pitches he actually swings at' : 'A typical MLB'} ${SC.model && SC.model.hand === 'L' ? 'vs LHP' : 'vs RHP'}${A.mix_src === 'his' ? ' (his mix, velo + movement)' : ' pitch mix'}, every pitch lined up to pass the same spot at <b>his commit point: ${A.commit_ms} ms before the plate</b> (his swing time from his swing length + bat speed, + the decision). At that point he reads each pitch and aims + times his swing where a fastball with that look would arrive; after that he can only adjust a little, and each swing carries his own spread (${A.cons > 1.05 ? 'looser' : A.cons < 0.95 ? 'tighter' : 'about average'} than the league's). Chart = where each pitch reached his bat (catcher's view): ● in play, ○ whiff, □ foul.${A.fitted ? '' : ' <i>Model not yet calibrated to MLB swings (run the calibration on the Mac).</i>'}</div>
    <div class="camb"><button id="cpl">At the plate</button><button id="ccm">From the commit point</button></div>
    <div style="display:flex;gap:12px;align-items:flex-start"><svg viewBox="0 0 ${W} ${H}" width="44%">${s}</svg>
    <div style="flex:1"><table><tr><th>Pitch</th><th>n</th><th>Velo</th><th>Whiff<br><span class="act">model / his</span></th><th>Foul<br><span class="act">model / his</span></th><th>In play<br><span class="act">model / his</span></th><th>Runs / swing</th></tr>${rows}</table>
    <div class="tiles" style="grid-template-columns:1fr 1fr 1fr"><div class="tile"><div>In play / foul / whiff</div><div>${c.fair || 0} / ${c.foul || 0} / ${c.whiff || 0}</div></div>
    <div class="tile"><div>Path robustness (runs / 100)</div><div>${sgn(A.runs100, 1)}</div></div>
    <div class="tile"><div>MLB-average path shape</div><div>${A.mlb_runs100 === null || A.mlb_runs100 === undefined ? '–' : sgn(A.mlb_runs100, 1)}</div></div></div></div></div>`;
  const g = overlay(); if (!g) return;
  // every pitch flies in on the scene clock: a ball in its pitch colour from release, its trail drawn from the commit
  // point on, and -- when it reaches his bat -- the outcome ball (green in play / orange foul / grey whiff)
  const tc = A.t_commit ?? -0.17;
  const P = A.rows.map(r => {
    const col = new THREE.Color(PC[r.type] || '#555');
    const fly = r.fly || r.line.map((q, i) => [...q, tc + i * 0.01]);
    const trailPts = fly.filter(q => q[3] >= tc);
    const geo = new THREE.BufferGeometry().setFromPoints(trailPts.map(V));
    const trail = new THREE.Line(geo, new THREE.LineBasicMaterial({color: col, transparent: true, opacity: .3}));
    const pb = ball(fly[0], col, 1.45, 1), out = ball(r.at, new THREE.Color(OUTC[r.result]), 1.45, .95);
    out.visible = false; g.add(trail, pb, out);
    const hit = r.result !== 'whiff';
    return {fly, trailPts, geo, pb, out, hit, tm: r.t};
  });
  const tok = OVL;
  function step() {
    if (OVL !== tok || !R3) return;
    const t = R3.time();
    P.forEach(p => {
      const f = p.fly, end = p.hit ? Math.min(p.tm, f[f.length - 1][3]) : f[f.length - 1][3];
      if (t < f[0][3] || t > end + (p.hit ? 0 : 0.02)) p.pb.visible = false;
      else {
        let k = 1; while (k < f.length - 1 && f[k][3] < t) k++;
        const a = f[k - 1], c = f[k], u = Math.max(0, Math.min(1, (t - a[3]) / ((c[3] - a[3]) || 1)));
        p.pb.position.copy(V([a[0] + (c[0] - a[0]) * u, a[1] + (c[1] - a[1]) * u, a[2] + (c[2] - a[2]) * u]));
        p.pb.visible = true;
      }
      p.out.visible = t >= p.tm;
      let n = 0; while (n < p.trailPts.length && p.trailPts[n][3] <= t) n++;
      p.geo.setDrawRange(0, n);
    });
    requestAnimationFrame(step);
  }
  step();
  const mk = ball(A.commit, 0x16202b, 0.5, .5); g.add(mk);
  document.getElementById('cpl').onclick = () => { mk.visible = true; R3.preset('Open side'); };
  document.getElementById('ccm').onclick = () => { R3.look(A.eye || [A.commit[0], A.commit[1] + 150, A.commit[2] + 10], SC.contact, 9); };
}
// ---------- the real bat, painted: at each instant, every spot on the barrel priced if the pitch met it right then
let PAINT = null;
const WOOD = [154, 123, 82];
function paintOff() {
  if (!PAINT) return;
  const bt = PAINT.R && PAINT.R.bat;
  if (bt) { bt.material.map = null; bt.material.color.set(0x9a7b52); bt.material.needsUpdate = true; }
  if (PAINT.mk && PAINT.R) PAINT.R.scene.remove(PAINT.mk);
  PAINT = null;
}
function paintOn(mode) {
  paintOff(); atContact();
  const HT = LY && LY.heat; if (!R3 || !R3.bat || !HT) return;
  const cv = document.createElement('canvas'); cv.width = 64; cv.height = 128;
  const ctx = cv.getContext('2d'), tex = new THREE.CanvasTexture(cv);
  R3.bat.material.map = tex; R3.bat.material.color.set(0xffffff); R3.bat.material.needsUpdate = true;
  let mk = null;
  if (mode === 'timing') { mk = ball([0, 0, 0], 0xffffff, 1.5, 1); mk.material.color.set(0x16202b); mk.visible = false; R3.scene.add(mk); }
  PAINT = {mode, tex, ctx, img: ctx.createImageData(64, 128), mk, last: -2, R: R3};
  closeUp();
  loopPaint();
}
function closeUp() {                                // the bat seen from the pitch's side, a little above, near contact
  const HT = LY && LY.heat; if (!R3 || !HT) return;
  const k = HT.t.reduce((b, x, i) => Math.abs(x) < Math.abs(HT.t[b]) ? i : b, 0), f = HT.f[k], C = SC.contact;
  if (!f || !(f[0] || f[1] || f[2])) return;
  R3.look([C[0] + f[0] * 60, C[1] + f[1] * 60, C[2] + f[2] * 60 + 20], [C[0], C[1], C[2] + 1], 26);
}
function paintAt(k) {
  const P = PAINT, HT = LY.heat, d = P.img.data, R = k >= 0 ? HT.runs[k] : null;
  let fl = null, sl = null;
  const f = k >= 0 ? HT.f[k] : null;
  if (R && f && (f[0] || f[1] || f[2])) {
    const qi = P.R.bat.quaternion.clone().invert(), n2 = v => { const L = Math.hypot(v.x, v.z) || 1; return [v.x / L, v.z / L]; };
    fl = n2(V(f).applyQuaternion(qi)); sl = n2(V(HT.s[k]).applyQuaternion(qi));
  }
  const L0 = HT.lam[0], L1 = HT.lam[HT.lam.length - 1], nD = HT.D.length;
  for (let y = 0; y < 128; y++) {
    const ly = 16.5 - (y + 0.5) / 128 * 33, lam = 10.5 - ly;          // local y (+ = tip) -> inches from the sweet spot
    for (let x = 0; x < 64; x++) {
      let c = WOOD;
      if (fl && lam >= L0 - 0.5 && lam <= L1 + 0.5) {
        const th = (x + 0.5) / 64 * 2 * Math.PI, dx = Math.sin(th), dz = Math.cos(th);
        // height across the bat (above / below its centre line, as the ball sees it) sets the offset: the top of
        // the barrel = undercut, the bottom = topped -- wrapped all the way round so it reads from any camera
        const cs = dx * sl[0] + dz * sl[1];
        const li = Math.max(0, Math.min(R.length - 1, Math.round(lam - L0)));
        const fd = Math.max(0, Math.min(nD - 1.001, (2.68 * cs - HT.D[0]) / 0.2)), d0 = Math.floor(fd), w1 = fd - d0;
        const a0 = R[li][d0], a1 = R[li][d0 + 1];
        const v = a0 === null ? a1 : a1 === null ? a0 : a0 + (a1 - a0) * w1; c = v === null ? WOOD : heat(v);
      }
      const o = 4 * (y * 64 + x); d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
    }
  }
  P.ctx.putImageData(P.img, 0, 0); P.tex.needsUpdate = true;
}
function loopPaint() {
  const P = PAINT; if (!P || P.R !== R3) return;
  requestAnimationFrame(loopPaint);
  const HT = LY.heat, t = R3.time(), n = HT.t.length;
  let k = -1;
  if (t >= HT.t[0] - 0.0015 && t <= HT.t[n - 1] + 0.0015) { k = Math.round((t - HT.t[0]) / (HT.t[1] - HT.t[0])); k = Math.max(0, Math.min(n - 1, k)); }
  if (k !== P.last) { P.last = k; paintAt(k); }
  const tip = document.getElementById('ptip');
  if (P.mode === 'timing') {                       // the bat at time t = the bat this pitch meets if he's t ms early
    const ms = t * 1000, rows = LY.timing.rows;
    let r = null; rows.forEach(x => { if (Math.abs(x.shift - ms) <= 1.6) r = x; });
    P.mk.visible = !!(r && r.result !== 'whiff'); if (r && r.result !== 'whiff') P.mk.position.copy(V(r.at));
    document.querySelectorAll('#tstrip td[data-s]').forEach(td => td.style.outline = r && +td.dataset.s === r.shift ? '2px solid #16202b' : '');
    if (tip) tip.textContent = Math.abs(ms) > 31 ? `Bat ${ms > 0 ? 'past' : 'before'} the timing window (${ms > 0 ? '+' : ''}${ms.toFixed(0)} ms) -- play or scrub the swing.`
      : r ? `${r.shift > 0 ? r.shift + ' ms early' : r.shift < 0 ? -r.shift + ' ms late' : 'On time'}: ${r.result === 'whiff' ? 'misses it' : `${r.ev} mph, ${r.la}°, spray ${r.spray}° → ${sgn(r.runs)} runs`} (● on the bat)` : '';
  } else if (tip) {
    tip.textContent = k < 0 ? 'Bat outside the contact window -- play or scrub the swing to watch the barrel light up.'
      : `${(t * 1000) > 0 ? '+' : ''}${(t * 1000).toFixed(0)} ms from his contact: the barrel as it is right now, priced if the pitch met each spot.`;
  }
}
function setTab(t) {
  TAB = t; paintOff(); if (R3) R3.preset('Catcher'); document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.t === t));
  const lay = document.getElementById('lay'), lg = document.getElementById('lg');
  if (t === 'Swing' || !LY) { lay.hidden = true; if (lg) lg.hidden = false; clearOverlay(); fit(); return; }
  lay.hidden = false; if (lg) lg.hidden = true;
  ({Contact: showContact, Timing: showTiming, Arsenal: showArsenal})[t](); fit();
}
if (LY) {
  const tb = document.getElementById('tabs');
  ['Swing', 'Contact', 'Timing', 'Arsenal'].forEach(t => { const b = document.createElement('button'); b.textContent = t; b.dataset.t = t;
    if (t === 'Swing') b.classList.add('on'); b.onclick = () => setTab(t); tb.appendChild(b); });
}
"""
