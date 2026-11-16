"""Point-of-contact map for the Bat Path tab (top-down on the plate, catcher's view)."""
from __future__ import annotations

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
.bp th:first-child,.bp td:first-child{text-align:left}
.bp td{padding:5px 9px;border-bottom:1px solid #e6eaed;text-align:right;font-family:ui-monospace,Menlo,monospace}
</style>"""


def _ev_color(ev) -> str:
    """Blue (soft) -> white (~88) -> red (hard)."""
    if ev is None or pd.isna(ev):
        return "#9aa4ad"
    t = float(np.clip((ev - 88) / 18, -1, 1))
    if t >= 0:
        r, g, b = 255, int(255 - 200 * t), int(255 - 210 * t)
    else:
        r, g, b = int(255 + 205 * t), int(255 + 130 * t), 255
    return f"#{r:02x}{g:02x}{b:02x}"


def poc_map_html(d: pd.DataFrame, color_by: str = "Exit Velo", estimated: bool = False) -> str:
    """d: one row per contact with pf, pxc, ev, la, result, pgroup, bip, date, ptype."""
    have = d[d["pf"].notna() & d["pxc"].notna()]
    if have.empty:
        return CSS + '<div class="bp"><div class="card">No measured contact points for this selection.</div></div>'
    W, H, S = 420, 470, 6.0
    xs = lambda v: W / 2 + v * S
    ys = lambda v: 200 - v * S                                    # +30" out front at top, -40" behind at bottom
    g = []
    for v in range(-40, 35, 5):
        g.append(f'<line x1="30" y1="{ys(v):.1f}" x2="{W - 8}" y2="{ys(v):.1f}" stroke="#eef1f3" '
                 f'stroke-dasharray="{"" if v == 0 else "3 3"}"/><text x="25" y="{ys(v) + 3:.1f}" class="ax" '
                 f'text-anchor="end">{v}</text>')
    for v in range(-25, 30, 5):
        g.append(f'<text x="{xs(v):.1f}" y="{H - 6}" class="ax" text-anchor="middle">{v}</text>')
    h = PLATE_HALF
    g.append(f'<polygon points="{xs(-h):.1f},{ys(0):.1f} {xs(h):.1f},{ys(0):.1f} {xs(h):.1f},{ys(-h):.1f} '
             f'{xs(0):.1f},{ys(-17):.1f} {xs(-h):.1f},{ys(-h):.1f}" fill="#f6f8f9" stroke="#0e7c86" stroke-width="1.8"/>')
    g.append(f'<text x="{W - 10}" y="{ys(30) + 10:.1f}" class="ax" text-anchor="end">toward the pitcher ↑</text>')
    fouls, inplay = have[~have["bip"]], have[have["bip"]]
    for part, hollow in ((fouls, True), (inplay, False)):
        for r in part.itertuples():
            if color_by == "Pitch Group":
                col = GROUP_COLOR.get(r.pgroup, "#9aa4ad")
            else:
                col = "#9aa4ad" if hollow else _ev_color(r.ev)
            x, y = float(np.clip(xs(r.pxc), 8, W - 8)), float(np.clip(ys(r.pf), 8, H - 18))
            tip = (f"{r.date} · {r.ptype} · {r.result}" + (f" · {r.ev:.0f} mph, {r.la:.0f}°" if not pd.isna(r.ev) else "")
                   + f" · {r.pf:+.1f}\" front, {r.pxc:+.1f}\" side")
            style = (f'fill="none" stroke="{col}" stroke-width="1.4"' if hollow and color_by != "Pitch Group"
                     else f'fill="{col}" fill-opacity=".85" stroke="#33404c" stroke-width=".6"')
            g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{4 if hollow else 5}" {style}><title>{escape(tip)}</title></circle>')
    for grp, gg in have.groupby("pgroup"):                        # average point by pitch group
        if len(gg) < 3:
            continue
        x, y, col = xs(gg["pxc"].mean()), ys(gg["pf"].mean()), GROUP_COLOR.get(grp, "#33404c")
        g.append(f'<g stroke="#16202b" stroke-width="3"><line x1="{x - 8:.1f}" y1="{y:.1f}" x2="{x + 8:.1f}" '
                 f'y2="{y:.1f}"/><line x1="{x:.1f}" y1="{y - 8:.1f}" x2="{x:.1f}" y2="{y + 8:.1f}"/></g>'
                 f'<g stroke="{col}" stroke-width="1.6"><line x1="{x - 7:.1f}" y1="{y:.1f}" x2="{x + 7:.1f}" '
                 f'y2="{y:.1f}"/><line x1="{x:.1f}" y1="{y - 7:.1f}" x2="{x:.1f}" y2="{y + 7:.1f}"/></g>')
    how = ("<b>*</b> front to back estimated -- not on Savant's stance board yet. " if estimated else "") + \
        "Front to back = MLB's contact point vs the plate (Statcast's contact from his hips − his stance depth); " \
        "side to side = where the ball was. "
    key = ("dots = pitch group (fouls and balls in play)" if color_by == "Pitch Group" else
           "filled = in play, colored by exit velo (blue soft → red hard); hollow = foul")
    legend = " ".join(f'<span style="color:{c}">✚</span> {k} avg' for k, c in GROUP_COLOR.items())
    chart = (f'<div class="card"><div class="t">Point of contact{"*" if estimated else ""} '
             f'<span class="faint">· {len(have)} contacts · inches · catcher\'s view</span></div>'
             f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}">{"".join(g)}</svg>'
             f'<div class="leg">{how}{key}. {legend}. Hover a dot for the pitch.</div></div>')
    return CSS + f'<div class="bp">{chart}{_summary(have)}</div>'


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
#lg i{font-style:normal;font-weight:800}
"""


def path3d_html(scene: dict, title: str, sub: str) -> str:
    """Self-contained page: the 3D view of one scene, square, filling its frame's width."""
    import json

    from ui.aar import S3D_CORE, THREE_JS
    data = json.dumps(scene).replace("</", "<\\/")
    ctl = ('<div id="ctl"><button id="s3p">❚❚</button><select id="s3sp"><option value="4">1/4 speed</option>'
           '<option value="10" selected>1/10</option><option value="25">1/25</option><option value="60">1/60</option>'
           '</select><span class="scw"><input id="s3sc" type="range" min="0" max="1000" value="0"><span id="s3tickbp" '
           'class="tick bp" title="Bat parallel (inferred)"></span><span id="s3tick" class="tick" title="Contact">'
           '</span></span><button id="s3bp">Bat parallel</button><button id="s3k">Contact</button>'
           '<span class="clk" id="s3c"></span></div>')
    legend = ('<div id="lg"><i style="color:#e07b24">━</i> modeled pitch &nbsp; <i style="color:#2f5fb3">━</i> his '
              'average bat path, last 40 ms before contact (from his measured swings) &nbsp; <i style="color:#7b5cc4">┅ ●'
              '</i> back to bat parallel + follow-through, inferred from his tilt, attack angle + direction &nbsp; '
              '<i style="color:#c0392b">●</i> his average contact point · drag to orbit, scroll to zoom</div>')
    js = """
function fit() {
  const h = document.documentElement.scrollHeight + 4;
  window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setFrameHeight', height: h}, '*');
}
const SC = JSON.parse(document.getElementById('s3d-data').textContent);
let lastW = 0;
function start() {
  const w = stage.clientWidth; if (Math.abs(w - lastW) < 8) return; lastW = w;
  document.getElementById('s3msg').style.display = 'none';
  if (R3) R3.stop(); R3 = build(SC); fit();
}
loadThree(start);
window.addEventListener('resize', () => { if (window.THREE) start(); });
window.addEventListener('load', fit); fit();
"""
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{_S3D_CSS}</style></head><body>'
            f'<div id="wrap"><div id="hd"><b>{escape(title)}</b><span>{escape(sub)}</span></div>'
            f'<div id="stage"><div id="cams"></div><div class="msg" id="s3msg">Loading 3D…</div></div>{ctl}{legend}</div>'
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
