"""Infield / Outfield AAR page: tiles (hover = season / last 30 days / MLB), a field chart of every ball that was his,
the selected play's read-out, a sortable play table (double-click = video), double plays (infield) or arm
(outfield), and what's measured / modeled. No starting spots or routes -- MLB doesn't publish them."""
from __future__ import annotations

import json
import math
from html import escape

import numpy as np
import pandas as pd

from core.aar import SAVANT_VIDEO

TEAL = "#4a8b96"
ROW_KEYS = ["n", "inning", "half", "outs", "on", "batter", "pitcher", "stand", "ev", "la", "bb", "dir", "dist", "hx",
            "hy", "result", "des", "pos", "p_out", "share", "made", "opa", "runs", "rng", "conv", "stars", "kind", "video"]

CSS_EXTRA = """
#fa{font:13px/1.4 Inter,system-ui,-apple-system,sans-serif;color:#16202b}
#fa .top{display:grid;grid-template-columns:330px 1fr;gap:12px;margin-top:10px}
#fa .card{background:#fff;border:1px solid #e6eaed;border-radius:12px;padding:12px 14px}
#fa .lab{font-size:11px;color:#6a7681;text-transform:uppercase;letter-spacing:.4px;font-weight:600}
#fa .tiles{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
#fa .tile{border-radius:10px;padding:8px 10px;border:1px solid #e6eaed;position:relative;cursor:default}
#fa .tile .k{font-size:11px;color:#33404c} #fa .tile .v{font:700 20px ui-monospace,Menlo,monospace;margin-top:2px}
#fa .tile .c{font-size:10.5px;color:#33404c;opacity:.8;margin-top:2px}
#fa .pop{display:none;position:absolute;left:30%;top:calc(100% - 6px);background:#16202b;color:#fff;border-radius:8px;
  padding:7px 9px;font-size:11px;width:230px;box-shadow:0 6px 18px rgba(0,0,0,.18);z-index:5}
#fa .tile:hover .pop{display:block} #fa .pop b{color:#9fd3da} #fa .pop table{width:100%;border-collapse:collapse;margin-top:3px}
#fa .pop td{padding:1px 0} #fa .pop td:last-child{text-align:right;font-family:ui-monospace,Menlo,monospace}
#fa .charts{display:grid;grid-template-columns:1.15fr 1fr;gap:14px}
#fa h4{margin:0 0 4px;font-size:13px} #fa .s{color:#6a7681;font-size:11.5px;margin-bottom:6px;line-height:1.45}
#fa .leg{display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:#6a7681;margin-top:4px;align-items:center}
#fa .gbar{height:8px;width:120px;border-radius:4px;background:linear-gradient(90deg,#d63328,#f2c01e,#2e9d5b)}
#fa table.t{width:100%;border-collapse:collapse;margin-top:6px;font-size:12.5px}
#fa table.t th{font-size:10.5px;color:#6a7681;text-align:right;padding:6px;border-bottom:1px solid #e6eaed;font-weight:600;
  white-space:nowrap;cursor:pointer;user-select:none} #fa table.t th:hover{color:#16202b}
#fa table.t th.asc:after{content:" ▲";font-size:9px} #fa table.t th.desc:after{content:" ▼";font-size:9px}
#fa table.t td{text-align:right;padding:6px;border-bottom:1px solid #f0f2f4;white-space:nowrap}
#fa table.t .l{text-align:left} #fa table.t tr.sel td{background:#eef6f7} #fa table.t tbody tr{cursor:pointer}
#fa .pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;font-weight:600}
#fa .ok{background:#e3f4ea;color:#1f7a45} #fa .no{background:#fbe5e3;color:#b0281e} #fa .na{background:#eef1f3;color:#6a7681}
#fa .mono{font-family:ui-monospace,Menlo,monospace}
#fa .tbar{display:flex;align-items:center;gap:8px;margin-top:12px;flex-wrap:wrap}
#fa .tbar .ttl{font-weight:700;font-size:14px} #fa .faint{color:#9aa5ae;font-size:11.5px}
#fa .chip{border:1px solid #d9dee2;border-radius:14px;padding:3px 10px;font-size:11.5px;background:#fff;cursor:pointer}
#fa .chip.on{border-color:#4a8b96;color:#4a8b96;font-weight:600}
#fa .note{font-size:11.5px;color:#6a7681;margin-top:8px;line-height:1.5}
#fa .two{display:grid;grid-template-columns:1.4fr 1fr;gap:12px;margin-top:12px}
#fa .det table{width:100%;border-collapse:collapse} #fa .det td{padding:4px 0;border-bottom:1px solid #f0f2f4;font-size:12.5px}
#fa .det td:last-child{text-align:right;font-family:ui-monospace,Menlo,monospace}
#fa .det a{color:#4a8b96;font-weight:600;text-decoration:none}
#fa svg .pt{cursor:pointer}
#fa svg text.pn{font:800 13px system-ui;fill:#16202b;stroke:#fff;stroke-width:3px;paint-order:stroke}
#fa svg text.fd{font:500 13px system-ui;fill:#6a7681;stroke:none}
@media (max-width:900px){#fa .top,#fa .charts,#fa .two{grid-template-columns:1fr}}
"""


def _ok(v) -> bool:
    try:
        return v is not None and not (isinstance(v, float) and math.isnan(v)) and v is not pd.NA
    except TypeError:
        return False


def _rows(df: pd.DataFrame | None, keys: list[str] | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    out = []
    for r in df.to_dict("records"):
        out.append({k: (r.get(k) if _ok(r.get(k)) else None) if not isinstance(r.get(k), (bool,)) else bool(r.get(k))
                    for k in (keys or list(r))})
    for r in out:
        for k, v in r.items():
            if hasattr(v, "item"):
                r[k] = v.item()
    return out


def _tile(k: str, v: str, c: str, tone: float | None, comp: list | None = None) -> str:
    bg = "#eef3f6" if tone is None else ("#e3f2e8" if tone > 0.05 else "#fbe9e7" if tone < -0.05 else "#eef3f6")
    pop = ""
    if comp:
        rows = "".join(f"<tr><td>{escape(str(a))}</td><td>{'–' if b is None else (f'{b:+.2f}' if isinstance(b, float) else b)}"
                       f"</td></tr>" for a, b in comp)
        pop = f'<div class="pop"><b>{escape(k)} · per game</b><table>{rows}</table></div>'
    return (f'<div class="tile" style="background:{bg}">{pop}<div class="k">{escape(k)}</div>'
            f'<div class="v">{v}</div><div class="c">{c}</div></div>')


def tiles_html(line: dict, view: str, comps: dict | None) -> str:
    cm = comps or {}
    g = lambda lab, v: (cm.get(lab) and [["This game", round(v, 2)]] + cm[lab]) or None
    s = lambda v: f"{v:+.2f}"
    best = line.get("best")
    best_t = f"{round(best['p_out'] * 100)}%" if best else "–"
    best_c = f"play {best['n']} · {best['inning']}th inning" if best else "no outs made"
    common = [_tile("Outs vs expected", s(line["opa"]), f"{line['plays']} plays · {line['outs']} outs made", line["opa"],
                    g("Outs vs expected", line["opa"]))]
    if view == "Infield":
        t = [_tile("Fielding runs", s(line["runs"]), f"range {line['rng']:+.2f} · finish {line['conv']:+.2f} outs",
                   line["runs"], g("Fielding runs", line["runs"])),
             _tile("Double plays", f"{line['dp_made']} of {line['dp_n']}",
                   f"MLB expects {line['dp_exp']:.1f} · {line['dp_runs']:+.2f} runs" if line["dp_n"] else "no chances",
                   line["dp_runs"] if line["dp_n"] else None, g("Double-play runs", line["dp_runs"])),
             _tile("Total runs saved", s(line["total"]), "fielding + double plays", line["total"],
                   g("Total runs saved", line["total"]))]
    else:
        t = [_tile("Range runs", s(line["runs"]), "outs vs expected × run value", line["runs"],
                   g("Range runs", line["runs"])),
             _tile("Arm runs", s(line["arm_runs"]), f"{line['arm_n']} balls fielded with runners on" if line["arm_n"]
                   else "no runners on when he fielded it", line["arm_runs"] if line["arm_n"] else None,
                   g("Arm runs", line["arm_runs"])),
             _tile("Total runs saved", s(line["total"]), "range + arm", line["total"], g("Total runs saved", line["total"]))]
    tail = [_tile("Hardest out", best_t, best_c + " (out chance)", None),
            _tile("Should-haves", str(line.get("miss", 0)), "his balls (50%+ his) not turned into outs",
                  -1 if line.get("miss") else None)]
    return "".join(common + t + tail)


def field_svg(view: str, dims: dict | None, outline: dict | None = None, park: str = "") -> str:
    """The park the game was played in (home at the bottom; feet): the traced wall, dirt and grass when the outline
    matches this venue (core.aar.park_outline, same as the hitter's spray chart), otherwise straight lines between
    the published fence distances. Points are drawn by the page's script."""
    from ui.aar import LINE_ANGLES, _around_center, _by_spray
    segs = (outline or {}).get("segments") or {}
    P = lambda x, y: f"{x:.1f},{-y:.1f}"
    g, note = [], ""
    if segs:
        fx, fy = _by_spray(*segs["outfield_outer"])
        g.append(f'<polygon points="0,0 {" ".join(P(a, c) for a, c in zip(fx, fy))}" fill="#eef5ef" stroke="#5f8f96" '
                 f'stroke-width="1.6" stroke-linejoin="round"/>')
        if "infield_outer" in segs:
            ix, iy = _by_spray(*segs["infield_outer"])
            g.append(f'<polygon points="0,0 {" ".join(P(a, c) for a, c in zip(ix, iy))}" fill="#f3ece2" stroke="#c9b48f" stroke-width=".8"/>')
        if "infield_inner" in segs:
            ix, iy = _around_center(*segs["infield_inner"])
            g.append(f'<polygon points="{" ".join(P(a, c) for a, c in zip(ix, iy))}" fill="#eef5ef" stroke="#c9b48f" stroke-width=".8"/>')
        for a_, c_ in ((fx[0], fy[0]), (fx[-1], fy[-1])):
            g.append(f'<line x1="0" y1="0" x2="{a_:.1f}" y2="{-c_:.1f}" stroke="#b9c7cb" stroke-width="1"/>')
        deep, wide = float(np.max(fy)), float(np.max(np.abs(fx)))
        dist = np.hypot(fx, fy)
        labels = [(fx[0], fy[0], "start", 6, (dims or {}).get("leftLine")), (fx[-1], fy[-1], "end", -6, (dims or {}).get("rightLine")),
                  (fx[int(np.argmax(dist))], fy[int(np.argmax(dist))], "middle", 0, (dims or {}).get("center"))]
    else:
        d = dims or {"leftLine": 330, "left": 365, "leftCenter": 385, "center": 400, "rightCenter": 385, "right": 365,
                     "rightLine": 325}
        pts = [(d[k] * math.sin(math.radians(a)), d[k] * math.cos(math.radians(a))) for k, a in LINE_ANGLES.items() if k in d]
        g.append(f'<polygon points="0,0 {" ".join(P(a, c) for a, c in pts)}" fill="#eef5ef" stroke="#7fa3a9" '
                 f'stroke-width="1.3" stroke-dasharray="6 4"/>')
        arc = " ".join(P(95 * math.sin(math.radians(a)), 95 * math.cos(math.radians(a)) + 60.5) for a in range(-75, 76, 5))
        g.append(f'<polygon points="0,0 {arc}" fill="#f3ece2" stroke="#ddd2c2" opacity=".9"/>')
        for a_, c_ in (pts[0], pts[-1]):
            g.append(f'<line x1="0" y1="0" x2="{a_:.1f}" y2="{-c_:.1f}" stroke="#b9c7cb"/>')
        deep, wide = max(c for _, c in pts), max(abs(a) for a, _ in pts)
        labels = [(pts[0][0], pts[0][1], "start", 6, d.get("leftLine")), (pts[-1][0], pts[-1][1], "end", -6, d.get("rightLine")),
                  (0, d.get("center", deep), "middle", 0, d.get("center"))]
        note = " (approximate fence -- published distances)"
    dia = "0,0 -63.6,-63.6 0,-127.3 63.6,-63.6"
    g.append(f'<polygon points="{dia}" fill="none" stroke="#c9b48f" stroke-width=".8"/>')
    g += [f'<rect x="{x - 2:.1f}" y="{y - 2:.1f}" width="4" height="4" fill="#fff" stroke="#9aa5ae" stroke-width=".6" '
          f'transform="rotate(45 {x:.1f} {y:.1f})"/>' for x, y in ((-63.6, -63.6), (0, -127.3), (63.6, -63.6))]
    if view == "Outfield":
        for x, y, anchor, dx, v in labels:
            if v:
                g.append(f'<text class="fd" x="{x + dx:.1f}" y="{-y - 7:.1f}" text-anchor="{anchor}">{v:.0f}</text>')
        vb = f"{-wide - 20:.0f} {-deep - 26:.0f} {2 * wide + 40:.0f} {deep + 38:.0f}"
    else:
        vb = "-125 -185 250 197"
    cap = f'<div class="s" style="margin:2px 0 0">{escape(park + note)}</div>' if (park or note) else ""
    return (f'<svg id="fld" viewBox="{vb}" width="100%" preserveAspectRatio="xMidYMax meet">{"".join(g)}'
            f'<g id="pts"></g></svg>{cap}')


JS = r"""
const P = JSON.parse(document.getElementById('plays-data').textContent);
const VIEW = document.getElementById('view-data').textContent.trim();
const VID = 'https://baseballsavant.mlb.com/sporty-videos?playId=';
let SEL = P.length ? (P.find(r => r.made && r.p_out === Math.min(...P.filter(x => x.made).map(x => x.p_out))) || P[0]).n : null;
let FILT = 'all', SORT = {k: 'n', dir: 1};
function col(p) {                           // red (hard / unlikely) -> yellow -> green (routine)
  const S = [[0, [214, 51, 40]], [.5, [242, 192, 30]], [1, [46, 157, 91]]];
  p = Math.max(0, Math.min(1, p ?? 0));
  for (let i = 1; i < S.length; i++) if (p <= S[i][0]) {
    const [a, ca] = S[i - 1], [b, cb] = S[i], f = (p - a) / (b - a);
    return `rgb(${ca.map((c, j) => Math.round(c + (cb[j] - c) * f)).join(',')})`;
  }
  return 'rgb(46,157,91)';
}
const sg = (v, d = 2) => v === null || v === undefined ? '–' : (v > 0.0049 ? '+' : '') + Number(v).toFixed(d);
const pct = v => v === null || v === undefined ? '–' : Math.round(v * 100) + '%';
const miss = r => !r.made && r.p_out * r.share >= 0.5;
const FILTERS = {all: r => true, out: r => r.made, hit: r => !r.made && r.kind === 'Hit', err: r => r.kind === 'Error',
                 miss, tough: r => r.stars >= 3};
const COLS = [['n', '#', 'l'], ['inning', 'Inn', 'l'], ['batter', 'Batter', 'l'], ['ev', 'EV'], ['la', 'LA'],
  ['bb', 'Ball', 'l'], ['dir', 'Direction', 'l'], ['dist', VIEW === 'Infield' ? 'Fielded at' : 'Distance'], ['p_out', 'Out chance'], ['share', 'His share'],
  ['result', 'Result', 'l'], ['opa', 'Outs vs exp.'], ['runs', 'Runs']];
// infield: where the ball was fielded (ft from home, from Statcast's hit coordinates); outfield: flight distance
const fieldedAt = r => r.hx === null || r.hy === null ? null : Math.round(Math.hypot(r.hx, r.hy));
if (VIEW === 'Infield') P.forEach(r => { r.flight = r.dist; r.dist = fieldedAt(r); });
const dcell = r => r.dist ? r.dist + ' ft' : '–';
function table() {
  const rows = P.filter(FILTERS[FILT]).slice().sort((a, b) => {
    const x = a[SORT.k], y = b[SORT.k];
    if (x === y) return 0; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1;
    return (x > y ? 1 : -1) * SORT.dir;
  });
  const th = COLS.map(([k, lab, c]) => `<th data-k="${k}" class="${c || ''} ${SORT.k === k ? (SORT.dir > 0 ? 'asc' : 'desc') : ''}">${lab}</th>`).join('');
  const tr = rows.map(r => {
    const res = `<span class="pill ${r.made ? 'ok' : 'no'}">${r.result}</span>`;
    const c = col(r.p_out);
    return `<tr data-n="${r.n}" class="${r.n === SEL ? 'sel' : ''}"><td class="l">${r.n}</td><td class="l">${r.half}${r.inning}</td>
      <td class="l">${r.batter || ''}</td><td>${r.ev ?? '–'}</td><td>${r.la ?? '–'}°</td><td class="l">${r.bb}</td><td class="l">${r.dir}</td>
      <td>${dcell(r)}</td><td><b style="color:${c};filter:brightness(.8)">${pct(r.p_out)}</b></td>
      <td>${pct(r.share)}</td><td class="l">${res}</td>
      <td class="mono" style="color:${r.opa > 0.005 ? '#1f7a45' : r.opa < -0.005 ? '#b0281e' : '#6a7681'}">${sg(r.opa)}</td>
      <td class="mono">${sg(r.runs)}</td></tr>`;
  }).join('');
  document.getElementById('tbl').innerHTML = `<thead><tr>${th}</tr></thead><tbody>${tr || '<tr><td class="l" colspan="13" style="color:#9aa5ae">No plays in this filter.</td></tr>'}</tbody>`;
  document.querySelectorAll('#tbl th').forEach(h => h.onclick = () => {
    const k = h.dataset.k; SORT = SORT.k === k ? {k, dir: -SORT.dir} : {k, dir: ['opa', 'runs', 'ev', 'dist'].includes(k) ? -1 : 1}; table(); });
  document.querySelectorAll('#tbl tbody tr[data-n]').forEach(t => {
    t.onclick = () => pick(+t.dataset.n);
    t.ondblclick = () => { const r = P.find(x => x.n === +t.dataset.n); if (r && r.video) window.open(VID + r.video, '_blank'); };
  });
  document.getElementById('shown').textContent = `${rows.length} of ${P.length} plays`;
}
function points() {
  const g = document.getElementById('pts'); if (!g) return;
  g.innerHTML = P.filter(FILTERS[FILT]).filter(r => r.hx !== null && r.hy !== null).map(r => {
    // infield chart: a ball that got through is drawn where it left the infield (its direction, 165 ft out)
    const d = Math.hypot(r.hx, r.hy), k = VIEW === 'Infield' && d > 165 ? 165 / d : 1;
    const x = r.hx * k, y = -r.hy * k, c = col(r.p_out), s = r.n === SEL, R = VIEW === 'Outfield' ? 7 : 3.4, w = VIEW === 'Outfield' ? 1 : 0.5;
    const mark = r.made ? `<circle cx="${x}" cy="${y}" r="${R}" fill="${c}" stroke="#16202b" stroke-width="${s ? w * 2.4 : w}"/>`
      : `<g stroke="${c}" stroke-width="${R * 0.45}"><line x1="${x - R}" y1="${y - R}" x2="${x + R}" y2="${y + R}"/><line x1="${x - R}" y1="${y + R}" x2="${x + R}" y2="${y - R}"/></g>` +
        (s ? `<circle cx="${x}" cy="${y}" r="${R * 1.5}" fill="none" stroke="#16202b" stroke-width="${w * 1.6}"/>` : '');
    return `<g class="pt" data-n="${r.n}">${mark}<text class="pn" x="${x + R * 1.2}" y="${y - R * 0.9}" style="font-size:${R * 1.9}px;stroke-width:${R * 0.32}px">${r.n}</text></g>`;
  }).join('');
  g.querySelectorAll('.pt').forEach(e => { e.onclick = () => pick(+e.dataset.n);
    e.ondblclick = () => { const r = P.find(x => x.n === +e.dataset.n); if (r && r.video) window.open(VID + r.video, '_blank'); }; });
}
function detail() {
  const r = P.find(x => x.n === SEL), el = document.getElementById('det');
  if (!r) { el.innerHTML = '<div class="s">No plays for him in this game.</div>'; return; }
  const row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  const why = r.made ? `He made the out: credit = 1 − ${pct(r.p_out)} = <b style="color:#1f7a45">${sg(r.opa)}</b> outs.`
    : `Not turned into an out. Balls like this are outs ${pct(r.p_out)} of the time, and his position makes that out ${pct(r.share)} of the time: credit = −${pct(r.p_out)} × ${pct(r.share)} = <b style="color:#b0281e">${sg(r.opa)}</b> outs.`;
  el.innerHTML = `<h4>Play ${r.n} · ${r.half}${r.inning} · ${r.batter || 'batter'}</h4>
    <div class="s">${r.des || ''}</div>
    <table>${row('Exit velo · launch', `${r.ev ?? '–'} mph · ${r.la ?? '–'}°`)}${row('Ball · direction', `${r.bb} · ${r.dir}`)}
    ${VIEW === 'Infield' ? (r.dist ? row('Fielded at', r.dist + ' ft from home') : '') + (r.flight ? row('Flight distance', r.flight + ' ft') : '')
      : (r.dist ? row('Flight distance', r.dist + ' ft') : '')}${row('Out chance (MLB)', pct(r.p_out))}
    ${row('His position\'s share', pct(r.share))}${row('Difficulty', '★'.repeat(r.stars) + `<span style="color:#d9dee2">${'★'.repeat(5 - r.stars)}</span>`)}
    ${row('Outs vs expected', sg(r.opa))}${row('Runs', sg(r.runs))}
    ${VIEW === 'Infield' && r.made ? row('Range · finish', `${sg(r.rng)} · ${sg(r.conv)}`) : ''}</table>
    <div class="note">${why}</div>${r.video ? `<div class="note"><a href="${VID + r.video}" target="_blank" rel="noopener">▶ Video</a></div>` : ''}`;
}
function pick(n) { SEL = n; table(); points(); detail(); }
document.querySelectorAll('#fchips .chip').forEach(c => c.onclick = () => {
  FILT = c.dataset.f; document.querySelectorAll('#fchips .chip').forEach(x => x.classList.toggle('on', x === c));
  const vis = P.filter(FILTERS[FILT]); if (vis.length && !vis.some(r => r.n === SEL)) SEL = vis[0].n;
  table(); points(); detail(); });
function resize() {
  const h = document.documentElement.scrollHeight + 8;
  window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setFrameHeight', height: h}, '*');
}
table(); points(); detail(); window.addEventListener('load', resize); resize();
"""


def _side_table(view: str, dp: pd.DataFrame | None, arm: pd.DataFrame | None) -> str:
    vid = lambda r: (f' <a href="{SAVANT_VIDEO.format(r["video"])}" target="_blank" rel="noopener" style="color:{TEAL}">▶</a>'
                     if r.get("video") else "")
    if view == "Infield":
        if dp is None or dp.empty:
            body = '<div class="note">No double-play chances (ground ball he fielded, runner on 1st, fewer than 2 out).</div>'
        else:
            tr = "".join(
                f'<tr><td class="l">{"▲" if r["half"] == "▲" else "▼"}{r["inning"]}</td><td class="l">{escape(str(r.get("batter") or ""))}</td>'
                f'<td>{r["ev"]}</td><td class="l">{r["dir"]}</td><td>{round(r["p_dp"] * 100)}%</td>'
                f'<td class="l"><span class="pill {"ok" if r["dp"] else "no"}">{"Turned" if r["dp"] else "Not turned"}</span>{vid(r)}</td>'
                f'<td class="mono" style="color:{"#1f7a45" if r["runs"] > 0 else "#b0281e"}">{r["runs"]:+.2f}</td></tr>'
                for r in _rows(dp))
            body = ('<table class="t"><tr><th class="l">Inn</th><th class="l">Batter</th><th>EV</th><th class="l">Direction</th>'
                    f'<th>MLB DP rate</th><th class="l">What happened</th><th>Runs</th></tr>{tr}</table>'
                    '<div class="note">Double-play chances: ground balls he fielded with a runner on 1st and fewer than 2 out. '
                    'Runs = (turned or not − how often MLB turns that ball) × what a double play is worth.</div>')
        return f'<div class="card"><div class="lab">Double plays</div>{body}</div>'
    if arm is None or arm.empty:
        body = '<div class="note">No balls fielded with runners on.</div>'
    else:
        tr = "".join(
            f'<tr><td class="l">{r["half"]}{r["inning"]}</td><td class="l">{escape(str(r["result"]))}</td><td class="l">{r["on"]}</td>'
            f'<td class="l" style="white-space:normal;max-width:330px;font-size:11.5px;color:#33404c">{escape(str(r.get("des") or ""))}{vid(r)}</td>'
            f'<td class="mono" style="color:{"#1f7a45" if r["runs"] > 0.005 else "#b0281e" if r["runs"] < -0.005 else "#6a7681"}">{r["runs"]:+.2f}</td></tr>'
            for r in _rows(arm))
        body = ('<table class="t"><tr><th class="l">Inn</th><th class="l">Play</th><th class="l">On base</th>'
                f'<th class="l">What happened</th><th>Runs</th></tr>{tr}</table>'
                '<div class="note">Balls he fielded with runners on. Runs = what MLB runners usually get out of that ball '
                '(same ball, base / out state) − what they got here: runners held, extra bases taken, outfield assists.</div>')
    return f'<div class="card"><div class="lab">Arm · runners on balls he fielded</div>{body}</div>'


def component_html(c: dict, line: dict, view: str, header: str = "", comps: dict | None = None,
                   dims: dict | None = None, has_model: bool = True, outline: dict | None = None, park: str = "") -> str:
    from ui.aar import CSS
    plays = _rows(c.get("plays"), ROW_KEYS)
    chips = "".join(f'<span class="chip{" on" if k == "all" else ""}" data-f="{k}">{v}</span>' for k, v in
                    (("all", "All"), ("out", "Outs he made"), ("hit", "Hits"), ("err", "Errors"),
                     ("miss", "Should-haves"), ("tough", "3★+ plays")))
    note = ("" if has_model else '<div class="note">The fielding model isn\'t trained on this computer yet '
            '(<code>python -m pipeline.train_fielding</code>) -- the plays fill in once it is.</div>')
    left = (f'<div class="card"><div class="lab">{view} · this game <span class="faint">· hover a tile to compare</span></div>'
            f'<div class="tiles">{tiles_html(line, view, comps)}</div>'
            f'<div class="note">Credit per play: out made by him = 1 − out chance; a hit or error = −(out chance × his '
            f'position\'s share). Runs = credit × what an out is worth vs a hit on that kind of ball.</div>{note}</div>')
    title = "Every ball that was his" if view == "Outfield" else "Every ball that was his (infield)"
    charts = (f'<div class="card"><div class="charts"><div><h4>{title}</h4><div class="s">Where each ball landed or was '
              f'picked up. ● out he made · ✕ a hit or error that was partly his -- coloured by how often MLB makes the out. '
              f'Click a play (here or in the table); double-click for the video.</div>{field_svg(view, dims, outline, park)}'
              f'<div class="leg"><span>hard</span><span class="gbar"></span><span>routine</span></div></div>'
              f'<div class="det" id="det"></div></div></div>')
    bar = (f'<div class="tbar"><span class="ttl">Plays</span><span id="fchips">{chips}</span>'
           f'<span class="faint" id="shown"></span><span class="faint">· click a column to sort · double-click a play '
           f'for the video</span></div>')
    src = ('<div class="card"><div class="lab">Where it comes from</div><div class="note">'
           '<b>Measured</b>: exit velo, launch angle, direction, flight distance, who fielded it and what happened '
           '(Statcast); names + video (MLB game feed).<br><b>Modeled</b> (IQ fielding model, trained on every MLB ball in '
           'play): how often a ball like this is an out, which position usually makes it, double-play and runner '
           'expectations -- the same model as the profile\'s Defense tab, so his games add up to his season there.<br>'
           '<b>Not used</b>: where he started or the route he ran -- MLB doesn\'t publish them, so nothing here assumes '
           'them. Every ball is judged by the ball itself.</div></div>')
    side = _side_table(view, c.get("dp"), c.get("arm"))
    data = json.dumps(plays).replace("</", "<\\/")
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}{CSS_EXTRA}</style></head><body>'
            f'<div id="fa">{header}<div class="top">{left}{charts}</div>{bar}'
            f'<div class="card" style="margin-top:6px"><table class="t" id="tbl"></table></div>'
            f'<div class="two">{side}{src}</div></div>'
            f'<script type="application/json" id="plays-data">{data}</script>'
            f'<script type="application/json" id="view-data">{view}</script>'
            f'<script>{JS}</script></body></html>')


def estimate_height(n: int) -> int:
    return 980 + 36 * n
