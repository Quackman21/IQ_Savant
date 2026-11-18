"""ABS tab on the player profile: a Core-4-style summary (coloured vs the MLB average for that role) and one
interactive zone: every challenge he made as a hot / cold map or as the pitches themselves, Successful / Failed /
Not counted toggles, catcher's or pitcher's view, click to inspect, double-click to play the video."""
from __future__ import annotations

import json
import math
from html import escape

import pandas as pd

from core import abs_stats
from ui.aar import INK, LINE, MUTED, PT_COLOR, TEAL
from ui.hitting import GREEN, RED, _mix

FMT = {"G": "{:.0f}", "Challenges": "{:.0f}", "Chal/G": "{:.2f}", "Overturned": "{:.0f}", "Failed": "{:.0f}",
       "Overturn %": "pct", "ABS Runs": "{:+.2f}", "Challenge cost": "{:+.2f}", "Net ABS Runs": "{:+.2f}", "Net/G": "{:+.3f}",
       "Not counted": "{:.0f}"}
TIPS = {"Challenges": "Challenges that count (not counted ones left out)",
        "Chal/G": "Challenges per game he played in this role",
        "Overturn %": "Share of his counted challenges that were overturned",
        "ABS Runs": "Runs the overturns were worth to his side (a ball made a strike, or the reverse, in that count)",
        "Challenge cost": "What his failed challenges cost: each one = the average value of the missed calls against "
                          "his team that game (blended toward the season average). Free on the last pitch.",
        "Net ABS Runs": "ABS Runs + Challenge cost -- what his challenging was worth overall",
        "Net/G": "Net ABS runs per game in this role",
        "Not counted": "Last pitch of the game, the call stood, and it was right by more than 3 inches -- "
                       "a free challenge burned on a pitch that wasn't close"}


def _fmt(k, v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    f = FMT.get(k, "{}")
    if f == "pct":
        return f"{100 * v:.0f}%"
    s = f.format(v)
    return s.replace("+0.000", "0.000").replace("-0.000", "0.000").replace("+0.00", "0.00").replace("-0.00", "0.00")


def _zval(k, v, ref) -> float:
    if not ref or v is None or (isinstance(v, float) and math.isnan(v)):
        return math.nan
    avg, sd = ref
    if avg is None or sd is None or (isinstance(sd, float) and (math.isnan(sd) or sd == 0)):
        return math.nan
    return (v - avg) / sd * abs_stats.GRADED.get(k, 1)


def _color(k, v, ref) -> str:
    """Same colouring as the Hitting / Core 4 tables: green = better, red = worse, deeper = more STDEVS off."""
    if k not in abs_stats.GRADED:
        return ""
    z = _zval(k, v, ref)
    if math.isnan(z):
        return ""
    t = min(abs(z) / 2.0, 1.0) * 0.85
    return f"background:{_mix(GREEN if z > 0 else RED, t)};color:#16202b"


# the column groups (separators like Core 4)
GROUPS = [("Volume", ["G", "Challenges", "Chal/G"]), ("Results", ["Overturned", "Failed", "Overturn %"]),
          ("Value", ["ABS Runs", "Challenge cost", "Net ABS Runs", "Net/G"]), ("", ["Not counted"])]
LEAD = ["Year", "Period", "Type", "Team"]


def _popup(k, v, ref, who: str, season: str, role: str) -> str:
    avg = (ref or (math.nan, math.nan))[0]
    z = _zval(k, v, ref)
    delta = (v - avg) if (v is not None and avg is not None and not (isinstance(avg, float) and math.isnan(avg))
                          and not (isinstance(v, float) and math.isnan(v))) else math.nan
    dtxt = "-" if isinstance(delta, float) and math.isnan(delta) else (
        f"{100 * delta:+.0f}%" if FMT.get(k) == "pct" else _fmt(k, delta))
    ztxt = "-" if math.isnan(z) else f"{z:.2f}"
    return (f'<div class="pop"><small>Click elsewhere to dismiss</small><table>'
            f'<tr><th></th><th>{escape(k)}</th><th>STDEVS</th></tr>'
            f'<tr><td>{escape(season)} MLB AVG ({escape(role.lower())})</td><td class="num">{_fmt(k, avg)}</td><td></td></tr>'
            f'<tr><td>{escape(who)}</td><td class="num">{_fmt(k, v)}</td><td></td></tr>'
            f'<tr><td>Delta</td><td class="num">{dtxt}</td><td class="num" style="{_color(k, v, ref)}">{ztxt}</td></tr>'
            f'</table></div>')


def summary_html(rows: list[tuple[str, dict]], ref: dict, role: str, who: str = "", teams: dict | None = None,
                 show_avg: bool = True) -> str:
    """Core-4-style table: rows = [(season, summarize() dict)], ref = league()[role] (latest season)."""
    from ui.core4 import SEASON_CSS
    from ui.hitting import CSS as HT_CSS
    teams = teams or {}
    keys = [k for _, ks in GROUPS for k in ks]
    firsts = {ks[0] for _, ks in GROUPS}
    n = len(keys)
    grp = '<tr>' + f'<th class="grp" colspan="{len(LEAD)}"></th>' + "".join(
        f'<th class="grp sep" colspan="{len(ks)}">{escape(g)}</th>' for g, ks in GROUPS) + "</tr>"
    head = "<tr>" + "".join(f'<th style="text-align:left;cursor:default">{h}</th>' for h in LEAD)
    head += "".join(f'<th class="{"r " if i > n / 2 else ""}{"sep" if k in firsts else ""}" '
                    f'data-tip="{escape(TIPS.get(k, k))}">{escape(k)}</th>' for i, k in enumerate(keys)) + "</tr>"
    body = ""
    for lab, v in rows:
        body += ('<tr class="season">' + f'<td class="lbl">{escape(lab)}</td><td class="lbl">Season</td>'
                 f'<td class="lbl">Reg</td><td class="lbl">{escape(teams.get(lab, ""))}</td>')
        for k in keys:
            sep = " sep" if k in firsts else ""
            if k in abs_stats.GRADED:
                body += (f'<td class="c{sep}" tabindex="0" style="{_color(k, v.get(k), ref.get(k))}">'
                         f'{_fmt(k, v.get(k))}{_popup(k, v.get(k), ref.get(k), who, lab, role)}</td>')
            else:
                body += f'<td class="cnt{sep}">{_fmt(k, v.get(k))}</td>'
        body += "</tr>"
    if ref and show_avg:
        body += ('<tr class="avg"><td class="lbl">MLB AVG</td><td class="lbl">'
                 f'{ {"Hitting": "Hitters", "Catching": "Catchers", "Pitching": "Pitchers"}[role]}</td>'
                 '<td class="lbl"></td><td class="lbl"></td>' + "".join(
                     f'<td class="{"sep" if k in firsts else ""}">{_fmt(k, (ref.get(k) or (None,))[0])}</td>'
                     for k in keys) + "</tr>")
    note = ('<div class="ht-note">Coloured by STDEVS vs MLB ' + role.lower() + ' challenges (green = better, red = '
            'worse); click a coloured number for MLB AVG / Delta / STDEVS. Counting stats on the MLB AVG row are per '
            'player. Not counted = last pitch of the game, the call stood, and it was right by 3"+. Challenge cost = '
            'each failed challenge costs the average missed call against his team that game (blended to the '
            'season).</div>')
    fix = ('<style>.ht table.pin .pop td,.ht table.pin .pop th{position:static;min-width:0;max-width:none;'
           'box-shadow:none;white-space:nowrap}.ht tr.season .pop td,.ht tr.season .pop th{border:1px solid #e3e7ea;'
           'font-weight:600}</style>')
    return HT_CSS + SEASON_CSS + fix + f'<div class="ht"><table class="pin">{grp}{head}{body}</table></div>{note}'


# ----------------------------------------------------------------------------- the zone
def zone_rows(t: pd.DataFrame) -> list[dict]:
    from core.hit_metrics import pitch_group
    out = []
    pg = pitch_group(t) if not t.empty else pd.Series(dtype=object)
    for i, r in enumerate(t.to_dict("records")):
        g = lambda k: (None if r.get(k) is None or (isinstance(r.get(k), float) and math.isnan(r.get(k))) else r.get(k))
        res = "Not counted" if not bool(r.get("counted")) else r.get("result")
        out.append({"i": i + 1, "date": str(r.get("game_date"))[:10], "game": f'{g("away_team") or ""} @ {g("home_team") or ""}',
                    "inn": f'{"▲" if str(r.get("inning_topbot")) == "Top" else "▼"}{int(r["inning"]) if g("inning") else ""}',
                    "count": f'{int(g("balls") or 0)}-{int(g("strikes") or 0)}', "pt": g("pitch_type") or "",
                    "grp": (pg.iloc[i] if len(pg) else None) or "", "velo": g("release_speed"),
                    "px": g("plate_x"), "pz": g("plate_z"), "top": g("sz_top"), "bot": g("sz_bot"),
                    "res": res, "from": g("from_call") or "", "to": g("to_call") or "", "margin": g("margin_in"),
                    "runs": g("runs"), "cost": g("cost"), "net": g("net"), "final": bool(r.get("final_pitch")),
                    "video": (str(r.get("play_id")) if g("play_id") and str(r.get("play_id")) not in ("nan", "None") else "")})
    return out


JS = r"""
const R = JSON.parse(document.getElementById('d').textContent);
const PT = JSON.parse(document.getElementById('pt').textContent);
const on = {Successful: true, Failed: true, 'Not counted': false};
let view = 'catcher', mode = 'heat', sel = null;
const COL = {Successful: '#2e9d5b', Failed: '#c0392b', 'Not counted': '#9aa5ae'};
const ok = v => v !== null && v !== undefined && v === v;
const med = a => { const s = a.filter(ok).sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
const TOP = med(R.map(r => r.top)) || 3.4, BOT = med(R.map(r => r.bot)) || 1.6;
const W = 360, H = 400, X = x => W / 2 + (view === 'pitcher' ? -x : x) * 70, Z = z => 340 - (z - 0.5) * 70;
function vis() { return R.filter(r => on[r.res] && ok(r.px) && ok(r.pz)); }
function shape(r, x, y, s) {
  const st = r.res === 'Not counted' ? `fill="#fff" stroke="${COL[r.res]}" stroke-width="2"` : `fill="${COL[r.res]}" stroke="#fff" stroke-width="1"`;
  if (r.grp === 'Breaking') return `<polygon points="${x},${y - s * 1.15} ${x - s},${y + s * .8} ${x + s},${y + s * .8}" ${st}/>`;
  if (r.grp === 'Offspeed') return `<rect x="${x - s * .85}" y="${y - s * .85}" width="${s * 1.7}" height="${s * 1.7}" ${st}/>`;
  return `<circle cx="${x}" cy="${y}" r="${s}" ${st}/>`;
}
function heat(pts) {
  if (!pts.length) return '';
  const cell = 10, sig = 26, out = [];
  let mx = 0; const g = [];
  for (let y = 0; y < H; y += cell) for (let x = 0; x < W; x += cell) {
    let d = 0; for (const p of pts) { const dx = X(p.px) - x - cell / 2, dy = Z(p.pz) - y - cell / 2; d += Math.exp(-(dx * dx + dy * dy) / (2 * sig * sig)); }
    g.push([x, y, d]); mx = Math.max(mx, d);
  }
  for (const [x, y, d] of g) {
    const t = d / mx; if (t < 0.06) continue;
    // cold -> hot: blue, white, red
    const c = t < .5 ? [Math.round(70 + 370 * t), Math.round(120 + 260 * t), 255] : [255, Math.round(255 - 330 * (t - .5)), Math.round(255 - 400 * (t - .5))];
    out.push(`<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="rgb(${c.map(v => Math.max(0, Math.min(255, v))).join(',')})" opacity="${(0.25 + 0.6 * t).toFixed(2)}"/>`);
  }
  return out.join('');
}
function draw() {
  const pts = vis(), hw = 17 / 24, mid = (TOP + BOT) / 2, hh = (TOP - BOT) / 2;
  const box = (k, dash, w) => `<rect x="${X(-hw * k) < X(hw * k) ? X(-hw * k) : X(hw * k)}" y="${Z(mid + hh * k)}" width="${Math.abs(X(hw * k) - X(-hw * k))}" height="${Z(mid - hh * k) - Z(mid + hh * k)}" fill="none" stroke="${k === 1 ? '#4a8b96' : '#9fb9be'}" stroke-width="${w}" stroke-dasharray="${dash}"/>`;
  let s = (mode === 'heat' ? heat(pts) : '') + box(1.33, '2 3', 1) + box(1, '', 2) + box(0.67, '5 4', 1);
  // the plate (point toward the catcher in his view, toward the pitcher in the pitcher's)
  const py = 368, pd = view === 'pitcher' ? -1 : 1;
  s += `<polygon points="${X(-hw)},${py} ${X(hw)},${py} ${X(hw)},${py + 6 * pd} ${X(0)},${py + 14 * pd} ${X(-hw)},${py + 6 * pd}" fill="#f1f4f5" stroke="#9fb9be"/>`;
  const left = view === 'pitcher' ? '1B side' : '3B side', right = view === 'pitcher' ? '3B side' : '1B side';
  s += `<text x="8" y="${H - 6}" class="ax">← ${left}</text><text x="${W - 8}" y="${H - 6}" class="ax" text-anchor="end">${right} →</text>`;
  if (mode === 'points') for (const r of pts) {
    const x = Math.max(8, Math.min(W - 8, X(r.px))), y = Math.max(8, Math.min(H - 30, Z(r.pz)));
    s += `<g class="pt${sel === r.i ? ' sel' : ''}" data-i="${r.i}">${shape(r, x, y, 8)}<text x="${x}" y="${y + 2.8}" class="n">${r.i}</text></g>`;
  }
  document.getElementById('z').innerHTML = s;
  document.querySelectorAll('g.pt').forEach(g => { g.onclick = () => pick(+g.dataset.i); g.ondblclick = () => play(+g.dataset.i); });
  const n = pts.length, sc = pts.filter(r => r.res === 'Successful').length, f = pts.filter(r => r.res === 'Failed').length;
  document.getElementById('cnt').textContent = `${n} shown · ${sc} successful · ${f} failed`;
  table();
}
function fmtRuns(v) { return ok(v) ? (v > 0.0005 ? '+' : '') + v.toFixed(3) : '–'; }
function table() {
  const rows = R.filter(r => on[r.res]);
  document.getElementById('tb').innerHTML = rows.map(r => `<tr data-i="${r.i}" class="${sel === r.i ? 'sel' : ''}"><td>${r.i}</td>` +
    `<td>${r.video ? `<a href="https://baseballsavant.mlb.com/sporty-videos?playId=${r.video}" target="_blank" rel="noopener">▶</a>` : '–'}</td>` +
    `<td>${r.date}</td><td>${r.game}</td><td>${r.inn}</td><td>${r.count}</td><td><b style="color:${PT[r.pt] || '#16202b'}">${r.pt}</b></td>` +
    `<td>${ok(r.velo) ? r.velo.toFixed(1) : '–'}</td><td>${r.from} → ${r.to}${r.final ? ' <span class="faint">(last pitch)</span>' : ''}</td>` +
    `<td><span class="chip" style="background:${COL[r.res]}22;color:${COL[r.res]}">${r.res}</span></td>` +
    `<td>${ok(r.margin) ? (r.margin > 0 ? r.margin.toFixed(1) + '″ out' : (-r.margin).toFixed(1) + '″ in') : '–'}</td><td>${fmtRuns(r.runs)}</td><td>${ok(r.cost) && r.cost ? fmtRuns(-r.cost) : '–'}</td><td><b>${fmtRuns(r.net)}</b></td></tr>`).join('') ||
    '<tr><td colspan="14" class="faint" style="text-align:center;padding:14px">No challenges</td></tr>';
  document.querySelectorAll('#tb tr[data-i]').forEach(tr => { tr.onclick = e => { if (e.target.closest('a')) return; pick(+tr.dataset.i); };
    tr.ondblclick = e => { if (e.target.closest('a')) return; window.getSelection().removeAllRanges(); play(+tr.dataset.i); }; });
}
function pick(i) { sel = sel === i ? null : i; if (sel && mode === 'heat') setMode('points'); else draw();
  const tr = document.querySelector(`#tb tr[data-i="${sel}"]`); if (tr) tr.scrollIntoView({block: 'nearest'}); }
function play(i) { const r = R.find(x => x.i === i); if (r && r.video) window.open('https://baseballsavant.mlb.com/sporty-videos?playId=' + r.video, '_blank', 'noopener'); }
function setMode(m) { mode = m; document.querySelectorAll('#mode button').forEach(b => b.classList.toggle('on', b.dataset.v === m)); draw(); }
document.querySelectorAll('#mode button').forEach(b => b.onclick = () => setMode(b.dataset.v));
document.querySelectorAll('#view button').forEach(b => b.onclick = () => { view = b.dataset.v;
  document.querySelectorAll('#view button').forEach(x => x.classList.toggle('on', x === b)); draw(); });
document.querySelectorAll('.rc').forEach(b => b.onclick = () => { on[b.dataset.v] = !on[b.dataset.v]; b.classList.toggle('on', on[b.dataset.v]); draw(); });
draw();
"""


def zone_html(t: pd.DataFrame, role: str, default_view: str = "catcher") -> str:
    rows = zone_rows(t)
    data = json.dumps(rows).replace("</", "<\\/")
    nc = sum(r["res"] == "Not counted" for r in rows)
    css = f"""
*{{box-sizing:border-box}} body{{margin:0;font:13px system-ui,sans-serif;color:{INK}}}
.wrap{{display:grid;grid-template-columns:380px 1fr;gap:14px;align-items:start}}
.card{{background:#fff;border:1px solid {LINE};border-radius:10px;padding:10px 12px}}
.ctl{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}}
.seg{{display:inline-flex;border:1px solid {LINE};border-radius:8px;overflow:hidden}}
.seg button{{font:600 12px system-ui;border:none;background:#fff;padding:4px 10px;cursor:pointer;color:{INK}}}
.seg button.on{{background:{TEAL};color:#fff}}
.rc{{font:600 12px system-ui;border:1.5px solid var(--c);color:var(--c);background:#fff;border-radius:12px;padding:3px 10px;
  cursor:pointer;opacity:.45}} .rc.on{{background:var(--c);color:#fff;opacity:1}}
svg{{width:100%;height:auto;display:block}} svg text.ax{{font:500 10px system-ui;fill:#8a96a0}}
svg text.n{{font:700 7.5px system-ui;fill:#fff;text-anchor:middle;pointer-events:none}}
g.pt{{cursor:pointer}} g.pt.sel circle,g.pt.sel rect,g.pt.sel polygon{{stroke:#111;stroke-width:2.4;
  filter:drop-shadow(0 0 4px rgba(255,196,0,.95))}}
.faint{{color:{MUTED};font-weight:400}} #cnt{{font-size:11.5px;color:{MUTED};margin-top:4px}}
.tw{{max-height:470px;overflow:auto}}
table{{border-collapse:collapse;width:100%;white-space:nowrap}}
th{{font:600 11px system-ui;color:{MUTED};padding:6px;border-bottom:1.5px solid {TEAL};text-align:left;position:sticky;top:0;background:#fff}}
td{{font:12.5px ui-monospace,Menlo,monospace;padding:5px 6px;border-bottom:1px solid {LINE}}}
tr[data-i]{{cursor:pointer}} tr[data-i]:hover td{{background:#f2f7f8}} tr.sel td{{background:#fff4c7!important}}
td a{{color:{TEAL};text-decoration:none;font-weight:700}}
.chip{{display:inline-block;padding:1px 8px;border-radius:9px;font:600 11px system-ui}}
.hint{{font-size:11.5px;color:{MUTED};margin-top:6px}}
@media (max-width:820px){{.wrap{{grid-template-columns:1fr}}}}
"""
    view_on = lambda v: ' class="on"' if v == default_view else ""
    ctl = (f'<div class="ctl"><div class="seg" id="mode"><button class="on" data-v="heat">Heat map</button>'
           f'<button data-v="points">Pitches</button></div><div class="seg" id="view">'
           f'<button data-v="catcher"{view_on("catcher")}>Catcher\'s view</button>'
           f'<button data-v="pitcher"{view_on("pitcher")}>Pitcher\'s view</button></div></div>'
           '<div class="ctl"><button class="rc on" data-v="Successful" style="--c:#2e9d5b">Successful</button>'
           '<button class="rc on" data-v="Failed" style="--c:#c0392b">Failed</button>'
           f'<button class="rc" data-v="Not counted" style="--c:#7d8890">Not counted ({nc})</button></div>')
    left = (f'<div class="card">{ctl}<svg viewBox="0 0 360 400" id="z"></svg><div id="cnt"></div>'
            '<div class="hint">Heat map: where he challenges (hot = most). Pitches: green = overturned, red = stood, '
            'hollow = not counted; ● FB ▲ BB ■ OS. Click to find it in the list, double-click to play the video.'
            '</div></div>')
    right = ('<div class="card tw"><table><thead><tr><th>#</th><th>Video</th><th>Date</th><th>Game</th><th>Inn</th>'
             '<th>Count</th><th>Type</th><th>Velo</th><th>Call</th><th>Result</th><th>Zone</th><th>ABS Runs</th><th>Cost</th><th>Net</th></tr>'
             '</thead><tbody id="tb"></tbody></table></div>')
    script = JS.replace("let view = 'catcher'", f"let view = '{default_view}'")
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>'
            f'<div class="wrap">{left}{right}</div><script type="application/json" id="d">{data}</script>'
            f'<script type="application/json" id="pt">{json.dumps(PT_COLOR)}</script><script>{script}</script>'
            '</body></html>')
