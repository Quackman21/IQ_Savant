"""Catching AAR panel (one Streamlit component, runs its own clicks):

    [ his game behind the plate + game stats (Framing / Blocking / Throwing / Total) ]  [ framing ][ blocking ][ ground ]
    [ quick filters: pitchers, what to show                                                                        ]
    [ ONE spreadsheet-style table: click a header to sort or filter it (values / ranges), like Excel or Sheets      ]

Balls in play are left out. Every filter narrows the charts AND the game-stat tiles, so "extra strikes from this
pitcher on low pitches" is two clicks. Click any pitch (chart dot or row) to light it up everywhere; the cube next
to its video opens the pitch's full flight in 3D (to the dirt, or 3 ft past the middle of the plate)."""
from __future__ import annotations

import json
import math
from html import escape

import pandas as pd

from ui.aar import CAMERA, CUBE, INK, LINE, MUTED, PT_COLOR, TEAL, THREE_JS, _chart, _shape

GAIN, LOSS, EVEN = "#2e9d5b", "#c0392b", "#b7c0c7"
TIER_COLOR = {"Easy": "#9aa5ae", "Medium": "#e0a400", "Tough": "#d9582b", "": "#5b6bc8"}
PLATE_HALF_FT = 17 / 24


def _ok(v) -> bool:
    try:
        return v is not None and v == v and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _num(v, nd=3):
    return round(float(v), nd) if _ok(v) else None


def _mark(r: dict, x: float, y: float, fill: str, extra: str = "", ring: str = "") -> str:
    kind = r.get("pgroup") or ""
    dy = 1.6 if kind == "Breaking" else 0
    rr = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9.6" fill="none" stroke="{ring}" stroke-width="1.6"/>'
          if ring else "")
    return (f'<g class="pt" data-n="{r["n"]}" {extra}>{rr}{_shape(kind, x, y, 7.2, fill)}'
            f'<text x="{x:.1f}" y="{y + dy + 2.5:.1f}" text-anchor="middle" style="font-size:6.8px">{r["n"]}</text></g>')


def _zone_box(xs, zs, top, bot, ks=((1.33, "2 3", 1, "#9fb9be"), (1, "", 1.8, TEAL), (0.67, "5 4", 1, "#9fb9be"))):
    hw, mid, hh = PLATE_HALF_FT, (top + bot) / 2, (top - bot) / 2
    return "".join(
        f'<rect x="{xs(-hw * k):.1f}" y="{zs(mid + hh * k):.1f}" width="{xs(hw * k) - xs(-hw * k):.1f}" '
        f'height="{zs(mid - hh * k) - zs(mid + hh * k):.1f}" fill="none" stroke="{c}" stroke-width="{w}" '
        f'stroke-dasharray="{d}"/>' for k, d, w, c in ks)


def _sz(p: pd.DataFrame) -> tuple[float, float]:
    top = p["sz_top"].median() if p["sz_top"].notna().any() else 3.4
    bot = p["sz_bot"].median() if p["sz_bot"].notna().any() else 1.6
    return float(top), float(bot)


def framing_svg(rows: list[dict], p: pd.DataFrame) -> str:
    """Every take (called strike / ball), coloured by what the call did for him."""
    W, H = 270, 300
    xs = lambda x: W / 2 + x * 62
    zs = lambda z: 258 - (z - 0.25) * 52
    top, bot = _sz(p)
    g = [_zone_box(xs, zs, top, bot),
         f'<polygon points="{xs(-PLATE_HALF_FT):.1f},270 {xs(PLATE_HALF_FT):.1f},270 {xs(PLATE_HALF_FT):.1f},276 '
         f'{xs(0):.1f},284 {xs(-PLATE_HALF_FT):.1f},276" fill="#f1f4f5" stroke="#9fb9be"/>']
    for r in rows:
        if not r["take"] or not (_ok(r["px"]) and _ok(r["pz"])):
            continue
        fr = r["frame_runs"]
        col = EVEN if not _ok(fr) or abs(fr) < 0.02 else (GAIN if fr > 0 else LOSS)
        ring = INK if r["strike"] else ""
        x, y = max(9, min(W - 9, xs(r["px"]))), max(9, min(H - 22, zs(r["pz"])))
        g.append(_mark(r, x, y, col, ring=ring))
    leg = (f'<i style="color:{GAIN}">●</i> strike gained &nbsp;<i style="color:{LOSS}">●</i> strike lost &nbsp;'
           f'<i style="color:{EVEN}">●</i> as expected &nbsp;◯ ring = called strike'
           f'<br><span class="faint">takes only · ● FB ▲ BB ■ OS · dotted = shadow</span>')
    return _chart("Framing", "catcher's view", f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>', leg,
                  tip="Every taken pitch. Gained = called a strike more often than an average umpire would "
                      "(+0.02 runs or more); lost = the reverse.")


def _block_color(r: dict) -> str:
    return TIER_COLOR.get(r.get("tier") or "", TIER_COLOR[""])


def blocking_svg(rows: list[dict], p: pd.DataFrame) -> str:
    """Blocking opportunities where they crossed the plate, down to the ground."""
    W, H = 270, 300
    xs = lambda x: W / 2 + x * 50
    zs = lambda z: 250 - (z + 0.6) * 45                     # 0.6 ft below the ground (the flight carried on) .. ~4.9 ft
    top, bot = _sz(p)
    g = [_zone_box(xs, zs, top, bot, ((1, "", 1.6, TEAL),)),
         f'<rect x="6" y="{zs(0):.1f}" width="{W - 12}" height="{H - zs(0) - 4:.1f}" fill="#f6f8f9"/>',
         f'<line x1="6" x2="{W - 6}" y1="{zs(0):.1f}" y2="{zs(0):.1f}" stroke="#9fb9be" stroke-width="1.6"/>',
         f'<line x1="6" x2="{W - 6}" y1="{zs(0.5):.1f}" y2="{zs(0.5):.1f}" stroke="#9fb9be" stroke-dasharray="3 3"/>',
         f'<text x="{W - 8}" y="{zs(0.5) - 3:.1f}" class="ax" text-anchor="end">6" (dirt ball)</text>',
         f'<text x="{W - 8}" y="{zs(0) + 11:.1f}" class="ax" text-anchor="end">ground</text>']
    for r in rows:
        if not (_ok(r["px"]) and _ok(r["pz"])) or not (r["opp"] or r["dirt"]):
            continue
        x, y = max(9, min(W - 9, xs(r["px"]))), max(9, min(H - 9, zs(r["pz"])))
        g.append(_mark(r, x, y, _block_color(r), extra=f'data-opp="{int(bool(r["opp"]))}"',
                       ring=LOSS if r["got_by"] else ""))
    leg = _tier_legend()
    return _chart("Blocking opportunities", "crossing the plate", f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>',
                  leg, tip="Runner on base, not in play, not a HBP. Colour = an average catcher's chance it gets "
                           "by (WP or PB). Red ring = it got by.")


def _tier_legend() -> str:
    return (f'<i style="color:{TIER_COLOR["Easy"]}">●</i> easy &lt;5% &nbsp;<i style="color:{TIER_COLOR["Medium"]}">●'
            f'</i> medium 5-15% &nbsp;<i style="color:{TIER_COLOR["Tough"]}">●</i> tough 15%+ &nbsp;'
            f'<span style="color:{LOSS}">◯</span> got by<br><span class="faint">chance it gets by an average catcher'
            f'</span>')


def ground_svg(rows: list[dict]) -> str:
    """Where each blocking chance first reaches the dirt (measured flight, carried to the ground). The plate is
    drawn wider than true scale side to side so the spread across it reads."""
    W, H = 270, 330
    XF, YF = 46.0, 30.0                                      # px per foot: side to side wider than front to back
    xs = lambda x: W / 2 + x * XF
    ys = lambda y: 250 - y * YF                              # y = ft in front of the plate's front edge
    g = []
    for v in range(-2, 9):
        g.append(f'<line x1="22" x2="{W - 6}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="#eef1f3" '
                 f'stroke-dasharray="{"" if v == 0 else "3 3"}"/><text x="18" y="{ys(v) + 3:.1f}" class="ax" '
                 f'text-anchor="end">{v}</text>')
    for v in (-2, -1, 0, 1, 2):
        g.append(f'<text x="{xs(v):.1f}" y="{H - 4}" class="ax" text-anchor="middle">{v}</text>')
    h, back = PLATE_HALF_FT, 17 / 12
    g.append(f'<polygon points="{xs(-h):.1f},{ys(0):.1f} {xs(h):.1f},{ys(0):.1f} {xs(h):.1f},{ys(-h):.1f} '
             f'{xs(0):.1f},{ys(-back):.1f} {xs(-h):.1f},{ys(-h):.1f}" fill="#f6f8f9" stroke="{TEAL}" stroke-width="1.6"/>')
    g.append(f'<line x1="22" x2="{W - 6}" y1="{ys(17 / 24 - 3 - 17 / 12):.1f}" y2="{ys(17 / 24 - 3 - 17 / 12):.1f}" '
             f'stroke="#9fb9be" stroke-dasharray="2 3"/>')
    g.append(f'<text x="{W - 8}" y="{ys(8) + 9:.1f}" class="ax" text-anchor="end">toward the mound ↑</text>')
    for r in rows:
        if not r["dirt"] or not (_ok(r["gx"]) and _ok(r["gy"])):
            continue
        x, y = max(9, min(W - 9, xs(r["gx"]))), max(9, min(H - 16, ys(r["gy"])))
        g.append(_mark(r, x, y, _block_color(r), extra=f'data-opp="{int(bool(r["opp"]))}"',
                       ring=LOSS if r["got_by"] else ""))
    leg = (_tier_legend() + '<br><span class="faint">ft · catcher\'s view · plate widened side to side · dashed = '
           '3 ft past the middle of the plate</span> <span class="faint" id="air"></span>')
    return _chart("Where it hit the dirt", "blocking chances", f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>', leg,
                  tip="Statcast's measured flight (release speed + movement), carried past the plate until the "
                      "bottom of the ball reaches the ground. Only pitches that land before 3 ft past the middle "
                      "of the plate.")


# ----------------------------------------------------------------------------- the rows the table / tiles run on
ROW_KEYS = ["n", "video", "inning", "inn", "count", "on", "pitcher", "p_throws", "batter", "result", "play", "ptype",
            "pgroup", "velo", "hb", "ivb", "zone", "gz", "fzone", "fz", "region", "strike_prob", "frame_runs", "extra", "take",
            "strike", "swing", "opp", "dirt", "wp_chance", "got_by", "block_runs", "tier", "steal", "cs_chance", "cs",
            "throw_runs", "catch_runs", "px", "pz", "gx", "gy", "orig_call", "abs", "abs_by", "abs_runs", "abs_cost", "abs_net", "has3d"]


def rows(p: pd.DataFrame, scenes3d: dict | None = None) -> list[dict]:
    s3 = set(int(k) for k in (scenes3d or {}))
    out = []
    for r in p.to_dict("records"):
        d = {}
        for k in ROW_KEYS:
            v = r.get(k)
            if k == "inn":
                v = f'{r.get("half")}{r.get("inning")}'
            elif k == "fz":
                from core.catch_aar import FRAME_ZONES
                v = FRAME_ZONES.index(r.get("fzone")) if r.get("fzone") in FRAME_ZONES else None
            elif k == "has3d":
                v = int(r["n"]) in s3
            elif k in ("take", "strike", "swing", "opp", "dirt"):
                v = bool(v)
            elif k == "got_by":
                v = bool(_ok(v) and v > 0)
            elif k in ("n", "inning", "gz"):
                v = int(v) if _ok(v) else None
            elif k == "cs":
                v = None if not _ok(v) else int(v)
            elif isinstance(v, float) or (v is not None and not isinstance(v, (str, bool, int))):
                v = _num(v, 4)
            elif v is None or (isinstance(v, float) and v != v):
                v = None
            d[k] = v if not (isinstance(v, str) and v == "nan") else ""
        out.append(d)
    return out


TIPS = {"ABS": "His ABS challenges: successful / made", "Innings": "Innings he caught at least one pitch"}


def line_html(line: dict) -> str:
    cols = ["Pitches", "Innings", "Pitchers", "SB", "CS", "WP", "PB", "ABS"]
    return ('<table class="line"><tr>' + "".join(f'<th title="{TIPS.get(c, "")}">{c}</th>' for c in cols) + "</tr><tr>"
            + "".join(f"<td>{line.get(c, '–')}</td>" for c in cols) + "</tr></table>")


CSS_EXTRA = f"""
.qf{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:0 0 8px}}
.qf .lab{{font:600 12px system-ui;color:{MUTED};margin-right:2px}} .qf .sep{{width:12px}}
.pchip{{font:600 12px system-ui;border:1.5px solid {TEAL};color:{TEAL};background:#fff;border-radius:12px;
  padding:3px 10px;cursor:pointer;opacity:.5}} .pchip.on{{background:{TEAL};color:#fff;opacity:1}}
.fl{{display:inline-flex;align-items:center;gap:4px;font:600 11.5px system-ui;background:#eaf3f4;color:#2c6670;
  border-radius:10px;padding:2px 4px 2px 9px}} .fl button{{border:none;background:none;cursor:pointer;color:#2c6670;
  font:700 13px system-ui;padding:0 4px}}
table.p th{{cursor:pointer;user-select:none}} table.p th:hover{{color:{TEAL}}}
table.p th .ar{{font-size:9px;margin-left:3px;color:{TEAL}}} table.p th.fx{{color:{TEAL}}}
table.p th .fn{{display:inline-block;width:6px;height:6px;border-radius:50%;background:{TEAL};margin-left:4px;
  vertical-align:middle}}
tr.crow{{cursor:pointer}} tr.crow:hover td{{background:#f2f7f8}}
tr.crow.sel td{{background:#fff4c7!important;box-shadow:inset 0 1px 0 #ffd34d,inset 0 -1px 0 #ffd34d}}
.has-sel tr.crow:not(.sel) td{{opacity:.55}}
#menu{{position:absolute;z-index:40;background:#fff;border:1px solid {LINE};border-radius:9px;
  box-shadow:0 8px 24px rgba(0,0,0,.18);width:250px;font:12.5px system-ui;display:none}}
#menu.on{{display:block}} #menu .sec{{padding:6px 10px;border-bottom:1px solid {LINE}}}
#menu .sb{{display:block;width:100%;text-align:left;border:none;background:none;padding:5px 2px;cursor:pointer;
  font:12.5px system-ui;color:{INK}}} #menu .sb:hover{{color:{TEAL}}}
#menu input[type=text],#menu input[type=number]{{width:100%;font:12.5px system-ui;padding:4px 6px;border:1px solid {LINE};
  border-radius:6px}} #menu .vals{{max-height:220px;overflow:auto;margin-top:6px}}
#menu label{{display:flex;gap:6px;align-items:center;padding:2px 0;cursor:pointer}} #menu label .c{{margin-left:auto;
  color:{MUTED};font-size:11px}} #menu .rng{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
#menu .ft{{display:flex;gap:6px;justify-content:flex-end;padding:6px 10px}}
#menu .ft button{{font:600 12px system-ui;border:1px solid {LINE};background:#fff;border-radius:6px;padding:4px 9px;
  cursor:pointer}}
.cmode{{display:flex;justify-content:center;margin:2px 0 4px}}
.cmode .seg button{{font-size:11px;padding:3px 8px}}
.hide-nonopp g.pt[data-opp="0"]{{display:none}}
.note{{font-size:11px;color:{MUTED};margin-top:6px;line-height:1.5}}
td.rc{{font-variant-numeric:tabular-nums}}
#tiles .tile{{cursor:help}} #tiles .tile:hover{{border-color:{TEAL}}}
#zp svg{{width:170px;display:block;margin:2px auto 0}}
#zp .zc polygon,#zp .zc rect{{fill:#eef1f3;stroke:#fff;stroke-width:2;cursor:pointer}}
#zp .zc.on polygon,#zp .zc.on rect{{fill:{TEAL}}} #zp .zc:hover polygon,#zp .zc:hover rect{{stroke:{INK}}}
#zp .zc.off polygon,#zp .zc.off rect{{fill:#f8f9fa;cursor:default}} #zp .zc.off{{pointer-events:none}}
#zp text{{pointer-events:none;text-anchor:middle}} #zp .zn{{font:700 12px system-ui;fill:{INK}}}
#zp .zk{{font:600 9.5px system-ui;fill:{MUTED}}} #zp .zc.on .zn{{fill:#fff}} #zp .zc.on .zk{{fill:#dbeef1}}
#zp .zc.off .zn{{fill:#c4ccd2}} #zp .zc path{{fill:#eef1f3;stroke:#fff;stroke-width:2;cursor:pointer}}
#zp .zc.on path{{fill:#e8a33d}} #zp .zc.on polygon,#zp .zc.on rect{{fill:#e8a33d}}
#zp .zc.on .zk{{fill:#fff3df}} #zp .zc.off path{{fill:#f8f9fa}}
#zp .zs{{display:flex;justify-content:space-between;font-size:10.5px;color:{MUTED};margin:4px 2px}}
#zp .zb{{display:flex;gap:6px;justify-content:center}} #zp .zb button{{font:600 11.5px system-ui;border:1px solid {LINE};
  background:#fff;border-radius:6px;padding:3px 9px;cursor:pointer}}
tr.crow td{{user-select:none}}
#ctip{{position:absolute;z-index:45;background:#fff;border:1px solid {LINE};border-radius:9px;padding:8px 10px;
  box-shadow:0 8px 24px rgba(0,0,0,.16);font:12.5px system-ui;min-width:210px;display:none;pointer-events:none}}
#ctip.on{{display:block}} #ctip .th{{font:700 12.5px system-ui;margin-bottom:5px}}
#ctip table{{width:100%;border-collapse:collapse}} #ctip td{{padding:3px 0;border-bottom:1px solid #f0f2f4}}
#ctip td:last-child{{text-align:right;font-family:ui-monospace,monospace;padding-left:14px}}
"""

JS = r"""
const root = document.getElementById('aar');
const ROWS = JSON.parse(document.getElementById('rows-data').textContent);
const BYN = Object.fromEntries(ROWS.map(r => [r.n, r]));
const HAS_MODEL = JSON.parse(document.getElementById('model-data').textContent);
const COLS = [
  {k: 'n', l: '#', t: 'num', a: 'r', f: 'int'},
  {k: 'video', l: 'Video', t: 'none', a: 'c'},
  {k: 'inn', l: 'Inn', t: 'cat', a: 'r', by: 'n'},
  {k: 'count', l: 'Count', t: 'cat', a: 'r'},
  {k: 'on', l: 'On', t: 'cat', a: 'l'},
  {k: 'pitcher', l: 'Pitcher', t: 'cat', a: 'l'},
  {k: 'batter', l: 'Batter', t: 'cat', a: 'l'},
  {k: 'result', l: 'Pitch Result', t: 'cat', a: 'l'},
  {k: 'play', l: 'Play Result', t: 'cat', a: 'l'},
  {k: 'abs', l: 'ABS', t: 'cat', a: 'c'},
  {k: 'ptype', l: 'Type', t: 'cat', a: 'c'},
  {k: 'velo', l: 'Velo', t: 'num', a: 'r', f: 1},
  {k: 'hb', l: 'HB', t: 'num', a: 'r', f: 1},
  {k: 'ivb', l: 'IVB', t: 'num', a: 'r', f: 1},
  {k: 'fzone', l: 'Zone', t: 'cat', a: 'l', by: 'fz'},
  {k: 'zone', l: 'Gameday Zone', t: 'cat', a: 'l', by: 'gz'},
  {k: 'strike_prob', l: 'Strike %', t: 'num', a: 'r', f: 'pct'},
  {k: 'frame_runs', l: 'Framing Runs', t: 'num', a: 'r', f: 'runs'},
  {k: 'wp_chance', l: 'WP/PB %', t: 'num', a: 'r', f: 'pct1'},
  {k: 'block_runs', l: 'Blocking Runs', t: 'num', a: 'r', f: 'runs'},
  {k: 'cs_chance', l: 'CS %', t: 'num', a: 'r', f: 'pct'},
  {k: 'throw_runs', l: 'Throwing Runs', t: 'num', a: 'r', f: 'runs'},
  {k: 'catch_runs', l: 'Total Runs', t: 'num', a: 'r', f: 'runs'},
];
const COLK = Object.fromEntries(COLS.map(c => [c.k, c]));
const PT = JSON.parse(document.getElementById('pt-data').textContent);
const CAM = document.getElementById('ico-cam').innerHTML, CUBEI = document.getElementById('ico-cube').innerHTML;
const filt = {};                       // key -> Set (cat) | {lo, hi} (num)
let sort = {k: 'n', dir: 1}, show = 'all', chartMode = 'opp';
const sel = new Set();
const ok = v => v !== null && v !== undefined && v === v;
function fmt(v, f) {
  if (!ok(v)) return '–';
  if (f === 'int') return String(v);
  if (f === 'pct') return Math.round(v * 100) + '%';
  if (f === 'pct1') return (v * 100).toFixed(1) + '%';
  if (f === 'runs') { const s = v.toFixed(3); return (v > 0.0005 ? '+' : '') + (s === '-0.000' ? '0.000' : s); }
  return (+v).toFixed(f || 0);
}
function runsStyle(v) {
  if (!ok(v) || Math.abs(v) < 0.005) return '';
  const t = Math.min(1, Math.abs(v) / 0.25);
  return v > 0 ? `background:rgba(46,157,91,${0.12 + 0.55 * t});${t > .6 ? 'color:#fff' : ''}`
               : `background:rgba(192,57,43,${0.12 + 0.55 * t});${t > .6 ? 'color:#fff' : ''}`;
}
const disp = (r, k) => { const v = r[k]; return ok(v) && v !== '' ? String(v) : '(blank)'; };
function passShow(r) {
  if (show === 'take') return r.take;
  if (show === 'opp') return r.opp;
  if (show === 'dirt') return r.dirt;
  if (show === 'steal') return !!r.steal;
  if (show === 'abs') return !!r.abs;
  if (show === 'swing') return r.swing;
  return true;
}
function passes(r, skip) {
  if (!passShow(r)) return false;
  for (const [k, f] of Object.entries(filt)) {
    if (k === skip) continue;
    const c = COLK[k];
    if (c.t === 'cat') { if (!f.has(disp(r, k))) return false; }
    else {
      const v = r[k];
      if ((f.lo !== null && f.lo !== '') || (f.hi !== null && f.hi !== '')) {
        if (!ok(v)) return false;
        const sc = c.f === 'pct' || c.f === 'pct1' ? 100 : 1;
        if (f.lo !== null && f.lo !== '' && v * sc < +f.lo) return false;
        if (f.hi !== null && f.hi !== '' && v * sc > +f.hi) return false;
      }
    }
  }
  return true;
}
function cell(r, c) {
  const v = r[c.k];
  if (c.k === 'video') {
    const cam = r.video ? `<a class="vid" href="https://baseballsavant.mlb.com/sporty-videos?playId=${r.video}" target="_blank" rel="noopener" title="Play this pitch">${CAM}</a>` : '';
    const s3 = r.has3d ? `<button class="s3d" data-n="${r.n}" title="Pitch flight in 3D">${CUBEI}</button>` : '';
    return cam || s3 ? `<span class="vcell">${cam}${s3}</span>` : '–';
  }
  if (c.k === 'ptype') return `<b style="color:${PT[v] || '#16202b'}">${v || '–'}</b>`;
  if (c.k === 'result' && r.orig_call) return `${v} <span class="tag x" title="ABS challenge overturned the umpire's call; framing uses his original call">ABS: was ${r.orig_call}</span>`;
  if (c.k === 'abs') return v ? `<span class="chip" style="background:${v === 'Successful' ? '#cdebd6;color:#14532d' : '#f6d3d0;color:#7a1f1a'}">${v}</span>`
    : (r.abs_by ? `<span class="faint" title="Challenged by the ${r.abs_by.toLowerCase()}">${r.abs_by}</span>` : '–');
  if (c.k === 'play' && r.got_by) return `<b style="color:#c0392b">${v}</b>`;
  if (c.t === 'num') return fmt(v, c.f);
  return ok(v) && v !== '' ? String(v).replace(/</g, '&lt;') : '–';
}
function sortVal(r, c) { const k = c.by || c.k; return r[k]; }
function render() {
  const live = ROWS.filter(r => passes(r));
  const c = COLK[sort.k];
  live.sort((a, b) => {
    const x = sortVal(a, c), y = sortVal(b, c);
    if (!ok(x) || x === '') return 1; if (!ok(y) || y === '') return -1;
    return (x < y ? -1 : x > y ? 1 : 0) * sort.dir || a.n - b.n;
  });
  const head = COLS.map(c => `<th class="${c.a}${filt[c.k] ? ' fx' : ''}" data-k="${c.k}">${c.l}` +
    (sort.k === c.k ? `<span class="ar">${sort.dir > 0 ? '▲' : '▼'}</span>` : '') + (filt[c.k] ? '<span class="fn"></span>' : '') + '</th>').join('');
  const body = live.map(r => `<tr class="crow${sel.has(r.n) ? ' sel' : ''}" data-n="${r.n}">` + COLS.map(c =>
    `<td class="${c.a}${c.t === 'num' ? ' rc' : ''}" style="${c.f === 'runs' ? runsStyle(r[c.k]) : ''}">${cell(r, c)}</td>`).join('') + '</tr>').join('');
  document.getElementById('tbl').innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body ||
    `<tr><td colspan="${COLS.length}" style="text-align:center;color:#6a7681;padding:20px">No pitches match these filters.</td></tr>`}</tbody>`;
  const keep = new Set(live.map(r => r.n));
  document.querySelectorAll('g.pt').forEach(g => {
    g.classList.toggle('filt', !keep.has(+g.dataset.n)); g.classList.toggle('sel', sel.has(+g.dataset.n)); });
  root.classList.toggle('has-sel', sel.size > 0);
  root.classList.toggle('hide-nonopp', chartMode === 'opp');
  document.getElementById('seln').textContent = sel.size ? sel.size + ' selected' : '';
  document.getElementById('shown').textContent = live.length < ROWS.length ? `${live.length} of ${ROWS.length} pitches` : `${ROWS.length} pitches`;
  const air = live.filter(r => r.opp && !r.dirt).length;
  document.getElementById('air').textContent = air ? `· ${air} more chance${air > 1 ? 's' : ''} caught before reaching the dirt` : '';
  document.querySelectorAll('.pchip').forEach(b => b.classList.toggle('on', !filt.pitcher || filt.pitcher.has(b.dataset.v)));
  chips(); tiles(live); wire();
}
function chips() {
  const fl = Object.keys(filt).map(k => {
    const f = filt[k], c = COLK[k];
    const txt = c.t === 'cat' ? (f.size === 1 ? [...f][0] : `${f.size} values`)
      : [ok(f.lo) && f.lo !== '' ? '≥ ' + f.lo : '', ok(f.hi) && f.hi !== '' ? '≤ ' + f.hi : ''].filter(Boolean).join(' ');
    return `<span class="fl">${c.l}: ${String(txt).replace(/</g, '&lt;')}<button data-k="${k}" title="Clear">×</button></span>`;
  }).join('');
  document.getElementById('fls').innerHTML = fl;
  document.querySelectorAll('#fls button').forEach(b => b.onclick = () => { delete filt[b.dataset.k]; render(); });
}
// ---- game stats (follow every filter)
const S = (a, k) => a.reduce((s, r) => s + (ok(r[k]) ? +r[k] : 0), 0);
const signed = (v, d = 1) => (v > 0.0005 ? '+' : '') + v.toFixed(d);
function tiles(live) {
  const m = document.getElementById('metric').value;
  const t = [];
  const M = v => HAS_MODEL ? v : '–';
  if (m === 'framing') {
    const tk = live.filter(r => r.take), sh = tk.filter(r => String(r.fzone || '').startsWith('Shadow'));
    const k = tk.filter(r => r.strike).length, e = S(tk, 'strike_prob'), ex = S(tk, 'extra');
    const shk = sh.filter(r => r.strike).length, she = S(sh, 'strike_prob');
    t.push(['Takes', tk.length], ['Called strikes', k], ['Expected strikes', M(e.toFixed(1))], ['Extra strikes', M(signed(ex))],
           ['Framing runs', M(signed(S(tk, 'frame_runs'), 2))], ['Shadow takes', sh.length],
           ['Shadow strike %', sh.length ? Math.round(100 * shk / sh.length) + '%' : '–'],
           ['Expected', sh.length ? M(Math.round(100 * she / sh.length) + '%') : '–']);
  } else if (m === 'blocking') {
    const o = live.filter(r => r.opp), dirt = o.filter(r => r.dirt), gen = o.filter(r => (r.wp_chance || 0) >= .05);
    const got = o.filter(r => r.got_by).length, ex = S(o, 'wp_chance');
    const tough = o.filter(r => r.tier === 'Tough'), tk = tough.filter(r => !r.got_by).length;
    t.push(['Opportunities', o.length], ['In the dirt', dirt.length], ['Genuine (5%+)', M(gen.length)],
           ['WP + PB', got], ['Expected WP + PB', M(ex.toFixed(2))], ['Blocks above avg', M(signed(ex - got, 2))],
           ['Blocking runs', M(signed(S(o, 'block_runs'), 2))], ['Tough blocks', M(tough.length ? `${tk} / ${tough.length}` : '–')]);
  } else if (m === 'throwing') {
    const a = live.filter(r => r.steal), cs = a.filter(r => r.cs === 1).length, e = S(a, 'cs_chance');
    t.push(['Steal attempts', a.length], ['Caught', cs], ['Stolen', a.length - cs], ['Expected CS', M(e.toFixed(2))],
           ['CS above exp', M(signed(cs - e, 2))], ['Throwing runs', M(signed(S(a, 'throw_runs'), 2))],
           ['CS %', a.length ? Math.round(100 * cs / a.length) + '%' : '–'], ['Expected CS %', a.length ? M(Math.round(100 * e / a.length) + '%') : '–']);
  } else if (m === 'abs') {
    const a = live.filter(r => r.abs), s = a.filter(r => r.abs === 'Successful').length;
    t.push(['Challenges', a.length], ['Successful', s], ['Failed', a.length - s],
           ['Success %', a.length ? Math.round(100 * s / a.length) + '%' : '–'], ['ABS runs', signed(S(a, 'abs_runs'), 2)],
           ['Challenge cost', signed(-S(a, 'abs_cost'), 2)], ['Net ABS runs', signed(S(a, 'abs_net'), 2)]);
  } else {
    const f = S(live, 'frame_runs'), b = S(live, 'block_runs'), th = S(live, 'throw_runs');
    t.push(['Framing runs', M(signed(f, 2))], ['Blocking runs', M(signed(b, 2))], ['Throwing runs', M(signed(th, 2))],
           ['Catcher runs', M(signed(f + b + th, 2))]);
  }
  const col = v => { const x = parseFloat(String(v)); return /^[+-]/.test(String(v)) && Math.abs(x) >= .005 ?
    (x > 0 ? 'color:#2e9d5b' : 'color:#c0392b') : ''; };
  document.getElementById('tiles').innerHTML = t.map(([l, v]) =>
    `<div class="tile" data-l="${l}" data-v="${v}"><div class="l">${l}</div><div class="v" style="${col(v)}">${v}</div></div>`).join('');
  document.querySelectorAll('#tiles .tile').forEach(el => {
    el.onmouseenter = () => showTip(el, m); el.onmouseleave = () => tip.classList.remove('on'); });
}
// ---- hover a tile: his season / last 30 days / MLB average (per game caught; rates pooled)
const COMPS = JSON.parse(document.getElementById('comps-data').textContent);
const tip = document.getElementById('ctip');
const RATE = new Set(['Success %', 'Shadow strike %', 'Expected', 'Tough blocks', 'CS %', 'Expected CS %']);
function showTip(el, m) {
  const l = el.dataset.l, c = (COMPS[m] || {})[l], g = (COMPS[m] || {})._games || [];
  const gm = Object.fromEntries(g);
  const rows = c ? c.map(([who, v]) => `<tr><td>${who}${gm[who] ? ` <span class="faint">${gm[who]} g</span>` : ''}</td><td>${v}</td></tr>`).join('')
                 : '<tr><td colspan="2" class="faint">No season numbers for this one yet</td></tr>';
  const how = RATE.has(l) ? 'rate' : 'per game caught';
  tip.innerHTML = `<div class="th">${l} <span class="faint">· ${how}</span></div><table><tr><td><b>This game</b>` +
    `${Object.keys(filt).length || show !== 'all' ? ' <span class="faint">(filtered)</span>' : ''}</td><td><b>${el.dataset.v}</b></td></tr>${rows}</table>`;
  const r = el.getBoundingClientRect(), rr = root.getBoundingClientRect();
  tip.style.left = Math.max(4, r.left - rr.left) + 'px'; tip.style.top = (r.bottom - rr.top + 6) + 'px';
  tip.classList.add('on');
}
// ---- header menu (sort + filter, like a spreadsheet)
const menu = document.getElementById('menu');
function openMenu(th) {
  const c = COLK[th.dataset.k]; if (!c || c.t === 'none') return;
  const r0 = th.getBoundingClientRect(), rr = root.getBoundingClientRect();
  menu.style.left = Math.min(r0.left - rr.left, rr.width - 260) + 'px';
  menu.style.top = (r0.bottom - rr.top + 4) + 'px';
  let h = `<div class="sec"><button class="sb" data-s="1">Sort ${c.t === 'num' ? 'low → high' : 'A → Z'}</button>` +
          `<button class="sb" data-s="-1">Sort ${c.t === 'num' ? 'high → low' : 'Z → A'}</button></div>`;
  if (c.t === 'cat') {
    const base = ROWS.filter(r => passes(r, c.k)), cnt = {};
    base.forEach(r => { const v = disp(r, c.k); cnt[v] = (cnt[v] || 0) + 1; });
    const vals = [...new Set(ROWS.map(r => disp(r, c.k)))].sort((a, b) => {
      const ra = ROWS.find(r => disp(r, c.k) === a), rb = ROWS.find(r => disp(r, c.k) === b);
      const x = ra[c.by || c.k], y = rb[c.by || c.k]; return ok(x) && ok(y) && x !== y ? (x < y ? -1 : 1) : a.localeCompare(b); });
    const cur = filt[c.k];
    if (c.k === 'zone') h += zonePicker(cnt);
    if (c.k === 'fzone') h += framePicker(cnt);
    h += `<div class="sec"><input type="text" id="mq" placeholder="Search"><div class="vals"><label><input type="checkbox" id="mall"` +
         `${!cur ? ' checked' : ''}> <b>(Select all)</b></label>` + vals.map(v => `<label class="mv"><input type="checkbox" value="${v.replace(/"/g, '&quot;')}"` +
         `${!cur || cur.has(v) ? ' checked' : ''}> ${v.replace(/</g, '&lt;')}<span class="c">${cnt[v] || 0}</span></label>`).join('') + '</div></div>';
  } else {
    const cur = filt[c.k] || {lo: '', hi: ''}, pc = c.f === 'pct' || c.f === 'pct1';
    h += `<div class="sec">Show values${pc ? ' (%)' : ''} between<div class="rng" style="margin-top:6px"><input type="number" step="any" id="mlo" placeholder="min" value="${cur.lo ?? ''}">` +
         `<input type="number" step="any" id="mhi" placeholder="max" value="${cur.hi ?? ''}"></div></div>`;
  }
  h += `<div class="ft"><button id="mclr">Clear filter</button><button id="mok">Done</button></div>`;
  menu.innerHTML = h; menu.classList.add('on'); menu.dataset.k = c.k;
  menu.querySelectorAll('.sb').forEach(b => b.onclick = () => { sort = {k: c.k, dir: +b.dataset.s}; closeMenu(); render(); });
  menu.querySelector('#mclr').onclick = () => { delete filt[c.k]; closeMenu(); render(); };
  menu.querySelector('#mok').onclick = () => closeMenu();
  if (c.t === 'cat') {
    const boxes = () => [...menu.querySelectorAll('label.mv input')];
    const sync = () => { const on = boxes().filter(b => b.checked).map(b => b.value);
      if (on.length === boxes().length) delete filt[c.k]; else filt[c.k] = new Set(on);
      menu.querySelector('#mall').checked = on.length === boxes().length; render(); };
    boxes().forEach(b => b.onchange = sync);
    menu.querySelector('#mall').onchange = e => { boxes().forEach(b => b.checked = e.target.checked); sync(); };
    if (c.k === 'zone' || c.k === 'fzone') {
      const box = v => boxes().find(b => b.value === v);
      const paint = () => menu.querySelectorAll('#zp .zc').forEach(z => {
        const b = box(z.dataset.v); z.classList.toggle('on', !!(b && b.checked)); });
      menu.querySelectorAll('#zp .zc').forEach(z => z.onclick = () => {
        const b = box(z.dataset.v); if (!b) return; b.checked = !b.checked; sync(); paint(); });
      menu.querySelector('#zall').onclick = () => { boxes().forEach(b => b.checked = true); sync(); paint(); };
      menu.querySelector('#znone').onclick = () => { boxes().forEach(b => b.checked = false); sync(); paint(); };
      menu.querySelector('#mall').addEventListener('change', paint);
      boxes().forEach(b => b.addEventListener('change', paint));
      paint();
    }
    menu.querySelector('#mq').oninput = e => { const q = e.target.value.toLowerCase();
      menu.querySelectorAll('label.mv').forEach(l => l.style.display = l.textContent.toLowerCase().includes(q) ? '' : 'none'); };
  } else {
    const upd = () => { const lo = menu.querySelector('#mlo').value, hi = menu.querySelector('#mhi').value;
      if (lo === '' && hi === '') delete filt[c.k]; else filt[c.k] = {lo, hi}; render(); };
    menu.querySelector('#mlo').oninput = upd; menu.querySelector('#mhi').oninput = upd;
  }
  resize();
}
function closeMenu() { menu.classList.remove('on'); }
// Statcast's zones as a picture (catcher's view): 1-9 the strike zone, 11-14 the four corners outside it
// Savant's framing split as one picture (catcher's view): click a piece to show / hide it
function framePicker(cnt) {
  const R = (x, y, w, h) => `<rect x="${x}" y="${y}" width="${w}" height="${h}"/>`;
  const P = {'Heart': [R(60, 60, 80, 80), 100, 98],
    'Shadow - Top Left': [R(20, 20, 40, 40), 40, 38], 'Shadow - Top': [R(60, 20, 80, 40), 100, 38],
    'Shadow - Top Right': [R(140, 20, 40, 40), 160, 38], 'Shadow - Left': [R(20, 60, 40, 80), 40, 98],
    'Shadow - Right': [R(140, 60, 40, 80), 160, 98], 'Shadow - Bottom Left': [R(20, 140, 40, 40), 40, 158],
    'Shadow - Bottom': [R(60, 140, 80, 40), 100, 158], 'Shadow - Bottom Right': [R(140, 140, 40, 40), 160, 158],
    'Chase / Waste': ['<path fill-rule="evenodd" d="M0,0H200V200H0Z M20,20V180H180V20Z"/>', 100, 11]};
  const short = {'Heart': 'Heart', 'Shadow - Top Left': 'Top L', 'Shadow - Top': 'Top', 'Shadow - Top Right': 'Top R',
                 'Shadow - Left': 'Left', 'Shadow - Right': 'Right', 'Shadow - Bottom Left': 'Bot L',
                 'Shadow - Bottom': 'Bottom', 'Shadow - Bottom Right': 'Bot R', 'Chase / Waste': 'Chase / Waste'};
  const g = Object.entries(P).map(([v, [shape, x, y]]) => {
    const n = cnt[v] || 0, has = ROWS.some(r => r.fzone === v);
    return `<g class="zc${has ? '' : ' off'}" data-v="${v}"><title>${v} · ${n} pitch${n === 1 ? '' : 'es'}</title>${shape}` +
      `` + (v === 'Chase / Waste' ? `<text x="${x}" y="${y + 3}" class="zn">Chase / Waste <tspan class="zk">${n}</tspan></text></g>`
      : `<text x="${x}" y="${y}" class="zn">${short[v]}</text><text x="${x}" y="${y + 12}" class="zk">${n}</text></g>`);
  }).join('');
  return `<div class="sec" id="zp"><svg viewBox="-2 -2 204 204">${g}` +
    `<rect x="78" y="196" width="44" height="3" fill="#c9d1d6"/></svg>` +
    `<div class="zs"><span>← glove side</span><span>catcher's view</span><span>arm side →</span></div>` +
    `<div class="zb"><button id="zall">Select all</button><button id="znone">Clear all</button></div></div>`;
}
function zonePicker(cnt) {
  const name = {};
  ROWS.forEach(r => { if (ok(r.gz)) name[r.gz] = disp(r, 'zone'); });
  const cells = [];
  const cell = (z, shape, lx, ly) => {
    const v = name[z], n = v ? (cnt[v] || 0) : 0;
    cells.push(`<g class="zc${v ? '' : ' off'}" data-v="${v ? v.replace(/"/g, '&quot;') : ''}"><title>${v || z + ' · no pitches'}</title>${shape}` +
      `<text x="${lx}" y="${ly}" class="zn">${z}</text><text x="${lx}" y="${ly + 13}" class="zk">${n || ''}</text></g>`);
  };
  const poly = p => `<polygon points="${p}"/>`;
  cell(11, poly('0,0 100,0 100,40 40,40 40,100 0,100'), 20, 22);
  cell(12, poly('100,0 200,0 200,100 160,100 160,40 100,40'), 180, 22);
  cell(13, poly('0,100 40,100 40,160 100,160 100,200 0,200'), 20, 180);
  cell(14, poly('200,100 200,200 100,200 100,160 160,160 160,100'), 180, 180);
  for (let z = 1; z <= 9; z++) {
    const x = 40 + ((z - 1) % 3) * 40, y = 40 + Math.floor((z - 1) / 3) * 40;
    cell(z, `<rect x="${x}" y="${y}" width="40" height="40"/>`, x + 20, y + 19);
  }
  return `<div class="sec" id="zp"><svg viewBox="-2 -2 204 204">${cells.join('')}</svg>` +
    `<div class="zs"><span>← glove side</span><span>catcher's view</span><span>arm side →</span></div>` +
    `<div class="zb"><button id="zall">Select all</button><button id="znone">Clear all</button></div></div>`;
}
document.addEventListener('click', e => { if (menu.classList.contains('on') && !menu.contains(e.target) && !e.target.closest('th')) closeMenu(); });
function wire() {
  document.querySelectorAll('#tbl th').forEach(th => th.onclick = e => { e.stopPropagation();
    if (menu.classList.contains('on') && menu.dataset.k === th.dataset.k) closeMenu(); else openMenu(th); });
  document.querySelectorAll('tr.crow').forEach(tr => tr.onclick = e => {
    if (e.target.closest('a') || e.target.closest('button')) return;
    const n = +tr.dataset.n; sel.has(n) ? sel.delete(n) : sel.add(n); paintSel(); });
  document.querySelectorAll('button.s3d').forEach(b => b.onclick = e => { e.stopPropagation(); open3d(b.dataset.n, b); });
}
document.querySelectorAll('g.pt').forEach(g => {
  g.addEventListener('click', () => { const n = +g.dataset.n; sel.has(n) ? sel.delete(n) : sel.add(n); paintSel(); });
  g.addEventListener('dblclick', () => playVideo(+g.dataset.n));
});
function paintSel() {
  document.querySelectorAll('tr.crow').forEach(tr => tr.classList.toggle('sel', sel.has(+tr.dataset.n)));
  document.querySelectorAll('g.pt').forEach(g => g.classList.toggle('sel', sel.has(+g.dataset.n)));
  root.classList.toggle('has-sel', sel.size > 0);
  document.getElementById('seln').textContent = sel.size ? sel.size + ' selected' : '';
}
// double-click a pitch (row or chart dot): play its video
function playVideo(n) {
  const r = BYN[n]; if (!r || !r.video) return;
  window.open('https://baseballsavant.mlb.com/sporty-videos?playId=' + r.video, '_blank', 'noopener');
}
document.getElementById('tbl').addEventListener('dblclick', e => {
  if (e.target.closest('a') || e.target.closest('button') || e.target.closest('th')) return;
  const tr = e.target.closest('tr.crow'); if (tr) { window.getSelection().removeAllRanges(); playVideo(+tr.dataset.n); }
});
document.querySelectorAll('.pchip').forEach(b => b.onclick = () => {
  const all = [...document.querySelectorAll('.pchip')].map(x => x.dataset.v);
  let cur = filt.pitcher ? new Set(filt.pitcher) : new Set(all);
  if (!filt.pitcher) cur = new Set([b.dataset.v]);                // first click = just this pitcher
  else cur.has(b.dataset.v) ? cur.delete(b.dataset.v) : cur.add(b.dataset.v);
  if (!cur.size || cur.size === all.length) delete filt.pitcher; else filt.pitcher = cur;
  render(); });
document.querySelectorAll('#show button').forEach(b => b.onclick = () => { show = b.dataset.v;
  document.querySelectorAll('#show button').forEach(x => x.classList.toggle('on', x === b)); render(); });
document.querySelectorAll('.cmode button').forEach(b => b.onclick = () => { chartMode = b.dataset.v;
  document.querySelectorAll('.cmode button').forEach(x => x.classList.toggle('on', x.dataset.v === chartMode)); render(); });
document.getElementById('clear').onclick = () => { sel.clear(); render(); };
document.getElementById('reset').onclick = () => { Object.keys(filt).forEach(k => delete filt[k]); sort = {k: 'n', dir: 1};
  show = 'all'; document.querySelectorAll('#show button').forEach(x => x.classList.toggle('on', x.dataset.v === 'all')); render(); };
document.getElementById('metric').onchange = () => render();
function resize() {
  const h = document.documentElement.scrollHeight + 4;
  window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setFrameHeight', height: h}, '*');
}
"""

OVERLAY = """<div id="ovl"><div id="dlg"><div class="dh"><b id="s3t"></b><span class="faint" id="s3s"></span>
<button class="x" id="s3x" title="Close (Esc)">&times;</button></div><div class="db"><div><div id="stage">
<div id="cams"></div><div class="msg" id="s3msg">Loading 3D…</div></div><div id="ctl"><button id="s3p">❚❚</button>
<select id="s3sp"><option value="2">1/2 speed</option><option value="4" selected>1/4</option>
<option value="10">1/10</option><option value="25">1/25</option></select><span class="scw"><input id="s3sc" type="range"
min="0" max="1000" value="0"><span id="s3tick" class="tick" title="Front of the plate"></span></span>
<button id="s3k" title="Jump to the front of the plate">Plate</button><button id="s3e" title="Jump to the end">End</button>
<span class="clk" id="s3c"></span><a id="s3v" target="_blank" rel="noopener">Video</a></div></div>
<div id="rd"></div></div></div></div>"""

P3D_JS = r"""
let stage = document.getElementById('stage'), R3 = null;
const S3 = JSON.parse(document.getElementById('s3d-data').textContent);
function loadThree(cb) {
  if (window.THREE) return cb();
  const s = document.createElement('script'); s.src = THREE_SRC; s.onload = cb;
  s.onerror = () => { document.getElementById('s3msg').textContent = "Couldn't load the 3D library (needs an internet connection)."; };
  document.head.appendChild(s);
}
const V = p => new THREE.Vector3(p[0], p[2], -p[1]);
const T = q => q[q.length - 1];
function at(pts, t) {
  if (t <= T(pts[0])) return pts[0];
  for (let k = 1; k < pts.length; k++) if (T(pts[k]) >= t) {
    const a = pts[k - 1], b = pts[k], f = (t - T(a)) / ((T(b) - T(a)) || 1);
    return [0, 1, 2].map(i => a[i] + (b[i] - a[i]) * f);
  }
  return pts[pts.length - 1];
}
const pc = v => ok(v) ? Math.round(v * 100) + '%' : '–';
function readouts(sc) {
  const r = sc.read, row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  const side = ok(r.gx) ? (Math.abs(r.gx) < 0.05 ? 'middle' : `${Math.abs(r.gx * 12).toFixed(0)}″ to the catcher's ${r.gx > 0 ? 'right' : 'left'}`) : '–';
  document.getElementById('rd').innerHTML =
    '<h5>Pitch</h5><table>' + row(r.ptype + ' velo', fmt(r.velo, 1) + ' mph') + row('HB / IVB', fmt(r.hb, 1) + '″ / ' + fmt(r.ivb, 1) + '″') +
    row('Pitcher', r.pitcher) + row('Batter', r.batter) + row('Count', r.count) + row('Runners', r.on) + '</table>' +
    '<h5>Where</h5><table>' + row('Zone', r.zone || '–') + row('Region', r.region || '–') +
    (r.dirt ? row('Hits the dirt', `${r.gy >= 0 ? fmt(r.gy, 1) + ' ft in front of the plate' : fmt(-r.gy, 1) + ' ft past the front edge'}`) + row('Side to side', side)
            : row('Hits the dirt', 'no -- reaches the catcher in the air')) + '</table>' +
    '<h5>What happened</h5><table>' + row('Pitch', r.result) + (r.play ? row('Play', r.play) : '') + '</table>' +
    '<h5>Catcher models</h5><table>' + row('Strike chance (avg umpire)', pc(r.strike_prob)) +
    (ok(r.frame_runs) ? row('Framing runs', fmt(r.frame_runs, 'runs')) : '') +
    (r.opp ? row('Chance it gets by', ok(r.wp_chance) ? (r.wp_chance * 100).toFixed(1) + '%' : '–') + row('Blocking runs', fmt(r.block_runs, 'runs')) : row('Blocking chance', 'no -- bases empty')) +
    (r.abs ? row('His ABS challenge', `<b style="color:${r.abs === 'Successful' ? '#2e9d5b' : '#c0392b'}">${r.abs}</b>`) :
      r.abs_by ? row('ABS challenge', 'by the ' + r.abs_by.toLowerCase()) : '') +
    (r.steal ? row('Steal of ' + r.steal + ' · CS chance', pc(r.cs_chance)) + row('Throwing runs', fmt(r.throw_runs, 'runs')) : '') + '</table>' +
    '<div class="lg"><i style="color:#e07b24">━</i> measured flight (Statcast release speed + movement, anchored where it crossed the plate), ' +
    'from release to ' + (r.dirt ? 'where it reaches the dirt <i style="color:#c0392b">◯</i>' : '3 ft past the middle of the plate') +
    '. Dashed box = his strike zone at the front of the plate.</div>';
}
function build(sc) {
  const W = stage.clientWidth, H = stage.clientHeight;
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0xf4f7f8);
  const cam = new THREE.PerspectiveCamera(40, W / H, 1, 20000);
  const ren = new THREE.WebGLRenderer({antialias: true}); ren.setSize(W, H); ren.setPixelRatio(Math.min(2, window.devicePixelRatio));
  stage.appendChild(ren.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x99aabb, 0.95));
  const dl = new THREE.DirectionalLight(0xffffff, 0.5); dl.position.set(100, 300, 200); scene.add(dl);
  const L = (pts, mat) => { const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map(V)), mat);
    if (mat.isLineDashedMaterial) l.computeLineDistances(); return l; };
  const dark = new THREE.LineBasicMaterial({color: 0x16202b});
  const shp = new THREE.Shape([[-8.5, 0], [8.5, 0], [8.5, -8.5], [0, -17], [-8.5, -8.5]].map(([x, y]) => new THREE.Vector2(x, y)));
  const plate = new THREE.Mesh(new THREE.ShapeGeometry(shp), new THREE.MeshBasicMaterial({color: 0xdde4e8, side: THREE.DoubleSide}));
  plate.rotation.x = -Math.PI / 2; plate.position.y = 0.2; scene.add(plate);
  scene.add(L([[-8.5, 0, .3], [8.5, 0, .3], [8.5, -8.5, .3], [0, -17, .3], [-8.5, -8.5, .3], [-8.5, 0, .3]], dark));
  for (const sg of [-1, 1]) scene.add(L([[sg * 14.5, -44.5, .3], [sg * 62.5, -44.5, .3], [sg * 62.5, 27.5, .3], [sg * 14.5, 27.5, .3], [sg * 14.5, -44.5, .3]], dark));
  scene.add(L([[-22, -51 - 43, .3], [-22, -51, .3], [22, -51, .3], [22, -51 - 43, .3]], dark));          // catcher's box
  scene.add(L([[-8.5, 0, sc.sz[0]], [8.5, 0, sc.sz[0]], [8.5, 0, sc.sz[1]], [-8.5, 0, sc.sz[1]], [-8.5, 0, sc.sz[0]]],
    new THREE.LineDashedMaterial({color: 0x16202b, dashSize: 2, gapSize: 1.5})));
  const midBack = 8.5 - 17 - 36;                                                                            // 3 ft past the middle
  scene.add(L([[-30, midBack, .4], [30, midBack, .4]], new THREE.LineDashedMaterial({color: 0x9fb9be, dashSize: 3, gapSize: 2})));
  const P = sc.pitch;
  scene.add(L(P, new THREE.LineBasicMaterial({color: 0xe07b24, transparent: true, opacity: .25})));
  const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: 0xe07b24})); scene.add(trail);
  const ball = new THREE.Mesh(new THREE.SphereGeometry(1.45, 20, 14), new THREE.MeshLambertMaterial({color: 0xffffff, emissive: 0x332211}));
  scene.add(ball);
  const last = P[P.length - 1];
  if (sc.dirt) { const ring = new THREE.Mesh(new THREE.RingGeometry(2.4, 3.4, 32), new THREE.MeshBasicMaterial({color: 0xc0392b, side: THREE.DoubleSide}));
    ring.rotation.x = -Math.PI / 2; ring.position.copy(V([last[0], last[1], 0.5])); scene.add(ring); }
  // camera
  const tgt = V([0, 40, 20]), st = {th: 0, ph: 1.38, r: 190};
  const PRESETS = {Catcher: {th: 0, ph: 1.38, r: 190}, '3B side': {th: -Math.PI / 2, ph: 1.4, r: 230},
                   '1B side': {th: Math.PI / 2, ph: 1.4, r: 230}, Overhead: {th: 0.0001, ph: 0.06, r: 330},
                   Pitcher: {th: Math.PI, ph: 1.42, r: 260}};
  function place() { cam.position.set(tgt.x + st.r * Math.sin(st.ph) * Math.sin(st.th), tgt.y + st.r * Math.cos(st.ph),
                                      tgt.z + st.r * Math.sin(st.ph) * Math.cos(st.th)); cam.lookAt(tgt); }
  const cams = document.getElementById('cams'); cams.innerHTML = '';
  Object.entries(PRESETS).forEach(([k, v], i) => { const b = document.createElement('button'); b.textContent = k; if (!i) b.classList.add('on');
    b.onclick = () => { Object.assign(st, v); place(); cams.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); }; cams.appendChild(b); });
  let drag = null;
  ren.domElement.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; ren.domElement.setPointerCapture(e.pointerId); });
  ren.domElement.addEventListener('pointermove', e => { if (!drag) return;
    st.th -= (e.clientX - drag[0]) * 0.008; st.ph = Math.max(0.05, Math.min(1.55, st.ph - (e.clientY - drag[1]) * 0.008));
    drag = [e.clientX, e.clientY]; place(); cams.querySelectorAll('button').forEach(x => x.classList.remove('on')); });
  ren.domElement.addEventListener('pointerup', () => { drag = null; });
  ren.domElement.addEventListener('wheel', e => { e.preventDefault(); st.r = Math.max(40, Math.min(2000, st.r * (1 + e.deltaY * 0.001))); place(); }, {passive: false});
  place();
  const t0 = T(P[0]), t1 = T(last), S = {t: t0, playing: true, last: null, raf: 0, hold: 0};
  const sc_ = document.getElementById('s3sc');
  function frame(t) {
    const pts = P.filter(q => T(q) <= t).map(V); pts.push(V(at(P, t)));
    trail.geometry.dispose(); trail.geometry = new THREE.BufferGeometry().setFromPoints(pts.length > 1 ? pts : [V(P[0]), V(P[0])]);
    ball.position.copy(V(at(P, t)));
    document.getElementById('s3c').textContent = `${t >= 0 ? '+' : ''}${(t * 1000).toFixed(0)} ms from the plate`;
    ren.render(scene, cam);
  }
  function tick(now) {
    if (S.playing && S.last !== null) {
      if (S.t >= t1) { S.hold += now - S.last; if (S.hold > 900) { S.t = t0; S.hold = 0; } }
      else S.t = Math.min(t1, S.t + (now - S.last) / 1000 / +document.getElementById('s3sp').value);
      sc_.value = Math.round((S.t - t0) / (t1 - t0) * 1000);
    }
    S.last = now; frame(S.t); S.raf = requestAnimationFrame(tick);
  }
  S.raf = requestAnimationFrame(tick);
  const pause = () => { S.playing = false; document.getElementById('s3p').textContent = '▶'; };
  document.getElementById('s3p').onclick = e => { S.playing = !S.playing; e.target.textContent = S.playing ? '❚❚' : '▶'; };
  document.getElementById('s3k').onclick = () => { pause(); S.t = Math.max(t0, Math.min(t1, 0)); sc_.value = Math.round((S.t - t0) / (t1 - t0) * 1000); };
  document.getElementById('s3e').onclick = () => { pause(); S.t = t1; sc_.value = 1000; };
  document.getElementById('s3tick').style.left = (100 * Math.max(0, Math.min(1, (0 - t0) / (t1 - t0)))) + '%';
  sc_.oninput = e => { pause(); S.t = t0 + (t1 - t0) * e.target.value / 1000; };
  return {stop() { cancelAnimationFrame(S.raf); ren.dispose(); ren.domElement.remove(); }};
}
const ovl = document.getElementById('ovl'), dlg = document.getElementById('dlg');
function open3d(n, btn) {
  const sc = S3[n]; if (!sc) return;
  const r = sc.read;
  document.getElementById('s3t').textContent = `${r.inning} · ${r.count} ${r.ptype} ${fmt(r.velo, 1)} · ${r.result}`;
  document.getElementById('s3s').textContent = 'drag to orbit · scroll to zoom';
  const v = document.getElementById('s3v');
  if (r.video) { v.href = 'https://baseballsavant.mlb.com/sporty-videos?playId=' + r.video; v.style.display = ''; } else v.style.display = 'none';
  readouts(sc);
  const docH = Math.max(document.documentElement.scrollHeight, 760);
  ovl.style.height = docH + 'px'; ovl.classList.add('on');
  const rowTop = btn.getBoundingClientRect().top + window.scrollY;
  dlg.style.top = Math.max(10, Math.min(rowTop - 160, docH - dlg.offsetHeight - 10)) + 'px';
  const msg = document.getElementById('s3msg'); msg.style.display = '';
  loadThree(() => { msg.style.display = 'none'; if (R3) R3.stop(); R3 = build(sc); });
}
function close3d() { ovl.classList.remove('on'); if (R3) { R3.stop(); R3 = null; } }
document.getElementById('s3x').onclick = close3d;
ovl.addEventListener('click', e => { if (e.target === ovl) close3d(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') { if (ovl.classList.contains('on')) close3d(); closeMenu(); } });
"""

METRICS = {"framing": "Framing", "blocking": "Blocking", "throwing": "Throwing", "abs": "ABS challenges",
           "total": "Total"}


def component_html(p: pd.DataFrame, line: dict, header: str = "", scenes3d: dict | None = None,
                   has_model: bool = True, comps: dict | None = None) -> str:
    from ui.aar import CSS
    scenes3d = scenes3d or {}
    rs = rows(p, scenes3d)
    opts = "".join(f'<option value="{k}">{v}</option>' for k, v in METRICS.items())
    note = ("" if has_model else '<div class="note">Model columns fill in once the catcher models are trained on '
            'this computer (<code>python -m pipeline.train_catching</code> + the swing-decision model).</div>')
    left = (f'<div class="left"><div class="card">{line_html(line)}</div>'
            f'<div class="card tiles"><div class="hd"><span class="lab">Game stats '
            f'<span class="faint">· follow the filters · hover to compare</span></span><select id="metric">{opts}</select></div>'
            f'<div class="row" id="tiles"></div>{note}</div></div>')
    cmode = ('<div class="cmode"><div class="seg"><button class="on" data-v="opp">Blocking chances</button>'
             '<button data-v="all">Every pitch in the dirt</button></div></div>')
    charts = (f'<div class="card"><div class="charts">{framing_svg(rs, p)}{blocking_svg(rs, p)}{ground_svg(rs)}</div>'
              f'{cmode}</div>')
    pitchers = [x for x in dict.fromkeys(p["pitcher"].astype(str)) if x]
    pch = "".join(f'<button class="pchip on" data-v="{escape(x)}">{escape(x)}</button>' for x in pitchers)
    show = ('<div class="seg" id="show">' + "".join(
        f'<button data-v="{k}"{" class=on" if k == "all" else ""}>{v}</button>' for k, v in
        (("all", "All"), ("take", "Takes"), ("swing", "Swings"), ("opp", "Blocking chances"), ("dirt", "In the dirt"),
         ("steal", "Steal attempts"), ("abs", "ABS challenges"))) + '</div>')
    bar = ('<div class="bar"><span class="ttl">Pitches received</span><span class="sel-n" id="seln"></span>'
           '<span class="faint" id="shown"></span><button id="reset">Clear filters + sort</button>'
           '<button id="clear">Clear selection</button></div>'
           f'<div class="qf"><span class="lab">Pitchers</span>{pch}<span class="sep"></span>'
           f'<span class="lab">Show</span>{show}<span class="sep"></span><span id="fls"></span></div>'
           '<div class="qf"><span class="faint">Click any column header to sort it or filter it by its values '
           '(Zone, Result, Pitcher …) or a range (Velo, Strike %, runs …). Double-click a pitch to play its video. Balls in play are left out.</span></div>')
    data = json.dumps(rs).replace("</", "<\\/")
    s3 = json.dumps({str(k): v for k, v in scenes3d.items()}).replace("</", "<\\/")
    cj = json.dumps(comps or {}).replace("</", "<\\/")
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}{CSS_EXTRA}</style></head><body>'
            f'<div id="aar" style="position:relative">{header}<div class="top">{left}{charts}</div>{bar}'
            f'<div class="wrap"><table class="p" id="tbl"></table></div><div id="menu"></div><div id="ctip"></div></div>{OVERLAY}'
            f'<template id="ico-cam">{CAMERA}</template><template id="ico-cube">{CUBE}</template>'
            f'<script type="application/json" id="rows-data">{data}</script>'
            f'<script type="application/json" id="s3d-data">{s3}</script>'
            f'<script type="application/json" id="pt-data">{json.dumps(PT_COLOR)}</script>'
            f'<script type="application/json" id="comps-data">{cj}</script>'
            f'<script type="application/json" id="model-data">{json.dumps(bool(has_model))}</script>'
            f'<script>const THREE_SRC = "{THREE_JS}";{JS}{P3D_JS}render();window.addEventListener("load", resize);'
            f'</script></body></html>')


def estimate_height(n: int) -> int:
    return 640 + 33 * n + 60
