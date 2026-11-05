"""Hitting tables: grouped Overview + Rates / Contact / Exit.

* Hover a column header -> its definition (from the stat catalog).
* Click a number -> popup (bottom-right) with MLB AVG, the player, Delta and
  STDEVS. Pure HTML/CSS: no extra code runs in the browser.
* Cells are coloured by STDEVS: green = better / red = worse for skill stats,
  yellow = more / purple = less for style stats (LA, GB%, Pull% ...).
"""
from __future__ import annotations

import math
from html import escape

from core import hit_metrics as hm

CSS = """
<style>
.ht{overflow-x:auto;-webkit-overflow-scrolling:touch;padding-bottom:96px;margin-bottom:-84px}
.ht table{border-collapse:collapse;width:100%;white-space:nowrap;font-family:system-ui,sans-serif;color:#16202b;
  border:1px solid #d9dee2}
.ht th{font:700 13px system-ui,sans-serif;color:#16202b;text-align:center;padding:9px 10px;
  border-bottom:1.5px solid #9aa4ad;background:#fff;position:relative;cursor:help}
.ht th.grp{font-size:15px;letter-spacing:.4px;cursor:default;border-bottom:1px solid #d9dee2}
.ht td{font:14px ui-monospace,"SF Mono",Menlo,Consolas,monospace;text-align:center;padding:8px 10px;
  border-bottom:1px solid #e6eaed;position:relative}
.ht td.lbl{font-family:system-ui,sans-serif;text-align:left}
.ht td.cnt{color:#33404c}
.ht .sep{border-left:1.5px solid #6a7681}
.ht tr.avg td{color:#6a7681;font-style:italic;background:#fafbfb}
.ht th[data-tip]:hover::after{content:attr(data-tip);white-space:pre-line;position:absolute;top:100%;left:0;
  z-index:50;background:#5f5f5f;color:#fff;font:400 13px/1.35 system-ui,sans-serif;text-align:left;
  padding:8px 11px;border-radius:6px;min-width:240px;max-width:420px;box-shadow:0 2px 8px rgba(0,0,0,.25)}
.ht th.r[data-tip]:hover::after{left:auto;right:0}
.ht td.c{cursor:pointer}
.ht td.c:focus{outline:2px solid #16202b;outline-offset:-2px}
.ht .pop{display:none;position:fixed;right:22px;bottom:22px;z-index:9999;background:#fff;border-radius:8px;
  box-shadow:0 6px 24px rgba(0,0,0,.22);padding:12px 14px;cursor:default}
.ht td.c:focus .pop,.ht td.c:focus-within .pop{display:block}
.ht .pop table{border:1px solid #d9dee2;width:auto}
.ht .pop th,.ht .pop td{font:600 14px system-ui,sans-serif;padding:8px 16px;text-align:center;cursor:default;
  border:1px solid #e3e7ea;background:#fff;color:#16202b}
.ht .pop td:first-child{text-align:right;font-weight:700}
.ht .pop td.num{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-weight:500}
.ht .pop small{display:block;text-align:right;color:#8a949d;font:11px system-ui,sans-serif;margin-bottom:6px}
.ht-note{font:12px system-ui,sans-serif;color:#6a7681;margin:4px 0 10px}
</style>
"""

# colours: (r, g, b)
GREEN, RED, YELLOW, PURPLE = (76, 175, 80), (231, 92, 92), (246, 196, 66), (106, 82, 160)


def _mix(rgb, t: float) -> str:
    t = max(0.0, min(1.0, t))
    r, g, b = (round(255 + (c - 255) * t) for c in rgb)
    return f"rgb({r},{g},{b})"


def _z(v, ref) -> float:
    if not ref:
        return math.nan
    avg, sd = ref
    if v is None or any(isinstance(x, float) and math.isnan(x) for x in (v, avg, sd)) or not sd:
        return math.nan
    return (v - avg) / sd


def color(key: str, z: float) -> str:
    if math.isnan(z):
        return ""
    good = hm.CATALOG[key].good
    t = min(abs(z) / 2.0, 1.0) * 0.85
    if good == 0:
        rgb = YELLOW if z > 0 else PURPLE
    else:
        rgb = GREEN if z * good > 0 else RED
    bg = _mix(rgb, t)
    fg = "#fff" if (rgb == PURPLE and t > 0.55) else "#16202b"
    return f"background:{bg};color:{fg}"


def _popup(key, v, ref, z, ctx, samples, lg_samples) -> str:
    s = hm.CATALOG[key]
    avg = ref[0] if ref else math.nan
    lg_pa, lg_bip = lg_samples
    pa, bip = samples
    zcell = "-" if math.isnan(z) else f"{z:.2f}"
    return (f'<div class="pop"><small>Click elsewhere to dismiss</small><table>'
            f'<tr><th></th><th>PA</th><th>BIP</th><th>{escape(s.label)}</th><th>STDEVS</th></tr>'
            f'<tr><td>{escape(ctx["season_label"])} MLB AVG</td><td class="num">{hm.fmt("PA", lg_pa)}</td>'
            f'<td class="num">{hm.fmt("BIP", lg_bip)}</td><td class="num">{hm.fmt(key, avg)}</td><td></td></tr>'
            f'<tr><td>{escape(ctx["player"])}</td><td class="num">{hm.fmt("PA", pa)}</td>'
            f'<td class="num">{hm.fmt("BIP", bip)}</td><td class="num">{hm.fmt(key, v)}</td><td></td></tr>'
            f'<tr><td>Delta</td><td></td><td></td><td class="num">{hm.fmt_delta(key, v - avg) if ref else "-"}</td>'
            f'<td class="num" style="{color(key, z)}">{zcell}</td></tr></table></div>')


def cell(key: str, v, ref_row: dict, ctx: dict, samples, sep: bool = False) -> str:
    """One stat cell, coloured, with a click popup (counts get no popup/colour)."""
    s = hm.CATALOG[key]
    cls = "sep " if sep else ""
    if s.kind == "count":
        return f'<td class="{cls}cnt">{hm.fmt(key, v)}</td>'
    ref = (ref_row or {}).get(key)
    z = _z(v, ref)
    lg = ((ref_row or {}).get("PA", (math.nan,))[0], (ref_row or {}).get("BIP", (math.nan,))[0])
    pop = _popup(key, v, ref, z, ctx, samples, lg) if ctx.get("popups", True) else ""
    return f'<td class="{cls}c" tabindex="0" style="{color(key, z)}">{hm.fmt(key, v)}{pop}</td>'


def _th(key: str, right: bool, sep: bool = False) -> str:
    s = hm.CATALOG[key]
    cls = " ".join(c for c in ("r" if right else "", "sep" if sep else "") if c)
    return f'<th class="{cls}" data-tip="{escape(s.desc)}">{escape(s.label)}</th>'


# ----------------------------------------------------------------------------
def overview_html(label: str, vals, ref_row: dict, ctx: dict) -> str:
    """Two stacked lines per group (COUNTS | RATES | CONTACT | BIP)."""
    if vals is None:
        return CSS + '<div class="ht-note">No Statcast data for this selection.</div>'
    groups = hm.OVERVIEW
    total = sum(len(g[1]) for g in groups)
    head = '<tr><th class="grp"></th>' + "".join(
        f'<th class="grp sep" colspan="{len(l1)}">{name}</th>' for name, l1, _ in groups) + "</tr>"
    samples = (vals.get("PA"), vals.get("BIP"))
    body = ""
    for line in (1, 2):
        pos = 0
        hdr, row = '<tr><th>Year</th>', f'<tr><td class="lbl">{escape(label)}</td>'
        for name, l1, l2 in groups:
            keys = l1 if line == 1 else l2
            for i, k in enumerate(keys):
                pos += 1
                hdr += _th(k, pos > total / 2, sep=(i == 0))
                row += cell(k, vals.get(k), ref_row, ctx, samples, sep=(i == 0))
        body += hdr + "</tr>" + row + "</tr>"
    return CSS + f'<div class="ht"><table>{head}{body}</table></div>'


SPLIT_LABEL = {v: k for k, v in hm.SPLITS.items()}


def tab_html(keys: list[str], rows: list, label: str, ref_for, ctx: dict,
             zone_dim: str | None, show_avgs: bool = False) -> str:
    """rows = hm.player_rows(); ref_for(labels) -> {stat: (avg, sd)}."""
    if not rows:
        return CSS + '<div class="ht-note">No Statcast data for this selection.</div>'
    split_dims = [k for k in rows[0][0] if k != "ZONE"]
    lead = ["YEAR", "ZONE"] + [SPLIT_LABEL.get(s, s).upper() for s in split_dims]
    n = len(keys)
    head = "<tr>" + "".join(f'<th style="text-align:left">{h}</th>' for h in lead)
    head += "".join(_th(k, i > n / 2, sep=(i == 0 or k == keys[2] if len(keys) > 2 else False))
                    for i, k in enumerate(keys)) + "</tr>"
    body = ""
    for labels, vals in rows:
        ref = ref_for(labels)
        samples = (vals.get("PA"), vals.get("BIP"))
        lead_cells = f'<td class="lbl">{escape(label)}</td><td class="lbl">{escape(labels["ZONE"])}</td>' + \
            "".join(f'<td class="lbl">{escape(labels[s])}</td>' for s in split_dims)
        body += "<tr>" + lead_cells + "".join(
            cell(k, vals.get(k), ref, ctx, samples, sep=(i == 0 or (len(keys) > 2 and k == keys[2])))
            for i, k in enumerate(keys)) + "</tr>"
        if show_avgs and ref:
            body += '<tr class="avg"><td class="lbl">MLB AVG</td>' + \
                f'<td class="lbl">{escape(labels["ZONE"])}</td>' + \
                "".join(f'<td class="lbl">{escape(labels[s])}</td>' for s in split_dims) + \
                "".join(f'<td class="{"sep" if (i == 0 or (len(keys) > 2 and k == keys[2])) else ""}">'
                        f'{"" if hm.CATALOG[k].kind == "count" else hm.fmt(k, ref.get(k, (math.nan,))[0])}</td>'
                        for i, k in enumerate(keys)) + "</tr>"
    return CSS + f'<div class="ht"><table>{head}{body}</table></div>'
