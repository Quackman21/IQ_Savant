"""Core 4 tables. Same look and behaviour as the Hitting tables (hover a header
for its definition, click a number for MLB AVG / Delta / STDEVS), plus grade
columns shown on the chosen scale (20-80 / Percentile / Plus)."""
from __future__ import annotations

import math
from html import escape

from core import core4 as c4
from core import hit_metrics as hm
from ui.hitting import CSS, _popup, _z, cell, color

LEAD = ["Year", "Period", "Age", "Type", "Team", "vs"]
SEASON_CSS = """<style>
.ht tr.season td{border-top:3px solid #16202b;border-bottom:3px solid #16202b;font-weight:700}
.ht tr.season td.lbl{font-weight:700}
</style>"""


def _th(key: str, right: bool, sep: bool, scale: str) -> str:
    cls = " ".join(c for c in ("r" if right else "", "sep" if sep else "") if c)
    return (f'<th class="{cls}" data-tip="{escape(c4.desc_of(key, scale))}">'
            f'{escape(c4.label_of(key))}</th>')


def _pct_text(z: float) -> str:
    return "-" if math.isnan(z) else f"{c4.to_scale(z, 'Percentile'):.0f}"


def _cell(key: str, vals, ref: dict, ctx: dict, samples, sep: bool, scale: str, display: str) -> str:
    cls = "sep " if sep else ""
    kind = c4.kind_of(key)
    if kind == "grade":
        g = c4.GRADES[key]
        v, z = c4.grade(key, vals, ref, scale)
        if not g.stat:                                   # blank until our model exists
            return f'<td class="{cls.strip()}"></td>'
        lg = ((ref or {}).get("PA", (math.nan,))[0], (ref or {}).get("BIP", (math.nan,))[0])
        pop = _popup(g.stat, vals.get(g.stat), (ref or {}).get(g.stat), z, ctx, samples, lg)
        return (f'<td class="{cls}c" tabindex="0" style="{color(g.stat, z)}">'
                f'{c4.fmt_grade(v)}{pop}</td>')
    if kind == "derived":
        v = c4.derived(key, vals, ref)
        base = {"xwOBA MLB": "xwOBA", "xwOBACON MLB": "xwOBACON", "BALL FLIGHT 600": "xwOBACON"}[key]
        z = _z(vals.get(base), (ref or {}).get(base))
        fmt = c4.DERIVED[key][1]
        txt = "-" if math.isnan(v) else (f"{v:.3f}".replace("0.", ".", 1) if fmt == "rate3" else f"{v:+.1f}")
        if display == "Percentiles":
            txt = _pct_text(z)
        return f'<td class="{cls}" style="{color(base, z)}">{txt}</td>'
    # catalog stat
    s = hm.CATALOG[key]
    if display == "Percentiles" and s.kind != "count":
        z = _z(vals.get(key), (ref or {}).get(key))
        return f'<td class="{cls}" style="{color(key, z)}">{_pct_text(z)}</td>'
    return cell(key, vals.get(key), ref, ctx, samples, sep=sep)


def table_html(keys: list[str], rows: list[dict], ctx: dict, scale: str, display: str) -> str:
    """rows = [{"lead": {Year, Period, Age, Type, Team, vs}, "vals": Series, "ref": {stat: (avg, sd)}}]"""
    if not rows:
        return CSS + '<div class="ht-note">No Statcast data for the selected periods.</div>'
    n = len(keys)
    first_after_shared = len(c4.SHARED)
    seps = {0, 2, first_after_shared}
    head = "<tr>" + "".join(f'<th style="text-align:left;cursor:default">{h}</th>' for h in LEAD)
    head += "".join(_th(k, i > n / 2, i in seps, scale) for i, k in enumerate(keys)) + "</tr>"
    body = ""
    for r in rows:
        vals, ref = r["vals"], r["ref"]
        samples = (vals.get("PA"), vals.get("BIP"))
        ctx_r = dict(ctx, season_label=str(r["lead"]["Year"]))
        tr = ' class="season"' if r.get("season_row") else ""
        body += f"<tr{tr}>" + "".join(f'<td class="lbl">{escape(str(r["lead"].get(h, "")))}</td>' for h in LEAD)
        body += "".join(_cell(k, vals, ref, ctx_r, samples, i in seps, scale, display) for i, k in enumerate(keys))
        body += "</tr>"
    note = ""
    if display == "Percentiles":
        note = '<div class="ht-note">Percentiles vs. MLB hitters (from each stat\'s average and spread).</div>'
    return CSS + SEASON_CSS + f'<div class="ht"><table>{head}{body}</table></div>{note}'
