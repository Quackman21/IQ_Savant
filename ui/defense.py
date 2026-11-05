"""Defense tables (same look as Hitting / Core 4: hover headers, bold Season rows)."""
from __future__ import annotations

import math
from html import escape

from core import defense as dfn
from ui.core4 import SEASON_CSS
from ui.hitting import CSS, GREEN, RED, _mix


def _fmt(v, f: str) -> str:
    if f in ("blank", "grade"):
        return ""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if f == "int":
        return f"{int(round(v)):,}"
    if f == "num1":
        return f"{v:.1f}"
    if f == "num2":
        return f"{v:.2f}"
    if f == "pct":
        return f"{100 * v:.1f}%"
    if f == "time":
        return f"{v:.2f}s"
    if f == "mph":
        return f"{v:.1f}mph"
    if f == "pctile":
        n = int(round(v))
        suf = "th" if 10 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"
    return str(v)


def _pct_style(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    z = (p - 50) / 50
    return f"background:{_mix(GREEN if z > 0 else RED, min(abs(z), 1) * 0.8)}"


def table_html(spec: list, rows: list[dict]) -> str:
    """rows = [{"lead": {...LEAD...}, "vals": {key: value}, "pct": {key: percentile}, "season_row": bool}]"""
    if not rows:
        return CSS + '<div class="ht-note">No games at these positions in the selected periods.</div>'
    cols, seps, first = [], set(), True
    for item in spec:
        if item == "|":
            first = True
            continue
        if first:
            seps.add(len(cols))
            first = False
        cols.append(item)
    n = len(cols)
    head = "<tr>" + "".join(f'<th style="text-align:left;cursor:default">{h}</th>' for h in dfn.LEAD)
    head += "".join(f'<th class="{"sep " if i in seps else ""}{"r" if i > n / 2 else ""}" '
                    f'data-tip="{escape(desc)}">{escape(label)}</th>'
                    for i, (_, label, desc, _f) in enumerate(cols)) + "</tr>"
    body = ""
    for r in rows:
        tr = ' class="season"' if r.get("season_row") else ""
        body += f"<tr{tr}>" + "".join(f'<td class="lbl">{escape(str(r["lead"].get(h, "")))}</td>' for h in dfn.LEAD)
        for i, (key, _, _, f) in enumerate(cols):
            v = r["vals"].get(key)
            style = _pct_style(v) if f == "pctile" else _pct_style(r.get("pct", {}).get(key))
            body += f'<td class="{"sep" if i in seps else ""}" style="{style}">{escape(_fmt(v, f))}</td>'
        body += "</tr>"
    return CSS + SEASON_CSS + f'<div class="ht"><table class="pin">{head}{body}</table></div>'
