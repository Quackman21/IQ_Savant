"""Pitching display: tables (same style as Core 4 / Defense) and the chart tabs."""
from __future__ import annotations

import math
from html import escape

import pandas as pd

from core import pitch_metrics as pm
from ui.core4 import SEASON_CSS
from ui.defense import _fmt
from ui.hitting import CSS

PITCH_CSS = """<style>.ht td.pt{font-weight:800;text-align:center}
/* a season = one bold block: thick line above the first pitch row and below the last */
.ht tr.season td{border-top:1px solid #e6eaed;border-bottom:1px solid #e6eaed}
.ht tr.season.s-first td{border-top:3px solid #16202b}
.ht tr.season.s-last td{border-bottom:3px solid #16202b}
</style>"""


def _cell_fmt(v, f: str) -> str:
    if f == "pct0":
        return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.0f}%"
    if f == "clock":
        return pm.clock(v)
    return _fmt(v, f)


def table_html(spec: list, rows: list[dict], split_dims: list[str]) -> str:
    """rows = [{"year", "period", "labels": {...,"Pitch"}, "vals": {...}, "season_row": bool}]"""
    if not rows:
        return CSS + '<div class="ht-note">No pitches for this selection.</div>'
    cols, seps, first = [], set(), True
    for item in spec:
        if item == "|":
            first = True
            continue
        if first:
            seps.add(len(cols))
            first = False
        cols.append(item)
    lead = ["Year", "Period"] + [pm.SPLIT_LABEL[d] for d in split_dims] + ["Pitch"]
    n = len(cols)
    head = "<tr>" + "".join(f'<th style="text-align:left;cursor:default">{h}</th>' for h in lead)
    head += "".join(f'<th class="{"sep " if i in seps else ""}{"r" if i > n / 2 else ""}" '
                    f'data-tip="{escape(desc)}">{escape(label)}</th>'
                    for i, (_, label, desc, _f) in enumerate(cols)) + "</tr>"
    body = ""
    keyof = lambda r: (r["year"], r["period"], r.get("season_row"))
    for k, r in enumerate(rows):
        lab = r["labels"]
        pt = lab["Pitch"]
        color = pm.TEXT_ON.get(pt, pm.PITCH_COLORS.get(pt, "#16202b"))
        tr = ""
        if r.get("season_row"):
            cls = ["season"]
            if k == 0 or keyof(rows[k - 1]) != keyof(r):
                cls.append("s-first")
            if k == len(rows) - 1 or keyof(rows[k + 1]) != keyof(r):
                cls.append("s-last")
            tr = f' class="{" ".join(cls)}"'
        body += (f'<tr{tr}><td class="lbl">{r["year"]}</td><td class="lbl">{escape(str(r["period"]))}</td>'
                 + "".join(f'<td class="lbl">{escape(str(lab.get(d, "")))}</td>' for d in split_dims)
                 + f'<td class="pt" style="color:{color}">{escape(pt)}</td>')
        for i, (key, _, _, f) in enumerate(cols):
            body += f'<td class="{"sep" if i in seps else ""}">{escape(_cell_fmt(r["vals"].get(key), f))}</td>'
        body += "</tr>"
    return CSS + SEASON_CSS + PITCH_CSS + f'<div class="ht"><table class="pin">{head}{body}</table></div>'


# ----------------------------------------------------------------------------
# Charts (Altair ships with Streamlit)
# ----------------------------------------------------------------------------
def _color_scale(codes):
    codes = [c for c in codes if c]
    return dict(domain=codes, range=[pm.PITCH_COLORS.get(c, "#888888") for c in codes])


def plot_charts(d: pd.DataFrame):
    import altair as alt
    pts = d.dropna(subset=["velo", "ivb", "hb_cat"])[["pitch", "velo", "ivb", "hb_cat"]]
    if pts.empty:
        return []
    order = pts["pitch"].value_counts().index.tolist()
    color = alt.Color("pitch:N", title="Pitch", scale=alt.Scale(**_color_scale(order)), sort=order)
    base = alt.Chart(pts).mark_circle(size=22, opacity=0.45)
    tip = ["pitch", alt.Tooltip("velo:Q", format=".1f"), alt.Tooltip("ivb:Q", format=".1f"),
           alt.Tooltip("hb_cat:Q", format=".1f", title="hb")]
    out = []
    for title, x, y, xs, ys in (("Horz vs Speed", "hb_cat", "velo", "Horizontal break (in)", "Velo"),
                                ("Vert vs Speed", "ivb", "velo", "Induced vertical break (in)", "Velo"),
                                ("Horz vs Vert", "hb_cat", "ivb", "Horizontal break (in)", "IVB (in)")):
        out.append(base.encode(x=alt.X(f"{x}:Q", title=xs, scale=alt.Scale(zero=False)),
                               y=alt.Y(f"{y}:Q", title=ys, scale=alt.Scale(zero=False)),
                               color=color, tooltip=tip).properties(title=title, height=300))
    return out


def usage_chart(d: pd.DataFrame, by_count: bool):
    import altair as alt
    if d.empty:
        return None
    dd = d.dropna(subset=["hand"])
    if by_count:
        dd = dd.dropna(subset=["count"])
    keys = ["hand"] + (["count"] if by_count else [])
    tot = dd.groupby(keys).size().rename("tot")
    u = dd.groupby(keys + ["pitch"]).size().rename("n").reset_index().join(tot, on=keys)
    u["share"] = u["n"] / u["tot"]
    order = d["pitch"].value_counts().index.tolist()
    ch = alt.Chart(u).mark_bar(size=46).encode(
        x=alt.X("hand:N", title=None, sort=["vLHH", "vRHH"]),
        y=alt.Y("share:Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("pitch:N", scale=alt.Scale(**_color_scale(order)), sort=order, title="Pitch"),
        order=alt.Order("pitch:N"),
        tooltip=["hand", "pitch", alt.Tooltip("share:Q", format=".1%"), "n"])
    if by_count:
        return ch.properties(width=170, height=320).facet(
            column=alt.Column("count:N", sort=pm.COUNT_ORDER, title=None))
    return ch.properties(height=360, title="Pitch Usage")


def location_chart(d: pd.DataFrame, row_dims: list[str]):
    """Catcher's-view plate location per pitch type (dots in pitch color; grade colouring comes with our model)."""
    import altair as alt
    dd = d.dropna(subset=["plate_x", "plate_z"])
    if dd.empty:
        return None
    order = d["pitch"].value_counts().index.tolist()
    top, bot = d["sz_top"].mean(), d["sz_bot"].mean()
    if math.isnan(top) or math.isnan(bot):
        top, bot = 3.4, 1.6
    if row_dims:
        dd = dd.dropna(subset=row_dims)                 # e.g. counts outside the three count groups
        if dd.empty:
            return None
        dd = dd.assign(facet_row=dd[row_dims].astype(str).agg(" · ".join, axis=1))
    else:
        dd = dd.assign(facet_row="All")
    dots = alt.Chart().mark_circle(size=16, opacity=0.55).encode(
        x=alt.X("plate_x:Q", scale=alt.Scale(domain=[-2.2, 2.2]), title=None, axis=None),
        y=alt.Y("plate_z:Q", scale=alt.Scale(domain=[0, 5]), title=None, axis=None),
        color=alt.Color("pitch:N", scale=alt.Scale(**_color_scale(order)), legend=None))
    # strike zone drawn from the same data (faceted charts can't mix data sources): one box per panel
    zone = alt.Chart().transform_aggregate(n="count()", groupby=["pitch", "facet_row"]).mark_rect(
        fill=None, stroke="#16202b", strokeWidth=1.5).encode(
        x=alt.datum(-17 / 24), x2=alt.datum(17 / 24), y=alt.datum(float(bot)), y2=alt.datum(float(top)))
    layer = alt.layer(dots, zone, data=dd).properties(width=120, height=140)
    return layer.facet(column=alt.Column("pitch:N", sort=order, title=None),
                       row=alt.Row("facet_row:N", title=None)).resolve_scale(color="shared")


TREND_PROPS = {"Velo": "velo", "Spin": "spin", "IVB": "ivb", "HB": "hb", "Extension": "ext",
               "Release Height": "ht", "Release Side": "side", "VAA": "vaa", "Arm Angle": "arm"}


def trend_frame(d: pd.DataFrame, prop: str, mode: str, pitch: str | None) -> pd.DataFrame:
    """5-game rolling average by appearance, or 400-pitch rolling average, per pitch type."""
    col = TREND_PROPS[prop]
    dd = d.dropna(subset=[col]).sort_values(["game_date", "game_pk"])
    if pitch:
        dd = dd[dd["pitch"] == pitch]
    out = []
    for pt, g in dd.groupby("pitch"):
        if mode == "By Appearance":
            per_game = g.groupby(["game_date", "game_pk"])[col].mean().reset_index()
            per_game["value"] = per_game[col].rolling(5, min_periods=1).mean()
            per_game["x"] = pd.to_datetime(per_game["game_date"])
            out.append(per_game.assign(pitch=pt)[["x", "value", "pitch"]])
        else:
            g = g.reset_index(drop=True)
            out.append(pd.DataFrame({"x": pd.to_datetime(g["game_date"]),
                                     "value": g[col].rolling(400, min_periods=50).mean(), "pitch": pt}).dropna())
    return pd.concat(out) if out else pd.DataFrame(columns=["x", "value", "pitch"])


def trend_chart(tf: pd.DataFrame, prop: str, mode: str):
    import altair as alt
    if tf.empty:
        return None
    order = tf["pitch"].value_counts().index.tolist()
    title = f"5-Game Rolling {prop}" if mode == "By Appearance" else f"400-Pitch Rolling {prop}"
    lines = alt.Chart(tf).mark_line(point=alt.OverlayMarkDef(size=14)).encode(
        x=alt.X("x:T", title=None), y=alt.Y("value:Q", title=prop, scale=alt.Scale(zero=False)),
        color=alt.Color("pitch:N", scale=alt.Scale(**_color_scale(order)), sort=order, title="Pitch"),
        tooltip=[alt.Tooltip("x:T", title="Date"), "pitch", alt.Tooltip("value:Q", format=".2f")])
    years = sorted(tf["x"].dt.year.unique())[1:]
    if years:
        rules = alt.Chart(pd.DataFrame({"x": [pd.Timestamp(year=y, month=1, day=1) for y in years]})).mark_rule(
            color="#16202b", strokeWidth=3).encode(x="x:T")
        return (lines + rules).properties(title=title, height=360)
    return lines.properties(title=title, height=360)
