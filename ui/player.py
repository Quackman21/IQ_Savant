"""Player Profile display pieces: white header card and stat tables."""
from __future__ import annotations

from html import escape

HEADSHOT = ("https://img.mlbstatic.com/mlb-photos/image/upload/"
            "d_people:generic:headshot:67:current.png/w_300,q_auto:best/v1/people/{id}/headshot/67/current")
LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"

CSS = """
<style>
.pp-head{display:grid;grid-template-columns:150px 1fr 110px;gap:8px 28px;align-items:start;
  background:#fff;border:1px solid #e3e7ea;border-radius:6px;padding:22px 26px;
  font-family:system-ui,sans-serif;color:#16202b}
.pp-head .hs{width:150px;height:150px;border-radius:50%;object-fit:cover;background:#eef1f3;
  border:1px solid #e3e7ea}
.pp-head .nm{font-size:32px;line-height:1.1;margin:2px 0 4px}
.pp-head .nm span{font-weight:300}
.pp-head .nm b{font-weight:700;margin-left:8px}
.pp-head .tm{font-size:16px;color:#33404c;margin-bottom:14px}
.pp-head .grid{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:14px 22px}
.pp-head .f{display:flex;justify-content:space-between;align-items:baseline;gap:10px;
  border-bottom:1.5px solid #16202b;padding-bottom:3px;white-space:nowrap}
.pp-head .f small{font-size:12px;letter-spacing:.3px;color:#33404c}
.pp-head .f span{font-size:16px;overflow:hidden;text-overflow:ellipsis}
.pp-head .f span.act{color:#1e9e6a;font-weight:600}
.pp-head .entry{grid-column:1/-1;justify-content:flex-start}
.pp-head .entry span{font-size:15px;margin-left:10px}
.pp-head .logo{display:flex;flex-direction:column;align-items:center;gap:6px;font-size:12px;color:#33404c}
.pp-head .logo div{width:66px;height:66px;border-radius:50%;border:2px solid #4a8b96;
  display:flex;align-items:center;justify-content:center}
.pp-head .logo img{width:44px;height:44px;object-fit:contain}
@media (max-width:820px){
  .pp-head{grid-template-columns:1fr;justify-items:start}
  .pp-head .grid{grid-template-columns:repeat(2,minmax(120px,1fr))}
  .pp-head .logo{display:none}}
.pp-t{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:4px 0 12px}
.pp-t table{border-collapse:collapse;width:100%;white-space:nowrap;font-family:system-ui,sans-serif;color:#16202b}
.pp-t th{font:600 11px system-ui,sans-serif;color:#6a7681;text-align:right;padding:6px 9px;
  border-bottom:1.5px solid #4a8b96}
.pp-t td{font:13px ui-monospace,"SF Mono",Menlo,Consolas,monospace;text-align:right;padding:5px 9px;
  border-bottom:1px solid #e6eaed}
.pp-t th:first-child,.pp-t td:first-child{text-align:left;font-family:system-ui,sans-serif;position:sticky;left:0;background:#fff}
.pp-t tr.avg td{color:#6a7681;font-style:italic}
.pp-pct{display:grid;grid-template-columns:150px 1fr 40px;gap:6px 12px;align-items:center;
  font:13px system-ui,sans-serif;color:#16202b;max-width:640px;margin:6px 0 14px}
.pp-pct .bar{height:10px;background:#eef1f3;border-radius:5px;overflow:hidden}
.pp-pct .bar i{display:block;height:100%;border-radius:5px}
.pp-pct b{font:600 13px ui-monospace,monospace;text-align:right}
.pp-empty{color:#6a7681;font:14px system-ui,sans-serif;padding:8px 0}
</style>
"""


def header_html(pid: int, b: dict, team: dict | None, level: str, entry: str) -> str:
    team = team or {}
    tid = b.get("team_id")
    team_line = f'{escape(level)} | {escape(b.get("team_name") or "Free Agent")}' if tid else "Free Agent"

    def f(label, value, cls=""):
        return (f'<div class="f"><small>{label}</small>'
                f'<span class="{cls}">{escape(str(value))}</span></div>')
    grid = "".join([
        f("POS", b["pos"]), f("H/W", b["hw"]),
        f("STATUS", b["status"], "act" if b["status"] == "ACTIVE" else ""), f("B/T", b["bt"]),
        f("AGE", b["age"]), f("DOB", b["dob"]), f("MLB DEBUT", b["debut"]), f("BORN", b["born"]),
    ])
    logo = (f'<div class="logo"><div><img src="{LOGO.format(id=tid)}" alt=""></div>'
            f'{escape(team.get("abbr", ""))}</div>') if tid else "<div></div>"
    return (CSS + f'<div class="pp-head"><img class="hs" src="{HEADSHOT.format(id=pid)}" alt="">'
            f'<div><div class="nm"><span>{escape(b["first"])}</span><b>{escape(b["last"])}</b></div>'
            f'<div class="tm">{team_line}</div>'
            f'<div class="grid">{grid}<div class="f entry"><small>ENTRY</small>'
            f'<span>{escape(entry or "-")}</span></div></div></div>{logo}</div>')


def table_html(rows: dict[str, dict], first_col: str = "", avg_rows: set | None = None) -> str:
    """{row label: {column: value}} -> table. Columns come from the first row."""
    rows = {k: v for k, v in rows.items() if v}
    if not rows:
        return CSS + '<div class="pp-empty">No data for this selection.</div>'
    cols = list(next(iter(rows.values())).keys())
    head = "".join(f"<th>{escape(c)}</th>" for c in cols)
    body = "".join(
        f'<tr class="{"avg" if avg_rows and k in avg_rows else ""}"><td>{escape(str(k))}</td>'
        + "".join(f"<td>{escape(str(r.get(c, '-')))}</td>" for c in cols) + "</tr>"
        for k, r in rows.items())
    return CSS + f'<div class="pp-t"><table><thead><tr><th>{escape(first_col)}</th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _pct_color(p: float) -> str:
    """Savant-style: blue (low) -> grey -> red (high)."""
    p = max(0, min(100, p)) / 100
    lo, mid, hi = (50, 92, 214), (180, 180, 180), (214, 46, 46)
    a, b, t = (lo, mid, p * 2) if p < .5 else (mid, hi, (p - .5) * 2)
    return "rgb({},{},{})".format(*(round(x + (y - x) * t) for x, y in zip(a, b)))


def percentile_html(pcts: dict[str, float]) -> str:
    if not pcts:
        return CSS + '<div class="pp-empty">No percentile ranks for this selection.</div>'
    rows = "".join(f'<span>{escape(k)}</span><div class="bar"><i style="width:{v:.0f}%;'
                   f'background:{_pct_color(v)}"></i></div><b>{v:.0f}</b>' for k, v in pcts.items())
    return CSS + f'<div class="pp-pct">{rows}</div>'


def empty_html(msg: str) -> str:
    return CSS + f'<div class="pp-empty">{escape(msg)}</div>'
