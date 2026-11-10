"""Video check for IQ xwOBA: outfield balls in the air, every version's xwOBA side by side.

Runs automatically at the end of  python -m pipeline.train_xwoba --compare  (which saves every tested
version's score for every batted ball). Pull a new sample any time without re-running:

    python -m pipeline.xwoba_videos                          # 32 outfield air balls: 8 outs, 8 1B, 8 2B, 8 XBH+HR
    python -m pipeline.xwoba_videos --n 40 --result double   # only doubles
    python -m pipeline.xwoba_videos --direction pull --min-ev 100     # pulled, 100+ mph
    python -m pipeline.xwoba_videos --direction center --result field_out

Output: models/artifacts/xwoba_video_check.html (opens in your browser) + .csv
Every number is from a model that did NOT see that season (same as the comparison report).
"""
from __future__ import annotations

import sys
import webbrowser
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCORED = ROOT / "data" / "train" / "compare_scored.csv.gz"
ART = ROOT / "models" / "artifacts"
HTML = ART / "xwoba_video_check.html"
CSV = ART / "xwoba_video_check.csv"
SAVANT_VIDEO = "https://baseballsavant.mlb.com/sporty-videos?playId={}"
SAVANT_GAME = "https://baseballsavant.mlb.com/gamefeed?gamePk={}"
FT_PER_UNIT = 2.5                       # Statcast hit-coordinate units -> feet (approx.)
OUTFIELD_FT = 200                       # landed / caught this far out = outfield
RESULT_GROUPS = {"out": {"field_out", "sac_fly", "force_out", "double_play", "sac_fly_double_play",
                         "fielders_choice_out", "grounded_into_double_play", "triple_play", "field_error"},
                 "single": {"single"}, "double": {"double"}, "xbh": {"triple", "home_run"}}


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def save_scored(results: list[dict]) -> None:
    """Called by the comparison: one file with every version's held-out score per batted ball."""
    keep = ["game_date", "game_pk", "at_bat_number", "batter", "stand", "events", "launch_speed",
            "launch_angle", "hc_x", "hc_y", "woba_value", "estimated_woba_using_speedangle"]
    base = results[0]["scored"]
    out = base[[c for c in keep if c in base]].copy()
    for r in results:
        if len(r["scored"]) == len(out):
            out[f"x_{r['variant']}"] = r["scored"]["x"].values
    SCORED.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SCORED, index=False)


def outfield_air(d: pd.DataFrame) -> pd.Series:
    from models import xwoba_iq as xq
    f = xq.features(d)
    dist = np.hypot(pd.to_numeric(d["hc_x"], errors="coerce") - 125.42,
                    198.27 - pd.to_numeric(d["hc_y"], errors="coerce")) * FT_PER_UNIT
    return f["la"].between(10, 50) & (dist >= OUTFIELD_FT)


def direction(d: pd.DataFrame) -> pd.Series:
    from models import xwoba_iq as xq
    p = xq.features(d)["spray_pull"]
    word = np.select([p >= xq.DIR_LINE, p <= -xq.DIR_LINE], ["Pull", "Oppo"], "Center")
    return pd.Series([f"{w} {abs(a):.0f}°" if pd.notna(a) else "" for w, a in zip(word, p)], index=d.index)


def pick(d: pd.DataFrame, argv) -> pd.DataFrame:
    m = outfield_air(d)
    if _arg(argv, "--min-ev"):
        m &= pd.to_numeric(d["launch_speed"], errors="coerce") >= float(_arg(argv, "--min-ev"))
    if _arg(argv, "--direction"):
        m &= direction(d).str.lower().str.startswith(_arg(argv, "--direction").lower())
    n = int(_arg(argv, "--n", 32))
    seed = int(_arg(argv, "--seed", 11))
    res = _arg(argv, "--result")
    if res:
        evs = RESULT_GROUPS.get(res, {res})
        pool = d[m & d["events"].isin(evs)]
        return pool.sample(min(n, len(pool)), random_state=seed)
    parts = [d[m & d["events"].isin(evs)] for evs in RESULT_GROUPS.values()]
    per = max(1, n // len(parts))
    return pd.concat([p.sample(min(per, len(p)), random_state=seed) for p in parts])


def last_pitch_ids(game_pk: int) -> dict[int, str]:
    """at_bat_number -> playId of the at-bat's last pitch (the ball in play)."""
    from pipeline.decision_videos import play_ids
    out: dict[int, tuple[int, str]] = {}
    for (ab, pn), pid in play_ids(game_pk).items():
        if ab not in out or pn > out[ab][0]:
            out[ab] = (pn, pid)
    return {ab: pid for ab, (_, pid) in out.items()}


def _desc(name: str) -> str:
    from models import xwoba_iq as xq
    return xq.describe(name)


def main(argv: list[str]) -> None:
    if not SCORED.exists():
        raise SystemExit("No saved scores yet -- run  python -m pipeline.train_xwoba --compare  first.")
    d = pd.read_csv(SCORED, low_memory=False)
    s = pick(d, argv).copy()
    if s.empty:
        raise SystemExit("No batted balls match those filters.")
    versions = [c for c in d.columns if c.startswith("x_")]
    try:
        from core import mlb_api
        names = mlb_api.get_people(s["batter"].astype(int).unique().tolist())
    except Exception:
        names = {}
    links, cache = [], {}
    for _, r in s.iterrows():
        gp = int(r["game_pk"])
        if gp not in cache:
            try:
                cache[gp] = last_pitch_ids(gp)
            except Exception as e:
                print(f"  game {gp}: couldn't reach MLB ({type(e).__name__})", flush=True)
                cache[gp] = {}
        pid = cache[gp].get(int(r["at_bat_number"]))
        links.append(SAVANT_VIDEO.format(pid) if pid else SAVANT_GAME.format(gp))
    s["batter_name"] = s["batter"].astype(int).map(names).fillna(s["batter"].astype(str))
    s["direction"] = direction(s)
    s["video"] = links
    s = s.sort_values(["events", "launch_speed"], ascending=[True, False])
    s.to_csv(CSV, index=False)

    head = ["Date", "Batter", "EV", "LA", "Direction", "Result", "Actual"] + \
           [c[2:] for c in versions] + ["Savant", "Video"]
    rows = ""
    for _, r in s.iterrows():
        vals = [r[c] for c in versions]
        cells = [str(r["game_date"])[:10], r["batter_name"], f"{r['launch_speed']:.1f}", f"{r['launch_angle']:.0f}°",
                 r["direction"], str(r["events"]).replace("_", " "), f"{r['woba_value']:.3f}"]
        tds = "".join(f"<td>{escape(c)}</td>" for c in cells)
        tds += "".join(f'<td class="x">{v:.3f}</td>' if pd.notna(v) else "<td>-</td>" for v in vals)
        sv = r.get("estimated_woba_using_speedangle")
        tds += f'<td class="x">{sv:.3f}</td>' if pd.notna(sv) else "<td>-</td>"
        label = "Watch" if "sporty" in r["video"] else "Game page"
        tds += f'<td><a href="{escape(r["video"])}" target="_blank">{label}</a></td>'
        rows += f"<tr>{tds}</tr>"
    found = sum("sporty" in v for v in s["video"])
    ART.mkdir(parents=True, exist_ok=True)
    HTML.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>xwOBA video check</title>
<style>body{{font-family:-apple-system,Segoe UI,sans-serif;background:#fff;color:#16202b;margin:24px}}
table{{border-collapse:collapse;font-size:13px}}th,td{{padding:6px 10px;border-bottom:1px solid #e6eaed;
text-align:left;white-space:nowrap}}th{{background:#f4f6f8;text-transform:uppercase;font-size:11px}}
td.x{{font-variant-numeric:tabular-nums;font-weight:600}}a{{color:#0b5cad;font-weight:700}}
p{{color:#555;max-width:980px}}</style></head><body>
<h2>xwOBA video check: outfield balls in the air</h2>
<p>Every version's expected wOBA for the same ball (scored by a model that did NOT see that season), next to
what actually happened and Savant's number. Direction is pull-adjusted (Pull / Center / Oppo, 15° lines).
Ask of each ball: which number best matches the quality and direction of what you see?</p>
<p>{"<br>".join(escape(c[2:] + ": " + _desc(c[2:])) for c in versions)}</p>
<table><tr>{"".join(f"<th>{escape(h)}</th>" for h in head)}</tr>{rows}</table></body></html>""")
    print(f"{found}/{len(s)} batted balls have a direct video link. Opening {HTML}")
    try:
        webbrowser.open(HTML.as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
