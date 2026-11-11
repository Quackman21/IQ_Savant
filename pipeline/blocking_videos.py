"""Video links for blocking chances -- check the 5% "genuine opportunity" line with your own eyes.

    python -m pipeline.blocking_videos                          # 30 random pitches right around the 5% line (3-8%)
    python -m pipeline.blocking_videos --tier tough --result blocked    # the big stops
    python -m pipeline.blocking_videos --tier tough --result missed     # tough ones that got away
    python -m pipeline.blocking_videos --min 0.02 --max 0.05 --n 40      # just under the line: are these real chances?
    python -m pipeline.blocking_videos --season 2025

  --tier easy | medium | tough     (Statcast lines: easy < 5%, medium 5-15%, tough 15%+)
  --result blocked | missed        (missed = wild pitch or passed ball)
  --min / --max                    WP+PB chance range (0-1)

Output: models/artifacts/blocking_video_check.html (opens in your browser) + .csv
Needs: the saved season (data/raw), its runner events (pipeline.fetch_events) and the trained blocking model.
"""
from __future__ import annotations

import sys
import webbrowser
from html import escape
from pathlib import Path

import pandas as pd

ART = Path(__file__).resolve().parent.parent / "models" / "artifacts"
CSV = ART / "blocking_video_check.csv"
HTML = ART / "blocking_video_check.html"
BASES = {0: "empty", 1: "1st", 2: "2nd", 3: "1st+2nd", 4: "3rd", 5: "1st+3rd", 6: "2nd+3rd", 7: "loaded"}


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def sample(argv) -> pd.DataFrame:
    from models import catching as cm
    from pipeline import build_league_reference as bl
    from pipeline import fetch_events
    from pipeline.train_decisions import latest_full_season
    year = int(_arg(argv, "--season", latest_full_season()))
    b = cm.load()
    if b is None:
        raise SystemExit("Blocking model not trained yet -- run  python -m pipeline.train_catching  first.")
    print(f"Scoring {year} blocking chances…", flush=True)
    raw = bl.pull_season(year)
    if "game_type" in raw:
        raw = raw[raw["game_type"].astype(str).eq("R")]
    df = raw[cm.block_rows(raw)].reset_index(drop=True)
    events = fetch_events.season_events(year)
    parts = [cm.block_table(df[m], events, sub) for m, sub in cm._bundles(df, b)
             if sub.get("block") is not None and m.any()]          # skip the era with no pitches this season
    if not parts:
        raise SystemExit(f"No blocking chances scored for {year}.")
    t = pd.concat(parts)
    lo = float(_arg(argv, "--min", 0.03 if not _arg(argv, "--tier") else 0))
    hi = float(_arg(argv, "--max", 0.08 if not _arg(argv, "--tier") else 1))
    keep = t["p"].between(lo, hi)
    if _arg(argv, "--tier"):
        keep &= t["tier"].eq(_arg(argv, "--tier"))
    if _arg(argv, "--result"):
        keep &= t["y"].eq(1 if _arg(argv, "--result") == "missed" else 0)
    if keep.sum() == 0:
        raise SystemExit("No pitches match those filters.")
    n = int(_arg(argv, "--n", 30))
    t = t[keep].sample(min(n, int(keep.sum())), random_state=int(_arg(argv, "--seed", 7))).sort_values("p")
    d = df.loc[t.index]
    try:
        from core import mlb_api
        names = mlb_api.get_people(sorted(d["fielder_2"].dropna().astype(int).unique().tolist()))
    except Exception:
        names = {}
    return pd.DataFrame({
        "date": d["game_date"].astype(str).str[:10], "game_pk": d["game_pk"].astype(int),
        "at_bat_number": d["at_bat_number"].astype(int), "pitch_number": d["pitch_number"].astype(int),
        "catcher": d["fielder_2"].map(lambda p: names.get(int(p), p) if p == p else ""),
        "runners_outs": [f"{BASES.get(int(bc), bc)}, {int(o)} out" for bc, o in
                         zip(cm.base_code(d), pd.to_numeric(d["outs_when_up"], errors="coerce").fillna(0))],
        "pitch_type": d["pitch_type"], "mph": pd.to_numeric(d["release_speed"], errors="coerce").round(1),
        "height_in": (pd.to_numeric(d["plate_z"], errors="coerce") * 12).round(0),
        "zone": t["zone"], "description": d["description"],
        "wp_pb_chance": t["p"], "tier": t["tier"], "result": t["y"].map({1: "WP/PB", 0: "blocked"}),
        "wp_pb_cost": t["val"].round(2), "credit_runs": t["runs"].round(3)})


def write_page(d: pd.DataFrame, title: str) -> None:
    cols = ["date", "catcher", "runners_outs", "pitch_type", "mph", "height_in", "zone", "description",
            "wp_pb_chance", "tier", "result", "wp_pb_cost", "credit_runs"]
    color = {"blocked": "#1a7f37", "WP/PB": "#c8102e"}
    head = "".join(f"<th>{escape(c.replace('_', ' '))}</th>" for c in cols) + "<th>video</th>"
    rows = ""
    for _, r in d.iterrows():
        cells = ""
        for c in cols:
            v = r[c]
            txt = "" if pd.isna(v) else (f"{v:.1%}" if c == "wp_pb_chance" else
                                         f"{v:+.3f}" if c == "credit_runs" else str(v))
            style = f' style="color:{color.get(str(v), "inherit")};font-weight:700"' if c == "result" else ""
            cells += f"<td{style}>{escape(txt)}</td>"
        vid = r.get("video")
        has = isinstance(vid, str) and vid.startswith("http")
        link = vid if has else f"https://baseballsavant.mlb.com/gamefeed?gamePk={int(r['game_pk'])}"
        rows += f'<tr>{cells}<td><a href="{escape(link)}" target="_blank">{"Watch" if has else "Game page"}</a></td></tr>'
    HTML.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>body{{font-family:-apple-system,Segoe UI,sans-serif;background:#fff;color:#16202b;margin:24px}}
table{{border-collapse:collapse;font-size:13px}}th,td{{padding:6px 10px;border-bottom:1px solid #e6eaed;
text-align:left;white-space:nowrap}}th{{background:#f4f6f8;text-transform:uppercase;font-size:11px}}
a{{color:#0b5cad;font-weight:700}}p{{color:#555;max-width:900px}}</style></head><body>
<h2>{escape(title)}</h2>
<p>WP/PB chance = how often an average catcher lets this pitch get away (pitch, location, count, hands).
Genuine opportunity = 5%+ (Statcast's Medium + Tough). Zone is from the catcher's view: glove = his left,
arm = his right. WP/PB cost = runs a wild pitch costs in that base / out situation. Credit = (chance - what
happened) x cost -- what goes into DRS. Height = inches above the ground as it crossed the plate.</p>
<table><tr>{head}</tr>{rows}</table></body></html>""")


def main(argv: list[str]) -> None:
    from pipeline.decision_videos import SAVANT_VIDEO, play_ids
    d = sample(argv)
    bits = [f"{k[2:]} {_arg(argv, k)}" for k in ("--tier", "--result", "--min", "--max") if _arg(argv, k)]
    title = "Blocking check: " + (" / ".join(bits) if bits else "right around the 5% line (3-8% chance)")
    cache: dict = {}
    links = []
    for _, r in d.iterrows():
        gp = int(r["game_pk"])
        if gp not in cache:
            try:
                cache[gp] = play_ids(gp)
            except Exception:
                cache[gp] = {}
        pid = cache[gp].get((int(r["at_bat_number"]), int(r["pitch_number"])))
        links.append(SAVANT_VIDEO.format(pid) if pid else "")
    d["video"] = links
    ART.mkdir(parents=True, exist_ok=True)
    d.to_csv(CSV, index=False)
    write_page(d, title)
    print(f"{len(d)} pitches, {sum(bool(x) for x in links)} with video. Saved {HTML.name} + {CSV.name}")
    try:
        webbrowser.open(HTML.as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
