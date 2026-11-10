"""Video links for graded swing decisions -- check the model with your own eyes.

Each pitch's Savant video has an ID (playId) in MLB's game feed; this looks them up (one small call
per game) and writes a page you can click through, plus the CSV with a video column.

    python -m pipeline.decision_videos                         # the 20 pitches from the training report
    python -m pipeline.decision_videos --count 0-0 --decision take --grade bad      # filter (new sample)
    python -m pipeline.decision_videos --n 40 --grade bad      # 40 random bad decisions
    python -m pipeline.decision_videos --count 0-0 --zone sub-heart --decision take   # IQ zone filter
      zones: heart, sub-heart, edge, "edge (off)", chase, waste
    python -m pipeline.decision_videos --season 2026           # which season's saved pitches (default latest)

Output: models/artifacts/decisions_video_check.html (opens in your browser) + .csv
"""
from __future__ import annotations

import sys
import webbrowser
from html import escape
from pathlib import Path

import pandas as pd

ART = Path(__file__).resolve().parent.parent / "models" / "artifacts"
CSV = ART / "decisions_video_check.csv"
HTML = ART / "decisions_video_check.html"
SAVANT_VIDEO = "https://baseballsavant.mlb.com/sporty-videos?playId={}"
SAVANT_GAME = "https://baseballsavant.mlb.com/gamefeed?gamePk={}"


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def play_ids(game_pk: int) -> dict[tuple[int, int], str]:
    """(at_bat_number, pitch_number) -> Savant playId for every pitch in a game."""
    from core import mlb_api
    feed = mlb_api.get_game_feed(int(game_pk))
    out = {}
    for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        ab = int(play.get("about", {}).get("atBatIndex", -1)) + 1          # feed is 0-based, Statcast 1-based
        for ev in play.get("playEvents", []):
            if ev.get("isPitch") and ev.get("playId") and ev.get("pitchNumber") is not None:
                out[(ab, int(ev["pitchNumber"]))] = ev["playId"]
    return out


def new_sample(argv) -> pd.DataFrame:
    """Score the saved season and pick a filtered random sample (same columns as the report's file)."""
    from core.statcast_clean import clean
    from models import decisions_iq as dq
    from pipeline import train_decisions as td
    year = int(_arg(argv, "--season", td.latest_full_season()))
    path = td.TRAIN_DIR / f"pitches_{year}.csv.gz"
    if not path.exists():
        raise SystemExit(f"No saved pitches for {year} -- run python -m pipeline.train_decisions first.")
    if dq.load() is None:
        raise SystemExit("Decision model not trained yet -- run python -m pipeline.train_decisions first.")
    print(f"Scoring {year} pitches…", flush=True)
    df = clean(pd.read_csv(path, low_memory=False)).reset_index(drop=True)
    sc = dq.score(df)
    st = dq.state(df)
    keep = sc["z"].notna()
    if _arg(argv, "--count"):
        keep &= st["count"].eq(_arg(argv, "--count"))
    if _arg(argv, "--zone"):
        keep &= dq.iq_zone(df).str.lower().eq(_arg(argv, "--zone").lower().replace("_", " "))
    if _arg(argv, "--decision"):
        keep &= sc["action"].eq(_arg(argv, "--decision"))
    g = _arg(argv, "--grade")
    if g:
        keep &= {"good": sc["good"].eq(1), "bad": sc["good"].eq(0), "neutral": sc["good"].isna()}[g]
    n = int(_arg(argv, "--n", 20))
    if keep.sum() == 0:
        raise SystemExit("No pitches match those filters.")
    pick = sc[keep].sample(min(n, int(keep.sum())), random_state=int(_arg(argv, "--seed", 7))).index
    out_path = ART / "_tmp_sample.csv"
    td.video_sample(df.loc[pick], sc.loc[pick], out_path, n=len(pick))
    out = pd.read_csv(out_path)
    out_path.unlink(missing_ok=True)
    return out


def write_page(d: pd.DataFrame, title: str) -> None:
    cols = [c for c in ["date", "batter", "inning", "count", "zone", "runners_outs", "pitch_type", "mph", "decision",
                        "what_happened", "grade", "model_strike_chance", "avg_hitter_swings", "z",
                        "runs_vs_avg"] if c in d]
    color = {"GOOD": "#1a7f37", "BAD": "#c8102e", "neutral": "#6b7280"}
    head = "".join(f"<th>{escape(c.replace('_', ' '))}</th>" for c in cols) + "<th>video</th>"
    rows = ""
    for _, r in d.iterrows():
        cells = ""
        for c in cols:
            v = r[c]
            txt = "" if pd.isna(v) else (f"{v:.0%}" if c in ("model_strike_chance", "avg_hitter_swings") else str(v))
            style = f' style="color:{color.get(str(v), "inherit")};font-weight:700"' if c == "grade" else ""
            cells += f"<td{style}>{escape(txt)}</td>"
        vid = r.get("video")
        has = isinstance(vid, str) and vid.startswith("http")
        link = vid if has else SAVANT_GAME.format(int(r["game_pk"]))
        label = "Watch" if has else "Game page"
        rows += f'<tr>{cells}<td><a href="{escape(link)}" target="_blank">{label}</a></td></tr>'
    HTML.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>body{{font-family:-apple-system,Segoe UI,sans-serif;background:#fff;color:#16202b;margin:24px}}
table{{border-collapse:collapse;font-size:13px}}th,td{{padding:6px 10px;border-bottom:1px solid #e6eaed;
text-align:left;white-space:nowrap}}th{{background:#f4f6f8;text-transform:uppercase;font-size:11px}}
a{{color:#0b5cad;font-weight:700}}p{{color:#555;max-width:900px}}</style></head><body>
<h2>{escape(title)}</h2>
<p>Grade = the model's call for a league-average hitter (count, then location, runners / outs only on close
calls). z = how clear the call was (|z| 0.25+ = clear; under 0.05 = coin flip). Strike chance = league-umpire
chance a take is called a strike. Avg hitter swings = how often a league-average hitter swings at this pitch.</p>
<table><tr>{head}</tr>{rows}</table></body></html>""")


def main(argv: list[str]) -> None:
    filtered = any(a in argv for a in ("--count", "--zone", "--decision", "--grade", "--n", "--seed"))
    if filtered:
        d = new_sample(argv)
        bits = [_arg(argv, k) for k in ("--count", "--zone", "--decision", "--grade") if _arg(argv, k)]
        title = "Swing decision check" + (": " + " / ".join(bits) if bits else "")
    else:
        if not CSV.exists():
            raise SystemExit("No video-check file yet -- run python -m pipeline.train_decisions first.")
        d = pd.read_csv(CSV)
        title = "Swing decision check: 20 random graded pitches"
    links, missing = [], 0
    cache: dict = {}
    for _, r in d.iterrows():
        gp = int(r["game_pk"])
        if gp not in cache:
            try:
                cache[gp] = play_ids(gp)
            except Exception as e:
                print(f"  game {gp}: couldn't reach MLB ({type(e).__name__})", flush=True)
                cache[gp] = {}
        pid = cache[gp].get((int(r["at_bat"]), int(r["pitch"])))
        missing += pid is None
        links.append(SAVANT_VIDEO.format(pid) if pid else None)
    d["video"] = links
    d.to_csv(CSV if not filtered else ART / "decisions_video_check_filtered.csv", index=False)
    write_page(d, title)
    print(f"{len(d) - missing}/{len(d)} pitches have a video link"
          + (f" ({missing} fall back to the game page)" if missing else "") + f"\nOpening {HTML}")
    try:
        webbrowser.open(HTML.as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
