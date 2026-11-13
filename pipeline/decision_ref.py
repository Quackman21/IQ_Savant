"""League references for the Decision Grades (overall + 0-0, <2k, 2k, FB, BB, OS).

    python -m pipeline.decision_ref                  # every season 2015 -> now that's missing
    python -m pipeline.decision_ref --all            # rebuild all
    python -m pipeline.decision_ref --season 2026
    python -m pipeline.decision_ref --top 2026       # eye test: grade every qualified hitter, top / bottom 10

Scores every decision of the season with IQ Swing Decisions (~1-2 min a season), from the saved pitches.
pipeline.update keeps the current season fresh.
"""
from __future__ import annotations

import sys
import time
from datetime import date

import pandas as pd

FIRST = 2015


def _season(season: int):
    from models import decisions_iq as dq
    from pipeline import build_league_reference as bl
    raw = bl.pull_season(season)
    if "game_type" in raw:
        raw = raw[raw["game_type"].astype(str).eq("R")]
    raw = raw.reset_index(drop=True)
    sc = dq.score(raw)
    return raw, sc["runs_vs_avg"], sc["quality"].notna()


def build(season: int) -> None:
    from models import decision_grade as dg
    t0 = time.time()
    raw, runs, graded = _season(season)
    ref = dg.build_ref(raw, runs, graded, season)
    p = dg.save_ref(ref)
    a = ref["splits"]["all"]
    print(f"[{season}] Decision Grade reference saved ({a['n_qualified']} qualified hitters, "
          f"{ref['qualified_pa']:.0f}+ PA) -> {p.name} in {time.time() - t0:.0f}s", flush=True)


def top(season: int) -> None:
    from models import decision_grade as dg
    from models import power
    raw, runs, graded = _season(season)
    ref = dg.load_ref(season) or dg.build_ref(raw, runs, graded, season)
    q, _ = power.qualified(raw)
    rows = []
    for b in q:
        m = raw["batter"] == b
        g = dg.score(raw[m], runs[m], graded[m], ref)
        rows.append({"batter": int(b), **{k.strip("_"): 50 + 10 * v for k, v in g.items() if k.endswith("_Z")},
                     "runs": runs[m].sum()})
    t = pd.DataFrame(rows).sort_values("DEC_Z", ascending=False)
    try:
        from core import mlb_api
        names = mlb_api.get_people(t["batter"].tolist())
    except Exception:
        names = {}
    f = lambda r: f"{names.get(r.batter, r.batter)} {r.DEC_Z:.0f} ({r.runs:+.0f} runs)"
    print(f"\n{season} Decision Grade, {len(t)} MLB-qualified hitters:")
    print("  top:    " + " | ".join(f(r) for r in t.head(10).itertuples()))
    print("  bottom: " + " | ".join(f(r) for r in t.tail(10).itertuples()))
    print(f"  spread check: middle 50% from {t.DEC_Z.quantile(.25):.0f} to {t.DEC_Z.quantile(.75):.0f} "
          "(should be about 43 to 57)")
    for k, lab in (("DEC00_Z", "0-0"), ("DECLT2_Z", "<2k"), ("DEC2K_Z", "2k"), ("DECFB_Z", "FB"),
                   ("DECBB_Z", "BB"), ("DECOS_Z", "OS")):
        print(f"  {lab:<4} grade: middle 50% {t[k].quantile(.25):.0f}-{t[k].quantile(.75):.0f}, "
              f"r with overall = {t[k].corr(t.DEC_Z):.2f}")


def main(argv: list[str]) -> None:
    from models import decision_grade as dg
    if "--top" in argv:
        top(int(argv[argv.index("--top") + 1]))
        return
    last = date.today().year
    if "--season" in argv:
        todo = [int(argv[argv.index("--season") + 1])]
    elif "--all" in argv:
        todo = list(range(FIRST, last + 1))
    else:
        todo = [y for y in range(FIRST, last + 1) if dg.load_ref(y) is None]
    if not todo:
        print("Decision Grade references are all built.")
    for y in todo:
        try:
            build(y)
        except Exception as e:
            print(f"[{y}] Decision Grade reference FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
