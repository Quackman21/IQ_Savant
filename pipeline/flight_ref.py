"""League references for the Ball Flight Grade: average + spread of every ingredient among MLB-qualified hitters.

    python -m pipeline.flight_ref                    # every season 2015 -> now that's missing
    python -m pipeline.flight_ref --all              # rebuild all
    python -m pipeline.flight_ref --season 2026
    python -m pipeline.flight_ref --top 2025 2026    # eye test: grade every qualified hitter, top / bottom 10

Uses the saved pitches (data/raw), well under a minute a season. pipeline.update keeps the current season fresh.
"""
from __future__ import annotations

import sys
import time
from datetime import date

FIRST = 2015
WATCH = {663728: "Cal Raleigh", 691718: "Pete Crow-Armstrong"}


def _raw(season: int):
    from pipeline import build_league_reference as bl
    raw = bl.pull_season(season)
    if "game_type" in raw:
        raw = raw[raw["game_type"].astype(str).eq("R")]
    return raw.reset_index(drop=True)


def build(season: int) -> None:
    from models import flight_grade as fg
    t0 = time.time()
    ref = fg.build_ref(_raw(season), season)
    p = fg.save_ref(ref)
    print(f"[{season}] Ball Flight Grade reference saved ({ref['n_qualified']} qualified hitters, "
          f"{ref['qualified_pa']:.0f}+ PA) -> {p.name} in {time.time() - t0:.0f}s", flush=True)


def top(season: int) -> None:
    import pandas as pd

    from models import flight_grade as fg
    from models import power
    raw = _raw(season)
    ref = fg.load_ref(season) or fg.build_ref(raw, season)
    q, _ = power.qualified(raw)
    rows = []
    for b in q:
        g = fg.score(raw[raw["batter"] == b], ref)
        rows.append({"batter": int(b), "grade": 50 + 10 * g["_BF_Z"], "parts": g["_BF_PARTS"]})
    t = pd.DataFrame(rows).dropna(subset=["grade"]).sort_values("grade", ascending=False).reset_index(drop=True)
    try:
        from core import mlb_api
        names = mlb_api.get_people(t["batter"].tolist())
    except Exception:
        names = {}
    f = lambda r: f"{names.get(r.batter, r.batter)} {r.grade:.0f}"
    print(f"\n{season} Ball Flight Grade, {len(t)} MLB-qualified hitters:")
    print("  top:    " + " | ".join(f(r) for r in t.head(10).itertuples()))
    print("  bottom: " + " | ".join(f(r) for r in t.tail(10).itertuples()))
    print(f"  spread check: middle 50% from {t.grade.quantile(.25):.0f} to {t.grade.quantile(.75):.0f} "
          "(should be about 43 to 57)")
    for b, name in WATCH.items():
        hit = t.index[t["batter"] == b]
        if len(hit):
            r = t.loc[hit[0]]
            print(f"  {name}: {r.grade:.0f}, #{hit[0] + 1} of {len(t)}  [{r.parts}]")


def main(argv: list[str]) -> None:
    from models import flight_grade as fg
    if "--top" in argv:
        for y in [a for a in argv[argv.index("--top") + 1:] if a.isdigit()]:
            top(int(y))
        return
    last = date.today().year
    if "--season" in argv:
        todo = [int(argv[argv.index("--season") + 1])]
    elif "--all" in argv:
        todo = list(range(FIRST, last + 1))
    else:
        todo = [y for y in range(FIRST, last + 1) if fg.load_ref(y) is None]
    if not todo:
        print("Ball Flight Grade references are all built.")
    for y in todo:
        try:
            build(y)
        except Exception as e:
            print(f"[{y}] Ball Flight Grade reference FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
