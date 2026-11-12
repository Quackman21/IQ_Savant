"""League references for the Bat Speed (power) Grade: average + spread of every ingredient among regulars.

    python -m pipeline.power_ref                  # every season 2015 -> latest that's missing
    python -m pipeline.power_ref --all            # rebuild all
    python -m pipeline.power_ref --season 2026
    python -m pipeline.power_ref --top 2026       # eye test: grade every regular, print top / bottom 10

Uses the saved pitches (data/raw), ~1 minute a season. pipeline.update keeps the current season fresh.
"""
from __future__ import annotations

import sys
import time
from datetime import date

FIRST = 2015


def build(season: int) -> None:
    from models import power
    from pipeline import build_league_reference as bl
    t0 = time.time()
    ref = power.build_ref(bl.pull_season(season), season)
    p = power.save_ref(ref)
    kind = "bat speed + exit velo" if "sd_bat" in ref else "exit velo only"
    print(f"[{season}] power reference saved ({kind}, {ref['n_regulars']} regulars) -> {p.name} "
          f"in {time.time() - t0:.0f}s", flush=True)


def missing(last: int) -> list[int]:
    from models import power
    return [y for y in range(FIRST, last + 1) if power.load_ref(y) is None]


def top(season: int) -> None:
    import pandas as pd

    from models import power
    from pipeline import build_league_reference as bl
    ref = power.load_ref(season)
    if not ref:
        build(season)
        ref = power.load_ref(season)
    raw = bl.pull_season(season)
    if "game_type" in raw:
        raw = raw[raw["game_type"].astype(str).eq("R")]
    bip = raw[raw["type"].eq("X")].groupby("batter").size()
    rows = []
    for b in bip[bip >= power.MIN_REF_BIP].index:
        g = power.score(raw[raw["batter"] == b], ref)
        rows.append({"batter": int(b), "power": 50 + 10 * g["_POWER_Z"], "sc": 50 + 10 * g["_SCBS_Z"],
                     "bip": int(bip[b])})
    t = pd.DataFrame(rows).sort_values("power", ascending=False)
    try:
        from core import mlb_api
        names = mlb_api.get_people(t["batter"].tolist())
    except Exception:
        names = {}
    f = lambda r: f"{names.get(r.batter, r.batter)} {r.power:.0f}" + (f" (SC {r.sc:.0f})" if r.sc == r.sc else "")
    print(f"\n{season} Bat Speed (power) Grade, {len(t)} regulars ({power.MIN_REF_BIP}+ BIP):")
    print("  top:    " + " | ".join(f(r) for r in t.head(10).itertuples()))
    print("  bottom: " + " | ".join(f(r) for r in t.tail(10).itertuples()))
    print(f"  spread check: middle 50% from {t.power.quantile(.25):.0f} to {t.power.quantile(.75):.0f} "
          "(should be about 43 to 57)")


def main(argv: list[str]) -> None:
    if "--top" in argv:
        top(int(argv[argv.index("--top") + 1]))
        return
    last = date.today().year
    if "--season" in argv:
        todo = [int(argv[argv.index("--season") + 1])]
    elif "--all" in argv:
        todo = list(range(FIRST, last + 1))
    else:
        todo = missing(last)
    if not todo:
        print("Power references are all built.")
    for y in todo:
        try:
            build(y)
        except Exception as e:
            print(f"[{y}] power reference FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
