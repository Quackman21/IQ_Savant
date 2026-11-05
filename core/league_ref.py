"""MLB reference numbers for every hitting stat: league average and the spread
(standard deviation) across MLB hitters, for every row type the tables show
(All, In Zone, vs LHP, In Zone + vs LHP, ...).

Built offline by pipeline/build_league_reference.py and saved to
data/league/hitting_<season>.csv.gz. Pages only read the small file.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from core import hit_metrics as hm

PER_HITTER = {"max", "q90", "top8"}
DATA = Path(__file__).resolve().parent.parent / "data" / "league"


def filter_sets() -> list[list[str]]:
    zones = list(hm.ZONE_GROUPINGS.values())
    splits = list(hm.SPLITS.values())
    sets = [[]] + [[z] for z in zones] + [[s] for s in splits] + [[z, s] for z, s in product(zones, splits)]
    return sets


def build(d: pd.DataFrame, min_pa: int = hm.SD_MIN_PA, model: str | None = None) -> pd.DataFrame:
    """d = hm.prep() output for a whole season (all batters).
    Returns long table: dims, vals, stat, avg, sd, n (qualified hitters)."""
    qualified = d.groupby("batter")["pa"].sum()
    qualified = set(qualified[qualified >= min_pa].index)
    rows = []
    for dims in filter_sets():
        key = "+".join(dims) or "all"
        league = hm.compute(d, by=dims or None)
        per = hm.compute(d[d["batter"].isin(qualified)], by=["batter"] + dims)
        if dims:
            spread = per.groupby(level=dims if len(dims) > 1 else dims[0]).std()
            counts = per.groupby(level=dims if len(dims) > 1 else dims[0]).size()
        else:
            spread = per.std().to_frame("All").T
            counts = pd.Series({"All": len(per)})
        # rates use the pooled league number; per-hitter shapes (max / 90th / top 1/8 EV)
        # use the average hitter's value instead of the single hardest ball in MLB
        if dims:
            hitter_avg = per.groupby(level=dims if len(dims) > 1 else dims[0]).mean()
        else:
            hitter_avg = per.mean().to_frame("All").T
        for idx in league.index:
            vals = "|".join(idx) if isinstance(idx, tuple) else str(idx)
            for stat in hm.ALL_KEYS:
                per_hitter = hm.CATALOG[stat].kind in PER_HITTER
                avg = (hitter_avg.at[idx, stat] if idx in hitter_avg.index else np.nan) if per_hitter \
                    else league.at[idx, stat]
                rows.append({"dims": key, "vals": vals, "stat": stat,
                             "avg": avg,
                             "sd": spread.at[idx, stat] if idx in spread.index else np.nan,
                             "n": int(counts.get(idx, 0))})
    out = pd.DataFrame(rows)
    out["model"] = model or "default"
    return out


def save(ref: pd.DataFrame, season: int) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f"hitting_{season}.csv.gz"
    ref.to_csv(p, index=False)
    return p


def load(season: int) -> dict:
    """{(dims, vals): {stat: (avg, sd)}} or {} if not built yet."""
    p = DATA / f"hitting_{season}.csv.gz"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out: dict = {}
    for r in df.itertuples():
        out.setdefault((r.dims, str(r.vals)), {})[r.stat] = (r.avg, r.sd)
    return out
