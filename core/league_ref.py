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


def _artifact_hash(active: str, kind: str = "xwoba") -> str:
    """Retraining a model changes its file -> league numbers become out of date."""
    if active != "iq_v1":
        return ""
    import hashlib
    if kind == "xwoba":
        from models import xwoba_iq as m
    else:
        from models import decisions_iq as m
    try:
        return hashlib.sha1(m.ARTIFACT.read_bytes()).hexdigest()[:12]
    except OSError:
        return ""


def _sa_hash() -> str:
    """Retraining SAxwOBA changes its MLB averages."""
    import hashlib
    from models import saxwoba
    try:
        return hashlib.sha1(saxwoba.ARTIFACT.read_bytes()).hexdigest()[:12]
    except OSError:
        return ""


def signature() -> str:
    """Fingerprint of everything that changes the numbers: stat definitions, the pitch
    flags in prep(), the reference builder, and which xwOBA / decision models are active.
    If it changes, saved reference files are out of date. (Comments and page layout
    changes don't affect it.)"""
    import hashlib
    import inspect

    from models import decisions, framing, xwoba
    parts = [inspect.getsource(framing.MODELS[framing.ACTIVE]), str(framing.RUNS_PER_STRIKE),repr(sorted((k, s.kind, s.a, s.b) for k, s in hm.CATALOG.items())),
             inspect.getsource(hm.prep), inspect.getsource(build), str(hm.SD_MIN_PA), str(hm.SPRAY_SCALE),
             xwoba.ACTIVE, inspect.getsource(xwoba.MODELS[xwoba.ACTIVE]), _artifact_hash(xwoba.ACTIVE),
             decisions.ACTIVE, inspect.getsource(decisions.MODELS[decisions.ACTIVE]),
             _artifact_hash(decisions.ACTIVE, "decisions"), _sa_hash(),
             _artifact_hash("iq_v1", "decisions"), _fielding_hash()]                 # framing uses the called-strike model too
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]


def _meta_path(season: int) -> Path:
    return DATA / f"hitting_{season}.json"


def save(ref: pd.DataFrame, season: int, last_game: str | None = None) -> Path:
    import json
    from datetime import datetime
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f"hitting_{season}.csv.gz"
    ref.to_csv(p, index=False)
    _meta_path(season).write_text(json.dumps({
        "signature": signature(), "built": datetime.now().isoformat(timespec="seconds"),
        "last_game": last_game}, indent=1))
    return p


def meta(season: int) -> dict:
    import json
    try:
        return json.loads(_meta_path(season).read_text())
    except (OSError, ValueError):
        return {}


def status(season: int) -> str:
    """'missing' | 'stale' (built with older definitions) | 'current'."""
    if not all((DATA / f"{k}_{season}.csv.gz").exists() for k in ("hitting", "catching", "fielding")):
        return "missing"
    return "current" if meta(season).get("signature") == signature() else "stale"


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


def save_catching(daily: pd.DataFrame, season: int) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f"catching_{season}.csv.gz"
    daily.to_csv(p, index=False)
    return p


def save_fielding(daily: pd.DataFrame, season: int) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f"fielding_{season}.csv.gz"
    daily.to_csv(p, index=False)
    return p


def load_fielding(season: int) -> pd.DataFrame:
    p = DATA / f"fielding_{season}.csv.gz"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype={"game_date": str})


def _fielding_hash() -> str:
    """Retraining fielding / catching -- or new play-by-play events -- changes the fielding files."""
    import hashlib
    from models import catching, fielding
    h = hashlib.sha1()
    for f in (fielding.ARTIFACT, catching.ARTIFACT):
        try:
            h.update(f.read_bytes())
        except OSError:
            pass
    raw = DATA.parent / "raw"
    h.update(",".join(sorted(p.name for p in raw.glob("events_*.csv.gz"))).encode() if raw.exists() else b"")
    return h.hexdigest()[:12]


def load_catching(season: int) -> pd.DataFrame:
    p = DATA / f"catching_{season}.csv.gz"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype={"game_date": str})
