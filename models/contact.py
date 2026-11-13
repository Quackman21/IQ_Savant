"""IQ Bat-To-Ball Grade -- how good a hitter is at making contact WHEN HE SWINGS (takes never count).

Weights from the Bat-to-Ball study (python -m pipeline.contact_study, 2015-2026), chosen so a swing and miss on a
hittable pitch costs the most:

    Cont% over expected     45%   every swing graded by how hard that pitch is to hit (league contact model:
    Z-Cont% over expected   20%   location, pitch type, velo, movement, count, hands). A whiff over the heart
                                  (~89% league contact) costs ~4x a whiff on a waste pitch (~23%).
    Cont%                   15%   contact per swing
    H-Cont%                 10%   contact on swings at heart pitches -- direct penalty for missing cookies
    C-Cont%                 10%   contact on chase swings

Each ingredient is regressed toward the league average by its own sample (swings before it is 70% skill, from the
study), then graded vs MLB-qualified hitters (3.1 PA per team game): 50 = average qualified hitter, 10 = one spread.
League contact model: models/artifacts/contact.joblib (python -m pipeline.train_contact), one model per era
(2015-2022 / 2023+). League references: data/league/contact_<year>.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "models" / "artifacts" / "contact.joblib"
REF_DIR = ROOT / "data" / "league"
SPLIT = 2023
FEATURES = ["nx", "nz", "velo", "hb", "ivb", "balls", "strikes", "same", "ptype"]
PTYPES = ["CH", "CS", "CU", "EP", "FA", "FC", "FF", "FO", "FS", "KC", "KN", "SC", "SI", "SL", "ST", "SV"]

W = {"Cont% over exp": .45, "Z-Cont% over exp": .20, "Cont%": .15, "H-Cont%": .10, "C-Cont%": .10}
INGREDIENTS = list(W)
SWINGS_FOR_70 = {"Cont% over exp": 140, "Z-Cont% over exp": 245, "Cont%": 165, "H-Cont%": 580, "C-Cont%": 475}
K = {m: n * 0.3 / 0.7 for m, n in SWINGS_FOR_70.items()}     # reliability = n / (n + k), n = his swings
REGRESS = True

_cache: dict = {}


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Pitch features from the batter's view (inside / away, not left / right)."""
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
    top, bot = num("sz_top"), num("sz_bot")
    nx = num("plate_x") / (17 / 24)
    lefty_bat = df["stand"].astype(str).eq("L") if "stand" in df else pd.Series(False, index=df.index)
    lefty_p = df["p_throws"].astype(str).eq("L") if "p_throws" in df else pd.Series(False, index=df.index)
    pt = df["pitch_type"].astype(str) if "pitch_type" in df else pd.Series("", index=df.index)
    return pd.DataFrame({
        "nx": np.where(lefty_bat, -nx, nx),
        "nz": (num("plate_z") - (top + bot) / 2) / ((top - bot) / 2),
        "velo": num("release_speed"),
        "hb": np.where(lefty_p, -1, 1) * num("pfx_x"),
        "ivb": num("pfx_z"),
        "balls": num("balls"), "strikes": num("strikes"),
        "same": (df.get("stand", pd.Series("", index=df.index)).astype(str)
                 == df.get("p_throws", pd.Series("?", index=df.index)).astype(str)).astype(int),
        "ptype": pt.map({p: i for i, p in enumerate(PTYPES)}).fillna(-1),
    }, index=df.index)


def tag_swings(df: pd.DataFrame) -> pd.DataFrame:
    """swing / contact / in-zone / attack-zone flags, the app's exact definitions."""
    from core import hit_metrics as hm
    desc = df["description"].astype(str)
    zone = pd.to_numeric(df["zone"], errors="coerce") if "zone" in df else pd.Series(np.nan, index=df.index)
    return pd.DataFrame({"batter": df["batter"] if "batter" in df else 0,
                         "swing": desc.isin(hm.SWINGS), "contact": desc.isin(hm.SWINGS) & ~desc.isin(hm.WHIFFS),
                         "in_z": zone.between(1, 9), "att": hm.attack_zone(df)}, index=df.index)


def load():
    if "m" not in _cache:
        if not ARTIFACT.exists():
            _cache["m"] = None
        else:
            import joblib
            _cache["m"] = joblib.load(ARTIFACT)
    return _cache["m"]


def expected(df: pd.DataFrame, bundle=None) -> pd.Series:
    """League contact chance for every swing in df (NaN for takes / no model)."""
    b = bundle or load()
    out = pd.Series(np.nan, index=df.index)
    if b is None or df.empty:
        return out
    t = tag_swings(df)
    sw = df[t["swing"]]
    if sw.empty:
        return out
    X = features(sw)[FEATURES].fillna(-1)
    yr = pd.to_datetime(sw["game_date"], errors="coerce").dt.year if "game_date" in sw else pd.Series(2026, index=sw.index)
    for era, m in (("pre", yr < b.get("split", SPLIT)), ("post", yr >= b.get("split", SPLIT))):
        model = b["eras"].get(era) or b["eras"].get("post" if era == "pre" else "pre")
        if m.any() and model is not None:
            out[sw.index[m.values]] = model.predict_proba(X[m.values])[:, 1]
    return out


def hitter_table(df: pd.DataFrame, xcont: pd.Series | None = None) -> pd.DataFrame:
    """The five ingredients for every batter in df (swings only)."""
    t = tag_swings(df)
    t["x"] = expected(df) if xcont is None else xcont
    s = t[t["swing"]]
    g = s.groupby("batter")
    out = pd.DataFrame({"swings": g.size()})
    out["Cont%"] = g["contact"].mean()
    over = s.assign(o=s["contact"].astype(float) - s["x"]).dropna(subset=["o"])
    out["Cont% over exp"] = over.groupby("batter")["o"].mean()
    out["Z-Cont% over exp"] = over[over["in_z"]].groupby("batter")["o"].mean()
    out["H-Cont%"] = s[s["att"].eq("Heart")].groupby("batter")["contact"].mean()
    out["C-Cont%"] = s[s["att"].eq("Chase")].groupby("batter")["contact"].mean()
    return out


def build_ref(season_df: pd.DataFrame, year: int) -> dict:
    """League average + spread of each ingredient among MLB-qualified hitters, one season."""
    from models import power
    d = season_df
    if "game_type" in d:
        d = d[d["game_type"].astype(str).eq("R")]
    t = hitter_table(d)
    q, need = power.qualified(d)
    reg = t[t.index.isin(q)]
    stats = {m: (float(reg[m].mean()), float(reg[m].std())) for m in INGREDIENTS if reg[m].notna().sum() >= 30}
    sw = tag_swings(d)
    sw = sw[sw["swing"]]
    ref = {"year": year, "stats": stats, "n_qualified": int(len(reg)), "qualified_pa": float(need),
           "league_contact_by_zone": {z: float(sw.loc[sw["att"].eq(z), "contact"].mean())
                                      for z in ("Heart", "Shadow", "Chase", "Waste")}}
    if all(m in stats for m in W) and len(reg) >= 30:
        comp = sum(v * (reg[m].fillna(stats[m][0]) - stats[m][0]) / stats[m][1] for m, v in W.items())
        ref["sd_blend"] = float(comp.std())
    return ref


def save_ref(ref: dict) -> Path:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    p = REF_DIR / f"contact_{ref['year']}.json"
    p.write_text(json.dumps(ref, indent=1))
    return p


def load_ref(year: int) -> dict | None:
    p = REF_DIR / f"contact_{year}.json"
    return json.loads(p.read_text()) if p.exists() else None


def score(df: pd.DataFrame, ref: dict | None) -> dict:
    """Bat-To-Ball Grade (as z) for one hitter over any period. Keys: _B2B_Z, _B2B_PARTS, _B2B_KIND."""
    out = {"_B2B_Z": math.nan, "_B2B_PARTS": "", "_B2B_KIND": "Bat-to-ball (swings only)"}
    if df is None or df.empty or not ref or "sd_blend" not in ref or load() is None:
        return out
    d = df.reset_index(drop=True)                            # combined seasons can repeat row numbers
    d["batter"] = 0
    t = hitter_table(d)
    if t.empty:
        return out
    row = t.iloc[0]
    n = float(row.get("swings", 0) or 0)
    z = {}
    for m in INGREDIENTS:
        mean, sd = ref["stats"][m]
        x = row.get(m, np.nan)
        if x != x or n <= 0 or not sd:
            z[m] = 0.0
            continue
        shrink = n / (n + K[m]) if REGRESS else 1.0
        z[m] = (x - mean) * shrink / sd
    out["_B2B_Z"] = sum(v * z[m] for m, v in W.items()) / ref["sd_blend"]
    out["_B2B_PARTS"] = " | ".join(f"{m} {50 + 10 * z[m]:.0f}" for m in W) + f" ({int(n)} swings)"
    return out
