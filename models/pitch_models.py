"""IQ pitch models: expected rates, grades and shape clusters. Everything comes from public Statcast pitch data.

Per pitch (all from what the ball did and where it went -- never the hitter or the outcome):
  xsw    P(swing)                                   -> xSw%   (averaged over every pitch)
  xbl    P(ball | taken)                            -> xBl%   (averaged over the pitches actually taken)
  xwf    P(miss | swung at)                         -> xWf%   (averaged over the actual swings)
  xiprv  expected batting-side run value if put in play -> xIPRV (averaged over the actual balls in play)
  xrv    expected pitcher-side run value (balls in play = expected value of the contact) -> xResults
  act    the same, from the ball's flight + release only (no location), by batter hand -> Action vL / vR
  shape  Action with the velo replaced by the velo expected for that shape -> Shape, Shape vL / vR
  loc    the same, from location + count only (no flight) -> Location
  ares   actual pitcher-side run value -> aResults
  cluster  league shape archetype (k-means on velo / movement / fastball differences, arm side mirrored)

GRADES (20-80): a group's mean, against every pitcher's season with that pitch type (200+ pitches):
  z = (mean - league mean) / league SD, shrunk toward average by n / (n + k). k is measured, not chosen: from how
  well a pitcher's odd pitches agree with his even ones (Spearman-Brown). The site turns z into 20-80 / percentile.
Command is NOT here -- it needs the intended location (catcher's target, from video).

Two eras (2015-2022, 2023+), like Stuff+. Trained by  python -m pipeline.train_pitch_models
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models import pitch_quality as pq

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "pitch_models.joblib"
ALL = pq.STUFF_FEATURES + [f for f in pq.LOC_FEATURES if f not in pq.STUFF_FEATURES]
ACTION = pq.STUFF_FEATURES + ["same_hand"]
SHAPE_VELO = ["ivb", "hb", "spin", "ext", "rel_z", "rel_x", "arm", "d_ivb", "d_hb", "ptype"]
CLUSTER = ["velo", "ivb", "hb", "d_velo", "d_ivb", "d_hb"]
GRADED = {"ARESULTS": "ares", "XRESULTS": "xrv", "ACTL": "act", "ACTR": "act", "SHAPE": "shape",
          "SHAPE_L": "shape", "SHAPE_R": "shape", "LOCATION": "loc"}
HAND = {"ACTL": "L", "ACTR": "R", "SHAPE_L": "L", "SHAPE_R": "R"}
COLS = ["xsw", "xbl", "xwf", "xiprv", "xrv", "act", "shape", "loc", "ares", "cluster"]

_cache: dict = {}


def load() -> dict | None:
    if "m" not in _cache:
        if not ARTIFACT.exists():
            _cache["m"] = None
        else:
            import joblib
            _cache["m"] = joblib.load(ARTIFACT)
    return _cache["m"]


def trained() -> bool:
    return ARTIFACT.exists()


def actual_rv(df: pd.DataFrame) -> pd.Series:
    """Pitcher-side actual run value per pitch (+ = good for the pitcher)."""
    rv = pd.to_numeric(df["delta_run_exp"], errors="coerce") if "delta_run_exp" in df else pd.Series(np.nan, index=df.index)
    balls = df["description"].isin({"ball", "blocked_ball"}) if "description" in df else pd.Series(False, index=df.index)
    if balls.any() and rv[balls].mean() > 0:                 # batting side -> flip
        rv = -rv
    return rv


def shape_features(f: pd.DataFrame, b: dict) -> pd.DataFrame:
    """Action inputs with the velo swapped for the velo expected from the pitch's shape."""
    g = f.copy()
    ok = g[SHAPE_VELO].notna().all(axis=1) & g["velo"].notna()
    if ok.any():
        ev = pd.Series(b["shape_velo"].predict(g.loc[ok, SHAPE_VELO]), index=g.index[ok])
        g.loc[ok, "d_velo"] = g.loc[ok, "d_velo"] + (ev - g.loc[ok, "velo"])
        g.loc[ok, "velo"] = ev
    return g


def cluster_of(f: pd.DataFrame, b: dict) -> pd.Series:
    c = b.get("cluster")
    out = pd.Series(None, index=f.index, dtype=object)
    if not c:
        return out
    x = f[CLUSTER].copy()
    x[["d_velo", "d_ivb", "d_hb"]] = x[["d_velo", "d_ivb", "d_hb"]].fillna(0)
    ok = x.notna().all(axis=1)
    if ok.any():
        z = (x[ok].to_numpy(float) - c["mu"]) / c["sd"]
        lab = c["km"].predict(z)
        out[ok] = [c["names"][int(i)] for i in lab]
    return out


def _score_era(df: pd.DataFrame, f: pd.DataFrame, b: dict) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=df.index, columns=COLS, dtype=object)
    ok = f["velo"].notna() & f[["x_in", "z_norm"]].notna().all(axis=1)
    if ok.any():
        X = f.loc[ok, ALL]
        out.loc[ok, "xsw"] = b["sw"].predict_proba(X)[:, 1]
        out.loc[ok, "xbl"] = b["bl"].predict_proba(X)[:, 1]
        out.loc[ok, "xwf"] = b["wf"].predict_proba(X)[:, 1]
        out.loc[ok, "xiprv"] = b["iprv"].predict(X)
        out.loc[ok, "xrv"] = b["xrv"].predict(X)
    oka = f["velo"].notna()
    if oka.any():
        out.loc[oka, "act"] = b["act"].predict(f.loc[oka, ACTION])
        out.loc[oka, "shape"] = b["act"].predict(shape_features(f[oka], b)[ACTION])
    okl = f[["x_in", "z_norm"]].notna().all(axis=1)
    if okl.any():
        out.loc[okl, "loc"] = b["loc"].predict(f.loc[okl, pq.LOC_FEATURES])
    out["ares"] = actual_rv(df)
    out["cluster"] = cluster_of(f, b)
    for c in COLS[:-1]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def score(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Per pitch: the COLS above (NaN / None until trained)."""
    b = bundle or load()
    out = pd.DataFrame({c: np.nan for c in COLS}, index=df.index)
    out["cluster"] = None
    if b is None or df.empty:
        return out
    f = pq.features(df)
    if "eras" not in b:
        return _score_era(df, f, b)
    yr = pd.to_datetime(df["game_date"]).dt.year if "game_date" in df else pd.Series(9999, index=df.index)
    for e, m in (("pre", yr < b["split"]), ("post", yr >= b["split"])):
        if m.any() and b["eras"].get(e):
            out.loc[m, COLS] = _score_era(df[m], f[m], b["eras"][e])[COLS].values
    for c in COLS[:-1]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _era_bundle(b: dict, years) -> dict | None:
    if b is None:
        return None
    if "eras" not in b:
        return b
    y = max(years) if len(years) else 9999
    return b["eras"].get("post" if y >= b["split"] else "pre") or next(iter(b["eras"].values()))


def grade_z(g: pd.DataFrame, key: str, bundle: dict | None = None) -> float | None:
    """z for one table row (g = prepped pitches of one pitch type) -- None if not trained / too few pitches."""
    b = _era_bundle(bundle or load(), pd.to_datetime(g["game_date"]).map(lambda d: d.year).unique()
                    if "game_date" in g and len(g) else [])
    if b is None or "grades" not in b:
        return None
    col = GRADED[key]
    if col not in g:
        return None
    rows = g if key not in HAND else g[g["stand"].astype(str).eq(HAND[key])]
    v = pd.to_numeric(rows[col], errors="coerce").dropna()
    if v.empty:
        return None
    pt = pq.PTYPE.get(str(g["pitch"].iloc[0])) if "pitch" in g else None
    ref = b["grades"].get(key, {})
    r = ref.get("by_type", {}).get(pt) or ref.get("all")
    if not r or not r.get("sd"):
        return None
    n = len(v)
    z = (v.mean() - r["mean"]) / r["sd"] * n / (n + ref.get("k", 100))
    return float(z)


def cluster_names(bundle: dict | None = None) -> list[str]:
    b = _era_bundle(bundle or load(), [9999])
    return list((b or {}).get("cluster", {}).get("names", []))
