"""IQ xwOBA: expected wOBA for every batted ball.

Swing only -- no sprint speed. Inputs per batted ball (which ones depends on the VARIANT):
  is_left     batter side
  ev, la      exit velocity, launch angle
  spray_pull  exact spray angle, pull-adjusted (+ = pulled for both sides)          [v1]
  spray_air   exact spray angle, ONLY for balls in the air (LA >= 10 deg)
              -- on ground balls Statcast's hit coordinate is where the ball was fielded,
                 so the exact angle mostly records whether it found a hole (the result)
  spray_dir   general direction for every ball: -1 oppo / 0 middle / +1 pull (+-15 deg,
              same lines as the site's PULL% / MIDDLE% / OPPO%)
  park        ballpark -- used ONLY in training so park quirks are blamed on the park.
              When scoring, every ball is valued in each MLB park and the results are
              averaged equally (a true neutral park, not "where games happened to be").

The model predicts P(out), P(1B), P(2B), P(3B), P(HR). The expected wOBA of a ball is those
probabilities times that season's standard wOBA weights, then scaled so each season's
league IQ xwOBACON equals the league's actual wOBACON (calibration). Walks / HBP / strikeouts
keep their actual (season) wOBA values -- that happens in models/xwoba.py.

Trained by pipeline/train_xwoba.py -> models/artifacts/xwoba_iq_v1.joblib
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "xwoba_iq_v1.joblib"
CLASSES = ["out", "single", "double", "triple", "home_run"]
FEATURES = ["is_left", "ev", "la", "spray_pull"]          # v1 (bundles saved before variants)
AIR_LA = 10                                                # launch angle where "in the air" starts
DIR_LINE = 15                                              # pull / middle / oppo line (degrees)
SPRAY_SCALE = 0.75

# name -> (inputs, park-neutral?, description)
VARIANTS = {
    "v1":        (["is_left", "ev", "la", "spray_pull"], False,
                  "v1 as trained: exact spray on every ball, parks mixed together"),
    "park":      (["is_left", "ev", "la", "spray_pull"], True,
                  "v1 + park-neutral"),
    "no_spray":  (["is_left", "ev", "la"], True,
                  "park-neutral, no direction at all"),
    "dir":       (["is_left", "ev", "la", "spray_dir"], True,
                  "park-neutral, pull/middle/oppo only (every ball)"),
    "air":       (["is_left", "ev", "la", "spray_air"], True,
                  "park-neutral, exact spray on air balls only"),
    "air_dir":   (["is_left", "ev", "la", "spray_air", "spray_dir"], True,
                  "park-neutral, exact spray on air balls + pull/middle/oppo on every ball"),
    "fly":       (["is_left", "ev", "la", "spray_fly"], True,
                  "park-neutral, exact spray only on true fly balls (LA 25-45)"),
    "fly_hard":  (["is_left", "ev", "la", "spray_hr"], True,
                  "park-neutral, exact spray only on 95+ mph balls at LA 20-45 (homer-or-out balls)"),
    "blur8":     (["is_left", "ev", "la", "spray_pull"], True,
                  "park-neutral, spray on every ball, blurred +-8 deg (washes out the holes)"),
    "blur12":    (["is_left", "ev", "la", "spray_pull"], True,
                  "park-neutral, spray on every ball, blurred +-12 deg"),
    "blur18":    (["is_left", "ev", "la", "spray_pull"], True,
                  "park-neutral, spray on every ball, blurred +-18 deg"),
    "hr_dir":    (["is_left", "ev", "la"], True,
                  "park-neutral; direction ONLY decides home run vs not (spray on air balls); rest no direction"),
    "by_type":   (["is_left", "ev", "la", "spray_pull"], True,
                  "park-neutral, blur by ball type: exact on fly balls, +-12 liners, +-18 grounders"),
}
DEFAULT_VARIANT = "no_spray"
# hr_dir: a second model gives P(home run) WITH direction (a pulled 105 / 25 clears the fence, to center it's
# caught -- fence distance, not fielders). Outs / 1B / 2B / 3B come from the no-direction model, rescaled to
# make room. Direction never touches singles / doubles, which is where gaps and holes leak the result.
HR_FEATURES = ["is_left", "ev", "la", "spray_air"]
# Blur: when SCORING, a ball is valued at its spray angle and at angles either side of it
# (evenly spaced across +-width) and the values averaged. Holes between fielders are only a few
# degrees wide, so they wash out; broad direction (pull in the air, oppo on the ground, down the
# line) survives. Training is unchanged.
BLUR = {"blur8": 8, "blur12": 12, "blur18": 18, "by_type": "by_type"}
# blur by ball type: (launch angle below, blur width). Fly balls (LA 25+) keep the exact angle --
# pulled vs center in the air is real carry / distance. Liners: outfield gaps washed out. Grounders:
# holes between infielders washed out, broad pull / oppo kept.
BY_TYPE = [(10, 18), (25, 12)]
BLUR_STEPS = 5
SPRAY_LIMIT = 45                                           # foul lines (pull-adjusted degrees)

HITS = {"single", "double", "triple", "home_run"}
OUTS = {"field_out", "force_out", "grounded_into_double_play", "double_play", "fielders_choice",
        "fielders_choice_out", "field_error", "sac_fly", "sac_fly_double_play", "triple_play",
        "other_out"}                                   # reached on error counts as an out (standard)
EXCLUDE = {"sac_bunt", "sac_bunt_double_play", "catcher_interf"}


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Every candidate input. Missing EV / LA / spray stay NaN (the model handles them).
    'park' is filled from a 'venue' column when present (training only)."""
    c = lambda n: pd.to_numeric(df[n], errors="coerce") if n in df else pd.Series(np.nan, index=df.index)
    stand = df["stand"].astype(str) if "stand" in df else pd.Series("R", index=df.index)
    spray = np.degrees(np.arctan((c("hc_x") - 125.42) / (198.27 - c("hc_y")))) * SPRAY_SCALE
    left = stand.eq("L")
    pull = pd.Series(np.where(left, spray, -spray), index=df.index)
    la = c("launch_angle")
    air = la >= AIR_LA                                   # NaN LA -> not air
    direction = pd.Series(np.select([pull >= DIR_LINE, pull <= -DIR_LINE], [1.0, -1.0], 0.0), index=df.index)
    direction[pull.isna()] = np.nan
    return pd.DataFrame({"is_left": left.astype(float), "ev": c("launch_speed"), "la": la,
                         "spray_pull": pull, "spray_air": pull.where(air), "spray_dir": direction,
                         "spray_fly": pull.where(la.between(25, 45)),
                         "spray_hr": pull.where(la.between(20, 45) & (c("launch_speed") >= 95)),
                         "park": c("venue")}, index=df.index)


def label(events: pd.Series) -> pd.Series:
    """Outcome class per batted ball (NaN = leave out of training)."""
    ev = events.astype(str)
    out = pd.Series(np.nan, index=events.index, dtype=object)
    out[ev.isin(OUTS)] = "out"
    for h in HITS:
        out[ev.eq(h)] = h
    return out


def season_weights(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Standard wOBA weights per season, read from Savant's own woba_value on each hit type."""
    if "woba_value" not in df or "events" not in df:
        return {}
    year = pd.to_datetime(df["game_date"]).dt.year if "game_date" in df else pd.Series(0, index=df.index)
    w = {}
    for y, g in df.assign(_y=year).groupby("_y"):
        vals = pd.to_numeric(g["woba_value"], errors="coerce")
        row = {h: float(vals[g["events"].eq(h)].median()) for h in HITS if g["events"].eq(h).any()}
        if len(row) == 4:
            w[int(y)] = row
    return w


# ----------------------------------------------------------------------------
# Model inputs for a trained bundle
# ----------------------------------------------------------------------------
def model_inputs(feats: pd.DataFrame, bundle: dict, cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Columns the model was trained on. Park becomes a category code (unknown -> missing)."""
    cols = list(cols or bundle.get("features", FEATURES))
    X = feats[cols].copy()
    if bundle.get("parks"):
        X["park"] = feats["park"].map(bundle["parks"]).astype(float)
        cols = cols + ["park"]
    return X[cols], cols


def park_codes(venues: pd.Series) -> tuple[dict, list[int]]:
    """venue id -> category code, and the codes of real MLB home parks that make up the neutral
    average (at least a quarter of a typical park's batted balls). One-off sites (London, Tokyo,
    Rickwood, Little League Classic...) help training but aren't part of the average."""
    v = venues.dropna().astype(int)
    counts = v.value_counts()
    codes = {int(vid): i for i, vid in enumerate(sorted(counts.index))}
    floor = 0.25 * counts.median() if len(counts) else 0
    neutral = [codes[int(vid)] for vid, n in counts.items() if n >= floor]
    return codes, sorted(neutral)


def _park_avg(m, X: pd.DataFrame, bundle: dict) -> np.ndarray:
    if bundle.get("parks") and bundle.get("neutral_parks"):
        acc = np.zeros((len(X), len(m.classes_)))
        Xp = X.copy()
        for code in bundle["neutral_parks"]:
            Xp["park"] = float(code)
            acc += m.predict_proba(Xp)
        return acc / len(bundle["neutral_parks"])
    return m.predict_proba(X)


def blur_widths(X: pd.DataFrame, blur) -> pd.Series | None:
    """Blur width (degrees) for every ball, or None for no blur."""
    if not blur:
        return None
    if blur == "by_type":
        la = X["la"]
        w = np.select([la < BY_TYPE[0][0], la < BY_TYPE[1][0]], [BY_TYPE[0][1], BY_TYPE[1][1]], 0.0)
        w = np.where(la.isna(), BY_TYPE[1][1], w)                     # unknown LA: middle blur
        return pd.Series(w, index=X.index, dtype=float)
    return pd.Series(float(blur), index=X.index)


def raw_probabilities(X: pd.DataFrame, bundle: dict, feats: pd.DataFrame | None = None) -> pd.DataFrame:
    """P(class) per ball. Park-neutral bundles: averaged over every MLB park equally.
    Blurred bundles: also averaged over spray angles either side of the real one."""
    m = bundle["model"]
    width = blur_widths(X, bundle.get("blur") or 0)
    if width is not None and "spray_pull" in X:
        p = np.zeros((len(X), len(m.classes_)))
        Xb = X.copy()
        for u in np.linspace(-1, 1, BLUR_STEPS):
            Xb["spray_pull"] = (X["spray_pull"] + u * width).clip(-SPRAY_LIMIT, SPRAY_LIMIT)
            p += _park_avg(m, Xb, bundle)
        p /= BLUR_STEPS
    else:
        p = _park_avg(m, X, bundle)
    out = pd.DataFrame(p, columns=list(m.classes_), index=X.index).reindex(columns=CLASSES, fill_value=0.0)
    if bundle.get("hr_model") is not None and feats is not None:
        Xh, _ = model_inputs(feats.loc[X.index], bundle, HR_FEATURES)
        ph = _park_avg(bundle["hr_model"], Xh, bundle)[:, 1]
        rest = (1 - ph) / (1 - out["home_run"]).clip(lower=1e-6)
        for c in CLASSES:
            out[c] = ph if c == "home_run" else out[c] * rest
    return out


def probs_from_feats(feats: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """P(class) per ball. Blended bundles (spray strength): (1 - s) x no-spray + s x spray version, so
    EV / LA always lead and spray adjusts."""
    X, _ = model_inputs(feats, bundle)
    p = raw_probabilities(X, bundle, feats)
    if bundle.get("blend"):
        lam = bundle["blend"]["lam"]
        p = (1 - lam) * probs_from_feats(feats, bundle["blend"]["base"]) + lam * p
    return p


def describe(name: str) -> str:
    if "@" in name:
        v, lam = name.split("@")
        return f"no_spray + {float(lam):.0%} of {v}'s spray adjustment ({VARIANTS[v][2]})"
    return VARIANTS[name][2] if name in VARIANTS else name


def probabilities(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    bundle = bundle or load()
    return probs_from_feats(features(df), bundle)


# ----------------------------------------------------------------------------
_model_cache: dict = {}


def load() -> dict | None:
    """{'model', 'weights', 'trained_on', 'variant', 'features', 'parks', 'neutral_parks',
    'calib', 'report'} or None if not trained yet."""
    if "m" not in _model_cache:
        if not ARTIFACT.exists():
            return None
        import joblib
        _model_cache["m"] = joblib.load(ARTIFACT)
    return _model_cache["m"]


def _nearest(d: dict, y: int):
    return d.get(y) if y in d else d[min(d, key=lambda t: abs(t - y))]


def score_batted_balls(df: pd.DataFrame, bundle: dict | None = None, calibrate: bool = True) -> pd.Series:
    """Expected wOBA value for each batted ball (NaN for rows that aren't balls in play).
    Works on any Statcast frame: a season, a player, a single game -- or one ball."""
    bundle = bundle or load()
    out = pd.Series(np.nan, index=df.index)
    if bundle is None or df.empty:
        return out
    bip = df["type"].eq("X") if "type" in df else pd.Series(True, index=df.index)
    if not bip.any():
        return out
    d = df[bip]
    probs = probabilities(d, bundle)
    # weights: this data's own season weights where available, else the nearest trained season
    own = season_weights(d)
    trained = bundle["weights"]
    years = pd.to_datetime(d["game_date"]).dt.year if "game_date" in d else pd.Series(max(trained), index=d.index)
    calib = bundle.get("calib") or {}
    val = pd.Series(0.0, index=d.index)
    for y in years.unique():
        w = own.get(int(y)) or _nearest(trained, int(y))
        m = years.eq(y)
        val[m] = sum(probs.loc[m, h] * w[h] for h in HITS)
        if calibrate and calib:
            val[m] *= _nearest(calib, int(y))
    out[bip] = val
    return out


def calibration(df: pd.DataFrame, raw: pd.Series) -> dict[int, float]:
    """Per season: league actual wOBACON / league model xwOBACON (uncalibrated)."""
    ok = (pd.to_numeric(df["woba_denom"], errors="coerce").fillna(0) > 0) & raw.notna()
    y = pd.to_datetime(df["game_date"]).dt.year
    out = {}
    for yr, g in df[ok].groupby(y[ok]):
        pred = raw[g.index].mean()
        if pred > 0:
            out[int(yr)] = float(pd.to_numeric(g["woba_value"], errors="coerce").mean() / pred)
    return out

