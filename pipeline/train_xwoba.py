"""Train our xwOBA model (IQ xwOBA). Light: downloads BATTED BALLS ONLY (about a fifth of
the pitches the full data update needs), saves them locally so re-training is instant.
Ballparks come from MLB's schedule (one small call per season, also saved locally).

    python -m pipeline.train_xwoba --compare           # test every version side by side (saves nothing)
    python -m pipeline.train_xwoba --compare v1,air_dir   # only some versions
    python -m pipeline.train_xwoba                     # train the default version (no_spray), save, report
    python -m pipeline.train_xwoba --variant air       # train a different version
    python -m pipeline.train_xwoba --activate          # (with any of the above) switch the site to it
    python -m pipeline.train_xwoba --use-savant        # switch the site back to Savant's xwOBA
    python -m pipeline.train_xwoba --refresh           # re-download the current season first

Seasons: 2023 -> present (pitch-clock / shift-rule era). Swing only: no sprint speed.
Versions: see models/xwoba_iq.VARIANTS.
"""
from __future__ import annotations

import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FIRST_SEASON = 2023
TRAIN_DIR = Path(__file__).resolve().parent.parent / "data" / "train"
KEEP = ["game_date", "game_pk", "at_bat_number", "batter", "stand", "type", "events", "launch_speed",
        "launch_angle", "hc_x", "hc_y", "woba_value", "woba_denom", "estimated_woba_using_speedangle"]
MIN_BIP_HALF = 100
SEASON_MIN_BIP = 200


# ----------------------------------------------------------------------------
# 1. Batted balls (downloaded once, kept in data/train/, not pushed to GitHub)
# ----------------------------------------------------------------------------
def _week(start: str, end: str) -> pd.DataFrame:
    from core import savant
    wait = 15
    for attempt in range(1, 6):
        try:
            return savant.batted_balls(start, end)
        except Exception as e:
            if attempt == 5:
                raise RuntimeError(f"Savant kept failing for {start}..{end}: {e}") from e
            print(f"    {start}..{end} failed ({type(e).__name__}); retry {attempt}/4 in {wait}s", flush=True)
            time.sleep(wait)
            wait *= 2
    return pd.DataFrame()


def season_bip(year: int, refresh: bool = False) -> pd.DataFrame:
    path = TRAIN_DIR / f"bip_{year}.csv.gz"
    current = year == date.today().year and date.today().month <= 10
    if path.exists() and not (refresh and current):
        return pd.read_csv(path, low_memory=False)
    print(f"[{year}] downloading batted balls a week at a time…", flush=True)
    d, last = date(year, 3, 1), min(date(year, 10, 15), date.today())
    parts = []
    while d <= last:
        e = min(d + timedelta(days=6), last)
        wk = _week(d.isoformat(), e.isoformat())
        if not wk.empty:
            parts.append(wk[[c for c in KEEP if c in wk.columns]])
        print(f"    {e}  ({sum(len(p) for p in parts):,} batted balls)", flush=True)
        d = e + timedelta(days=1)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEEP)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def season_venues(year: int, refresh: bool = False) -> pd.DataFrame:
    """game_pk -> ballpark, from MLB's schedule (cached in data/train/)."""
    path = TRAIN_DIR / f"venues_{year}.csv"
    current = year == date.today().year and date.today().month <= 10
    if path.exists() and not (refresh and current):
        return pd.read_csv(path)
    from core import mlb_api
    v = mlb_api.get_game_venues(year)
    df = pd.DataFrame([(pk, vid, name) for pk, (vid, name) in v.items()],
                      columns=["game_pk", "venue", "venue_name"])
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def with_venue(bip: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    if venues.empty or "game_pk" not in bip:
        return bip.assign(venue=np.nan)
    return bip.drop(columns=["venue"], errors="ignore").merge(
        venues[["game_pk", "venue"]], on="game_pk", how="left")


# ----------------------------------------------------------------------------
# 2. Model
# ----------------------------------------------------------------------------
def make_model(categorical=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=80, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=0,
                                          categorical_features=categorical)


def fit(df: pd.DataFrame, variant: str = None, calibrate: bool = True) -> dict:
    from models import xwoba_iq as xq
    variant = variant or xq.DEFAULT_VARIANT
    if "@" in variant:                                   # spray strength: no_spray + lam x spray adjustment
        v, lam = variant.split("@")
        b = fit(df, v, calibrate=False)
        b["blend"] = {"lam": float(lam), "base": fit(df, "no_spray", calibrate=False)}
        b["variant"], b["description"] = variant, xq.describe(variant)
        if calibrate:
            b["calib"] = xq.calibration(df, xq.score_batted_balls(df, b, calibrate=False))
        return b
    cols, park_neutral, desc = xq.VARIANTS[variant]
    feats = xq.features(df)
    y = xq.label(df["events"])
    keep = y.notna()
    bundle = {"variant": variant, "description": desc, "features": cols, "blur": xq.BLUR.get(variant, 0),
              "weights": xq.season_weights(df),
              "trained_on": sorted(pd.to_datetime(df["game_date"]).dt.year.unique().tolist())}
    if park_neutral and feats["park"].notna().any():
        bundle["parks"], bundle["neutral_parks"] = xq.park_codes(feats["park"])
    X, used = xq.model_inputs(feats[keep], bundle)
    cat = [c == "park" for c in used] if "park" in used else None
    bundle["model"] = make_model(cat).fit(X, y[keep])
    if variant == "hr_dir":                              # P(home run) WITH direction
        Xh, used_h = xq.model_inputs(feats[keep], bundle, xq.HR_FEATURES)
        cat_h = [c == "park" for c in used_h] if "park" in used_h else None
        bundle["hr_model"] = make_model(cat_h).fit(Xh, (y[keep] == "home_run").astype(int))
    if calibrate:
        raw = xq.score_batted_balls(df, bundle, calibrate=False)
        bundle["calib"] = xq.calibration(df, raw)
    return bundle


# ----------------------------------------------------------------------------
# 3. Testing: every season scored by a model trained WITHOUT that season
# ----------------------------------------------------------------------------
def _player_halves(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per hitter per season: metric in 1st half vs actual wOBACON in 2nd half."""
    d = df[df["woba_denom"].fillna(0) > 0].copy()
    d["year"] = pd.to_datetime(d["game_date"]).dt.year
    d["half"] = np.where(pd.to_datetime(d["game_date"]).dt.month <= 6, 1, 2)
    g = d.groupby(["year", "batter", "half"]).agg(n=("woba_value", "size"), metric=(col, "mean"),
                                                   actual=("woba_value", "mean")).reset_index()
    a, b = g[g.half == 1], g[g.half == 2]
    m = a.merge(b, on=["year", "batter"], suffixes=("_1", "_2"))
    return m[(m.n_1 >= MIN_BIP_HALF) & (m.n_2 >= MIN_BIP_HALF)]


def future_r(s: pd.DataFrame, col: str) -> tuple[float, int]:
    h = _player_halves(s, col)
    return (h["metric_1"].corr(h["actual_2"]), len(h)) if len(h) > 10 else (np.nan, 0)


def same_season_r(s: pd.DataFrame, col: str, min_bip: int = 200) -> float:
    """How closely the number matches what actually happened that season (not the goal by itself:
    an expected stat that matches results too closely is carrying the luck in them)."""
    d = s[s["woba_denom"].fillna(0) > 0]
    g = d.groupby([pd.to_datetime(d["game_date"]).dt.year, "batter"]).agg(
        n=("woba_value", "size"), m=(col, "mean"), a=("woba_value", "mean"))
    g = g[g.n >= min_bip]
    return g["m"].corr(g["a"]) if len(g) > 10 else np.nan


def evaluate(all_bip: dict[int, pd.DataFrame], variant: str) -> dict:
    """Leave-one-season-out: each season scored by a model trained WITHOUT it."""
    from models import xwoba_iq as xq
    years = sorted(all_bip)
    order = sorted(xq.CLASSES)                                         # log_loss expects sorted labels
    labs, Ps, frames = [], [], []
    for y in years:
        train = pd.concat([all_bip[t] for t in years if t != y], ignore_index=True)
        test = all_bip[y].copy()
        if train.empty or test.empty:
            continue
        b = fit(train, variant, calibrate=False)
        feats_t = xq.features(test)
        lab = xq.label(test["events"])
        keep = lab.notna()
        Ps.append(xq.probs_from_feats(feats_t[keep], b)[order].values)
        labs.append(lab[keep].values)
        test["raw"] = xq.score_batted_balls(test, b, calibrate=False)
        test["year"] = y
        frames.append(test)
    return finish(variant, pd.concat(frames, ignore_index=True), np.concatenate(labs), np.vstack(Ps))


def finish(name: str, frame: pd.DataFrame, lab: np.ndarray, P: np.ndarray) -> dict:
    """Scores -> every number in the report. Each season calibrated to its own league wOBACON."""
    from sklearn.metrics import log_loss

    from models import xwoba_iq as xq
    order = sorted(xq.CLASSES)
    base = pd.Series(lab).value_counts(normalize=True).reindex(order).fillna(1e-6).values
    ok = frame["woba_denom"].fillna(0) > 0
    seasons, drift, x = [], [], pd.Series(np.nan, index=frame.index)
    for y, g in frame.groupby("year"):
        o = ok[g.index]
        actual, raw = g.loc[o, "woba_value"].mean(), g.loc[o, "raw"].mean()
        drift.append(raw - actual)
        seasons.append((y, int((~g["events"].isna()).sum()), actual, raw,
                        g.loc[o, "estimated_woba_using_speedangle"].mean()))
        x[g.index] = g["raw"] * (actual / raw if raw > 0 else 1.0)
    frame = frame.assign(x=x)
    s = frame[ok]
    r, n = future_r(s, "x")
    err = float(np.sqrt(((s["x"] - s["woba_value"]) ** 2).mean()))
    return {"variant": name, "seasons": seasons, "drift": drift, "future_r": r, "n": n,
            "same_r": same_season_r(s, "x"), "log_loss": float(log_loss(lab, P, labels=order)),
            "baseline": float(log_loss(lab, np.tile(base, (len(lab), 1)), labels=order)),
            "ball_err": err, "scored": s, "frame": frame, "lab": lab, "P": P}


def blend(ns: dict, v: dict, lam: float) -> dict:
    """no_spray + lam x (spray version - no_spray), from scores already made (no retraining)."""
    f = ns["frame"][["game_date", "batter", "events", "woba_value", "woba_denom",
                     "estimated_woba_using_speedangle", "year"]].copy()
    f["raw"] = (1 - lam) * ns["frame"]["raw"].values + lam * v["frame"]["raw"].values
    res = finish(f"{v['variant']}@{lam:g}", f, ns["lab"], (1 - lam) * ns["P"] + lam * v["P"])
    for k in ("frame", "P", "lab"):                          # keep memory down: blends only need scores
        res.pop(k, None)
    return res


BEST_CANDIDATES = ["no_spray", "air", "fly", "hr_dir", "blur18"]
STRENGTHS = (0.25, 0.5, 0.75)
MIN_PULL_AIR = 0.10


def choose(results: list[dict]) -> tuple[str, str]:
    """The rule, fixed in advance: must use spray meaningfully (pull-air credit >= MIN_PULL_AIR) and must
    NOT be reliably worse than no_spray at predicting the future (gap range reaches 0); among those, the
    most accurate per ball (lowest BALL ERR)."""
    spray = [r for r in results if r["variant"] != "no_spray"]
    ok = [r for r in spray if r.get("pull_air", -1) >= MIN_PULL_AIR and r.get("gap_hi", -1) >= 0]
    if ok:
        b = min(ok, key=lambda r: r["ball_err"])
        return b["variant"], (f"{len(ok)} version(s) passed both tests (pull-air credit >= +{MIN_PULL_AIR:.2f} and "
                              f"not reliably worse than no_spray at predicting); {b['variant']} is the most "
                              f"accurate per ball (BALL ERR {b['ball_err']:.4f}).")
    ok2 = [r for r in spray if r.get("pull_air", -1) >= MIN_PULL_AIR] or spray
    b = max(ok2, key=lambda r: r["future_r"])
    return b["variant"], ("No version passed both tests. Picked the spray version that loses the least "
                          f"predictive value while still crediting pulled air balls: {b['variant']} "
                          f"(future r {b['future_r']:.3f}, gap {b.get('gap', float('nan')):+.3f}).")


def report_one(res: dict) -> str:
    from models import xwoba_iq as xq
    lines = [f"IQ xwOBA ({res['variant']}: {xq.describe(res['variant'])})",
             "Test report -- each season scored by a model trained WITHOUT that season", ""]
    lines.append(f"{sum(t[1] for t in res['seasons']):,} batted balls | log loss {res['log_loss']:.4f} "
                 f"(no-skill baseline {res['baseline']:.4f}) | per-ball error {res['ball_err']:.3f}")
    lines += ["", "League check before calibration: actual wOBACON vs IQ xwOBACON vs Savant xwOBACON",
              "(calibration then sets IQ = actual for each season)"]
    for y, _, a, iq, sv in res["seasons"]:
        lines.append(f"  {y}: {a:.3f}  vs  {iq:.3f}  vs  {sv:.3f}")
    s = res["scored"]
    lines += ["", f"Predicting the future (hitters with {MIN_BIP_HALF}+ batted balls in each half):",
              "correlation of a 1st-half number with the hitter's ACTUAL 2nd-half wOBACON (higher = better)",
              f"  {'IQ xwOBACON':<17} r = {res['future_r']:.3f}   ({res['n']} hitter-seasons)"]
    for label, col in (("Savant xwOBACON", "estimated_woba_using_speedangle"), ("Actual wOBACON", "woba_value")):
        r, n = future_r(s, col)
        lines.append(f"  {label:<17} r = {r:.3f}   ({n} hitter-seasons)")
    return "\n".join(lines)


def reliability(results: list[dict], draws: int = 1000) -> str:
    """Are the FUTURE r differences real? Resample the hitter-seasons 1,000 times (same hitters for
    every version), give each version's 95% range and its gap to no_spray, and show r season by season."""
    base = next((r for r in results if r["variant"] == "no_spray"), results[0])
    halves = {}
    for r in results:
        h = _player_halves(r["scored"], "x").set_index(["year", "batter"])
        halves[r["variant"]] = h
    sv = _player_halves(base["scored"], "estimated_woba_using_speedangle").set_index(["year", "batter"])
    halves["Savant"] = sv
    keys = sorted(set.intersection(*(set(h.index) for h in halves.values())))
    if len(keys) < 30:
        return ""
    actual = halves[base["variant"]].loc[keys, "actual_2"].values
    X = {k: h.loc[keys, "metric_1"].values for k, h in halves.items()}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), (draws, len(keys)))

    def boot(m):
        a, b = m[idx], actual[idx]
        a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    rs = {k: boot(v) for k, v in X.items()}
    ref = rs[base["variant"]]
    years = sorted({y for y, _ in keys})
    yr_idx = {y: [i for i, (yy, _) in enumerate(keys) if yy == y] for y in years}
    L = ["", f"How reliable are the FUTURE r numbers? ({len(keys)} hitter-seasons resampled {draws:,} times)",
         f"  RANGE = 95% of resamples.  GAP = this version minus {base['variant']}, same hitters each time;",
         "  if the GAP range doesn't cross 0, the difference is real, not luck of which hitters were in the sample.",
         "", f"  {'VERSION':<12}{'FUTURE r':>9}{'RANGE':>16}{'GAP':>9}{'GAP RANGE':>18}   " +
         "".join(f"{y:>7}" for y in years)]
    for k, v in rs.items():
        point = np.corrcoef(X[k], actual)[0, 1]
        lo, hi = np.percentile(v, [2.5, 97.5])
        gap = v - ref
        glo, ghi = np.percentile(gap, [2.5, 97.5])
        for r_ in results:
            if r_["variant"] == k:
                r_["gap"], r_["gap_lo"], r_["gap_hi"] = float(np.mean(gap)), glo, ghi
        by_year = "".join(f"{np.corrcoef(X[k][yr_idx[y]], actual[yr_idx[y]])[0, 1]:>7.3f}" for y in years)
        g = "" if k == base["variant"] else f"{point - np.corrcoef(X[base['variant']], actual)[0, 1]:>+9.3f}" \
            f"{f'{glo:+.3f} to {ghi:+.3f}':>18}"
        L.append(f"  {k:<12}{point:>9.3f}{f'{lo:.3f}-{hi:.3f}':>16}{g if g else '':<27}   {by_year}")
    return "\n".join(L)


def report_compare(results: list[dict]) -> str:
    from models import xwoba_iq as xq
    s = results[0]["scored"]
    sv_r, n = future_r(s, "estimated_woba_using_speedangle")
    act_r, _ = future_r(s, "woba_value")
    w = max(len(r["variant"]) for r in results) + 2
    lines = ["IQ xwOBA -- version comparison",
             "Every number: each season scored by a model trained WITHOUT that season.", "",
             "Columns:",
             "  FUTURE r   1st-half number vs the hitter's ACTUAL 2nd-half wOBACON. THE main test. Higher = better.",
             "  SAME r     season number vs that season's actual wOBACON (200+ BIP). How closely it matches",
             "             what happened -- higher is NOT automatically better: luck matches results too.",
             "  LOG LOSS   how well it predicts each ball's outcome (lower = better). Park-neutral versions",
             "             are scored in the average park, not the real one, so they give up a little here.",
             "  BALL ERR   per-ball accuracy: typical gap between a ball's xwOBA and the wOBA it produced",
             "             (lower = each ball's number is closer to what happened).",
             "  DRIFT      worst season: league IQ xwOBACON minus actual, BEFORE calibration fixes it.",
             "  name@0.5 = no_spray + 50% of that version's spray adjustment (EV / LA lead, spray adjusts).", "",
             f"  {'VERSION':<{w}}{'FUTURE r':>9}{'SAME r':>9}{'LOG LOSS':>10}{'BALL ERR':>10}{'DRIFT':>9}   WHAT IT IS"]
    for r in results:
        d = max(r["drift"], key=abs)
        lines.append(f"  {r['variant']:<{w}}{r['future_r']:>9.3f}{r['same_r']:>9.3f}{r['log_loss']:>10.4f}"
                     f"{r['ball_err']:>10.4f}{d:>+9.3f}   {xq.describe(r['variant'])}")
    lines += [f"  {'(no-skill)':<{w}}{'':>9}{'':>9}{results[0]['baseline']:>10.4f}",
              "", "References (same hitters):",
              f"  {'Savant xwOBA':<{w}}{sv_r:>9.3f}{same_season_r(s, 'estimated_woba_using_speedangle'):>9.3f}"
              "   (uses sprint speed on some ground balls)",
              f"  {'Actual wOBACON':<{w}}{act_r:>9.3f}{1:>9.3f}",
              f"  {'Savant per-ball error':<{w}}{float(np.sqrt(((s['estimated_woba_using_speedangle'] - s['woba_value']) ** 2).mean())):>9.4f}",
              f"  ({n} hitter-seasons with {MIN_BIP_HALF}+ batted balls in each half)"]
    return "\n".join(lines)


def direction_skill(results: list[dict]) -> str:
    """Is the direction credit a SKILL? Credit = a version's value minus no_spray's on the same ball.
      REPEAT   hitter's 1st-half credit vs his 2nd-half credit (skill repeats, holes don't)
      PULL-AIR season credit vs the hitter's pulled-air-ball rate (EV > 80, LA 10-45, pull 10+ deg)
      OPPO-GB  season credit vs the hitter's oppo-ground-ball rate (LA < 10, oppo 15+ deg)
      SIZE     how far direction moves hitters' season xwOBACON (SD across hitters, wOBA points)"""
    from models import xwoba_iq as xq
    base = next((r for r in results if r["variant"] == "no_spray"), None)
    if base is None:
        return "\n(direction skill check needs no_spray in the comparison)"
    s = base["scored"][["game_date", "batter", "woba_value", "woba_denom"]].copy()
    f = xq.features(base["scored"])
    s["pull_air"] = ((f["ev"] > 80) & f["la"].between(10, 45) & (f["spray_pull"] >= 10)).astype(float)
    s["oppo_gb"] = ((f["la"] < 10) & (f["spray_pull"] <= -15)).astype(float)
    s["year"] = pd.to_datetime(s["game_date"]).dt.year
    lines = ["", "Direction as a SKILL (credit = this version minus no_spray, same balls):",
             "  REPEAT    a hitter's 1st-half direction credit vs his 2nd-half credit (higher = real skill;",
             "            holes and luck don't repeat)",
             "  PULL-AIR  season credit vs his rate of pulled air balls     (should be clearly +)",
             "  OPPO-GB   season credit vs his rate of opposite-field grounders (+ if the model rewards it)",
             "  SIZE      how much direction moves hitters' season xwOBACON (SD, in points of wOBA)", "",
             f"  {'VERSION':<12}{'REPEAT':>8}{'PULL-AIR':>10}{'OPPO-GB':>9}{'SIZE':>7}"]
    for r in results:
        if r["variant"] == "no_spray" or len(r["scored"]) != len(s):
            continue
        s["c"] = r["scored"]["x"].values - base["scored"]["x"].values
        h = _player_halves(s.assign(woba_value=s["c"]), "c")       # metric = credit, both halves
        rep = h["metric_1"].corr(h["actual_2"]) if len(h) > 10 else np.nan
        g = s.groupby(["year", "batter"]).agg(n=("c", "size"), c=("c", "mean"), pa=("pull_air", "mean"),
                                             og=("oppo_gb", "mean"))
        g = g[g.n >= SEASON_MIN_BIP]
        r["repeat"], r["pull_air"], r["oppo_gb"] = rep, g["c"].corr(g["pa"]), g["c"].corr(g["og"])
        lines.append(f"  {r['variant']:<12}{rep:>8.2f}{r['pull_air']:>+10.2f}{r['oppo_gb']:>+9.2f}"
                     f"{1000 * g['c'].std():>7.0f}")
    return "\n".join(lines)


def gb_spray_chart(results: list[dict], path: Path) -> str:
    """The 'hole' check: hard-hit ground balls, value by exact spray angle. If a version follows the
    spikes in ACTUAL value (balls through the holes) it is learning the defense, not the swing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from models import xwoba_iq as xq
    base = results[0]["scored"]
    f = xq.features(base)
    gb = (f["la"] < xq.AIR_LA) & f["ev"].between(85, 105) & f["spray_pull"].between(-45, 45)
    bins = np.arange(-45, 46, 3)
    cut = pd.cut(f.loc[gb, "spray_pull"], bins)
    mid = [(i.left + i.right) / 2 for i in cut.cat.categories]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    actual = base.loc[gb, "woba_value"].groupby(cut, observed=False).mean().values
    ax.plot(mid, actual, color="#16202b", lw=2.6, marker="o", ms=4, label="Actual (what happened)")
    colors = ["#c8102e", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf",
              "#bcbd22", "#7f7f7f", "#393b79", "#d62728"]
    lines = ["", "Ground-ball hole check (85-105 mph ground balls, 3-degree bins, pull-adjusted spray):",
             "  how much each version swings with the actual spikes (higher = following the holes).",
             "  no_spray is the floor (~0.5: hitters' EV / LA differ by direction) -- near it = no hole leak:"]
    for r, c in zip(results, colors):
        x = r["scored"].loc[gb, "x"].groupby(cut, observed=False).mean().values
        ax.plot(mid, x, color=c, lw=1.6, label=r["variant"])
        ok = ~np.isnan(actual) & ~np.isnan(x)
        follow = np.corrcoef(np.diff(actual[ok]), np.diff(x[ok]))[0, 1] if ok.sum() > 3 else np.nan
        lines.append(f"    {r['variant']:<10} {follow:+.2f}")
    ax.axvline(-15, color="#999", ls=":", lw=1)
    ax.axvline(15, color="#999", ls=":", lw=1)
    ax.set_xlabel("Spray angle (degrees)   <- opposite field      |      pulled ->")
    ax.set_ylabel("wOBA value per ground ball")
    ax.set_title("Hard-hit ground balls: actual value vs each version, by exact direction")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
def load_all(refresh: bool) -> dict[int, pd.DataFrame]:
    all_bip = {}
    for y in range(FIRST_SEASON, date.today().year + 1):
        df = season_bip(y, refresh)
        if df.empty:
            continue
        try:
            ven = season_venues(y, refresh)
        except Exception as e:                       # no park list -> park-neutral versions can't run
            print(f"[{y}] couldn't get ballparks from MLB ({type(e).__name__}); continuing without", flush=True)
            ven = pd.DataFrame()
        df = with_venue(df, ven)
        all_bip[y] = df
        print(f"[{y}] {len(df):,} batted balls ready ({df['venue'].notna().mean():.0%} matched to a ballpark)",
              flush=True)
    return all_bip


def main(argv: list[str]) -> None:
    from models import xwoba
    from models import xwoba_iq as xq

    if "--use-savant" in argv:
        xwoba.set_active("savant")
        print("Site switched back to Savant xwOBA. Run `python -m pipeline.update` when you step away "
              "so MLB averages match.")
        return
    t0 = time.time()
    all_bip = load_all("--refresh" in argv)
    if len(all_bip) < 2:
        print("Need at least two seasons of batted balls -- check the internet connection and try again.")
        return
    xq.ARTIFACT.parent.mkdir(parents=True, exist_ok=True)

    if "--best" in argv:
        i = argv.index("--best")
        cands = argv[i + 1].split(",") if i + 1 < len(argv) and not argv[i + 1].startswith("--") else BEST_CANDIDATES
        if "no_spray" not in cands:
            cands = ["no_spray"] + cands
        results = []
        for k, v in enumerate(cands, 1):
            print(f"\n[{k}/{len(cands)}] testing {v}: {xq.describe(v)} …", flush=True)
            results.append(evaluate(all_bip, v))
            print(f"    future r {results[-1]['future_r']:.3f}   per-ball error {results[-1]['ball_err']:.4f}   "
                  f"({time.time() - t0:.0f}s so far)", flush=True)
        pure = list(results)
        for v in pure[1:]:
            for lam in STRENGTHS:
                results.append(blend(pure[0], v, lam))
        print(f"\nScored {len(results) - len(pure)} spray-strength blends. Writing the report…", flush=True)
        rep = report_compare(results) + "\n" + reliability(results) + "\n" + direction_skill(results)
        choice, why = choose(results)
        rep += "\n\nTHE PICK: " + choice + " -- " + xq.describe(choice) + "\n  " + why
        chosen = next(r for r in results if r["variant"] == choice)
        shown = pure + ([chosen] if chosen not in pure else [])
        png = xq.ARTIFACT.parent / "xwoba_gb_spray_check.png"
        rep += gb_spray_chart(shown, png)
        (xq.ARTIFACT.parent / "xwoba_compare_report.txt").write_text(rep + "\n")
        print("\n" + rep, flush=True)
        for r in pure:                                     # free memory before the final fit
            r.pop("frame", None)
            r.pop("P", None)
        print(f"\nTraining the final model ({choice}) on every season…", flush=True)
        import joblib
        bundle = fit(pd.concat(all_bip.values(), ignore_index=True), choice)
        bundle["report"] = rep
        joblib.dump(bundle, xq.ARTIFACT, compress=3)
        (xq.ARTIFACT.parent / "xwoba_iq_v1_report.txt").write_text(rep + "\n")
        print(f"Saved {xq.ARTIFACT.name} ({choice}) in {time.time() - t0:.0f}s", flush=True)
        try:
            from pipeline import xwoba_videos
            xwoba_videos.save_scored(shown)
            print("\nBuilding the batted-ball video check (outfield balls in the air)…", flush=True)
            xwoba_videos.main([])
        except Exception as e:
            print(f"(video check skipped: {type(e).__name__}: {e})")
        if "--activate" in argv:
            xwoba.set_active("iq_v1")
            print("\nThe site now uses IQ xwOBA (" + choice + "). Run `python -m pipeline.update` when you step away.")
        else:
            print("\nNot switched on yet. After looking at the report + videos:  "
                  "python -m pipeline.train_xwoba --use-saved --activate")
        return

    if "--use-saved" in argv and "--activate" in argv:
        if xq.load() is None:
            print("No trained model saved yet.")
            return
        xwoba.set_active("iq_v1")
        print(f"The site now uses the saved IQ xwOBA ({xq.load().get('variant')}). "
              "Run `python -m pipeline.update` when you step away.")
        return

    if "--compare" in argv:
        i = argv.index("--compare")
        pick = argv[i + 1].split(",") if i + 1 < len(argv) and not argv[i + 1].startswith("--") else list(xq.VARIANTS)
        bad = [v for v in pick if v not in xq.VARIANTS]
        if bad:
            print(f"Unknown version(s): {bad}. Choose from {list(xq.VARIANTS)}")
            return
        results = []
        for k, v in enumerate(pick, 1):
            print(f"\n[{k}/{len(pick)}] testing {v}: {xq.VARIANTS[v][2]} …", flush=True)
            results.append(evaluate(all_bip, v))
            print(f"    future r {results[-1]['future_r']:.3f}   ({time.time() - t0:.0f}s so far)", flush=True)
        rep = report_compare(results) + "\n" + reliability(results) + "\n" + direction_skill(results)
        png = xq.ARTIFACT.parent / "xwoba_gb_spray_check.png"
        rep += gb_spray_chart(results, png)
        (xq.ARTIFACT.parent / "xwoba_compare_report.txt").write_text(rep + "\n")
        print("\n" + rep)
        print(f"\nSaved models/artifacts/xwoba_compare_report.txt and {png.name} ({time.time() - t0:.0f}s).")
        print("Nothing was switched. To train one:  python -m pipeline.train_xwoba --variant NAME")
        try:                                          # video check: outfield air balls, every version side by side
            from pipeline import xwoba_videos
            xwoba_videos.save_scored(results)
            print("\nBuilding the batted-ball video check (outfield balls in the air)…", flush=True)
            xwoba_videos.main([])
        except Exception as e:
            print(f"(video check skipped: {type(e).__name__}: {e})")
        return

    variant = xq.DEFAULT_VARIANT
    if "--variant" in argv:
        variant = argv[argv.index("--variant") + 1]
        if variant.split("@")[0] not in xq.VARIANTS:
            print(f"Unknown version {variant}. Choose from {list(xq.VARIANTS)}")
            return
    print(f"\nTesting {variant} on seasons the model hasn't seen…", flush=True)
    rep = report_one(evaluate(all_bip, variant))
    print("\n" + rep + "\n", flush=True)

    print("Training the final model on every season…", flush=True)
    import joblib
    bundle = fit(pd.concat(all_bip.values(), ignore_index=True), variant)
    bundle["report"] = rep
    joblib.dump(bundle, xq.ARTIFACT, compress=3)
    (xq.ARTIFACT.parent / "xwoba_iq_v1_report.txt").write_text(rep + "\n")
    print(f"Saved {xq.ARTIFACT} ({xq.ARTIFACT.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")

    if "--activate" in argv:
        xwoba.set_active("iq_v1")
        print("\nThe site now uses IQ xwOBA. Player pages switch right away. MLB averages / colours switch "
              "after the next data update (python -m pipeline.update) -- run it when you step away.")
    else:
        print("\nNot switched on yet. Look over the report, then run:  "
              f"python -m pipeline.train_xwoba --variant {variant} --activate")


if __name__ == "__main__":
    main(sys.argv[1:])
