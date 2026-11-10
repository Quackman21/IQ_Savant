"""Train SAxwOBA (models/saxwoba.py): exit velo + launch angle + spray angle, nothing else.

Uses the batted balls already saved in data/train/ (no big download).

    python -m pipeline.train_saxwoba                # 2023+ (saved batted balls), test, train, save, report
    python -m pipeline.train_saxwoba --since 2015   # every Statcast season (uses the saved full downloads)

Filled-in tracking data is left out of training: when Statcast missed a ball (mostly 2015-2019) it filled in
default exit velo / launch angle values that depend on the outcome. Those show up as the exact same
EV / LA pair repeated far more often than real contact ever repeats (IMPUTED_MIN+ times in a season).

The site shows it right away as SAxwOBA / SAxwOBACON next to Savant's xwOBA / xwOBACON. MLB averages
for the new columns fill in after the next  python -m pipeline.update.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


IMPUTED_MIN = 50          # same exact EV + LA this many times in one season = filled-in, not measured
SHIFT_BAN = 2023          # same split as models/saxwoba.SHIFT_BAN


def imputed(df: pd.DataFrame) -> pd.Series:
    """True for batted balls whose EV / LA look filled in rather than measured."""
    ev = pd.to_numeric(df["launch_speed"], errors="coerce").round(1)
    la = pd.to_numeric(df["launch_angle"], errors="coerce").round(0)
    yr = pd.to_datetime(df["game_date"]).dt.year
    key = pd.DataFrame({"y": yr, "ev": ev, "la": la})
    n = key.groupby(["y", "ev", "la"])["ev"].transform("size")
    return (n >= IMPUTED_MIN) & ev.notna()


def load_seasons(since: int) -> dict[int, pd.DataFrame]:
    """Batted balls per season: the saved batted-ball files (2023+) or the saved full downloads."""
    from datetime import date

    from pipeline import build_league_reference as bl
    from pipeline import train_xwoba as tx
    out = {}
    for y in range(since, date.today().year + 1):
        path = tx.TRAIN_DIR / f"bip_{y}.csv.gz"
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
        elif bl.raw_path(y).exists() or bl.season_complete(y):
            raw = bl.pull_season(y)
            df = raw[raw["type"].astype(str).eq("X")][[c for c in tx.KEEP if c in raw.columns]]
        else:
            df = tx.season_bip(y)
        if df.empty:
            continue
        bad = imputed(df)
        out[y] = df[~bad].reset_index(drop=True)
        print(f"[{y}] {len(out[y]):,} batted balls ready ({bad.sum():,} filled-in left out)", flush=True)
    return out


def fit(df: pd.DataFrame) -> dict:
    from models import saxwoba as sa
    from models import xwoba_iq as xq
    from pipeline.train_xwoba import make_model
    y = xq.label(df["events"])
    keep = y.notna()
    model = make_model().fit(sa.features(df[keep])[sa.FEATURES], y[keep])
    return {"model": model, "features": sa.FEATURES, "weights": xq.season_weights(df),
            "trained_on": sorted(pd.to_datetime(df["game_date"]).dt.year.unique().tolist())}


def report(all_bip: dict[int, pd.DataFrame]) -> str:
    from sklearn.metrics import log_loss

    from models import saxwoba as sa
    from models import xwoba_iq as xq
    from pipeline.train_xwoba import MIN_BIP_HALF, future_r
    years = sorted(all_bip)
    order = sorted(xq.CLASSES)
    L = ["SAxwOBA -- exit velo + launch angle + spray angle. No sprint speed, side, park or calibration.",
         "Each season scored by a model trained WITHOUT that season.", "",
         "Per season: log loss (lower = better, no-skill in brackets) | per-ball error vs Savant's"]
    parts = []
    for yr in years:
        train = pd.concat([all_bip[t] for t in years if t != yr], ignore_index=True)
        test = all_bip[yr].copy()
        b = fit(train)
        lab = xq.label(test["events"])
        keep = lab.notna()
        p = pd.DataFrame(b["model"].predict_proba(sa.features(test[keep])[sa.FEATURES]),
                         columns=list(b["model"].classes_)).reindex(columns=order, fill_value=1e-9).values
        base = lab[keep].value_counts(normalize=True).reindex(order).fillna(1e-6).values
        test["sa"] = sa.score_batted_balls(test, b)
        ok = test["woba_denom"].fillna(0) > 0
        t = test[ok]
        err = lambda col: float(np.sqrt(((t[col] - t["woba_value"]) ** 2).mean()))
        L.append(f"  {yr}: {log_loss(lab[keep], p, labels=order):.4f} ({log_loss(lab[keep], np.tile(base, (keep.sum(), 1)), labels=order):.4f})"
                 f" | {err('sa'):.4f} vs {err('estimated_woba_using_speedangle'):.4f}")
        parts.append(t)
    s = pd.concat(parts, ignore_index=True)
    L += ["", "League average on contact: actual wOBACON vs SAxwOBACON vs Savant xwOBACON"]
    for yr, g in s.groupby(pd.to_datetime(s["game_date"]).dt.year):
        L.append(f"  {yr}: {g['woba_value'].mean():.3f}  vs  {g['sa'].mean():.3f}  vs  "
                 f"{g['estimated_woba_using_speedangle'].mean():.3f}")
    L += ["", f"Per-ball error, all seasons (typical gap between a ball's number and what it produced; lower = closer):",
          f"  SAxwOBA {np.sqrt(((s['sa'] - s['woba_value']) ** 2).mean()):.4f}   Savant xwOBA "
          f"{np.sqrt(((s['estimated_woba_using_speedangle'] - s['woba_value']) ** 2).mean()):.4f}",
          "", f"Predicting the future (hitters with {MIN_BIP_HALF}+ batted balls each half): 1st-half number vs "
          "ACTUAL 2nd-half wOBACON"]
    yrs = pd.to_datetime(s["game_date"]).dt.year
    for title, part in (("all seasons", s), (f"{SHIFT_BAN}+ only (shift ban -- today's game)", s[yrs >= SHIFT_BAN])):
        if part.empty:
            continue
        L.append(f"  {title}:")
        for lab_, col in (("SAxwOBACON", "sa"), ("Savant xwOBACON", "estimated_woba_using_speedangle"),
                          ("Actual wOBACON", "woba_value")):
            r, n = future_r(part, col)
            L.append(f"    {lab_:<17} r = {r:.3f}   ({n} hitter-seasons)")
        e = lambda c: float(np.sqrt(((part[c] - part["woba_value"]) ** 2).mean()))
        L.append(f"    per-ball error: SAxwOBA {e('sa'):.4f}   Savant {e('estimated_woba_using_speedangle'):.4f}")
    return "\n".join(L)


def main(argv: list[str]) -> None:
    import joblib

    from models import saxwoba as sa
    t0 = time.time()
    since = int(argv[argv.index("--since") + 1]) if "--since" in argv else SHIFT_BAN
    all_bip = load_seasons(since)
    if len(all_bip) < 2:
        print("Need at least two seasons of batted balls.")
        return
    pre = {y: d for y, d in all_bip.items() if y < sa.SHIFT_BAN}
    post = {y: d for y, d in all_bip.items() if y >= sa.SHIFT_BAN}
    eras = [("SHIFT ERA (2015-2022)", "pre", pre), ("SHIFT BAN (2023+)", "post", post)]
    reps, models = [], {}
    for title, key, part in eras:
        if len(part) < 2:
            continue
        print(f"\n{title}: testing on seasons the model hasn't seen…", flush=True)
        reps.append(f"===== {title} -- its own model, trained only on these seasons =====\n" + report(part))
        print(f"{title}: training on every season of the era…", flush=True)
        models[key] = fit(pd.concat(part.values(), ignore_index=True))
    if not models:
        print("Need at least two seasons of batted balls in an era.")
        return
    rep = "\n\n".join(reps)
    print("\n" + rep + "\n", flush=True)
    bundle = models["post"] if list(models) == ["post"] else {"eras": models, "split": sa.SHIFT_BAN,
                                                             "features": sa.FEATURES}
    bundle["report"] = rep
    sa.ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, sa.ARTIFACT, compress=3)
    (sa.ARTIFACT.parent / "saxwoba_report.txt").write_text(rep + "\n")
    print(f"Saved {sa.ARTIFACT.name} in {time.time() - t0:.0f}s.\n\nSAxwOBA / SAxwOBACON now show on player pages "
          "next to Savant's xwOBA. MLB averages for them fill in after the next  python -m pipeline.update.")


if __name__ == "__main__":
    main(sys.argv[1:])
