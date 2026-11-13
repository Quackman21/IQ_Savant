"""Train the league contact model for the Bat-To-Ball Grade and build each season's league reference.

    python -m pipeline.train_contact              # train (2015-2022 + 2023+ models), then every season's reference
    python -m pipeline.train_contact --refs-only  # just rebuild the season references
    python -m pipeline.train_contact --top 2026   # eye test: grade every qualified hitter, top / bottom 10

Uses the saved pitches (data/raw). The model answers one question for every swing: how often does an average
hitter make contact on THIS pitch (location, pitch type, velo, movement, count, hands)? Each era is tested on the
games it didn't train on first (odd / even games), then trained on everything.
"""
from __future__ import annotations

import sys
import time
from datetime import date

import numpy as np
import pandas as pd

FIRST = 2015


def _swings(year: int) -> pd.DataFrame:
    from models import contact as cm
    from pipeline import build_league_reference as bl
    raw = bl.pull_season(year)
    if "game_type" in raw:
        raw = raw[raw["game_type"].astype(str).eq("R")]
    t = cm.tag_swings(raw)
    sw = raw[t["swing"]]
    X = cm.features(sw)[cm.FEATURES]
    X["y"] = t.loc[sw.index, "contact"].astype(int).values
    X["game_pk"] = pd.to_numeric(sw["game_pk"], errors="coerce").values
    return X.dropna(subset=["nx", "nz", "velo"])


def _clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from models import contact as cm
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
                                          categorical_features=[cm.FEATURES.index("ptype")], early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20, random_state=0)


def train(years: list[int]) -> dict:
    import joblib
    from sklearn.metrics import roc_auc_score
    from models import contact as cm
    eras, report = {}, []
    for key, ys in (("pre", [y for y in years if y < cm.SPLIT]), ("post", [y for y in years if y >= cm.SPLIT])):
        if not ys:
            continue
        t0 = time.time()
        D = pd.concat([_swings(y) for y in ys], ignore_index=True)
        X, y = D[cm.FEATURES].fillna(-1), D["y"]
        fold = D["game_pk"].fillna(0).astype(int) % 2
        m = _clf().fit(X[fold == 0], y[fold == 0])
        p = m.predict_proba(X[fold == 1])[:, 1]
        held = y[fold == 1]
        bins = pd.cut(p, [0, .3, .5, .7, .8, .9, 1.0])
        cal = pd.DataFrame({"pred": p, "act": held.values}).groupby(bins, observed=True).mean()
        line = (f"{key} ({ys[0]}-{ys[-1]}): {len(D):,} swings | test AUC {roc_auc_score(held, p):.3f} | "
                "predicted vs actual contact: " + "  ".join(f"{r.pred:.2f}->{r.act:.2f}" for _, r in cal.iterrows()))
        print(line, flush=True)
        report.append(line)
        eras[key] = _clf().fit(X, y)
        print(f"  trained in {(time.time() - t0) / 60:.1f} min", flush=True)
        del D, X, y
    bundle = {"eras": eras, "split": cm.SPLIT, "report": "\n".join(report)}
    cm.ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cm.ARTIFACT, compress=3)
    (cm.ARTIFACT.parent / "contact_model_report.txt").write_text(bundle["report"] + "\n")
    cm._cache.clear()
    return bundle


def build_ref(season: int) -> None:
    from models import contact as cm
    from pipeline import build_league_reference as bl
    t0 = time.time()
    ref = cm.build_ref(bl.pull_season(season), season)
    p = cm.save_ref(ref)
    zones = ", ".join(f"{k} {v:.0%}" for k, v in ref["league_contact_by_zone"].items())
    print(f"[{season}] Bat-To-Ball reference saved ({ref['n_qualified']} qualified hitters, "
          f"{ref['qualified_pa']:.0f}+ PA; league contact {zones}) -> {p.name} in {time.time() - t0:.0f}s", flush=True)


def top(season: int) -> None:
    from models import contact as cm
    from models import power
    from pipeline import build_league_reference as bl
    ref = cm.load_ref(season)
    raw = bl.pull_season(season)
    if "game_type" in raw:
        raw = raw[raw["game_type"].astype(str).eq("R")]
    q, _ = power.qualified(raw)
    x = cm.expected(raw)
    rows = []
    for b in q:
        g = raw[raw["batter"] == b]
        t = cm.hitter_table(g.assign(batter=0), x.loc[g.index])
        if t.empty:
            continue
        r = t.iloc[0]
        n = r["swings"]
        z = {m: (r[m] - ref["stats"][m][0]) * n / (n + cm.K[m]) / ref["stats"][m][1] if r[m] == r[m] else 0
             for m in cm.INGREDIENTS}
        rows.append({"batter": int(b), "grade": 50 + 10 * sum(v * z[m] for m, v in cm.W.items()) / ref["sd_blend"]})
    t = pd.DataFrame(rows).sort_values("grade", ascending=False)
    try:
        from core import mlb_api
        names = mlb_api.get_people(t["batter"].tolist())
    except Exception:
        names = {}
    f = lambda r: f"{names.get(r.batter, r.batter)} {r.grade:.0f}"
    print(f"\n{season} Bat-To-Ball Grade, {len(t)} MLB-qualified hitters:")
    print("  top:    " + " | ".join(f(r) for r in t.head(10).itertuples()))
    print("  bottom: " + " | ".join(f(r) for r in t.tail(10).itertuples()))
    print(f"  spread check: middle 50% from {t.grade.quantile(.25):.0f} to {t.grade.quantile(.75):.0f} "
          "(should be about 43 to 57)")


def main(argv: list[str]) -> None:
    from pipeline.train_decisions import latest_full_season
    if "--top" in argv:
        top(int(argv[argv.index("--top") + 1]))
        return
    last = latest_full_season()
    years = list(range(FIRST, last + 1))
    if "--refs-only" not in argv:
        train(years)
    for y in range(FIRST, date.today().year + 1):
        try:
            build_ref(y)
        except Exception as e:
            print(f"[{y}] Bat-To-Ball reference FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
