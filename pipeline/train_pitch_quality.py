"""Train IQ Stuff+ and Location+ (models/pitch_quality.py) on every pitch since 2015, one model per era.

Reads the saved full downloads (data/raw/) -- no new download.

    python -m pipeline.train_pitch_quality              # 2015 -> latest season, test, train, save, report
    python -m pipeline.train_pitch_quality --since 2023 # only today's game

Tested like the other models: each half of the games scored by a model trained on the other half.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

COLS = ["game_date", "game_pk", "pitcher", "batter", "p_throws", "stand", "pitch_type", "release_speed", "pfx_x",
        "pfx_z", "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z", "arm_angle",
        "plate_x", "plate_z", "sz_top", "sz_bot", "balls", "strikes", "description", "type", "events",
        "delta_run_exp", "estimated_woba_using_speedangle", "game_type"]
SKIP = {"pitchout", "intent_ball", "automatic_ball", "automatic_strike", "foul_pitchout"}
MIN_HALF = 300            # pitches per half for the repeatability test
MIN_BOARD = 1000          # pitches for the pitcher leaderboard
MIN_PITCH = 250           # pitches of one type for the best-pitches list


def load(years: list[int]) -> pd.DataFrame:
    from pipeline import build_league_reference as bl
    from pipeline.train_decisions import compact
    parts = []
    for y in years:
        raw = bl.pull_season(y)
        d = compact(raw[[c for c in COLS if c in raw.columns]])
        d = d[~d["description"].astype(str).isin(SKIP)]
        print(f"[{y}] {len(d):,} pitches ready", flush=True)
        parts.append(d)
        del raw
    df = pd.concat(parts, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")
    return df


def _reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                         min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=20, random_state=0)


def fit(df: pd.DataFrame, f: pd.DataFrame | None = None) -> dict:
    from models import pitch_quality as pq
    f = pq.features(df) if f is None else f
    rv = pd.to_numeric(df["delta_run_exp"], errors="coerce")
    if rv[df["description"].isin(["ball", "blocked_ball"])].mean() < 0:
        rv = -rv                                                     # batting side
    bip = df["type"].astype(str).eq("X")
    xq = pd.to_numeric(df["estimated_woba_using_speedangle"], errors="coerce")
    cnt = f["balls"].astype("Int64").astype(str) + "-" + f["strikes"].astype("Int64").astype(str)
    ok = bip & rv.notna() & xq.notna()
    table = rv[ok].groupby(cnt[ok]).mean().to_dict()
    slope = float(np.cov(xq[ok], rv[ok])[0, 1] / xq[ok].var()) if ok.sum() > 100 else 0.85
    y = pq.target(df, table, slope, float(xq[ok].mean()))
    s_ok = y.notna() & f["velo"].notna()
    l_ok = y.notna() & f[["x_in", "z_norm"]].notna().all(axis=1)
    stuff = _reg().fit(f.loc[s_ok, pq.STUFF_FEATURES], y[s_ok])
    loc = _reg().fit(f.loc[l_ok, pq.LOC_FEATURES], y[l_ok])
    sp = stuff.predict(f.loc[s_ok, pq.STUFF_FEATURES])
    lp = pd.Series(loc.predict(f.loc[l_ok, pq.LOC_FEATURES]), index=f.index[l_ok])
    g = f.loc[l_ok, "ptype"]
    return {"stuff": stuff, "loc": loc, "stuff_mean": float(sp.mean()), "stuff_sd": float(sp.std()),
            "loc_mean": lp.groupby(g).mean().to_dict(), "loc_sd": lp.groupby(g).std().to_dict(),
            "loc_mean_all": float(lp.mean()), "loc_sd_all": float(lp.std()),
            "bip_table": table, "slope": slope, "xq_mean": float(xq[ok].mean()),
            "trained_on": sorted(pd.to_datetime(df["game_date"]).dt.year.unique().tolist())}


def out_of_fold(df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    from models import pitch_quality as pq
    fold = pd.to_numeric(df["game_pk"], errors="coerce").fillna(0).astype(int) % 2
    parts = []
    for k in (0, 1):
        b = fit(df[fold != k], f[fold != k])
        parts.append(pq._score_one(f[fold == k], b))
    return pd.concat(parts).reindex(df.index)


def report(df: pd.DataFrame, f: pd.DataFrame, sc: pd.DataFrame, b: dict) -> str:
    from models import pitch_quality as pq
    y = pq.target(df, b["bip_table"], b["slope"], b["xq_mean"])
    actual = -pd.to_numeric(df["delta_run_exp"], errors="coerce")
    if actual[df["description"].isin(["ball", "blocked_ball"])].mean() > 0:
        actual = -actual                                             # pitcher side, actual
    d = pd.DataFrame({"pitcher": df["pitcher"].astype(int), "year": pd.to_datetime(df["game_date"]).dt.year,
                      "half": np.where(pd.to_datetime(df["game_date"]).dt.month <= 6, 1, 2),
                      "pt": df["pitch_type"].astype(str), "ptype": f["ptype"], "velo": f["velo"], "ivb": f["ivb"],
                      "hb": f["hb"], "stuff": sc["stuff_plus"], "loc": sc["loc_plus"], "rv": actual * 100,
                      "y": y})
    L = ["IQ Stuff+ / Location+ -- test report",
         "Each half of the games scored by a model trained on the OTHER half. 100 = average, 10 = 1 SD (pitch level).",
         "Run value is from the pitcher's side (+ = good for the pitcher); RV/100 = runs saved per 100 pitches.", ""]
    L.append(f"Pitch-level: Stuff+ vs that pitch's run value r = {d['stuff'].corr(d['y']):.3f} | "
             f"Location+ r = {d['loc'].corr(d['y']):.3f}  (single pitches are noisy -- small r is normal)")
    L += ["", "Average by pitch type:", f"  {'TYPE':<18}{'PITCHES':>11}{'STUFF+':>9}{'LOCATION+':>11}"]
    for code, g in d.groupby("ptype"):
        L.append(f"  {pq.PTYPE_NAME.get(int(code), '?'):<18}{len(g):>11,}{g['stuff'].mean():>9.1f}{g['loc'].mean():>11.1f}")

    g = d.groupby(["year", "pitcher", "half"]).agg(n=("rv", "size"), stuff=("stuff", "mean"), loc=("loc", "mean"),
                                                  rv=("rv", "mean")).reset_index()
    a, b2 = g[g.half == 1], g[g.half == 2]
    m = a.merge(b2, on=["year", "pitcher"], suffixes=("_1", "_2"))
    m = m[(m.n_1 >= MIN_HALF) & (m.n_2 >= MIN_HALF)]
    L += ["", f"Repeatable? pitcher's 1st half vs 2nd half ({MIN_HALF}+ pitches each half, {len(m)} pitcher-seasons):",
          f"  Stuff+     r = {m['stuff_1'].corr(m['stuff_2']):.3f}",
          f"  Location+  r = {m['loc_1'].corr(m['loc_2']):.3f}",
          f"  Actual RV/100  r = {m['rv_1'].corr(m['rv_2']):.3f}",
          "", "Predicting the 2nd half's ACTUAL RV/100 from the 1st half (higher = better):",
          f"  Stuff+         r = {m['stuff_1'].corr(m['rv_2']):.3f}",
          f"  Location+      r = {m['loc_1'].corr(m['rv_2']):.3f}",
          f"  Actual RV/100  r = {m['rv_1'].corr(m['rv_2']):.3f}"]
    if len(m) > 20:
        X = np.column_stack([m["stuff_1"], m["loc_1"], np.ones(len(m))])
        coef = np.linalg.lstsq(X, m["rv_2"], rcond=None)[0]
        L.append(f"  Stuff+ and Location+ together  r = {np.corrcoef(X @ coef, m['rv_2'])[0, 1]:.3f}")

    last = int(d["year"].max())
    cur = d[d.year == last]
    p = cur.groupby("pitcher").agg(n=("rv", "size"), stuff=("stuff", "mean"), loc=("loc", "mean"), rv=("rv", "mean"))
    p = p[p.n >= MIN_BOARD]
    pp = cur.groupby(["pitcher", "pt"]).agg(n=("rv", "size"), stuff=("stuff", "mean"), velo=("velo", "mean"),
                                           ivb=("ivb", "mean"), hb=("hb", "mean"), rv=("rv", "mean"))
    pp = pp[pp.n >= MIN_PITCH]
    try:
        from core import mlb_api
        names = mlb_api.get_people(sorted(set(p.index) | set(pp.index.get_level_values(0))))
    except Exception:
        names = {}
    nm = lambda i: names.get(int(i), str(i))
    if len(p) >= 10:
        L += ["", f"{last} pitchers by Stuff+ ({MIN_BOARD}+ pitches) -- eye test:",
              f"  {'PITCHER':<24}{'PITCHES':>8}{'STUFF+':>8}{'LOC+':>7}{'RV/100':>8}"]
        top = p.sort_values("stuff", ascending=False)
        for part in (top.head(10), top.tail(5)):
            for i, r in part.iterrows():
                L.append(f"  {nm(i):<24}{int(r.n):>8,}{r.stuff:>8.1f}{r['loc']:>7.1f}{r.rv:>+8.2f}")
            if part is not top.tail(5):
                L.append("  ...")
    if len(pp) >= 10:
        L += ["", f"{last} best individual pitches by Stuff+ ({MIN_PITCH}+ thrown):",
              f"  {'PITCHER':<24}{'PITCH':>6}{'N':>7}{'STUFF+':>8}{'VELO':>7}{'IVB':>7}{'HB':>7}{'RV/100':>8}"]
        for (i, pt), r in pp.sort_values("stuff", ascending=False).head(15).iterrows():
            L.append(f"  {nm(i):<24}{pt:>6}{int(r.n):>7,}{r.stuff:>8.1f}{r.velo:>7.1f}{r.ivb:>7.1f}{r.hb:>7.1f}"
                     f"{r.rv:>+8.2f}")
    return "\n".join(L)


def main(argv: list[str]) -> None:
    import joblib

    from models import pitch_quality as pq
    from pipeline.train_decisions import latest_full_season
    t0 = time.time()
    since = int(argv[argv.index("--since") + 1]) if "--since" in argv else 2015
    years = list(range(since, latest_full_season() + 1))
    eras = [("SHIFT ERA (2015-2022)", "pre", [y for y in years if y < pq.SHIFT_BAN]),
            ("PITCH CLOCK + SHIFT BAN (2023+)", "post", [y for y in years if y >= pq.SHIFT_BAN])]
    models, reps = {}, []
    for title, key, ys in eras:
        if not ys:
            continue
        df = load(ys)
        print(f"\n{title}: {len(df):,} pitches. Building inputs…", flush=True)
        f = pq.features(df)
        print(f"{title}: testing (each half of the games scored by a model trained on the other half)…", flush=True)
        sc = out_of_fold(df, f)
        print(f"{title}: training on every pitch…", flush=True)
        models[key] = fit(df, f)
        reps.append(f"===== {title} -- its own models, trained only on these seasons =====\n"
                    + report(df, f, sc, models[key]))
        print("\n" + reps[-1] + "\n", flush=True)
        del df, f, sc
    bundle = models["post"] if list(models) == ["post"] else (
        models["pre"] if list(models) == ["pre"] else {"eras": models, "split": pq.SHIFT_BAN})
    rep = "\n\n".join(reps)
    bundle["report"] = rep
    pq.ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, pq.ARTIFACT, compress=3)
    (pq.ARTIFACT.parent / "pitch_quality_report.txt").write_text(rep + "\n")
    print(f"Saved {pq.ARTIFACT.name} + pitch_quality_report.txt in {(time.time() - t0) / 60:.0f} min. "
          "Stuff+ / Location+ now show on the Pitching > Summary tab.")


if __name__ == "__main__":
    main(sys.argv[1:])
