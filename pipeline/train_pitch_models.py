"""Train the IQ pitch models (models/pitch_models.py): expected rates, grades, shape clusters. One set per era.

Reads the saved full downloads (data/raw/) -- no new download.

    python -m pipeline.train_pitch_models              # 2015 -> latest full season
    python -m pipeline.train_pitch_models --since 2023 # only the pitch-clock / shift-ban era

Tested on held-out games: models trained on the even game_pk's, scored on the odd ones; then refit on everything.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MAX_FIT = 2_000_000       # rows per model fit (random sample) -- plenty for these inputs, keeps a laptop happy
MIN_GROUP = 200           # pitches per pitcher-season-pitch type for the grade references
K_CLUSTERS = 14


def _clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=0)


def _reg():
    from pipeline.train_pitch_quality import _reg as r
    return r()


def _cap(idx: pd.Index) -> pd.Index:
    if len(idx) <= MAX_FIT:
        return idx
    return pd.Index(np.random.default_rng(0).choice(idx.to_numpy(), MAX_FIT, replace=False))


def targets(df: pd.DataFrame, f: pd.DataFrame) -> dict:
    """The masks + targets every model trains on."""
    from core import pitch_metrics as pm
    from models import pitch_quality as pq
    desc = df["description"].astype(str)
    swing, take = desc.isin(pm.SWINGS), desc.isin(pm.TAKES)
    rv_bat = pd.to_numeric(df["delta_run_exp"], errors="coerce")
    if rv_bat[desc.isin(["ball", "blocked_ball"])].mean() < 0:
        rv_bat = -rv_bat
    bip = df["type"].astype(str).eq("X")
    xq = pd.to_numeric(df["estimated_woba_using_speedangle"], errors="coerce")
    cnt = f["balls"].astype("Int64").astype(str) + "-" + f["strikes"].astype("Int64").astype(str)
    ok = bip & rv_bat.notna() & xq.notna()
    table = rv_bat[ok].groupby(cnt[ok]).mean().to_dict()
    slope = float(np.cov(xq[ok], rv_bat[ok])[0, 1] / xq[ok].var()) if ok.sum() > 100 else 0.85
    xrv = pq.target(df, table, slope, float(xq[ok].mean()) if ok.any() else 0.37)
    return {"swing": swing, "take": take, "ball": desc.isin(pm.BALLS - {"hit_by_pitch", "intent_ball", "pitchout"}),
            "whiff": desc.isin(pm.WHIFFS), "bip": bip, "rv_bat": rv_bat, "xrv": xrv}


def fit(df: pd.DataFrame, f: pd.DataFrame, rows: pd.Index | None = None) -> dict:
    """Every model on `rows` (default: all)."""
    from models import pitch_models as pmod
    from models import pitch_quality as pq
    rows = df.index if rows is None else rows
    t = targets(df, f)
    full = f[pmod.ALL].notna().all(axis=1) & f.index.isin(rows)
    flight = f["velo"].notna() & f.index.isin(rows)
    loc_ok = f[["x_in", "z_norm"]].notna().all(axis=1) & f.index.isin(rows)
    X = lambda m, cols: f.loc[_cap(f.index[m]), cols]
    y = lambda m, s: s.loc[_cap(f.index[m])]
    b = {}
    m = full
    b["sw"] = _clf().fit(X(m, pmod.ALL), y(m, t["swing"]).astype(int))
    m = full & t["take"]
    b["bl"] = _clf().fit(X(m, pmod.ALL), y(m, t["ball"]).astype(int))
    m = full & t["swing"]
    b["wf"] = _clf().fit(X(m, pmod.ALL), y(m, t["whiff"]).astype(int))
    m = full & t["bip"] & t["rv_bat"].notna()
    b["iprv"] = _reg().fit(X(m, pmod.ALL), y(m, t["rv_bat"]))
    m = full & t["xrv"].notna()
    b["xrv"] = _reg().fit(X(m, pmod.ALL), y(m, t["xrv"]))
    m = flight & t["xrv"].notna()
    b["act"] = _reg().fit(X(m, pmod.ACTION), y(m, t["xrv"]))
    m = loc_ok & t["xrv"].notna()
    b["loc"] = _reg().fit(X(m, pq.LOC_FEATURES), y(m, t["xrv"]))
    m = flight & f[pmod.SHAPE_VELO].notna().all(axis=1)
    b["shape_velo"] = _reg().fit(X(m, pmod.SHAPE_VELO), y(m, f["velo"]))
    b["cluster"] = clusters(df.loc[rows], f.loc[rows])
    return b


def clusters(df: pd.DataFrame, f: pd.DataFrame) -> dict | None:
    from sklearn.cluster import MiniBatchKMeans
    from models import pitch_models as pmod
    x = f[pmod.CLUSTER].copy()
    x[["d_velo", "d_ivb", "d_hb"]] = x[["d_velo", "d_ivb", "d_hb"]].fillna(0)
    ok = x.notna().all(axis=1)
    if ok.sum() < 1000:
        return None
    idx = x.index[ok]
    if len(idx) > 600_000:
        idx = pd.Index(np.random.default_rng(1).choice(idx.to_numpy(), 600_000, replace=False))
    a = x.loc[idx].to_numpy(float)
    mu, sd = a.mean(0), a.std(0)
    km = MiniBatchKMeans(n_clusters=K_CLUSTERS, random_state=0, n_init=5, batch_size=8192).fit((a - mu) / sd)
    lab = km.labels_
    pt = df.loc[idx, "pitch_type"].astype(str).to_numpy()
    names = []
    for k in range(K_CLUSTERS):
        m = lab == k
        top = pd.Series(pt[m]).value_counts().index[0] if m.any() else "?"
        v, ivb, hb = a[m, 0].mean(), a[m, 1].mean(), a[m, 2].mean()
        names.append(f"{top} {v:.0f} mph, {ivb:+.0f} IVB, {hb:+.0f} HB")
    seen = {}
    for i, n in enumerate(names):                                   # keep names unique
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[i] = f"{n} ({seen[n]})"
    return {"km": km, "mu": mu, "sd": sd, "names": names}


def grade_refs(df: pd.DataFrame, f: pd.DataFrame, sc: pd.DataFrame) -> dict:
    """League reference per grade: mean / SD of pitcher-season-pitch-type means (200+), by pitch type, + the
    shrinkage k from split-half (odd vs even pitches) agreement."""
    from models import pitch_models as pmod
    base = pd.DataFrame({"p": df["pitcher"].astype(int).to_numpy(), "y": pd.to_datetime(df["game_date"]).dt.year.to_numpy(),
                         "pt": df["pitch_type"].astype(str).to_numpy(), "ptype": f["ptype"].to_numpy(),
                         "stand": df["stand"].astype(str).to_numpy()}, index=df.index)
    out = {}
    for key, col in pmod.GRADED.items():
        d = base.assign(v=sc[col].to_numpy())
        if key in pmod.HAND:
            d = d[d["stand"] == pmod.HAND[key]]
        d = d.dropna(subset=["v", "ptype"])
        d["half"] = d.groupby(["p", "y", "pt"]).cumcount() % 2
        g = d.groupby(["p", "y", "pt"]).agg(n=("v", "size"), v=("v", "mean"), ptype=("ptype", "first"))
        g = g[g["n"] >= MIN_GROUP]
        if g.empty:
            continue
        by = {int(k): {"mean": float(x["v"].mean()), "sd": float(x["v"].std())} for k, x in g.groupby("ptype")
              if len(x) >= 20}
        h = d.groupby(["p", "y", "pt", "half"])["v"].agg(["mean", "size"]).unstack("half")
        h = h[(h[("size", 0)] >= MIN_GROUP / 2) & (h[("size", 1)] >= MIN_GROUP / 2)]
        r = float(np.corrcoef(h[("mean", 0)], h[("mean", 1)])[0, 1]) if len(h) > 20 else 0.0
        nh = float(h[("size", 0)].mean()) if len(h) else 100.0
        k = max(nh * (1 - r) / r, 1.0) if r > 0.02 else 1e5
        out[key] = {"by_type": by, "all": {"mean": float(g["v"].mean()), "sd": float(g["v"].std())},
                    "k": float(k), "split_half_r": r, "half_n": nh, "groups": int(len(g))}
    return out


def holdout_report(df: pd.DataFrame, f: pd.DataFrame, title: str) -> str:
    """Train on even game_pk's, score the odd ones."""
    from sklearn.metrics import log_loss, roc_auc_score
    from models import pitch_models as pmod
    fold = pd.to_numeric(df["game_pk"], errors="coerce").fillna(0).astype(int) % 2
    b = fit(df, f, df.index[fold == 0])
    test = df.index[fold == 1]
    sc = pmod._score_era(df.loc[test], f.loc[test], b)
    t = {k: v.loc[test] for k, v in targets(df, f).items()}
    L = [f"{title}: held-out games ({len(test):,} pitches), models trained on the other half", ""]
    for name, col, mask, yv in (("xSw%  (swing)", "xsw", pd.Series(True, index=test), t["swing"]),
                                ("xBl%  (ball | take)", "xbl", t["take"], t["ball"]),
                                ("xWf%  (miss | swing)", "xwf", t["swing"], t["whiff"])):
        m = mask & sc[col].notna()
        p, yy = sc.loc[m, col].clip(1e-4, 1 - 1e-4), yv[m].astype(int)
        base = np.full(len(yy), yy.mean())
        L.append(f"  {name:<22} predicted {p.mean():.1%} vs actual {yy.mean():.1%} | AUC {roc_auc_score(yy, p):.3f} | "
                 f"log loss {log_loss(yy, p):.4f} vs {log_loss(yy, base):.4f} with no model")
    for name, col, mask, yv in (("xIPRV (run value, in play)", "xiprv", t["bip"], t["rv_bat"]),
                                ("xResults (expected RV)", "xrv", pd.Series(True, index=test), t["xrv"])):
        m = mask & sc[col].notna() & yv.notna()
        L.append(f"  {name:<22} r = {np.corrcoef(sc.loc[m, col], yv[m])[0, 1]:.3f} with the pitch's (expected) value")
    # does xResults predict next-half ACTUAL results better than actual results do?
    d = pd.DataFrame({"p": df.loc[test, "pitcher"].astype(int), "y": pd.to_datetime(df.loc[test, "game_date"]).dt.year,
                      "h": np.where(pd.to_datetime(df.loc[test, "game_date"]).dt.month <= 6, 1, 2),
                      "x": sc["xrv"], "a": sc["ares"]})
    g = d.groupby(["y", "p", "h"]).agg(n=("a", "size"), x=("x", "mean"), a=("a", "mean")).reset_index()
    m = g[g.h == 1].merge(g[g.h == 2], on=["y", "p"], suffixes=("1", "2"))
    m = m[(m.n1 >= 300) & (m.n2 >= 300)]
    if len(m) > 20:
        L += ["", f"  1st half -> 2nd half ACTUAL run value ({len(m)} pitcher-seasons, 300+ pitches each half):",
              f"    xResults r = {m['x1'].corr(m['a2']):.3f}   aResults r = {m['a1'].corr(m['a2']):.3f}"]
    return "\n".join(L)


def grade_report(refs: dict) -> str:
    L = ["", "Grades: split-half agreement (odd vs even pitches) and the shrinkage it implies",
         f"  {'GRADE':<10}{'GROUPS':>8}{'r':>8}{'k':>9}   (a group of k pitches is graded halfway to average)"]
    for key, r in refs.items():
        L.append(f"  {key:<10}{r['groups']:>8,}{r['split_half_r']:>8.2f}{r['k']:>9.0f}")
    return "\n".join(L)


def eye_test(df: pd.DataFrame, f: pd.DataFrame, sc: pd.DataFrame, b: dict) -> str:
    from models import pitch_models as pmod
    last = int(pd.to_datetime(df["game_date"]).dt.year.max())
    m = pd.to_datetime(df["game_date"]).dt.year.eq(last)
    d = pd.DataFrame({"p": df.loc[m, "pitcher"].astype(int), "pt": df.loc[m, "pitch_type"].astype(str),
                      "velo": f.loc[m, "velo"], "ivb": f.loc[m, "ivb"], "hb": f.loc[m, "hb"], "x": sc.loc[m, "xrv"],
                      "s": sc.loc[m, "shape"], "c": sc.loc[m, "cluster"]})
    g = d.groupby(["p", "pt"]).agg(n=("x", "size"), x=("x", "mean"), s=("s", "mean"), velo=("velo", "mean"),
                                   ivb=("ivb", "mean"), hb=("hb", "mean")).reset_index()
    g = g[g.n >= 250]
    try:
        from core import mlb_api
        names = mlb_api.get_people(sorted(g["p"].unique().tolist()))
    except Exception:
        names = {}
    L = ["", f"{last} best pitches by xResults (250+ thrown), runs saved per 100:",
         f"  {'PITCHER':<24}{'PITCH':>6}{'N':>7}{'xRV/100':>9}{'SHAPE':>8}{'VELO':>7}{'IVB':>7}{'HB':>7}"]
    for _, r in g.sort_values("x", ascending=False).head(15).iterrows():
        L.append(f"  {names.get(int(r.p), str(r.p)):<24}{r.pt:>6}{int(r.n):>7,}{100 * r.x:>+9.2f}{100 * r.s:>+8.2f}"
                 f"{r.velo:>7.1f}{r.ivb:>7.1f}{r.hb:>7.1f}")
    cl = d["c"].value_counts()
    L += ["", "Shape clusters (share of pitches):"] + [f"  {k:<40}{v / len(d):>7.1%}" for k, v in cl.items()]
    return "\n".join(L)


def main(argv: list[str]) -> None:
    import joblib

    from models import pitch_models as pmod
    from models import pitch_quality as pq
    from pipeline.train_decisions import latest_full_season
    from pipeline.train_pitch_quality import load
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
        f = pq.features(df)
        print(f"\n{title}: {len(df):,} pitches. Testing on held-out games…", flush=True)
        rep = holdout_report(df, f, title)
        print(rep, flush=True)
        print(f"{title}: training on every pitch…", flush=True)
        b = fit(df, f)
        sc = pmod._score_era(df, f, b)
        b["grades"] = grade_refs(df, f, sc)
        extra = grade_report(b["grades"]) + eye_test(df, f, sc, b)
        print(extra, flush=True)
        rep += extra
        b["trained_on"] = ys
        models[key] = b
        reps.append(f"===== {title} =====\n{rep}")
        del df, f, sc
    bundle = models[next(iter(models))] if len(models) == 1 else {"eras": models, "split": pq.SHIFT_BAN}
    text = "\n\n".join(reps)
    bundle["report"] = text
    pmod.ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, pmod.ARTIFACT, compress=3)
    (pmod.ARTIFACT.parent / "pitch_models_report.txt").write_text(text + "\n")
    print(f"\nSaved {pmod.ARTIFACT.name} + pitch_models_report.txt in {(time.time() - t0) / 60:.0f} min.")


if __name__ == "__main__":
    main(sys.argv[1:])
