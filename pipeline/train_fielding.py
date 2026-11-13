"""Train IQ fielding (models/fielding.py): out chance, who makes the out, who fields it, double plays, arm.

Reads the saved full downloads (data/raw/) -- no new download. One set of models per era.

    python -m pipeline.train_fielding            # 2015 -> latest season, test, train, save, report
    python -m pipeline.train_fielding --since 2023

Then  python -m pipeline.update  rebuilds every season's fielding file so the Defense page fills in.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

COLS = ["game_date", "game_pk", "at_bat_number", "inning", "inning_topbot", "outs_when_up", "events", "des",
        "type", "description", "hit_location", "launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y", "stand",
        "on_1b", "on_2b", "on_3b", "delta_run_exp", "plate_z", "sz_bot", "game_type"] + \
       [f"fielder_{k}" for k in range(2, 10)]
MIN_HALF = 80             # responsible plays per half for the repeatability test
MIN_INN = 400             # innings for the leaderboards


def load(years: list[int]) -> pd.DataFrame:
    from pipeline import build_league_reference as bl
    parts = []
    for y in years:
        raw = bl.pull_season(y)
        d = raw[[c for c in COLS if c in raw.columns]]
        if "game_type" in d:
            d = d[d["game_type"].astype(str).eq("R")]
        print(f"[{y}] {len(d):,} pitches ready", flush=True)
        parts.append(d.reset_index(drop=True))
        del raw
    return pd.concat(parts, ignore_index=True)


def _clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=100,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=0)


def _reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=100,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, random_state=0)


def fit(df: pd.DataFrame) -> dict:
    from models import fielding as fm
    d = df[fm.balls_in_play(df)]
    f = fm.features(d)
    ev = d["events"].astype(str)
    out = ev.isin(fm.OUTS).astype(int)
    rv = pd.to_numeric(d["delta_run_exp"], errors="coerce")
    sign = 1.0 if rv[ev.isin(["single", "double"])].mean() > 0 else -1.0
    rv = rv * sign                                                     # batting side
    m_out = _clf().fit(f[fm.FEATURES], out)
    loc = f["fielder"].between(1, 9)
    who = _clf().fit(f.loc[loc & (out == 1), fm.FEATURES], f.loc[loc & (out == 1), "fielder"].astype(int))
    fielded = _clf().fit(f.loc[loc, fm.FEATURES], f.loc[loc, "fielder"].astype(int))
    bucket = fm.run_bucket(f["la"])
    run_value = {}
    for bk in ("GB", "LD", "FB", "PU"):
        m = bucket.eq(bk) & rv.notna()
        if (m & (out == 0)).sum() > 50 and (m & (out == 1)).sum() > 50:
            run_value[bk] = float(rv[m & (out == 0)].mean() - rv[m & (out == 1)].mean())
    dp_opp = (f["la"] < 10) & f["on1"].eq(1) & (f["outs"] < 2) & f["fielder"].isin(fm.INFIELD)
    got = ev[dp_opp].isin(fm.DP).astype(int)
    dp = _clf().fit(f.loc[dp_opp, fm.FEATURES + ["fielder"]], got) if got.nunique() > 1 else None
    dp_value = float(rv[dp_opp & ~ev.isin(fm.DP)].mean() - rv[dp_opp & ev.isin(fm.DP)].mean()) if dp is not None else 0.0
    arm_opp = f["fielder"].isin(fm.OUTFIELD) & (f["on1"] + f["on2"] + f["on3"] > 0) & rv.notna()
    arm = _reg().fit(f.loc[arm_opp, fm.ARM_FEATURES], rv[arm_opp]) if arm_opp.sum() > 200 else None
    # after he reaches it: out chance once an infielder fields it (Conversion) + fielding-error chance (Ball Security)
    conv, ferr = None, None
    got_it = f["fielder"].isin(fm.INFIELD)
    if got_it.sum() > 500:
        Xc = f.loc[got_it, fm.FEATURES + ["fielder"]]
        yc = fm.converted(d[got_it]).astype(int)
        conv = _clf().fit(Xc, yc) if yc.nunique() > 1 else None
        ye = fm.fielding_error(d[got_it]).astype(int)
        ferr = _clf().fit(Xc, ye) if ye.sum() > 50 else None
    return {"out": m_out, "who": who, "fielded": fielded, "conv": conv, "ferr": ferr, "run_value": run_value, "dp": dp, "dp_value": dp_value,
            "arm": arm, "rv_sign": sign, "use_speed": fm.USE_SPEED,
            "trained_on": sorted(pd.to_datetime(df["game_date"]).dt.year.unique().tolist())}


def out_of_fold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each half of the games credited by models trained on the other half. Returns (credits, daily)."""
    from models import fielding as fm
    fold = pd.to_numeric(df["game_pk"], errors="coerce").fillna(0).astype(int) % 2
    credits, daily = [], []
    for k in (0, 1):
        b = fit(df[fold != k])
        part = df[fold == k]
        credits.append(fm.play_credits(part, b))
        daily.append(fm.daily_table(part, b))
    return pd.concat(credits, ignore_index=True), pd.concat(daily, ignore_index=True)


def report(df: pd.DataFrame, credits: pd.DataFrame, daily: pd.DataFrame, b: dict) -> str:
    from models import fielding as fm
    L = ["IQ fielding -- test report",
         "Each half of the games credited by models trained on the OTHER half. Runs from the fielding team's side.", ""]
    c = credits[credits.get("primary", pd.Series(dtype=float)).fillna(0) == 1] if "primary" in credits else credits
    if "p_out" in c:
        L += ["Out-chance check (balls in play, grouped by the model's out chance):",
              "  MODEL SAID     ACTUALLY OUT     PLAYS"]
        pc = c.drop_duplicates(subset=["game_pk", "p_out", "out"])
        for lo in np.arange(0, 1, 0.1):
            m = pc["p_out"].between(lo, lo + 0.1, inclusive="left" if lo < 0.9 else "both")
            if m.sum():
                L.append(f"  {lo:>4.0%}-{lo + 0.1:<4.0%}   {pc.loc[m, 'p_out'].mean():>5.0%} -> {pc.loc[m, 'out'].mean():>5.0%}"
                         f"{m.sum():>12,}")
    L.append(f"\n  run value of an out vs a hit by ball type: " +
             ", ".join(f"{k} {v:.2f}" for k, v in b["run_value"].items()) +
             f" | double play vs not: {b['dp_value']:.2f} runs")

    d = daily.copy()
    d["year"] = pd.to_datetime(d["game_date"]).dt.year
    d["half"] = np.where(pd.to_datetime(d["game_date"]).dt.month <= 6, 1, 2)
    L += ["", f"Repeatable? player's 1st half vs 2nd half at a position ({MIN_HALF}+ responsible plays each half):",
          f"  {'POS':<5}{'OPA / play r':>14}{'DRS / inning r':>16}{'PLAYERS':>9}"]
    for k, name in fm.POS.items():
        g = d[d.pos == k].groupby(["year", "player", "half"]).sum(numeric_only=True).reset_index()
        g["drs"] = g["runs"] + (g["dp_runs"] if k in fm.INFIELD else g["arm_runs"])
        a, b2 = g[g.half == 1], g[g.half == 2]
        m = a.merge(b2, on=["year", "player"], suffixes=("_1", "_2"))
        m = m[(m.n_1 >= MIN_HALF) & (m.n_2 >= MIN_HALF) & (m.outs_1 > 0) & (m.outs_2 > 0)]
        if len(m) > 10:
            r1 = (m.opa_1 / m.n_1).corr(m.opa_2 / m.n_2)
            r2 = (m.drs_1 / m.outs_1).corr(m.drs_2 / m.outs_2)
            L.append(f"  {name:<5}{r1:>14.3f}{r2:>16.3f}{len(m):>9}")

    last = int(d["year"].max())
    cur = d[d.year == last]
    try:
        from core import mlb_api
        names = mlb_api.get_people(sorted(cur["player"].astype(int).unique().tolist()))
    except Exception:
        names = {}
    L += ["", f"{last} DRS leaders by position ({MIN_INN}+ innings) -- eye test (models trained on the other half):"]
    for k, name in fm.POS.items():
        s = fm.summarize(cur, "0000", "9999", [k])
        rows = [(p, v) for p, v in s.items() if v["IQ_INN"] >= MIN_INN]
        if len(rows) < 6:
            continue
        rows.sort(key=lambda x: -x[1]["DRS"])
        extra = "DP" if k in fm.INFIELD else "ARM"
        L.append(f"  {name}:  " + " | ".join(
            f"{names.get(p, p)} {v['DRS']:+.0f}" for p, v in rows[:4]) + "   ...   " + " | ".join(
            f"{names.get(p, p)} {v['DRS']:+.0f}" for p, v in rows[-2:]))
    return "\n".join(L)


def main(argv: list[str]) -> None:
    import joblib

    from models import fielding as fm
    from pipeline.train_decisions import latest_full_season
    t0 = time.time()
    since = int(argv[argv.index("--since") + 1]) if "--since" in argv else 2015
    years = list(range(since, latest_full_season() + 1))
    eras = [("SHIFT ERA (2015-2022)", "pre", [y for y in years if y < fm.SHIFT_BAN]),
            ("SHIFT BAN (2023+)", "post", [y for y in years if y >= fm.SHIFT_BAN])]
    models, reps = {}, []
    for title, key, ys in eras:
        if not ys:
            continue
        df = load(ys)
        print(f"\n{title}: testing (each half of the games credited by models trained on the other half)…", flush=True)
        credits, daily = out_of_fold(df)
        print(f"{title}: training on every ball in play…", flush=True)
        models[key] = fit(df)
        reps.append(f"===== {title} -- its own models =====\n" + report(df, credits, daily, models[key]))
        print("\n" + reps[-1] + "\n", flush=True)
        del df, credits, daily
    bundle = models["post"] if list(models) == ["post"] else (
        models["pre"] if list(models) == ["pre"] else {"eras": models, "split": fm.SHIFT_BAN})
    rep = "\n\n".join(reps)
    bundle["report"] = rep
    fm.ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, fm.ARTIFACT, compress=3)
    (fm.ARTIFACT.parent / "fielding_report.txt").write_text(rep + "\n")
    print(f"Saved {fm.ARTIFACT.name} + fielding_report.txt in {(time.time() - t0) / 60:.0f} min. "
          "Run  python -m pipeline.update  to rebuild the fielding files the Defense page reads.")


if __name__ == "__main__":
    main(sys.argv[1:])
