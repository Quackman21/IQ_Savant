"""Train IQ catcher blocking + throwing (models/catching.py), one model each per era.

Needs the play-by-play runner events (wild pitches, passed balls, steals) -- downloads any missing season
first (pipeline.fetch_events, one call per game, step-away job the first time). Pitches come from data/raw/.

    python -m pipeline.train_catching            # 2015 -> latest season
    python -m pipeline.train_catching --since 2023

Then  python -m pipeline.update  rebuilds the fielding files so blocking / throwing fill in on any period.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

COLS = ["game_date", "game_pk", "at_bat_number", "pitch_number", "description", "type", "plate_x", "plate_z",
        "sz_bot", "release_speed", "pfx_x", "pfx_z", "pitch_type", "p_throws", "stand", "balls", "strikes",
        "outs_when_up", "on_1b", "on_2b", "on_3b", "fielder_2", "game_type", "inning", "inning_topbot",
        "bat_score", "post_bat_score"]
MIN_HALF_PITCHES = 300
MIN_BOARD = 1500          # blocking opportunities, for the leaderboards


def load(years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Runners-on pitches, runner events, and what a WP / PB costs in each base / out state (era run expectancy)."""
    from models import catching as cm
    from pipeline import build_league_reference as bl
    from pipeline import fetch_events
    parts, evs, re_parts = [], [], []
    for y in years:
        raw = bl.pull_season(y)
        d = raw[[c for c in COLS if c in raw.columns]]
        if "game_type" in d:
            d = d[d["game_type"].astype(str).eq("R")]
        re_parts.append(pd.Series(cm.run_expectancy(d)))
        on = d["on_1b"].notna() | d["on_2b"].notna() | d["on_3b"].notna()
        d = d[on].reset_index(drop=True)                      # only runners-on pitches matter here
        ev = fetch_events.season_events(y)
        print(f"[{y}] {len(d):,} pitches with runners on, {len(ev):,} runner events", flush=True)
        parts.append(d)
        evs.append(ev)
        del raw
    re = pd.concat(re_parts, axis=1).mean(axis=1).to_dict()           # era run expectancy (season average)
    return pd.concat(parts, ignore_index=True), pd.concat(evs, ignore_index=True), cm.wp_values(re)


def _clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=100,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=0)


def fit(df: pd.DataFrame, events: pd.DataFrame, wpv: dict | None = None) -> dict:
    from models import catching as cm
    d = df[cm.block_rows(df)]
    y = cm.wp_pb_flag(d, events)
    block = _clf().fit(cm.features(d)[cm.BLOCK_FEATURES], y.astype(int)) if y.nunique() > 1 else None
    att = cm.steal_attempts(df, events)
    throw = None
    if not att.empty and att["cs"].nunique() > 1:
        throw = _clf().fit(cm.throw_features(att)[cm.THROW_FEATURES], att["cs"].astype(int))
    wpv = wpv or {}
    avg = float(np.mean(list(wpv.values()))) if wpv else 0.28
    return {"block": block, "throw": throw, "wp_values": wpv, "wp_value_avg": avg, "cs_runs": cm.CS_RUNS,
            "use_runner_speed": cm.USE_RUNNER_SPEED,
            "trained_on": sorted(pd.to_datetime(df["game_date"]).dt.year.unique().tolist())}


def out_of_fold(df: pd.DataFrame, events: pd.DataFrame, wpv: dict | None = None) -> pd.DataFrame:
    from models import catching as cm
    fold = pd.to_numeric(df["game_pk"], errors="coerce").fillna(0).astype(int) % 2
    parts = []
    for k in (0, 1):
        b = fit(df[fold != k], events, wpv)
        parts.append(cm.daily_rows(df[fold == k], events, b))
    return pd.concat(parts, ignore_index=True)


def report(df: pd.DataFrame, events: pd.DataFrame, daily: pd.DataFrame, wpv: dict) -> str:
    from models import catching as cm
    d = daily.copy()
    tot = d[cm.CATCH_COLS].sum()
    on = df["on_1b"].notna() | df["on_2b"].notna() | df["on_3b"].notna()
    L = ["IQ catcher blocking / throwing -- test report",
         "Each half of the games scored by models trained on the OTHER half. Runs from the fielding team's side.", "",
         "BLOCKING (Statcast method: every pitch with a runner on, not in play / HBP; WP + PB chance per pitch)",
         f"  Opportunities: {int(tot.c_n):,}  |  expected WP+PB {tot.c_exp:,.0f} vs actual {tot.c_act:,.0f} (should match)",
         f"  Tiers (Statcast lines): Easy {1 - tot.g_n / tot.c_n:.1%}  Medium {tot.med_n / tot.c_n:.1%}  "
         f"Tough {tot.tough_n / tot.c_n:.1%}   (Statcast says 95%+ of chances are easy)",
         f"  Genuine opportunities (Medium + Tough): {int(tot.g_n):,} | blocked {tot.g_blk / tot.g_n:.1%} "
         f"(expected {tot.g_exp / tot.g_n:.1%})",
         f"  Tough: {int(tot.tough_n):,} | blocked {tot.tough_blk / max(tot.tough_n, 1):.1%}", "",
         "  What a WP / PB costs by situation (runners move up a base; from this era's run expectancy):",
         "    RUNNERS     0 OUT   1 OUT   2 OUT"]
    names_b = {1: "1st", 2: "2nd", 3: "1st+2nd", 4: "3rd", 5: "1st+3rd", 6: "2nd+3rd", 7: "loaded"}
    for b_, nm in names_b.items():
        L.append(f"    {nm:<10}" + "".join(f"{wpv.get((b_, o), np.nan):>8.2f}" for o in range(3)))
    L.append(f"  (Statcast uses {cm.STATCAST_BLOCK_RUNS} runs everywhere; DRS uses these)")
    L += ["", f"THROWING: {int(tot.t_n):,} steal attempts | expected caught {tot.t_exp:,.0f} vs actual {tot.t_cs:,.0f}"]
    d["year"] = pd.to_datetime(d["game_date"]).dt.year
    d["half"] = np.where(pd.to_datetime(d["game_date"]).dt.month <= 6, 1, 2)
    g = d.groupby(["year", "player", "half"])[cm.CATCH_COLS].sum().reset_index()
    a, b = g[g.half == 1], g[g.half == 2]
    m = a.merge(b, on=["year", "player"], suffixes=("_1", "_2"))
    mb = m[(m.c_n_1 >= MIN_HALF_PITCHES) & (m.c_n_2 >= MIN_HALF_PITCHES)]
    mt = m[(m.t_n_1 >= 15) & (m.t_n_2 >= 15)]
    L += ["", "Repeatable? catcher's 1st half vs 2nd half:",
          f"  blocks above avg per opportunity r = {((mb.c_exp_1 - mb.c_act_1) / mb.c_n_1).corr((mb.c_exp_2 - mb.c_act_2) / mb.c_n_2):.3f}  ({len(mb)})",
          f"  throwing runs per attempt        r = {(mt.t_runs_1 / mt.t_n_1).corr(mt.t_runs_2 / mt.t_n_2):.3f}  ({len(mt)})"]
    last = int(d["year"].max())
    s = d[d.year == last].groupby("player")[cm.CATCH_COLS].sum()
    s = s[s.c_n >= MIN_BOARD]
    s["baa"] = s.c_exp - s.c_act
    try:
        from core import mlb_api
        names = mlb_api.get_people(s.index.astype(int).tolist())
    except Exception:
        names = {}
    if len(s) >= 8:
        for title, col in (("Blocks Above Average", "baa"), ("Blocking runs (situational)", "c_runs"),
                           ("Big Stop runs", "big_runs"), ("Throwing runs", "t_runs")):
            top = s.sort_values(col, ascending=False)
            L += ["", f"{last} {title} ({MIN_BOARD}+ opportunities) -- eye test:"]
            L.append("  " + " | ".join(f"{names.get(int(i), i)} {r[col]:+.1f}" for i, r in top.head(5).iterrows())
                     + "   ...   " + " | ".join(f"{names.get(int(i), i)} {r[col]:+.1f}" for i, r in top.tail(3).iterrows()))
    return "\n".join(L)


def main(argv: list[str]) -> None:
    import joblib

    from models import catching as cm
    from pipeline.train_decisions import latest_full_season
    t0 = time.time()
    since = int(argv[argv.index("--since") + 1]) if "--since" in argv else 2015
    years = list(range(since, latest_full_season() + 1))
    eras = [("SHIFT ERA (2015-2022)", "pre", [y for y in years if y < cm.SHIFT_BAN]),
            ("2023+", "post", [y for y in years if y >= cm.SHIFT_BAN])]
    models, reps = {}, []
    for title, key, ys in eras:
        if not ys:
            continue
        df, events, wpv = load(ys)
        print(f"\n{title}: testing…", flush=True)
        daily = out_of_fold(df, events, wpv)
        print(f"{title}: training…", flush=True)
        models[key] = fit(df, events, wpv)
        reps.append(f"===== {title} =====\n" + report(df, events, daily, wpv))
        print("\n" + reps[-1] + "\n", flush=True)
        del df, events, daily
    bundle = models["post"] if list(models) == ["post"] else (
        models["pre"] if list(models) == ["pre"] else {"eras": models, "split": cm.SHIFT_BAN})
    rep = "\n\n".join(reps)
    bundle["report"] = rep
    cm.ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cm.ARTIFACT, compress=3)
    (cm.ARTIFACT.parent / "catching_report.txt").write_text(rep + "\n")
    print(f"Saved {cm.ARTIFACT.name} + catching_report.txt in {(time.time() - t0) / 60:.0f} min. "
          "Run  python -m pipeline.update  so blocking / throwing fill in on the Defense page.")


if __name__ == "__main__":
    main(sys.argv[1:])
