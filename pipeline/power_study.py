"""Power study (analysis only -- nothing on the site changes). Which bat speed / exit velocity metrics best
predict a hitter's run value, and what weight should each get in the Bat Speed (power) grade?

    caffeinate -i python -m pipeline.power_study

Uses the saved pitches in data/raw (no downloads). Every test predicts NEXT season, so exit-velocity metrics
don't win just because this season's xwOBA is built from exit velocity.

  1. Every metric, one at a time: how stable (odd vs even games), how strongly it relates to next season's
     run value -- on contact (the power target) and per plate appearance (shows the strikeout trade-off)
  2. How the metrics overlap (correlations between them)
  3. Step test -- raw bat speed first, then APPLICATION (exit velo beyond what his bat speed + the pitch speed
     should produce: squared-up %, smash factor, blasts), then the remaining exit-velocity metrics. How much
     does each step add? (and the reverse order)
  4. Weights: ridge regression, all weights positive, each metric only credited for what it adds;
     cross-validated by hitter so the fit can't memorise anyone. Compared with "bat speed weighted most".
  5. Are the weights stable from one season pair to the next?
  6. Eye test: top and bottom hitters by the composite
  7. Exit-velocity-only version for seasons / levels without bat tracking (2015+, by era)


Output: models/artifacts/power_study_report.txt (+ power_study_hitters.csv, every hitter-season and metric)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "models" / "artifacts"
REPORT = ART / "power_study_report.txt"
FIRST_BAT = 2023                 # bat tracking (mid-2023 on)
SPLIT = 2023
MIN_BIP, MIN_SWINGS = 150, 300   # hitter-seasons used for the season tests
MIN_HALF_BIP = 70
SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "missed_bunt"}
COLS = ["game_pk", "game_date", "game_type", "batter", "description", "type", "launch_speed", "launch_angle",
        "release_speed", "bat_speed", "swing_length", "woba_value", "woba_denom", "iso_value", "babip_value",
        "estimated_woba_using_speedangle", "bb_type"]

BAT = ["Avg Bat Speed", "Top 8th Bat Speed"]
APPLY = ["Squared-Up%", "Smash Factor", "Blast%", "Smash (Top 8th)"]
EV = ["Avg EV", "90th EV", "HHEV (Top 8th EV)", "Max EV (BIP-adj)", "EV>95%", "EV (Fly)", "90th EV (Fly)",
      "Max EV (Fly, BIP-adj)"]
OTHER = ["Swing Length"]
EV_ONLY_ERA = EV


def _top8(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    k = max(1, int(np.ceil(len(s) / 8)))
    return float(np.sort(s.values)[-k:].mean())


def load(year: int) -> pd.DataFrame:
    from pipeline import build_league_reference as bl
    raw = bl.pull_season(year)
    d = raw[[c for c in COLS if c in raw.columns]].copy()
    del raw
    if "game_type" in d:
        d = d[d["game_type"].astype(str).eq("R")]
    for c in ("launch_speed", "launch_angle", "release_speed", "bat_speed", "swing_length", "woba_value",
              "woba_denom", "iso_value", "estimated_woba_using_speedangle"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = np.nan
    d["year"] = year
    # pitch speed matters: a faster pitch comes off the bat harder (~0.2 mph EV per mph of pitch)
    lg_pitch = d.loc[d["type"].eq("X"), "release_speed"].mean()
    d["ev_adj"] = d["launch_speed"] - 0.2 * (d["release_speed"] - lg_pitch)
    return d


def hitter_table(d: pd.DataFrame) -> pd.DataFrame:
    """Every metric for every hitter in d."""
    bunt = d["description"].astype(str).str.contains("bunt") | d["bb_type"].astype(str).eq("bunt") \
        if "bb_type" in d else d["description"].astype(str).str.contains("bunt")
    sw = d[d["description"].isin(SWINGS) & ~bunt]
    bip = d[d["type"].eq("X") & d["launch_speed"].notna() & ~bunt]
    fly = bip[(bip["launch_speed"] > 70) & (bip["launch_angle"] >= 10) & (bip["launch_angle"] < 51)]
    out = pd.DataFrame(index=pd.Index(sorted(d["batter"].dropna().unique()), name="batter"))

    # bat speed: Statcast "competitive" swings -- drop each hitter's slowest 10% (checks / defensive swings)
    bs = sw.dropna(subset=["bat_speed"])
    if len(bs):
        cut = bs.groupby("batter")["bat_speed"].transform(lambda s: s.quantile(0.10))
        comp = bs[bs["bat_speed"] >= cut]
        g = comp.groupby("batter")
        out["swings_tracked"] = bs.groupby("batter").size()
        out["Avg Bat Speed"] = g["bat_speed"].mean()
        out["Top 8th Bat Speed"] = g["bat_speed"].agg(_top8)
        out["Swing Length"] = g["swing_length"].mean()
        # application: exit velo vs the most his bat speed + this pitch could produce (Savant's squared-up)
        b = bip.dropna(subset=["bat_speed", "release_speed"])
        b = b[b["bat_speed"] > 30]
        mx = 1.23 * b["bat_speed"] + 0.23 * b["release_speed"]
        b = b.assign(smash=(b["launch_speed"] / mx).clip(0, 1.2))
        b = b.assign(sq=(b["smash"] >= 0.8).astype(float),
                     blast=((b["smash"] * 100 + b["bat_speed"]) >= 164).astype(float))
        gb = b.groupby("batter")
        out["bip_tracked"] = gb.size()
        out["Squared-Up%"] = gb["sq"].mean()
        out["Smash Factor"] = gb["smash"].mean()
        out["Blast%"] = gb["blast"].mean()
        out["Smash (Top 8th)"] = gb["smash"].agg(_top8)
    else:
        for c in BAT + APPLY + OTHER + ["swings_tracked", "bip_tracked"]:
            out[c] = np.nan

    g = bip.groupby("batter")["ev_adj"]
    out["BIP"] = g.size()
    out["Avg EV"] = g.mean()
    out["90th EV"] = g.quantile(0.90)
    out["HHEV (Top 8th EV)"] = g.agg(_top8)
    out["EV>95%"] = bip.assign(h=(bip["ev_adj"] > 95).astype(float)).groupby("batter")["h"].mean()
    out["max_raw"] = g.max()
    gf = fly.groupby("batter")["ev_adj"]
    out["EV (Fly)"] = gf.mean()
    out["90th EV (Fly)"] = gf.quantile(0.90)
    out["max_fly_raw"], out["fly_n"] = gf.max(), gf.size()

    # outcomes
    pa = d[d["woba_denom"] > 0]
    gp = pa.groupby("batter")
    out["PA"] = gp["woba_denom"].sum()
    out["wOBA"] = gp["woba_value"].sum() / out["PA"]
    out["wOBAcon"] = bip.groupby("batter")["woba_value"].mean()
    out["ISOcon"] = bip.groupby("batter")["iso_value"].mean()
    out["xwOBAcon"] = bip.groupby("batter")["estimated_woba_using_speedangle"].mean()
    return out


def bip_adjust_max(t: pd.DataFrame) -> pd.DataFrame:
    """Max EV rises with more balls in play. Remove that: regress max on log(BIP) across hitters, keep the residual
    (+ the league average), so 80-BIP and 500-BIP hitters are on the same footing."""
    t = t.copy()
    for raw, n, name in (("max_raw", "BIP", "Max EV (BIP-adj)"), ("max_fly_raw", "fly_n", "Max EV (Fly, BIP-adj)")):
        ok = t[raw].notna() & (t[n] >= 20)
        if ok.sum() < 30:
            t[name] = np.nan
            continue
        x = np.log(t.loc[ok, n])
        b = np.polyfit(x, t.loc[ok, raw], 1)
        t[name] = t[raw] - b[0] * (np.log(t[n].clip(lower=1)) - x.mean())
        t.loc[~ok, name] = np.nan
    return t


# ----------------------------------------------------------------------------------------------- tests
def zstd(x: pd.DataFrame) -> pd.DataFrame:
    return (x - x.mean()) / x.std()


def cv_r2(X: pd.DataFrame, y: pd.Series, groups: pd.Series, positive=True, alpha=1.0, folds=5) -> float:
    """Cross-validated R^2 (folds by hitter) of a ridge fit."""
    from sklearn.linear_model import Ridge
    ok = X.notna().all(axis=1) & y.notna()
    X, y, groups = X[ok], y[ok], groups[ok]
    if len(y) < 50:
        return np.nan
    uniq = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    fold_of = {g: i % folds for i, g in enumerate(uniq)}
    f = groups.map(fold_of)
    pred = pd.Series(np.nan, index=y.index)
    for k in range(folds):
        tr, te = f != k, f == k
        mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
        m = Ridge(alpha=alpha, positive=positive).fit((X[tr] - mu) / sd, y[tr])
        pred[te] = m.predict((X[te] - mu) / sd)
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def weights(X: pd.DataFrame, y: pd.Series, alpha=1.0) -> pd.Series:
    from sklearn.linear_model import Ridge
    ok = X.notna().all(axis=1) & y.notna()
    m = Ridge(alpha=alpha, positive=True).fit(zstd(X[ok]), y[ok])
    w = pd.Series(m.coef_, index=X.columns)
    return w / w.sum() if w.sum() > 0 else w


def pairs(tabs: dict, years: list[int]) -> pd.DataFrame:
    """This season's metrics joined to NEXT season's outcomes (same hitter)."""
    rows = []
    for y in years:
        if y + 1 not in tabs:
            continue
        a, b = tabs[y]["all"], tabs[y + 1]["all"]
        a = a[(a["BIP"] >= MIN_BIP)]
        b = b[(b["BIP"] >= MIN_BIP)][["wOBA", "wOBAcon", "ISOcon", "xwOBAcon", "PA", "BIP"]].add_prefix("next_")
        rows.append(a.join(b, how="inner").assign(year=y))
    return pd.concat(rows).reset_index() if rows else pd.DataFrame()


def reliability(tabs: dict, years: list[int], metrics: list[str]) -> pd.DataFrame:
    out = []
    for m in metrics:
        rs, ns = [], []
        for y in years:
            o, e = tabs[y]["odd"], tabs[y]["even"]
            idx = o.index[o["BIP"] >= MIN_HALF_BIP].intersection(e.index[e["BIP"] >= MIN_HALF_BIP])
            if m in o and len(idx) >= 30 and o.loc[idx, m].notna().sum() >= 30:
                rs.append(o.loc[idx, m].corr(e.loc[idx, m]))
                ns.append(float(((o.loc[idx, "BIP"] + e.loc[idx, "BIP"]) / 2).mean()))
        r = float(np.nanmean(rs)) if rs else np.nan
        n_half = float(np.nanmean(ns)) if ns else np.nan
        need = n_half * (0.7 / 0.3) * (1 - r) / r if r == r and r > 0 else np.nan
        out.append({"metric": m, "split_r": r, "bip_for_70": need})
    return pd.DataFrame(out).set_index("metric")


def main(argv: list[str]) -> None:
    from pipeline.train_decisions import latest_full_season
    t0 = time.time()
    last = latest_full_season()
    years = list(range(2015, last + 1))
    tabs = {}
    for y in years:
        d = load(y)
        has_bat = d["bat_speed"].notna().mean() > 0.05
        print(f"[{y}] {len(d):,} pitches, bat tracking {'yes' if has_bat else 'no'} -- building hitter metrics…",
              flush=True)
        tabs[y] = {k: bip_adjust_max(hitter_table(v)) for k, v in
                   {"all": d, "odd": d[d.game_pk % 2 == 1], "even": d[d.game_pk % 2 == 0]}.items()}
        del d
        print(f"[{y}] done ({(time.time() - t0) / 60:.0f} min)", flush=True)

    bat_years = [y for y in years if y >= FIRST_BAT and tabs[y]["all"]["Top 8th Bat Speed"].notna().sum() > 100]
    allm = BAT + APPLY + EV + OTHER
    L = ["IQ POWER STUDY -- which bat speed / exit velocity metrics predict run value, and how much weight each gets",
         f"Regular season. Hitter-seasons with {MIN_BIP}+ balls in play (both seasons for next-season tests). "
         "Exit velocities adjusted for pitch speed; max EVs adjusted for number of balls in play.",
         f"Bat tracking seasons: {', '.join(map(str, bat_years)) or 'none found'}", "",
         "HOW TO READ",
         "  SPLIT r        odd vs even games, same season: is it a real, stable skill? (higher = better)",
         "  BIP FOR .70    balls in play before the number is 70% skill",
         "  NEXT CON r     relation to NEXT season's run value on contact (wOBA on contact) -- the power target",
         "  NEXT ISO r     relation to next season's ISO on contact (extra-base damage)",
         "  NEXT PA r      relation to next season's wOBA per plate appearance (includes strikeouts / walks)",
         "  SAME xCON r    same season vs xwOBA on contact -- shown only to prove the trap: EV metrics look great",
         "                 here because xwOBA is built from EV. Don't use this column to pick weights.",
         "  WEIGHT         share of the grade from ridge regression (each metric credited only for what it ADDS)"]

    P = pairs(tabs, bat_years)
    if len(P):
        rel = reliability(tabs, bat_years, allm)
        L += ["", "=" * 112, f"1. EVERY METRIC ON ITS OWN -- bat tracking seasons ({len(P)} hitter season-pairs: "
              + ", ".join(f"{y}->{y + 1}" for y in sorted(P['year'].unique())) + ")",
              f"  {'METRIC':<24}{'GROUP':<13}{'SPLIT r':>8}{'BIP FOR .70':>12}{'NEXT CON r':>11}{'NEXT ISO r':>11}"
              f"{'NEXT PA r':>10}{'SAME xCON r':>12}"]
        same = pd.concat([tabs[y]["all"][tabs[y]["all"]["BIP"] >= MIN_BIP] for y in bat_years])
        grp = {**{m: "bat speed" for m in BAT}, **{m: "application" for m in APPLY}, **{m: "exit velo" for m in EV},
               **{m: "other" for m in OTHER}}
        rows = []
        for m in allm:
            rows.append((m, grp[m], rel.loc[m, "split_r"], rel.loc[m, "bip_for_70"], P[m].corr(P["next_wOBAcon"]),
                         P[m].corr(P["next_ISOcon"]), P[m].corr(P["next_wOBA"]), same[m].corr(same["xwOBAcon"])))
        for m, g_, sr, need, c, i_, pa_, s in sorted(rows, key=lambda r: -(r[4] if r[4] == r[4] else -9)):
            nd = f"{need:,.0f}" if need == need else "-"
            L.append(f"  {m:<24}{g_:<13}{sr:>8.2f}{nd:>12}{c:>11.2f}{i_:>11.2f}{pa_:>10.2f}{s:>12.2f}")

        key = ["Avg Bat Speed", "Top 8th Bat Speed", "Squared-Up%", "Smash Factor", "Blast%", "Avg EV", "90th EV",
               "HHEV (Top 8th EV)", "EV>95%", "EV (Fly)", "90th EV (Fly)"]
        C = P[key].corr()
        short = ["AvgBS", "T8BS", "SQ%", "Smash", "Blast", "AvgEV", "90EV", "HHEV", "EV95", "EVfly", "90fly"]
        L += ["", "2. HOW THE METRICS OVERLAP (correlation between them; 1.00 = the same information)",
              "  " + " " * 7 + "".join(f"{s:>7}" for s in short)]
        for s, m in zip(short, key):
            L.append(f"  {s:<7}" + "".join(f"{C.loc[m, k]:>7.2f}" for k in key))

        y_ = P["next_wOBAcon"]
        grp_id = P["batter"]
        steps = [("bat speed only", BAT), ("+ application", BAT + APPLY), ("+ exit velo metrics", BAT + APPLY + EV)]
        rev = [("exit velo only", EV), ("+ bat speed", EV + BAT), ("+ application", EV + BAT + APPLY)]
        L += ["", "3. STEP TEST -- how much of next season's run value on contact each layer explains "
                  "(cross-validated R^2, folds by hitter)"]
        for title, seq in (("Bat speed first (your order)", steps), ("Exit velo first (reverse order)", rev)):
            L.append(f"  {title}:")
            prev = 0.0
            for name, cols in seq:
                r2 = cv_r2(P[cols], y_, grp_id)
                L.append(f"    {name:<24} R^2 = {r2:.3f}   (adds {r2 - prev:+.3f})")
                prev = r2
        L.append("  Same steps for next season's ISO on contact and wOBA per PA:")
        for tgt in ("next_ISOcon", "next_wOBA"):
            vals = [cv_r2(P[cols], P[tgt], grp_id) for _, cols in steps]
            L.append(f"    {tgt[5:]:<10} bat speed {vals[0]:.3f} | + application {vals[1]:.3f} | + exit velo {vals[2]:.3f}")

        L += ["", "4. WEIGHTS (ridge, all positive, each metric credited only for what it adds)"]
        feats = BAT + APPLY + EV
        res = {}
        for tgt, lab in (("next_wOBAcon", "NEXT CON"), ("next_ISOcon", "NEXT ISO"), ("next_wOBA", "NEXT PA")):
            res[lab] = weights(P[feats], P[tgt])
        W = pd.DataFrame(res)
        L.append(f"  {'METRIC':<24}{'GROUP':<13}" + "".join(f"{c:>10}" for c in W.columns))
        for m in W.sort_values("NEXT CON", ascending=False).index:
            L.append(f"  {m:<24}{grp[m]:<13}" + "".join(f"{W.loc[m, c]:>10.0%}" for c in W.columns))
        tot = W.groupby(pd.Series(grp)).sum().reindex(["bat speed", "application", "exit velo"])
        L.append("  " + "-" * 60)
        for gname, r in tot.iterrows():
            L.append(f"  {('TOTAL ' + gname):<37}" + "".join(f"{v:>10.0%}" for v in r.values))
        data_r2 = cv_r2(P[feats], y_, grp_id)
        z = zstd(P[feats])
        prior = (0.30 * z["Avg Bat Speed"] + 0.30 * z["Top 8th Bat Speed"] + 0.40 * z[APPLY + EV].mean(axis=1))
        L += ["", f"  Composite with the data's weights:          cross-validated R^2 = {data_r2:.3f}  "
                  f"(r = {np.sqrt(max(data_r2, 0)):.2f})",
              f"  Your prior (bat speed 60%, the rest 40%):  r = {prior.corr(y_):.2f} with next season's run value "
              "on contact"]

        two = BAT + APPLY
        w2 = weights(P[two], y_)
        r2_two = cv_r2(P[two], y_, grp_id)
        L += ["", "  4b. TWO-PART VERSION -- bat speed + application only (exit velo IS bat speed x application, so this",
              "      asks whether the clean two-part grade loses anything vs using every metric):",
              f"      cross-validated R^2 = {r2_two:.3f} vs {data_r2:.3f} with every metric   weights: "
              + ", ".join(f"{k} {v:.0%}" for k, v in w2.sort_values(ascending=False).items())]

        L += ["", "5. ARE THE WEIGHTS STABLE? (fitted on each season pair alone, NEXT CON target; by group)"]
        for y in sorted(P["year"].unique()):
            sub = P[P["year"] == y]
            if len(sub) < 80:
                continue
            w = weights(sub[feats], sub["next_wOBAcon"])
            tg = w.groupby(pd.Series(grp)).sum()
            L.append(f"    {y}->{y + 1} ({len(sub)} hitters): bat speed {tg.get('bat speed', 0):.0%} | application "
                     f"{tg.get('application', 0):.0%} | exit velo {tg.get('exit velo', 0):.0%}   top: "
                     + ", ".join(f"{k} {v:.0%}" for k, v in w.sort_values(ascending=False).head(3).items()))

        last_y = bat_years[-1]
        s = tabs[last_y]["all"]
        s = s[s["BIP"] >= MIN_BIP].dropna(subset=feats)
        if len(s):
            comp = (zstd(s[feats]) * W["NEXT CON"]).sum(axis=1)
            s = s.assign(grade=50 + 10 * comp / comp.std())
            try:
                from core import mlb_api
                names = mlb_api.get_people(s.index.astype(int).tolist())
            except Exception:
                names = {}
            top = s.sort_values("grade", ascending=False)
            L += ["", f"6. {last_y} EYE TEST -- composite power grade with the data's weights (20-80, {MIN_BIP}+ BIP)"]
            fmt = lambda r: f"{names.get(int(r.name), r.name)} {r['grade']:.0f}"
            L.append("  top:    " + " | ".join(fmt(r) for _, r in top.head(10).iterrows()))
            L.append("  bottom: " + " | ".join(fmt(r) for _, r in top.tail(6).iterrows()))
    else:
        L += ["", "No bat tracking seasons with enough hitters were found -- bat speed sections skipped."]

    L += ["", "=" * 112, "7. EXIT-VELO-ONLY VERSION (seasons / levels without bat tracking), by era, NEXT CON target"]
    for title, ys in ((f"2015-{SPLIT - 1}", [y for y in years if y < SPLIT]), (f"{SPLIT}+", [y for y in years if y >= SPLIT])):
        Pe = pairs(tabs, [y for y in ys if y + 1 in ys])
        if len(Pe) < 100:
            continue
        w = weights(Pe[EV], Pe["next_wOBAcon"])
        r2 = cv_r2(Pe[EV], Pe["next_wOBAcon"], Pe["batter"])
        L.append(f"  {title} ({len(Pe)} pairs, cross-validated R^2 = {r2:.3f}):")
        for m in w.sort_values(ascending=False).index:
            L.append(f"    {m:<24} on its own r = {Pe[m].corr(Pe['next_wOBAcon']):.2f}   weight {w[m]:>4.0%}")

    out = "\n".join(L)
    ART.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(out + "\n")
    pd.concat([tabs[y]["all"].assign(year=y) for y in years]).to_csv(ART / "power_study_hitters.csv")
    print("\n" + out)
    print(f"\nSaved {REPORT} + power_study_hitters.csv in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main(sys.argv[1:])
