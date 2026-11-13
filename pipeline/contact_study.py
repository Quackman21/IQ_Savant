"""Bat-to-Ball study (analysis only -- nothing on the site changes). Which contact metrics best measure a hitter's
bat-to-ball skill, and what weight should each get in the Bat-To-Ball Grade?

    caffeinate -i python -m pipeline.contact_study          (Mac)   |   python -m pipeline.contact_study   (PC)

Uses the saved pitches in data/raw (no downloads). Every test predicts NEXT season.

Metrics (the app's exact definitions):  Cont%, Z-Cont%, Z-Cont% (Hard), Z-Cont% (Soft), H-Cont%, S-Cont%, C-Cont%,
W-Cont%  -- plus candidates: Cont% over expected and Z-Cont% over expected (contact above what a league model
expects from location, pitch type, velo, movement, count, hands -- separates the bat from the swing decisions and
the pitch mix), and Squared-Up% (2023+, contact quality).

Targets (next season): strikeouts avoided (-K%), wOBA per PA at the SAME power (partial r, power held equal --
contact hitters tend to have less power), and whiff rate avoided (-Whiff%).

Output: models/artifacts/contact_study_report.txt (+ contact_study_hitters.csv)
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
REPORT = ART / "contact_study_report.txt"
SPLIT = 2023
MIN_SWINGS, MIN_HALF = 250, 120
APP = ["Cont%", "Z-Cont%", "Z-Cont% (Hard)", "Z-Cont% (Soft)", "H-Cont%", "S-Cont%", "C-Cont%", "W-Cont%"]
XC = ["Cont% over exp", "Z-Cont% over exp"]
SQ = ["Squared-Up%"]
COLS = ["game_pk", "game_date", "game_type", "batter", "description", "events", "type", "zone", "plate_x", "plate_z",
        "sz_top", "sz_bot", "pitch_type", "release_speed", "pfx_x", "pfx_z", "balls", "strikes", "stand", "p_throws",
        "launch_speed", "bat_speed", "woba_value", "woba_denom", "estimated_woba_using_speedangle", "launch_angle"]
FEAT = ["nx", "nz", "velo", "hb", "ivb", "balls", "strikes", "same", "ptype"]


def load(year: int) -> pd.DataFrame:
    from core import hit_metrics as hm
    from pipeline import build_league_reference as bl
    raw = bl.pull_season(year)
    d = raw[[c for c in COLS if c in raw.columns]].copy()
    del raw
    if "game_type" in d:
        d = d[d["game_type"].astype(str).eq("R")]
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
    desc = d["description"].astype(str)
    d["swing"] = desc.isin(hm.SWINGS)
    d["whiff"] = desc.isin(hm.WHIFFS)
    d["contact"] = d["swing"] & ~d["whiff"]
    zone = num("zone")
    d["in_z"] = zone.between(1, 9)
    d["hard"] = d["pitch_type"].isin(hm.FASTBALLS)
    d["att"] = hm.attack_zone(d)
    top, bot = num("sz_top"), num("sz_bot")
    d["nx"] = num("plate_x") / (17 / 24)
    d["nz"] = (num("plate_z") - (top + bot) / 2) / ((top - bot) / 2)
    lefty_p = d["p_throws"].astype(str).eq("L")
    d["velo"], d["hb"], d["ivb"] = num("release_speed"), np.where(lefty_p, -1, 1) * num("pfx_x"), num("pfx_z")
    d["nx"] = np.where(d["stand"].astype(str).eq("L"), -d["nx"], d["nx"])       # inside / away, not left / right
    d["balls"], d["strikes"] = num("balls"), num("strikes")
    d["same"] = (d["stand"].astype(str) == d["p_throws"].astype(str)).astype(int)
    d["ptype"] = d["pitch_type"].map({p: i for i, p in enumerate(sorted(d["pitch_type"].dropna().unique()))})
    d["year"] = year
    return d


def expected_contact(d: pd.DataFrame) -> pd.Series:
    """League contact chance for every swing; each half of the games scored by a model trained on the other half."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    sw = d[d["swing"]].dropna(subset=["nx", "nz", "velo"])
    out = pd.Series(np.nan, index=d.index)
    fold = sw["game_pk"].astype(int) % 2
    for k in (0, 1):
        tr, te = sw[fold != k], sw[fold == k]
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
                                           categorical_features=[FEAT.index("ptype")], random_state=0)
        m.fit(tr[FEAT].fillna(-1), tr["contact"].astype(int))
        out[te.index] = m.predict_proba(te[FEAT].fillna(-1))[:, 1]
    return out


def hitter_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("batter")
    sw = d[d["swing"]]
    out = pd.DataFrame({"swings": sw.groupby("batter").size()})
    rate = lambda m: d[d["swing"] & m].groupby("batter")["contact"].mean()
    out["Cont%"] = rate(pd.Series(True, index=d.index))
    out["Z-Cont%"] = rate(d["in_z"])
    out["Z-Cont% (Hard)"] = rate(d["in_z"] & d["hard"])
    out["Z-Cont% (Soft)"] = rate(d["in_z"] & ~d["hard"])
    for a, name in (("Heart", "H"), ("Shadow", "S"), ("Chase", "C"), ("Waste", "W")):
        out[f"{name}-Cont%"] = rate(d["att"].eq(a))
    xs = sw.assign(over=sw["contact"].astype(float) - sw["xcont"])
    out["Cont% over exp"] = xs.groupby("batter")["over"].mean()
    out["Z-Cont% over exp"] = xs[xs["in_z"]].groupby("batter")["over"].mean()
    b = d[d["type"].eq("X") & d["bat_speed"].notna() & d["launch_speed"].notna()] if "bat_speed" in d else d.iloc[:0]
    if len(b):
        mx = 1.23 * pd.to_numeric(b["bat_speed"]) + 0.23 * pd.to_numeric(b["release_speed"])
        out["Squared-Up%"] = (pd.to_numeric(b["launch_speed"]) / mx >= 0.8).groupby(b["batter"]).mean()
    else:
        out["Squared-Up%"] = np.nan
    pa = d[d["events"].notna() & d["events"].astype(str).ne("")]
    gp = pa.groupby("batter")
    out["PA"] = gp.size()
    out["K%"] = gp["events"].apply(lambda e: e.astype(str).str.startswith("strikeout").mean())
    out["Whiff%"] = sw.groupby("batter")["whiff"].mean()
    wd = d[pd.to_numeric(d["woba_denom"], errors="coerce") > 0]
    out["wOBA"] = wd.groupby("batter")["woba_value"].sum() / wd.groupby("batter")["woba_denom"].sum()
    bip = d[d["type"].eq("X")].assign(ev=lambda x: pd.to_numeric(x["launch_speed"], errors="coerce"))
    out["power"] = bip.groupby("batter")["ev"].quantile(0.9)              # power held equal in the wOBA test
    fly = bip[(bip["ev"] > 70) & (pd.to_numeric(bip["launch_angle"], errors="coerce").between(10, 50))]
    out["power_fly"] = fly.groupby("batter")["ev"].mean()
    return out


def partial_r(x: pd.Series, y: pd.Series, z: pd.DataFrame) -> float:
    """Correlation of x and y once z (power) is held equal."""
    ok = x.notna() & y.notna() & z.notna().all(axis=1)
    if ok.sum() < 30:
        return np.nan
    Z = np.column_stack([np.ones(ok.sum()), z[ok].values])
    rx = x[ok] - Z @ np.linalg.lstsq(Z, x[ok], rcond=None)[0]
    ry = y[ok] - Z @ np.linalg.lstsq(Z, y[ok], rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main(argv: list[str]) -> None:
    from pipeline.power_study import cv_r2, weights, zstd
    from pipeline.train_decisions import latest_full_season
    t0 = time.time()
    years = list(range(2015, latest_full_season() + 1))
    tabs = {}
    for y in years:
        d = load(y)
        print(f"[{y}] {len(d):,} pitches -- league contact model + hitter metrics…", flush=True)
        d["xcont"] = expected_contact(d)
        tabs[y] = {k: hitter_table(v) for k, v in
                   {"all": d, "odd": d[d.game_pk % 2 == 1], "even": d[d.game_pk % 2 == 0]}.items()}
        del d
        print(f"[{y}] done ({(time.time() - t0) / 60:.0f} min)", flush=True)

    L = ["IQ BAT-TO-BALL STUDY -- which contact metrics measure bat-to-ball skill, and how much weight each gets",
         f"Regular season. Hitter-seasons with {MIN_SWINGS}+ swings (both seasons for next-season tests).", "",
         "HOW TO READ",
         "  SPLIT r        odd vs even games, same season: is it a real, stable skill? (higher = better)",
         "  SWINGS FOR .70 swings before the number is 70% skill",
         "  NEXT -K% r     relation to NEXT season's strikeouts avoided (the main target: contact skill keeps you",
         "                 out of strikeouts)",
         "  NEXT -WHIFF r  relation to next season's whiffs avoided",
         "  NEXT wOBA r    relation to next season's wOBA per PA (raw -- contact hitters often have less power)",
         "  wOBA|POWER r   same, with POWER HELD EQUAL (90th EV + fly-ball EV): between two hitters with the same",
         "                 power, does better contact mean more runs?",
         "  WEIGHT         share of the grade from ridge regression (each metric credited only for what it ADDS)"]
    metrics_all = APP + XC + SQ
    for title, ys in ((f"2015-{SPLIT - 1}", [y for y in years if y < SPLIT]), (f"{SPLIT}+", [y for y in years if y >= SPLIT])):
        rows = []
        for y in ys:
            if y + 1 not in ys:
                continue
            a, b = tabs[y]["all"], tabs[y + 1]["all"]
            a = a[a["swings"] >= MIN_SWINGS]
            b = b[b["swings"] >= MIN_SWINGS][["K%", "Whiff%", "wOBA", "PA"]].add_prefix("next_")
            rows.append(a.join(b, how="inner").assign(year=y))
        if not rows:
            continue
        P = pd.concat(rows).reset_index()
        P["-K"], P["-Wh"] = -P["next_K%"], -P["next_Whiff%"]
        metrics = [m for m in metrics_all if P[m].notna().sum() >= 100]
        L += ["", "=" * 116, f"{title}  ({len(P)} hitter season-pairs)", "",
              "1. EVERY METRIC ON ITS OWN",
              f"  {'METRIC':<20}{'SPLIT r':>8}{'SWINGS FOR .70':>15}{'NEXT -K% r':>11}{'NEXT -WHIFF r':>14}"
              f"{'NEXT wOBA r':>12}{'wOBA|POWER r':>13}"]
        res = []
        for m in metrics:
            rs, ns = [], []
            for y in ys:
                o, e = tabs[y]["odd"], tabs[y]["even"]
                idx = o.index[o["swings"] >= MIN_HALF].intersection(e.index[e["swings"] >= MIN_HALF])
                if len(idx) >= 30 and o.loc[idx, m].notna().sum() >= 30:
                    rs.append(o.loc[idx, m].corr(e.loc[idx, m]))
                    ns.append(float(((o.loc[idx, "swings"] + e.loc[idx, "swings"]) / 2).mean()))
            r = float(np.nanmean(rs)) if rs else np.nan
            need = float(np.nanmean(ns)) * (0.7 / 0.3) * (1 - r) / r if r == r and r > 0 else np.nan
            res.append((m, r, need, P[m].corr(P["-K"]), P[m].corr(P["-Wh"]), P[m].corr(P["next_wOBA"]),
                        partial_r(P[m], P["next_wOBA"], P[["power", "power_fly"]])))
        for m, r, need, k, w, wo, pw in sorted(res, key=lambda x: -(x[3] if x[3] == x[3] else -9)):
            nd = f"{need:,.0f}" if need == need else "-"
            L.append(f"  {m:<20}{r:>8.2f}{nd:>15}{k:>11.2f}{w:>14.2f}{wo:>12.2f}{pw:>13.2f}")

        key = [m for m in metrics]
        C = P[key].corr()
        short = {"Cont%": "Cont", "Z-Cont%": "ZCont", "Z-Cont% (Hard)": "ZHard", "Z-Cont% (Soft)": "ZSoft",
                 "H-Cont%": "Heart", "S-Cont%": "Shadw", "C-Cont%": "Chase", "W-Cont%": "Waste",
                 "Cont% over exp": "xCont", "Z-Cont% over exp": "xZCon", "Squared-Up%": "SqUp"}
        L += ["", "2. HOW THE METRICS OVERLAP (correlation between them; 1.00 = the same information)",
              "  " + " " * 7 + "".join(f"{short[m]:>7}" for m in key)]
        for m in key:
            L.append(f"  {short[m]:<7}" + "".join(f"{C.loc[m, k]:>7.2f}" for k in key))

        grp = P["batter"]
        app = [m for m in APP if m in metrics]
        steps = [("Z-Cont% only", ["Z-Cont%"]),
                 ("+ Heart / Shadow / Chase / Waste", ["Z-Cont%", "H-Cont%", "S-Cont%", "C-Cont%", "W-Cont%"]),
                 ("+ Hard / Soft in zone + Cont%", app),
                 ("+ over-expected versions", app + [m for m in XC if m in metrics])]
        if "Squared-Up%" in metrics:
            steps.append(("+ Squared-Up%", app + [m for m in XC + SQ if m in metrics]))
        L += ["", "3. STEP TEST -- how much of next season's strikeouts avoided each layer explains (cross-validated R^2)"]
        prev = 0.0
        for name, cols in steps:
            r2 = cv_r2(P[cols], P["-K"], grp)
            L.append(f"    {name:<36} R^2 = {r2:.3f}   (adds {r2 - prev:+.3f})")
            prev = r2
        L.append(f"    (your 8 app metrics alone: R^2 = {cv_r2(P[app], P['-K'], grp):.3f}; the 2 over-expected alone: "
                 f"{cv_r2(P[[m for m in XC if m in metrics]], P['-K'], grp):.3f})")

        L += ["", "4. WEIGHTS (ridge, all positive, each metric credited only for what it adds)"]
        Wk = weights(P[metrics], P["-K"])
        Ww = weights(P[metrics], P["-Wh"])
        # wOBA with power held equal: residualize the target on power first
        ok = P[["power", "power_fly", "next_wOBA"]].notna().all(axis=1)
        Z = np.column_stack([np.ones(ok.sum()), P.loc[ok, ["power", "power_fly"]].values])
        resid = pd.Series(np.nan, index=P.index)
        resid[ok] = P.loc[ok, "next_wOBA"] - Z @ np.linalg.lstsq(Z, P.loc[ok, "next_wOBA"], rcond=None)[0]
        Wo = weights(P[metrics], resid)
        W = pd.DataFrame({"-K%": Wk, "-WHIFF": Ww, "wOBA|POWER": Wo})
        L.append(f"  {'METRIC':<20}" + "".join(f"{c:>12}" for c in W.columns))
        for m in W.sort_values("-K%", ascending=False).index:
            L.append(f"  {m:<20}" + "".join(f"{W.loc[m, c]:>12.0%}" for c in W.columns))
        L.append(f"  Composite with these weights (-K% target): cross-validated R^2 = {cv_r2(P[metrics], P['-K'], grp):.3f}")
        z = zstd(P[metrics])
        zc = z["Z-Cont%"]
        L.append(f"  Z-Cont% alone vs next season -K%: r = {zc.corr(P['-K']):.2f}   |   data composite: r = "
                 f"{(z * Wk).sum(axis=1).corr(P['-K']):.2f}")

        L += ["", "5. ARE THE WEIGHTS STABLE? (each season pair alone, -K% target, top 4)"]
        for y in sorted(P["year"].unique()):
            sub = P[P["year"] == y]
            if len(sub) < 80:
                continue
            w = weights(sub[metrics], sub["-K"])
            L.append(f"    {y}->{y + 1} ({len(sub)}): " + ", ".join(f"{k} {v:.0%}" for k, v in
                                                                  w.sort_values(ascending=False).head(4).items()))

        last = ys[-1]
        s = tabs[last]["all"]
        s = s[s["swings"] >= MIN_SWINGS].dropna(subset=metrics)
        if len(s):
            ref = pd.concat([tabs[y]["all"][tabs[y]["all"]["swings"] >= MIN_SWINGS] for y in ys])
            comp = sum(Wk[m] * (s[m] - ref[m].mean()) / ref[m].std() for m in metrics)
            s = s.assign(grade=50 + 10 * comp / comp.std())
            try:
                from core import mlb_api
                names = mlb_api.get_people(s.index.astype(int).tolist())
            except Exception:
                names = {}
            top = s.sort_values("grade", ascending=False)
            f = lambda r: f"{names.get(int(r.name), r.name)} {r['grade']:.0f}"
            L += ["", f"6. {last} EYE TEST -- composite with the -K% weights ({MIN_SWINGS}+ swings)",
                  "  top:    " + " | ".join(f(r) for _, r in top.head(10).iterrows()),
                  "  bottom: " + " | ".join(f(r) for _, r in top.tail(8).iterrows())]

    out = "\n".join(L)
    ART.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(out + "\n")
    pd.concat([tabs[y]["all"].assign(year=y) for y in years]).to_csv(ART / "contact_study_hitters.csv")
    print("\n" + out)
    print(f"\nSaved {REPORT} + contact_study_hitters.csv in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main(sys.argv[1:])
