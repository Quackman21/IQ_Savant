"""Ball Flight study (analysis only -- nothing on the site changes). How much of a hitter's run value comes from his
BALL FLIGHT (launch angle + direction) once bat speed / exit velocity is taken out -- and how to grade it.

    python -m pipeline.flight_study            (PC)   |   caffeinate -i python -m pipeline.flight_study   (Mac)
    python -m pipeline.flight_study --no-push   # don't push the results to GitHub when it finishes

FLIGHT PERCENTILE (the core candidate): for every ball in play, how its angle + direction ranked (by SAxwOBA) among
ALL balls hit exactly that hard that season. 0 = average. Power can't magnify it -- a 90-mph hitter with great flight
scores the same as a 105-mph hitter with the same habits. FLIGHT RUNS: the same idea in runs (power does magnify it).
Leak checks against the other grades (power / bat speed, contact, chase) and every test with those tools held equal,
so a metric only gets credit for what the other three grades can't explain. Angle-only versions of the app's metrics
(no exit-velo cut-offs) and Sweet Spot% (8-32 deg) are tested too.

  1. WHERE FLIGHT RUNS COME FROM -- by batted-ball type x direction (pulled fly balls, oppo grounders ...): typical
     value, how much hitters really differ, share of the REAL differences.
  2. EVERY METRIC ON ITS OWN -- flight runs (expected + actual) and the app's 11: GB%, FB%, WEAK FB%, LD%, LD+FB%,
     PULL%, MIDDLE%, OPPO%, AIRPULL%, FB-PULL%, PULL-AIR% (+ LA on hard-hit balls): stability, balls in play to
     trust, and NEXT season's flight value (actual outcomes, exit velo held equal) and runs with tools held equal.
  3. Do the 11 add anything on top of flight runs? Step test + the data's weights.
  4. Eye test.
Output: models/artifacts/flight_study_report.txt (+ flight_study_hitters.csv), pushed to GitHub when done.
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
REPORT = ART / "flight_study_report.txt"
CSV = ART / "flight_study_hitters.csv"
SPLIT = 2023
MIN_BIP, MIN_HALF = 200, 90
WOBA_SCALE = 1.2
PER_SEASON = 420                                   # balls in play for a typical regular
APP = ["GB%", "FB%", "WEAK FB%", "LD%", "LD+FB%", "PULL%", "MIDDLE%", "OPPO%", "AIRPULL%", "FB-PULL%", "PULL-AIR%"]
EXTRA = ["LA (hard-hit)"]
CORE = ["Flight percentile", "Flight runs/BIP (x)"]
ANGLE = ["GB% (angle only)", "FB% (angle only)", "Pop-up% (angle only)", "Sweet Spot%", "AIRPULL% (angle only)",
         "PULL-AIR% (angle only)", "LA sd (consistency)"]
TOOLS = ["power", "avg_ev", "contact", "chase"]
TYPES = ["Ground ball", "Line drive", "Fly ball", "Weak air / pop"]
DIRS = ["Pull", "Middle", "Oppo"]


def load(year: int) -> pd.DataFrame:
    from core import hit_metrics as hm
    from models import saxwoba
    from pipeline import build_league_reference as bl
    raw = bl.pull_season(year)
    if "game_type" in raw:
        raw = raw[raw["game_type"].astype(str).eq("R")]
    raw = raw.reset_index(drop=True)
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")
    ev, la = num("launch_speed"), num("launch_angle")
    desc = raw["description"].astype(str)
    bunt = desc.str.contains("bunt") | raw["events"].astype(str).str.contains("bunt") | \
        raw.get("des", pd.Series("", index=raw.index)).astype(str).str.contains("bunt")
    bip = raw["type"].eq("X") & ev.notna() & la.notna() & ~bunt
    b = raw[bip].copy()
    spray = hm.spray_angle(b)
    rhb = b["stand"].astype(str).eq("R")
    sx = saxwoba.expected_woba(b)
    d = pd.DataFrame({"game_pk": pd.to_numeric(b["game_pk"]), "batter": b["batter"], "year": year,
                      "ev": ev[bip], "la": la[bip], "spray": spray, "sx": sx,
                      "wv": pd.to_numeric(b["woba_value"], errors="coerce")}, index=b.index)
    d["ld"] = (d.la >= 10) & (d.la <= 30)
    d["gb"] = (d.ev > 50) & (d.la >= -30) & (d.la < 10)
    d["fb"] = (d.ev > 70) & (d.la >= 10) & (d.la < 51)
    d["weak_fb"] = ((d.la >= 51) & (d.la < 90)) | ((d.ev <= 70) & (d.la >= 10) & (d.la < 51))
    d["spr"] = d.spray.notna()
    d["pull"] = d.spr & np.where(rhb, d.spray <= -15, d.spray >= 15)
    d["oppo"] = d.spr & np.where(rhb, d.spray >= 15, d.spray <= -15)
    d["mid"] = d.spr & (d.spray > -15) & (d.spray < 15)
    d["airpull"] = d.spr & (d.ev > 80) & (d.la >= 10) & (d.la <= 45) & np.where(rhb, d.spray <= -10, d.spray >= 10)
    d["type"] = np.select([d.la < 10, d.la <= 30, (d.la < 51) & (d.ev > 70)],
                          ["Ground ball", "Line drive", "Fly ball"], "Weak air / pop")
    d["dir"] = np.select([d["pull"], d["oppo"], d["mid"]], DIRS, "Middle")
    # flight value: SAxwOBA minus an average ball hit exactly that hard (1-mph bins, this season)
    evb = d.ev.clip(40, 120).round()
    for col, base in (("sx", "flight_x"), ("wv", "flight_a")):
        avg = d.groupby(evb)[col].transform("mean")
        d[base] = (d[col] - avg) / WOBA_SCALE                     # runs per ball in play
    # FLIGHT PERCENTILE: how this ball's angle + direction ranked among all balls hit that hard (0 = average).
    # Power can't magnify it: a 90-mph hitter with great flight scores the same as a 105-mph one.
    d["flight_pct"] = d.groupby(evb)["sx"].rank(pct=True) - 0.5
    # angle / direction only (no exit-velo cut-offs)
    d["a_gb"] = d.la < 10
    d["a_fb"] = (d.la >= 10) & (d.la < 50)
    d["a_pop"] = d.la >= 50
    d["sweet"] = (d.la >= 8) & (d.la <= 32)
    d["a_airpull"] = d.spr & (d.la >= 10) & (d.la <= 45) & np.where(rhb, d.spray <= -10, d.spray >= 10)
    return d


def hitter_table(d: pd.DataFrame, tools: pd.DataFrame | None = None) -> pd.DataFrame:
    g = d.groupby("batter")
    out = pd.DataFrame({"BIP": g.size()})
    out["Flight runs/BIP (x)"] = g["flight_x"].mean()
    out["Flight runs/BIP (actual)"] = g["flight_a"].mean()
    out["Flight runs (x)"] = g["flight_x"].sum()
    out["GB%"], out["FB%"], out["WEAK FB%"], out["LD%"] = g["gb"].mean(), g["fb"].mean(), g["weak_fb"].mean(), g["ld"].mean()
    out["LD+FB%"] = (d["ld"] | d["fb"]).groupby(d["batter"]).mean()
    s = d[d["spr"]].groupby("batter")
    out["PULL%"], out["MIDDLE%"], out["OPPO%"], out["AIRPULL%"] = s["pull"].mean(), s["mid"].mean(), s["oppo"].mean(), s["airpull"].mean()
    out["FB-PULL%"] = d[d["fb"] & d["spr"]].groupby("batter")["pull"].mean()
    out["PULL-AIR%"] = d[d["pull"]].groupby("batter")["fb"].mean()
    out["LA (hard-hit)"] = d[d.ev >= 95].groupby("batter")["la"].mean()
    out["Flight percentile"] = g["flight_pct"].mean()
    out["GB% (angle only)"], out["FB% (angle only)"] = g["a_gb"].mean(), g["a_fb"].mean()
    out["Pop-up% (angle only)"], out["Sweet Spot%"] = g["a_pop"].mean(), g["sweet"].mean()
    out["AIRPULL% (angle only)"] = d[d["spr"]].groupby("batter")["a_airpull"].mean()
    out["PULL-AIR% (angle only)"] = d[d["pull"]].groupby("batter")["a_fb"].mean()
    out["LA sd (consistency)"] = g["la"].std()
    for t in TYPES:
        for r in DIRS:
            m = d["type"].eq(t) & d["dir"].eq(r)
            out[f"R: {t}, {r}"] = d.loc[m].groupby("batter")["flight_x"].sum().reindex(out.index).fillna(0) / out["BIP"]
    out["power"] = g["ev"].quantile(0.9)
    out["avg_ev"] = g["ev"].mean()
    return out


def main(argv: list[str]) -> None:
    from pipeline.decision_study import push, resid
    from pipeline.power_study import cv_r2, weights
    from pipeline.train_decisions import latest_full_season
    t0 = time.time()
    years = list(range(2015, latest_full_season() + 1))
    tabs, pa = {}, {}
    for y in years:
        d = load(y)
        print(f"[{y}] {len(d):,} balls in play -- flight values + hitter tables…", flush=True)
        tabs[y] = {k: hitter_table(v) for k, v in
                   {"all": d, "odd": d[d.game_pk % 2 == 1], "even": d[d.game_pk % 2 == 0]}.items()}
        del d
        # season wOBA + contact for the "runs with tools held equal" target
        from pipeline import build_league_reference as bl
        raw = bl.pull_season(y)
        raw = raw[raw["game_type"].astype(str).eq("R")] if "game_type" in raw else raw
        wd = raw[pd.to_numeric(raw["woba_denom"], errors="coerce") > 0]
        from core import hit_metrics as hm
        sw = raw[raw["description"].astype(str).isin(hm.SWINGS)]
        zone = pd.to_numeric(raw["zone"], errors="coerce")
        oz = raw[zone.notna() & ~zone.between(1, 9)]
        bs = pd.to_numeric(sw["bat_speed"], errors="coerce") if "bat_speed" in sw else pd.Series(np.nan, index=sw.index)
        pa[y] = pd.DataFrame({"wOBA": pd.to_numeric(wd["woba_value"], errors="coerce").groupby(wd["batter"]).sum()
                              / pd.to_numeric(wd["woba_denom"], errors="coerce").groupby(wd["batter"]).sum(),
                              "contact": 1 - sw["description"].astype(str).isin(hm.WHIFFS).groupby(sw["batter"]).mean(),
                              "chase": oz["description"].astype(str).isin(hm.SWINGS).groupby(oz["batter"]).mean(),
                              "bat_speed": bs.groupby(sw["batter"]).mean()})
        del raw
        print(f"[{y}] done ({(time.time() - t0) / 60:.0f} min)", flush=True)

    L = ["IQ BALL FLIGHT STUDY -- how much of a hitter's run value comes from WHERE he hits the ball (angle + direction),",
         "with bat speed / exit velocity taken out",
         f"Regular season, no bunts. Hitter-seasons with {MIN_BIP}+ balls in play. FLIGHT RUNS = SAxwOBA minus an average "
         "ball hit exactly that hard, / 1.2.", "",
         "HOW TO READ",
         "  RUNS/SZN        typical regular's flight runs from that piece over a season",
         "  REL r / BIP .70 odd vs even games -- is it real skill? / balls in play before it's 70% skill",
         "  AT STAKE        top-10% vs bottom-10% hitter there, runs per season",
         "  SKILL SHARE     share of the REAL (repeatable) differences in flight runs from that piece",
         "  NEXT FLIGHT     NEXT season's ACTUAL flight value per BIP (real outcomes, exit velo held equal) -- main target",
         "  NEXT RUNS|TOOLS next season's wOBA with power (90th EV, avg EV) + contact held equal",
         "  WEIGHT          ridge regression, each metric credited only for what it ADDS (signs set by the data)"]
    for title, ys in ((f"2015-{SPLIT - 1}", [y for y in years if y < SPLIT]), (f"{SPLIT}+", [y for y in years if y >= SPLIT])):
        L += ["", "=" * 112, title, "", "1. WHERE FLIGHT RUNS COME FROM (per BIP, adds up to the total)"]
        To, Te, po, pe, ref = [], [], {}, {}, []
        for y in ys:
            o, e, a = tabs[y]["odd"], tabs[y]["even"], tabs[y]["all"]
            q = o.index[o["BIP"] >= MIN_HALF].intersection(e.index[e["BIP"] >= MIN_HALF])
            To.append(o.loc[q, "Flight runs/BIP (x)"])
            Te.append(e.loc[q, "Flight runs/BIP (x)"])
            for t in TYPES:
                for r in DIRS:
                    k = f"R: {t}, {r}"
                    po.setdefault(k, []).append(o.loc[q, k])
                    pe.setdefault(k, []).append(e.loc[q, k])
            ref.append(a[a["BIP"] >= MIN_BIP])
        To, Te, R = pd.concat(To), pd.concat(Te), pd.concat(ref)
        true = float(np.cov(To, Te)[0, 1])
        L.append(f"  Total flight runs/BIP: odd vs even r = {To.corr(Te):.2f};  top-10% vs bottom-10% = "
                 f"{2.563 * np.sqrt(max(true, 0)) * PER_SEASON:.1f} runs a season")
        L.append(f"  {'PIECE':<26}{'RUNS/SZN':>9}{'REL r':>7}{'AT STAKE':>10}{'SKILL SHARE':>13}")
        rows = []
        for t in TYPES:
            for r in DIRS:
                k = f"R: {t}, {r}"
                a_, b_ = pd.concat(po[k]), pd.concat(pe[k])
                share = (np.cov(a_, Te)[0, 1] + np.cov(b_, To)[0, 1]) / 2 / true if true > 0 else np.nan
                rows.append((f"{t}, {r}", R[k].mean() * PER_SEASON, a_.corr(b_),
                             2.563 * np.sqrt(max(float(np.cov(a_, b_)[0, 1]), 0)) * PER_SEASON, share))
        for name, v, r, st, sh in sorted(rows, key=lambda x: -(x[4] if x[4] == x[4] else -9)):
            L.append(f"  {name:<26}{v:>+9.1f}{r:>7.2f}{st:>10.1f}{sh:>13.0%}")

        # ---- 2. metrics on their own
        prs = []
        for y in ys:
            if y + 1 not in ys:
                continue
            a, b = tabs[y]["all"].join(pa[y]), tabs[y + 1]["all"].join(pa[y + 1])
            a = a[a["BIP"] >= MIN_BIP]
            b = b[b["BIP"] >= MIN_BIP][["Flight runs/BIP (actual)", "wOBA"]].add_prefix("next_")
            prs.append(a.join(b, how="inner").assign(year=y))
        P = pd.concat(prs).reset_index()
        P["tgt"] = resid(P["next_Flight runs/BIP (actual)"], P[TOOLS])       # flight value, other tools held equal
        P["tools"] = resid(P["next_wOBA"], P[TOOLS])
        metrics = CORE + ["Flight runs/BIP (actual)"] + APP + ANGLE + EXTRA
        has_bs = P["bat_speed"].notna().sum() >= 100
        L += ["", f"2. EVERY METRIC ON ITS OWN ({len(P)} hitter season-pairs)",
              "   NEXT FLIGHT|TOOLS = next season's actual flight value with power (90th + avg EV), contact and chase "
              "held equal",
              "   LEAK columns = how tied it is to the OTHER grades' ingredients (near 0 = measures its own thing)",
              f"  {'METRIC':<26}{'REL r':>7}{'BIP .70':>9}{'NEXT FLIGHT|TOOLS':>18}{'NEXT RUNS|TOOLS':>16}"
              f"{'LEAK pwr':>9}{'LEAK bat':>9}{'LEAK con':>9}{'LEAK chase':>11}"]
        res = []
        for m in metrics:
            rs, ns = [], []
            for y in ys:
                o, e = tabs[y]["odd"], tabs[y]["even"]
                q = o.index[o["BIP"] >= MIN_HALF].intersection(e.index[e["BIP"] >= MIN_HALF])
                if o.loc[q, m].notna().sum() >= 30:
                    rs.append(o.loc[q, m].corr(e.loc[q, m]))
                    ns.append(float(((o.loc[q, "BIP"] + e.loc[q, "BIP"]) / 2).mean()))
            r = float(np.nanmean(rs)) if rs else np.nan
            need = float(np.nanmean(ns)) * (0.7 / 0.3) * (1 - r) / r if r == r and r > 0 else np.nan
            res.append((m, r, need, P[m].corr(P["tgt"]), P[m].corr(P["tools"]), P[m].corr(P["power"]),
                        P[m].corr(P["bat_speed"]) if has_bs else np.nan, P[m].corr(P["contact"]), P[m].corr(P["chase"])))
        for m, r, need, t_, tl, pw, bs_, cn, ch in sorted(res, key=lambda x: -abs(x[3]) if x[3] == x[3] else 9):
            nd = f"{need:,.0f}" if need == need else "-"
            L.append(f"  {m:<26}{r:>7.2f}{nd:>9}{t_:>+18.2f}{tl:>+16.2f}{pw:>+9.2f}"
                     + (f"{bs_:>+9.2f}" if bs_ == bs_ else f"{'-':>9}") + f"{cn:>+9.2f}{ch:>+11.2f}")

        # ---- 3. step test + weights (signs from the data: flip negatives so all weights can be positive)
        signed = []
        for m in APP + ANGLE + EXTRA:
            s = np.sign(P[m].corr(P["tgt"])) or 1
            P["±" + m] = s * P[m]
            signed.append(("±" + m, s))
        core = ["Flight percentile"]
        app_s = ["±" + m for m in APP]
        ang_s = ["±" + m for m in ANGLE]
        rows3 = [("flight percentile alone", core), ("flight runs (in runs) alone", ["Flight runs/BIP (x)"]),
                 ("+ angle-only metrics", core + ang_s), ("+ the 11 app metrics", core + ang_s + app_s),
                 ("the 11 app metrics alone", app_s), ("angle-only metrics alone", ang_s)]
        L += ["", "3. WHAT PREDICTS NEXT SEASON'S FLIGHT VALUE (tools held equal)? cross-validated R^2"]
        for name, cols in rows3:
            L.append(f"    {name:<34} R^2 = {cv_r2(P[cols], P['tgt'], P['batter']):.3f}")
        app_s = ang_s + app_s + ["±" + m for m in EXTRA]
        W = weights(P[core + app_s], P["tgt"])
        L.append("  data's weights: " + ", ".join(f"{k.replace('±', '')}{'' if dict(signed).get(k, 1) > 0 else ' (low)'} {v:.0%}"
                                             for k, v in W.sort_values(ascending=False).items() if v > 0.005))
        Wt = weights(P[core + app_s], P["tools"])
        L.append("  same, for runs with tools held equal: " + ", ".join(
            f"{k.replace('±', '')}{'' if dict(signed).get(k, 1) > 0 else ' (low)'} {v:.0%}"
            for k, v in Wt.sort_values(ascending=False).items() if v > 0.005))

        # ---- 4. eye test
        last = ys[-1]
        s = tabs[last]["all"].join(pa[last])
        s = s[s["BIP"] >= MIN_BIP]
        try:
            from core import mlb_api
            names = mlb_api.get_people(s.index.astype(int).tolist())
        except Exception:
            names = {}
        top = s.sort_values("Flight percentile", ascending=False)
        f = lambda r: f"{names.get(int(r.name), r.name)} {100 * r['Flight percentile']:+.1f} ({r['power']:.0f} 90thEV)"
        L += ["", f"4. {last} EYE TEST -- flight percentile ({MIN_BIP}+ BIP; +5 = his balls average the 55th percentile "
                  "of balls hit that hard). 90th EV shown to check power isn't driving it",
              "  top:    " + " | ".join(f(r) for _, r in top.head(10).iterrows()),
              "  bottom: " + " | ".join(f(r) for _, r in top.tail(8).iterrows())]

    out = "\n".join(L)
    ART.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(out + "\n", encoding="utf-8")
    pd.concat([tabs[y]["all"].join(pa[y]).assign(year=y) for y in years]).to_csv(CSV)
    print("\n" + out)
    print(f"\nSaved {REPORT} + {CSV.name} in {(time.time() - t0) / 60:.0f} min", flush=True)
    if "--no-push" not in argv:
        push([REPORT, CSV])


if __name__ == "__main__":
    main(sys.argv[1:])
