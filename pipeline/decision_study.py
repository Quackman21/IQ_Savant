"""Decision study (analysis only -- nothing on the site changes). Built around the RUN VALUE of a hitter's swing /
take decisions: where it comes from, what predicts it, and how the Decision Grade should be weighted.

    python -m pipeline.decision_study            (PC)   |   caffeinate -i python -m pipeline.decision_study   (Mac)
    python -m pipeline.decision_study --no-push   # don't push the results to GitHub when it finishes

Uses the saved pitches (data/raw) and the IQ Swing Decisions model. Every pitch gets the run value of his choice
vs what an average hitter would do on that same pitch (DEC RUNS). Swing rate itself is never rewarded.

  1. WHERE DECISION RUNS COME FROM -- each hitter's decision runs split by zone x swing / take (heart takes,
     heart takes in hitter's counts, chase swings, waste swings ...), by count and by pitch type: the typical
     value of each piece, how much hitters really differ there, and its share of the REAL (repeatable)
     differences in decision runs.
  2. EVERY METRIC ON ITS OWN -- GD%, DEC QUAL, DEC RUNS/100, heart / shadow / chase / waste swing rates, pre-2k
     and 2k chase, the heart-minus-chase gap: stability, pitches to trust, and NEXT season's run value with the
     hitter's tools held equal (power + contact), walks, strikeouts.
  3. GD% vs DEC RUNS/100 as the core (question 1) and how to weight 2-strike chasing (question 2) -- candidate
     grade mixes head to head, plus the data's own weights.
  4. Count / pitch-type splits: how many pitches each split needs (sets their regression).
  5. Eye test.
Output: models/artifacts/decision_study_report.txt (+ decision_study_hitters.csv), pushed to GitHub when done.
"""
from __future__ import annotations

import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "models" / "artifacts"
REPORT = ART / "decision_study_report.txt"
CSV = ART / "decision_study_hitters.csv"
SPLIT = 2023
MIN_PITCHES, MIN_HALF = 1000, 400          # graded decisions per hitter-season / per half
PER_SEASON = 2300                          # graded decisions for a typical starter (to express runs per season)
AHEAD = {(1, 0), (2, 0), (3, 0), (2, 1), (3, 1)}

ZONE_ACTION = ["Heart swings", "Heart takes", "Shadow swings", "Shadow takes", "Chase swings", "Chase takes",
               "Waste swings", "Waste takes"]
COUNTS = ["0-0", "Hitter's counts", "Other counts <2k", "Two strikes"]
PITCHES = ["Fastballs", "Breaking", "Offspeed"]
SPECIAL = ["Heart takes in hitter's counts", "Chase+waste swings <2k", "Chase+waste swings 2k"]

# metrics, oriented so HIGHER = BETTER for the weights (name, column, sign)
METRICS = [("GD%", "GD%", 1), ("DEC QUAL", "DEC QUAL", 1), ("DEC RUNS/100", "DEC RUNS/100", 1),
           ("H-Swing%", "H-Swing%", 1), ("S-Swing%", "S-Swing%", 0), ("C-Swing% (low)", "C-Swing%", -1),
           ("W-Swing% (low)", "W-Swing%", -1), ("Pre2k O-Swing% (low)", "Pre2k O-Swing%", -1),
           ("2k O-Swing% (low)", "2k O-Swing%", -1), ("Heart-minus-chase gap", "Gap", 1)]


def load(year: int) -> pd.DataFrame:
    from core import hit_metrics as hm
    from models import decisions_iq as dq
    from pipeline import build_league_reference as bl
    raw = bl.pull_season(year)
    if "game_type" in raw:
        raw = raw[raw["game_type"].astype(str).eq("R")]
    raw = raw.reset_index(drop=True)
    sc = dq.score(raw)
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")
    desc = raw["description"].astype(str)
    d = pd.DataFrame({"game_pk": num("game_pk"), "batter": raw["batter"], "year": year,
                      "swing": desc.isin(hm.SWINGS), "whiff": desc.isin(hm.WHIFFS),
                      "zone": num("zone"), "att": hm.attack_zone(raw), "pg": hm.pitch_group(raw),
                      "balls": num("balls"), "strikes": num("strikes"),
                      "good": sc["good"], "quality": sc["quality"], "runs": sc["runs_vs_avg"],
                      "events": raw["events"], "woba_value": num("woba_value"), "woba_denom": num("woba_denom"),
                      "type": raw["type"], "ev": num("launch_speed"), "la": num("launch_angle")})
    d["graded"] = d["quality"].notna()
    d["in_z"] = d["zone"].between(1, 9)
    d["two_k"] = d["strikes"] >= 2
    d["ahead"] = [(b, s) in AHEAD for b, s in zip(d["balls"], d["strikes"])]
    d["first"] = (d["balls"] == 0) & (d["strikes"] == 0)
    return d


def bucket_masks(d: pd.DataFrame) -> dict:
    sw, tk = d["swing"], ~d["swing"]
    m = {}
    for z in ("Heart", "Shadow", "Chase", "Waste"):
        m[f"{z} swings"], m[f"{z} takes"] = d["att"].eq(z) & sw, d["att"].eq(z) & tk
    m["0-0"] = d["first"]
    m["Hitter's counts"] = d["ahead"] & ~d["first"]
    m["Other counts <2k"] = ~d["ahead"] & ~d["first"] & ~d["two_k"]
    m["Two strikes"] = d["two_k"]
    m["Fastballs"], m["Breaking"], m["Offspeed"] = d["pg"].eq("Fastball"), d["pg"].eq("Breaking"), d["pg"].eq("Offspeed")
    m["Heart takes in hitter's counts"] = d["att"].eq("Heart") & tk & d["ahead"]
    cw = d["att"].isin(["Chase", "Waste"]) & sw
    m["Chase+waste swings <2k"], m["Chase+waste swings 2k"] = cw & ~d["two_k"], cw & d["two_k"]
    return m


def hitter_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d[d["graded"]]
    G = g.groupby("batter")
    out = pd.DataFrame({"graded": G.size()})
    out["GD%"] = g[g["good"].notna()].groupby("batter")["good"].mean()
    out["DEC QUAL"] = G["quality"].mean()
    out["DEC RUNS"] = G["runs"].sum()
    out["DEC RUNS/100"] = 100 * out["DEC RUNS"] / out["graded"]
    for name, mask in bucket_masks(g).items():                       # runs per 100 of ALL his graded pitches
        out["R: " + name] = 100 * g[mask].groupby("batter")["runs"].sum().reindex(out.index).fillna(0) / out["graded"]
    rate = lambda m: d[m].groupby("batter")["swing"].mean()
    for z, c in (("Heart", "H"), ("Shadow", "S"), ("Chase", "C"), ("Waste", "W")):
        out[f"{c}-Swing%"] = rate(d["att"].eq(z))
    out["Pre2k O-Swing%"] = rate(~d["in_z"] & ~d["two_k"])
    out["2k O-Swing%"] = rate(~d["in_z"] & d["two_k"])
    out["CW-Swing%"] = rate(d["att"].isin(["Chase", "Waste"]))
    out["Gap"] = out["H-Swing%"] - out["CW-Swing%"]
    out["SWING%"] = d.groupby("batter")["swing"].mean()
    for split, mask in (("00", d["first"]), ("lt2", ~d["two_k"]), ("2k", d["two_k"]), ("fb", d["pg"].eq("Fastball")),
                        ("bb", d["pg"].eq("Breaking")), ("os", d["pg"].eq("Offspeed"))):
        s = g[mask.reindex(g.index).fillna(False)]
        out[f"n_{split}"] = s.groupby("batter").size()
        out[f"R100_{split}"] = 100 * s.groupby("batter")["runs"].sum() / out[f"n_{split}"]
        out[f"GD_{split}"] = s[s["good"].notna()].groupby("batter")["good"].mean()
    pa = d[d["events"].notna() & d["events"].astype(str).ne("")]
    P = pa.groupby("batter")
    out["PA"] = P.size()
    out["K%"] = P["events"].apply(lambda e: e.astype(str).str.startswith("strikeout").mean())
    out["BB%"] = P["events"].apply(lambda e: e.astype(str).isin(["walk", "intent_walk"]).mean())
    wd = d[d["woba_denom"] > 0]
    out["wOBA"] = wd.groupby("batter")["woba_value"].sum() / wd.groupby("batter")["woba_denom"].sum()
    bip = d[d["type"].eq("X")]
    out["power"] = bip.groupby("batter")["ev"].quantile(0.9)
    out["power_fly"] = bip[(bip["ev"] > 70) & bip["la"].between(10, 50)].groupby("batter")["ev"].mean()
    sw = d[d["swing"]]
    out["contact"] = 1 - sw.groupby("batter")["whiff"].mean()
    return out


def resid(y: pd.Series, Z: pd.DataFrame) -> pd.Series:
    ok = y.notna() & Z.notna().all(axis=1)
    out = pd.Series(np.nan, index=y.index)
    A = np.column_stack([np.ones(ok.sum()), Z[ok].values])
    out[ok] = y[ok] - A @ np.linalg.lstsq(A, y[ok], rcond=None)[0]
    return out


def push(paths: list[Path]) -> None:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    git("add", *[str(p.relative_to(ROOT)) for p in paths])
    if git("diff", "--cached", "--quiet").returncode != 0:
        git("commit", "-m", "Decision study results")
    git("pull", "--rebase", "--autostash")
    r = git("push")
    print("Pushed the results to GitHub." if r.returncode == 0 else f"Push FAILED:\n{r.stderr}", flush=True)


def main(argv: list[str]) -> None:
    from pipeline.power_study import cv_r2, weights
    from pipeline.train_decisions import latest_full_season
    t0 = time.time()
    years = list(range(2015, latest_full_season() + 1))
    tabs = {}
    for y in years:
        d = load(y)
        print(f"[{y}] {d['graded'].sum():,} graded decisions -- hitter tables…", flush=True)
        tabs[y] = {k: hitter_table(v) for k, v in
                   {"all": d, "odd": d[d.game_pk % 2 == 1], "even": d[d.game_pk % 2 == 0]}.items()}
        del d
        print(f"[{y}] done ({(time.time() - t0) / 60:.0f} min)", flush=True)

    L = ["IQ DECISION STUDY -- the run value of swing / take decisions: where it comes from and how to grade it",
         f"Regular season. Hitter-seasons with {MIN_PITCHES:,}+ graded decisions. DEC RUNS = runs of his choice vs what "
         "an average hitter does on that same pitch.", "",
         "HOW TO READ",
         "  RUNS/SZN        typical hitter's decision runs from that piece over a season (league average is ~0 overall)",
         "  REL r           odd vs even games: do hitters really differ there?",
         "  AT STAKE        top-10% vs bottom-10% hitter there, runs per season",
         "  SKILL SHARE     share of the REAL (repeatable) differences in total decision runs from that piece",
         "  PITCHES .70     graded decisions before the number is 70% skill",
         "  NEXT RUNS|TOOLS NEXT season's wOBA with power + contact held equal: do better decisions mean more runs",
         "                  for the same tools? (the main target)",
         "  NEXT BB / -K    next season's walk rate / strikeouts avoided;  NEXT SELF  next season's same number"]

    for title, ys in ((f"2015-{SPLIT - 1}", [y for y in years if y < SPLIT]), (f"{SPLIT}+", [y for y in years if y >= SPLIT])):
        # ---------- 1. where decision runs come from
        L += ["", "=" * 118, f"{title}", "", "1. WHERE DECISION RUNS COME FROM"]
        T_o, T_e, parts_o, parts_e, ref = [], [], {}, {}, []
        for y in ys:
            o, e, a = tabs[y]["odd"], tabs[y]["even"], tabs[y]["all"]
            q = o.index[o["graded"] >= MIN_HALF].intersection(e.index[e["graded"] >= MIN_HALF])
            T_o.append(o.loc[q, "DEC RUNS/100"])
            T_e.append(e.loc[q, "DEC RUNS/100"])
            for b in ZONE_ACTION + COUNTS + PITCHES + SPECIAL:
                parts_o.setdefault(b, []).append(o.loc[q, "R: " + b])
                parts_e.setdefault(b, []).append(e.loc[q, "R: " + b])
            ref.append(a[a["graded"] >= MIN_PITCHES])
        To, Te, R = pd.concat(T_o), pd.concat(T_e), pd.concat(ref)
        true_tot = float(np.cov(To, Te)[0, 1])
        L.append(f"  Total DEC RUNS/100: odd vs even r = {To.corr(Te):.2f};  top-10% vs bottom-10% hitter = "
                 f"{2.563 * np.sqrt(max(true_tot, 0)) * PER_SEASON / 100:.1f} runs a season")
        for group, names in (("By zone and swing / take (adds up to the total)", ZONE_ACTION),
                             ("By count (adds up to the total)", COUNTS), ("By pitch type (adds up to the total)", PITCHES),
                             ("Specific pieces you asked about", SPECIAL)):
            L += ["", f"  {group}", f"  {'PIECE':<34}{'RUNS/SZN':>9}{'REL r':>7}{'AT STAKE':>10}{'SKILL SHARE':>13}"]
            for b in names:
                po, pe = pd.concat(parts_o[b]), pd.concat(parts_e[b])
                cov_true = float(np.cov(po, pe)[0, 1])
                share = (np.cov(po, Te)[0, 1] + np.cov(pe, To)[0, 1]) / 2 / true_tot if true_tot > 0 else np.nan
                L.append(f"  {b:<34}{R['R: ' + b].mean() * PER_SEASON / 100:>+9.1f}{po.corr(pe):>7.2f}"
                         f"{2.563 * np.sqrt(max(cov_true, 0)) * PER_SEASON / 100:>10.1f}{share:>13.0%}")

        # ---------- 2. every metric on its own
        rows = []
        for y in ys:
            if y + 1 not in ys:
                continue
            a, b = tabs[y]["all"], tabs[y + 1]["all"]
            a = a[a["graded"] >= MIN_PITCHES]
            b = b[b["graded"] >= MIN_PITCHES][["wOBA", "BB%", "K%", "DEC RUNS/100", "GD%", "DEC QUAL"]].add_prefix("next_")
            rows.append(a.join(b, how="inner").assign(year=y))
        P = pd.concat(rows).reset_index()
        P["target"] = resid(P["next_wOBA"], P[["power", "power_fly", "contact"]])     # run value, tools held equal
        P["-K"] = -P["next_K%"]
        L += ["", "2. EVERY METRIC ON ITS OWN" + f"  ({len(P)} hitter season-pairs)",
              f"  {'METRIC':<24}{'REL r':>7}{'PITCHES .70':>12}{'NEXT RUNS|TOOLS':>16}{'NEXT BB':>9}{'NEXT -K':>9}"
              f"{'NEXT SELF':>10}"]
        res = []
        for name, col, sign in METRICS + [("SWING% (not used)", "SWING%", 0)]:
            rs, ns = [], []
            for y in ys:
                o, e = tabs[y]["odd"], tabs[y]["even"]
                q = o.index[o["graded"] >= MIN_HALF].intersection(e.index[e["graded"] >= MIN_HALF])
                rs.append(o.loc[q, col].corr(e.loc[q, col]))
                ns.append(float(((o.loc[q, "graded"] + e.loc[q, "graded"]) / 2).mean()))
            r = float(np.nanmean(rs))
            need = float(np.nanmean(ns)) * (0.7 / 0.3) * (1 - r) / r if r > 0 else np.nan
            s = sign if sign else 1
            nxt = P["next_" + col] if "next_" + col in P else None
            res.append((name, r, need, (s * P[col]).corr(P["target"]), (s * P[col]).corr(P["next_BB%"]),
                        (s * P[col]).corr(P["-K"]), P[col].corr(nxt) if nxt is not None else np.nan))
        for name, r, need, rt, bb, k, self_ in sorted(res, key=lambda x: -(x[3] if x[3] == x[3] else -9)):
            nd = f"{need:,.0f}" if need == need else "-"
            L.append(f"  {name:<24}{r:>7.2f}{nd:>12}{rt:>16.2f}{bb:>9.2f}{k:>9.2f}"
                     + (f"{self_:>10.2f}" if self_ == self_ else f"{'':>10}"))

        # ---------- 3. candidate grade mixes
        for name, col, sign in METRICS:
            if sign:
                P[name] = sign * P[col]
        zst = lambda c: (P[c] - P[c].mean()) / P[c].std()
        mixes = {
            "A  GD% core: GD 50 | H 20 | chase 20 | waste 10": {"GD%": .5, "H-Swing%": .2, "C-Swing% (low)": .2, "W-Swing% (low)": .1},
            "B  RUNS core: DR/100 50 | H 20 | chase 20 | waste 10": {"DEC RUNS/100": .5, "H-Swing%": .2, "C-Swing% (low)": .2, "W-Swing% (low)": .1},
            "C  RUNS core, chase by count EQUAL (pre2k 10 / 2k 10)": {"DEC RUNS/100": .5, "H-Swing%": .2, "Pre2k O-Swing% (low)": .1, "2k O-Swing% (low)": .1, "W-Swing% (low)": .1},
            "D  RUNS core, 2k chase HEAVIER (pre2k 5 / 2k 15)": {"DEC RUNS/100": .5, "H-Swing%": .2, "Pre2k O-Swing% (low)": .05, "2k O-Swing% (low)": .15, "W-Swing% (low)": .1},
            "E  RUNS core, pre2k chase HEAVIER (pre2k 15 / 2k 5)": {"DEC RUNS/100": .5, "H-Swing%": .2, "Pre2k O-Swing% (low)": .15, "2k O-Swing% (low)": .05, "W-Swing% (low)": .1},
            "F  RUNS core + gap: DR/100 60 | gap 40": {"DEC RUNS/100": .6, "Heart-minus-chase gap": .4},
            "G  DEC RUNS/100 only": {"DEC RUNS/100": 1.0},
            "H  GD% only": {"GD%": 1.0},
            "I  DEC QUAL only": {"DEC QUAL": 1.0},
        }
        feats = [n for n, c, s in METRICS if s]
        Wd = weights(P[feats], P["target"])
        mixes["J  the data's own weights (runs | tools)"] = Wd[Wd > 0].to_dict()
        L += ["", "3. GRADE MIXES HEAD TO HEAD (each metric standardized; higher = better)",
              f"  {'MIX':<58}{'RUNS|TOOLS':>11}{'BB':>7}{'-K':>7}{'SELF DR':>9}" + "".join(f"{y}>{y + 1 - 2000}".rjust(9)
                                                                                       for y in sorted(P.year.unique()))]
        for name, w in mixes.items():
            c = sum(v * zst(k) for k, v in w.items())
            by = [c[P.year == y].corr(P.loc[P.year == y, "target"]) for y in sorted(P.year.unique())]
            L.append(f"  {name:<58}{c.corr(P['target']):>11.3f}{c.corr(P['next_BB%']):>7.3f}{c.corr(P['-K']):>7.3f}"
                     f"{c.corr(P['next_DEC RUNS/100']):>9.3f}" + "".join(f"{v:>9.3f}" for v in by))
        L.append("  data's weights: " + ", ".join(f"{k} {v:.0%}" for k, v in Wd.sort_values(ascending=False).items() if v > 0))

        # ---------- 4. splits
        L += ["", "4. COUNT / PITCH-TYPE SPLITS -- how fast each split becomes real skill (sets their regression)",
              f"  {'SPLIT':<10}{'PITCHES/SZN':>12}{'DR/100 REL r':>14}{'DR/100 PITCHES .70':>20}{'GD% REL r':>11}"
              f"{'GD% PITCHES .70':>17}"]
        for split, label in (("00", "0-0"), ("lt2", "<2k"), ("2k", "2k"), ("fb", "FB"), ("bb", "BB"), ("os", "OS")):
            out = [label, float(R[f"n_{split}"].median())]
            for col in (f"R100_{split}", f"GD_{split}"):
                rs, ns = [], []
                for y in ys:
                    o, e = tabs[y]["odd"], tabs[y]["even"]
                    q = o.index[o["graded"] >= MIN_HALF].intersection(e.index[e["graded"] >= MIN_HALF])
                    rs.append(o.loc[q, col].corr(e.loc[q, col]))
                    ns.append(float(((o.loc[q, f"n_{split}"] + e.loc[q, f"n_{split}"]) / 2).mean()))
                r = float(np.nanmean(rs))
                out += [r, float(np.nanmean(ns)) * (0.7 / 0.3) * (1 - r) / r if r > 0 else np.nan]
            L.append(f"  {out[0]:<10}{out[1]:>12,.0f}{out[2]:>14.2f}{out[3]:>20,.0f}{out[4]:>11.2f}{out[5]:>17,.0f}")

        # ---------- 5. eye test (mix B on the last season)
        last = ys[-1]
        s = tabs[last]["all"]
        s = s[s["graded"] >= MIN_PITCHES].copy()
        for name, col, sign in METRICS:
            if sign:
                s[name] = sign * s[col]
        w = mixes["B  RUNS core: DR/100 50 | H 20 | chase 20 | waste 10"]
        c = sum(v * (s[k] - s[k].mean()) / s[k].std() for k, v in w.items())
        s["grade"] = 50 + 10 * c / c.std()
        try:
            from core import mlb_api
            names = mlb_api.get_people(s.index.astype(int).tolist())
        except Exception:
            names = {}
        top = s.sort_values("grade", ascending=False)
        f = lambda r: f"{names.get(int(r.name), r.name)} {r['grade']:.0f} ({r['DEC RUNS']:+.0f} runs)"
        L += ["", f"5. {last} EYE TEST -- mix B (decision runs core), {MIN_PITCHES:,}+ graded decisions",
              "  top:    " + " | ".join(f(r) for _, r in top.head(8).iterrows()),
              "  bottom: " + " | ".join(f(r) for _, r in top.tail(6).iterrows())]

    out = "\n".join(L)
    ART.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(out + "\n", encoding="utf-8")
    pd.concat([tabs[y]["all"].assign(year=y) for y in years]).to_csv(CSV)
    print("\n" + out)
    print(f"\nSaved {REPORT} + {CSV.name} in {(time.time() - t0) / 60:.0f} min", flush=True)
    if "--no-push" not in argv:
        push([REPORT, CSV])


if __name__ == "__main__":
    main(sys.argv[1:])
