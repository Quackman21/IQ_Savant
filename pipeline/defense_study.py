"""Defense study (v20.0): how trustworthy the new fielding grades are, and count vs situational framing runs.

    caffeinate -i python -m pipeline.defense_study          # Mac  (PC: .venv\\Scripts\\python.exe -m ...)
    python -m pipeline.defense_study --no-rebuild           # skip re-scoring the seasons (files already new)

Steps
  1. Re-scores every season's fielding + framing files with the current models (needs
     python -m pipeline.train_fielding first, for the Conversion / Ball Security models). Saved pitches only.
  2. Fielding grades: split-half (odd / even games) reliability of Range, Conversion and Ball Security per chance ->
     chances before each is 70% skill (written to data/league/fielding_grades.json, which the grades use), year-to-
     year repeatability, and an eye test with names.
  3. Framing: count-valued runs (the profile) vs situational runs (count + runners + outs, for after-action reports):
     how much catchers' runs and ranks move, and which predicts next season better.
  4. Writes models/artifacts/defense_study_report.txt and pushes the results to GitHub.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "models" / "artifacts" / "defense_study_report.txt"
GROUPS = {"IF": [3, 4, 5, 6], "OF": [7, 8, 9]}
STATS = {"IF": ["range", "conv", "sec"], "OF": ["range"]}
NAMES = {"range": "Range", "conv": "Conversion", "sec": "Ball Security"}
MIN_HALF = {"range": 40, "conv": 30, "sec": 30}
MIN_INN = 300
MIN_FRAME = 2000


def rebuild(years: list[int]) -> None:
    from core import hit_metrics as hm
    from core import league_ref
    from models import fielding, framing
    from pipeline import build_league_reference as bl
    for y in years:
        t = time.time()
        raw = bl.pull_season(y)
        if "game_type" in raw:
            raw = raw[raw["game_type"].astype(str).eq("R")]
        clean = hm.clean(raw)
        league_ref.save_catching(framing.daily_table(framing.framing_pitches(clean)), y)
        ev = bl.RAW_DIR / f"events_{y}.csv.gz"
        events = pd.read_csv(ev) if ev.exists() else None
        league_ref.save_fielding(fielding.daily_table(clean, events=events), y)
        print(f"[{y}] fielding + framing files re-scored in {time.time() - t:.0f}s", flush=True)
        del raw, clean


def player_rates(daily: pd.DataFrame, positions: list[int]) -> pd.DataFrame:
    from models import fielding as fm
    s = fm.summarize(daily, "0000", "9999", positions)
    rows = []
    for p, v in s.items():
        r = {"player": p, "inn": v["IQ_INN"]}
        for stat in fm.GRADE_STATS:
            r[stat], r[stat + "_n"] = fm._rate(v, stat)
        rows.append(r)
    return pd.DataFrame(rows)


def for70(r: float, n: float) -> float:
    """Chances before a rate is 70% skill, from a split-half r at n chances per half (Spearman-Brown)."""
    if not r or r <= 0 or n != n:
        return float("nan")
    k = n * (1 - r) / r
    return 0.7 / 0.3 * k


def fielding_section(years: list[int]) -> tuple[list[str], dict]:
    from core import league_ref
    from models import fielding as fm
    L, out = ["", "=" * 100, "1. FIELDING GRADES -- can we trust them?", "=" * 100,
              "Split-half: each player's odd games vs even games, same season and position group. r = how much of the "
              "rate repeats; 'for .70' = chances before the rate is 70% skill (what the grades regress with).", ""], {}
    for g, pos in GROUPS.items():
        for stat in STATS[g]:
            pairs = []
            for y in years:
                d = league_ref.load_fielding(y)
                if d.empty or "conv_n" not in d or d["conv_n"].sum() == 0 and stat != "range":
                    continue
                d = d[d["pos"].isin(pos)]
                a = player_rates(d[d["game_pk"] % 2 == 1], pos)
                b = player_rates(d[d["game_pk"] % 2 == 0], pos)
                m = a.merge(b, on="player", suffixes=("_a", "_b"))
                m = m[(m[f"{stat}_n_a"] >= MIN_HALF[stat]) & (m[f"{stat}_n_b"] >= MIN_HALF[stat])]
                pairs.append(m.assign(year=y))
            if not pairs:
                L.append(f"  {g} {NAMES[stat]:<14} no data yet (run python -m pipeline.train_fielding, then this study)")
                continue
            m = pd.concat(pairs)
            if len(m) < 30:
                continue
            r = m[f"{stat}_a"].corr(m[f"{stat}_b"])
            n = float(pd.concat([m[f"{stat}_n_a"], m[f"{stat}_n_b"]]).mean())
            n70 = for70(r, n)
            if n70 == n70:
                out[f"{stat}_{g}"] = round(n70)
            L.append(f"  {g} {NAMES[stat]:<14} split-half r {r:.2f} at {n:.0f} chances a half "
                     f"-> {n70:,.0f} chances for .70   ({len(m)} player-seasons)")
    L += ["", "Year to year (regulars, 300+ innings both years): r of the rate this season vs next"]
    for g, pos in GROUPS.items():
        prev = None
        rs = {s_: [] for s_ in STATS[g]}
        for y in years:
            d = league_ref.load_fielding(y)
            if d.empty:
                prev = None
                continue
            cur = player_rates(d[d["pos"].isin(pos)], pos)
            cur = cur[cur.inn >= MIN_INN]
            if prev is not None:
                m = prev.merge(cur, on="player", suffixes=("_1", "_2"))
                for s_ in STATS[g]:
                    rs[s_].append(m[[f"{s_}_1", f"{s_}_2"]].dropna())
            prev = cur
        for s_, parts in rs.items():
            if parts:
                m = pd.concat(parts)
                if len(m) > 20:
                    L.append(f"  {g} {NAMES[s_]:<14} r {m.iloc[:, 0].corr(m.iloc[:, 1]):.2f}  ({len(m)} pairs)")
    # eye test, latest season, with the measured regression
    if out:
        fm.GRADE_FILE.parent.mkdir(parents=True, exist_ok=True)
        fm.GRADE_FILE.write_text(json.dumps({"for_70": out}, indent=1))
    last = years[-1]
    d = league_ref.load_fielding(last)
    if not d.empty:
        try:
            from core import mlb_api
            names = mlb_api.get_people(sorted(d["player"].astype(int).unique().tolist()))
        except Exception:
            names = {}
        L += ["", f"{last} eye test -- grades by position ({MIN_INN}+ innings), regressed:"]
        for k, name in fm.POS.items():
            spread = fm.grade_spread(d, [k])
            s = fm.summarize(d, "0000", "9999", [k])
            g = "IF" if k in fm.INFIELD else "OF"
            rows = [(p, fm.grades(v, spread, g)) for p, v in s.items() if v["IQ_INN"] >= MIN_INN]
            for key in fm.GRADE_KEYS[g].values():
                rr = sorted([(p, x[key]) for p, x in rows if key in x], key=lambda t: -t[1])
                if len(rr) >= 8:
                    f_ = lambda t: f"{names.get(t[0], t[0])} {t[1]:.0f}"
                    L.append(f"  {name} {key.replace('_', ' ').title():<20} top: " + " | ".join(map(f_, rr[:4]))
                             + "   bottom: " + " | ".join(map(f_, rr[-3:])))
    return L, out


def framing_section(years: list[int]) -> list[str]:
    from core import league_ref
    L = ["", "=" * 100, "2. FRAMING RUNS -- count only (profile) vs count + runners + outs (after-action reports)",
         "=" * 100, f"Catcher seasons with {MIN_FRAME:,}+ framing opportunities.", ""]
    rows = []
    for y in years:
        c = league_ref.load_catching(y)
        if c.empty or "runs_sit" not in c:
            continue
        g = c.groupby("catcher")[["n", "strikes", "exp", "runs", "runs_sit"]].sum()
        g = g[g.n >= MIN_FRAME].assign(year=y, xs=lambda t: t.strikes - t.exp)
        rows.append(g.reset_index())
    if not rows:
        return L + ["  no situational framing yet -- re-score the seasons (this study does it unless --no-rebuild)"]
    t = pd.concat(rows)
    t["rank_c"] = t.groupby("year")["runs"].rank(ascending=False)
    t["rank_s"] = t.groupby("year")["runs_sit"].rank(ascending=False)
    L += [f"  same season: r(count runs, situational runs) = {t.runs.corr(t.runs_sit):.3f}",
          f"  average rank change between the two: {(t.rank_c - t.rank_s).abs().mean():.1f} spots "
          f"(of ~{t.groupby('year').size().mean():.0f} catchers a season)",
          f"  biggest gap: {(t.runs_sit - t.runs).abs().max():.1f} runs; typical gap "
          f"{(t.runs_sit - t.runs).abs().median():.1f} runs", ""]
    nx = t[["catcher", "year", "n", "xs", "runs", "runs_sit"]].copy()
    nx["year"] -= 1
    m = t.merge(nx, on=["catcher", "year"], suffixes=("", "_next"))
    if len(m) > 20:
        rate = lambda c, n: m[c] / m[n]
        tgt = {"extra strikes / pitch": rate("xs_next", "n_next"), "count runs / pitch": rate("runs_next", "n_next"),
               "situational runs / pitch": rate("runs_sit_next", "n_next")}
        L.append("  Which of THIS season's numbers best predicts NEXT season? (r, per pitch)")
        L.append(f"  {'this season':<28}" + "".join(f"{k:>28}" for k in tgt))
        for lab, x in (("extra strikes", rate("xs", "n")), ("count runs", rate("runs", "n")),
                       ("situational runs", rate("runs_sit", "n"))):
            L.append(f"  {lab:<28}" + "".join(f"{x.corr(v):>28.3f}" for v in tgt.values()))
        L.append(f"  ({len(m)} catcher pairs)")
    return L


def main(argv: list[str]) -> None:
    from pipeline.decision_study import push
    from pipeline.train_decisions import latest_full_season
    t0 = time.time()
    years = list(range(2015, latest_full_season() + 1))
    if "--no-rebuild" not in argv:
        rebuild(years)
    L = ["IQ DEFENSE STUDY (v20.0)", f"seasons {years[0]}-{years[-1]}"]
    fl, k70 = fielding_section(years)
    L += fl + framing_section(years)
    L += ["", f"done in {(time.time() - t0) / 60:.0f} min"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)
    from models import fielding as fm
    paths = [REPORT] + ([fm.GRADE_FILE] if fm.GRADE_FILE.exists() else []) + \
            sorted((ROOT / "data" / "league").glob("fielding_*.csv.gz")) + \
            sorted((ROOT / "data" / "league").glob("catching_*.csv.gz"))
    if "--no-push" not in argv:
        push(paths, "Defense study results (v20.0)")
    try:
        import os
        import subprocess
        if sys.platform == "darwin":
            subprocess.run(["open", str(REPORT)])
        elif os.name == "nt":
            os.startfile(str(REPORT))                                  # noqa
    except Exception:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
