"""Fielding comparison study: our numbers vs Statcast OAA and SIS DRS -- agreement, which predicts next season best,
the best Range blend, and which 20-80 scale to use.

    caffeinate -i python -m pipeline.fielding_compare_study       # Mac  (PC: .venv\\Scripts\\python.exe -m ...)
    python -m pipeline.fielding_compare_study --no-push

Inputs
  * our saved fielding files (data/league/fielding_<year>.csv.gz -- already re-scored by pipeline.defense_study)
  * Statcast OAA by position: downloaded once a season, cached in data/compare/oaa_<year>_<pos>.csv
  * SIS DRS: data/compare/drs_<year>.csv for every season you saved (player name, position, DRS; component
    columns like range / positioning, good plays, errors are used when present). Seasons without a file are skipped.
Writes models/artifacts/fielding_compare_study.txt, pushes it to GitHub, opens it.
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
COMPARE = ROOT / "data" / "compare"
REPORT = ROOT / "models" / "artifacts" / "fielding_compare_study.txt"
POS = {3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}
GROUP = {k: ("IF" if k <= 6 else "OF") for k in POS}
MIN_INN = 300
FIRST = 2016                                 # OAA starts (outfield 2016, infield 2016+ on the board)


# ---------------------------------------------------------------------------- data
def oaa(year: int, pos: int) -> pd.DataFrame:
    """Statcast OAA for one position-season: player, oaa (cached)."""
    COMPARE.mkdir(parents=True, exist_ok=True)
    p = COMPARE / f"oaa_{year}_{pos}.csv"
    if p.exists():
        return pd.read_csv(p).astype({"player": int})
    try:
        import pybaseball as pb
        from pipeline.compare_fielding import _col
        b = pb.statcast_outs_above_average(year, pos, 1)
        idc, vc = _col(b, "player_id"), _col(b, "outs_above_average")
        out = pd.DataFrame({"player": pd.to_numeric(b[idc], errors="coerce"),
                            "oaa": pd.to_numeric(b[vc], errors="coerce")}).dropna().astype({"player": int})
    except Exception as e:
        print(f"  [{year} {POS[pos]}] OAA download failed: {type(e).__name__}", flush=True)
        return pd.DataFrame(columns=["player", "oaa"])
    out.to_csv(p, index=False)
    return out


def sis(year: int) -> pd.DataFrame:
    """SIS DRS file for a season, matched to our player ids by name: player_key, pos, drs (+ components)."""
    p = COMPARE / f"drs_{year}.csv"
    if not p.exists():
        return pd.DataFrame()
    from pipeline.compare_fielding import _col, norm
    d = pd.read_csv(p)
    nc, pc = _col(d, "player") or _col(d, "name"), _col(d, "pos")
    dc = _col(d, "drs", exclude=("rank", "/", "per")) or _col(d, "total")
    if not (nc and dc):
        print(f"  {p.name}: couldn't find name / DRS columns {list(d.columns)}", flush=True)
        return pd.DataFrame()
    out = pd.DataFrame({"key": d[nc].map(norm), "pos": d[pc].astype(str).str.upper() if pc else "",
                        "sis": pd.to_numeric(d[dc], errors="coerce")})
    for name, needles in (("sis_range", ("rpm",)), ("sis_range", ("range",)), ("sis_gfp", ("gfp",)),
                          ("sis_err", ("err",))):
        c = _col(d, *needles)
        if c and name not in out:
            out[name] = pd.to_numeric(d[c], errors="coerce")
    return out


def ours(year: int, names: dict) -> pd.DataFrame:
    from core import league_ref
    from models import fielding as fm
    from pipeline.compare_fielding import norm
    daily = league_ref.load_fielding(year)
    if daily.empty:
        return pd.DataFrame()
    rows = []
    for k in POS:
        spread = fm.grade_spread(daily, [k])
        for pid, v in fm.summarize(daily, "0000", "9999", [k]).items():
            if v["IQ_INN"] < MIN_INN:
                continue
            r = {"player": int(pid), "pos": POS[k], "pk": k, "inn": v["IQ_INN"], "drs": v["DRS"],
                 "range": fm._rate(v, "range")[0], "conv": fm._rate(v, "conv")[0], "n": v["_N"]}
            for sc in ("raw", "talent", "rank"):
                g = fm.grades(v, spread, GROUP[k], scale=sc)
                r[f"g_range_{sc}"] = g.get("RANGE_GRADE", g.get("RANGE", np.nan))
            rows.append(r)
    t = pd.DataFrame(rows)
    t["key"] = t["player"].map(lambda p: norm(names.get(p, "")))
    return t


def season_table(year: int, names: dict) -> pd.DataFrame:
    t = ours(year, names)
    if t.empty:
        return t
    o = pd.concat([oaa(year, k).assign(pk=k) for k in POS], ignore_index=True)
    t = t.merge(o, on=["player", "pk"], how="left")
    s = sis(year)
    if not s.empty:
        s = s[s["pos"].isin(POS.values()) | s["pos"].eq("")]
        if s["pos"].eq("").all():
            t = t.merge(s.drop(columns="pos").groupby("key").sum(), on="key", how="left")
        else:
            t = t.merge(s.groupby(["key", "pos"]).sum().reset_index(), on=["key", "pos"], how="left")
    per = 1200 / t["inn"]                                    # everything per a full season of innings
    t["drs_s"] = t["drs"] * per
    t["oaa_s"] = t["oaa"] * per
    if "sis" in t:
        t["sis_s"] = t["sis"] * per
    t["year"] = year
    return t


# ---------------------------------------------------------------------------- analysis
MEASURES = {"range": "OUR Range (outs/play)", "drs_s": "OUR DRS* / season", "oaa_s": "Statcast OAA / season",
            "sis_s": "SIS DRS / season"}


def agreement(t: pd.DataFrame) -> list[str]:
    L = ["", "1. SAME-SEASON AGREEMENT (r, players with 300+ innings at the position)", ""]
    have = [m for m in MEASURES if m in t and t[m].notna().sum() > 30]
    for g in ("IF", "OF"):
        s = t[t.pk.map(GROUP).eq(g)]
        L.append(f"  {g}: " + " | ".join(f"{MEASURES[a].split(' (')[0]} vs {MEASURES[b].split(' /')[0]} "
                                          f"{s[a].corr(s[b]):.2f}" for i, a in enumerate(have) for b in have[i + 1:]))
    L.append("")
    L.append(f"  {'POS':<5}" + "".join(f"{'ours Range vs ' + MEASURES[b].split(' /')[0]:>34}" for b in have[2:]))
    for k, name in POS.items():
        s = t[t.pk == k]
        L.append(f"  {name:<5}" + "".join(f"{s['range'].corr(s[b]):>34.2f}" for b in have[2:]))
    return L


def predict(t: pd.DataFrame) -> tuple[list[str], dict]:
    nx = t.copy()
    nx["year"] -= 1
    m = t.merge(nx, on=["player", "pk", "year"], suffixes=("", "_nx"))
    have = [k for k in MEASURES if k in t and m[k].notna().sum() > 30 and m[k + "_nx"].notna().sum() > 30]
    L = ["", "2. WHICH PREDICTS NEXT SEASON? (r of this season's number with next season's, same position, "
             f"{MIN_INN}+ innings both years)", "  Read across: a good measure predicts EVERY column, not just its own.",
         ""]
    best = {}
    for g in ("IF", "OF"):
        s = m[m.pk.map(GROUP).eq(g)]
        L.append(f"  {g}  ({len(s)} pairs)   next season ->" + "".join(f"{MEASURES[c].split(' /')[0][:22]:>24}"
                                                                   for c in have))
        for a in have:
            rs = [s[a].corr(s[c + "_nx"]) for c in have]
            best[(g, a)] = np.nanmean(rs)
            L.append(f"    {MEASURES[a]:<36}" + "".join(f"{x:>24.3f}" for x in rs) + f"   avg {np.nanmean(rs):.3f}")
        L.append("")
    return L, {"m": m, "have": have, "best": best}


def blend(m: pd.DataFrame, have: list[str]) -> list[str]:
    """Best Range = w * ours + (1 - w) * OAA (z-scored by position-season), by next-season prediction of the
    other public numbers (OAA, SIS) -- ours isn't allowed to grade its own homework."""
    L = ["", "3. BEST RANGE BLEND: w x OUR Range + (1 - w) x OAA, judged by how well it predicts next season's "
             "OAA and SIS DRS", ""]
    if "oaa_s" not in have:
        return L + ["  no OAA -- skipped"]
    z = lambda c: m.groupby(["year", "pk"])[c].transform(lambda x: (x - x.mean()) / x.std())
    zo, za = z("range"), z("oaa_s")
    targets = [c for c in ("oaa_s", "sis_s") if c in have]
    for g in ("IF", "OF"):
        msk = m.pk.map(GROUP).eq(g)
        row = []
        for w in np.arange(0, 1.01, 0.1):
            b = w * zo + (1 - w) * za
            row.append((w, np.nanmean([b[msk].corr(m.loc[msk, c + "_nx"]) for c in targets])))
        bw, br = max(row, key=lambda x: x[1])
        L.append(f"  {g}: " + " ".join(f"{w:.1f}:{r:.3f}" for w, r in row) + f"   -> best w = {bw:.1f} ({br:.3f})")
    return L


def scales(t: pd.DataFrame, names: dict) -> list[str]:
    last = t[t.year == t.year.max()]
    L = ["", f"4. WHICH 20-80 SCALE? {int(last.year.max())} Range grades, three ways (see models/fielding.py "
             "scale_factor)", "  raw = v20.0 (squeezed twice) | talent = vs true-talent spread | rank = regulars "
             "spread like the hitting grades", ""]
    for sc in ("raw", "talent", "rank"):
        g = last[f"g_range_{sc}"].dropna()
        L.append(f"  {sc:<7} middle 50% {g.quantile(.25):.0f}-{g.quantile(.75):.0f} | top {g.max():.0f} | "
                 f"bottom {g.min():.0f} | 60+: {int((g >= 60).sum())} players | 70+: {int((g >= 70).sum())}")
    L += ["", "  Top / bottom 5 by position (talent | rank):"]
    for k, name in POS.items():
        s = last[last.pk == k].sort_values("g_range_talent", ascending=False)
        if len(s) < 8:
            continue
        f = lambda r: f"{names.get(r.player, r.player)} {r.g_range_talent:.0f}|{r.g_range_rank:.0f}"
        L.append(f"  {name}: " + ", ".join(f(r) for r in s.head(5).itertuples()) + "   ...   "
                 + ", ".join(f(r) for r in s.tail(3).itertuples()))
    return L


def disagreements(t: pd.DataFrame, names: dict) -> list[str]:
    last = t[t.year == t.year.max()].copy()
    L = ["", f"5. BIGGEST DISAGREEMENTS, {int(last.year.max())} (rank within position, 1 = best)", ""]
    for other in ("oaa_s", "sis_s"):
        if other not in last or last[other].notna().sum() < 30:
            continue
        last["r_ours"] = last.groupby("pk")["drs_s"].rank(ascending=False)
        last["r_them"] = last.groupby("pk")[other].rank(ascending=False)
        last["gap"] = last["r_ours"] - last["r_them"]
        lab = MEASURES[other].split(" /")[0]
        for title, s in (("we like more", last.nsmallest(6, "gap")), ("they like more", last.nlargest(6, "gap"))):
            L.append(f"  vs {lab}, {title}: " + " | ".join(
                f"{names.get(r.player, r.player)} {r.pos} (ours #{r.r_ours:.0f}, theirs #{r.r_them:.0f})"
                for r in s.itertuples()))
        L.append("")
    return L


def main(argv: list[str]) -> None:
    from core import mlb_api
    from pipeline.decision_study import push
    from pipeline.train_decisions import latest_full_season
    t0 = time.time()
    years = list(range(FIRST, latest_full_season() + 1))
    tabs, names = [], {}
    for y in years:
        print(f"[{y}] building…", flush=True)
        try:
            from core import league_ref
            d = league_ref.load_fielding(y)
            if not d.empty:
                names.update(mlb_api.get_people(sorted(d["player"].astype(int).unique().tolist())))
        except Exception:
            pass
        t = season_table(y, names)
        if not t.empty:
            tabs.append(t)
    t = pd.concat(tabs, ignore_index=True)
    sis_years = sorted(t.loc[t.get("sis", pd.Series(dtype=float)).notna(), "year"].unique()) if "sis" in t else []
    L = ["IQ FIELDING COMPARISON STUDY -- ours (DRS*) vs Statcast OAA vs SIS DRS",
         f"seasons {years[0]}-{years[-1]} | SIS DRS files: {', '.join(map(str, sis_years)) or 'none yet'}"]
    L += agreement(t)
    pl, info = predict(t)
    L += pl + blend(info["m"], info["have"]) + scales(t, names) + disagreements(t, names)
    L += ["", f"done in {(time.time() - t0) / 60:.0f} min"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)
    if "--no-push" not in argv:
        push([REPORT] + sorted(COMPARE.glob("oaa_*.csv")), "Fielding comparison study results")
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
