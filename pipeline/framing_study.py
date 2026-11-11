"""Framing study (analysis only -- nothing on the site changes). Run it when you step away:

    caffeinate -i python -m pipeline.framing_study                  # 2015 -> latest season
    (a second run reuses the saved umpires + takes, so only the analysis re-runs)

Two questions:
  1. Which framing number is the most accurate -- measures the catcher (not the pitcher / umpire / luck) and
     predicts what he does next?
  2. Which zone matters most -- where are framing runs actually won and lost?

Steps
  * Home plate umpire for every game (MLB stats API boxscores, one call per game, saved data/raw/umps_<year>.csv.gz)
  * Every taken pitch: league-umpire strike chance (IQ Swing Decisions called-strike model), what a strike is worth
    in that count, zone, catcher / pitcher / batter / umpire  ->  data/framing_study/takes_<year>.csv.gz
  * Every metric for every catcher-season, plus odd / even games and 1st / 2nd half
  * "Adjusted" versions: one model per season that splits each pitch's extra strike between the catcher, the
    pitcher, the umpire and the batter (ridge regression = the mixed-model idea Baseball Prospectus uses)
  * Tests by era (2015-2022 and 2023+, no carryover): split-half r, pitches needed to trust it, 1st half -> 2nd half,
    this season -> next season, agreement with Savant's framing runs
  * SIS Strike Zone Runs Saved too, if you save it as data/compare/sis_framing.csv (player, season, strike zone runs)
  * Zones (catcher's view, every catcher right-handed: glove = his left, arm = his right): heart, top, bottom,
    glove, arm, 4 corners, waste -- also split by batter hand

Output: models/artifacts/framing_study_report.txt  (+ framing_study_catchers.csv, every catcher-season and metric)
"""
from __future__ import annotations

import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
STUDY_DIR = ROOT / "data" / "framing_study"
ART = ROOT / "models" / "artifacts"
REPORT = ART / "framing_study_report.txt"
TAKEN = {"called_strike", "ball"}
FLAT = 0.125
SPLIT = 2023
MIN_SEASON = 3000          # takes, for season-level tests
MIN_HALF = 1500            # takes in each half, for split-half tests
EDGE_IN, EDGE_OUT = 0.67, 1.33
ALPHAS = (500, 2000, 5000)
ZONE_ORDER = ["heart", "top", "bottom", "glove", "arm", "top-glove", "top-arm", "bottom-glove", "bottom-arm", "waste"]
WORKERS = 6


# ----------------------------------------------------------------------------------------------- umpires
def season_umps(year: int) -> pd.DataFrame:
    from core import mlb_api
    from pipeline import build_league_reference as bl
    path = bl.RAW_DIR / f"umps_{year}.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    games = [g["gamePk"] for g in mlb_api.get_season_games(year)
             if g.get("gameType") == "R" and g.get("status", {}).get("abstractGameState") == "Final"]
    print(f"[{year}] home plate umpires for {len(games):,} games…", flush=True)
    rows, failed, t0 = [], [], time.time()

    def one(pk):
        for attempt in range(4):
            try:
                box = mlb_api.get_boxscore(pk)
                for o in box.get("officials", []):
                    if o.get("officialType") == "Home Plate":
                        return {"game_pk": pk, "ump": o["official"]["id"], "ump_name": o["official"].get("fullName")}
                return {"game_pk": pk, "ump": np.nan, "ump_name": None}
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(pk)

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(one, pk): pk for pk in games}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception:
                failed.append(futs[f])
            if i % 500 == 0:
                print(f"    {i:,}/{len(games):,} games ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["game_pk", "ump", "ump_name"])
    if failed:
        print(f"[{year}] {len(failed)} games failed -- using what we have, not saving (run again to retry)", flush=True)
        return df
    bl.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    print(f"[{year}] umpires saved ({time.time() - t0:.0f}s)", flush=True)
    return df


# ----------------------------------------------------------------------------------------------- takes
def zone(xn: pd.Series, zn: pd.Series) -> np.ndarray:
    """Catcher's view, plate_x > 0 = catcher's right = ARM side (every catcher right-handed)."""
    ax, az = xn.abs(), zn.abs()
    r = np.maximum(ax, az)
    side = np.where(xn < 0, "glove", "arm")
    vert = np.where(zn > 0, "top", "bottom")
    cx, cz = ax >= EDGE_IN, az >= EDGE_IN
    corner = np.char.add(np.char.add(vert.astype(str), "-"), side.astype(str))
    return np.select([r < EDGE_IN, r >= EDGE_OUT, cx & cz, cz, cx], ["heart", "waste", corner, vert, side], "heart")


def season_takes(year: int) -> pd.DataFrame:
    path = STUDY_DIR / f"takes_{year}.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    from models import decisions_iq as dq
    from pipeline import build_league_reference as bl
    if dq.load() is None:
        raise SystemExit("Needs the IQ Swing Decisions model (called-strike chance) -- train_decisions first.")
    raw = bl.pull_season(year)
    if "game_type" in raw:
        raw = raw[raw["game_type"].astype(str).eq("R")]
    d = raw[raw["description"].isin(TAKEN)].dropna(
        subset=["plate_x", "plate_z", "sz_top", "sz_bot", "fielder_2", "pitcher", "batter"])
    del raw
    d = d.reset_index(drop=True)
    p = dq.called_strike_chance(d)
    val = dq.strike_value(d)
    xn = d["plate_x"] / (17 / 24)
    zn = (d["plate_z"] - (d["sz_top"] + d["sz_bot"]) / 2) / ((d["sz_top"] - d["sz_bot"]) / 2)
    umps = season_umps(year).drop_duplicates("game_pk").set_index("game_pk")["ump"]
    t = pd.DataFrame({"year": year, "game_pk": d["game_pk"].astype(int),
                      "game_date": pd.to_datetime(d["game_date"]).dt.date.astype(str),
                      "catcher": d["fielder_2"].astype(int), "pitcher": d["pitcher"].astype(int),
                      "batter": d["batter"].astype(int), "ump": d["game_pk"].map(umps),
                      "stand": d["stand"], "y": d["description"].eq("called_strike").astype(np.int8),
                      "p": p.astype(np.float32), "val": val.fillna(FLAT).astype(np.float32),
                      "zone": zone(xn, zn), "shadow": ((np.maximum(xn.abs(), zn.abs()) >= EDGE_IN)
                                                       & (np.maximum(xn.abs(), zn.abs()) < EDGE_OUT)).astype(np.int8)})
    t = t.dropna(subset=["p"])
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    t.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)
    print(f"[{year}] {len(t):,} takes saved, umpire known on {t['ump'].notna().mean():.1%}", flush=True)
    return t


# ----------------------------------------------------------------------------------------------- metrics
def adjusted(t: pd.DataFrame, alpha: float, groups=("catcher", "pitcher", "ump", "batter"),
             count_value: bool = False) -> pd.Series:
    """Catcher's share of the extra strikes per take once pitcher / umpire / batter get their share (ridge)."""
    from scipy import sparse
    from sklearn.linear_model import Ridge
    target = (t["y"] - t["p"]).to_numpy(np.float64)
    scale = 1.0
    if count_value:
        scale = float(t["val"].mean())
        target = target * t["val"].to_numpy(np.float64) / scale
    target = target - target.mean()
    blocks, cat_codes = [], None
    for g in groups:
        col = t[g].fillna(-1).astype(np.int64) if g != "batter" else t[g].astype(np.int64)
        codes, uniq = pd.factorize(col)
        blocks.append(sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)),
                                        shape=(len(codes), len(uniq))))
        if g == "catcher":
            cat_codes = uniq
    X = sparse.hstack(blocks).tocsr()
    m = Ridge(alpha=alpha, fit_intercept=False, solver="sparse_cg", max_iter=2000, tol=1e-6).fit(X, target)
    return pd.Series(m.coef_[:len(cat_codes)] * scale, index=cat_codes)


METRICS = {   # name -> (how to make a per-take rate from the stats table, is it in runs?)
    "Extra Strikes (all takes)": "xs / n",
    "Extra Strikes (shadow only)": "xs_sh / n_sh",
    "Extra Strikes / Game": "xs / games",
    "SLAA+": "100 * strikes / exp",
    "Framing Runs (count value)": "runs_c / n",
    "Framing Runs (flat 0.125)": "xs * 0.125 / n",
    "Adjusted (a=500)": "adj500",
    "Adjusted (a=2000)": "adj2000",
    "Adjusted (a=5000)": "adj5000",
    "Adjusted, no umpire (a=2000)": "adjnoump",
    "Adjusted, count value (a=2000)": "adjc",
}


def catcher_table(t: pd.DataFrame, with_adj: bool = True) -> pd.DataFrame:
    t = t.assign(xs=t["y"] - t["p"], runs_c=(t["y"] - t["p"]) * t["val"])
    g = t.groupby("catcher")
    s = pd.DataFrame({"n": g.size(), "games": g["game_pk"].nunique(), "strikes": g["y"].sum(), "exp": g["p"].sum(),
                      "xs": g["xs"].sum(), "runs_c": g["runs_c"].sum()})
    sh = t[t["shadow"] == 1].groupby("catcher")
    s["n_sh"], s["xs_sh"] = sh.size(), sh["xs"].sum()
    if with_adj:
        for a in ALPHAS:
            s[f"adj{a}"] = adjusted(t, a)
        s["adjnoump"] = adjusted(t, 2000, ("catcher", "pitcher", "batter"))
        s["adjc"] = adjusted(t, 2000, count_value=True)
    return s


def rates(s: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=s.index)
    for name, expr in METRICS.items():
        if expr in s:
            out[name] = s[expr]
        else:
            try:
                out[name] = s.eval(expr)
            except Exception:
                out[name] = np.nan
    return out


def season_tables(t: pd.DataFrame) -> dict:
    """Stats per catcher for the whole season and each split."""
    mid = t["game_date"].sort_values().iloc[len(t) // 2]
    splits = {"all": t, "odd": t[t.game_pk % 2 == 1], "even": t[t.game_pk % 2 == 0],
              "h1": t[t.game_date <= mid], "h2": t[t.game_date > mid]}
    return {k: catcher_table(v) for k, v in splits.items()}


# ----------------------------------------------------------------------------------------------- tests
def _r(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    return (float(a[ok].corr(b[ok])) if ok.sum() >= 10 else np.nan), int(ok.sum())


def savant_framing(year: int) -> pd.Series:
    """Savant catcher framing runs PER CALLED PITCH (player id -> value); empty if it can't be downloaded."""
    try:
        import pybaseball as pb
        f = pb.statcast_catcher_framing(year, min_called_p=1)
    except Exception:
        try:
            import pybaseball as pb
            f = pb.statcast_catcher_framing(year)
        except Exception:
            return pd.Series(dtype=float)
    cols = {c.lower(): c for c in f.columns}
    idc = cols.get("player_id") or cols.get("id")
    rc = cols.get("runs_extra_strikes") or next((c for lc, c in cols.items() if "run" in lc), None)
    nc = cols.get("n_called_pitches") or next((c for lc, c in cols.items() if "called" in lc or lc.startswith("n_")), None)
    if not (idc and rc and nc):
        return pd.Series(dtype=float)
    f = f[pd.to_numeric(f[idc], errors="coerce").notna()]
    n = pd.to_numeric(f[nc], errors="coerce")
    return pd.Series((pd.to_numeric(f[rc], errors="coerce") / n).values,
                     index=pd.to_numeric(f[idc]).astype(int).values).dropna()


def sis_framing(catcher_ids: list[int], years: list[int]) -> dict:
    """SIS Strike Zone Runs Saved (the framing piece of catcher DRS), if saved as data/compare/sis_framing*.csv.
    Needs a player name column and a strike zone runs column; a season column (or the year in the file name,
    e.g. sis_framing_2025.csv) -- otherwise it's taken as the latest season. -> {year: Series(player id -> runs)}"""
    import re
    from pipeline.compare_fielding import COMPARE_DIR, _col, norm
    files = sorted(COMPARE_DIR.glob("sis_framing*.csv"))
    if not files:
        return {}
    try:
        from core import mlb_api
        names = mlb_api.get_people(sorted(set(int(c) for c in catcher_ids)))
    except Exception:
        return {}
    by_name = {}
    for pid, nm in names.items():
        by_name.setdefault(norm(nm), pid)
    out: dict = {}
    for f in files:
        d = pd.read_csv(f)
        nc = _col(d, "player") or _col(d, "name")
        rc = (_col(d, "strike", "zone") or _col(d, "rsz") or _col(d, "framing") or _col(d, "sz", exclude=("season",)))
        yc = _col(d, "season") or _col(d, "year")
        if not (nc and rc):
            print(f"  {f.name}: couldn't find player / strike zone runs columns ({list(d.columns)})", flush=True)
            continue
        m = re.search(r"(20\d\d)", f.stem)
        d = d.assign(_y=pd.to_numeric(d[yc], errors="coerce") if yc else (int(m.group(1)) if m else years[-1]),
                     _id=d[nc].map(lambda x: by_name.get(norm(x))), _r=pd.to_numeric(d[rc], errors="coerce"))
        d = d.dropna(subset=["_y", "_id", "_r"])
        for y, g in d.groupby("_y"):
            out[int(y)] = pd.concat([out.get(int(y), pd.Series(dtype=float)),
                                     g.groupby("_id")["_r"].sum().rename(None)])
        print(f"  SIS framing: {f.name} -> {len(d)} catcher-seasons matched", flush=True)
    return out


def metric_tests(tabs: dict, years: list[int], sav: dict, sis: dict | None = None) -> pd.DataFrame:
    rows = []
    for name in METRICS:
        sh_r, sh_n, sh_sz, hh_own, hh_act, yy_own, yy_act, sv, si = [], [], [], [], [], [], [], [], []
        for y in years:
            T = tabs[y]
            R = {k: rates(v) for k, v in T.items()}
            ok = (T["odd"]["n"] >= MIN_HALF).reindex(T["all"].index, fill_value=False) & \
                 (T["even"]["n"] >= MIN_HALF).reindex(T["all"].index, fill_value=False)
            idx = ok[ok].index
            sh_r.append(R["odd"].loc[idx, name].corr(R["even"].loc[idx, name])) if len(idx) >= 10 else None
            sh_n.append(len(idx))
            sh_sz.append(float(np.mean((T["odd"].loc[idx, "n"] + T["even"].loc[idx, "n"]) / 2)) if len(idx) else np.nan)
            okh = (T["h1"]["n"] >= MIN_HALF).reindex(T["all"].index, fill_value=False) & \
                  (T["h2"]["n"] >= MIN_HALF).reindex(T["all"].index, fill_value=False)
            ih = okh[okh].index
            if len(ih) >= 10:
                hh_own.append(R["h1"].loc[ih, name].corr(R["h2"].loc[ih, name]))
                hh_act.append(R["h1"].loc[ih, name].corr(R["h2"].loc[ih, "Framing Runs (count value)"]))
            nxt = y + 1
            if nxt in tabs:
                a, b = T["all"], tabs[nxt]["all"]
                both = a.index[(a["n"] >= MIN_SEASON)].intersection(b.index[b["n"] >= MIN_SEASON])
                if len(both) >= 10:
                    ra, rb = rates(a).loc[both], rates(b).loc[both]
                    yy_own.append((ra[name].corr(rb[name]), len(both)))
                    yy_act.append((ra[name].corr(rb["Framing Runs (count value)"]), len(both)))
            if len(sav.get(y, [])):
                full = T["all"][T["all"]["n"] >= MIN_SEASON]
                sv.append(_r(rates(full)[name], sav[y].reindex(full.index)))
            if sis and len(sis.get(y, [])):
                full = T["all"][T["all"]["n"] >= MIN_SEASON]
                sis_rate = sis[y].reindex(full.index) / full["n"]          # SIS runs per take
                si.append(_r(rates(full)[name], sis_rate))
        wavg = lambda xs: (np.nansum([r * n for r, n in xs]) / np.nansum([n for r, n in xs if r == r])
                           if xs else np.nan)
        r_sh = float(np.nanmean(sh_r)) if sh_r else np.nan
        half = float(np.nanmean(sh_sz)) if sh_sz else np.nan
        need = half * (0.7 / 0.3) * (1 - r_sh) / r_sh if r_sh and r_sh > 0 else np.inf
        rows.append({"metric": name, "split_half_r": r_sh, "takes_for_r70": need,
                     "h1_to_h2_own": float(np.nanmean(hh_own)) if hh_own else np.nan,
                     "h1_to_h2_runs": float(np.nanmean(hh_act)) if hh_act else np.nan,
                     "next_yr_own": wavg(yy_own), "next_yr_runs": wavg(yy_act), "savant_r": wavg(sv), "sis_r": wavg(si),
                     "pairs": int(sum(n for _, n in yy_own))})
    return pd.DataFrame(rows).set_index("metric")


# ----------------------------------------------------------------------------------------------- zones
def zone_tables(t: pd.DataFrame, key: str) -> dict:
    t = t.assign(xs=t["y"] - t["p"], runs=(t["y"] - t["p"]) * t["val"], w=t["p"] * (1 - t["p"]))
    out = {}
    for split, d in {"all": t, "odd": t[t.game_pk % 2 == 1], "even": t[t.game_pk % 2 == 0]}.items():
        g = d.groupby(["catcher", key]).agg(n=("xs", "size"), xs=("xs", "sum"), runs=("runs", "sum"),
                                            w=("w", "sum"))
        tot = d.groupby("catcher").agg(N=("xs", "size"), R=("runs", "sum"))
        out[split] = (g, tot)
    vw = t.groupby(key).apply(lambda z: float(np.average(z["val"], weights=z["w"] + 1e-9)))
    out["value"] = vw
    return out


def zone_report(ztabs: list[dict], ztabs_next: list[tuple[dict, dict]], label: str, order: list[str]) -> list[str]:
    """Reliability, true spread, runs at stake, share of real framing differences, next-year weight."""
    rel, sd_rows, share_num, n_med, cov_tot, obs_share = {}, {}, {}, {}, [], {}
    per_zone_odd, per_zone_even, tot_odd, tot_even = {}, {}, [], []
    a_all, T_all = {}, []
    vals = pd.concat([z["value"] for z in ztabs], axis=1).mean(axis=1)
    for z in ztabs:
        (go, to), (ge, te), (ga, ta) = z["odd"], z["even"], z["all"]
        q = ta.index[ta["N"] >= MIN_SEASON]
        q = q.intersection(to.index[to["N"] >= MIN_HALF]).intersection(te.index[te["N"] >= MIN_HALF])
        To, Te = (to.loc[q, "R"] / to.loc[q, "N"]), (te.loc[q, "R"] / te.loc[q, "N"])
        tot_odd.append(To)
        tot_even.append(Te)
        T_all.append(ta.loc[q, "R"] / ta.loc[q, "N"])
        for zn in order:
            def piece(g, tt, col):
                s = g.xs(zn, level=1)[col] if zn in g.index.get_level_values(1) else pd.Series(dtype=float)
                return s.reindex(q).fillna(0)
            no, ne = piece(go, to, "n"), piece(ge, te, "n")
            ro = piece(go, to, "xs") / no.replace(0, np.nan)
            re_ = piece(ge, te, "xs") / ne.replace(0, np.nan)
            rel.setdefault(zn, []).append((ro, re_))
            per_zone_odd.setdefault(zn, []).append(piece(go, to, "runs") / to.loc[q, "N"])
            per_zone_even.setdefault(zn, []).append(piece(ge, te, "runs") / te.loc[q, "N"])
            a_all.setdefault(zn, []).append(piece(ga, ta, "runs") / ta.loc[q, "N"])
            n_med.setdefault(zn, []).append(piece(ga, ta, "n"))
    To, Te, Ta = pd.concat(tot_odd), pd.concat(tot_even), pd.concat(T_all)
    true_tot = float(np.cov(To.values, Te.values)[0, 1])
    rows = []
    for zn in order:
        ro = pd.concat([a for a, _ in rel[zn]])
        re_ = pd.concat([b for _, b in rel[zn]])
        ok = ro.notna() & re_.notna()
        r = float(ro[ok].corr(re_[ok])) if ok.sum() >= 10 else np.nan
        true_var = float(np.cov(ro[ok].values, re_[ok].values)[0, 1]) if ok.sum() >= 10 else np.nan
        sd = np.sqrt(true_var) if true_var and true_var > 0 else 0.0
        n_typ = float(pd.concat(n_med[zn]).median())
        stake = 2 * 1.2816 * sd * n_typ * float(vals.get(zn, FLAT))
        zo, ze = pd.concat(per_zone_odd[zn]), pd.concat(per_zone_even[zn])
        skill = (np.cov(zo.values, Te.values)[0, 1] + np.cov(ze.values, To.values)[0, 1]) / 2 / true_tot \
            if true_tot > 0 else np.nan
        za = pd.concat(a_all[zn])
        obs = float(np.cov(za.values, Ta.values)[0, 1] / np.var(Ta.values, ddof=1))
        rows.append((zn, n_typ, float(vals.get(zn, np.nan)), r, sd * 1000, stake, skill, obs))
    L = [f"{label}", f"  {'ZONE':<22}{'TAKES/SZN':>10}{'RUNS/K':>8}{'REL r':>7}{'SKILL SD':>9}"
                     f"{'RUNS AT STAKE':>14}{'SKILL SHARE':>12}{'RAW SHARE':>10}"]
    for zn, n_typ, v, r, sd, stake, skill, obs in sorted(rows, key=lambda x: -(x[6] if x[6] == x[6] else -9)):
        L.append(f"  {zn:<22}{n_typ:>10,.0f}{v:>8.3f}{r:>7.2f}{sd:>9.1f}{stake:>14.1f}{skill:>12.0%}{obs:>10.0%}")
    # next-season weights
    X, Y = [], []
    for (z, zn_next) in ztabs_next:
        ga, ta = z["all"]
        gb, tb = zn_next["all"]
        both = ta.index[ta["N"] >= MIN_SEASON].intersection(tb.index[tb["N"] >= MIN_SEASON])
        if len(both) < 10:
            continue
        cols = {}
        for zn in order:
            s = ga.xs(zn, level=1)["runs"] if zn in ga.index.get_level_values(1) else pd.Series(dtype=float)
            cols[zn] = s.reindex(both).fillna(0) / ta.loc[both, "N"]
        X.append(pd.DataFrame(cols))
        Y.append(tb.loc[both, "R"] / tb.loc[both, "N"])
    if X:
        X, Y = pd.concat(X), pd.concat(Y)
        A = np.column_stack([np.ones(len(X)), X.values])
        coef, *_ = np.linalg.lstsq(A, Y.values, rcond=None)
        r2 = 1 - np.var(Y.values - A @ coef) / np.var(Y.values)
        L += ["", f"  Predicting NEXT season's framing runs from this season's zones ({len(X)} catcher pairs, "
                  f"R^2 = {r2:.2f}):", f"  {'ZONE':<22}{'STD WEIGHT':>11}{'R^2 LOST IF DROPPED':>21}"]
        res = []
        for i, zn in enumerate(order):
            beta = coef[i + 1] * X[zn].std() / Y.std()
            A2 = np.delete(A, i + 1, axis=1)
            c2, *_ = np.linalg.lstsq(A2, Y.values, rcond=None)
            res.append((zn, beta, r2 - (1 - np.var(Y.values - A2 @ c2) / np.var(Y.values))))
        for zn, beta, drop in sorted(res, key=lambda x: -x[2]):
            L.append(f"  {zn:<22}{beta:>+11.2f}{drop:>21.3f}")
    return L


# ----------------------------------------------------------------------------------------------- report
def value_section(takes: dict, eras) -> list[str]:
    L = ["", "WHAT IS A STOLEN STRIKE WORTH? (flat 0.125 vs the count)",
         "  Runs a strike instead of a ball is worth, from our run expectancy by count. 'Where framing happens' =",
         "  weighted by p x (1 - p), i.e. the close pitches where a catcher can actually change the call."]
    for title, ys in eras:
        t = pd.concat([takes[y][["p", "val", "y"]] for y in ys])
        w = t["p"] * (1 - t["p"])
        L.append(f"  {title}:  every take {t['val'].mean():.3f}  |  where framing happens "
                 f"{np.average(t['val'], weights=w):.3f}  |  flat {FLAT}")
        by = t.assign(w=w).groupby(pd.cut(t["val"], [0, .08, .12, .18, .3, 1]), observed=True)["w"].sum()
        L.append("    share of framing chances by strike value: " +
                 "  ".join(f"{str(k)}: {v / by.sum():.0%}" for k, v in by.items()))
    return L


def main(argv: list[str]) -> None:
    from pipeline.train_decisions import latest_full_season
    t0 = time.time()
    since = int(argv[argv.index("--since") + 1]) if "--since" in argv else 2015
    years = list(range(since, latest_full_season() + 1))
    takes, tabs, sav, ztab, zhtab = {}, {}, {}, {}, {}
    for y in years:
        t = season_takes(y)
        t["stand"] = t["stand"].astype(str)
        print(f"[{y}] metrics + adjusted models…", flush=True)
        tabs[y] = season_tables(t)
        sav[y] = savant_framing(y)
        ztab[y] = zone_tables(t, "zone")
        t["zh"] = np.where(t["zone"].isin(["heart", "waste"]), t["zone"], t["zone"] + " vs " + t["stand"] + "HB")
        zhtab[y] = zone_tables(t, "zh")
        takes[y] = t[["p", "val", "y"]]
        print(f"[{y}] done ({(time.time() - t0) / 60:.0f} min)", flush=True)
    sis = sis_framing(sorted(set().union(*[tabs[y]["all"].index for y in years])), years)
    if not sis:
        print("  (no SIS framing file -- save data/compare/sis_framing.csv to add that column)", flush=True)
    eras = [(f"2015-{SPLIT - 1}", [y for y in years if y < SPLIT]), (f"{SPLIT}+", [y for y in years if y >= SPLIT])]
    eras = [(n, ys) for n, ys in eras if ys]

    L = ["IQ FRAMING STUDY -- which framing number is most accurate, and which zones matter most",
         f"Seasons {years[0]}-{years[-1]}, regular season, every taken pitch. Strike chance = league-average umpire "
         "(IQ Swing Decisions called-strike model).",
         f"Season tests use catchers with {MIN_SEASON:,}+ takes; split-half tests {MIN_HALF:,}+ takes in each half.",
         "", "HOW TO READ",
         "  split-half r    odd games vs even games, same season: is it measuring something real? (higher = better)",
         "  takes for .70   how many taken pitches before the number is 70% signal",
         "  1H->2H          1st half predicting the 2nd half (own = same metric; runs = actual count-valued runs)",
         "  next yr         this season predicting next season (same era only)",
         "  savant r        agreement with Savant's catcher framing runs per called pitch",
         "  SIS r           agreement with SIS Strike Zone Runs Saved (framing piece of catcher DRS), per take",
         "                  (Savant / SIS are other estimates, not the truth -- next season is the main test)",
         "  Adjusted        catcher's share after the pitcher, umpire and batter get theirs (a = how hard it pulls",
         "                  everyone toward average; bigger = more cautious)"]
    for title, ys in eras:
        res = metric_tests(tabs, ys, sav, sis)
        res = res.sort_values("next_yr_runs", ascending=False)
        L += ["", "=" * 110, f"WHICH METRIC IS MOST ACCURATE -- {title}  ({res['pairs'].max()} catcher season-pairs)",
              f"  {'METRIC':<34}{'SPLIT r':>8}{'TAKES FOR .70':>14}{'1H->2H own':>11}{'1H->2H runs':>12}"
              f"{'NEXT YR own':>12}{'NEXT YR runs':>13}{'SAVANT r':>9}{'SIS r':>7}"]
        for m, r in res.iterrows():
            need = f"{r.takes_for_r70:,.0f}" if np.isfinite(r.takes_for_r70) else "never"
            L.append(f"  {m:<34}{r.split_half_r:>8.2f}{need:>14}{r.h1_to_h2_own:>11.2f}{r.h1_to_h2_runs:>12.2f}"
                     f"{r.next_yr_own:>12.2f}{r.next_yr_runs:>13.2f}{r.savant_r:>9.2f}{r.sis_r:>7.2f}")
        best = res["next_yr_runs"].idxmax()
        L.append(f"  -> Best at predicting next season's framing runs: {best}")
        L += ["", f"  Split-half r by season ({title}):"]
        for y in ys:
            T = tabs[y]
            idx = T["odd"].index[T["odd"]["n"] >= MIN_HALF].intersection(T["even"].index[T["even"]["n"] >= MIN_HALF])
            ro, re_ = rates(T["odd"]).loc[idx], rates(T["even"]).loc[idx]
            L.append(f"    {y}: raw extra strikes {ro['Extra Strikes (all takes)'].corr(re_['Extra Strikes (all takes)']):.2f}"
                     f"   adjusted {ro['Adjusted (a=2000)'].corr(re_['Adjusted (a=2000)']):.2f}   ({len(idx)} catchers)")
        L += ["", "-" * 110, f"WHICH ZONES MATTER MOST -- {title}",
              "  TAKES/SZN       taken pitches there for a typical starting catcher",
              "  RUNS/K          what a stolen strike is worth there (count mix), runs",
              "  REL r           odd vs even games: is catcher skill there real?",
              "  SKILL SD        true spread between catchers, extra strikes per 1,000 takes there",
              "  RUNS AT STAKE   top-10% vs bottom-10% catcher there over a season, runs",
              "  SKILL SHARE     share of the REAL (repeatable) differences in total framing runs from this zone",
              "  RAW SHARE       share of the observed differences (includes luck)", ""]
        nxt = [(ztab[y], ztab[y + 1]) for y in ys if y + 1 in ys]
        L += zone_report([ztab[y] for y in ys], nxt, "  By zone (catcher's view: glove = his left, arm = his right)",
                         ZONE_ORDER)
        zh_order = ["heart", "waste"] + [f"{z} vs {h}HB" for z in ZONE_ORDER[1:-1] for h in ("R", "L")]
        nxt_h = [(zhtab[y], zhtab[y + 1]) for y in ys if y + 1 in ys]
        L += [""] + zone_report([zhtab[y] for y in ys], nxt_h,
                                "  By zone and batter hand (glove side = inside to a righty, away to a lefty)",
                                zh_order)
    L += value_section(takes, eras)

    # names for a last-season eye test
    last = years[-1]
    s = tabs[last]["all"]
    s = s[s["n"] >= MIN_SEASON].assign(raw_runs=lambda x: x["runs_c"], adj_runs=lambda x: x["adjc"] * x["n"])
    try:
        from core import mlb_api
        names = mlb_api.get_people(s.index.astype(int).tolist())
    except Exception:
        names = {}
    L += ["", f"{last} EYE TEST -- raw framing runs vs adjusted (catcher's share only), count valued"]
    for col, title in (("raw_runs", "raw"), ("adj_runs", "adjusted")):
        top = s.sort_values(col, ascending=False)
        L.append(f"  {title:>8}: " + " | ".join(f"{names.get(int(i), i)} {r[col]:+.1f}" for i, r in top.head(6).iterrows())
                 + "  ...  " + " | ".join(f"{names.get(int(i), i)} {r[col]:+.1f}" for i, r in top.tail(3).iterrows()))
    if last >= 2026:
        L += ["", "NOTE: 2026 has ABS challenges -- an overturned call shows as the final call, so a stolen strike that",
              "gets challenged away counts as a ball. Watch whether 2026 split-half r drops vs earlier seasons."]

    out = "\n".join(L)
    ART.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(out + "\n")
    rows = []
    for y in years:
        a = tabs[y]["all"].join(rates(tabs[y]["all"]), rsuffix="_rate")
        rows.append(a.assign(year=y))
    pd.concat(rows).reset_index().rename(columns={"index": "catcher"}).to_csv(ART / "framing_study_catchers.csv",
                                                                               index=False)
    print("\n" + out)
    print(f"\nSaved {REPORT.relative_to(ROOT)} + framing_study_catchers.csv in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main(sys.argv[1:])
