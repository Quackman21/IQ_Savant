"""How does our DRS line up with the public numbers? Season level, by position.

    python -m pipeline.compare_fielding                 # latest finished season
    python -m pipeline.compare_fielding --season 2025

Compares our full-season DRS (from the saved fielding file, built by  python -m pipeline.update ) with:
  * Savant Outs Above Average and Fielding Run Value (downloaded, by position)
  * Savant catcher blocking (Blocks Above Average) -- ours uses the same method, so it should line up closely
  * Sports Info Solutions DRS -- if you save their leaderboard as  data/compare/drs_<season>.csv
    (copy the table from fieldingbible.com into a spreadsheet and save as CSV; it needs a player name column,
    a position column and a total DRS column -- the names are matched to ours automatically)

Prints and saves models/artifacts/fielding_compare_<season>.txt:
  r      correlation (does the order of players agree?)
  SLOPE  how many of OUR runs per 1 of theirs (1.00 = same scale; 0.5 = ours are half the size)
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COMPARE_DIR = ROOT / "data" / "compare"
POS_NAME = {3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF", 2: "C"}
MIN_INN = 400


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = s.replace(".", "").replace(",", " ").replace(" jr", "").replace(" sr", "").replace(" ii", "")
    return " ".join(s.split())


def _col(df: pd.DataFrame, *needles, exclude=()) -> str | None:
    for c in df.columns:
        lc = str(c).lower()
        if all(n in lc for n in needles) and not any(x in lc for x in exclude):
            return c
    return None


def ours(season: int) -> pd.DataFrame:
    """Our full-season DRS per player per position (and catchers: framing + blocking + throwing)."""
    from core import league_ref
    from models import catching, fielding, framing
    daily = league_ref.load_fielding(season)
    if daily.empty:
        raise SystemExit(f"No fielding file for {season} -- run  python -m pipeline.update  first.")
    rows = []
    for k, name in POS_NAME.items():
        if k == 2:
            c = catching.summarize(daily, "0000", "9999")
            fr = framing.summarize(league_ref.load_catching(season), "0000", "9999")
            inn = fielding.summarize(daily, "0000", "9999", [2])
            for pid, v in c.items():
                f_rs = fr.get(pid, {}).get("F_RS", np.nan)
                drs = np.nansum([f_rs, v.get("B_RS", np.nan), v.get("T_RS", np.nan)])
                rows.append({"player": pid, "pos": name, "drs": drs, "inn": inn.get(pid, {}).get("IQ_INN", 0),
                             "baa": v.get("B_BAA", np.nan), "b_rs": v.get("B_RS", np.nan),
                             "b_rs_sc": v.get("B_RS_SC", np.nan), "opps": v.get("B_OPPS", np.nan)})
            continue
        for pid, v in fielding.summarize(daily, "0000", "9999", [k]).items():
            rows.append({"player": pid, "pos": name, "drs": v["DRS"], "inn": v["IQ_INN"], "opa": v["OPA"]})
    return pd.DataFrame(rows)


def _stats(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    ok = a.notna() & b.notna()
    if ok.sum() < 8:
        return np.nan, np.nan, int(ok.sum())
    slope = float(np.polyfit(b[ok], a[ok], 1)[0])
    return float(a[ok].corr(b[ok])), slope, int(ok.sum())


def savant_compare(o: pd.DataFrame, season: int) -> list[str]:
    from core import savant
    L = ["Savant (by position, players with "
         f"{MIN_INN}+ of our innings):", f"  {'POS':<5}{'vs OAA r':>10}{'vs FRV r':>10}{'SLOPE vs FRV':>14}{'N':>6}"]
    for k, name in POS_NAME.items():
        mine = o[(o.pos == name) & (o.inn >= MIN_INN)].set_index("player")["drs"]
        r_oaa = r_frv = slope = np.nan
        n = 0
        try:
            import pybaseball as pb
            if k != 2:
                oaa = pb.statcast_outs_above_average(season, k, 1)
                idc, vc = _col(oaa, "player_id"), _col(oaa, "outs_above_average")
                if idc and vc:
                    their = oaa.set_index(idc)[vc]
                    r_oaa, _, n = _stats(mine, their.reindex(mine.index))
        except Exception:
            pass
        try:
            frv = savant.fielding_run_value(season, k, 1)
            idc = _col(frv, "player_id") or _col(frv, "id")
            vc = _col(frv, "run_value", exclude=("arm", "range", "fram", "block", "throw")) or _col(frv, "total")
            if idc and vc:
                their = pd.to_numeric(frv.set_index(idc)[vc], errors="coerce")
                r_frv, slope, n = _stats(mine, their.reindex(mine.index))
        except Exception:
            pass
        L.append(f"  {name:<5}{r_oaa:>10.2f}{r_frv:>10.2f}{slope:>14.2f}{n:>6}")
    return L


def blocking_compare(o: pd.DataFrame, season: int) -> list[str]:
    """Our Blocks Above Average vs Savant's (same method, so r should be high and slope near 1)."""
    from core import savant
    L = ["", "Catcher blocking vs Savant (Blocks Above Average; Statcast method):"]
    try:
        sv = savant.catcher_blocking(season)
    except Exception as e:
        return L + [f"  couldn't download Savant's blocking leaderboard ({type(e).__name__}: {e})"]
    idc = _col(sv, "player_id") or _col(sv, "id")
    bc = (_col(sv, "blocks_above", exclude=("per", "rate")) or _col(sv, "baa"))
    rc = _col(sv, "run", exclude=("per",))
    if not (idc and bc):
        return L + [f"  couldn't find id / blocks-above-average columns (columns: {list(sv.columns)})"]
    c = o[o.pos == "C"].set_index("player")
    sv = sv.assign(_id=pd.to_numeric(sv[idc], errors="coerce")).dropna(subset=["_id"]).set_index("_id")
    their = pd.to_numeric(sv[bc], errors="coerce").reindex(c.index.astype(float))
    their.index = c.index
    r, sl, n = _stats(c["baa"], their)
    L.append(f"  Blocks Above Average:  r = {r:.2f}  slope = {sl:.2f}  ({n} catchers)")
    if rc:
        tr = pd.to_numeric(sv[rc], errors="coerce").reindex(c.index.astype(float))
        tr.index = c.index
        r2, sl2, n2 = _stats(c["b_rs_sc"], tr)
        r3, sl3, _ = _stats(c["b_rs"], tr)
        L.append(f"  Blocking runs (our Statcast-way runs):  r = {r2:.2f}  slope = {sl2:.2f}  ({n2})")
        L.append(f"  Blocking runs (our SITUATIONAL runs, into DRS):  r = {r3:.2f}  slope = {sl3:.2f}")
    m = c[["baa"]].assign(savant=their).dropna().sort_values("savant", ascending=False)
    if len(m):
        try:
            from core import mlb_api
            names = mlb_api.get_people(m.index.astype(int).tolist())
        except Exception:
            names = {}
        L += ["", f"  {'CATCHER':<26}{'SAVANT BAA':>11}{'OURS':>7}"]
        for pid, r_ in pd.concat([m.head(8), m.tail(5)]).iterrows():
            L.append(f"  {str(names.get(int(pid), pid)):<26}{r_['savant']:>+11.0f}{r_['baa']:>+7.0f}")
    return L


def drs_compare(o: pd.DataFrame, season: int) -> list[str]:
    path = COMPARE_DIR / f"drs_{season}.csv"
    if not path.exists():
        return ["", f"SIS DRS: no file yet. Save the Fielding Bible leaderboard as {path.relative_to(ROOT)} "
                    "(player name, position, DRS) and run this again."]
    sis = pd.read_csv(path)
    nc = _col(sis, "player") or _col(sis, "name")
    pc = _col(sis, "pos")
    dc = _col(sis, "drs", exclude=("rank", "/", "per")) or _col(sis, "total")
    if not (nc and dc):
        return ["", f"SIS DRS: couldn't find name / DRS columns in {path.name} (columns: {list(sis.columns)})"]
    from core import mlb_api
    names = mlb_api.get_people(sorted(o["player"].astype(int).unique().tolist()))
    o = o.assign(key=o["player"].map(lambda p: norm(names.get(int(p), ""))))
    tot = o.groupby("key").agg(drs=("drs", "sum"), inn=("inn", "sum"))
    sis = sis.assign(key=sis[nc].map(norm), sis=pd.to_numeric(sis[dc], errors="coerce"))
    L = ["", f"SIS DRS ({path.name}):"]
    if pc:
        L.append(f"  {'POS':<5}{'r':>7}{'SLOPE':>8}{'N':>6}")
        for name in POS_NAME.values():
            s_ = sis[sis[pc].astype(str).str.upper().str.contains(name)]
            m = s_.merge(o[o.pos == name], on="key")
            r, sl, n = _stats(m["drs"], m["sis"])
            L.append(f"  {name:<5}{r:>7.2f}{sl:>8.2f}{n:>6}")
    m = sis.groupby("key")["sis"].sum().to_frame().join(tot, how="inner")
    r, sl, n = _stats(m["drs"], m["sis"])
    L.append(f"  ALL  r = {r:.2f}  slope = {sl:.2f}  ({n} players matched)")
    top = m.sort_values("sis", ascending=False).head(15)
    L += ["", f"  {'PLAYER':<26}{'SIS DRS':>8}{'OURS':>7}"]
    for k, r_ in top.iterrows():
        L.append(f"  {k.title():<26}{r_['sis']:>+8.0f}{r_['drs']:>+7.0f}")
    return L


def main(argv: list[str]) -> None:
    from pipeline.train_decisions import latest_full_season
    season = int(argv[argv.index("--season") + 1]) if "--season" in argv else latest_full_season()
    o = ours(season)
    L = [f"IQ DRS vs public fielding numbers -- {season}",
         "r = do the player orders agree (1 = perfectly); SLOPE = our runs per 1 of theirs (1.00 = same scale).", ""]
    L += savant_compare(o, season)
    L += blocking_compare(o, season)
    L += drs_compare(o, season)
    out = "\n".join(L)
    print(out)
    path = ROOT / "models" / "artifacts" / f"fielding_compare_{season}.txt"
    path.write_text(out + "\n")
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
