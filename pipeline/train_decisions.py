"""Train IQ Swing Decisions (models/decisions_iq.py).

Needs EVERY pitch (not just batted balls) for one season -- run the first time when you step
away (same size as one season of the league update). Saved in data/train/ so re-training is instant.

    python -m pipeline.train_decisions                  # download (if needed), test, train, save, report
    python -m pipeline.train_decisions --season 2025    # a different season (default: latest full season)
    python -m pipeline.train_decisions --since 2015     # every season since 2015 (saved downloads reused)
    python -m pipeline.train_decisions --activate       # same, then switch the site's GD% stats to it
    python -m pipeline.train_decisions --use-placeholder   # switch back to the zone placeholder
"""
from __future__ import annotations

import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN_DIR = Path(__file__).resolve().parent.parent / "data" / "train"
KEEP = ["game_date", "game_pk", "at_bat_number", "pitch_number", "batter", "stand", "p_throws", "balls",
        "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b", "inning", "inning_topbot", "bat_score",
        "fld_score", "description", "des", "events", "type", "plate_x", "plate_z", "sz_top", "sz_bot",
        "release_speed", "pfx_x", "pfx_z", "pitch_type", "delta_run_exp", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom", "launch_speed", "launch_angle", "hc_x", "hc_y", "game_type"]
MIN_HALF = 250            # graded pitches per half for the repeatability test
MIN_SEASON = 600          # graded pitches per season for the walk / strikeout check


def latest_full_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 11 else today.year - 1


def season_pitches(year: int) -> pd.DataFrame:
    """One season, only the columns this model needs. Reads the saved full download (data/raw/) or the
    older trimmed file; downloads (and saves) the season only if neither exists."""
    from pipeline import build_league_reference as bl
    path = TRAIN_DIR / f"pitches_{year}.csv.gz"
    if bl.raw_path(year).exists() or not path.exists():
        df = bl.pull_season(year)
    else:
        df = pd.read_csv(path, low_memory=False)
    return df[[c for c in KEEP if c in df.columns]]


def compact(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink a season in memory (8+ million pitches have to fit on a laptop): text -> categories,
    decimals -> 32-bit, the play description -> just 'bunt' or ''."""
    from core.statcast_clean import clean
    d = clean(df).copy()
    if "game_type" in d:
        d = d[d["game_type"].astype(str).eq("R")].drop(columns="game_type")
    if "des" in d:
        d["des"] = np.where(d["des"].astype(str).str.contains("bunt", case=False, na=False), "bunt", "")
    ids = {"game_pk", "batter", "at_bat_number", "pitch_number", "on_1b", "on_2b", "on_3b"}
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].astype("category")
        elif d[c].dtype == "float64" and c not in ids:
            d[c] = d[c].astype("float32")
    return d


# ----------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------
def _gb_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                          min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20, random_state=0)


def _gb_reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                         min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=20, random_state=0)


def run_values(df: pd.DataFrame) -> pd.Series:
    """delta_run_exp from the BATTING team's side (a ball must be worth +)."""
    rv = pd.to_numeric(df["delta_run_exp"], errors="coerce")
    balls = df["description"].isin({"ball", "blocked_ball"})
    return -rv if rv[balls].mean() < 0 else rv


def tables(df: pd.DataFrame, rv: pd.Series, act: pd.Series, xq: pd.Series | None = None) -> dict:
    from models import decisions_iq as dq
    st = dq.state(df)
    desc = df["description"]
    bip = desc.isin(dq.IN_PLAY) & act.eq("swing")
    kinds = {"ball": desc.isin(dq.BALL) & act.eq("take"),
             "strike": desc.isin(dq.CALLED | dq.MISS),
             "bip": bip}
    if xq is not None:
        kinds["bip_dmg"] = bip & (xq >= dq.DAMAGE_XWOBA)                  # damage contact
        if "events" in df:
            kinds["bip_out"] = bip & df["events"].isin(dq.OUT_EVENTS)    # weak contact = an out
    out = {"count": {}, "bo": {}}
    for name, m in kinds.items():
        g = rv[m & rv.notna()]
        cm = g.groupby(st.loc[g.index, "count"]).mean()
        out["count"][name] = cm.to_dict()
        agg = g.groupby(st.loc[g.index, "bo"]).agg(["sum", "size"])
        prior = agg.index.str.split("|").str[0].map(cm).astype(float)
        out["bo"][name] = ((agg["sum"] + dq.SHRINK * prior) / (agg["size"] + dq.SHRINK)).dropna().to_dict()
    return out


def contact_target(df: pd.DataFrame) -> pd.Series:
    """Expected xwOBA of each ball in play, from whichever xwOBA model the site uses."""
    from models import xwoba
    return pd.to_numeric(xwoba.expected_woba(df), errors="coerce")


def fit(df: pd.DataFrame) -> dict:
    from models import decisions_iq as dq
    act = dq.action(df)
    f = dq.features(df)
    rv = run_values(df)
    desc = df["description"]
    loc = f[["x_in", "z_norm"]].notna().all(axis=1)

    takes = act.eq("take") & desc.isin(dq.CALLED | {"ball", "blocked_ball"}) & loc
    called = _gb_clf().fit(f.loc[takes, dq.CALLED_FEATURES], desc[takes].isin(dq.CALLED).astype(int))

    swings = act.eq("swing") & loc
    y = np.select([desc.isin(dq.MISS), desc.isin(dq.FOUL)], ["miss", "foul"], "bip")
    swing = _gb_clf().fit(f.loc[swings, dq.PITCH_FEATURES], pd.Series(y, index=df.index)[swings])

    decided = act.notna() & loc                       # league-average swing rate (for Decision Runs)
    swing_rate = _gb_clf().fit(f.loc[decided, dq.PITCH_FEATURES], act[decided].eq("swing").astype(int))

    bip = act.eq("swing") & desc.isin(dq.IN_PLAY) & loc
    xq = contact_target(df)
    bip &= xq.notna()
    contact = _gb_reg().fit(f.loc[bip, dq.PITCH_FEATURES], xq[bip])
    damage = _gb_clf().fit(f.loc[bip, dq.PITCH_FEATURES], (xq[bip] >= dq.DAMAGE_XWOBA).astype(int))
    x, r = xq[bip & rv.notna()], rv[bip & rv.notna()]
    slope = float(np.cov(x, r)[0, 1] / x.var()) if len(x) > 100 else 1.0
    return {"called": called, "swing": swing, "swing_rate": swing_rate, "contact": contact, "damage": damage,
            "tables": tables(df, rv, act, xq),
            "runs_per_xwoba": slope, "xq_mean": float(xq[bip].mean()),
            "trained_on": sorted(pd.to_datetime(df["game_date"]).dt.year.unique().tolist())}


# ----------------------------------------------------------------------------
# Test: each half of the games scored by a model trained on the other half
# ----------------------------------------------------------------------------
def out_of_fold(df: pd.DataFrame) -> pd.DataFrame:
    from models import decisions_iq as dq
    fold = pd.to_numeric(df["game_pk"], errors="coerce").fillna(0).astype(int) % 2
    parts = []
    for k in (0, 1):
        b = fit(df[fold != k])
        parts.append(dq.score(df[fold == k], b))
    return pd.concat(parts).reindex(df.index)


def _halves(d: pd.DataFrame, cols: dict) -> dict:
    d = d.assign(year=pd.to_datetime(d["game_date"]).dt.year,
                 half=np.where(pd.to_datetime(d["game_date"]).dt.month <= 6, 1, 2))
    out = {}
    for label, (val, mask) in cols.items():
        g = d[mask].groupby(["year", "batter", "half"])[val].agg(["mean", "size"]).reset_index()
        a, b = g[g.half == 1], g[g.half == 2]
        m = a.merge(b, on=["year", "batter"], suffixes=("_1", "_2"))
        m = m[(m.size_1 >= MIN_HALF) & (m.size_2 >= MIN_HALF)]
        out[label] = (m["mean_1"].corr(m["mean_2"]), len(m))
    return out


def report(df: pd.DataFrame, sc: pd.DataFrame, b: dict) -> str:
    from core import hit_metrics as hm
    from models import decisions, decisions_iq as dq
    st = dq.state(df)
    L = ["IQ Swing Decisions v1 -- test report",
         "Each half of the season's games scored by a model trained on the OTHER half.",
         f"Runners + outs at {dq.BASEOUT_WEIGHT:.0%} strength; they can't flip a decision count + location "
         f"call clear (|z| >= {dq.CLEAR}); neutral = |z| < {dq.NEUTRAL}.", ""]

    # 1. count values
    t = b["tables"]["count"]
    L += ["What each count is worth (runs, batting side) -- from real run values:",
          "  COUNT   BALL   STRIKE   IN PLAY   AT STAKE (ball - strike)"]
    for cnt in ["0-0", "1-0", "0-1", "2-0", "1-1", "0-2", "3-0", "2-1", "1-2", "3-1", "2-2", "3-2"]:
        bl, sk, bp = t["ball"].get(cnt, np.nan), t["strike"].get(cnt, np.nan), t["bip"].get(cnt, np.nan)
        L.append(f"  {cnt:<6}{bl:>+7.3f}{sk:>+9.3f}{bp:>+10.3f}{bl - sk:>12.3f}")
    L.append(f"  (contact: +{b['runs_per_xwoba']:.2f} runs per 1.000 of expected xwOBA above average)")

    # 2. grade mix
    g = sc.dropna(subset=["z"])
    tot = len(g)
    L += ["", f"Graded pitches: {tot:,}  |  good {g['good'].eq(1).mean():.1%}  bad {g['good'].eq(0).mean():.1%}  "
          f"neutral {g['good'].isna().mean():.1%}",
          "", "By count (share of decisions):     SWINGS good / bad / neutral     TAKES good / bad / neutral"]
    for cnt in ["0-0", "1-0", "0-1", "2-0", "1-1", "0-2", "3-0", "2-1", "1-2", "3-1", "2-2", "3-2"]:
        row = f"  {cnt:<6}"
        for a in ("swing", "take"):
            h = g[(st.loc[g.index, "count"] == cnt) & g["action"].eq(a)]
            if len(h):
                row += (f"   {a+'s':<6} {h['good'].eq(1).mean():>5.0%} / {h['good'].eq(0).mean():>4.0%} / "
                        f"{h['good'].isna().mean():>4.0%}  (n {len(h):,})")
        L.append(row)

    # 2b. tier check: the coaching rule (heart = swing, sub-heart = swing to a lesser extent, edges = take
    # before 2 strikes), in IQ zones
    zone = dq.iq_zone(df).loc[g.index]
    zs = dq.ZONE_ORDER
    L += ["", "Tier check -- share graded GOOD by IQ zone (Heart / Sub-heart / Edge are inside the zone):",
          f"  {'':<24}" + "".join(f"{z:>12}" for z in zs)]
    for lab, m in (("0-0", st.loc[g.index, "count"].eq("0-0")),
                   ("before 2 strikes", ~st.loc[g.index, "two_k"]),
                   ("2 strikes", st.loc[g.index, "two_k"])):
        for a in ("swing", "take"):
            row = f"  {lab + ' ' + a + 's':<24}"
            for z in zs:
                h = g[m & g["action"].eq(a) & zone.eq(z)]
                row += f"{h['good'].eq(1).mean():>12.0%}" if len(h) >= 30 else f"{'-':>12}"
            L.append(row)
    L.append("  (- = fewer than 30 pitches)")

    # 3. flip report
    L += ["", "Flip report -- how often each layer changed the grade from count + location alone:",
          f"  runners + outs: {g['flipped'].mean():.1%} of graded pitches",
          "  game state: not in v1"]
    bo = st.loc[g.index, "bo"].str.split("|", n=1).str[1]
    fr = g.groupby(bo)["flipped"].agg(["mean", "size"]).sort_values("mean", ascending=False).head(5)
    names = {0: "empty", 1: "1st", 2: "2nd", 3: "1st+2nd", 4: "3rd", 5: "1st+3rd", 6: "2nd+3rd", 7: "loaded"}
    for k, r in fr.iterrows():
        base, outs = k.split("|")
        L.append(f"    most: {names[int(base)]:<8} {outs} out  -> {r['mean']:.1%} flipped (n {int(r['size']):,})")

    # 4. called-strike check: are the chances right?
    tk = df["description"].isin(dq.CALLED | {"ball", "blocked_ball"}) & sc["p_called"].notna()
    pc, called = sc.loc[tk, "p_called"].astype(float), df.loc[tk, "description"].isin(dq.CALLED)
    L += ["", "Called-strike check (taken pitches, grouped by the model's strike chance):",
          "  MODEL SAID     ACTUALLY CALLED STRIKE     PITCHES"]
    for lo in np.arange(0, 1, 0.1):
        m = pc.between(lo, lo + 0.1, inclusive="left" if lo < 0.9 else "both")
        if m.sum():
            L.append(f"  {lo:>4.0%}-{lo + 0.1:<4.0%}   {pc[m].mean():>5.0%} avg -> {called[m].mean():>5.0%}"
                     f"{m.sum():>17,}")

    # 5. repeatability vs the placeholder
    att = hm.attack_zone(df)
    swing = df["description"].isin(hm.SWINGS)
    ph = decisions.placeholder_v0(df, swing, att)
    d = df[["game_date", "batter"]].assign(gd=sc["good"], q=sc["quality"], runs=sc["runs_vs_avg"] * 100, ph=ph)
    rep = _halves(d, {"IQ GD%": ("gd", d["gd"].notna()),
                      "IQ Decision Quality": ("q", d["q"].notna()),
                      "IQ Decision Runs / 100": ("runs", d["runs"].notna()),
                      "Placeholder GD%": ("ph", d["ph"].notna())})
    L += ["", f"Repeatable? hitter's 1st half vs 2nd half ({MIN_HALF}+ graded pitches each half; higher = real skill):"]
    for k_, (r, n) in rep.items():
        L.append(f"  {k_:<24} r = {r:.3f}   ({n} hitters)")

    # 6. does it relate to walks / strikeouts?
    ev = df["events"].astype(str)
    pa = df.groupby("batter")["events"].apply(lambda s_: s_.notna().sum())
    bb = df[ev.isin({"walk"})].groupby("batter").size().reindex(pa.index, fill_value=0) / pa
    k = df[ev.isin({"strikeout", "strikeout_double_play"})].groupby("batter").size().reindex(pa.index, fill_value=0) / pa
    s = d.groupby("batter").agg(gd=("gd", "mean"), n=("q", "count"), q=("q", "mean"), r100=("runs", "mean"),
                                total=("runs", "sum"), ph=("ph", "mean"))
    s["total"] /= 100
    s = s[s.n >= MIN_SEASON].join(pd.DataFrame({"bb": bb, "k": k}))
    L += ["", f"Same season, hitters with {MIN_SEASON}+ graded pitches ({len(s)}): correlation with",
          "                         BB%      K%     BB%-K%"]
    for lab, col in (("IQ GD%", "gd"), ("IQ Decision Quality", "q"), ("IQ Decision Runs/100", "r100"),
                     ("Placeholder GD%", "ph")):
        L.append(f"  {lab:<21}{s[col].corr(s['bb']):>+7.2f}{s[col].corr(s['k']):>+8.2f}"
                 f"{s[col].corr(s['bb'] - s['k']):>+9.2f}")
    L.append(f"  (GD% vs Decision Quality across hitters: r = {s['gd'].corr(s['q']):.2f} -- "
             "lower = they tell you more different things)")

    # 7. sanity: best / worst by Decision Runs
    if len(s) >= 10:
        try:
            from core import mlb_api
            names = mlb_api.get_people(s.index.astype(int).tolist())
        except Exception:
            names = {}
        L += ["", "Best and worst decision-makers by Decision Runs (does the list pass the eye test?):",
              f"  {'HITTER':<24}{'DEC RUNS':>9}{'GD%':>8}{'DEC QUAL':>10}{'BB%':>7}{'K%':>7}"]
        top = s.sort_values("total", ascending=False)
        for lab, part in (("top", top.head(5)), ("bottom", top.tail(5))):
            for bid, r in part.iterrows():
                nm = names.get(int(bid), str(bid))
                L.append(f"  {nm:<24}{r['total']:>+9.1f}{r['gd']:>8.1%}{r['q']:>+10.1f}{r['bb']:>7.1%}{r['k']:>7.1%}")
            if lab == "top":
                L.append("  ...")
    return "\n".join(L)


def video_sample(df: pd.DataFrame, sc: pd.DataFrame, path: Path, n: int = 20) -> None:
    """20 random graded pitches to check against video."""
    from models import decisions_iq as dq
    g = sc[sc["z"].notna()].sample(min(n, sc["z"].notna().sum()), random_state=7)
    d = df.loc[g.index]
    st = dq.state(d)
    out = pd.DataFrame({"date": d["game_date"], "game_pk": d["game_pk"], "inning": d["inning"],
                        "at_bat": d["at_bat_number"], "pitch": d["pitch_number"], "batter_id": d["batter"],
                        "count": st["count"], "zone": dq.iq_zone(d), "runners_outs": st["bo"].str.split("|", n=1).str[1],
                        "pitch_type": d["pitch_type"], "mph": d["release_speed"],
                        "plate_x": d["plate_x"], "plate_z": d["plate_z"], "what_happened": d["description"],
                        "decision": g["action"], "grade": g["good"].map({1.0: "GOOD", 0.0: "BAD"}).fillna("neutral"),
                        "model_strike_chance": g["p_called"].round(2), "avg_hitter_swings": g["p_swing_avg"].round(2),
                        "z": g["z"].round(2), "quality": g["quality"].round(0),
                        "runs_vs_avg": g["runs_vs_avg"].round(3)})
    try:
        from core import mlb_api
        names = mlb_api.get_people(out["batter_id"].astype(int).unique().tolist())
        out.insert(6, "batter", out["batter_id"].astype(int).map(names))
    except Exception:
        pass
    out.sort_values(["date", "game_pk", "at_bat", "pitch"]).to_csv(path, index=False)


# ----------------------------------------------------------------------------
def main(argv: list[str]) -> None:
    from models import decisions, decisions_iq as dq
    if "--use-placeholder" in argv:
        decisions.set_active("placeholder_v0")
        print("Site switched back to the placeholder decision model. Run `python -m pipeline.update` "
              "when you step away so MLB averages match.")
        return
    last = latest_full_season()
    if "--since" in argv:
        years = list(range(int(argv[argv.index("--since") + 1]), last + 1))
    else:
        years = [int(argv[argv.index("--season") + 1]) if "--season" in argv else last]
    t0 = time.time()
    import joblib
    eras = [("SHIFT ERA (2015-2022)", "pre", [y for y in years if y < dq.SHIFT_BAN]),
            ("PITCH CLOCK + SHIFT BAN (2023+)", "post", [y for y in years if y >= dq.SHIFT_BAN])]
    models, reps, last_df, last_sc = {}, [], None, None
    for title, key, ys in eras:                       # one era at a time: own model, own data, less memory
        if not ys:
            continue
        parts = []
        for year in ys:
            d = compact(season_pitches(year))
            print(f"[{year}] {len(d):,} pitches ready", flush=True)
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        for c in df.columns:                           # categories differ by season -> re-compact
            if df[c].dtype == object:
                df[c] = df[c].astype("category")
        del parts
        print(f"\n{title}: {len(df):,} pitches. Testing (each half of the games scored by a model trained on "
              "the other half)…", flush=True)
        sc = out_of_fold(df)
        print(f"{title}: training the final model on every pitch…", flush=True)
        models[key] = fit(df)
        reps.append(f"===== {title} -- its own model, trained only on these seasons =====\n"
                    + report(df, sc, models[key]))
        print("\n" + reps[-1] + "\n", flush=True)
        last_df, last_sc = df, sc
    rep = "\n\n".join(reps)
    bundle = models["post"] if list(models) == ["post"] else (
        models["pre"] if list(models) == ["pre"] else {"eras": models, "split": dq.SHIFT_BAN})
    bundle["report"] = rep
    dq.ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, dq.ARTIFACT, compress=3)
    (dq.ARTIFACT.parent / "decisions_iq_v1_report.txt").write_text(rep + "\n")
    sample = dq.ARTIFACT.parent / "decisions_video_check.csv"
    video_sample(last_df, last_sc, sample)            # video check from the newest era (today's game)
    print(f"Saved {dq.ARTIFACT.name}, the report and {sample.name} (20 pitches to check on video) "
          f"in {time.time() - t0:.0f}s")

    if "--activate" in argv:
        decisions.set_active("iq_v1")
        print("\nThe site's GD% stats now use IQ Swing Decisions. MLB averages switch after the next "
              "`python -m pipeline.update` -- run it when you step away.")
    else:
        print("\nNot switched on yet. Look over the report + video check, then:  "
              "python -m pipeline.train_decisions --activate")


if __name__ == "__main__":
    main(sys.argv[1:])
