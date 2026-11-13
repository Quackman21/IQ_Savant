"""IQ Ball Flight Grade -- how well a hitter steers the ball: in the air, to his pull side, at repeatable angles.

Bat speed / exit velo are kept OUT on purpose (that's the Bat Speed Grade). Every ingredient uses launch angle and
spray direction only -- no exit-velo cut-offs and no xwOBA, both of which let power (and contact) leak back in.
Weights from the Ball Flight study (python -m pipeline.flight_study, 2015-2026), the AirPull-heavy mix:

    AirPull%            45%   balls hit 10-45 deg AND to the pull side (the gold star: most valuable ball in baseball)
    Air%                30%   balls hit 10-50 deg (in the air, not on the ground or popped up)
    LA consistency      25%   launch-angle spread (lower = better)

v19.9: Pull-Air% dropped -- it double-counted AirPull% (the data gave it a negative weight once AirPull was in).
This mix predicts next season's ball-flight value better than v19.8's in BOTH halves of the data (2015-21 and
2022-25, so not tuned to one era) and leaks essentially nothing from power (r with 90th EV -.03, bat speed -.02).

Each ingredient is regressed toward the league average by its own sample (balls in play before it is 70% skill,
from the study), then graded vs MLB-qualified hitters: 50 = average qualified hitter, 10 = one spread.
League references: data/league/flight_<year>.json  (python -m pipeline.flight_ref)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REF_DIR = Path(__file__).resolve().parent.parent / "data" / "league"
W = {"AirPull%": .45, "Air%": .30, "LA spread": .25}
SIGN = {"AirPull%": 1, "Air%": 1, "LA spread": -1}
INGREDIENTS = list(W)
# balls in play before each ingredient is 70% skill -- flight study, both eras averaged
BIP_FOR_70 = {"AirPull%": 210, "Air%": 310, "LA spread": 410}
K = {m: n * 0.3 / 0.7 for m, n in BIP_FOR_70.items()}     # reliability = n / (n + k), n = his balls in play
MIN_BIP = 25
REGRESS = True


def balls(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ball in play (no bunts): batter, launch angle, pulled / has-direction flags."""
    from core import hit_metrics as hm
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
    la = num("launch_angle")
    s = lambda c: df[c].astype(str) if c in df else pd.Series("", index=df.index)
    bunt = s("description").str.contains("bunt") | s("events").str.contains("bunt") | s("des").str.contains("bunt")
    bip = s("type").eq("X") & la.notna() & ~bunt
    b = df[bip]
    if b.empty:
        return pd.DataFrame(columns=["batter", "la", "spr", "pull", "pull10"])
    spray = hm.spray_angle(b)
    rhb = b["stand"].astype(str).eq("R") if "stand" in b else pd.Series(True, index=b.index)
    spr = spray.notna()
    return pd.DataFrame({"batter": b["batter"] if "batter" in b else 0, "la": la[bip], "spr": spr,
                         "pull": spr & np.where(rhb, spray <= -15, spray >= 15),
                         "pull10": spr & np.where(rhb, spray <= -10, spray >= 10)}, index=b.index)


def hitter_table(df: pd.DataFrame) -> pd.DataFrame:
    """The three ingredients for every batter in df, plus BIP."""
    d = balls(df)
    if d.empty:
        return pd.DataFrame(columns=["BIP", *INGREDIENTS])
    air = (d["la"] >= 10) & (d["la"] < 50)
    d = d.assign(air=air, airpull=d["pull10"] & (d["la"] >= 10) & (d["la"] <= 45))
    g = d.groupby("batter")
    out = pd.DataFrame({"BIP": g.size()})
    out["AirPull%"] = d[d["spr"]].groupby("batter")["airpull"].mean()
    out["Air%"] = g["air"].mean()
    out["LA spread"] = g["la"].std()
    return out


def build_ref(season_df: pd.DataFrame, year: int) -> dict:
    """League average + spread of each ingredient among MLB-qualified hitters, one season."""
    from models import power
    d = season_df
    if "game_type" in d:
        d = d[d["game_type"].astype(str).eq("R")]
    t = hitter_table(d)
    q, need = power.qualified(d)
    reg = t[t.index.isin(q)]
    stats = {m: (float(reg[m].mean()), float(reg[m].std())) for m in INGREDIENTS if reg[m].notna().sum() >= 30}
    ref = {"year": year, "stats": stats, "n_qualified": int(len(reg)), "qualified_pa": float(need),
           "median_bip": float(reg["BIP"].median()) if len(reg) else 0}
    if all(m in stats for m in W) and len(reg) >= 30:
        comp = sum(v * SIGN[m] * (reg[m].fillna(stats[m][0]) - stats[m][0]) / stats[m][1] for m, v in W.items())
        ref["sd_blend"] = float(comp.std())
    return ref


def save_ref(ref: dict) -> Path:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    p = REF_DIR / f"flight_{ref['year']}.json"
    p.write_text(json.dumps(ref, indent=1))
    return p


def load_ref(year: int) -> dict | None:
    p = REF_DIR / f"flight_{year}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(m: str, x: float) -> str:
    return f"{x:.1f} deg" if m == "LA spread" else f"{100 * x:.1f}%"


def score(df: pd.DataFrame, ref: dict | None) -> dict:
    """Ball Flight Grade (as z) for one hitter over any period. Keys: _BF_Z, _BF_PARTS, _BF_KIND."""
    out = {"_BF_Z": math.nan, "_BF_PARTS": "", "_BF_KIND": "Ball flight (angle + direction, no exit velo)"}
    if df is None or df.empty or not ref or "sd_blend" not in ref:
        return out
    d = df.reset_index(drop=True)
    d["batter"] = 0
    t = hitter_table(d)
    if t.empty or t["BIP"].iloc[0] < MIN_BIP:
        return out
    row, n = t.iloc[0], float(t["BIP"].iloc[0])
    z, shown = {}, {}
    for m in INGREDIENTS:
        mean, sd = ref["stats"][m]
        x = row.get(m, np.nan)
        if x != x or not sd:
            z[m], shown[m] = 0.0, ""
            continue
        shrink = n / (n + K[m]) if REGRESS else 1.0
        z[m] = SIGN[m] * (x - mean) * shrink / sd
        shown[m] = f" ({_fmt(m, x)})"
    out["_BF_Z"] = sum(v * z[m] for m, v in W.items()) / ref["sd_blend"]
    out["_BF_PARTS"] = " | ".join(f"{m} {50 + 10 * z[m]:.0f}{shown[m]}" for m in W) + f" | {int(n)} BIP"
    return out
