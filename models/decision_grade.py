"""IQ Decision Grade (+ 0-0, <2k, 2k, FB, BB, OS Decision Grades) -- built around the RUN VALUE of decisions.

From the decision study (python -m pipeline.decision_study): decision runs are a real, repeatable skill (~24 runs a
season between a top- and bottom-10% hitter) and ~80% of the real differences come from chase and waste pitches
(swinging or laying off), split evenly before and with two strikes. Swing rate itself is never rewarded.

    DEC RUNS/100                                   55%   runs of every swing / take vs what an average hitter does
                                                         on that same pitch (heart takes counted at their true cost)
    Chase + waste swing rate, before 2 strikes     15%   (lower = better)
    Chase + waste swing rate, with 2 strikes       15%   (lower = better)
    Heart-minus-chase gap                          15%   heart swing rate minus chase + waste swing rate
                                                         (attack the heart, lay off the junk -- cancels swing rate)

Splits use the same mix inside that situation (0-0 and <2k have no 2-strike piece, 2k has no early piece -- the
weights are rescaled). Each ingredient is regressed by its own sample (from the study), then graded vs MLB-qualified
hitters in that same split: 50 = average qualified hitter, 10 = one spread. Too few decisions -> blank.
League references: data/league/decision_<year>.json  (python -m pipeline.decision_ref)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REF_DIR = Path(__file__).resolve().parent.parent / "data" / "league"
W = {"DEC RUNS/100": .55, "Chase+waste swing% <2k": .15, "Chase+waste swing% 2k": .15, "Heart-minus-chase gap": .15}
SIGN = {"DEC RUNS/100": 1, "Chase+waste swing% <2k": -1, "Chase+waste swing% 2k": -1, "Heart-minus-chase gap": 1}
SPLITS = {"all": "_DEC_Z", "00": "_DEC00_Z", "lt2": "_DECLT2_Z", "2k": "_DEC2K_Z",
          "fb": "_DECFB_Z", "bb": "_DECBB_Z", "os": "_DECOS_Z"}
# graded decisions (in that split) before DEC RUNS/100 is 70% skill -- decision study, both eras averaged
PITCHES_FOR_70 = {"all": 385, "00": 225, "lt2": 252, "2k": 300, "fb": 358, "bb": 221, "os": 205}
# the other ingredients need more pitches than DEC RUNS/100 (study: pre-2k chase ~1.1x, 2k chase ~1.8x, gap ~2x)
SLOWER = {"DEC RUNS/100": 1.0, "Chase+waste swing% <2k": 1.1, "Chase+waste swing% 2k": 1.8, "Heart-minus-chase gap": 2.0}
MIN_DECISIONS = {"all": 50, "00": 20, "lt2": 40, "2k": 25, "fb": 30, "bb": 20, "os": 15}
REGRESS = True


def frame(raw: pd.DataFrame, runs: pd.Series, graded: pd.Series) -> pd.DataFrame:
    """One row per pitch: batter, swing, attack zone, count, pitch group, decision runs, graded."""
    from core import hit_metrics as hm
    num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c in raw else pd.Series(np.nan, index=raw.index)
    strikes, balls = num("strikes"), num("balls")
    return pd.DataFrame({"batter": raw["batter"] if "batter" in raw else 0,
                         "swing": raw["description"].astype(str).isin(hm.SWINGS),
                         "att": hm.attack_zone(raw), "two_k": strikes >= 2, "first": (balls == 0) & (strikes == 0),
                         "pg": hm.pitch_group(raw), "runs": runs.reindex(raw.index).fillna(0),
                         "graded": graded.reindex(raw.index).fillna(False).astype(bool)}, index=raw.index)


def _mask(f: pd.DataFrame, split: str) -> pd.Series:
    return {"all": pd.Series(True, index=f.index), "00": f["first"], "lt2": ~f["two_k"], "2k": f["two_k"],
            "fb": f["pg"].eq("Fastball"), "bb": f["pg"].eq("Breaking"), "os": f["pg"].eq("Offspeed")}[split]


def ingredients(f: pd.DataFrame, split: str) -> pd.DataFrame:
    """Per batter: the four ingredients inside one split, plus n (graded decisions there)."""
    s = f[_mask(f, split)]
    g = s[s["graded"]].groupby("batter")
    out = pd.DataFrame({"n": g.size()})
    out["DEC RUNS/100"] = 100 * g["runs"].sum() / out["n"]
    cw = s["att"].isin(["Chase", "Waste"])
    rate = lambda m: s[m].groupby("batter")["swing"].mean()
    if split not in ("2k",):
        out["Chase+waste swing% <2k"] = rate(cw & ~s["two_k"])
    if split not in ("00", "lt2"):
        out["Chase+waste swing% 2k"] = rate(cw & s["two_k"])
    out["Heart-minus-chase gap"] = rate(s["att"].eq("Heart")) - rate(cw)
    return out


def build_ref(raw: pd.DataFrame, runs: pd.Series, graded: pd.Series, year: int) -> dict:
    """League average + spread of each ingredient, per split, among MLB-qualified hitters."""
    from models import power
    q, need = power.qualified(raw)
    f = frame(raw, runs, graded)
    ref = {"year": year, "qualified_pa": float(need), "splits": {}}
    for split in SPLITS:
        t = ingredients(f, split)
        reg = t[t.index.isin(q)]
        stats = {m: (float(reg[m].mean()), float(reg[m].std())) for m in W if m in reg and reg[m].notna().sum() >= 30}
        w = {m: v for m, v in W.items() if m in stats}
        tot = sum(w.values())
        sp = {"stats": stats, "n_qualified": int(len(reg)), "median_n": float(reg["n"].median()) if len(reg) else 0}
        if w and len(reg) >= 30:
            comp = sum(v / tot * SIGN[m] * (reg[m].fillna(stats[m][0]) - stats[m][0]) / stats[m][1] for m, v in w.items())
            sp["sd_blend"] = float(comp.std())
        ref["splits"][split] = sp
    return ref


def save_ref(ref: dict) -> Path:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    p = REF_DIR / f"decision_{ref['year']}.json"
    p.write_text(json.dumps(ref, indent=1))
    return p


def load_ref(year: int) -> dict | None:
    p = REF_DIR / f"decision_{year}.json"
    return json.loads(p.read_text()) if p.exists() else None


def score(raw: pd.DataFrame, runs: pd.Series, graded: pd.Series, ref: dict | None) -> dict:
    """All seven Decision Grades (as z) for one hitter over any period, plus each one's ingredients for hover."""
    out = {}
    for key in SPLITS.values():
        out[key], out[key[:-2] + "_PARTS"] = math.nan, ""
    if raw is None or raw.empty or not ref:
        return out
    f = frame(raw.assign(batter=0), runs, graded)
    for split, key in SPLITS.items():
        sp = ref["splits"].get(split, {})
        if "sd_blend" not in sp:
            continue
        t = ingredients(f, split)
        if t.empty or t["n"].iloc[0] < MIN_DECISIONS[split]:
            continue
        row, n = t.iloc[0], float(t["n"].iloc[0])
        w = {m: v for m, v in W.items() if m in sp["stats"]}
        tot = sum(w.values())
        z = {}
        for m in w:
            mean, sd = sp["stats"][m]
            x = row.get(m, np.nan)
            if x != x or not sd:
                z[m] = 0.0
                continue
            k = PITCHES_FOR_70[split] * 0.3 / 0.7 * SLOWER[m]
            shrink = n / (n + k) if REGRESS else 1.0
            z[m] = SIGN[m] * (x - mean) * shrink / sd
        out[key] = sum(v / tot * z[m] for m, v in w.items()) / sp["sd_blend"]
        out[key[:-2] + "_PARTS"] = " | ".join(f"{m} {50 + 10 * z[m]:.0f}" for m in w) + f" ({int(n)} decisions)"
    return out
