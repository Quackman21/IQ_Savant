"""IQ Bat Speed Grade and SC Bat Speed Grade.

Weights come from the power study (python -m pipeline.power_study, 2023-2026): every metric scored on how well it
predicts NEXT season's run value on contact. Contact-skill metrics (squared-up %, smash factor, blasts) are left
out -- they belong to Bat-To-Ball.

    Bat Speed Grade, seasons with bat tracking:
        bat speed 50%   avg bat speed 25, top 8th bat speed 25
        force in air 30% EV (fly) 15, 90th EV (fly) 15
        top-end 10%     90th EV 5, HHEV (top 8th EV) 5
        everyday 6%     avg EV 3, EV>95% 3
        max 4%          max EV 2, max EV (fly) 2          (max EVs adjusted for number of balls in play)
    No bat tracking (before 2023 / untracked): the bat speed half goes to 90th EV and HHEV (their closest stand-in).
    SC Bat Speed Grade: avg bat speed 50, top 8th bat speed 50.

Exit velocities are adjusted for pitch speed. Each ingredient is regressed toward the league average by its own
sample (how fast it becomes real skill, from the study), so a 10-game stretch can't grade 80 on noise. Then:
grade = 50 + 10 x (weighted z / spread of that blend among regulars).
League references: data/league/power_<year>.json  (python -m pipeline.power_ref)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REF_DIR = Path(__file__).resolve().parent.parent / "data" / "league"
SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "missed_bunt"}
REGRESS = True
MIN_REF_BIP = 150                          # regulars who set the league average and spread

BAT = ["Avg Bat Speed", "Top 8th Bat Speed"]
APPLY = ["Squared-Up%", "Smash Factor", "Blast%", "Smash (Top 8th)"]
OTHER = ["Swing Length"]
W_BAT = {"Avg Bat Speed": .25, "Top 8th Bat Speed": .25, "EV (Fly)": .15, "90th EV (Fly)": .15,
         "90th EV": .05, "HHEV (Top 8th EV)": .05, "Avg EV": .03, "EV>95%": .03,
         "Max EV (BIP-adj)": .02, "Max EV (Fly, BIP-adj)": .02}
W_EV = {"90th EV": .30, "HHEV (Top 8th EV)": .30, "EV (Fly)": .15, "90th EV (Fly)": .15, "Avg EV": .03,
        "EV>95%": .03, "Max EV (BIP-adj)": .02, "Max EV (Fly, BIP-adj)": .02}
W_SC = {"Avg Bat Speed": .5, "Top 8th Bat Speed": .5}
INPUTS = list(W_BAT)
# balls in play (fly balls for the fly stats) before each is 70% skill (power study, split-half) -> regression k
BIP_FOR_70 = {"Avg Bat Speed": 6, "Top 8th Bat Speed": 15, "EV (Fly)": 125, "90th EV (Fly)": 75, "90th EV": 46,
              "HHEV (Top 8th EV)": 34, "Avg EV": 90, "EV>95%": 120, "Max EV (BIP-adj)": 127,
              "Max EV (Fly, BIP-adj)": 127}
K = {m: n * 0.3 / 0.7 for m, n in BIP_FOR_70.items()}       # reliability = n / (n + k)
COUNT_OF = {m: ("fly_n" if "Fly" in m else "BIP") for m in INPUTS}
FLY_SHARE = 0.36                                             # fly balls per BIP (league), to put fly k in fly units


def _top8(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    k = max(1, int(np.ceil(len(s) / 8)))
    return float(np.sort(s.values)[-k:].mean())


def hitter_table(d: pd.DataFrame) -> pd.DataFrame:
    """Every metric for every hitter in d."""
    bunt = d["description"].astype(str).str.contains("bunt") | d["bb_type"].astype(str).eq("bunt") \
        if "bb_type" in d else d["description"].astype(str).str.contains("bunt")
    sw = d[d["description"].isin(SWINGS) & ~bunt]
    bip = d[d["type"].eq("X") & d["launch_speed"].notna() & ~bunt]
    fly = bip[(bip["launch_speed"] > 70) & (bip["launch_angle"] >= 10) & (bip["launch_angle"] < 51)]
    out = pd.DataFrame(index=pd.Index(sorted(d["batter"].dropna().unique()), name="batter"))

    # bat speed: Statcast "competitive" swings -- drop each hitter's slowest 10% (checks / defensive swings)
    bs = sw.dropna(subset=["bat_speed"])
    if len(bs):
        cut = bs.groupby("batter")["bat_speed"].transform(lambda s: s.quantile(0.10))
        comp = bs[bs["bat_speed"] >= cut]
        g = comp.groupby("batter")
        out["swings_tracked"] = bs.groupby("batter").size()
        out["Avg Bat Speed"] = g["bat_speed"].mean()
        out["Top 8th Bat Speed"] = g["bat_speed"].agg(_top8)
        out["Swing Length"] = g["swing_length"].mean()
        # application: exit velo vs the most his bat speed + this pitch could produce (Savant's squared-up)
        b = bip.dropna(subset=["bat_speed", "release_speed"])
        b = b[b["bat_speed"] > 30]
        mx = 1.23 * b["bat_speed"] + 0.23 * b["release_speed"]
        b = b.assign(smash=(b["launch_speed"] / mx).clip(0, 1.2))
        b = b.assign(sq=(b["smash"] >= 0.8).astype(float),
                     blast=((b["smash"] * 100 + b["bat_speed"]) >= 164).astype(float))
        gb = b.groupby("batter")
        out["bip_tracked"] = gb.size()
        out["Squared-Up%"] = gb["sq"].mean()
        out["Smash Factor"] = gb["smash"].mean()
        out["Blast%"] = gb["blast"].mean()
        out["Smash (Top 8th)"] = gb["smash"].agg(_top8)
    else:
        for c in BAT + APPLY + OTHER + ["swings_tracked", "bip_tracked"]:
            out[c] = np.nan

    g = bip.groupby("batter")["ev_adj"]
    out["BIP"] = g.size()
    out["Avg EV"] = g.mean()
    out["90th EV"] = g.quantile(0.90)
    out["HHEV (Top 8th EV)"] = g.agg(_top8)
    out["EV>95%"] = bip.assign(h=(bip["ev_adj"] > 95).astype(float)).groupby("batter")["h"].mean()
    out["max_raw"] = g.max()
    gf = fly.groupby("batter")["ev_adj"]
    out["EV (Fly)"] = gf.mean()
    out["90th EV (Fly)"] = gf.quantile(0.90)
    out["max_fly_raw"], out["fly_n"] = gf.max(), gf.size()

    # outcomes
    pa = d[d["woba_denom"] > 0]
    gp = pa.groupby("batter")
    out["PA"] = gp["woba_denom"].sum()
    out["wOBA"] = gp["woba_value"].sum() / out["PA"]
    out["wOBAcon"] = bip.groupby("batter")["woba_value"].mean()
    out["ISOcon"] = bip.groupby("batter")["iso_value"].mean()
    out["xwOBAcon"] = bip.groupby("batter")["estimated_woba_using_speedangle"].mean()
    return out



def prepare(df: pd.DataFrame, lg_pitch: float | None = None) -> tuple[pd.DataFrame, float]:
    """Numeric columns + pitch-speed-adjusted exit velo (a faster pitch comes off the bat ~0.2 mph harder per mph)."""
    d = df.copy()
    for c in ("launch_speed", "launch_angle", "release_speed", "bat_speed", "swing_length", "woba_value",
              "woba_denom", "iso_value", "estimated_woba_using_speedangle"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    if "batter" not in d:
        d["batter"] = 0
    if lg_pitch is None:
        lg_pitch = float(d.loc[d["type"].eq("X"), "release_speed"].mean())
    d["ev_adj"] = d["launch_speed"] - 0.2 * (d["release_speed"] - lg_pitch)
    return d, lg_pitch


def _max_slopes(t: pd.DataFrame) -> dict:
    out = {}
    for raw, n in (("max_raw", "BIP"), ("max_fly_raw", "fly_n")):
        ok = t[raw].notna() & (t[n] >= 20)
        if ok.sum() < 30:
            out[raw] = (0.0, 0.0)
            continue
        x = np.log(t.loc[ok, n])
        out[raw] = (float(np.polyfit(x, t.loc[ok, raw], 1)[0]), float(x.mean()))
    return out


def _with_max(t: pd.DataFrame, slopes: dict) -> pd.DataFrame:
    t = t.copy()
    for raw, n, name in (("max_raw", "BIP", "Max EV (BIP-adj)"), ("max_fly_raw", "fly_n", "Max EV (Fly, BIP-adj)")):
        b, xm = slopes.get(raw, (0.0, 0.0))
        t[name] = t[raw] - b * (np.log(t[n].clip(lower=1)) - xm)
    return t


def build_ref(season_df: pd.DataFrame, year: int) -> dict:
    """League average + spread of every ingredient among regulars, for one season."""
    d, lg_pitch = prepare(season_df)
    if "game_type" in d:
        d = d[d["game_type"].astype(str).eq("R")]
    t = hitter_table(d)
    slopes = _max_slopes(t)
    t = _with_max(t, slopes)
    reg = t[t["BIP"] >= MIN_REF_BIP]
    stats = {m: (float(reg[m].mean()), float(reg[m].std())) for m in INPUTS if reg[m].notna().sum() >= 30}
    ref = {"year": year, "lg_pitch": lg_pitch, "slopes": slopes, "stats": stats, "n_regulars": int(len(reg))}
    for name, w in (("bat", W_BAT), ("ev", W_EV), ("sc", W_SC)):
        if all(m in stats for m in w):
            comp = sum(v * (reg[m] - stats[m][0]) / stats[m][1] for m, v in w.items())
            ref[f"sd_{name}"] = float(comp.std())
    return ref


def save_ref(ref: dict) -> Path:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    p = REF_DIR / f"power_{ref['year']}.json"
    p.write_text(json.dumps(ref, indent=1))
    return p


def load_ref(year: int) -> dict | None:
    p = REF_DIR / f"power_{year}.json"
    return json.loads(p.read_text()) if p.exists() else None


def score(df: pd.DataFrame, ref: dict | None) -> dict:
    """Grades (as z) for one hitter's pitches over any period. Keys: _POWER_Z, _SCBS_Z, _POWER_PARTS, _POWER_KIND."""
    out = {"_POWER_Z": math.nan, "_SCBS_Z": math.nan, "_POWER_PARTS": "", "_POWER_KIND": ""}
    if df is None or df.empty or not ref or not ref.get("stats"):
        return out
    d, _ = prepare(df, ref["lg_pitch"])
    d["batter"] = 0                                         # one hitter (all his rows together)
    t = _with_max(hitter_table(d), ref["slopes"])
    if t.empty:
        return out
    row = t.iloc[0]
    stats = ref["stats"]
    z = {}
    for m in INPUTS:
        if m not in stats or not stats[m][1]:
            continue
        mean, sd = stats[m]
        x = row.get(m, np.nan)
        n = float(row.get(COUNT_OF[m], 0) or 0)
        if x != x or n <= 0:
            z[m] = 0.0                                       # no sample yet -> league average
            continue
        k = K[m] * (FLY_SHARE if COUNT_OF[m] == "fly_n" else 1.0)
        shrink = n / (n + k) if REGRESS else 1.0
        z[m] = (mean + (x - mean) * shrink - mean) / sd
    has_bat = row.get("Avg Bat Speed") == row.get("Avg Bat Speed") and "sd_bat" in ref
    w, sd_key, kind = (W_BAT, "sd_bat", "bat speed + exit velo") if has_bat else (W_EV, "sd_ev", "exit velo only")
    if sd_key in ref and all(m in z for m in w):
        out["_POWER_Z"] = sum(v * z[m] for m, v in w.items()) / ref[sd_key]
        out["_POWER_KIND"] = kind
    if has_bat and "sd_sc" in ref:
        out["_SCBS_Z"] = sum(v * z[m] for m, v in W_SC.items()) / ref["sd_sc"]
    out["_POWER_PARTS"] = " | ".join(f"{m} {50 + 10 * z[m]:.0f}" for m in w if m in z)
    return out
