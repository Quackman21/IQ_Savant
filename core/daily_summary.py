"""Day-of-competition summary from one day of Statcast pitches (pybaseball).

    df = pybaseball.statcast(start_dt=day, end_dt=day)
    s = summarize(df)

Leaders are per event (one pitch / one batted ball each), like Savant, except
Swing & Misses, which is per pitcher.
"""
from __future__ import annotations

import pandas as pd

TOP = 5
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "foul_tip_whiff", "missed_bunt"}
HITS = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR"}
STRIKEOUTS = {"strikeout", "strikeout_double_play"}
NOT_AB = {"walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play",
          "sac_bunt_double_play", "catcher_interf", "truncated_pa"}
# Savant's daily pitch mix groups some types together
PITCH_GROUPS = [("FF", {"FF"}), ("SI", {"SI"}), ("FC", {"FC"}), ("FS", {"FS", "FO", "SC"}),
                ("CH", {"CH"}), ("SL", {"SL", "ST", "SV"}), ("CU", {"CU", "KC", "CS"})]


def _teams(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(batting team, pitching team) abbreviation for every pitch."""
    top = df["inning_topbot"].eq("Top")
    bat = df["away_team"].where(top, df["home_team"])
    pit = df["home_team"].where(top, df["away_team"])
    return bat, pit


def _leaders(df: pd.DataFrame, col: str, who: str, team: pd.Series) -> list[dict]:
    rows = df[df[col].notna()].nlargest(TOP, col)
    return [{"player": int(r[who]), "team": team.loc[i], "value": float(r[col])}
            for i, r in rows.iterrows()]


def summarize(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    bat_team, pit_team = _teams(df)
    bip = df[df["type"].eq("X")]

    whiff_mask = df["description"].isin(WHIFFS)
    w = (df[whiff_mask].assign(_team=pit_team[whiff_mask])
         .groupby(["pitcher", "_team"]).size().sort_values(ascending=False).head(TOP))
    whiffs = [{"player": int(p), "team": t, "value": int(n)} for (p, t), n in w.items()]

    events = df["events"].dropna()
    ab = events[~events.isin(NOT_AB)]
    hits = events[events.isin(HITS.keys())]
    ev = bip["launch_speed"].dropna()
    brl = int((bip["launch_speed_angle"] == 6).sum()) if "launch_speed_angle" in bip else 0
    pt = df["pitch_type"].dropna()

    totals = {
        "Pitchers": df["pitcher"].nunique(),
        "Batters": df["batter"].nunique(),
        "Pitches": len(df),
        "Max Velo": round(df["release_speed"].max(), 1),
        "Min Velo": round(df["release_speed"].min(), 1),
    }
    for label, codes in PITCH_GROUPS:
        totals[f"{label}%"] = round(100 * pt.isin(codes).mean(), 1) if len(pt) else 0.0
    totals.update({
        "Max EV": round(ev.max(), 1) if len(ev) else "-",
        "HH": int((ev >= 95).sum()),
        "HH%": round(100 * (ev >= 95).mean(), 1) if len(ev) else 0.0,
        "Brl": brl,
        "Brl%": round(100 * brl / len(ev), 1) if len(ev) else 0.0,
        "AB": len(ab),
        "AVG": f"{len(hits) / len(ab):.3f}".lstrip("0") if len(ab) else "-",
        "H": len(hits),
    })
    for code, label in HITS.items():
        totals[label] = int((events == code).sum())
    totals.update({"SO": int(events.isin(STRIKEOUTS).sum()), "BIP": len(bip)})

    return {
        "exit_velo": _leaders(bip, "launch_speed", "batter", bat_team),
        "distance": _leaders(bip, "hit_distance_sc", "batter", bat_team),
        "pitch_velo": _leaders(df, "release_speed", "pitcher", pit_team),
        "whiffs": whiffs,
        "totals": totals,
    }


def player_ids(summary: dict) -> list[int]:
    ids = {r["player"] for k in ("exit_velo", "distance", "pitch_velo", "whiffs")
           for r in summary.get(k, [])}
    return sorted(ids)
