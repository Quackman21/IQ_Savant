"""ABS challenges (2026+): put every taken pitch back to the umpire's ORIGINAL call.

Framing = getting the umpire to call it. A challenge that overturns the call is a different skill, so framing is
scored on the original call; the challenges are kept separately (who challenged, overturned or not).

How the original call is found (no guessing about which call the data shows):
  1. MLB's play-by-play marks each challenged pitch and whether it was overturned (one call per game, saved in
     data/raw/abs_<year>.csv.gz; only new games are fetched on later runs).
  2. The call that STOOD is read from the count on the next pitch (a strike adds a strike, a ball adds a ball;
     if the at-bat ended there, a strikeout = strike, a walk = ball).
  3. Overturned -> original call = the opposite of the call that stood. Upheld -> the call stood.

    python -m core.abs_calls --inspect 2026      # prints how MLB marks a challenge in one game (to check the fields)
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ABS_FIRST = 2026
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
WORKERS = 6
COLS = ["game_pk", "at_bat_number", "pitch_number", "overturned", "challenger", "team", "review_type", "feed_call"]


def _is_review(obj) -> bool:
    return isinstance(obj, dict) and any(k in obj for k in ("isOverturned", "inProgress", "reviewType"))


def parse(pbp: dict, game_pk: int) -> list[dict]:
    """Challenged pitches in one game's play-by-play."""
    rows = []
    for play in pbp.get("allPlays", []):
        ab = int(play.get("about", {}).get("atBatIndex", -1)) + 1               # Statcast at_bat_number is 1-based
        last_pitch = None
        for ev in play.get("playEvents", []):
            det = ev.get("details", {}) or {}
            if ev.get("isPitch"):
                last_pitch = ev.get("pitchNumber")
                rd = ev.get("reviewDetails")
                if _is_review(rd) or det.get("hasReview"):
                    rd = rd or {}
                    rows.append({"game_pk": game_pk, "at_bat_number": ab, "pitch_number": last_pitch,
                                 "overturned": bool(rd.get("isOverturned")),
                                 "challenger": (rd.get("player") or {}).get("id"),
                                 "team": rd.get("challengeTeamId"), "review_type": rd.get("reviewType"),
                                 "feed_call": (det.get("call") or {}).get("code") or det.get("code")})
            else:                                   # some feeds log the challenge as its own event after the pitch
                text = f"{det.get('event', '')} {det.get('description', '')}".lower()
                rd = ev.get("reviewDetails")
                if last_pitch is not None and ("challenge" in text or _is_review(rd)) and \
                        not any(r["at_bat_number"] == ab and r["pitch_number"] == last_pitch for r in rows):
                    rd = rd or {}
                    rows.append({"game_pk": game_pk, "at_bat_number": ab, "pitch_number": last_pitch,
                                 "overturned": bool(rd.get("isOverturned")) or "overturn" in text,
                                 "challenger": (rd.get("player") or {}).get("id") or (ev.get("player") or {}).get("id"),
                                 "team": rd.get("challengeTeamId"), "review_type": rd.get("reviewType"),
                                 "feed_call": None})
    return rows


def season_challenges(year: int, games: list[int] | None = None) -> pd.DataFrame:
    """All challenged pitches for a season; fetches only games not checked yet."""
    from core import mlb_api
    path, done_path = RAW / f"abs_{year}.csv.gz", RAW / f"abs_{year}_games.txt"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)
    done = set(int(x) for x in done_path.read_text().split()) if done_path.exists() else set()
    if games is None:
        games = [g["gamePk"] for g in mlb_api.get_season_games(year)
                 if g.get("gameType") == "R" and g.get("status", {}).get("abstractGameState") == "Final"]
    todo = sorted(set(int(g) for g in games) - done)
    if not todo:
        return have
    print(f"[{year}] ABS challenges: checking {len(todo):,} games…", flush=True)
    rows, ok, t0 = [], [], time.time()

    def one(pk):
        for attempt in range(4):
            try:
                return pk, parse(mlb_api.get_play_by_play(pk), pk)
            except Exception:
                time.sleep(2 * (attempt + 1))
        return pk, None

    with ThreadPoolExecutor(WORKERS) as ex:
        for i, f in enumerate(as_completed([ex.submit(one, pk) for pk in todo]), 1):
            pk, r = f.result()
            if r is not None:
                rows += r
                ok.append(pk)
            if i % 500 == 0:
                print(f"    {i:,}/{len(todo):,} games ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat([have, pd.DataFrame(rows, columns=COLS)], ignore_index=True) if rows else have
    out = out.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"], keep="last")
    RAW.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, compression="gzip")
    done_path.write_text(" ".join(str(x) for x in sorted(done | set(ok))))
    print(f"[{year}] {len(out):,} challenged pitches ({out['overturned'].mean():.0%} overturned) "
          f"in {len(done | set(ok)):,} games ({time.time() - t0:.0f}s)", flush=True)
    if len(ok) > 200 and out.empty:
        print(f"[{year}] WARNING: no challenges found in {len(ok):,} games -- MLB may mark them differently. "
              f"Run  python -m core.abs_calls --inspect {year}  and send the output.", flush=True)
    return out


def call_that_stood(df: pd.DataFrame) -> pd.Series:
    """'S' / 'B' for each called pitch, from what happened next (count on the next pitch / how the at-bat ended)."""
    d = df[["game_pk", "at_bat_number", "pitch_number", "balls", "strikes"]].copy()
    d["events"] = df["events"] if "events" in df else np.nan
    d = d.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    g = d.groupby(["game_pk", "at_bat_number"])
    nb, ns = g["balls"].shift(-1), g["strikes"].shift(-1)
    out = pd.Series(np.nan, index=d.index, dtype=object)
    out[(ns - d["strikes"]) == 1] = "S"
    out[(nb - d["balls"]) == 1] = "B"
    ev = d["events"].astype(str)
    last = nb.isna()
    out[last & ev.str.startswith("strikeout")] = "S"
    out[last & ev.isin(["walk", "intent_walk"])] = "B"
    return out.reindex(df.index)


def original_calls(df: pd.DataFrame, fetch: bool = True) -> pd.DataFrame:
    """Statcast pitches -> same rows with description set to the ORIGINAL call on challenged takes,
    plus abs_challenged / abs_overturned / abs_challenger columns. Seasons before 2026 pass straight through."""
    if df is None or df.empty or "game_date" not in df or not {"game_pk", "at_bat_number", "pitch_number"} <= set(df):
        return df
    yrs = pd.to_datetime(df["game_date"]).dt.year
    out = df.copy()
    out["abs_challenged"], out["abs_overturned"], out["abs_challenger"] = False, False, np.nan
    for y in sorted(yrs[yrs >= ABS_FIRST].unique()):
        m = yrs == y
        games = pd.to_numeric(df.loc[m, "game_pk"], errors="coerce").dropna().astype(int).unique().tolist()
        ch = season_challenges(int(y), games) if fetch else (
            pd.read_csv(RAW / f"abs_{y}.csv.gz") if (RAW / f"abs_{y}.csv.gz").exists() else pd.DataFrame(columns=COLS))
        if ch.empty:
            continue
        key = ["game_pk", "at_bat_number", "pitch_number"]
        sub = df.loc[m, key].astype(int).reset_index().merge(ch.astype({k: int for k in key}), on=key, how="inner")
        if sub.empty:
            continue
        idx = sub["index"].values
        out.loc[idx, "abs_challenged"] = True
        out.loc[idx, "abs_overturned"] = sub["overturned"].astype(bool).values
        # numbers, not an object array: pandas 3 refuses object values in a float column (the v20.0 2026 crash)
        out.loc[idx, "abs_challenger"] = pd.to_numeric(sub["challenger"], errors="coerce").astype(float).values
        taken = out.loc[idx, "description"].isin(["called_strike", "ball", "blocked_ball"])
        idx = idx[taken.values]
        stood = call_that_stood(df.loc[m]).reindex(idx)
        stood = stood.fillna(out.loc[idx, "description"].map({"called_strike": "S", "ball": "B", "blocked_ball": "B"}))
        over = out.loc[idx, "abs_overturned"].astype(bool)
        orig = np.where(over, stood.map({"S": "B", "B": "S"}), stood)
        out.loc[idx, "description"] = pd.Series(orig, index=idx).map({"S": "called_strike", "B": "ball"}).values
    return out


def check(df: pd.DataFrame) -> str:
    """Self-check line for reports: does the data show the call that stood?"""
    c = df[df.get("abs_challenged", False) == True] if "abs_challenged" in df else df.iloc[0:0]  # noqa: E712
    if c.empty:
        return "ABS: no challenged pitches found."
    over = c[c["abs_overturned"]]
    return (f"ABS: {len(c):,} challenged pitches, {len(over):,} overturned ({len(over) / len(c):.0%}). "
            "Framing uses the umpire's ORIGINAL call on all of them.")


def inspect_game(year: int) -> None:
    """Print how MLB's play-by-play marks challenges in one recent game."""
    from core import mlb_api
    games = [g["gamePk"] for g in mlb_api.get_season_games(year)
             if g.get("gameType") == "R" and g.get("status", {}).get("abstractGameState") == "Final"]
    for pk in games[-40:]:
        pbp = mlb_api.get_play_by_play(pk)
        hits = []
        for play in pbp.get("allPlays", []):
            for ev in play.get("playEvents", []):
                s = json.dumps(ev).lower()
                if "challenge" in s or "review" in s or "overturn" in s:
                    hits.append(ev)
        if hits:
            print(f"game {pk}: {len(hits)} events mentioning challenge / review. First two:")
            for h in hits[:2]:
                print(json.dumps(h, indent=1)[:3000])
            print("parsed:", parse(pbp, pk)[:5])
            return
    print("No challenge / review mentions in the last 40 games.")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--inspect" in a:
        inspect_game(int(a[a.index("--inspect") + 1]))
