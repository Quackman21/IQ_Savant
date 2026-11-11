"""Runner events from MLB's free play-by-play: wild pitches, passed balls, stolen bases, caught stealing --
and the exact pitch each happened on, so they can be matched to the catcher (Statcast fielder_2).

    python -m pipeline.fetch_events                 # every season 2015 -> latest finished, skips saved seasons
    python -m pipeline.fetch_events --season 2026

Saved small: data/raw/events_<year>.csv.gz (one row per runner event). One call per game, 6 at a time.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

KEEP = {"wild_pitch", "passed_ball", "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
        "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home"}
WORKERS = 6


def parse(pbp: dict, game_pk: int) -> list[dict]:
    """One row per runner event in KEEP, tied to the pitch it happened on (the last pitch at / before it)."""
    rows = []
    for play in pbp.get("allPlays", []):
        ab = int(play.get("about", {}).get("atBatIndex", -1)) + 1           # Statcast at_bat_number is 1-based
        events = play.get("playEvents", [])
        for rn in play.get("runners", []):
            det = rn.get("details", {})
            et = det.get("eventType") or ""
            if et not in KEEP:                       # e.g. a dropped third strike: tagged strikeout, moved on a WP / PB
                why = f"{det.get('movementReason') or ''} {det.get('event') or ''}".lower()
                if "wild_pitch" in why or "wild pitch" in why:
                    et = "wild_pitch"
                elif "passed_ball" in why or "passed ball" in why:
                    et = "passed_ball"
                else:
                    continue
            idx = det.get("playIndex")
            pitch = None
            if isinstance(idx, int):
                for ev in reversed(events[: idx + 1]):
                    if ev.get("isPitch") and ev.get("pitchNumber") is not None:
                        pitch = int(ev["pitchNumber"])
                        break
            mv = rn.get("movement", {})
            rows.append({"game_pk": game_pk, "at_bat_number": ab, "pitch_number": pitch, "event": et,
                         "runner": (det.get("runner") or {}).get("id"), "start": mv.get("start"),
                         "end": mv.get("end"), "is_out": bool(mv.get("isOut"))})
    return rows


def season_events(year: int, refresh: bool = False) -> pd.DataFrame:
    from core import mlb_api
    from pipeline import build_league_reference as bl
    path = bl.RAW_DIR / f"events_{year}.csv.gz"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    games = [g["gamePk"] for g in mlb_api.get_season_games(year)
             if g.get("gameType") == "R" and g.get("status", {}).get("abstractGameState") == "Final"]
    print(f"[{year}] play-by-play for {len(games):,} games…", flush=True)
    rows, failed, t0 = [], [], time.time()

    def one(pk):
        for attempt in range(4):
            try:
                return parse(mlb_api.get_play_by_play(pk), pk)
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(pk)

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(one, pk): pk for pk in games}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows += f.result()
            except Exception:
                failed.append(futs[f])
            if i % 500 == 0:
                print(f"    {i:,}/{len(games):,} games ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["game_pk", "at_bat_number", "pitch_number", "event", "runner", "start",
                                     "end", "is_out"])
    if failed:
        print(f"[{year}] {len(failed)} games failed -- not saving; run again to retry", flush=True)
        return df
    bl.RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)
    print(f"[{year}] {len(df):,} runner events saved ({time.time() - t0:.0f}s)", flush=True)
    return df


def main(argv: list[str]) -> None:
    from pipeline.train_decisions import latest_full_season
    years = [int(argv[argv.index("--season") + 1])] if "--season" in argv else \
        list(range(2015, latest_full_season() + 1))
    for y in years:
        season_events(y, "--refresh" in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
