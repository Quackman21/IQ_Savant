"""One command to bring the site's data up to date. Run it when you step away:

    python -m pipeline.update            # rebuild what's needed, then commit + push
    python -m pipeline.update --check    # just list what's out of date (takes a second)
    python -m pipeline.update --no-push  # rebuild but don't commit / push
    python -m pipeline.update --light    # only add new games (what the desktop app runs every few hours);
                                         # full rebuilds are listed, not run -- those are step-away jobs

It rebuilds ONLY:
  * seasons that are missing,
  * seasons built with older stat definitions / models ("stale"),
  * the current season during the year (to add the latest games).
Everything already up to date is skipped. Safe to stop with Ctrl+C at any time --
run it again later and it continues where it left off.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import date

FIRST = 2015


def plan(today: date | None = None) -> list[tuple[int, str]]:
    from core import league_ref
    today = today or date.today()
    in_season = 3 <= today.month <= 10
    out = []
    for season in range(today.year, FIRST - 1, -1):          # newest first
        st = league_ref.status(season)
        if st != "current":
            out.append((season, st))
        elif season == today.year and in_season and league_ref.meta(season).get("built", "")[:10] < today.isoformat():
            out.append((season, "new games"))
    return out


def _git(*args) -> int:
    return subprocess.call(["git", *args])


def refresh_stance() -> None:
    """Savant's batting-stance board (point of contact vs the plate): re-downloaded daily during the season."""
    try:
        from core import stance_feed
        season = date.today().year if date.today().month >= 3 else date.today().year - 1
        if not stance_feed.FEED.exists():
            print("Stance board: finding Savant's download options (one time)...", flush=True)
            stance_feed.discover(season)
        if stance_feed.stale(season):
            print(f"Stance board {season}:", flush=True)
            stance_feed.refresh(season)
            _git("add", "data/stance", f"data/batting_stance_{season}.csv")
    except Exception as e:
        print(f"Stance board refresh skipped: {e}", flush=True)


def _weather_stamp(season: int):
    from core import weather
    return weather.ROOT / "data" / "weather" / f"refreshed_{season}.txt"


def weather_stale(season: int, today: date | None = None) -> bool:
    """Refresh at most once a day, and not at all once the season's last game is already in the files."""
    today = today or date.today()
    f = _weather_stamp(season)
    if not f.exists():
        return True
    try:
        built, last_game = f.read_text().split()[:2]
    except ValueError:
        return True
    if built >= today.isoformat():
        return False                                   # already done today
    from datetime import timedelta
    recent = (date.fromisoformat(built) - timedelta(days=3)).isoformat()
    return last_game >= recent                         # games were still being played when we last built


def _mark_weather(season: int) -> None:
    import pandas as pd
    from core import weather
    last = ""
    try:
        g = pd.read_csv(weather.ROOT / "data" / "weather" / f"games_{season}.csv")
        col = next(c for c in ("game_date", "date", "start_utc") if c in g)
        last = str(pd.to_datetime(g[col], errors="coerce", utc=True).max().date())
    except Exception:
        last = date.today().isoformat()
    _weather_stamp(season).write_text(f"{date.today().isoformat()} {last}\n")


def refresh_weather() -> None:
    """Current season's games + reported weather + hourly history at each park (daily during the season)."""
    try:
        from core import weather
        season = date.today().year
        if 3 <= date.today().month <= 11 and weather_stale(season):
            print(f"Weather {season}:", flush=True)
            weather.build_season(season, verbose=False, refresh_hourly=True)
            _mark_weather(season)
            _git("add", "data/weather")
    except Exception as e:
        print(f"Weather refresh skipped: {e}", flush=True)


def main(argv: list[str]) -> None:
    if "--check" not in argv:
        refresh_stance()
        refresh_weather()
    todo = plan()
    if not todo:
        if "--no-push" not in argv and "--check" not in argv and _git("diff", "--cached", "--quiet") != 0:
            _git("commit", "-m", "Data update: stance board")
            _git("push")
        print("Everything is up to date. Nothing to do.")
        return
    print("Out of date:", ", ".join(f"{y} ({why})" for y, why in todo), flush=True)
    if "--check" in argv:
        return
    if "--light" in argv:
        heavy = [y for y, why in todo if why != "new games"]
        todo = [(y, why) for y, why in todo if why == "new games"]
        if heavy:
            print(f"HEAVY: {len(heavy)} season(s) need a full rebuild ({', '.join(map(str, heavy))}) -- "
                  "run  caffeinate -i python -m pipeline.update  when you step away.", flush=True)
        if not todo:
            print("No new games to add.")
            return
    from pipeline import build_league_reference as builder
    from pipeline import power_ref
    for season, why in todo:
        builder.main([str(season), "--force"])
        try:
            power_ref.build(season)                 # Bat Speed Grade reference for that season
        except Exception as e:
            print(f"[{season}] power reference skipped: {e}", flush=True)
        try:
            from pipeline import train_contact
            train_contact.build_ref(season)         # Bat-To-Ball Grade reference for that season
        except Exception as e:
            print(f"[{season}] Bat-To-Ball reference skipped: {e}", flush=True)
        try:
            from pipeline import decision_ref
            decision_ref.build(season)              # Decision Grade references for that season
        except Exception as e:
            print(f"[{season}] Decision Grade reference skipped: {e}", flush=True)
        try:
            from pipeline import flight_ref
            flight_ref.build(season)                # Ball Flight Grade reference for that season
        except Exception as e:
            print(f"[{season}] Ball Flight Grade reference skipped: {e}", flush=True)
    if "--no-push" in argv:
        return
    _git("add", "data/league")          # includes power_<year>.json
    if _git("diff", "--cached", "--quiet") != 0:
        _git("commit", "-m", f"Data update: {', '.join(str(y) for y, _ in todo)}")
        _git("push")
    print("ALL DONE — data is up to date.")


if __name__ == "__main__":
    main(sys.argv[1:])
