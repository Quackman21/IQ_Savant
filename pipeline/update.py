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


def main(argv: list[str]) -> None:
    todo = plan()
    if not todo:
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
    if "--no-push" in argv:
        return
    _git("add", "data/league")          # includes power_<year>.json
    if _git("diff", "--cached", "--quiet") != 0:
        _git("commit", "-m", f"Data update: {', '.join(str(y) for y, _ in todo)}")
        _git("push")
    print("ALL DONE — data is up to date.")


if __name__ == "__main__":
    main(sys.argv[1:])
