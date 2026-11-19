"""One long job: everything the site has been built on, for the whole bat-tracking era (2023 -> now).

    caffeinate -i .venv/bin/python -m pipeline.backfill_era            # every season
    caffeinate -i .venv/bin/python -m pipeline.backfill_era 2025 2026  # just these
    ... --force                                                        # redo steps that already finished

Per season, in order (each step skips itself when its output already exists, so it's safe to stop and re-run):
  1. Savant's batting-stance board -- season, playoff, vs LHP / RHP (point of contact vs the plate, every hitter)
  2. every MLB contact swing (pipeline.fetch_league_swings; ~30-60 min a season, saved to ~/Desktop/swing_data)
  3. contact-type calibration -- where squared-up balls leave vs the bat's path, league-wide
     -> data/calibration/batted_ball_<season>.json
  4. hip-position fallback for hitters not on the stance board -> data/hip_group_<season>.json
  5. weather: every park (location, elevation, field bearing, roof) + every game's reported weather + hourly history
     -> data/weather/; then a check that it explains carry (tailwind -> longer, thinner air -> longer)
  6. Bat Path norms (fast swings) + 7. the bat cube (every competitive swing)
  8. the bat path through the zone by contact depth -> path_fit_<season>.json
  9. bat-ball collision calibration -> collision_<season>.json
  (the swing simulator's spray model is one file, trained on the newest season: python -m core.swing_sim <season>)
Then the regular data update (league reference + grades for every season that needs it) and one commit + push.
A log of the whole run goes to data/calibration/backfill_log.txt.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "data" / "calibration"
FIRST = 2023


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    CAL.mkdir(parents=True, exist_ok=True)
    with open(CAL / "backfill_log.txt", "a") as f:
        f.write(line + "\n")


def season_steps(season: int, force: bool) -> None:
    from core import batted_ball, stance_feed
    from pipeline import fetch_league_swings, hip_group

    log(f"===== {season} =====")
    # 1 stance board
    if force or not (stance_feed.DIR / str(season) / "refreshed.txt").exists() or season == date.today().year:
        if not stance_feed.FEED.exists():
            log("stance board: finding Savant's download options")
            stance_feed.discover(season, verbose=False)
        wrote = stance_feed.refresh(season, verbose=False)
        log(f"stance board: {', '.join(wrote) or 'nothing (Savant has no board for this season)'}")
    else:
        log("stance board: already have it")
    # 2 league contact swings
    lg_file = fetch_league_swings.DEFAULT_DIR / f"league_{season}.csv.gz"
    if force or not lg_file.exists() or season == date.today().year:
        log("league contact swings: downloading (resumable)")
        fetch_league_swings.fetch(season, verbose=True)
    lg = pd.read_csv(lg_file, low_memory=False)
    log(f"league contact swings: {len(lg):,} ({lg['batter'].nunique() if len(lg) else 0} hitters)")
    tracked = len(lg) and "attack_angle" in lg and lg["attack_angle"].notna().any()
    if not tracked:
        log("no bat tracking this season -- skipping the calibrations")
        return
    # 3 contact-type calibration
    out = CAL / f"batted_ball_{season}.json"
    if force or not out.exists():
        c = batted_ball.calibrate(lg)
        out.write_text(json.dumps(c))
        log(f"contact-type centres: launch {c['v_center']:+.1f} deg vs attack angle, spray {c['h_center']:+.1f} deg "
            f"vs attack direction ({c['n_squared_up']:,} squared-up balls)")
    else:
        log("contact-type centres: already have them")
    # 5 weather: parks, schedule + reported weather, hourly history; then check weather explains carry
    from core import weather
    wf = weather.DIR / f"games_{season}.csv"
    if force or not wf.exists() or season == date.today().year:
        weather.build_season(season, verbose=False, refresh_hourly=season == date.today().year)
        n = len(pd.read_csv(wf)) if wf.exists() else 0
        log(f"weather: {n:,} games, {len(list((weather.DIR / str(season)).glob('*.csv')))} parks with hourly history")
    else:
        log("weather: already have it")
    chk = CAL / f"carry_{season}.json"
    old = json.loads(chk.read_text()) if chk.exists() else {}
    if force or "by_source" not in old:
        res = {}
        for src in ("hourly", "reported"):
            c = weather.carry_check(lg, season, wind=src)
            res[src] = c
            if c.get("ft_per_mph_tailwind") is not None:
                log(f"weather check, {src} wind ({c['n']:,} well-hit fly balls): {c['ft_per_mph_tailwind']:+.2f} ft per "
                    f"mph of tailwind, {c['ft_per_pct_density']:+.2f} ft per 1% denser air"
                    + (f", thin-air parks {c['coors_minus_avg_ft']:+.0f} ft vs average" if c.get("coors_minus_avg_ft") else ""))
        chk.write_text(json.dumps({"by_source": res}))
    # 6 Bat Path normative ranges: MLB fast swings (75+ mph) by pitch group x zone x contact depth
    bn = CAL / f"bat_norms_{season}.json"
    if force or not bn.exists():
        try:
            from pipeline import bat_norms
            bat_norms.run(season)
            log(f"bat path norms: {json.loads(bn.read_text())['swings']['fast']:,} fast swings")
        except Exception as e:
            log(f"bat path norms skipped: {type(e).__name__}: {e}")
    # 7 the bat cube: every competitive swing with depth, plane match + bat angles (any hitter x combo, quickly)
    from pipeline import bat_cube
    cube_f = [bat_cube.OUT / f"swings_{season}.parquet", bat_cube.OUT / f"swings_{season}.csv.gz"]
    if force or not any(f.exists() for f in cube_f):
        try:
            bat_cube.run(season)
        except Exception as e:
            log(f"bat cube skipped: {type(e).__name__}: {e}")
    # 8 the bat path through the zone (league + per-hitter rates by contact depth) -> data/calibration/path_fit_<season>
    pf = CAL / f"path_fit_{season}.json"
    if force or not pf.exists():
        try:
            from core import path_fit
            r = path_fit.run(season)
            a = (r.get("league") or {}).get("all") or {}
            log(f"bat path rates: attack angle {a.get('attack_angle', {}).get('slope', float('nan')):+.3f} deg/in, "
                f"direction {a.get('attack_direction', {}).get('slope', float('nan')):+.3f} deg/in")
        except Exception as e:
            log(f"bat path rates skipped: {type(e).__name__}: {e}")
    else:
        log("bat path rates: already have them")
    # 9 bat-ball collision calibration (needs 8) -> data/calibration/collision_<season>.json
    cf = CAL / f"collision_{season}.json"
    if force or not cf.exists():
        try:
            from core import collision
            r = collision.run(season)
            log(f"collision: q {(r.get('measured') or {}).get('q', float('nan')):.3f} on {r.get('balls', 0):,} balls")
        except Exception as e:
            log(f"collision skipped: {type(e).__name__}: {e}")
    else:
        log("collision: already have it")
    # 4 hip fallback
    if force or not (ROOT / "data" / f"hip_group_{season}.json").exists():
        hip_group.main([str(lg_file)])
        log("hip-position fallback: done")
    else:
        log("hip-position fallback: already have it")


def main(argv: list[str]) -> None:
    force = "--force" in argv
    yrs = [int(a) for a in argv if a.isdigit()]
    today = date.today()
    last = today.year if today.month >= 3 else today.year - 1
    seasons = yrs or list(range(FIRST, last + 1))
    t0 = time.time()
    log(f"backfill: seasons {seasons}")
    for season in seasons:
        try:
            season_steps(season, force)
        except Exception as e:                       # one season failing shouldn't stop the others
            log(f"{season} FAILED: {type(e).__name__}: {e} -- re-run to retry")
    log("regular data update (league reference + grades)")
    try:
        from pipeline import update
        update.main([])
    except SystemExit:
        pass
    except Exception as e:
        log(f"data update FAILED: {type(e).__name__}: {e}")
    try:
        from pipeline.update import _git
        paths = ["data/calibration", "data/stance", "data/weather", *[f"data/hip_group_{s}.json" for s in seasons],
                 *[f"data/batting_stance_{s}.csv" for s in seasons]]
        _git("add", *[p for p in paths if (ROOT / p).exists()])
        if _git("diff", "--cached", "--quiet") != 0:
            _git("commit", "-m", f"Backfill: bat-tracking era {seasons[0]}-{seasons[-1]}")
            _git("push")
    except Exception as e:
        log(f"commit/push skipped: {e}")
    # wind: keep the source that explains carry (closest to the physics, ~3 ft per mph of tailwind)
    try:
        got = {}
        for s_ in seasons:
            f = CAL / f"carry_{s_}.json"
            for src, c in (json.loads(f.read_text()).get("by_source", {}) if f.exists() else {}).items():
                if c.get("ft_per_mph_tailwind") is not None:
                    got.setdefault(src, []).append(c["ft_per_mph_tailwind"])
        if got:
            best = min(got, key=lambda k: abs(np.mean(got[k]) - 3.0))
            (CAL / "wind_source.json").write_text(json.dumps({"source": best, "ft_per_mph": {k: round(float(np.mean(v)), 2)
                                                                                           for k, v in got.items()}}))
            log(f"wind source: {best} ({', '.join(f'{k} {np.mean(v):+.2f} ft/mph' for k, v in got.items())})")
    except Exception as e:
        log(f"wind source pick skipped: {e}")
    log(f"backfill done in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main(sys.argv[1:])
