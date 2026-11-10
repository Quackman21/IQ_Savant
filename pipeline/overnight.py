"""The overnight job -- one command, run it when you step away:

    caffeinate -i python -m pipeline.overnight

  0. git pull (pick up anything pushed from the other computer)
  1. Download every pitch, 2015 -> latest finished season, ONCE (saved in data/raw/, ~2-3 GB, not pushed).
     Seasons already saved are skipped, so a stopped run picks up where it left off.
  2. Train SAxwOBA -- two models, shift era (2015-2022) and shift ban (2023+), no carryover.
  3. Train IQ Swing Decisions -- two models, 2015-2022 and 2023+ -- and switch it on.
  4. Commit + push the trained models.
  5. Rebuild every season's MLB averages from the saved downloads (no second download) and push.

Everything is logged to data/overnight_log.txt. A step that fails is logged and the rest still run.
"""
from __future__ import annotations

import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "overnight_log.txt"
LOCK = ROOT / "data" / "overnight.lock"
FIRST = 2015


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def git(*args) -> int:
    return subprocess.call(["git", *args], cwd=ROOT)


def step(name: str, fn, results: list) -> None:
    t0 = time.time()
    log(f"START  {name}")
    try:
        fn()
        results.append((name, "done", time.time() - t0))
        log(f"DONE   {name}  ({(time.time() - t0) / 60:.0f} min)")
    except Exception as e:
        results.append((name, f"FAILED: {type(e).__name__}: {e}", time.time() - t0))
        log(f"FAILED {name}: {type(e).__name__}: {e}")
        with LOG.open("a") as f:
            f.write(traceback.format_exc() + "\n")


def _running(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def main(argv: list[str]) -> None:
    import os
    if LOCK.exists():
        try:
            other = int(LOCK.read_text().strip())
        except ValueError:
            other = -1
        if _running(other):
            print(f"Another overnight job is already running (process {other}). Not starting a second one -- two "
                  "jobs at once fight over the same files. Wait for it to finish (or stop it), then run again.")
            return
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()))
    try:
        _run(argv)
    finally:
        LOCK.unlink(missing_ok=True)


def _run(argv: list[str]) -> None:
    from pipeline import build_league_reference as bl
    from pipeline import train_decisions, train_saxwoba, update
    last = train_decisions.latest_full_season()
    results: list = []
    log(f"===== OVERNIGHT JOB: {FIRST}-{last} =====")

    git("config", "http.postBuffer", "524288000")       # big data pushes (default limit fails with HTTP 400)
    step("git pull", lambda: git("pull"), results)

    def download():
        for y in range(FIRST, last + 1):
            p = bl.ensure_season(y)
            log(f"  {y}: {'saved' if p else 'NOT saved'}")
    step(f"download every pitch {FIRST}-{last}", download, results)

    step("train SAxwOBA (shift era + shift ban)", lambda: train_saxwoba.main(["--since", str(FIRST)]), results)
    step("train IQ Swing Decisions (2015-2022 + 2023+) and switch on",
         lambda: train_decisions.main(["--since", str(FIRST), "--activate"]), results)

    def push_models():
        git("add", "-A")
        if git("diff", "--cached", "--quiet") != 0:
            git("commit", "-m", "Overnight: SAxwOBA + IQ Swing Decisions trained on 2015+ (era models)")
        git("push")
    step("commit + push the models", push_models, results)

    step("rebuild MLB averages (from saved downloads) + push", lambda: update.main([]), results)

    log("===== SUMMARY =====")
    for name, status, secs in results:
        log(f"  {status:<8.60} {name}  ({secs / 60:.0f} min)")
    log("Reports: models/artifacts/saxwoba_report.txt, decisions_iq_v1_report.txt, decisions_video_check.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
