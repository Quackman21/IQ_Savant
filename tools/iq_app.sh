#!/bin/bash
# IQ Savant desktop app -- what the Desktop icon runs.
#   start    sync (install new version zips in ~/Dev, test, commit, pull, push), start the site, open it,
#            keep the data fresh while it's open
#   open     open the site in the browser          stop     stop the site and the updater
#   status   "running" or "stopped"                restart  stop + start
# Log: data/app_log.txt
if [ -z "$IQ_APP_REAL" ]; then           # run from a copy, so installing a new version can't change this file mid-run
  tmp="$(mktemp "${TMPDIR:-/tmp}/iq_app.XXXXXX")"; cp "$0" "$tmp"
  IQ_APP_REAL="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" exec bash "$tmp" "$@"
fi
export IQ_APP_REAL
SELF="$IQ_APP_REAL"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"
cd "$REPO" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export GIT_TERMINAL_PROMPT=0
PY="$REPO/.venv/bin/python"
PORT=8501
URL="http://localhost:$PORT"
LOG="$REPO/data/app_log.txt"
LOOP_PID="$REPO/data/app_loop.pid"
EVERY=$((3 * 3600))          # data check every 3 hours while the site is open
mkdir -p "$REPO/data"

log() { echo "[$(date '+%Y-%m-%d %H:%M')] $*" >> "$LOG"; }
notify() { osascript -e "display notification \"$1\" with title \"IQ Savant\"" >/dev/null 2>&1; }
site_up() { lsof -ti "tcp:$PORT" -sTCP:LISTEN >/dev/null 2>&1; }

pull() {         # download the latest, keeping any local changes on top
  if git pull --rebase --autostash >> "$LOG" 2>&1; then log "git pull ok"
  else
    git rebase --abort >> "$LOG" 2>&1
    log "git pull FAILED (conflict) -- using what's here"; notify "Couldn't download the latest from GitHub (conflict) -- see data/app_log.txt"
  fi
}

install_zips() { # any IQ_Savant_v*.zip in ~/Dev: unzip over the project (oldest version first), then delete it
  local dev; dev="$(dirname "$REPO")"; local n=0
  while IFS= read -r z; do
    [ -f "$z" ] || continue
    if unzip -o -q "$z" -d "$REPO" >> "$LOG" 2>&1; then rm -f "$z"; log "installed $(basename "$z")"; n=$((n + 1))
    else log "couldn't unzip $(basename "$z")"; fi
  done < <(ls "$dev"/IQ_Savant_v*.zip 2>/dev/null | sort -V)
  [ $n -gt 0 ] && notify "Installed $n new version(s): v$(cat "$REPO/VERSION" 2>/dev/null)"
  return 0
}

sync() {         # install new versions -> test -> commit -> download latest -> push
  install_zips
  if ! "$PY" -m pytest -q -x >> "$LOG" 2>&1; then
    log "TESTS FAILED -- not pushing"; notify "Tests failed -- nothing pushed. Site still starts; see data/app_log.txt"
    pull; return
  fi
  git add -A >> "$LOG" 2>&1
  if ! git diff --cached --quiet; then
    git commit -q -m "Auto-sync from $(scutil --get ComputerName 2>/dev/null || hostname): v$(cat VERSION 2>/dev/null)" >> "$LOG" 2>&1
    log "committed local changes"
  fi
  pull
  if git push -q >> "$LOG" 2>&1; then log "git push ok"; else log "git push FAILED"; notify "Couldn't push to GitHub -- see data/app_log.txt"; fi
}

refresh() {      # light update: new games only; tells you when a full rebuild is due
  if [ -f "$REPO/data/overnight.lock" ]; then log "overnight job running -- skipping this data check"; return; fi
  pull
  out="$(caffeinate -i "$PY" -m pipeline.update --light 2>&1)"
  echo "$out" >> "$LOG"
  if echo "$out" | grep -q "^HEAVY:"; then
    notify "Some seasons need a full rebuild -- run the update when you step away."
  elif echo "$out" | grep -q "ALL DONE"; then
    notify "New games added."
  fi
}

loop() {
  echo $$ > "$LOOP_PID"
  sleep 120                                   # let the site load first
  while site_up; do
    refresh
    for _ in $(seq 1 $((EVERY / 60))); do site_up || break; sleep 60; done
  done
  rm -f "$LOOP_PID"
  log "site stopped -- updater stopped"
}

start() {
  if site_up; then open "$URL"; exit 0; fi
  if [ ! -x "$PY" ]; then osascript -e "display alert \"IQ Savant\" message \"No .venv found in $REPO\""; exit 1; fi
  log "===== starting IQ Savant ====="
  sync
  nohup "$PY" -m streamlit run app.py --server.headless true --server.port "$PORT" >> "$LOG" 2>&1 &
  for _ in $(seq 1 90); do site_up && break; sleep 1; done
  if site_up; then
    open "$URL"; log "site up at $URL"
  else
    log "site did NOT start -- see above"; notify "The site didn't start -- see data/app_log.txt"; exit 1
  fi
  nohup env -u IQ_APP_REAL bash "$SELF" loop >> "$LOG" 2>&1 &
}

stop() {
  [ -f "$LOOP_PID" ] && kill "$(cat "$LOOP_PID")" 2>/dev/null; rm -f "$LOOP_PID"
  pkill -f "$PY -m pipeline.update --light" 2>/dev/null
  pids="$(lsof -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null)"; [ -n "$pids" ] && kill $pids 2>/dev/null
  log "stopped"
}

case "${1:-start}" in
  start) start ;;
  open) open "$URL" ;;
  stop) stop ;;
  restart) stop; sleep 2; start ;;
  loop) loop ;;
  refresh) refresh ;;
  sync) sync ;;
  status) if site_up; then echo running; else echo stopped; fi ;;
  *) echo "usage: $SELF [start|open|stop|restart|status|refresh|sync]"; exit 1 ;;
esac
