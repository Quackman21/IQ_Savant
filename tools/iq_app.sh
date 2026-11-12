#!/bin/bash
# IQ Savant desktop app -- what the Desktop icon runs.
#   start    pull the latest code, start the site, open it, keep the data fresh while it's open
#   open     open the site in the browser          stop     stop the site and the updater
#   status   "running" or "stopped"                restart  stop + start
# Log: data/app_log.txt
REPO="$(cd "$(dirname "$0")/.." && pwd)"
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

pull() {
  if git pull --ff-only >> "$LOG" 2>&1; then log "git pull ok"; else log "git pull FAILED (local changes?) -- using what's here"; fi
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
  pull
  nohup "$PY" -m streamlit run app.py --server.headless true --server.port "$PORT" >> "$LOG" 2>&1 &
  for _ in $(seq 1 90); do site_up && break; sleep 1; done
  if site_up; then
    open "$URL"; log "site up at $URL"
  else
    log "site did NOT start -- see above"; notify "The site didn't start -- see data/app_log.txt"; exit 1
  fi
  nohup "$0" loop >> "$LOG" 2>&1 &
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
  status) if site_up; then echo running; else echo stopped; fi ;;
  *) echo "usage: $0 [start|open|stop|restart|status|refresh]"; exit 1 ;;
esac
