#!/bin/bash
# Auto-resume for the Claude follow-up (owner OK 2026-11-24): reruns code/couzin_claude_run.sh after every
# stop-on-quota exit (75), waiting $WAIT seconds between attempts, at most $TRIES attempts. Stops on success (0) or on
# any other failure. Optionally first waits for an already-running runner (pid $1) and reads how it ended from its log.
# The name avoids the "queue" pattern other studies' wrappers wait on (this job uses no Adaption credits).
# Usage: nohup caffeinate -i bash code/couzin_claude_autoresume.sh [running-pid] > scratch/couzin_claude_autoresume.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
WAIT=${WAIT:-1800}; TRIES=${TRIES:-60}; LOG=scratch/couzin_claude_run.log
if [ -n "$1" ]; then
  echo "=== $(date '+%F %T') waiting for the running runner (pid $1)"
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
  if grep -q "QUEUE DONE" "$LOG"; then echo "=== $(date '+%F %T') runner finished; nothing to resume"; exit 0; fi
  if ! grep -q "QUOTA STOP in" "$LOG"; then echo "=== $(date '+%F %T') runner ended without a quota stop; not resuming (see $LOG)"; exit 3; fi
  echo "=== $(date '+%F %T') runner stopped on the usage limit"; sleep "$WAIT"
fi
for i in $(seq 1 "$TRIES"); do
  cat "$LOG" >> scratch/couzin_claude_run_all.log 2>/dev/null
  echo "=== $(date '+%F %T') attempt $i"
  bash code/couzin_claude_run.sh > "$LOG" 2>&1; rc=$?
  if [ "$rc" = 0 ]; then echo "=== $(date '+%F %T') follow-up complete"; exit 0; fi
  if [ "$rc" != 75 ]; then echo "=== $(date '+%F %T') runner failed rc=$rc; not resuming (see $LOG)"; exit "$rc"; fi
  echo "=== $(date '+%F %T') usage limit; next attempt in $WAIT s"; sleep "$WAIT"
done
echo "=== $(date '+%F %T') gave up after $TRIES attempts"; exit 4
