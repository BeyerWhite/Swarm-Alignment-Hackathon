#!/bin/bash
# Auto-resume for a CLI follow-up (M-A4; same rule as couzin_claude_autoresume.sh): reruns
# `bash code/couzin_cli_run.sh` (with COUZIN_CLI_MODELS from the environment) after every stop-on-quota exit (75),
# waiting $WAIT s, at most $TRIES attempts; stops on success or any other failure.
# Usage: COUZIN_CLI_MODELS="gpt:gpt-6-luna" nohup caffeinate -i bash code/couzin_cli_autoresume.sh > scratch/couzin_gpt_autoresume.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
: "${COUZIN_CLI_MODELS:?set COUZIN_CLI_MODELS}"
FAM=${COUZIN_CLI_MODELS%%:*}
WAIT=${WAIT:-1800}; TRIES=${TRIES:-60}; LOG=scratch/couzin_${FAM}_run.log
for i in $(seq 1 "$TRIES"); do
  cat "$LOG" >> "scratch/couzin_${FAM}_run_all.log" 2>/dev/null
  echo "=== $(date '+%F %T') attempt $i"
  bash code/couzin_cli_run.sh > "$LOG" 2>&1; rc=$?
  if [ "$rc" = 0 ]; then echo "=== $(date '+%F %T') follow-up complete"; exit 0; fi
  if [ "$rc" != 75 ]; then echo "=== $(date '+%F %T') runner failed rc=$rc; not resuming (see $LOG)"; exit "$rc"; fi
  echo "=== $(date '+%F %T') usage limit or empty CLI reply; next attempt in $WAIT s"; sleep "$WAIT"
done
echo "=== $(date '+%F %T') gave up after $TRIES attempts"; exit 4
