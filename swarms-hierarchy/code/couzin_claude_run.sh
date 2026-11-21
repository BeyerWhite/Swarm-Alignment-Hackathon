#!/bin/bash
# Couzin amendment M-A1: the M4b design on Claude models through the `claude` CLI (COUZIN_CLAUDE_FOLLOWUP.md).
# No Adaption calls, so it neither waits for nor blocks the Adaption queues. Its name deliberately avoids the
# "queue" pattern other studies' wrappers wait on. Models run one after another (Haiku, then Sonnet).
# Stops on the first usage-limit message (exit 75): rerun the same command after the limit resets; every runner
# skips rows already written and the write-once steps are skipped when their file exists.
# Usage: nohup caffeinate -i bash code/couzin_claude_run.sh > scratch/couzin_claude_run.log 2>&1 &
#        DRY=1 bash code/couzin_claude_run.sh    (scripted agents in scratch/couzin_followup_dry; no CLI calls)
cd "$(dirname "$0")/.." || exit 1
MAIN=/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy
PY=$MAIN/../.venv/bin/python
MODELS=${COUZIN_CLAUDE_MODELS:-"claude:haiku claude:sonnet"}
if [ "$DRY" = 1 ]; then
  export COUZIN_FAKE=1 COUZIN_RESULTS=$PWD/scratch/couzin_followup_dry DRYF=${DRYF:-4}
else
  export COUZIN_RESULTS=$PWD/results/couzin_claude
  [ -f results/couzin_followup_freeze.json ] || { echo "=== amendment not stamped; not starting"; exit 5; }
  $PY code/couzin_followup.py check || { echo "=== amendment files changed; not starting"; exit 5; }
fi
mkdir -p "$COUZIN_RESULTS"
export LLM_MAX_CONC=${LLM_MAX_CONC:-8}
F="$PY -u code/couzin_followup.py"
run() {
  echo "=== $(date '+%F %T') $*"
  $F "$@" < /dev/null; local rc=$?
  if [ "$rc" = 75 ]; then echo "=== $(date '+%F %T') QUOTA STOP in '$*': rerun this script after the usage limit resets"; exit 75; fi
  [ "$rc" = 0 ] || { echo "=== STAGE FAILED rc=$rc $(date '+%F %T')"; exit "$rc"; }
}
run gate --models $MODELS
for M in $MODELS; do
  if ! $PY -c "import json,sys; sys.exit(0 if '$M' in json.load(open('$COUZIN_RESULTS/couzin_m0_gate.json'))['passed'] else 1)"; then
    echo "=== $M failed the M0 gate: skipped"; continue
  fi
  run kernel --model "$M"
  run rules --model "$M"
  run calib --model "$M"
  run forecast --model "$M"
  run pop --model "$M"
done
run eval
echo "=== QUEUE DONE $(date '+%F %T')"
