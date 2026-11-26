#!/bin/bash
# Couzin amendment M-A4: the M4b design on CLI models (same procedure as M-A1). Generalises couzin_claude_run.sh:
# results in results/couzin_<FAMILY> (FAMILY = first word before ":" of the first model, e.g. gpt).
# No Adaption calls. Stops on any usage-limit message or empty CLI reply (exit 75); rerun to resume.
# Usage: COUZIN_CLI_MODELS="gpt:gpt-6-luna" nohup caffeinate -i bash code/couzin_cli_run.sh > scratch/couzin_gpt_run.log 2>&1 &
#        DRY=1 COUZIN_CLI_MODELS=... bash code/couzin_cli_run.sh
cd "$(dirname "$0")/.." || exit 1
MAIN=/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy
PY=$MAIN/../.venv/bin/python
MODELS=${COUZIN_CLI_MODELS:?set COUZIN_CLI_MODELS}
FAM=${MODELS%%:*}
if [ "$DRY" = 1 ]; then
  export COUZIN_FAKE=1 COUZIN_RESULTS=$PWD/scratch/couzin_${FAM}_dry DRYF=${DRYF:-4}
else
  export COUZIN_RESULTS=$PWD/results/couzin_${FAM}
  [ -f results/couzin_followup_freeze.json ] || { echo "=== amendment not stamped; not starting"; exit 5; }
  $PY code/couzin_followup.py check || { echo "=== amendment files changed; not starting"; exit 5; }
fi
mkdir -p "$COUZIN_RESULTS"
export LLM_MAX_CONC=${LLM_MAX_CONC:-6} GPT_CALL_BUDGET=${GPT_CALL_BUDGET:-25000}
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
