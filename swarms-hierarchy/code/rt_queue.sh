#!/bin/bash
# Response-threshold study queue (RT_PREREGISTRATION.md): RT1 -> c choice -> pilots -> gate -> RT2 -> forecast freeze
# -> RT3 -> RT4 -> evaluation. One paced Adaption job at a time; waits for pid $2 (the Rosen phase 2 queue) first.
# The cap is the owner's (150 on 2026-11-19): before each stage ledger + estimate must stay at least 1 credit below it.
# Restart-safe: every stage skips calls and populations already written.
# Usage: bash code/rt_queue.sh <ledger-cap> [pid-to-wait-for]  > scratch/rt_queue.log 2>&1
cd "$(dirname "$0")/.." || exit 1
MAIN=/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy
PY=$MAIN/../.venv/bin/python
[ -n "$1" ] || { echo "=== no ledger cap given; not starting"; exit 2; }
CAP=$1
fail() { echo "=== STAGE FAILED rc=$1 $(date '+%F %T')"; exit "$1"; }
ledger() { $PY -c "import json;print(json.load(open('$MAIN/results/adaption_spend_total.json'))['total'])"; }
guard() {  # $1 stage name, $2 estimated credits
  local t; t=$(ledger)
  if $PY -c "import sys; sys.exit(0 if $t + $2 < $CAP - 1 else 1)"; then
    echo "=== $(date '+%F %T') $1 (ledger $t, estimate $2, cap $CAP)"
  else
    echo "=== $(date '+%F %T') STOP before $1: ledger $t + estimate $2 would come within 1 of the cap $CAP"; exit 7
  fi
}
if [ -n "$2" ]; then
  echo "=== $(date '+%F %T') waiting for pid $2"
  while kill -0 "$2" 2>/dev/null; do sleep 60; done
  echo "=== $(date '+%F %T') pid $2 exited"
fi
[ -f results/rt_prereg_freeze.json ] || { echo "=== pre-registration not frozen; not starting"; exit 5; }
unset RT_FAKE RT_RESULTS
set -a; . "$MAIN/.secrets/adaption.env"; set +a
export ADAPTION_CREDIT_CAP=$CAP ADAPTION_LEDGER_DIR=$MAIN/results ADAPTION_RPM=48 ADAPTION_429_SLEEP=30 ADAPTION_MAX_TRIES=40 LLM_MAX_CONC=12
M="adaption:gemma-4-31b-it adaption:qwen3.8-27b"
P="$PY -u code/chase_paced.py"

guard "RT1 comprehension + calibration" 0.15
$P rt_kernel comp --models $M < /dev/null || fail $?
$P rt_kernel calib --models $M < /dev/null || fail $?
$PY code/rt_kernel.py choose --models $M || fail $?
guard "RT1 pilots" 0.1
$P rt run --stage pilot --models $M < /dev/null || fail $?
$PY code/rt_kernel.py gate --models $M || fail $?
guard "RT2 kernel" 0.7
$P rt_kernel grid --models $M < /dev/null || fail $?
$P rt_kernel replay --models $M < /dev/null || fail $?
$P rt_kernel social --models $M < /dev/null || fail $?
$P rt_kernel topup --models $M < /dev/null || fail $?
$PY code/rt_kernel.py summary
if [ ! -f results/rt_predictions.json ]; then
  echo "=== $(date '+%F %T') freezing forecast"
  $PY -W ignore code/rt_forecast.py freeze --models $M --B 500 --procs 8 || fail $?
fi
guard "RT3 populations" 1.6
$P rt run --stage RT3 --models $M < /dev/null || fail $?
R4=$($PY -c "import json;print(json.load(open('results/rt_predictions.json'))['rt4_model'] or '')")
if [ -n "$R4" ]; then
  guard "RT4 group size ($R4)" 0.3
  $P rt run --stage RT4 --models $R4 < /dev/null || fail $?
else
  echo "=== RT4 not run: no model with tau above 0"
fi
$PY -W ignore code/rt_eval.py || fail $?
echo "=== $(date '+%F %T') RT queue done (ledger $(ledger))"
