#!/bin/bash
# RT-R (RT_REST_PREREGISTRATION.md): the RT design with the rest-reward framing (RT_FRAME=rest), Gemma-4-31B and
# Qwen3.8-27B, results in results/rest/. RT1 (r chosen from {0.5, 1, 2}), pilots, gate, RT2, write-once forecast,
# RT3, RT4 (registered rule), evaluation. Study guard 4 credits; the ledger stays 1 below the cap.
# Usage: bash code/rt_rest_queue.sh <ledger-cap>      (DRY=1 with RT_FAKE=1 RT_RESULTS=<dir>: scripted agents)
cd "$(dirname "$0")/.." || exit 1
MAIN=/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy
PY=$MAIN/../.venv/bin/python
[ -n "$1" ] || { echo "=== no ledger cap given; not starting"; exit 2; }
CAP=$1; GUARD=4
export RT_FRAME=rest
fail() { echo "=== STAGE FAILED rc=$1 $(date '+%F %T')"; exit "$1"; }
if [ -n "$DRY" ]; then
  ledger() { echo 0; }
  P="$PY -u code/rt_dry.py"
else
  unset RT_FAKE RT_FORECAST
  export RT_RESULTS=$PWD/results/rest
  set -a; . "$MAIN/.secrets/adaption.env"; set +a
  export ADAPTION_CREDIT_CAP=$CAP ADAPTION_LEDGER_DIR=$MAIN/results ADAPTION_RPM=48 ADAPTION_429_SLEEP=30 ADAPTION_MAX_TRIES=40 LLM_MAX_CONC=12
  ledger() { $PY -c "import json;print(json.load(open('$MAIN/results/adaption_spend_total.json'))['total'])"; }
  P="$PY -u code/chase_paced.py"
fi
unset RT_FORECAST
mkdir -p "$RT_RESULTS"
$PY code/rt_freeze.py check "$RT_RESULTS/rt_prereg_freeze.json" || { echo "=== RT-R freeze missing or files changed; not starting"; exit 5; }
START=$(ledger)
guard() {  # $1 stage name, $2 estimated credits
  local t; t=$(ledger)
  if ! $PY -c "import sys; sys.exit(0 if $t + $2 < $CAP - 1 and $t - $START + $2 <= $GUARD else 1)"; then
    echo "=== $(date '+%F %T') STOP before $1: ledger $t (start $START) + estimate $2 vs cap $CAP / study guard $GUARD"; exit 7
  fi
  echo "=== $(date '+%F %T') $1 (ledger $t, study so far $($PY -c "print(round($t - $START, 3))"), estimate $2, cap $CAP)"
}
M="adaption:gemma-4-31b-it adaption:qwen3.8-27b"

guard "RT-R RT1 comprehension + calibration" 0.15
$P rt_kernel comp --models $M < /dev/null || fail $?
$P rt_kernel calib --models $M < /dev/null || fail $?
$PY code/rt_kernel.py choose --models $M || fail $?
guard "RT-R pilots" 0.1
$P rt run --stage pilot --models $M < /dev/null || fail $?
$PY code/rt_kernel.py gate --models $M || fail $?
guard "RT-R RT2 kernel" 0.7
$P rt_kernel grid --models $M < /dev/null || fail $?
$P rt_kernel replay --models $M < /dev/null || fail $?
$P rt_kernel social --models $M < /dev/null || fail $?
$P rt_kernel topup --models $M < /dev/null || fail $?
$PY code/rt_kernel.py summary
if [ ! -f "$RT_RESULTS/rt_predictions.json" ]; then
  echo "=== $(date '+%F %T') freezing RT-R forecast"
  $PY -W ignore code/rt_forecast.py freeze --models $M --B 500 --procs 8 || fail $?
fi
guard "RT-R RT3 populations" 1.6
$P rt run --stage RT3 --models $M < /dev/null || fail $?
R4=$($PY -c "import json;print(json.load(open('$RT_RESULTS/rt_predictions.json'))['rt4_model'] or '')")
if [ -n "$R4" ]; then
  guard "RT-R RT4 group size ($R4)" 0.3
  $P rt run --stage RT4 --models $R4 < /dev/null || fail $?
else
  echo "=== RT-R RT4 not run: no model with tau above 0"
fi
$PY -W ignore code/rt_eval.py || fail $?
echo "=== $(date '+%F %T') RT-R queue done (ledger $(ledger))"
