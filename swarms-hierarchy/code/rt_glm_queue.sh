#!/bin/bash
# Amendment RT-A1 (RT_LOG.md): GLM-5.2 on the frozen RT design (effort frame, c = 0.75 as already chosen).
# RT1 (comprehension, calibration: descriptive, c stays 0.75), pilots, gate (added to the earlier gate), RT2, its own
# write-once forecast results/rt_predictions_glm.json, RT3, evaluation -> results/rt_eval_glm.json. RT4 is not run.
# Study guard: this queue's own spend stays below 20 credits; the ledger stays 1 below the cap.
# Usage: bash code/rt_glm_queue.sh <ledger-cap>      (DRY=1 with RT_FAKE=1 RT_RESULTS=<dir>: scripted agents)
cd "$(dirname "$0")/.." || exit 1
MAIN=/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy
PY=$MAIN/../.venv/bin/python
[ -n "$1" ] || { echo "=== no ledger cap given; not starting"; exit 2; }
CAP=$1; GUARD=20
RES=${RT_RESULTS:-$PWD/results}
fail() { echo "=== STAGE FAILED rc=$1 $(date '+%F %T')"; exit "$1"; }
if [ -n "$DRY" ]; then
  ledger() { echo 0; }
  P="$PY -u code/rt_dry.py"
else
  unset RT_FAKE RT_RESULTS; RES=$PWD/results
  set -a; . "$MAIN/.secrets/adaption.env"; set +a
  export ADAPTION_CREDIT_CAP=$CAP ADAPTION_LEDGER_DIR=$MAIN/results ADAPTION_RPM=48 ADAPTION_429_SLEEP=30 ADAPTION_MAX_TRIES=40 LLM_MAX_CONC=12
  ledger() { $PY -c "import json;print(json.load(open('$MAIN/results/adaption_spend_total.json'))['total'])"; }
  P="$PY -u code/chase_paced.py"
fi
$PY code/rt_freeze.py check "$RES/rt_A1_freeze.json" || { echo "=== RT-A1 freeze missing or files changed; not starting"; exit 5; }
START=$(ledger)
guard() {  # $1 stage name, $2 estimated credits
  local t; t=$(ledger)
  if ! $PY -c "import sys; sys.exit(0 if $t + $2 < $CAP - 1 and $t - $START + $2 <= $GUARD else 1)"; then
    echo "=== $(date '+%F %T') STOP before $1: ledger $t (start $START) + estimate $2 vs cap $CAP / study guard $GUARD"; exit 7
  fi
  echo "=== $(date '+%F %T') $1 (ledger $t, study so far $($PY -c "print(round($t - $START, 3))"), estimate $2, cap $CAP)"
}
G=adaption:glm-5.2
export RT_FORECAST=$RES/rt_predictions_glm.json

guard "RT-A1 RT1 comprehension + calibration (GLM)" 0.6
$P rt_kernel comp --models $G < /dev/null || fail $?
$P rt_kernel calib --models $G < /dev/null || fail $?
guard "RT-A1 pilots (GLM)" 0.4
$P rt run --stage pilot --models $G < /dev/null || fail $?
$PY code/rt_kernel.py gate --models $G || fail $?
guard "RT-A1 RT2 kernel (GLM)" 5.5
$P rt_kernel grid --models $G < /dev/null || fail $?
$P rt_kernel replay --models $G < /dev/null || fail $?
$P rt_kernel social --models $G < /dev/null || fail $?
$P rt_kernel topup --models $G < /dev/null || fail $?
$PY code/rt_kernel.py summary
if [ ! -f "$RT_FORECAST" ]; then
  echo "=== $(date '+%F %T') freezing GLM forecast"
  $PY -W ignore code/rt_forecast.py freeze --models $G --B 500 --procs 8 || fail $?
fi
guard "RT-A1 RT3 populations (GLM)" 13.0
$P rt run --stage RT3 --models $G < /dev/null || fail $?
$PY -W ignore code/rt_eval.py --out "$RES/rt_eval_glm.json" || fail $?
echo "=== $(date '+%F %T') RT-A1 GLM queue done (ledger $(ledger))"
