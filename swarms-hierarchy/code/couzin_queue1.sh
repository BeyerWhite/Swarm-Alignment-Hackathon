#!/bin/bash
# Couzin study, queue 1 (COUZIN_PREREGISTRATION.md §6): M0 comprehension items and label screen (6 models), the
# write-once gate, and a 3-group pilot on the primary model. Stops there: queue 2 is launched by hand after inspection.
# Waits for every pid given after the cap, then until no other study's queue script and no paced job runs.
# Guards: ledger + estimate < cap - 1 (the owner's cap, 150) and this study's spend + estimate <= 20 credits.
# Usage: bash code/couzin_queue1.sh <ledger-cap> [pid ...]  > scratch/couzin_queue1.log 2>&1
#        DRY=1 bash code/couzin_queue1.sh 150   (scripted agents, results in scratch/couzin_dry; no API calls)
cd "$(dirname "$0")/.." || exit 1
. code/couzin_common.sh "$@" || exit $?
shift
wait_turn "$@"
freeze_ok

guard "M0 comprehension items" 0.15
$P couzin quiz --models $ALL --draws 20 --parallel 10 < /dev/null || fail $?
guard "M0 label screen" 0.15
$P couzin labels --models $ALL --draws 20 --parallel 10 < /dev/null || fail $?
[ -f "$RES/couzin_m0_gate.json" ] || $PY code/couzin_kernel.py gate > /dev/null || fail $?
$PY -c "import json,sys; d=json.load(open('$RES/couzin_m0_gate.json')); print('=== gate: passed', d['passed'], 'labels', d['label_pair']); sys.exit(0 if d['primary_passed'] else 9)" \
  || { echo "=== primary model failed the M0 gate: stop (owner decision needed)"; exit 9; }

guard "pilot: 2 groups M3_N10_p0.2, 1 group M4_obs_nu6" 0.05
$P couzin pop --pilot --models $G --arms M3_N10_p0.2 --nseeds 2 --parallel 2 < /dev/null || fail $?
$P couzin pop --pilot --models $G --arms M4_obs_nu6 --nseeds 1 --parallel 1 < /dev/null || fail $?
echo "=== $(date '+%F %T') ledger after queue 1: $(ledger); study spend $(spent)"
echo "=== QUEUE DONE $(date '+%F %T')"
