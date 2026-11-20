#!/bin/bash
# Couzin study, queue 2 (COUZIN_PREREGISTRATION.md §6): M1 kernels -> write-once rules (omega, M4b model) ->
# M2 calibration -> Hartnett check + write-once forecast -> M3, M4, M4b, M5, M6, M7 populations -> evaluation.
# Launched by hand after queue 1 is inspected. Restart-safe: every runner skips rows already written and the
# write-once steps are skipped when their file exists.
# Usage: bash code/couzin_queue2.sh <ledger-cap> [pid ...]  > scratch/couzin_queue2.log 2>&1   (DRY=1 as queue 1)
cd "$(dirname "$0")/.." || exit 1
. code/couzin_common.sh "$@" || exit $?
shift
wait_turn "$@"
freeze_ok
[ -f "$RES/couzin_m0_gate.json" ] || { echo "=== no M0 gate; run queue 1 first"; exit 5; }
MODELS=$($PY -c "import json; print(' '.join(json.load(open('$RES/couzin_m0_gate.json'))['passed']))")
DR=$([ "$DRY" = 1 ] && echo 10 || echo 20)

guard "M1 kernels (gated models)" 2.5
$P couzin kernel --models $MODELS --draws $DR --parallel 12 < /dev/null || fail $?
$P couzin kernel --models $G --draws $((DR * 2)) --parallel 12 < /dev/null || fail $?
$P couzin kernel --robust --models $G --draws $DR --parallel 12 < /dev/null || fail $?
[ -f "$RES/couzin_m1_rules.json" ] || $PY code/couzin_kernel.py rules --models $MODELS > /dev/null || fail $?
M4B=$($PY -c "import json; print(json.load(open('$RES/couzin_m1_rules.json'))['m4b_model'] or '')")
echo "=== rules: M4b model '${M4B}'"

guard "M2 calibration in rounds" 0.3
$P couzin pop --models $G --arms M2_N10_p0.1 --nseeds $((24 / DRYF)) --parallel 6 < /dev/null || fail $?
$P couzin pop --models $G --arms M2c_obs_nu6 --nseeds $((8 / DRYF)) --parallel 4 < /dev/null || fail $?
[ -n "$M4B" ] && { $P couzin pop --models $M4B --arms M2c_obs_nu6 --nseeds $((8 / DRYF)) --parallel 4 < /dev/null || fail $?; }
if [ ! -f "$RES/couzin_forecast.json" ]; then
  $PY code/couzin_forecast.py freeze $([ "$DRY" = 1 ] && echo "--draws 40 --per 50") || fail $?
fi

stage() {   # $1 stage, $2 estimate
  guard "$1 populations" "$2"
  $PY code/couzin_forecast.py plan --stage "$1" | while read -r ARM MODEL NG; do
    NG=$(( (NG + DRYF - 1) / DRYF ))
    EXTRA=""; [ "$1" = M7 ] && EXTRA="--extra $D"
    echo "=== $(date '+%F %T') $ARM ($MODEL, $NG groups)"
    $P couzin pop --models "$MODEL" --arms "$ARM" --nseeds "$NG" --parallel 6 $EXTRA < /dev/null || exit $?
  done || fail $?
}
stage M3 3.5
stage M4 2.0
[ -n "$M4B" ] && stage M4b 1.2
stage M5 0.8
stage M6 0.7
stage M7 0.4
$PY code/couzin_eval.py > "$RES/couzin_eval.txt" 2>&1 || fail $?
echo "=== $(date '+%F %T') ledger after queue 2: $(ledger); study spend $(spent)"
echo "=== QUEUE DONE $(date '+%F %T')"
