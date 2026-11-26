#!/bin/bash
# Couzin amendment M-A4: the M4b design on Qwen3.5-397B (Adaption). Reuses the main study's M0 gate and M1 kernel for
# the model (seed), then rules -> M2c calibration -> forecast -> 5 arms x 24 groups -> eval, in results/couzin_qwen.
# Same guards as queues 1-2 (ledger + estimate < cap - 1; study spend + estimate <= 20). Launch through
# couzin_launch.sh 3 <cap> [pid ...] so it waits for its turn on the account.
cd "$(dirname "$0")/.." || exit 1
. code/couzin_common.sh "$@" || exit $?
wait_turn      # pids are waited for by couzin_launch.sh; here only the busy check
freeze_ok
[ "$DRY" = 1 ] || { $PY code/couzin_followup.py check || { echo "=== amendment files changed; not starting"; exit 5; }; }
Q=adaption:qwen3.5-397b
export COUZIN_RESULTS=$([ "$DRY" = 1 ] && echo "$PWD/scratch/couzin_qwen_dry" || echo "$PWD/results/couzin_qwen")
mkdir -p "$COUZIN_RESULTS"
F="$P couzin_followup"
$F seed --model $Q < /dev/null || fail $?
$F rules --model $Q < /dev/null || fail $?
guard "M-A4 Qwen calibration (M2c, 8 groups)" 0.2
$F calib --model $Q < /dev/null || fail $?
$F forecast --model $Q < /dev/null || fail $?
guard "M-A4 Qwen M4b populations" 3.0
$F pop --model $Q < /dev/null || fail $?
$F eval < /dev/null > "$COUZIN_RESULTS/couzin_followup_eval.txt" 2>&1 || fail $?
echo "=== $(date '+%F %T') ledger after queue 3: $(ledger); study spend $(spent)"
echo "=== QUEUE DONE $(date '+%F %T')"
