#!/bin/bash
# Hierarchy-efficiency study, stage G0 (refine-logs/HIERARCHY_PREREGISTRATION.md): skill battery -> frozen task
# selection -> comprehension quiz -> 4 pilot populations -> gates. Waits for pid $2 (the running Rosen phase-2
# queue), then refuses to start if any other paced queue is running. Stops after the gates: G1-G4 (hier_q1.sh) are
# launched by hand once G0 has been inspected. Restart-safe (every runner skips rows already written).
# Usage: bash code/hier_q0.sh <ledger-cap> [pid-to-wait-for]  > scratch/hier_q0.log 2>&1
cd "$(dirname "$0")/.." || exit 1
. code/hier_common.sh
[ -n "$1" ] || { echo "=== no ledger cap given; not starting"; exit 2; }
CAP=$1
wait_for "$2"
other_queue
verify
env_setup

guard "G0 battery + comprehension" 1.2
$P hier_measure battery --parallel 12 < /dev/null || fail $?
[ -f results/hier_g0_selection.json ] || $PY code/hier_measure.py select || fail $?
$P hier_measure comp --parallel 12 < /dev/null || fail $?
guard "G0 pilot" 0.3
$P office pilot --seeds 7001 7002 --parallel 4 < /dev/null || fail $?
$PY code/hier_measure.py gate || fail $?
echo "=== $(date '+%F %T') G0 done (ledger $(ledger)); inspect results/hier_g0_*.json before hier_q1.sh"
