#!/bin/bash
# Hierarchy-efficiency study, stages G1-G4 (refine-logs/HIERARCHY_PREREGISTRATION.md), launched by hand after G0.
# G1a -> kernels -> G1b/G1d/G1e -> G1c (+ registered top-up) -> kernels -> write-once freeze -> G2 -> benchmarks
# -> G3 re-check -> write-once G3 freeze -> G3 -> G4 -> evaluation. Models: those that passed the G0 gates.
# Restart-safe: runners skip rows already written; freezes are write-once and are not redone.
# Usage: bash code/hier_q1.sh <ledger-cap> [pid-to-wait-for]  > scratch/hier_q1.log 2>&1
cd "$(dirname "$0")/.." || exit 1
. code/hier_common.sh
[ -n "$1" ] || { echo "=== no ledger cap given; not starting"; exit 2; }
CAP=$1
wait_for "$2"
other_queue
verify
[ -f results/hier_g0_gate.json ] || { echo "=== G0 gate missing"; exit 4; }
M=$($PY -c "import json;g=json.load(open('results/hier_g0_gate.json'));assert g['task_gate'];print(' '.join(g['passed_models']))") \
  || { echo "=== G0 task gate failed; not starting"; exit 4; }
echo "=== models passing G0: $M"
env_setup

guard "G1a worker skill" 0.4
$P hier_measure g1a --models $M < /dev/null || fail $?
$PY code/hier_measure.py summarize || fail $?
guard "G1b/G1d/G1e kernels" 1.6
for part in g1b g1d g1e; do $P hier_measure $part --models $M < /dev/null || fail $?; done
guard "G1c appointed managers" 1.4
$P office g1c --seeds 61001 61002 61003 61004 61005 61006 61007 61008 --models $M --parallel 8 < /dev/null || fail $?
$PY code/hier_measure.py summarize || fail $?
$PY code/hier_measure.py reliability; rc=$?
if [ $rc -eq 3 ]; then
  guard "G1c registered top-up (8 more seeds)" 1.4
  $P office g1c --seeds 61009 61010 61011 61012 61013 61014 61015 61016 --models $M --parallel 8 < /dev/null || fail $?
  $PY code/hier_measure.py summarize || fail $?
  $PY code/hier_measure.py reliability   # reported; the study goes on whatever its value (pre-registration)
elif [ $rc -ne 0 ]; then fail $rc; fi

[ -f results/hier_freeze.json ] || $PY code/hier_sim.py freeze || fail $?
echo "=== freeze sha256 $(shasum -a 256 results/hier_freeze.json | cut -d' ' -f1)"
guard "G2 sorting populations" 3.5
$P office g2 --seeds 62001 62002 62003 62004 62005 62006 62007 62008 62009 62010 62011 62012 --parallel 8 \
  < /dev/null || fail $?
guard "G2 benchmark arms" 2.5
$P office bench --seeds 62001 62002 62003 62004 62005 62006 --parallel 8 < /dev/null || fail $?
guard "G3 disposition re-check" 0.3
$P hier_measure g3check < /dev/null || fail $?
[ -f results/hier_freeze_g3.json ] || $PY code/hier_sim.py freeze_g3 || fail $?
echo "=== G3 freeze sha256 $(shasum -a 256 results/hier_freeze_g3.json | cut -d' ' -f1)"
guard "G3 disposition populations" 1.8
$P office g3 --seeds 62001 62002 62003 62004 62005 62006 62007 62008 62009 62010 62011 62012 --parallel 8 \
  < /dev/null || fail $?
guard "G4 seeded records" 0.6
$P office g4 --seeds 64001 64002 64003 64004 64005 64006 64007 64008 --parallel 8 < /dev/null || fail $?
$PY code/hier_eval.py || fail $?
echo "=== $(date '+%F %T') phase 1 complete (ledger $(ledger))"
