#!/bin/bash
# Rosen phase 2, continued (ROSEN_PHASE2_PREREGISTRATION.md + amendment R-A3), after the owner raised the credit cap
# to 150 on 2026-11-19. Replaces rosen_p2_queue.sh, which ran under the 78.06 cap from .secrets/adaption.env and
# was stopped cleanly after R2x so that the Chase E10 queue could run. Waits for pid $1 (the E10 queue).
# The cap is the owner's: <ledger-cap> is exported as ADAPTION_CREDIT_CAP for this run, and before each stage
# ledger + estimate must stay at least 1 credit below it. Restart-safe: run commands skip populations already written.
# Usage: bash code/rosen_p2_queue2.sh <ledger-cap> [pid-to-wait-for]  > scratch/rosen_p2_queue2.log 2>&1
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
[ -f results/rosen_p2_prereg_freeze.json ] || { echo "=== pre-registration not frozen; not starting"; exit 5; }
[ -f results/rosen_p2_RA3_freeze.json ] || { echo "=== amendment R-A3 not frozen; not starting"; exit 5; }
[ -f results/rosen_r2x_predictions.json ] || { echo "=== R2x forecast missing; not starting"; exit 4; }

set -a; . "$MAIN/.secrets/adaption.env"; set +a
export ADAPTION_CREDIT_CAP=$CAP ADAPTION_LEDGER_DIR=$MAIN/results ADAPTION_RPM=48 ADAPTION_429_SLEEP=30 ADAPTION_MAX_TRIES=40 LLM_MAX_CONC=12
G=adaption:gemma-4-31b-it; Q=adaption:qwen3.8-27b; L=adaption:glm-5.2
S2="41001 41002 41003 41004 41005 41006 41007 41008"
P="$PY -u code/chase_paced.py"

guard "R2x populations (remainder)" 5.0
$P rosen run --stage R2x --models $G $Q $L --seeds 41101 41102 41103 41104 41105 41106 41107 41108 \
  --plan team:2:1,3,5,7,9,13 team:1:3 autarky:2:1,3,7 --parallel 10 < /dev/null || fail $?

guard "R2r reasoning populations" 5.5
$P rosen_p2 run2 --stage R2r --reasoning low --models $G --seeds $S2 \
  --plan team:2:1,3,5,7,9,11,13 team:1:3 autarky:2:1,3,7 --parallel 12 < /dev/null || fail $?

guard "R2b m=3 sweep" 0.5
$P rosen_p2 run2 --stage R2b --models $G --seeds $S2 --plan team:3:1,3,5,7,9,11,13 --parallel 10 < /dev/null || fail $?

if [ ! -f results/rosen_r6_acc_code.jsonl ]; then
  guard "R6 codebook battery (R-A3)" 0.2
  $P rosen_p2 nbacc --models $G $Q $L < /dev/null || fail $?
fi

guard "R6 notebook populations (codebook, R-A3)" 0.5
$P rosen_p2 run2 --stage R6 --variant notebook --models $G --seeds $S2 --plan team:1:1,3,5,7,9 --parallel 10 < /dev/null || fail $?

guard "R5 N-scaling" 1.4
$P rosen_p2 run2 --stage R5 --n-agents 4 --models $G --seeds $S2 --plan team:2:1,3,5,7,9,11 --parallel 10 < /dev/null || fail $?
$P rosen_p2 run2 --stage R5 --n-agents 16 --models $G --seeds $S2 --plan team:2:1,3,5,7,9,11 --parallel 10 < /dev/null || fail $?

guard "R4 Yang-Borland" 2.4
$P rosen_p2 yb --models $G $L --seeds $S2 --parallel 10 < /dev/null || fail $?

guard "R3 Becker-Murphy" 6.2
$P rosen_p2 bm --models $G $L --seeds $S2 --parallel 10 < /dev/null || fail $?

echo "=== $(date '+%F %T') ledger after phase 2: $(ledger)"
echo "=== QUEUE DONE $(date '+%F %T')"
