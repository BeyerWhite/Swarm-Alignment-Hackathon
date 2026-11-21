#!/bin/bash
# Rosen follow-ups and phase 2 (ROSEN_PHASE2_PREREGISTRATION.md), in the registered order. Waits for pid $1 (the Chase
# queue) so only one paced job uses the account. Before each stage: ledger + estimate must stay below 99 credits.
# Restart-safe: every run command skips populations already written.
# Usage: bash code/rosen_p2_queue.sh [pid-to-wait-for]  > scratch/rosen_p2_queue.log 2>&1
cd "$(dirname "$0")/.." || exit 1
MAIN=/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy
PY=$MAIN/../.venv/bin/python
fail() { echo "=== STAGE FAILED rc=$1 $(date '+%F %T')"; exit "$1"; }
ledger() { $PY -c "import json;print(json.load(open('$MAIN/results/adaption_spend_total.json'))['total'])"; }
guard() {  # $1 stage name, $2 estimated credits
  local t; t=$(ledger)
  if $PY -c "import sys; sys.exit(0 if $t + $2 < 99 else 1)"; then
    echo "=== $(date '+%F %T') $1 (ledger $t, estimate $2)"
  else
    echo "=== $(date '+%F %T') STOP before $1: ledger $t + estimate $2 would reach the cap"; exit 7
  fi
}
if [ -n "$1" ]; then
  echo "=== $(date '+%F %T') waiting for pid $1"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
  echo "=== $(date '+%F %T') pid $1 exited"
fi
[ -f results/rosen_p2_prereg_freeze.json ] || { echo "=== pre-registration not frozen; not starting"; exit 5; }

set -a; . "$MAIN/.secrets/adaption.env"; set +a
export ADAPTION_LEDGER_DIR=$MAIN/results ADAPTION_RPM=48 ADAPTION_429_SLEEP=30 ADAPTION_MAX_TRIES=40 LLM_MAX_CONC=12
G=adaption:gemma-4-31b-it; Q=adaption:qwen3.8-27b; L=adaption:glm-5.2
S2="41001 41002 41003 41004 41005 41006 41007 41008"
P="$PY -u code/chase_paced.py"

guard "R6 battery" 0.2
$P rosen_p2 nbacc --models $G $Q $L < /dev/null || fail $?

guard "R1x kernel" 1.0
$P rosen_kernel extend --models $G $Q $L < /dev/null || fail $?

if [ ! -f results/rosen_r2x_predictions.json ]; then
  if [ -f results/rosen_r2x_pop.jsonl ]; then echo "=== REFUSING: R2x rows exist before the forecast"; exit 4; fi
  echo "=== $(date '+%F %T') freeze R2x forecast"
  $PY -u code/rosen_forecast2.py freeze --models $G $Q $L --B 200 < /dev/null || fail $?
  SHA=$($PY -c "import json;print(json.load(open('results/rosen_r2x_predictions.json'))['sha256'])")
  printf '\n**%s — R2x forecast frozen (written by the queue script).** `results/rosen_r2x_predictions.json`, stamp %s, file sha256 %s.\n' \
    "$(date '+%F %H:%M')" "$SHA" "$(shasum -a 256 results/rosen_r2x_predictions.json | cut -d' ' -f1)" >> refine-logs/ROSEN_LOG.md
fi

guard "R2x populations" 6.5
$P rosen run --stage R2x --models $G $Q $L --seeds 41101 41102 41103 41104 41105 41106 41107 41108 \
  --plan team:2:1,3,5,7,9,13 team:1:3 autarky:2:1,3,7 --parallel 10 < /dev/null || fail $?

guard "R2r reasoning populations" 5.5
$P rosen_p2 run2 --stage R2r --reasoning low --models $G --seeds $S2 \
  --plan team:2:1,3,5,7,9,11,13 team:1:3 autarky:2:1,3,7 --parallel 12 < /dev/null || fail $?

guard "R2b m=3 sweep" 0.5
$P rosen_p2 run2 --stage R2b --models $G --seeds $S2 --plan team:3:1,3,5,7,9,11,13 --parallel 10 < /dev/null || fail $?

guard "R6 notebook populations" 0.5
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
