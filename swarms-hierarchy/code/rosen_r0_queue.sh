#!/bin/bash
# Rosen R0: waits for the Chase queue (pid $1) to exit so the two studies never share the rate limit, then runs
# comprehension, the reasoning probe and the pilot populations, then the gate summary. Restart-safe only per step.
# Usage: bash code/rosen_r0_queue.sh <chase-queue-pid>  > scratch/rosen_r0_queue.log 2>&1
cd "$(dirname "$0")/.." || exit 1
MAIN=/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy
PY=$MAIN/../.venv/bin/python
MODELS=(adaption:gemma-4-31b-it adaption:qwen3.8-27b adaption:glm-5.2 adaption:gemma-4-26b-a4b-it)
fail() { echo "=== STAGE FAILED rc=$1 $(date '+%F %T')"; exit "$1"; }

echo "=== $(date '+%F %T') waiting for Chase queue pid $1"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') Chase queue exited; its last lines:"
tail -n 2 "$MAIN/scratch/chase_f_queue.log"

set -a; . "$MAIN/.secrets/adaption.env"; set +a
export ADAPTION_LEDGER_DIR=$MAIN/results ADAPTION_RPM=48 ADAPTION_429_SLEEP=30 ADAPTION_MAX_TRIES=40 LLM_MAX_CONC=8
echo "=== $(date '+%F %T') ledger before R0: $(cat "$MAIN/results/adaption_spend_total.json")"

echo "=== $(date '+%F %T') R0 comprehension"
$PY -u code/chase_paced.py rosen_kernel comp --models "${MODELS[@]}" < /dev/null || fail $?
echo "=== $(date '+%F %T') R0 reasoning probe"
$PY -u code/chase_paced.py rosen_kernel probe --models "${MODELS[@]}" < /dev/null || fail $?
echo "=== $(date '+%F %T') R0 pilot populations"
$PY -u code/chase_paced.py rosen pilot --models "${MODELS[@]}" --F 1 13 --seeds 7 8 --parallel 8 < /dev/null || fail $?
$PY code/rosen.py gate
echo "=== $(date '+%F %T') ledger after R0: $(cat "$MAIN/results/adaption_spend_total.json")"
echo "=== R0 DONE $(date '+%F %T')"
