#!/bin/bash
# Launcher for the RT follow-ups: GLM-5.2 on the frozen design (RT-A1), then the rest-reward framing (RT-R).
# Its name deliberately matches no study's "queue" pattern (the hierarchy G0 wrapper waits while any queue[N].sh
# command line exists; a waiting launcher under such a name could deadlock with it).
# Waits for the pids given, then until no other study's stage script or paced job runs, then runs both queues.
# Usage: nohup caffeinate -i bash code/rt_phase2_launch.sh <ledger-cap> [pid ...] > scratch/rt_phase2.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
CAP=$1; shift
[ -n "$CAP" ] || { echo "=== no ledger cap given; not starting"; exit 2; }
busy() {   # a running stage script of another study (not a waiting "bash -c" wrapper) or any paced job
  local pat='^(caffeinate -i )?bash (code|scratch)/[A-Za-z0-9_]*(queue[0-9]*|_q[0-9]+)\.sh'
  pgrep -fl "$pat" 2>/dev/null | grep -v " code/rt_" | grep -q . && return 0
  pgrep -f "chase_paced\.py" > /dev/null
}
for p in "$@"; do
  echo "=== $(date '+%F %T') waiting for pid $p"
  while kill -0 "$p" 2>/dev/null; do sleep 60; done
done
echo "=== $(date '+%F %T') waiting until no other stage script or paced job runs"
while true; do
  if ! busy; then sleep 120; busy || break; fi
  sleep 120
done
echo "=== $(date '+%F %T') account free"
bash code/rt_glm_queue.sh "$CAP"
echo "=== $(date '+%F %T') GLM queue exited rc=$?"
bash code/rt_rest_queue.sh "$CAP"
echo "=== $(date '+%F %T') RT-R queue exited rc=$?"
