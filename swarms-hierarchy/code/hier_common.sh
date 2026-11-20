#!/bin/bash
# Shared helpers for the hierarchy-efficiency queues (sourced, not run). Bash, not zsh (zsh does not word-split).
MAIN=/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy
PY=$MAIN/../.venv/bin/python
fail() { echo "=== STAGE FAILED rc=$1 $(date '+%F %T')"; exit "$1"; }
ledger() { $PY -c "import json;print(json.load(open('$MAIN/results/adaption_spend_total.json'))['total'])"; }
guard() {  # $1 stage name, $2 estimated credits: ledger + estimate must stay at least 1 credit below the cap
  local t; t=$(ledger)
  if $PY -c "import sys; sys.exit(0 if $t + $2 < $CAP - 1 else 1)"; then
    echo "=== $(date '+%F %T') $1 (ledger $t, estimate $2, cap $CAP)"
  else
    echo "=== $(date '+%F %T') STOP before $1: ledger $t + estimate $2 would come within 1 of the cap $CAP"; exit 7
  fi
}
verify() {  # the code and pre-registration must match the write-once stamp
  [ -f results/hier_prereg_freeze.json ] || { echo "=== pre-registration not frozen; not starting"; exit 5; }
  $PY - <<'EOF' || { echo "=== code differs from the pre-registration stamp; not starting"; exit 6; }
import hashlib, json, sys
st = json.load(open("results/hier_prereg_freeze.json"))
bad = [f for f, h in st["files"].items() if hashlib.sha256(open(f, "rb").read()).hexdigest() != h]
print("verify:", "ok" if not bad else f"MISMATCH {bad}"); sys.exit(1 if bad else 0)
EOF
}
wait_for() {
  if [ -n "$1" ]; then
    echo "=== $(date '+%F %T') waiting for pid $1"
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
    echo "=== $(date '+%F %T') pid $1 exited"
  fi
}
other_queue() {  # refuse to start while another study's paced queue is running (one paced job at a time)
  # amendment H-A1: any study's queue script (rosen_p2_queue2.sh, tour_queue1.sh, rt_queue.sh, chase_e14_queue.sh, ...)
  if pgrep -f "queue[0-9]*\.sh" >/dev/null; then
    echo "=== $(date '+%F %T') another paced queue is running: $(pgrep -fl 'queue[0-9]*\.sh' | head -3)"; exit 8
  fi
}
env_setup() {
  set -a; . "$MAIN/.secrets/adaption.env"; set +a
  export ADAPTION_CREDIT_CAP=$CAP ADAPTION_LEDGER_DIR=$MAIN/results ADAPTION_RPM=48 ADAPTION_429_SLEEP=30 \
         ADAPTION_MAX_TRIES=40 LLM_MAX_CONC=12
  P="$PY -u code/chase_paced.py"
}
