# Sourced by couzin_queue1.sh / couzin_queue2.sh. $1 = ledger cap.
MAIN=/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy
PY=$MAIN/../.venv/bin/python
[ -n "$1" ] || { echo "=== no ledger cap given; not starting"; return 2; }
CAP=$1
BUDGET=${COUZIN_BUDGET:-20}
G=adaption:gemma-4-31b-it
ALL="adaption:gemma-4-31b-it adaption:gemma-4-26b-a4b-it adaption:qwen3.8-27b adaption:qwen3.5-397b adaption:glm-5.2 adaption:deepseek-v4-flash"
D=adaption:deepseek-v4-flash
fail() { echo "=== STAGE FAILED rc=$1 $(date '+%F %T')"; exit "$1"; }
if [ "$DRY" = 1 ]; then
  RES=$PWD/scratch/couzin_dry; mkdir -p "$RES"
  export COUZIN_FAKE=1 COUZIN_RESULTS=$RES
  P="$PY -u code/couzin.py"
  ledger() { echo 0; }
  spent() { echo 0; }
  guard() { echo "=== DRY $1 (estimate $2)"; }
  wait_turn() { echo "=== DRY: not waiting"; }
  freeze_ok() { :; }
  DRYF=${DRYF:-4}                     # dry run: groups divided by this factor
else
  RES=$PWD/results
  P="$PY -u code/chase_paced.py"
  DRYF=1
  ledger() { $PY -c "import json;print(json.load(open('$MAIN/results/adaption_spend_total.json'))['total'])"; }
  spent() { $PY code/couzin_budget.py; }
  guard() {
    local t s; t=$(ledger); s=$(spent)
    if ! $PY -c "import sys; sys.exit(0 if $t + $2 < $CAP - 1 else 1)"; then
      echo "=== $(date '+%F %T') STOP before $1: ledger $t + estimate $2 would come within 1 of the cap $CAP"; exit 7
    fi
    if ! $PY -c "import sys; sys.exit(0 if $s + $2 <= $BUDGET else 1)"; then
      echo "=== $(date '+%F %T') STOP before $1: study spend $s + estimate $2 would exceed the study guard $BUDGET"; exit 8
    fi
    echo "=== $(date '+%F %T') $1 (ledger $t, study $s, estimate $2, cap $CAP, guard $BUDGET)"
  }
  busy() {   # another study's queue script (not this one) or any paced job
    pgrep -f "queue[0-9]*\.sh" | grep -vx "$$" | while read -r p; do
      ps -o command= -p "$p" | grep -q "couzin_queue" || echo "$p"; done | grep -q . && return 0
    pgrep -f "chase_paced\.py" > /dev/null
  }
  wait_turn() {
    for p in "$@"; do
      echo "=== $(date '+%F %T') waiting for pid $p"
      while kill -0 "$p" 2>/dev/null; do sleep 60; done
    done
    echo "=== $(date '+%F %T') waiting until no other queue script or paced job runs"
    while true; do
      if ! busy; then sleep 120; busy || break; fi
      sleep 120
    done
    echo "=== $(date '+%F %T') account free"
  }
  freeze_ok() {
    [ -f results/couzin_prereg_freeze.json ] || { echo "=== pre-registration not frozen; not starting"; exit 5; }
    $PY code/couzin_freeze.py check || { echo "=== frozen files changed; not starting"; exit 5; }
  }
  set -a; . "$MAIN/.secrets/adaption.env"; set +a
  export ADAPTION_CREDIT_CAP=$CAP ADAPTION_LEDGER_DIR=$MAIN/results ADAPTION_RPM=48 ADAPTION_429_SLEEP=30 ADAPTION_MAX_TRIES=40 LLM_MAX_CONC=12
fi
