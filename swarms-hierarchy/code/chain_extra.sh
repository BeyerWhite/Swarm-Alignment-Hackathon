#!/bin/bash
cd /Users/saraansari/Claude/Outputs/Swarms
set -a; . .secrets/adaption.env; set +a
PY=scratch/.venv/bin/python
(
  while pgrep -f "bnr_exp.py calib" >/dev/null; do sleep 30; done
  $PY code/bnr_exp.py predict > scratch/bnr_predict.log 2>&1
  LLM_MAX_CONC=6 $PY code/bnr_exp.py run --parallel 3 > scratch/bnr_run.log 2>&1
  echo "bnr done $(date)"
) &
(
  while pgrep -f "code/board.py" >/dev/null; do sleep 30; done
  LLM_MAX_CONC=6 $PY code/hysteresis.py run --parallel 3 > scratch/hyst_run.log 2>&1
  echo "hyst done $(date)"
) &
wait
echo "extra done $(date)"
