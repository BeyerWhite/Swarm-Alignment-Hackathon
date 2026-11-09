#!/bin/bash
cd /Users/saraansari/Claude/Outputs/Swarms
set -a; . .secrets/adaption.env; set +a
PY=scratch/.venv/bin/python
ADAPTION_429_SLEEP=6 ADAPTION_MAX_TRIES=40 LLM_MAX_CONC=32 $PY code/pilotD.py calib --workers 32 > scratch/pilotD_calib.log 2>&1
$PY code/pilotD.py predict > scratch/pilotD_predict.log 2>&1 || exit 1
ADAPTION_429_SLEEP=6 ADAPTION_MAX_TRIES=40 LLM_MAX_CONC=48 $PY code/pilotD.py run --parallel 10 > scratch/pilotD_run.log 2>&1
$PY code/pilotD.py eval > scratch/pilotD_eval.log 2>&1
echo "pilotD done $(date)" >> scratch/pilotD_eval.log
