#!/bin/bash
cd /Users/saraansari/Claude/Outputs/Swarms
set -a; . .secrets/adaption.env; set +a
PY=scratch/.venv/bin/python
LLM_MAX_CONC=24 $PY code/pilotD.py calib --workers 24 > scratch/pilotD_calib.log 2>&1
$PY code/pilotD.py predict > scratch/pilotD_predict.log 2>&1 || exit 1
LLM_MAX_CONC=40 $PY code/pilotD.py run --parallel 10 > scratch/pilotD_run.log 2>&1
$PY code/pilotD.py eval > scratch/pilotD_eval.log 2>&1
echo "pilotD done $(date)" >> scratch/pilotD_eval.log
