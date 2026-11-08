#!/bin/bash
cd /Users/saraansari/Claude/Outputs/Swarms
set -a; . .secrets/adaption.env; set +a
PY=scratch/.venv/bin/python
LLM_MAX_CONC=2 $PY code/fsweep.py calib --workers 2 > scratch/fsweep_calib.log 2>&1
$PY code/fsweep.py predict > scratch/fsweep_predict.log 2>&1
while pgrep -f "threshold.py run" >/dev/null || pgrep -f "x2b.py run" >/dev/null; do sleep 60; done
LLM_MAX_CONC=8 $PY code/fsweep.py run --parallel 4 > scratch/fsweep_run.log 2>&1
echo "fsweep done $(date)" >> scratch/fsweep_run.log
