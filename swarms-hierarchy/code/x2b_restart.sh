#!/bin/bash
# When the threshold sweep exits, restart X2b with more concurrency (resume skips finished runs).
cd /Users/saraansari/Claude/Outputs/Swarms
set -a; . .secrets/adaption.env; set +a
while pgrep -f "threshold.py run" >/dev/null; do sleep 30; done
pkill -f "x2b.py run"; sleep 5
LLM_MAX_CONC=14 scratch/.venv/bin/python code/x2b.py run --parallel 8 >> scratch/x2b_run.log 2>&1
