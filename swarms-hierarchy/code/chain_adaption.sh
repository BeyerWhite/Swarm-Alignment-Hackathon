#!/bin/bash
# Waits for 6 Adaption calibration files, freezes per-model predictions (parallel), then runs populations.
cd /Users/saraansari/Claude/Outputs/Swarms
set -a; . .secrets/adaption.env; set +a
PY=scratch/.venv/bin/python
MODELS="deepseek-v4-flash qwen3.8-27b gemma-4-31b-it gemma-4-26b-a4b-it qwen3.5-397b glm-5.2"
SWEEP="gemma-4-31b-it deepseek-v4-flash"
# a calibration is complete when its streamed file has all 2,136 rows and its process has exited
for m in $MODELS; do
  while pgrep -f "calibrate.py --model adaption:$m" >/dev/null || [ "$(wc -l < results/calib_v2_adaption_$m.jsonl 2>/dev/null || echo 0)" -lt 2136 ]; do sleep 30; done
done
echo "calibration complete $(date)"
for m in $MODELS; do
  CFG=$($PY -c "
import json,sys
m='adaption:$m'; sweep='$m' in '$SWEEP'.split()
c=[dict(model=m,cond=k,rho=0.5,N=8) for k in ['norm0','ended','ended+comp','norm100']]
if sweep: c+=[dict(model=m,cond=k,rho=r,N=8) for k in ['norm0','ended+comp','norm100'] for r in (0.1,0.25,0.9)]
print(json.dumps(c))")
  nohup $PY code/predict.py --configs "$CFG" --B 25 --S 8 --out results/predictions_v2_adaption_$m.json > scratch/pred_$m.log 2>&1 &
done
wait
echo "predictions frozen $(date)"; ls results/predictions_v2_adaption_*.json
run_pop() {
  m=$1
  CFG=$($PY -c "
import json
m='adaption:$m'; sweep='$m' in '$SWEEP'.split()
c=[dict(model=m,cond=k,rho=0.5,N=8,seed=s) for k in ['norm0','ended','ended+comp','norm100'] for s in (301,302,303)]
if sweep: c+=[dict(model=m,cond=k,rho=r,N=8,seed=s) for k in ['norm0','ended+comp','norm100'] for r in (0.1,0.25,0.9) for s in (301,302,303)]
print(json.dumps(c))")
  LLM_MAX_CONC=8 $PY code/run_populations.py --parallel 8 --configs "$CFG" --out results/population_v2_adaption_$m.jsonl >> scratch/pop_$m.log 2>&1
}
# two models at a time (<= 16 concurrent Adaption calls)
set -- $MODELS
while [ $# -gt 0 ]; do
  run_pop $1 & [ $# -gt 1 ] && run_pop $2 &
  wait; shift; [ $# -gt 0 ] && shift
done
echo "populations done $(date)"
