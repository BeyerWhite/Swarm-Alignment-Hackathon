#!/bin/bash
# Per Adaption model: wait for grid calibration -> freeze GRID predictions -> state-matched calibration ->
# freeze STATE predictions -> (all models) populations, two models at a time. Every freeze precedes populations.
cd /Users/saraansari/Claude/Outputs/Swarms
set -a; . .secrets/adaption.env; set +a
PY=scratch/.venv/bin/python
MODELS="deepseek-v4-flash qwen3.8-27b gemma-4-31b-it gemma-4-26b-a4b-it qwen3.5-397b glm-5.2"
SWEEP="gemma-4-31b-it deepseek-v4-flash"
cfg() {  # $1 model, $2 with_seeds(0/1)
  $PY -c "
import json
m='adaption:$1'; sweep='$1' in '$SWEEP'.split(); seeds=(301,302,303)
c=[dict(model=m,cond=k,rho=0.5,N=8) for k in ['norm0','ended','ended+comp','norm100']]
if sweep: c+=[dict(model=m,cond=k,rho=r,N=8) for k in ['norm0','ended+comp','norm100'] for r in (0.1,0.25,0.9)]
if $2: c=[dict(x,seed=s) for x in c for s in seeds]
print(json.dumps(c))"
}
prep() {
  m=$1
  while pgrep -f "calibrate.py --model adaption:$m " >/dev/null || [ "$(wc -l < results/calib_v2_adaption_$m.jsonl 2>/dev/null || echo 0)" -lt 2136 ]; do sleep 30; done
  [ -f results/predictions_v2_adaption_$m.json ] || $PY code/predict.py --configs "$(cfg $m 0)" --B 25 --S 8 --out results/predictions_v2_adaption_$m.json > scratch/pred_$m.log 2>&1
  LLM_MAX_CONC=4 $PY code/statecal.py --model adaption:$m --workers 4 > scratch/statecal_$m.log 2>&1
  [ -f results/predictions_v2s_adaption_$m.json ] || $PY code/predict.py --with_state --kernels nw bonabeau --configs "$(cfg $m 0)" --B 25 --S 8 --out results/predictions_v2s_adaption_$m.json > scratch/preds_$m.log 2>&1
  echo "prepared $m $(date)"
}
for m in $MODELS; do prep $m & done
wait
echo "all predictions frozen $(date)"
run_pop() {
  LLM_MAX_CONC=8 $PY code/run_populations.py --parallel 8 --configs "$(cfg $1 1)" --out results/population_v2_adaption_$1.jsonl >> scratch/pop_$1.log 2>&1
}
set -- $MODELS
while [ $# -gt 0 ]; do
  run_pop $1 & [ $# -gt 1 ] && run_pop $2 &
  wait; shift; [ $# -gt 0 ] && shift
done
echo "populations done $(date)"
