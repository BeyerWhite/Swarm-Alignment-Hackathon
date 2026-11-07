#!/bin/bash
cd /Users/saraansari/Claude/Outputs/Swarms
while [ "$(date +%H%M)" -lt 1855 ]; do sleep 60; done
until echo "Reply OK" | claude -p --model haiku 2>&1 | grep -qv -i "limit"; do sleep 300; done
LLM_MAX_CONC=12 python3 code/statecal.py --model claude:haiku --workers 12 > scratch/statecal_haiku_resume.log 2>&1
python3 code/predict.py --with_state --kernels nw bonabeau --configs '[{"model":"claude:haiku","cond":"norm0","rho":0.5,"N":8},{"model":"claude:haiku","cond":"ended","rho":0.5,"N":8},{"model":"claude:haiku","cond":"ended+comp","rho":0.5,"N":8},{"model":"claude:haiku","cond":"norm100","rho":0.5,"N":8}]' --B 25 --S 8 --out results/predictions_v2s_RETRO_claude_haiku.json > scratch/preds_haiku_retro.log 2>&1
echo "haiku retro done $(date)"
