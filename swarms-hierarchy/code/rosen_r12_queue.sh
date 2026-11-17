#!/bin/bash
# Rosen R1 -> freeze -> R2, applying the pre-registered gate (results/rosen_r0_gate.json) mechanically.
# Usage: bash code/rosen_r12_queue.sh [pid-to-wait-for]  > scratch/rosen_r12_queue.log 2>&1
cd "$(dirname "$0")/.." || exit 1
MAIN=/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy
PY=$MAIN/../.venv/bin/python
fail() { echo "=== STAGE FAILED rc=$1 $(date '+%F %T')"; exit "$1"; }
if [ -n "$1" ]; then while kill -0 "$1" 2>/dev/null; do sleep 60; done; fi
[ -f results/rosen_r0_gate.json ] || { echo "=== no R0 gate file; not starting"; exit 5; }

set -a; . "$MAIN/.secrets/adaption.env"; set +a
export ADAPTION_LEDGER_DIR=$MAIN/results ADAPTION_RPM=48 ADAPTION_429_SLEEP=30 ADAPTION_MAX_TRIES=40 LLM_MAX_CONC=12

K_MODELS=(adaption:gemma-4-31b-it adaption:qwen3.8-27b adaption:glm-5.2 adaption:gemma-4-26b-a4b-it)
read -r -a POP_MODELS <<< "$($PY - <<'EOF'
import json
g = json.load(open("results/rosen_r0_gate.json"))
pref = ["adaption:gemma-4-31b-it", "adaption:qwen3.8-27b", "adaption:glm-5.2"]
sub = "adaption:gemma-4-26b-a4b-it"
out, used = [], False
for m in pref:
    if g.get(m, {}).get("passes"):
        out.append(m)
    elif not used and g.get(sub, {}).get("passes"):
        out.append(sub); used = True
print(" ".join(out))
EOF
)"
read -r -a R_MODELS <<< "$($PY -c "
import json; g = json.load(open('results/rosen_r0_gate.json'))
print(' '.join(m for m in ('adaption:gemma-4-31b-it', 'adaption:glm-5.2') if (g.get(m, {}).get('probe_valid') or 0) >= 0.8))")"
echo "=== $(date '+%F %T') population models: ${POP_MODELS[*]:-none}; reasoning arm: ${R_MODELS[*]:-none}"
echo "=== ledger before R1: $(cat "$MAIN/results/adaption_spend_total.json")"

echo "=== $(date '+%F %T') R1 kernel"
$PY -u code/chase_paced.py rosen_kernel kernel --models "${K_MODELS[@]}" < /dev/null || fail $?
echo "=== $(date '+%F %T') R1 top-up"
$PY -u code/chase_paced.py rosen_kernel topup --models "${K_MODELS[@]}" < /dev/null || fail $?
if [ ${#R_MODELS[@]} -gt 0 ]; then
  echo "=== $(date '+%F %T') R1 reasoning arm"
  $PY -u code/chase_paced.py rosen_kernel reason --models "${R_MODELS[@]}" < /dev/null || fail $?
fi
[ ${#POP_MODELS[@]} -gt 0 ] || { echo "=== no model passed the gate; stopping before R2"; exit 6; }

if [ ! -f results/rosen_r2_predictions.json ]; then
  if [ -f results/rosen_r2_pop.jsonl ]; then echo "=== REFUSING: R2 rows exist before the forecast"; exit 4; fi
  echo "=== $(date '+%F %T') freeze R2 forecast"
  $PY -u code/rosen_forecast.py freeze --models "${POP_MODELS[@]}" --B 200 < /dev/null || fail $?
  SHA=$($PY -c "import json;print(json.load(open('results/rosen_r2_predictions.json'))['sha256'])")
  printf '\n**%s — R2 forecast frozen (written by the queue script).** `results/rosen_r2_predictions.json`, stamp %s, file sha256 %s. Models: %s.\n' \
    "$(date '+%F %H:%M')" "$SHA" "$(shasum -a 256 results/rosen_r2_predictions.json | cut -d' ' -f1)" "${POP_MODELS[*]}" >> refine-logs/ROSEN_LOG.md
fi

echo "=== $(date '+%F %T') R2 populations"
$PY -u code/chase_paced.py rosen run --stage R2 --models "${POP_MODELS[@]}" \
  --seeds 41001 41002 41003 41004 41005 41006 41007 41008 \
  --plan team:2:1,3,5,7,9,11,13 team:1:3,7 hidden:2:3,7 autarky:2:1,3,5,7,9,11,13 autarky:3:1,3 \
  --parallel 10 < /dev/null || fail $?
echo "=== ledger after R2: $(cat "$MAIN/results/adaption_spend_total.json")"
echo "=== QUEUE DONE $(date '+%F %T')"
