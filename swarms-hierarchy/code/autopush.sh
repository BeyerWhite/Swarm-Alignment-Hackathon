#!/bin/bash
# Push a snapshot whenever one of the background chains finishes, and a final one when all are done.
cd /Users/saraansari/Claude/Outputs/Swarms
for c in chain_adaption2 chain_extra after_reset; do
  while pgrep -f "code/$c.sh" >/dev/null; do sleep 60; done
  python3 code/evaluate.py > scratch/eval_latest.log 2>&1 || true
  python3 code/figures.py claude:haiku adaption:gemma-4-31b-it > /dev/null 2>&1 || true
  code/push_results.sh "Auto-snapshot: $c finished ($(date '+%H:%M %Z'))" >> scratch/autopush.log 2>&1 || echo "push failed $(date)" >> scratch/autopush.log
done
echo "all chains done $(date)" >> scratch/autopush.log
