#!/bin/bash
# Sync the project into github.com/BeyerWhite/Swarm-Alignment-Hackathon under swarms-hierarchy/ and push.
# Never pushes secrets, scratch, the venv, or the raw (uncompressed) call log. Aborts if an API key is found.
set -euo pipefail
SRC=/Users/saraansari/Claude/Outputs/Swarms
REPO=$SRC/scratch/repo
MSG=${1:-"Update swarms-hierarchy results"}
[ -d $REPO/.git ] || git clone -q https://github.com/BeyerWhite/Swarm-Alignment-Hackathon $REPO
cd $REPO && git fetch -q origin && git reset -q --hard origin/main   # scratch clone; content is re-synced below
DST=$REPO/swarms-hierarchy; mkdir -p $DST
rsync -a --delete \
  --exclude '.secrets/' --exclude 'scratch/' --exclude '.aris/' --exclude '__pycache__/' --exclude '.DS_Store' \
  --exclude 'results/calls.jsonl' --exclude '.gitignore' --exclude 'README.md' \
  $SRC/RESEARCH_BRIEF.md $SRC/code $SRC/figures $SRC/idea-stage $SRC/refine-logs $SRC/results $DST/
[ -f $SRC/NARRATIVE_REPORT.md ] && cp $SRC/NARRATIVE_REPORT.md $DST/
gzip -9 -c $SRC/results/calls.jsonl > $DST/results/calls.jsonl.gz
# secret scan (key prefix + the literal key value from the local secrets file)
KEY=$(sed -n 's/^ADAPTION_API_KEY=//p' $SRC/.secrets/adaption.env 2>/dev/null || true)
if grep -rIl --exclude=push_results.sh -e 'pt_live_' ${KEY:+-e "$KEY"} $DST >/dev/null 2>&1 || zgrep -q 'pt_live_' $DST/results/calls.jsonl.gz; then
  echo "ABORT: secret-like string found in staged files"; exit 1
fi
big=$(find $DST -type f -size +95M); [ -z "$big" ] || { echo "ABORT: files >95MB: $big"; exit 1; }
sed -i '' "s/^\*\*Snapshot taken .*\*\*$/**Snapshot taken $(date '+%Y-%m-%d %H:%M %Z').**/" $DST/README.md 2>/dev/null || true
git add -A swarms-hierarchy
if git diff --cached --quiet; then echo "nothing to push"; exit 0; fi
git commit -q -m "$MSG

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push -q origin main && echo "pushed $(git rev-parse --short HEAD)"
