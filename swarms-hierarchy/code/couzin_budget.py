"""Couzin study spend: sum of Adaption costs in this worktree's results/calls.jsonl (only this study's calls are
logged there; a missing cost counts 0.001, as in llm.py). Prints the total in credits."""
import json
from pathlib import Path
f = Path(__file__).resolve().parents[1] / "results" / "calls.jsonl"
tot = 0.0
if f.exists():
    for line in open(f):
        r = json.loads(line)
        if not r.get("model", "").startswith("adaption:") or not r.get("tag", "").startswith("couzin"):
            continue
        u = r.get("usage") or {}
        tot += float(u["cost"]) if u.get("cost") is not None else (0.001 if r.get("response") or u else 0.0)
print(f"{tot:.4f}")
