"""Couzin study spend in credits (amendment M-A2).

This worktree's llm.py does not log costs in results/calls.jsonl, so each Couzin Adaption call there is matched to
its entry in the shared per-call ledger (adaption_spend.jsonl in ADAPTION_LEDGER_DIR or the main checkout): same
model, nearest timestamp within 2 s, each ledger entry used once (both are written within milliseconds of each
other). An unmatched call with a reply counts 0.001 (the llm.py fail-closed charge). Prints the total."""
import bisect, collections, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MAIN = "/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy/results"
CALLS = ROOT / "results" / "calls.jsonl"
LEDGER = Path(os.environ.get("ADAPTION_LEDGER_DIR", MAIN)) / "adaption_spend.jsonl"


def spent():
    calls = collections.defaultdict(list)
    if CALLS.exists():
        for line in open(CALLS):
            r = json.loads(line)
            if r.get("model", "").startswith("adaption:") and r.get("tag", "").startswith("couzin"):
                calls[r["model"].split(":", 1)[1]].append((r["t"], bool(r.get("response"))))
    if not calls:
        return 0.0
    t0 = min(t for v in calls.values() for t, _ in v) - 60
    led = collections.defaultdict(list)
    for line in open(LEDGER):
        r = json.loads(line)
        if r["t"] >= t0 and r["model"] in calls:
            led[r["model"]].append((r["t"], r["cost"]))
    tot = 0.0
    for m, cs in calls.items():
        L = sorted(led[m]); lt = [x[0] for x in L]; used = set()
        for t, replied in sorted(cs):
            i = bisect.bisect_left(lt, t)
            near = [j for j in range(i - 3, i + 3) if 0 <= j < len(L) and j not in used and abs(lt[j] - t) <= 2]
            if near:
                j = min(near, key=lambda j: abs(lt[j] - t)); used.add(j); tot += L[j][1]
            elif replied:
                tot += 0.001
    return tot


if __name__ == "__main__":
    print(f"{spent():.4f}")
