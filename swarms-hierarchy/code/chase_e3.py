"""E3 / E4 population runner for the Chase study (LLM in the loop).

  chase_e3.py run --cells "model|framing" ... [--controls ...] [--sizes 4 8] [--arms OWN LOG]
  chase_e3.py rewind --members m1 m2 m3 m4 --framing comp --arms OWN LOG --groups 40
Group counts come from chase_e2.PLAN / PLAN_CONTROL unless --groups is given. Seeds are 4000+g (groups of four),
8000+g (N = 8), 9000+g (rewind); the same seeds are used in every arm. A finished (model, framing, arm, N, seed)
is skipped on restart. Refuses to run a cell that has no frozen forecast in results/chase_e2_predictions.json.
-> results/chase_e3_pop.jsonl, results/chase_e4_rewind.jsonl
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from chase_pop import run, LLMDecider
from chase_e2 import PLAN, PLAN_CONTROL, ROUNDS, RHO

ROOT = Path(__file__).resolve().parents[1]
SEED0 = {4: 4000, 8: 8000}


def _done(path, keys):
    out = set()
    if Path(path).exists():
        for l in open(path):
            r = json.loads(l)
            out.add(tuple(r.get(k) for k in keys))
    return out


def frozen_cells(path):
    if not Path(path).exists():
        return set()
    p = json.loads(Path(path).read_text())
    return {(e["model"], e["framing"], e["arm"], e["N"]) for e in p["predictions"]}


def cmd_run(a):
    out = ROOT / "results" / "chase_e3_pop.jsonl"
    done = _done(out, ("model", "framing", "arm", "N", "seed"))
    frozen = frozen_cells(a.predictions)
    jobs = []
    for group, plan in ((a.cells, PLAN), (a.controls, PLAN_CONTROL)):
        for cell in group:
            model, framing = cell.split("|")
            for N in a.sizes:
                for arm, G in plan.get(N, {}).items():
                    if a.arms and arm not in a.arms:
                        continue
                    if (model, framing, arm, N) not in frozen:
                        sys.exit(f"no frozen forecast for {(model, framing, arm, N)}; run chase_e2.py predict first")
                    for g in range(a.groups or G):
                        seed = SEED0[N] + g
                        if (model, framing, arm, N, seed) not in done:
                            jobs.append((model, framing, arm, N, seed))
    jobs.sort(key=lambda j: (j[4] % 1000, j[3]))          # interleave arms and cells: early stops stay balanced
    print(len(jobs), "groups to run", flush=True)

    def one(j):
        model, framing, arm, N, seed = j
        rec = run(N, RHO, ROUNDS[N], arm, LLMDecider(model, arm, framing), seed=seed,
                  meta=dict(model=model, framing=framing, stage="E3"), out=str(out))
        return j, rec["invalid"]

    with ThreadPoolExecutor(a.parallel) as ex:
        for k, fu in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            j, inv = fu.result()
            print(k, j, "invalid", inv, flush=True)


def cmd_rewind(a):
    out = ROOT / "results" / "chase_e4_rewind.jsonl"
    done = _done(out, ("framing", "arm", "seed"))
    jobs = [(arm, 9000 + g) for g in range(a.groups) for arm in a.arms if (a.framing, arm, 9000 + g) not in done]
    print(len(jobs), "groups to run", flush=True)

    def one(j):
        arm, seed = j
        # membership is fixed: agent index i is always members[i]; only names, pairings and coin flips change
        rec = run(len(a.members), RHO, ROUNDS[4], arm, LLMDecider(list(a.members), arm, a.framing, tag="chase_e4"),
                  seed=seed, meta=dict(members=list(a.members), framing=a.framing, stage="E4"), out=str(out))
        return j, rec["invalid"]

    with ThreadPoolExecutor(a.parallel) as ex:
        for k, fu in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            print(k, *fu.result(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "rewind"])
    ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--controls", nargs="*", default=[])
    ap.add_argument("--sizes", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--groups", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--members", nargs="*", default=[])
    ap.add_argument("--framing", default="comp")
    ap.add_argument("--predictions", default=str(ROOT / "results" / "chase_e2_predictions.json"))
    a = ap.parse_args()
    (cmd_run if a.cmd == "run" else cmd_rewind)(a)


if __name__ == "__main__":
    main()
