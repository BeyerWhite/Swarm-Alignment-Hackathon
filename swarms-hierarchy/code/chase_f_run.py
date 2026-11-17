"""Follow-up population runner (amendment C8: E7, E8, E9). The engine (chase_pop.run, LLMDecider) is unchanged.

  chase_f_run.py --stage E9 --cells "model|framing" ... --size 4 --plan OWN=40 LOG=40 DYAD=20 \
                 --seed0 14000 --out results/chase_e9_pop.jsonl [--predictions FILE] [--parallel 6] [--dry]

Seeds are seed0 + g, the same in every arm and cell; an arm with fewer groups uses the first ones. A finished
(model, framing, arm, N, seed) is skipped on restart. With --predictions, a (model, framing, arm, N) without a frozen
forecast in that file is refused.
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from chase_pop import run, LLMDecider
from chase_e2 import ROUNDS, RHO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--size", type=int, required=True, choices=[4, 8])
    ap.add_argument("--plan", nargs="+", required=True, help="ARM=groups ...")
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--predictions", default="")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    plan = {x.split("=")[0]: int(x.split("=")[1]) for x in a.plan}
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            done.add((r["model"], r["framing"], r["arm"], r["N"], r["seed"]))
    frozen = None
    if a.predictions:
        p = json.loads(Path(a.predictions).read_text())
        frozen = {(e["model"], e["framing"], e["arm"], e["N"]) for e in p["predictions"]}
    jobs = []
    for cell in a.cells:
        model, framing = cell.split("|")
        for arm, G in plan.items():
            if frozen is not None and (model, framing, arm, a.size) not in frozen:
                sys.exit(f"no frozen forecast for {(model, framing, arm, a.size)}")
            jobs += [(model, framing, arm, a.size, a.seed0 + g) for g in range(G)
                     if (model, framing, arm, a.size, a.seed0 + g) not in done]
    jobs.sort(key=lambda j: j[4])                       # interleave arms and cells
    print(len(jobs), "groups to run ->", out, flush=True)
    if a.dry:
        return

    def one(j):
        model, framing, arm, N, seed = j
        rec = run(N, RHO, ROUNDS[N], arm, LLMDecider(model, arm, framing, tag="chase_" + a.stage.lower()), seed=seed,
                  meta=dict(model=model, framing=framing, stage=a.stage), out=str(out))
        return j, rec["invalid"]

    with ThreadPoolExecutor(a.parallel) as ex:
        for k, fu in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            j, inv = fu.result()
            print(k, j, "invalid", inv, flush=True)


if __name__ == "__main__":
    main()
