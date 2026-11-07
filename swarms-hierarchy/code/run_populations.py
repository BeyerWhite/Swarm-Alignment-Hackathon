"""Run several LLM-in-the-loop v2 populations concurrently (each population is sequential in rounds)."""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from population2 import LLMDecider, run, order_params
from calibrate import CONDS

ROOT = Path(__file__).resolve().parents[1]


def one(cfg, out):
    dec = LLMDecider(cfg["model"], CONDS[cfg["cond"]])
    rec = run(cfg["N"], cfg["rho"], cfg.get("mu", 0.1), cfg.get("rounds", 40), dec, seed=cfg["seed"],
              meta=dict(model=cfg["model"], cond=cfg["cond"], tag=cfg.get("tag", "v2")), out=out)
    o = order_params(rec)
    print(json.dumps({**cfg, **{k: round(v, 3) for k, v in o.items()}, "invalid": rec["invalid"]}), flush=True)
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "results" / "population_v2.jsonl"))
    a = ap.parse_args()
    cfgs = json.loads(a.configs)
    if Path(a.out).exists():                       # resume: skip configs already completed
        have = {(r["model"], r["cond"], r["rho"], r["N"], r["seed"]) for r in map(json.loads, open(a.out))}
        cfgs = [c for c in cfgs if (c["model"], c["cond"], c["rho"], c["N"], c["seed"]) not in have]
        print(f"resuming: {len(cfgs)} configs left", flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda c: one(c, a.out), cfgs))
