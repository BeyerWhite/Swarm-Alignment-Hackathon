"""Step 2a: does a seeded leader persist? Equal start vs seeded start (one agent at the top of a norm100-like
hierarchy, others matched to a simulated norm100 steady state). Predictions from frozen kernels (state-matched NW
+ Bonabeau, grid+state calibration) are written once BEFORE any LLM run.
  predict -> results/hyst_predictions.json ; run -> results/hyst_pop.jsonl ; eval
"""
import argparse, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from population2 import KERNELS, KernelDecider, LLMDecider, run
from predict import load_calib, load_statecal
from calibrate import CONDS

ROOT = Path(__file__).resolve().parents[1]
N, RHO, MU, T = 8, 0.5, 0.1, 40


def calib_rows(model, cond):
    c = load_calib(); s = load_statecal()
    d = pd.concat([c, s[[x for x in s.columns if x in c.columns]]], ignore_index=True) if len(s) else c
    return d[(d.model == model) & (d.cond == cond)]


def seeded_h0(model, seed=0):
    """Sorted final scores of a simulated norm100 population (kernel only), top agent first."""
    k = KERNELS["nw"](calib_rows(model, "norm100"))
    rec = run(N, RHO, MU, 60, KernelDecider(k), seed=seed)
    return sorted(rec["history"][-1]["h"], reverse=True)


def traj_metrics(rec):
    H = np.array([x["h"] for x in rec["history"]])
    lead0 = int(np.argmax(rec["h0"])) if rec.get("h0") and np.ptp(rec["h0"]) > 0 else None
    sd = H.std(axis=1)
    lost = next((t for t in range(len(H)) if lead0 is not None and int(np.argmax(H[t])) != lead0), len(H))
    return dict(sd_early=float(sd[:5].mean()), sd_late=float(sd[-10:].mean()),
                leader_rank1_rounds=int(lost) if lead0 is not None else None,
                leader_final_rank=int((H[-1] > H[-1][lead0]).sum()) + 1 if lead0 is not None else None)


def cmd_predict(a):
    out = ROOT / "results" / "hyst_predictions.json"
    if out.exists():
        sys.exit("frozen; refusing to overwrite")
    h0 = seeded_h0(a.model)
    res = dict(model=a.model, h0=h0, N=N, rho=RHO, mu=MU, rounds=T, preds=[])
    rng = np.random.default_rng(0)
    for cond in a.conds:
        d = calib_rows(a.model, cond)
        for kern in ("nw", "bonabeau"):
            for arm in ("equal", "seeded"):
                samp = []
                for b in range(a.B):
                    k = KERNELS[kern](d, rng)
                    for s in range(a.S):
                        rec = run(N, RHO, MU, T, KernelDecider(k), seed=int(rng.integers(1e9)),
                                  h0=h0 if arm == "seeded" else None)
                        samp.append(traj_metrics(rec))
                res["preds"].append(dict(cond=cond, kernel=kern, arm=arm,
                                         **{m: [x[m] for x in samp] for m in samp[0]}))
                p = res["preds"][-1]
                print(cond, kern, arm, "sd_late", round(np.mean(p["sd_late"]), 2),
                      "leader rank1 rounds", None if arm == "equal" else round(np.mean(p["leader_rank1_rounds"]), 1))
    res["sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    out.write_text(json.dumps(res)); print("frozen", res["sha256"])


def cmd_run(a):
    P = json.load(open(ROOT / "results" / "hyst_predictions.json"))
    cfgs = [(c, arm, s) for c in a.conds for arm in ("equal", "seeded") for s in (701, 702, 703)]
    def go(x):
        cond, arm, s = x
        rec = run(N, RHO, MU, T, LLMDecider(a.model, CONDS[cond]), seed=s,
                  h0=P["h0"] if arm == "seeded" else None, meta=dict(model=a.model, cond=cond, arm=arm, tag="hyst"),
                  out=ROOT / "results" / "hyst_pop.jsonl")
        print(json.dumps(dict(cond=cond, arm=arm, seed=s, **traj_metrics(rec))), flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def cmd_eval(a):
    P = json.load(open(ROOT / "results" / "hyst_predictions.json"))
    recs = [json.loads(l) for l in open(ROOT / "results" / "hyst_pop.jsonl")]
    for cond in a.conds:
        for arm in ("equal", "seeded"):
            o = [traj_metrics(r) for r in recs if r["cond"] == cond and r["arm"] == arm]
            if not o:
                continue
            line = f"{cond:10s} {arm:6s} obs sd_late={np.mean([x['sd_late'] for x in o]):.2f}"
            if arm == "seeded":
                line += f" leader_rank1={np.mean([x['leader_rank1_rounds'] for x in o]):.1f}"
            for kern in ("nw", "bonabeau"):
                p = [q for q in P["preds"] if q["cond"] == cond and q["arm"] == arm and q["kernel"] == kern][0]
                s = np.array(p["sd_late"]); line += f" | {kern} sd_late={s.mean():.2f} [{np.quantile(s,.05):.2f},{np.quantile(s,.95):.2f}]"
                if arm == "seeded":
                    l = np.array(p["leader_rank1_rounds"]); line += f" rank1={l.mean():.1f} [{np.quantile(l,.05):.0f},{np.quantile(l,.95):.0f}]"
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "run", "eval"])
    ap.add_argument("--model", default="adaption:gemma-4-31b-it")
    ap.add_argument("--conds", nargs="+", default=["norm0", "ended", "norm50"])
    ap.add_argument("--B", type=int, default=15)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    {"predict": cmd_predict, "run": cmd_run, "eval": cmd_eval}[a.cmd](a)
