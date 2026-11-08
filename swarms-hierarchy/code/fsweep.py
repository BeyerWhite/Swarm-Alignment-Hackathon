"""Loss-weight sweep (follow-up, 2026-11-08): does the Bonabeau loss weight F move the hierarchy threshold as
Lacasa & Luque predict?  Protocol mean-field threshold (each agent meets ~2*rho contests per round):
        rho_c(F) = 2 mu (N-1) / [ s_F (1-mu) N (1+F) ],   mu = 0.1,  N = 8,
with s_F the finite-difference status sensitivity measured under F-wording ("the other agent gets -F").
F = 1 is the running threshold sweep (results/threshold_*); here F in {2, 3}.
With F > 1 the mean score drifts to about (1-mu) rho (1-F) / mu, so calibration uses mean levels {-12, -6, 0}
plus state-matched calls at states visited by surrogate populations simulated with the grid kernel.

  calib    grid + state-matched calibration                      -> results/fsweep_calib.jsonl
  predict  frozen predictions (write-once)                         -> results/fsweep_predictions.json
  run      LLM populations (N=8, rho sweep, 3 seeds, 80 rounds)    -> results/fsweep_pop.jsonl
  eval     onset vs rho_c(F), coverage, comparison with F = 1     -> results/fsweep_eval.json
"""
import argparse, hashlib, json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from llm import call, parse_choice
from prompts_v2 import SYSTEM, pair_prompt
from calibrate import CONDS
from population2 import KERNELS, KernelDecider, LLMDecider, run, order_params
from evaluate import seed_mean_dist

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
MODEL, COND = "adaption:gemma-4-26b-a4b-it", "ended+comp"
MU, N, T, WIN = 0.1, 8, 80, 40
FS = [2.0, 3.0]
RHOS = [0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7]
SEEDS = [1301, 1302, 1303]
DELTAS = [-16, -8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 16]
MEANS = [-12.0, -6.0, 0.0]


def disp(x):
    return float(f"{x:+.1f}")


def calib_df(F):
    rows = [json.loads(l) for l in open(R / "fsweep_calib.jsonl")]
    rows = [r for r in rows if r["F"] == F and r["choice"]]
    df = pd.DataFrame(rows)
    df["lead"] = (df.choice == "LEAD").astype(float)
    df["delta"] = (df.h_me.map(disp) - df.h_other.map(disp)).round(2)
    df["mean"] = ((df.h_me.map(disp) + df.h_other.map(disp)) / 2).round(2)
    return df


def ask(jobs, F, tag, workers):
    out = open(R / "fsweep_calib.jsonl", "a"); lock = threading.Lock()
    def one(j):
        resp = call(MODEL, j["sys"], j["prompt"], tag=tag)
        with lock:
            out.write(json.dumps(dict(F=F, design=j["design"], h_me=j["h_me"], h_other=j["h_other"],
                                      response=resp, choice=parse_choice(resp))) + "\n"); out.flush()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))


def cmd_calib(a):
    rng = random.Random(41)
    for F in FS:
        jobs = []
        for dlt in DELTAS:
            for m in MEANS:
                for _ in range(8 if abs(dlt) <= 1 else 4):
                    me, ot = rng.sample(range(1, 100), 2)
                    hs, ho = m + dlt / 2, m - dlt / 2
                    jobs.append(dict(design="grid", h_me=hs, h_other=ho, sys=SYSTEM.format(me=me, n=N),
                                     prompt=pair_prompt(me, ot, hs, ho, rng, CONDS[COND], f=F)))
        ask(jobs, F, "fsweep_calib", a.workers)
        # state-matched: states visited by populations simulated with this grid kernel across the rho sweep
        k = KERNELS["flexible"](calib_df(F))
        states = []
        srng = np.random.default_rng(int(F))
        for rho in RHOS:
            for _ in range(3):
                rec = run(N, rho, MU, T, KernelDecider(k), seed=int(srng.integers(1e9)), F=F)
                h = np.zeros(N)
                for x in rec["history"]:
                    d = [round(v, 1) for v in h]
                    for i, j, *_ in x["events"]:
                        states += [(d[i], d[j]), (d[j], d[i])]
                    h = np.array(x["h"])
        jobs = []
        for _ in range(400):
            hs, ho = rng.choice(states)
            me, ot = rng.sample(range(1, 100), 2)
            jobs.append(dict(design="state", h_me=hs, h_other=ho, sys=SYSTEM.format(me=me, n=N),
                             prompt=pair_prompt(me, ot, hs, ho, rng, CONDS[COND], f=F)))
        ask(jobs, F, "fsweep_statecal", a.workers)
        print("calibrated F =", F, flush=True)


def s_draws(df, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    g = df[df.design == "grid"]
    cells = {x: (g[g.delta == x].lead.sum(), (g.delta == x).sum()) for x in sorted(set(g.delta)) if abs(x) <= 1 and x != 0}
    pos = [x for x in cells if x > 0 and -x in cells]
    return np.mean([4 * ((rng.beta(cells[x][0] + .5, cells[x][1] - cells[x][0] + .5, n)
                          - rng.beta(cells[-x][0] + .5, cells[-x][1] - cells[-x][0] + .5, n)) / 2) / x for x in pos], axis=0)


def rho_c(s, F):
    return 2 * MU * (N - 1) / (s * (1 - MU) * N * (1 + F))


def cmd_predict(a):
    out = R / "fsweep_predictions.json"
    if out.exists():
        sys.exit("frozen; refusing to overwrite")
    res = dict(model=MODEL, cond=COND, mu=MU, N=N, rounds=T, window=WIN, Fs=FS, rhos=RHOS, seeds=SEEDS, F={}, preds=[])
    rng = np.random.default_rng(0)
    for F in FS:
        df = calib_df(F)
        sd = s_draws(df)
        rc = rho_c(sd[sd > 0], F)
        res["F"][str(F)] = dict(n_calib=len(df), s_median=float(np.median(sd)), s_90=[float(np.quantile(sd, .05)), float(np.quantile(sd, .95))],
                                rho_c_median=float(np.median(rc)), rho_c_90=[float(np.quantile(rc, .05)), float(np.quantile(rc, .95))],
                                frac_s_nonpositive=float((sd <= 0).mean()),
                                mean_score_fixed_point={str(r): (1 - MU) * r * (1 - F) / MU for r in RHOS})
        print("F", F, res["F"][str(F)], flush=True)
        for kern in ("nw", "bonabeau", "null"):
            for rho in RHOS:
                qs, sds = [], []
                for b in range(a.B):
                    k = KERNELS[kern](df, rng)
                    for s in range(a.S):
                        o = order_params(run(N, rho, MU, T, KernelDecider(k), seed=int(rng.integers(1e9)), F=F), WIN)
                        qs.append(o["q"]); sds.append(o["sd_h"])
                res["preds"].append(dict(F=F, kernel=kern, rho=rho, q=qs, sd_h=sds, B=a.B, S=a.S))
            print(" ", kern, [round(float(np.mean(p["q"])), 2) for p in res["preds"] if p["F"] == F and p["kernel"] == kern], flush=True)
    res["sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    out.write_text(json.dumps(res)); print("frozen", res["sha256"])


def cmd_run(a):
    P = json.load(open(R / "fsweep_predictions.json"))
    a7 = R / "fsweep_addendum_A7.json"
    extra = json.load(open(a7))["extra_rhos"] if a7.exists() else []
    cfgs = [(F, rho, s) for F in FS for rho in RHOS + extra for s in SEEDS]
    a8 = R / "fsweep_addendum_A8.json"
    if a8.exists():
        cfgs += [(e["F"], e["rho"], s) for e in json.load(open(a8))["extra"] for s in e["seeds"]]
    def go(c):
        F, rho, s = c
        rec = run(N, rho, MU, T, LLMDecider(MODEL, CONDS[COND]), seed=s, F=F,
                  meta=dict(model=MODEL, cond=COND, tag="fsweep", pred_sha=P["sha256"]), out=R / "fsweep_pop.jsonl")
        o = order_params(rec, WIN)
        print(json.dumps(dict(F=F, rho=rho, seed=s, q=round(o["q"], 3), sd_h=round(o["sd_h"], 3))), flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def cmd_eval(a):
    P = json.load(open(R / "fsweep_predictions.json"))
    recs = [json.loads(l) for l in open(R / "fsweep_pop.jsonl")]
    out = dict(F=P["F"], rows=[])
    for F in FS:
        rows = []
        for rho in RHOS:
            rr = [r for r in recs if r["F"] == F and r["rho"] == rho]
            if not rr:
                continue
            qs = [order_params(r, WIN)["q"] for r in rr]; k = len(qs)
            row = dict(F=F, rho=rho, k=k, q_obs=float(np.mean(qs)))
            for kern in ("nw", "bonabeau", "null"):
                p = [x for x in P["preds"] if x["F"] == F and x["kernel"] == kern and x["rho"] == rho][0]
                dd = seed_mean_dist(p["q"], p["B"], p["S"], k)
                row[kern] = [float(dd.mean()), float(np.quantile(dd, .05)), float(np.quantile(dd, .95))]
            row["above_null"] = row["q_obs"] > row["null"][2]
            row["in_nw"] = row["nw"][1] <= row["q_obs"] <= row["nw"][2]
            rows.append(row)
        on = [r["rho"] for r in rows if r["above_null"]]
        out[f"onset_F{F:g}"] = min(on) if on else None
        out["rows"] += rows
        f = P["F"][str(F)]
        print(f"F={F:g}: s={f['s_median']:.2f} rho_c={f['rho_c_median']:.2f} [{f['rho_c_90'][0]:.2f},{f['rho_c_90'][1]:.2f}]"
              f"  observed onset {out[f'onset_F{F:g}']}  NW coverage {sum(r['in_nw'] for r in rows)}/{len(rows)}")
        for r in rows:
            print(f"   rho={r['rho']:.2f} q={r['q_obs']:.2f} NW {r['nw'][0]:.2f}[{r['nw'][1]:.2f},{r['nw'][2]:.2f}] null p95 {r['null'][2]:.2f}{' *' if r['above_null'] else ''}")
    json.dump(out, open(R / "fsweep_eval.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "predict", "run", "eval"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--B", type=int, default=12)
    ap.add_argument("--S", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args()
    {"calib": cmd_calib, "predict": cmd_predict, "run": cmd_run, "eval": cmd_eval}[a.cmd](a)
