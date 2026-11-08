"""Bonabeau threshold test (follow-up, 2026-11-08). Lacasa & Luque's mean-field instability threshold, adapted to
our protocol (each agent initiates with prob rho and is targeted at the same mean rate, so it meets ~2*rho contests
per round; the Lacasa-Luque value is halved):
        rho_c = 2 mu (N-1) / [ s (1-mu) N (1+F) ],   mu = 0.1, F = 1,
with s the measured finite-difference status sensitivity (eq. 3 of the paper; posterior draws). Model/framing:
Gemma-4-26B-A4B, neutral wording + competence sentence (s ~ 0.29, so rho_c ~ 0.34 at N=8, 0.36 at N=16).

  predict -> results/threshold_predictions.json (write-once): mean-field rho_c distribution; frozen kernel
             simulations (state-matched NW, Bonabeau, status-blind null) of q and sd(h) at every (rho, N).
  run     -> results/threshold_pop.jsonl  (N=8: 3 seeds; N=16: 2 seeds; 80 rounds; metrics over last 40)
  eval

Pre-declared analysis:
  (a) onset rho_on = smallest rho whose observed seed-mean q exceeds the null kernel's 95th percentile for the
      seed mean; compare with the mean-field rho_c 90% interval and with the kernel-simulated onset;
  (b) coverage of observed seed means by the frozen NW-kernel 90% intervals across the sweep;
  (c) finite size: slope of q vs rho around rho_c for N=16 vs N=8.
"""
import argparse, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from population2 import KERNELS, KernelDecider, LLMDecider, run, order_params
from calibrate import CONDS
from hysteresis import calib_rows
from evaluate import seed_mean_dist

ROOT = Path(__file__).resolve().parents[1]
MODEL, COND = "adaption:gemma-4-26b-a4b-it", "ended+comp"
MU, F, T, WIN = 0.1, 1.0, 80, 40
RHOS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0]
SEEDS = {8: [1101, 1102, 1103], 16: [1201, 1202]}


def s_draws(d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    cells = {x: (d[d.delta == x].lead.sum(), (d.delta == x).sum()) for x in sorted(set(d.delta)) if abs(x) <= 1 and x != 0}
    pos = [x for x in cells if x > 0 and -x in cells]
    return np.mean([4 * ((rng.beta(cells[x][0] + .5, cells[x][1] - cells[x][0] + .5, n)
                          - rng.beta(cells[-x][0] + .5, cells[-x][1] - cells[-x][0] + .5, n)) / 2) / x for x in pos], axis=0)


def rho_c(s, N):
    return 2 * MU * (N - 1) / (s * (1 - MU) * N * (1 + F))


def cmd_predict(a):
    out = ROOT / "results" / "threshold_predictions.json"
    if out.exists():
        sys.exit("frozen; refusing to overwrite")
    from predict import load_calib
    grid = load_calib(); grid = grid[(grid.model == MODEL) & (grid.cond == COND)]
    sd = s_draws(grid)
    res = dict(model=MODEL, cond=COND, mu=MU, F=F, rounds=T, window=WIN, rhos=RHOS, seeds=SEEDS,
               s_median=float(np.median(sd)), s_90=[float(np.quantile(sd, .05)), float(np.quantile(sd, .95))],
               rho_c={str(N): dict(median=float(np.median(rho_c(sd[sd > 0], N))),
                                   q05=float(np.quantile(rho_c(sd[sd > 0], N), .05)),
                                   q95=float(np.quantile(rho_c(sd[sd > 0], N), .95)),
                                   frac_s_nonpositive=float((sd <= 0).mean())) for N in SEEDS},
               preds=[])
    print("s median", res["s_median"], "90%", res["s_90"], "rho_c", res["rho_c"])
    rows = calib_rows(MODEL, COND)
    rng = np.random.default_rng(0)
    for kern in ("nw", "bonabeau", "null"):
        for N in SEEDS:
            for rho in RHOS:
                qs, sds = [], []
                for b in range(a.B):
                    k = KERNELS[kern](rows, rng)
                    for s in range(a.S):
                        rec = run(N, rho, MU, T, KernelDecider(k), seed=int(rng.integers(1e9)))
                        o = order_params(rec, WIN); qs.append(o["q"]); sds.append(o["sd_h"])
                res["preds"].append(dict(kernel=kern, N=N, rho=rho, q=qs, sd_h=sds, B=a.B, S=a.S))
            print(kern, N, [round(float(np.mean(p["q"])), 2) for p in res["preds"] if p["kernel"] == kern and p["N"] == N])
    res["sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    out.write_text(json.dumps(res)); print("frozen", res["sha256"])


def cmd_run(a):
    P = json.load(open(ROOT / "results" / "threshold_predictions.json"))
    cfgs = [(N, rho, s) for N, ss in SEEDS.items() for rho in RHOS for s in ss]
    def go(c):
        N, rho, s = c
        rec = run(N, rho, MU, T, LLMDecider(MODEL, CONDS[COND]), seed=s,
                  meta=dict(model=MODEL, cond=COND, tag="threshold", pred_sha=P["sha256"]),
                  out=ROOT / "results" / "threshold_pop.jsonl")
        o = order_params(rec, WIN)
        print(json.dumps(dict(N=N, rho=rho, seed=s, q=round(o["q"], 3), sd_h=round(o["sd_h"], 3))), flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def cmd_eval(a):
    P = json.load(open(ROOT / "results" / "threshold_predictions.json"))
    recs = [json.loads(l) for l in open(ROOT / "results" / "threshold_pop.jsonl")]
    obs = pd.DataFrame([dict(N=r["N"], rho=r["rho"], **order_params(r, WIN)) for r in recs])
    out = dict(rho_c=P["rho_c"], s_median=P["s_median"], rows=[])
    for N in SEEDS:
        onset = {}
        for rho in RHOS:
            o = obs[(obs.N == N) & (obs.rho == rho)]
            if o.empty:
                continue
            k = len(o); row = dict(N=N, rho=rho, seeds=k, q_obs=float(o.q.mean()), sd_obs=float(o.sd_h.mean()))
            for kern in ("nw", "bonabeau", "null"):
                p = [x for x in P["preds"] if x["kernel"] == kern and x["N"] == N and x["rho"] == rho][0]
                for m in ("q", "sd_h"):
                    dist = seed_mean_dist(p[m], p["B"], p["S"], k)
                    row[f"{m}_{kern}_mean"] = float(dist.mean())
                    row[f"{m}_{kern}_lo"], row[f"{m}_{kern}_hi"] = float(np.quantile(dist, .05)), float(np.quantile(dist, .95))
                if kern == "null":
                    row["q_null_p95"] = float(np.quantile(seed_mean_dist(p["q"], p["B"], p["S"], k), .95))
            row["above_null"] = row["q_obs"] > row["q_null_p95"]
            row["covered_nw_q"] = row["q_nw_lo"] <= row["q_obs"] <= row["q_nw_hi"]
            out["rows"].append(row)
        rr = [r for r in out["rows"] if r["N"] == N]
        on = [r["rho"] for r in rr if r["above_null"]]
        out[f"onset_N{N}"] = min(on) if on else None
        print(f"N={N}: rho_c mean-field {P['rho_c'][str(N)]['median']:.2f} [{P['rho_c'][str(N)]['q05']:.2f},{P['rho_c'][str(N)]['q95']:.2f}]"
              f"  observed onset (q above null p95): {out[f'onset_N{N}']}")
        for r in rr:
            print(f"   rho={r['rho']:.2f} q={r['q_obs']:.2f} | NW {r['q_nw_mean']:.2f} [{r['q_nw_lo']:.2f},{r['q_nw_hi']:.2f}]"
                  f" | null p95 {r['q_null_p95']:.2f} {'*' if r['above_null'] else ''}")
    json.dump(out, open(ROOT / "results" / "threshold_eval.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "run", "eval"])
    ap.add_argument("--B", type=int, default=12)
    ap.add_argument("--S", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args()
    {"predict": cmd_predict, "run": cmd_run, "eval": cmd_eval}[a.cmd](a)
