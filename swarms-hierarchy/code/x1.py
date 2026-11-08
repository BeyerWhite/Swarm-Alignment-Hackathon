"""X1: mixed-model populations (quenched disorder). Same protocol as population2 (N=8, rho=.5, mu=.1, 40 rounds,
synchronous, displayed 0.1, lone LEAD wins, +1/-1); identical prompt for all agents; agent type = model family.

  predict -> results/x1_predictions.json   (write-once, hash-stamped; per-type kernels; null = all majority type)
  run     -> results/x1_pop.jsonl
  eval
Compositions: A = minority (assertive) model, B = majority model. Types assigned to random positions per seed.
"""
import argparse, hashlib, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pointbiserialr
sys.path.insert(0, str(Path(__file__).parent))
from population2 import KERNELS, run
from predict import load_calib, load_statecal
from calibrate import CONDS
from llm import call, parse_choice
from prompts_v2 import SYSTEM, TASKS, pair_prompt

ROOT = Path(__file__).resolve().parents[1]
N, RHO, MU, T = 8, 0.5, 0.1, 40
ARMS = {  # name: (minority model, majority model, n_minority, cond)
    "ds1_gemma7": ("adaption:deepseek-v4-flash", "adaption:gemma-4-31b-it", 1, "norm100"),
    "ds4_gemma4": ("adaption:deepseek-v4-flash", "adaption:gemma-4-31b-it", 4, "norm100"),
    "ds1_glm7_w0": ("adaption:deepseek-v4-flash", "adaption:glm-5.2", 1, "norm0"),
}
SEEDS = [801, 802, 803, 804, 805]


def types_for(arm, seed):
    a, b, k, _ = ARMS[arm]
    rng = random.Random(seed)
    idx = set(rng.sample(range(N), k))
    return [a if i in idx else b for i in range(N)]


def kernel_rows(model, cond):
    c = load_calib(); s = load_statecal()
    d = pd.concat([c, s[[x for x in s.columns if x in c.columns]]], ignore_index=True) if len(s) else c
    return d[(d.model == model) & (d.cond == cond)]


class MixedKernelDecider:
    def __init__(self, kern_by_agent):
        self.k, self.invalid = kern_by_agent, 0

    def __call__(self, pairs, h, names, N_, rp, rs, mu, Q, F):
        out = []
        for i, j in pairs:
            hi, hj = round(h[i], 1), round(h[j], 1); m = (hi + hj) / 2
            out.append(("LEAD" if rs.random() < self.k[i](hi - hj, m) else "DEFER",
                        "LEAD" if rs.random() < self.k[j](hj - hi, m) else "DEFER"))
        return out


class MixedLLMDecider:
    def __init__(self, models, cond):
        self.models, self.cond, self.invalid = models, CONDS[cond], 0

    def __call__(self, pairs, h, names, N_, rp, rs, mu, Q, F):
        jobs = []
        for i, j in pairs:
            task = rp.choice(TASKS)
            for a, b in ((i, j), (j, i)):
                jobs.append((self.models[a], SYSTEM.format(me=names[a], n=N_),
                             pair_prompt(names[a], names[b], round(h[a], 1), round(h[b], 1), rp, self.cond,
                                         q=Q, f=F, mu=mu, task=task)))

        def one(jb):
            for _ in range(3):
                c = parse_choice(call(jb[0], jb[1], jb[2], tag="x1_pop"))
                if c:
                    return c
            return None
        with ThreadPoolExecutor(min(len(jobs), 8) or 1) as ex:
            ch = list(ex.map(one, jobs))
        out = []
        for k in range(len(pairs)):
            c = [ch[2 * k], ch[2 * k + 1]]
            for z in range(2):
                if c[z] is None:
                    self.invalid += 1; c[z] = "LEAD" if rs.random() < .5 else "DEFER"
            out.append(tuple(c))
        return out


def metrics(rec, types, minority):
    H = np.array([x["h"] for x in rec["history"]])
    is_min = np.array([t == minority for t in types])
    final = H[-10:].mean(0)
    rank = (-final).argsort().argsort() + 1                 # 1 = top
    top_t = [int(np.argmax(x)) for x in H]
    held = [is_min[t] for t in top_t]
    first = next((t for t, x in enumerate(held) if x), None)
    within = np.mean([final[is_min].std() if is_min.sum() > 1 else 0, final[~is_min].std()])
    tot = H[-20:].var(axis=1).mean()
    return dict(
        pb_corr=float(pointbiserialr(is_min.astype(int), -rank)[0]) if 0 < is_min.sum() < N else 0.0,
        rank1_minority=bool(is_min[int(np.argmax(final))]),
        minority_mean_rank=float(rank[is_min].mean()),
        type_gap=float(final[is_min].mean() - final[~is_min].mean()), within_sd=float(within),
        first_rank1_minority=first, frac_rounds_minority_top=float(np.mean(held)),
        sd_h=float(H[-20:].std(axis=1).mean()), q=float(H[-20:].mean(0).var() / tot) if tot > 0 else 0.0)


def cmd_predict(a):
    out = ROOT / "results" / "x1_predictions.json"
    if out.exists():
        sys.exit("frozen; refusing to overwrite")
    res = dict(N=N, rho=RHO, mu=MU, rounds=T, arms={}, kernels=["nw", "flexible"], null="all agents majority type")
    rng = np.random.default_rng(0)
    for arm, (A, B, k, cond) in ARMS.items():
        dA, dB = kernel_rows(A, cond), kernel_rows(B, cond)
        res["arms"][arm] = dict(minority=A, majority=B, k=k, cond=cond, n_calib=[len(dA), len(dB)], preds={})
        for kern in ("nw", "flexible", "null"):
            samp = []
            for b in range(a.B):
                kb = KERNELS["nw" if kern == "null" else kern](dB, rng)
                ka = kb if kern == "null" else KERNELS[kern](dA, rng)
                for s in range(a.S):
                    seed = int(rng.integers(1e9)); ty = types_for(arm, seed)
                    rec = run(N, RHO, MU, T, MixedKernelDecider([ka if t == A else kb for t in ty]), seed=seed)
                    samp.append(metrics(rec, ty, A))
            res["arms"][arm]["preds"][kern] = {m: [x[m] for x in samp] for m in samp[0]}
            p = res["arms"][arm]["preds"][kern]
            print(f"{arm:12s} {kern:8s} P(rank1=minority)={np.mean(p['rank1_minority']):.2f} "
                  f"pb_corr={np.mean(p['pb_corr']):.2f} type_gap={np.mean(p['type_gap']):.2f} sd_h={np.mean(p['sd_h']):.2f}")
    res["decision_rule"] = ("'Lone assertive agent leads' iff P(rank1=DeepSeek) across seeds is significantly above 1/8 "
                            "(one-sided binomial, alpha=.05) AND the identity (nw) kernel's predictive 90% interval for "
                            "the seed-mean contains the observed rate. Leads but kernel misses => interaction effects.")
    res["sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    out.write_text(json.dumps(res)); print("frozen", res["sha256"])


def cmd_run(a):
    P = json.load(open(ROOT / "results" / "x1_predictions.json"))
    cfgs = [(arm, s) for arm in a.arms for s in SEEDS]
    def go(c):
        arm, s = c; A, B, k, cond = ARMS[arm]; ty = types_for(arm, s)
        rec = run(N, RHO, MU, T, MixedLLMDecider(ty, cond), seed=s,
                  meta=dict(arm=arm, types=ty, cond=cond, tag="x1", pred_sha=P["sha256"]),
                  out=ROOT / "results" / "x1_pop.jsonl")
        print(json.dumps(dict(arm=arm, seed=s, **metrics(rec, ty, A))), flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def cmd_eval(a):
    from scipy.stats import binomtest
    P = json.load(open(ROOT / "results" / "x1_predictions.json"))
    recs = [json.loads(l) for l in open(ROOT / "results" / "x1_pop.jsonl")]
    for arm, spec in P["arms"].items():
        rr = [r for r in recs if r["arm"] == arm]
        if not rr:
            continue
        obs = [metrics(r, r["types"], spec["minority"]) for r in rr]
        k = sum(o["rank1_minority"] for o in obs)
        null_p = spec["k"] / N                      # composition null: minority share of agents
        line = f"{arm}: rank1(last-10 mean)=minority {k}/{len(obs)} (binom p vs composition null {null_p:.3f} = {binomtest(k, len(obs), null_p, 'greater').pvalue:.3f})"
        for m in ("type_gap", "pb_corr", "sd_h"):
            line += f" | {m} obs={np.mean([o[m] for o in obs]):.2f}"
            for kern in ("nw", "null"):
                from evaluate import seed_mean_dist
                v = np.array(P["arms"][arm]["preds"][kern][m], float)
                dist = seed_mean_dist(v, 20, 8, len(obs))
                line += f" {kern}={v.mean():.2f}[{np.quantile(dist,.05):.2f},{np.quantile(dist,.95):.2f}]"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "run", "eval"])
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    {"predict": cmd_predict, "run": cmd_run, "eval": cmd_eval}[a.cmd](a)
