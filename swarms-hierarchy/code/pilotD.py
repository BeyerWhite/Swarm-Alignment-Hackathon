"""Prospective pilot D (2026-11-08 evening): does unshrunk, monotone calibration at visited states predict NEW LLM
populations better than the pre-registered Jeffreys estimator? Targeted pilot (5 seeds per condition) on two
historically high-residual conditions, so selection is acknowledged; everything below is frozen before any new
population is run.

Design
  conditions   Gemma-4-26B norm100, Qwen3.5-397B norm0 (protocol: N = 8, rho = 0.5, mu = 0.1, 40 rounds, window 20)
  anchors      simulate 24 populations each with the old-grid isotonic and Jeffreys kernels; project visited displayed
               states to the nearest mean slice {-4, 0, 4} and |delta| rounded to 0.2 (|delta| <= 16, delta != 0);
               take the 10 most visited positive deltas per condition plus their mirrors -> 20 anchors x 30 calls
  kernels      2 x 2: {Jeffreys (frozen pipeline: Beta(k+1/2, n-k+1/2) cell draws), isotonic (weighted PAVA per mean
               slice, direction = sign of calibration s; uncertainty: Binomial(n, p_iso) resampling with p clipped to
               [1/(2(n+1)), 1 - 1/(2(n+1))] for the draws only, then refit)} x {old grid, old grid + new anchors}
  primary      sd(h) of the 5-seed mean. Candidate = isotonic(new); reference = Jeffreys(new).
               Success in a condition: LPD(candidate) > LPD(reference) and |obs - pred mean| <= 0.5.
               Secondary: q; the old-grid kernels (separates sampling from estimator effects); 90% coverage.
  power        frozen with the predictions: P(candidate LPD > reference LPD) when 5-seed means are drawn from the
               candidate predictive distribution.

  anchors | calib | predict | run | eval        (outputs results/pilotD_*)
"""
import argparse, hashlib, json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from llm import call, parse_choice
from prompts_v2 import SYSTEM, pair_prompt
from calibrate import CONDS
from population2 import KernelDecider, LLMDecider, run, order_params, kernel_flexible
from predict import load_calib
from posthoc_attenuation import _grid, _pava, s_sign
from evaluate import seed_mean_dist, lpd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
CONDITIONS = [("adaption:gemma-4-26b-a4b-it", "norm100"), ("adaption:qwen3.5-397b", "norm0")]
SLICES = [-4.0, 0.0, 4.0]
B, S, SEEDS = 25, 8, [4101, 4102, 4103, 4104, 4105]
CALLS_PER_ANCHOR = 30
ANCH, CAL, PRED, POP = R / "pilotD_anchors.json", R / "pilotD_calib.jsonl", R / "pilotD_predictions.json", R / "pilotD_pop.jsonl"


def iso_kernel(df, direction, rng=None):
    c = df.groupby(["delta", "mean"]).lead.agg(["sum", "count"])
    rate = c["sum"] / c["count"]

    def fit(rt):
        out = rt.copy()
        for m in sorted(set(c.index.get_level_values("mean"))):
            idx = sorted([i for i in c.index if i[1] == m], key=lambda i: i[0])
            out.loc[idx] = _pava(rt.loc[idx].values * direction, c["count"].loc[idx].values.astype(float)) * direction
        return out
    p = fit(rate)
    if rng is not None:
        n = c["count"].values
        lam = 1 / (2 * (n + 1))
        pc = np.clip(p.values, lam, 1 - lam)
        p = fit(pd.Series(rng.binomial(n, pc) / n, index=c.index))
    return _grid(p)


def disp(x):
    return float(f"{x:+.1f}")


def new_rows(model, cond, snapshot=None):
    rows = snapshot if snapshot is not None else [json.loads(l) for l in open(CAL)]
    rows = [r for r in rows if r["model"] == model and r["cond"] == cond]
    df = pd.DataFrame(rows)
    df["choice"] = df.response.map(parse_choice)
    df = df[df.choice.notna()].copy()
    df["lead"] = (df.choice == "LEAD").astype(float)
    df["delta"] = (df.h_me.map(disp) - df.h_other.map(disp)).round(2)
    df["mean"] = ((df.h_me.map(disp) + df.h_other.map(disp)) / 2).round(2)
    return df[["delta", "mean", "lead"]]


def old_rows(model, cond):
    d = load_calib()
    return d[(d.model == model) & (d.cond == cond)][["delta", "mean", "lead"]]


def sim(k, seed):
    return order_params(run(8, 0.5, 0.1, 40, KernelDecider(k), seed=seed), 20)


def cmd_anchors(a):
    if ANCH.exists():
        sys.exit("anchors frozen")
    out = {}
    for m, c in CONDITIONS:
        d = old_rows(m, c); sg = s_sign(d)
        cnt = {}
        for kern in (iso_kernel(d, sg), kernel_flexible(d)):
            for s in range(24):
                rec = run(8, 0.5, 0.1, 40, KernelDecider(kern), seed=20000 + s)
                H = [np.zeros(8)] + [np.array(x["h"]) for x in rec["history"]]
                for t, x in enumerate(rec["history"]):
                    h = H[t]
                    for i, j, *_ in x["events"]:
                        for p_, q_ in ((i, j), (j, i)):
                            hp, hq = round(h[p_], 1), round(h[q_], 1)
                            dl = round(round((hp - hq) / 0.2) * 0.2, 1)
                            ms = min(SLICES, key=lambda z: abs(z - (hp + hq) / 2))
                            if 0 < dl <= 16:
                                cnt[(dl, ms)] = cnt.get((dl, ms), 0) + 1
        top = sorted(cnt, key=lambda k: -cnt[k])[:10]
        out[f"{m}|{c}"] = dict(s_sign=sg, anchors=[[dl, ms] for dl, ms in top] + [[-dl, ms] for dl, ms in top],
                               visits={f"{dl}|{ms}": cnt[(dl, ms)] for dl, ms in top})
        print(m, c, out[f"{m}|{c}"]["anchors"], flush=True)
    out["sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    ANCH.write_text(json.dumps(out, indent=1))


def cmd_calib(a):
    A = json.load(open(ANCH))
    rng = random.Random(77)
    have = {}                                                     # resume: count calls already made per anchor
    if CAL.exists():
        for l in open(CAL):
            r = json.loads(l)
            k_ = (r["model"], r["cond"], round(r["h_me"] - r["h_other"], 1), round((r["h_me"] + r["h_other"]) / 2, 1))
            have[k_] = have.get(k_, 0) + 1
    jobs = []
    for m, c in CONDITIONS:
        for dl, ms in A[f"{m}|{c}"]["anchors"]:
            for _ in range(max(0, CALLS_PER_ANCHOR - have.get((m, c, round(dl, 1), round(ms, 1)), 0))):
                me, ot = rng.sample(range(1, 100), 2)
                hm, ho = ms + dl / 2, ms - dl / 2
                jobs.append(dict(model=m, cond=c, h_me=hm, h_other=ho, sys=SYSTEM.format(me=me, n=8),
                                 prompt=pair_prompt(me, ot, hm, ho, rng, CONDS[c])))
    lock = threading.Lock(); f = open(CAL, "a")
    def one(j):
        r = call(j["model"], j["sys"], j["prompt"], tag="pilotD_calib")
        with lock:
            f.write(json.dumps(dict(model=j["model"], cond=j["cond"], h_me=j["h_me"], h_other=j["h_other"],
                                    response=r, choice=parse_choice(r))) + "\n"); f.flush()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs))


def predictive(kfun, seed):
    rng = np.random.default_rng(seed); out = []
    for b in range(B):
        k = kfun(rng)
        for s in range(S):
            out.append(sim(k, int(rng.integers(1e9))))
    return {m: [x[m] for x in out] for m in ("sd_h", "q")}


def cmd_predict(a):
    assert not POP.exists(), "population data exist"
    A = json.load(open(ANCH))
    raw = open(CAL, "rb").read()                                  # one immutable snapshot, hashed as read
    snap = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    need = {(m, c, round(dl, 1), ms): CALLS_PER_ANCHOR for m, c in CONDITIONS for dl, ms in A[f"{m}|{c}"]["anchors"]}
    got = {}
    for r in snap:
        key = (r["model"], r["cond"], round(r["h_me"] - r["h_other"], 1), round((r["h_me"] + r["h_other"]) / 2, 1))
        got[key] = got.get(key, 0) + 1
    missing = {str(k): need[k] - got.get(k, 0) for k in need if got.get(k, 0) < need[k]}
    assert not missing, f"calibration incomplete: {missing}"
    valid = sum(1 for r in snap if r.get("choice"))
    res = dict(conditions=CONDITIONS, B=B, S=S, seeds=SEEDS, protocol=dict(N=8, rho=0.5, mu=0.1, rounds=40, window=20),
               anchors_sha=A["sha256"], calib_sha=hashlib.sha256(raw).hexdigest(), calib_rows=len(snap), calib_valid=valid,
               invalid_policy="unparseable calibration responses are dropped (attempted 30 per anchor)",
               source_sha={f: hashlib.sha256(open(Path(__file__).parent / f, "rb").read()).hexdigest()
                           for f in ("pilotD.py", "population2.py", "prompts_v2.py", "posthoc_attenuation.py", "llm.py")},
               old_grid_sha={}, directions={}, preds={}, operating={})
    for m, c in CONDITIONS:
        old = old_rows(m, c); new = pd.concat([old, new_rows(m, c, snap)], ignore_index=True)
        res["old_grid_sha"][f"{m}|{c}"] = hashlib.sha256(old.to_csv(index=False).encode()).hexdigest()
        sg = s_sign(old); res["directions"][f"{m}|{c}"] = sg
        P = {}
        for name, d in (("old", old), ("new", new)):
            P[f"jeffreys_{name}"] = predictive(lambda rng, d=d: kernel_flexible(d, rng), 1)
            P[f"iso_{name}"] = predictive(lambda rng, d=d: iso_kernel(d, sg, rng), 2)
        res["preds"][f"{m}|{c}"] = P
        k = len(SEEDS)
        cand = seed_mean_dist(P["iso_new"]["sd_h"], B, S, k); ref = seed_mean_dist(P["jeffreys_new"]["sd_h"], B, S, k)
        g = np.random.default_rng(3)
        def succ(x):
            return lpd(x, cand, B * S) > lpd(x, ref, B * S) and abs(x - cand.mean()) <= 0.5
        res["operating"][f"{m}|{c}"] = dict(
            p_success_if_candidate_true=float(np.mean([succ(x) for x in g.choice(cand, 400)])),
            p_success_if_reference_true=float(np.mean([succ(x) for x in g.choice(ref, 400)])),
            cand_mean=float(cand.mean()), ref_mean=float(ref.mean()), n_new_rows=int(len(new) - len(old)),
            note="conditional operating characteristics under the fitted predictive models; not power against misspecification")
        print(m, c, {kk: round(float(np.mean(v["sd_h"])), 2) for kk, v in P.items()}, res["operating"][f"{m}|{c}"], flush=True)
    res["rule"] = ("Primary sd(h) mean of exactly the 5 declared seeds. Success per condition: LPD(iso_new) > LPD(jeffreys_new) and "
                   "|obs - mean(iso_new)| <= 0.5. Secondary: q, old-grid kernels, 90% coverage. Targeted pilot on historically "
                   "selected conditions; isotonic predictive draws are clipped (regularized), the point estimate is unshrunk PAVA.")
    blob = json.dumps(res, sort_keys=True)
    res["sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    with open(PRED, "x") as f:                                    # exclusive creation: refuse to overwrite
        f.write(json.dumps(res))
    print("frozen", res["sha256"])


def load_frozen():
    P = json.load(open(PRED)); sha = P.pop("sha256")
    assert hashlib.sha256(json.dumps(P, sort_keys=True).encode()).hexdigest() == sha, "prediction hash mismatch"
    P["sha256"] = sha
    return P


def cmd_run(a):
    P = load_frozen(); pr = P["protocol"]
    done = {(r["model"], r["cond"], r["seed"]) for r in map(json.loads, open(POP))} if POP.exists() else set()
    cfgs = [(m, c, s) for m, c in P["conditions"] for s in P["seeds"] if (m, c, s) not in done]
    def go(cfg):
        m, c, s = cfg
        rec = run(pr["N"], pr["rho"], pr["mu"], pr["rounds"], LLMDecider(m, CONDS[c]), seed=s,
                  meta=dict(model=m, cond=c, tag="pilotD", pred_sha=P["sha256"]), out=POP)
        print(m, c, s, round(order_params(rec, pr["window"])["sd_h"], 2), flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def cmd_eval(a):
    P = load_frozen(); recs = [json.loads(l) for l in open(POP)]
    out = dict(pred_sha=P["sha256"])
    for m, c in P["conditions"]:
        rr = [r for r in recs if r["model"] == m and r["cond"] == c]; k = len(rr)
        assert sorted(r["seed"] for r in rr) == sorted(P["seeds"]), f"seed set mismatch for {m} {c}"
        assert all(r.get("pred_sha") == P["sha256"] and r["rounds"] == P["protocol"]["rounds"] for r in rr), "protocol/hash mismatch"
        obs = pd.DataFrame([order_params(r, 20) for r in rr]).mean()
        row = dict(k=k, obs_sd=float(obs.sd_h), obs_q=float(obs.q))
        for name, v in P["preds"][f"{m}|{c}"].items():
            for met in ("sd_h", "q"):
                dd = seed_mean_dist(v[met], B, S, k); lo, hi = np.quantile(dd, [.05, .95]); x = float(obs[met])
                row[f"{name}_{met}"] = dict(pred=float(dd.mean()), lo=float(lo), hi=float(hi), hit=bool(lo <= x <= hi), lpd=lpd(x, dd, B * S))
        row["success"] = bool(row["iso_new_sd_h"]["lpd"] > row["jeffreys_new_sd_h"]["lpd"] and abs(row["obs_sd"] - row["iso_new_sd_h"]["pred"]) <= 0.5)
        out[f"{m}|{c}"] = row
        print(m, c, "obs sd", round(row["obs_sd"], 2), {n: (round(row[f'{n}_sd_h']['pred'], 2), round(row[f'{n}_sd_h']['lpd'], 2), row[f'{n}_sd_h']['hit'])
                                                      for n in P["preds"][f"{m}|{c}"]}, "success", row["success"])
    json.dump(out, open(R / "pilotD_eval.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["anchors", "calib", "predict", "run", "eval"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--parallel", type=int, default=10)
    a = ap.parse_args()
    {"anchors": cmd_anchors, "calib": cmd_calib, "predict": cmd_predict, "run": cmd_run, "eval": cmd_eval}[a.cmd](a)
