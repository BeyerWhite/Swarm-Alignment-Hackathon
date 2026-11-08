"""Round-1 review fix: re-evaluate follow-up experiments (hysteresis, X1, X2, BNR-rules) against their frozen
predictions using the SAME statistic on both sides (mean over k observed seeds) and a predictive distribution
that keeps each calibration draw shared across the k seeds of a replicate (evaluate.seed_mean_dist).
Frozen samples are stored b-major (B calibration draws x S simulations each). Writes results/reeval_followups.json.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import seed_mean_dist

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
out = {}


def interval(samples, B, S, k, censor=None):
    v = np.asarray(samples, float)
    if censor is not None:
        v = np.minimum(v, censor)
    d = seed_mean_dist(v, B, S, k)
    return float(d.mean()), float(np.quantile(d, .05)), float(np.quantile(d, .95))


def line(name, obs, pi):
    m, lo, hi = pi
    flag = "inside" if lo - 1e-9 <= obs <= hi + 1e-9 else ("ABOVE" if obs > hi else "BELOW")
    print(f"  {name:34s} obs={obs:7.2f}  pred={m:6.2f} 90%[{lo:6.2f},{hi:6.2f}]  {flag}")
    return dict(obs=obs, pred=m, lo=lo, hi=hi, flag=flag)


# ---------------- hysteresis ----------------
sys.argv = ["x"]
from hysteresis import traj_metrics
P = json.load(open(R / "hyst_predictions.json"))
recs = [json.loads(l) for l in open(R / "hyst_pop.jsonl")]
B, S = 15, 8
print("HYSTERESIS (Gemma-31B; rank-1 duration censored at 40 rounds on both sides)")
out["hyst"] = {}
for p in P["preds"]:
    rr = [r for r in recs if r["cond"] == p["cond"] and r["arm"] == p["arm"]]
    if not rr:
        continue
    o = [traj_metrics(r) for r in rr]; k = len(o)
    key = f'{p["cond"]}/{p["arm"]}/{p["kernel"]}'
    out["hyst"][key] = {"sd_late": line(key + " sd_late", float(np.mean([x["sd_late"] for x in o])),
                                        interval(p["sd_late"], B, S, k))}
    if p["arm"] == "seeded":
        obs = [x["leader_rank1_rounds"] for x in o]
        out["hyst"][key]["rank1"] = line(key + " leader rank-1 rounds", float(np.mean(obs)),
                                         interval(p["leader_rank1_rounds"], B, S, k, censor=40))
        out["hyst"][key]["rank1"]["per_seed"] = obs

# ---------------- X1 ----------------
from x1 import metrics as x1_metrics, ARMS
P = json.load(open(R / "x1_predictions.json"))
recs = [json.loads(l) for l in open(R / "x1_pop.jsonl")]
B, S = 20, 8
print("\nX1 (endpoint 'rank 1' = highest mean score over the last 10 rounds, as frozen)")
out["x1"] = {}
for arm, spec in P["arms"].items():
    rr = [r for r in recs if r["arm"] == arm]
    o = [x1_metrics(r, r["types"], spec["minority"]) for r in rr]; k = len(o)
    n_min = spec["k"]; null_p = n_min / 8
    succ = sum(x["rank1_minority"] for x in o)
    final_round = sum(bool(np.array([t == spec["minority"] for t in r["types"]])[int(np.argmax(r["history"][-1]["h"]))]) for r in rr)
    pv = binomtest(succ, k, null_p, "greater").pvalue
    print(f" {arm}: last-10-avg rank1=minority {succ}/{k} (composition null {null_p:.3f}, p={pv:.4f}); "
          f"final-round rank1=minority {final_round}/{k}")
    out["x1"][arm] = dict(rank1_last10=succ, rank1_final_round=final_round, n=k, null=null_p, p=pv)
    for m in ("rank1_minority", "type_gap", "pb_corr", "sd_h"):
        for kern in ("nw", "flexible", "null"):
            out["x1"][arm][f"{m}/{kern}"] = line(f"{m} [{kern}]", float(np.mean([float(x[m]) for x in o])),
                                                 interval(P["arms"][arm]["preds"][kern][m], B, S, k))

# ---------------- X2 ----------------
P = json.load(open(R / "x2_predictions.json"))
recs = [json.loads(l) for l in open(R / "x2_pop.jsonl")]
B, S = 20, 10
print("\nX2 (L = fraction with k <= 0.1 T at T=30, as frozen; compared with the IDENTICAL discrete simulator)")
out["x2"] = {}
for p in P["preds"]:
    rr = [r for r in recs if r["cond"] == p["cond"] and r["r"] == p["r"]]
    if not rr:
        continue
    key = f'{p["cond"]}/r={p["r"]}/{p["kernel"]}'
    out["x2"][key] = line(key, float(np.mean([r["L"] for r in rr])), interval(p["L"], B, S, len(rr)))

# ---------------- BNR-rules (bnr_exp) ----------------
P = json.load(open(R / "bnr_predictions.json"))
recs = [json.loads(l) for l in open(R / "bnr_pop.jsonl")]
B, S = 20, 5
print("\nBNR-rules (bnr_exp; rate-2 normalisation, deterministic decline)")
out["bnr"] = {}
for p in P["preds"]:
    rr = [r for r in recs if r["cond"] == p["cond"] and r["r"] == p["r"]]
    if not rr:
        continue
    key = f'{p["cond"]}/r={p["r"]}/{p["kernel"]}'
    out["bnr"][key] = line(key, float(np.mean([r["L"] for r in rr])), interval(p["L"], B, S, len(rr)))

json.dump(out, open(R / "reeval_followups.json", "w"), indent=1)
