"""Score LLM populations against frozen predictions (pre-declared in refine-logs/EXPERIMENT_PLAN.md).

Primary metrics: sd_h and q, reported per model (Haiku is the pre-declared primary comparison).
Per kernel and config with k observed seeds:
  predictive distribution of the k-seed mean = draw one calibration replicate b (shared across seeds),
  then k population outcomes from that replicate's S simulations; KDE log predictive density, 90% interval
  coverage, standardised error.
Diagnostic: conditional winner log-score on observed contests, replaying exact displayed round-start scores.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
sys.path.insert(0, str(Path(__file__).parent))
from population2 import order_params, METRICS, KERNELS
from predict import load_calib, calib_sha, rows_sha, load_statecal

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ["sd_h", "q"]
PROTO = ("mu", "rounds")


def seed_mean_dist(samples, B, S, k, n=4000, seed=0):
    """Predictive distribution of the mean of k NEW runs: variance tau^2 + sigma^2/k in expectation, where
    tau^2 = between-calibration-draw variance (block means shrunk by their sampling noise sigma^2/S) and
    sigma^2 = within-draw run variance (k distinct runs, finite-population corrected)."""
    rng = np.random.default_rng(seed)
    A = np.asarray(samples, float).reshape(B, S)
    mu_b = A.mean(1); mu = mu_b.mean()
    sig2 = A.var(axis=1, ddof=1).mean()
    var_mu = mu_b.var(ddof=1) if B > 1 else 0.0
    tau2 = max(var_mu - sig2 / S, 0.0)
    c = np.sqrt(tau2 / var_mu) if var_mu > 0 else 0.0
    b = rng.integers(0, B, n)
    s = np.argsort(rng.random((n, S)), axis=1)[:, :k]
    dev = (A[b[:, None], s].mean(1) - mu_b[b]) * np.sqrt(S / (S - k)) if S > k else 0.0
    return mu + c * (mu_b[b] - mu) + dev


def lpd(x, dist):
    dist = np.asarray(dist)
    if dist.std() < 1e-9:
        return 0.0 if abs(x - dist.mean()) < 1e-6 else -50.0      # convention for degenerate predictions
    return float(np.log(max(gaussian_kde(dist)(x)[0], 1e-22)))


def displayed_states(rec):
    """Exact displayed (0.1-rounded) round-start scores for every round, replayed at full precision."""
    N, Q, F, mu = rec["N"], rec["Q"], rec["F"], rec["mu"]
    h = np.zeros(N) if rec.get("h0") is None else np.array(rec["h0"], float)
    out = []
    for x in rec["history"]:
        out.append(np.array([round(v, 1) for v in h]))          # identical operation to the deciders (np.float64)
        dh = np.zeros(N)
        for i, j, ci, cj, w in x["events"]:
            l = j if w == i else i
            dh[w] += Q; dh[l] -= F
        h = (h + dh) * (1 - mu)
    return out


def conditional_winner_logscore(recs, kernels):
    tot = {k: 0.0 for k in kernels}; n = 0
    for r in recs:
        for disp, x in zip(displayed_states(r), r["history"]):
            for i, j, ci, cj, w in x["events"]:
                hi, hj = disp[i], disp[j]
                m = (hi + hj) / 2
                n += 1
                for name, f in kernels.items():
                    pi, pj = f(hi - hj, m), f(hj - hi, m)
                    pw = pi * (1 - pj) + 0.5 * (pi * pj + (1 - pi) * (1 - pj))
                    tot[name] += np.log(max(pw if w == i else 1 - pw, 1e-9))
    return {k: v / max(n, 1) for k, v in tot.items()}, n


def main():
    files = sorted((ROOT / "results").glob("predictions_v2*.json"))
    recs = [json.loads(l) for f in sorted((ROOT / "results").glob("population_v2*.jsonl")) for l in open(f)]
    calib_sha_now = calib_sha()
    calib = load_calib()
    st = load_statecal()
    calib_st = pd.concat([calib, st[[c for c in st.columns if c in calib.columns or c == "design"]]],
                         ignore_index=True) if len(st) else calib
    rows, diag, seen = [], [], set()
    for fpath in files:
        P = json.load(open(fpath))
        for e in P["predictions"]:
            key = (e["model"], e["cond"], e["rho"], e["N"])
            dupkey = key + (tuple(e.get("kernels", [])), fpath.name)
            if dupkey in seen:
                raise ValueError(f"duplicate prediction config {key} in {fpath.name}")
            seen.add(dupkey)
            rr = [r for r in recs if (r["model"], r["cond"], r["rho"], r["N"]) == key
                  and all(abs(r[p] - P[p]) < 1e-9 for p in PROTO)]
            seeds = [r["seed"] for r in rr]
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"duplicate seeds for {key}")
            if not rr:
                continue
            obs = pd.DataFrame([order_params(r, P["window"]) for r in rr])
            k = len(rr)
            kerns = e.get("kernels", ["flexible", "bonabeau", "logistic3", "null"])
            for kern in kerns:
                for m in METRICS:
                    dist = seed_mean_dist(e[kern][m]["samples"], P["B"], P["S"], k)
                    x = obs[m].mean()
                    lo, hi = np.quantile(dist, [.05, .95])
                    rows.append(dict(file=fpath.name, design="state" if "nw" in kerns else "grid", model=e["model"], cond=e["cond"], rho=e["rho"], N=e["N"],
                                     kernel=kern, metric=m, seeds=k, obs_mean=round(x, 3),
                                     obs_sd=round(obs[m].std(ddof=1), 3) if k > 1 else None,
                                     pred_mean=round(dist.mean(), 3), pred90=f"[{lo:.2f}, {hi:.2f}]",
                                     covered=bool(lo <= x <= hi),
                                     z=round((x - dist.mean()) / max(dist.std(), 1e-9), 2), lpd=round(lpd(x, dist), 2)))
            src = calib_st if "nw" in kerns else calib
            d = src[(src.model == e["model"]) & (src.cond == e["cond"])]
            ks = {kern: KERNELS[kern](d) for kern in kerns}
            cw, n = conditional_winner_logscore(rr, ks)
            diag.append(dict(model=e["model"], cond=e["cond"], rho=e["rho"], design="state" if "nw" in kerns else "grid",
                             contests=n,
                             calib_matches_freeze=(e.get("calib_rows_sha256") == rows_sha(d)),
                             invalid=sum(r["invalid"] for r in rr),
                             **{f"cwls_{kk}": round(v, 4) for kk, v in cw.items()}))
    df = pd.DataFrame(rows)
    prim = df[df.metric.isin(PRIMARY)]
    summary = prim.groupby(["model", "design", "kernel"]).agg(lpd_sum=("lpd", "sum"), coverage=("covered", "mean"),
                                                    mean_abs_z=("z", lambda z: np.abs(z).mean())).round(3)
    out = dict(pred_files={f.name: json.load(open(f))["sha256"] for f in files}, calib_sha256=calib_sha_now,
               table=rows, kernel_summary=summary.reset_index().to_dict("records"), winner_logscore=diag)
    Path(ROOT / "results" / "eval_v2.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 250)
    print(prim.drop(columns=["file"]).to_string(index=False)); print(); print(summary); print()
    print(pd.DataFrame(diag).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
