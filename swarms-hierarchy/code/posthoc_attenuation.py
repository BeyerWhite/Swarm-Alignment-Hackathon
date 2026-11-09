"""POST-HOC (not pre-registered; 2026-11-08 afternoon, after all population runs). Tests whether the systematic
under-prediction of population inequality is kernel attenuation: every pre-registered estimator shrinks sparse,
saturated calibration cells (~5 calls each) toward 1/2 (Jeffreys rates, Beta(k+1/2, n-k+1/2) draws, NW pseudo-count),
which acts as an artificial lapse rate and flattens P_win.

New estimators (frozen prediction files are not touched):
  flex_mle     kernel_flexible with near-flat Beta(k+0.05, n-k+0.05) cell draws
  flex_iso     per mean level, weighted isotonic regression of LEAD rate in delta (direction = sign of calibration s),
               cell-bootstrap uncertainty, then the same bilinear interpolation
  nw_noshrink  kernel_nw with alpha = 0 (grid + state-matched rows, as the frozen NW predictions)
Reference: the frozen flexible (Jeffreys) and null predictions, read from the frozen files.

Evaluated exactly like the paper: B x S simulations per config (25 x 8 common design; 15 x 8 seeded leaders),
seed_mean_dist for the observed number of seeds, 90% intervals, z, composite LPD with the n_eff bandwidth rule.
Writes results/posthoc_attenuation.{json,csv}.
"""
import json, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from population2 import kernel_nw, KernelDecider, run, order_params
from predict import load_calib, load_statecal
from evaluate import seed_mean_dist, lpd
from hysteresis import traj_metrics

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
METRICS = ["sd_h", "q"]


def _grid(rate):
    g = rate.unstack("mean")
    g = g.sort_index().interpolate(method="index", axis=0, limit_direction="both")
    g = g.T.sort_index().interpolate(method="index", axis=0, limit_direction="both").T
    D, M, Z = np.array(g.index, float), np.array(g.columns, float), g.values

    def p(delta, mean):
        d = min(max(delta, D[0]), D[-1]); m = min(max(mean, M[0]), M[-1])
        return float(np.interp(m, M, np.array([np.interp(d, D, Z[:, k]) for k in range(len(M))])))
    return p


def kernel_flex_mle(df, rng=None, eps=0.05):
    c = df.groupby(["delta", "mean"]).lead.agg(["sum", "count"])
    if rng is None:
        rate = c["sum"] / c["count"]
    else:
        rate = pd.Series(rng.beta(c["sum"] + eps, c["count"] - c["sum"] + eps), index=c.index)
    return _grid(rate)


def _pava(y, w):
    """Weighted pool-adjacent-violators, non-decreasing."""
    blocks = [[yi, wi, 1] for yi, wi in zip(y, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            a, b = blocks[i], blocks[i + 1]
            wt = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / wt, wt, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[v] * n for v, _, n in blocks])


def kernel_flex_iso(df, rng=None, direction=1):
    c = df.groupby(["delta", "mean"]).lead.agg(["sum", "count"])
    if rng is not None:                                   # cell bootstrap: resample each cell's calls
        s = rng.binomial(c["count"].values, (c["sum"] / c["count"]).values)
        c = c.assign(sum=s)
    rate = c["sum"] / c["count"]
    out = rate.copy()
    for m in sorted(set(c.index.get_level_values("mean"))):
        idx = [i for i in c.index if i[1] == m]
        idx.sort(key=lambda i: i[0])
        y = rate.loc[idx].values * direction
        w = c["count"].loc[idx].values.astype(float)
        fit = _pava(y, w) * direction
        out.loc[idx] = fit
    return _grid(out)


def kernel_nw_noshrink(df, rng=None):
    return kernel_nw(df, rng=rng, alpha=0.0)


def s_sign(df):
    g = df.groupby("delta").lead.mean()
    v = [(g.get(d, np.nan) - g.get(-d, np.nan)) / d for d in (0.4, 1.0) if d in g.index and -d in g.index]
    return 1 if np.nanmean(v) >= 0 else -1


def predictive(kfun, d, B, S, sim, seed):
    rng = np.random.default_rng(seed)
    out = []
    for b in range(B):
        k = kfun(d, rng)
        for s in range(S):
            out.append(sim(k, int(rng.integers(1e9))))
    return out


def main(B=25, S=8):
    calib = load_calib(); st = load_statecal()
    calib_st = pd.concat([calib, st[[c for c in st.columns if c in calib.columns]]], ignore_index=True)
    recs = [json.loads(l) for f in sorted(R.glob("population_v2*.jsonl")) for l in open(f)]
    rows = []
    files = [f for f in sorted(R.glob("predictions_v2_*.json"))]
    for fpath in files:
        P = json.load(open(fpath))
        for e in P["predictions"]:
            if e["rho"] != 0.5 or e["N"] != 8:
                continue
            rr = [r for r in recs if (r["model"], r["cond"], r["rho"], r["N"]) == (e["model"], e["cond"], 0.5, 8)
                  and r.get("h0") is None and abs(r["mu"] - P["mu"]) < 1e-9 and r["rounds"] == P["rounds"]]
            if not rr:
                continue
            k = len(rr)
            obs = pd.DataFrame([order_params(r, P["window"]) for r in rr]).mean()
            d = calib[(calib.model == e["model"]) & (calib.cond == e["cond"])]
            dst = calib_st[(calib_st.model == e["model"]) & (calib_st.cond == e["cond"])]
            sgn = s_sign(d)
            sim = lambda kern, seed: order_params(run(8, 0.5, P["mu"], P["rounds"], KernelDecider(kern), seed=seed), P["window"])
            preds = {"flexible": {m: e["flexible"][m]["samples"] for m in METRICS},
                     "null": {m: e["null"][m]["samples"] for m in METRICS}}
            for name, kf, data in (("flex_mle", kernel_flex_mle, d),
                                   ("flex_iso", lambda df, rng: kernel_flex_iso(df, rng, sgn), d),
                                   ("nw_noshrink", kernel_nw_noshrink, dst)):
                samp = predictive(kf, data, B, S, sim, seed=zlib.crc32(f"{e['model']}|{e['cond']}|{name}".encode()))
                preds[name] = {m: [x[m] for x in samp] for m in METRICS}
            for name, pm in preds.items():
                for m in METRICS:
                    dist = seed_mean_dist(pm[m], B, S, k)
                    lo, hi = np.quantile(dist, [.05, .95]); x = float(obs[m])
                    rows.append(dict(design="common", estimator=name, model=e["model"], cond=e["cond"], metric=m, seeds=k,
                                     obs=round(x, 3), pred=round(float(dist.mean()), 3), lo=round(float(lo), 3),
                                     hi=round(float(hi), 3), z=round((x - dist.mean()) / max(dist.std(), 1e-9), 2),
                                     hit=bool(lo - 1e-9 <= x <= hi + 1e-9), lpd=round(lpd(x, dist, B * S), 2)))
            print(e["model"].split(":")[-1], e["cond"], k, {n: round(float(np.mean(preds[n]["sd_h"])), 2) for n in preds},
                  "obs", round(float(obs["sd_h"]), 2), flush=True)
    # seeded leaders (Gemma-4-31B), same h0 and B=15, S=8 as the frozen hysteresis predictions
    HP = json.load(open(R / "hyst_predictions.json"))
    H = [json.loads(l) for l in open(R / "hyst_pop.jsonl")]
    for cond in ("norm0", "ended", "norm50"):
        rs = [r for r in H if r["cond"] == cond and r.get("h0")]
        obs_held = float(np.mean([traj_metrics(r)["leader_rank1_rounds"] for r in rs]))
        d = calib[(calib.model == HP["model"]) & (calib.cond == cond)]
        dst = calib_st[(calib_st.model == HP["model"]) & (calib_st.cond == cond)]
        sim = lambda kern, seed: traj_metrics(run(8, HP["rho"], HP["mu"], HP["rounds"], KernelDecider(kern), seed=seed, h0=HP["h0"]))["leader_rank1_rounds"]
        preds = {"nw (frozen)": [p for p in HP["preds"] if p["cond"] == cond and p["kernel"] == "nw" and p["arm"] == "seeded"][0]["leader_rank1_rounds"]}
        sgn = s_sign(d)
        for name, kf, data in (("flex_mle", kernel_flex_mle, d), ("flex_iso", lambda df, rng: kernel_flex_iso(df, rng, sgn), d),
                               ("nw_noshrink", kernel_nw_noshrink, dst)):
            preds[name] = predictive(kf, data, 15, 8, sim, seed=zlib.crc32(f"seeded|{cond}|{name}".encode()))
        for name, v in preds.items():
            dist = seed_mean_dist(v, 15, 8, len(rs))
            lo, hi = np.quantile(dist, [.05, .95])
            rows.append(dict(design="seeded_leader", estimator=name, model=HP["model"], cond=cond, metric="rounds_held",
                             seeds=len(rs), obs=round(obs_held, 2), pred=round(float(dist.mean()), 2), lo=round(float(lo), 2),
                             hi=round(float(hi), 2), z=round((obs_held - dist.mean()) / max(dist.std(), 1e-9), 2),
                             hit=bool(lo - 1e-9 <= obs_held <= hi + 1e-9), lpd=None))
        print("seeded", cond, {r["estimator"]: r["pred"] for r in rows if r["design"] == "seeded_leader" and r["cond"] == cond}, "obs", obs_held, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R / "posthoc_attenuation.csv", index=False)
    c = df[(df.design == "common") & ~df.model.str.contains("deepseek")]
    summ = []
    for est, g in c.groupby("estimator"):
        sd = g[g.metric == "sd_h"]
        margin = g.groupby("model").lpd.sum() - c[c.estimator == "null"].groupby("model").lpd.sum()
        summ.append(dict(estimator=est, frac_obs_gt_pred_sd=round(float((sd.obs > sd.pred).mean()), 3),
                         mean_resid_sd=round(float((sd.obs - sd.pred).mean()), 3), mean_z_sd=round(float(sd.z.mean()), 2),
                         coverage_all=f"{int(g.hit.sum())}/{len(g)}",
                         lpd_margin_vs_null_min=round(float(margin.min()), 1), lpd_margin_vs_null_max=round(float(margin.max()), 1)))
    S_ = pd.DataFrame(summ)
    print(S_.to_string(index=False))
    json.dump(dict(note="POST-HOC; not pre-registered", summary=summ, rows=rows), open(R / "posthoc_attenuation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
