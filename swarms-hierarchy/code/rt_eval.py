"""Response-threshold study: evaluation of the pre-registered hypotheses (RT_PREREGISTRATION.md, section 5).

  H1 shape        grid + top-up rows: AIC(linear stimulus) - AIC(log stimulus, full) > 4; n with bootstrap 90% CI
  H2 reinforcement tau = kappa + omega > 0: one-sided bootstrap p = share of frozen bootstrap refits with tau <= 0
  H3 K effect     dol_excess(K = 8) > dol_excess(K = 0), stratified by D: permutation of K within D (one-sided)
  H4 forecast     log predictive density (Gaussian approximation of each forecast) of the full kernel against the
                  history-blind and position nulls over the 10 RT3 arms x 5 metrics; 90% coverage k of n
  H5 demand       K = 8 arms: dol_excess against D, step (break between adjacent D) vs line, dAIC > 4
  H6a replacement spec_gain (A specialists among the remaining agents, after minus before) > 0: exact sign-flip test
  H6b             spec_post and waitA_post against the full and history-blind forecasts (descriptive)
  H7 sufficiency  others hidden vs shown (K = 8, D = 0.5): dol_excess, two-sided permutation; forecast interval
  H8 size         RT4 model: Spearman(N, dol_excess) over N = 4, 8 (RT3 arm), 16 > 0, one-sided permutation
  Holm over H2, H3, H6a within each model (alpha 0.05); RT-R (RT_FRAME=rest) adds H1r: n > 0, bootstrap p.

  rt_eval.py [--out results/rt_eval.json]
"""
import argparse, json, sys
from itertools import product
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import rt
import rt_forecast as F
from rt import R

LPD_METRICS = ("dol_excess", "persist", "work_rate", "mean_waiting", "gini_work")
COV_METRICS = ("dol", "dol_excess", "gini_work", "persist", "work_rate", "mean_waiting", "points")
PERMS = 20000


def load_pops(stage, model):
    p = R / f"rt_{stage.lower()}_pop.jsonl"
    if not p.exists():
        return []
    return [r for r in map(json.loads, open(p)) if r["model"] == model]


def pop_metrics(r):
    m = rt.metrics(r)
    if r.get("remove_at"):
        m.update(rt.removal_metrics(r))
    return m


def perm_strat(x, g, s, rng, one_sided=True):
    """Stratified two-group permutation: statistic sum over strata of mean(g=1) - mean(g=0)."""
    x, g, s = map(np.asarray, (x, g, s))

    def stat(gg):
        return sum(x[(s == k) & (gg == 1)].mean() - x[(s == k) & (gg == 0)].mean() for k in np.unique(s))
    obs = stat(g)
    null = []
    for _ in range(PERMS):
        gg = g.copy()
        for k in np.unique(s):
            idx = np.where(s == k)[0]; gg[idx] = rng.permutation(g[idx])
        null.append(stat(gg))
    null = np.array(null)
    p = (np.sum(null >= obs) + 1) / (PERMS + 1) if one_sided else (np.sum(np.abs(null) >= abs(obs)) + 1) / (PERMS + 1)
    return float(obs), float(p)


def signflip(d):
    d = np.asarray(d, float)
    obs = d.mean()
    null = [np.mean(d * np.array(sg)) for sg in product((1, -1), repeat=len(d))]
    return float(obs), float(np.mean(np.array(null) >= obs - 1e-12))


def step_vs_line(D, y):
    D, y = np.asarray(D, float), np.asarray(y, float)
    n = len(y)

    def aic(rss, k):
        return n * np.log(rss / n) + 2 * (k + 1)
    b = np.polyfit(D, y, 1)
    a_line = aic(((y - np.polyval(b, D)) ** 2).sum(), 2)
    lev = sorted(set(D)); best = None
    for cut in [(lev[i] + lev[i + 1]) / 2 for i in range(len(lev) - 1)]:
        lo, hi = y[D < cut], y[D > cut]
        rss = ((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum()
        a = aic(rss, 3)
        if best is None or a < best[0]:
            best = (a, cut, float(lo.mean()), float(hi.mean()))
    return dict(dAIC_line_minus_step=float(a_line - best[0]), step=bool(best[0] < a_line - 4), break_at=best[1],
                levels=best[2:])


def holm(ps):
    keys = sorted(ps, key=lambda k: ps[k])
    out, m, run = {}, len(ps), 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def lpd(fc, obs_by_arm, arms):
    tot, terms = 0.0, 0
    for a in arms:
        for k in LPD_METRICS:
            f = fc.get(a, {}).get(k); o = obs_by_arm.get(a, {}).get(k)
            if f is None or o is None or not np.isfinite(o):
                continue
            sd = max(f["sd"], 1e-3)
            tot += -0.5 * np.log(2 * np.pi * sd * sd) - 0.5 * ((o - f["mean"]) / sd) ** 2; terms += 1
    return float(tot), terms


def evaluate_model(m, pred, rng):
    out = {}
    rows = F.load_rows(m)
    grid = [r for r in rows if r["set"] == "grid"]
    full_g, lin_g = F.fit(grid, "full"), F.fit(grid, "linear")
    nb = [p["theta"][F.NAMES.index("n")] for p in pred["params_boot"][m]["full"]]
    out["H1"] = dict(aic_linear_minus_log=lin_g["aic"] - full_g["aic"], log_better=bool(lin_g["aic"] - full_g["aic"] > 4),
                     n=pred["fits"][m]["full"]["theta"][F.NAMES.index("n")],
                     n_ci90=[float(np.percentile(nb, 5)), float(np.percentile(nb, 95))])
    boot = pred["params_boot"][m]["full"]
    par = {k: [p["theta"][i] for p in boot] for i, k in enumerate(F.NAMES)}
    tau = np.array(par["kappa"]) + np.array(par["omega"])
    pt = dict(zip(F.NAMES, pred["fits"][m]["full"]["theta"]))
    ci = lambda v: [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
    out["H2"] = dict(tau=pt["kappa"] + pt["omega"], tau_ci90=ci(tau), p=float((np.sum(tau <= 0) + 1) / (len(tau) + 1)),
                     kappa=pt["kappa"], kappa_ci90=ci(par["kappa"]), omega=pt["omega"], omega_ci90=ci(par["omega"]),
                     rho_ci90=ci(par["rho"]),
                     sig_ci90=[float(np.percentile(par["sig"], 5)), float(np.percentile(par["sig"], 95))],
                     b1_ci90=[float(np.percentile(par["b1"], 5)), float(np.percentile(par["b1"], 95))])
    pops = load_pops("RT3", m)
    met = [(r["arm"], pop_metrics(r)) for r in pops]
    by_arm = {}
    for a, x in met:
        by_arm.setdefault(a, []).append(x)
    obs = {a: {k: float(np.nanmean([x[k] for x in v])) for k in v[0]} for a, v in by_arm.items()}
    out["n_pops"] = {a: len(v) for a, v in by_arm.items()}
    main = [(a, x) for a, x in met if "others=shown|T=24|N=8|rm=None" in a]
    if main:
        K = [int(a.split("|")[0][2:]) for a, _ in main]; D = [a.split("|")[1] for a, _ in main]
        y = [x["dol_excess"] for _, x in main]
        d, p = perm_strat(y, [int(k == 8) for k in K], D, rng)
        out["H3"] = dict(diff_sum_over_D=d, diff_mean=d / len(set(D)), p=p,
                         by_D={dd: dict(K8=float(np.mean([v for v, k, e in zip(y, K, D) if k == 8 and e == dd])),
                                        K0=float(np.mean([v for v, k, e in zip(y, K, D) if k == 0 and e == dd])))
                               for dd in sorted(set(D))})
        k8 = [(float(dd[2:]), v) for v, k, dd in zip(y, K, D) if k == 8]
        out["H5"] = step_vs_line([a for a, _ in k8], [v for _, v in k8])
    arms3 = [rt.arm_key(a) for a in rt.arms("RT3")]
    fcs = pred["forecasts"][m]
    out["H4"] = {}
    for v in ("full", "histblind", "position", "flex"):
        l, n = lpd(fcs[v], obs, arms3)
        cov = [(fcs[v][a][k]["lo"] <= obs[a][k] <= fcs[v][a][k]["hi"]) for a in arms3 if a in obs
               for k in COV_METRICS + F.RM_METRICS if k in fcs[v].get(a, {}) and k in obs[a] and np.isfinite(obs[a][k])]
        out["H4"][v] = dict(lpd=l, terms=n, coverage=f"{sum(cov)} of {len(cov)}")
    out["H4"]["full_minus_histblind"] = out["H4"]["full"]["lpd"] - out["H4"]["histblind"]["lpd"]
    out["H4"]["full_minus_position"] = out["H4"]["full"]["lpd"] - out["H4"]["position"]["lpd"]
    rm = [x for a, x in met if "rm=16" in a]
    if rm:
        g, p = signflip([x["spec_gain"] for x in rm])
        ra = rt.arm_key(rt.arms("RT3")[-1])
        out["H6a"] = dict(spec_gain_mean=g, p=p)
        out["H6b"] = {k: dict(obs=obs[ra][k], full=[fcs["full"][ra][k]["lo"], fcs["full"][ra][k]["hi"]],
                              histblind=[fcs["histblind"][ra][k]["lo"], fcs["histblind"][ra][k]["hi"]])
                      for k in ("spec_pre", "spec_post", "spec_gain", "a_share_post", "waitA_pre", "waitA_peak",
                                "waitA_post")}
    hid = [x["dol_excess"] for a, x in met if "others=hidden" in a]
    sh = [x["dol_excess"] for a, x in met if a == rt.arm_key(dict(K=8, D=0.5))]
    if hid and sh:
        d, p = perm_strat(hid + sh, [1] * len(hid) + [0] * len(sh), [0] * (len(hid) + len(sh)), rng, one_sided=False)
        ha = rt.arm_key(dict(K=8, D=0.5, others="hidden"))
        out["H7"] = dict(diff=d, p_two_sided=p, hidden_in_forecast=bool(
            fcs["full"][ha]["dol_excess"]["lo"] <= obs[ha]["dol_excess"] <= fcs["full"][ha]["dol_excess"]["hi"]))
    if pred.get("rt4_model") == m:
        p4 = load_pops("RT4", m)
        xs = [(r["N"], pop_metrics(r)["dol_excess"]) for r in p4] + [(8, v) for v in sh]
        if p4:
            Ns, ys = np.array([a for a, _ in xs]), np.array([b for _, b in xs])
            rho = spearmanr(Ns, ys).correlation
            null = [spearmanr(rng.permutation(Ns), ys).correlation for _ in range(5000)]
            out["H8"] = dict(rho=float(rho), p=float((np.sum(np.array(null) >= rho) + 1) / 5001),
                             by_N={int(n): float(np.mean(ys[Ns == n])) for n in sorted(set(Ns))})
    if rt.FRAME == "rest":     # RT-R: an uptake threshold exists iff work rises with backlog (n > 0), confirmatory
        out["H1r"] = dict(n=out["H1"]["n"], n_ci90=out["H1"]["n_ci90"], p=float((np.sum(np.array(nb) <= 0) + 1) / (len(nb) + 1)))
    ps = {k: out[k]["p"] for k in ("H1r", "H2", "H3", "H6a") if k in out}
    out["holm"] = holm(ps) if ps else {}
    out["observed_by_arm"] = obs
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(R / "rt_eval.json"))
    a = p.parse_args()
    pred = json.load(open(rt.FORECAST))
    rng = np.random.default_rng(20261120)
    res = dict(forecast_sha=pred["sha256"], c=pred["c"], rt4_model=pred.get("rt4_model"), models={})
    for m in pred["models"]:
        res["models"][m] = evaluate_model(m, pred, rng)
    json.dump(res, open(a.out, "w"), indent=1)
    for m, r in res["models"].items():
        print(f"== {m}")
        for k in ("H1", "H1r", "H2", "H3", "H4", "H5", "H6a", "H6b", "H7", "H8", "holm"):
            if k in r:
                print(f"  {k}: {json.dumps(r[k])[:400]}")


if __name__ == "__main__":
    main()
