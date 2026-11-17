"""Evaluate R1 kernels and R2 populations against the frozen forecast and the registered hypotheses.

Written and tested on simulated data before any R2 population was run. Reads results/rosen_r2_predictions.json,
results/rosen_kernel.jsonl and results/rosen_r2_pop.jsonl; writes results/rosen_r2_eval.json.

Usage: rosen_eval.py [--B 1000]
"""
import argparse, json, sys
from collections import defaultdict
from math import comb
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rosen
from rosen import TYPES, metrics
from rosen_forecast import ridge_logit, arm_key, ARMS

R = rosen.R
SCORED = ("dol", "build_rate", "benches", "points", "coverage")


def loglik(X, y, b):
    z = np.asarray(X, float) @ b
    return float(np.sum(np.asarray(y, float) * z - np.logaddexp(0, z)))


def h1(rows, B, rng):
    """AIC: logistic in Rosen value per remaining round vs logistic in F; myopia index per m."""
    core = [r for r in rows if r["set"] == "core" and r.get("reasoning", "none") == "none"]
    y = [r["action"] == "build" for r in core]
    Xv = [[1, r["m"] * (1 - r["F"] / r["H"]) - 1] for r in core]
    Xf = [[1, r["F"] / 16] for r in core]
    Xfull = [[1, r["F"] / 16, r["H"] / 16, r["m"]] for r in core]
    out = {}
    for name, X in (("value", Xv), ("F_only", Xf), ("F_H_m", Xfull)):
        b = ridge_logit(X, y, lam=0.0 + 1e-6)
        out[f"aic_{name}"] = 2 * len(b) - 2 * loglik(X, y, b)
        out[f"coef_{name}"] = b.tolist()
    out["delta_aic_F_minus_value"] = out["aic_F_only"] - out["aic_value"]
    out["H1_value_beats_F"] = out["delta_aic_F_minus_value"] > 4
    b = out["coef_F_H_m"]
    out["signs_F_neg_H_pos_m_pos"] = [b[1] < 0, b[2] > 0, b[3] > 0]

    def myopia(rr, m):
        sub = [r for r in rr if r["m"] == m]
        X = [[1, r["F"] / 16, r["H"] / 16] for r in sub]; yy = [r["action"] == "build" for r in sub]
        if not sub or len(set(yy)) < 2:
            return float("nan")
        bb = ridge_logit(X, yy, lam=1e-3)
        if abs(bb[1]) < 1e-9:
            return float("nan")
        return float((-bb[2] / bb[1]) / (1 - 1 / m))         # dF50/dH relative to the rational slope (1 - 1/m)
    by = defaultdict(list)
    for r in core:
        by[r["cell"]].append(r)
    for m in (2.0, 3.0):
        est = myopia(core, m)
        boots = []
        for _ in range(B):
            rr = [x for v in by.values() for x in (v[i] for i in rng.integers(0, len(v), len(v)))]
            boots.append(myopia(rr, m))
        boots = np.array(boots); boots = boots[~np.isnan(boots)]
        out[f"myopia_m{m:g}"] = dict(est=est, ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                                     if len(boots) else None)
    # cell table for the report
    out["cells"] = {c: dict(n=len(v), p_build=float(np.mean([x["action"] == "build" for x in v])))
                    for c, v in sorted(by.items())}
    return out


def perm_greater(x1, x0, B, rng):
    """One-sided permutation p for mean(x1) > mean(x0)."""
    x1, x0 = np.asarray(x1, float), np.asarray(x0, float)
    x1, x0 = x1[~np.isnan(x1)], x0[~np.isnan(x0)]
    if not (len(x1) and len(x0)):
        return None
    d = x1.mean() - x0.mean(); pool = np.r_[x1, x0]
    perm = [(lambda i: pool[i[:len(x1)]].mean() - pool[i[len(x1):]].mean())(rng.permutation(len(pool))) for _ in range(B)]
    return dict(diff=float(d), p=float((1 + np.sum(np.array(perm) >= d - 1e-12)) / (B + 1)), n1=len(x1), n0=len(x0))


def fisher_greater(a1, n1, a0, n0):
    """One-sided Fisher exact p for rate a1/n1 > a0/n0."""
    K, N = a1 + a0, n1 + n0
    tot = comb(N, K)
    return float(sum(comb(n1, k) * comb(n0, K - k) for k in range(a1, min(K, n1) + 1)) / tot) if tot else None


def step_vs_line(F, y):
    F, y = np.asarray(F, float), np.asarray(y, float)
    ok = ~np.isnan(y); F, y = F[ok], y[ok]
    n = len(y)
    def aic(rss, k):
        return n * np.log(max(rss, 1e-12) / n) + 2 * k
    A = np.c_[np.ones(n), F]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    line = aic(float(np.sum((y - A @ beta) ** 2)), 3)                 # slope, intercept, variance
    grid = sorted(set(F)); best = None
    for lo, hi in zip(grid[:-1], grid[1:]):
        c = (lo + hi) / 2
        rss = float(np.sum((y[F < c] - y[F < c].mean()) ** 2) + np.sum((y[F > c] - y[F > c].mean()) ** 2))
        a = aic(rss, 4)                                                # two levels, break, variance
        if best is None or a < best[0]:
            best = (a, c, float(y[F < c].mean()), float(y[F > c].mean()))
    return dict(aic_line=line, aic_step=best[0], delta=line - best[0], break_at=best[1], level_lo_F=best[2],
                level_hi_F=best[3], H3_step_beats_line=bool(line - best[0] > 4))


def holm(ps):
    keys = [k for k, p in ps.items() if p is not None]
    order = sorted(keys, key=lambda k: ps[k]); m = len(order); out = {}; run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); out[k] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    P = json.load(open(R / "rosen_r2_predictions.json"))
    K = [json.loads(l) for l in open(R / "rosen_kernel.jsonl")]
    pops = [json.loads(l) for l in open(R / "rosen_r2_pop.jsonl")] if (R / "rosen_r2_pop.jsonl").exists() else []
    out = dict(pred_sha=P["sha256"], models={})
    for mod in P["models"]:
        row = dict(H1=h1([r for r in K if r["model"] == mod], min(a.B, 500), rng))
        recs = defaultdict(list)
        for r in pops:
            if r["model"] == mod and r.get("pred_sha") == P["sha256"]:
                recs[arm_key(r["arm"], r["m"], r["F"])].append(r)
        M = {k: [metrics(r) for r in v] for k, v in recs.items()}
        g = lambda k, mt: [x[mt] for x in M.get(k, [])]
        # forecast coverage
        cov, inside = {}, []
        for k in P["arms"]:
            if not M.get(k):
                continue
            cov[k] = {}
            for mt in SCORED:
                f = P["forecast"][mod][k].get(mt)
                v = [x for x in g(k, mt) if not np.isnan(x)]
                if f is None or not v:
                    continue
                obs = float(np.mean(v)); ok = f["lo"] - 1e-9 <= obs <= f["hi"] + 1e-9
                cov[k][mt] = dict(obs=obs, **f, inside=ok, n=len(v), rational=P["rational"][k].get(mt))
                inside.append(ok)
        row["forecast_coverage"] = dict(inside=int(sum(inside)), n=len(inside), arms=cov)
        # H2 corner
        team_low = [r for k, v in recs.items() for r in v if k in [arm_key("team", 2.0, F) for F in (1, 3, 5, 7)]]
        n_one = n_b = 0; use = []
        for r in team_low:
            for a_ in map(str, r["ids"]):
                nr = sum(r["stats"][a_]["bench"][j] >= r["F"] for j in TYPES)
                n_b += nr >= 1; n_one += nr == 1
            mm = metrics(r)
            if not np.isnan(mm["use_share"]):
                use.append(mm["use_share"])
        row["H2"] = dict(builders=n_b, one_bench=n_one, share_one=n_one / n_b if n_b else None,
                         median_pop_use_share=float(np.median(use)) if use else None,
                         supported=bool(n_b and n_one / n_b >= 0.8 and use and np.median(use) >= 0.9))
        # H3 jump
        Fx, Dy = [], []
        for F in (1, 3, 5, 7, 9, 11, 13):
            for x in M.get(arm_key("team", 2.0, F), []):
                Fx.append(F); Dy.append(x["dol"])
        row["H3"] = step_vs_line(Fx, Dy) if len(set(Fx)) >= 3 else None
        fc = [(F, P["forecast"][mod][arm_key("team", 2.0, F)]["dol"]["point"]) for F in (1, 3, 5, 7, 9, 11, 13)]
        row["H3_forecast_curve"] = fc
        # H4a, H4b
        row["H4a"] = perm_greater(g(arm_key("team", 2.0, 3), "dol"), g(arm_key("team", 1.0, 3), "dol"), a.B, rng)
        row["H4a_F7"] = perm_greater(g(arm_key("team", 2.0, 7), "dol"), g(arm_key("team", 1.0, 7), "dol"), a.B, rng)
        def multi(k):
            n = c = 0
            for r in recs.get(k, []):
                for a_ in map(str, r["ids"]):
                    n += 1; c += sum(r["stats"][a_]["bench"][j] >= r["F"] for j in TYPES) >= 2
            return c, n
        (c1, n1), (c0, n0) = multi(arm_key("autarky", 2.0, 1)), multi(arm_key("team", 2.0, 1))
        row["H4b"] = dict(autarky_multi=c1, autarky_n=n1, team_multi=c0, team_n=n0,
                          p=fisher_greater(c1, n1, c0, n0) if n1 and n0 else None)
        row["H4b_descriptive_m3"] = {k: dict(zip(("multi", "n"), multi(k)), rational=P["rational"][k]["multi_ready"])
                                     for k in (arm_key("autarky", 3.0, 1), arm_key("autarky", 3.0, 3))}
        # H5
        row["H5_cov"] = perm_greater(g(arm_key("team", 2.0, 3), "coverage"), g(arm_key("hidden", 2.0, 3), "coverage"), a.B, rng)
        row["H5_dup"] = perm_greater([-x for x in g(arm_key("hidden", 2.0, 3), "dup")],
                                     [-x for x in g(arm_key("team", 2.0, 3), "dup")], a.B, rng)
        if row["H5_dup"]:
            row["H5_dup"]["note"] = "diff = mean dup(team) - mean dup(hidden), sign flipped: positive means team duplicates less"
        ps = dict(H3=None, H4a=(row["H4a"] or {}).get("p"), H4b=row["H4b"]["p"], H5_cov=(row["H5_cov"] or {}).get("p"),
                  H5_dup=(row["H5_dup"] or {}).get("p"))
        row["holm"] = holm(ps)
        # descriptive: who takes which type (first-round position of agents that start each type)
        pos = defaultdict(list)
        for k, v in recs.items():
            if not k.startswith("team|m=2"):
                continue
            for r in v:
                seen = set()
                for h in r["history"]:
                    for i, a_ in enumerate(h["order"], 1):
                        act = h["acts"].get(str(a_))
                        if act and act[0] == "build" and a_ not in seen:
                            seen.add(a_); pos[act[1]].append(i)
        row["who_takes_what"] = {j: dict(n=len(v), mean_turn_position=float(np.mean(v)) if v else None) for j, v in pos.items()}
        row["arm_means"] = {k: {mt: float(np.nanmean([x[mt] for x in v])) if v and not all(np.isnan(x[mt]) for x in v)
                                else None for mt in P["metrics"]} | {"n": len(v)} for k, v in M.items()}
        out["models"][mod] = row
        print(mod, "| forecast inside", row["forecast_coverage"]["inside"], "of", row["forecast_coverage"]["n"],
              "| H1 dAIC", round(row["H1"]["delta_aic_F_minus_value"], 1), "| H2", row["H2"]["share_one"],
              "| H3", row["H3"] and round(row["H3"]["delta"], 1), row["H3"] and row["H3"]["break_at"],
              "| H4a", row["H4a"], "| H4b", row["H4b"]["p"], "| H5", ps["H5_cov"], ps["H5_dup"])
    json.dump(out, open(R / "rosen_r2_eval.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
