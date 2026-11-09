"""POST-HOC (no LLM calls): remaining items of the attenuation critique (refine-logs/attenuation/critique.md).

A-ext  unshrunk estimators on the other regimes: threshold sweep (Gemma-4-26B ended+comp, N = 8/16), mixed-model X1
       populations, and the N = 700 run. Reduced B x S (stated in the output) because these are diagnostics.
B1     known-kernel dose-response: bias of predicted vs true sd(h) at 0.5x, 1x and 4x the real per-cell call counts.
B2     lapse decomposition: fit P_win(D) = lam + (1 - 2 lam) sigma(eta D) to Jeffreys-smoothed and raw cell win
       probabilities (eq. 1, per mean slice), per model x framing; report lam_J - lam_raw vs 1/(2(n+1)).
B3     coverage check for the four largest residual cells: population LEAD rates vs grid rates at matched displayed
       states (|D_pop - D_grid| <= 0.15, |mean| <= 5), Wilson intervals; share of decisions with |mean| > 4.
Writes results/posthoc_extra.json.
"""
import json, sys, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from population2 import kernel_nw, kernel_flexible, KernelDecider, run, order_params
from predict import load_calib, load_statecal
from posthoc_attenuation import s_sign
from posthoc_mechanism import point, sd_mean
from evaluate import seed_mean_dist, displayed_states

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
calib = load_calib(); st = load_statecal()
cst = pd.concat([calib, st[[c for c in st.columns if c in calib.columns]]], ignore_index=True)
grid = lambda m, c: calib[(calib.model == m) & (calib.cond == c)]
allr = lambda m, c: cst[(cst.model == m) & (cst.cond == c)]
out = {}


def interval(samples, B, S, k):
    d = seed_mean_dist(samples, B, S, k); lo, hi = np.quantile(d, [.05, .95])
    return dict(pred=float(d.mean()), lo=float(lo), hi=float(hi))      # full precision; round only for display


def draws(kfun, B, S, simf, seed):
    rng = np.random.default_rng(seed); v = []
    for b in range(B):
        k = kfun(rng)
        for s in range(S):
            v.append(simf(k, int(rng.integers(1e9))))
    return v


def iso_draw(d, sg):
    from posthoc_attenuation import kernel_flex_iso
    return lambda rng: kernel_flex_iso(d, rng, sg)


# ---------------- A-ext: threshold sweep
TH = json.load(open(R / "threshold_predictions.json"))
recs = [json.loads(l) for l in open(R / "threshold_pop.jsonl")]
m, c = TH["model"], TH["cond"]
g, a_ = grid(m, c), allr(m, c); sg = s_sign(g)
B, S = 10, 6
rows = []
for N in (8, 16):
    for rho in TH["rhos"]:
        rr = [r for r in recs if r["N"] == N and r["rho"] == rho]
        if not rr:
            continue
        k = len(rr); qo = float(np.mean([order_params(r, TH["window"])["q"] for r in rr]))
        simf = lambda kern, seed: order_params(run(N, rho, TH["mu"], TH["rounds"], KernelDecider(kern), seed=seed), TH["window"])["q"]
        fz = [p for p in TH["preds"] if p["kernel"] == "nw" and p["N"] == N and p["rho"] == rho][0]
        row = dict(N=N, rho=rho, obs_q=round(qo, 3), nw_frozen=interval(fz["q"], fz["B"], fz["S"], k))
        row["nw_noshrink"] = interval(draws(lambda rng: kernel_nw(a_, rng, alpha=0.0), B, S, simf, 1), B, S, k)
        row["flex_iso"] = interval(draws(iso_draw(g, sg), B, S, simf, 2), B, S, k)
        for kk in ("nw_frozen", "nw_noshrink", "flex_iso"):
            row[kk]["hit"] = row[kk]["lo"] <= qo <= row[kk]["hi"]
        rows.append(row); print("threshold", N, rho, qo, {kk: row[kk]["pred"] for kk in ("nw_frozen", "nw_noshrink", "flex_iso")}, flush=True)
out["threshold"] = dict(B=B, S=S, rows=rows,
                        summary={kk: dict(hits=sum(r[kk]["hit"] for r in rows), n=len(rows),
                                          mean_resid=round(float(np.mean([r["obs_q"] - r[kk]["pred"] for r in rows])), 3))
                                 for kk in ("nw_frozen", "nw_noshrink", "flex_iso")})
print("threshold summary", out["threshold"]["summary"], flush=True)

# ---------------- A-ext: X1 mixed populations
import x1
X1P = json.load(open(R / "x1_predictions.json"))
xrecs = [json.loads(l) for l in open(R / "x1_pop.jsonl")]
xrows = []
for arm, (mina, maja, nmin, cond) in x1.ARMS.items():
    rr = [r for r in xrecs if r.get("arm") == arm]
    if not rr:
        continue
    obs = [x1.metrics(r, x1.types_for(arm, r["seed"]), mina) for r in rr]
    o = dict(rank1_minority=float(np.mean([z["rank1_minority"] for z in obs])), type_gap=float(np.mean([z["type_gap"] for z in obs])),
             sd_h=float(np.mean([z["sd_h"] for z in obs])))
    res = dict(arm=arm, k=len(rr), obs=o)
    for name, kf in (("nw_noshrink", lambda d, rng: kernel_nw(d, rng, alpha=0.0)), ("nw_frozen_method", lambda d, rng: kernel_nw(d, rng, alpha=1.0))):
        rng = np.random.default_rng(5); samp = []
        dA, dB = x1.kernel_rows(mina, cond), x1.kernel_rows(maja, cond)
        for b in range(B):
            ka, kb = kf(dA, rng), kf(dB, rng)
            for s in range(S):
                seed = int(rng.integers(1e9)); types = x1.types_for(arm, seed % 10000)
                rec = run(8, 0.5, 0.1, 40, x1.MixedKernelDecider([ka if t == mina else kb for t in types]), seed=seed)
                samp.append(x1.metrics(rec, types, mina))
        res[name] = {kk: interval([z[kk] for z in samp], B, S, len(rr)) for kk in ("rank1_minority", "type_gap", "sd_h")}
    xrows.append(res); print("x1", arm, o, {n: {kk: res[n][kk]["pred"] for kk in res[n]} for n in ("nw_noshrink", "nw_frozen_method")}, flush=True)
out["x1"] = dict(B=B, S=S, rows=xrows)

# ---------------- A-ext: N = 700
import pop700
PP = json.load(open(R / "pop700_predictions.json")); prec = [json.loads(l) for l in open(R / "pop700_pop.jsonl")][0]
po = pop700.metrics(prec)
m7, c7 = PP["model"], prec["cond"]
res7 = dict(obs=dict(sd_h=po["sd_h"], q=po["q"]))
for name, kf, d in (("flex_iso", None, grid(m7, c7)), ("nw_noshrink", None, allr(m7, c7))):
    rng = np.random.default_rng(7); samp = []
    for b in range(3):
        k = kernel_nw(d, rng, alpha=0.0) if name == "nw_noshrink" else iso_draw(d, s_sign(d))(rng)
        for s in range(2):
            samp.append(pop700.metrics(pop700.simulate(KernelDecider(k), int(rng.integers(1e9)))))
    res7[name] = {kk: dict(mean=round(float(np.mean([z[kk] for z in samp])), 3), lo=round(float(np.min([z[kk] for z in samp])), 3),
                           hi=round(float(np.max([z[kk] for z in samp])), 3)) for kk in ("sd_h", "q")}
    print("N700", name, res7[name], "obs", res7["obs"], flush=True)
out["n700"] = dict(B=3, S=2, note="range over 6 runs", **res7)

# ---------------- B1: dose-response in calls per cell
dose = []
rng = np.random.default_rng(21)
for m, c in [("adaption:gemma-4-31b-it", "norm100"), ("claude:haiku", "norm100")]:
    d = grid(m, c); sg = s_sign(d); true_k = point(d, "iso", sg); cells = d.groupby(["delta", "mean"]).size()
    true_sd = sd_mean(true_k, 48, seed0=5000)
    for mult in (0.5, 1, 4):
        res = {"jeffreys": [], "raw": []}
        for rep in range(6):
            syn = pd.DataFrame([dict(delta=dl, mean=mn, lead=float(x)) for (dl, mn), n in cells.items()
                                for x in rng.random(max(1, int(round(n * mult)))) < true_k(dl, mn)])
            for e in res:
                res[e].append(sd_mean(point(syn, e, sg), 12, seed0=7000 + 100 * rep))
        dose.append(dict(model=m, cond=c, mult=mult, median_n=float(np.median(cells) * mult), true_sd=true_sd,
                         **{f"bias_{e}": round(float(np.mean(v) - true_sd), 3) for e, v in res.items()},
                         **{f"sd_{e}": round(float(np.std(v)), 3) for e, v in res.items()}))
        print("dose", dose[-1], flush=True)
out["dose_response"] = dose

# ---------------- B2: lapse decomposition
def pwin_cells(d, eps):
    c = d.groupby(["delta", "mean"]).lead.agg(["sum", "count"])
    r = (c["sum"] + eps) / (c["count"] + 2 * eps)
    pts = []
    for (dl, mn), v in r.items():
        if dl > 0 and (-dl, mn) in r.index:
            pts.append((dl, 0.5 + 0.5 * (v - r[(-dl, mn)]), c.loc[(dl, mn), "count"]))
    return pts


def fit_lapse(pts):
    D = np.array([p[0] for p in pts]); Y = np.array([p[1] for p in pts])
    def loss(w):
        lam = 1 / (1 + np.exp(-w[0])) * 0.5; eta = w[1]
        return np.sum((lam + (1 - 2 * lam) / (1 + np.exp(-eta * D)) - Y) ** 2)
    best = min((minimize(loss, [x0, e0], method="Nelder-Mead") for x0 in (-3, 0) for e0 in (-2, 2)), key=lambda r: r.fun)
    return float(1 / (1 + np.exp(-best.x[0])) * 0.5), float(best.x[1])


lap = []
for (m, c), d in calib.groupby(["model", "cond"]):
    if "deepseek" in m or len(d) < 100:
        continue
    lj, ej = fit_lapse(pwin_cells(d, 0.5)); lr, er = fit_lapse(pwin_cells(d, 0.0))
    n = float(d.groupby(["delta", "mean"]).size().median())
    lap.append(dict(model=m, cond=c, lam_jeffreys=round(lj, 3), lam_raw=round(lr, 3), diff=round(lj - lr, 3),
                    expected=round(1 / (2 * (n + 1)), 3), eta_jeffreys=round(ej, 2), eta_raw=round(er, 2)))
L = pd.DataFrame(lap)
out["lapse"] = dict(rows=lap, median_diff=float(L["diff"].median()), median_expected=float(L.expected.median()))
print("lapse median diff", out["lapse"]["median_diff"], "expected", out["lapse"]["median_expected"], flush=True)

# ---------------- B3: coverage residual in four cells
def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; den = 1 + z * z / n; c0 = p + z * z / (2 * n); hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c0 - hw) / den, 3), round((c0 + hw) / den, 3))


precs = [json.loads(l) for f in sorted(R.glob("population_v2*.jsonl")) for l in open(f)]
cov = []
for m, c in [("adaption:gemma-4-26b-a4b-it", "norm0"), ("adaption:gemma-4-26b-a4b-it", "norm100"),
             ("adaption:qwen3.5-397b", "norm0"), ("claude:haiku", "ended")]:
    rr = [r for r in precs if r["model"] == m and r["cond"] == c and r["rho"] == 0.5 and r["N"] == 8 and r.get("h0") is None]
    dec = []
    for r in rr:
        for disp, x in zip(displayed_states(r), r["history"]):
            for i, j, ci, cj, w in x["events"]:
                for a_, b_, ch in ((i, j, ci), (j, i, cj)):
                    dec.append((round(disp[a_] - disp[b_], 1), (disp[a_] + disp[b_]) / 2, ch == "LEAD"))
    D = pd.DataFrame(dec, columns=["d", "m", "lead"])
    g = grid(m, c); cells = []
    for (dl, mn), gg in g.groupby(["delta", "mean"]):
        sel = D[(np.abs(D.d - dl) <= 0.15) & (np.abs(D.m - mn) <= 2) & (np.abs(D.m) <= 5)]
        if len(sel) >= 10:
            cells.append(dict(delta=dl, mean=mn, grid_k=int(gg.lead.sum()), grid_n=len(gg), grid_ci=wilson(int(gg.lead.sum()), len(gg)),
                              pop_k=int(sel.lead.sum()), pop_n=len(sel), pop_ci=wilson(int(sel.lead.sum()), len(sel))))
    cov.append(dict(model=m, cond=c, decisions=len(D), share_outside_hull=round(float((D.m.abs() > 4).mean()), 3), cells=cells))
    print("coverage", m, c, len(D), cov[-1]["share_outside_hull"], len(cells), flush=True)
out["coverage"] = cov
json.dump(out, open(R / "posthoc_extra.json", "w"), indent=1, default=float)
print("done")
