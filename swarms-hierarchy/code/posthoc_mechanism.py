"""POST-HOC mechanism checks for the attenuation explanation (no LLM calls; responds to round-1 review).

(a) Matched decomposition on identical data, point estimates only (no posterior draws), S simulated seeds each:
      Jeffreys vs raw vs isotonic on the grid rows; NW alpha=1 vs alpha=0 on identical grid + state rows.
    Reports, over the 27 status-sensitive common-design cells: positive-residual count and mean sd(h) residual.
(b) Known-kernel simulation: treat a fitted kernel as the truth, regenerate calibration data with the real per-cell call
    counts, refit with each estimator, and compare predicted with true population sd(h). Isolates estimator bias.
(c) Artificial-lapse sweep: l' = lam + (1 - 2 lam) l on a deference and an anti-deference kernel, sd(h) vs lam.
Writes results/posthoc_mechanism.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from population2 import kernel_nw, KernelDecider, run, order_params
from predict import load_calib, load_statecal
from posthoc_attenuation import _grid, _pava, s_sign

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def point(df, how, direction=1):
    c = df.groupby(["delta", "mean"]).lead.agg(["sum", "count"])
    if how == "jeffreys":
        rate = (c["sum"] + 0.5) / (c["count"] + 1)
    else:
        rate = c["sum"] / c["count"]
    if how == "iso":
        out = rate.copy()
        for m in sorted(set(c.index.get_level_values("mean"))):
            idx = sorted([i for i in c.index if i[1] == m], key=lambda i: i[0])
            out.loc[idx] = _pava(rate.loc[idx].values * direction, c["count"].loc[idx].values.astype(float)) * direction
        rate = out
    return _grid(rate)


def sd_mean(k, S, seed0=0):
    return float(np.mean([order_params(run(8, 0.5, 0.1, 40, KernelDecider(k), seed=seed0 + s), 20)["sd_h"] for s in range(S)]))


def main(S=24):
    calib = load_calib(); st = load_statecal()
    cst = pd.concat([calib, st[[c for c in st.columns if c in calib.columns]]], ignore_index=True)
    recs = [json.loads(l) for f in sorted(R.glob("population_v2*.jsonl")) for l in open(f)]
    obs = {}
    for r in recs:
        if r["rho"] == 0.5 and r["N"] == 8 and r.get("h0") is None and r["rounds"] == 40:
            obs.setdefault((r["model"], r["cond"]), []).append(order_params(r, 20)["sd_h"])
    rows = []
    for (m, c), v in sorted(obs.items()):
        if "deepseek" in m:
            continue
        d = calib[(calib.model == m) & (calib.cond == c)]
        if len(d) < 100:
            continue
        ds = cst[(cst.model == m) & (cst.cond == c)]
        sg = s_sign(d)
        row = dict(model=m, cond=c, obs=float(np.mean(v)), s_sign=sg,
                   jeffreys=sd_mean(point(d, "jeffreys"), S), raw=sd_mean(point(d, "raw"), S),
                   iso=sd_mean(point(d, "iso", sg), S),
                   nw_a1=sd_mean(kernel_nw(ds, alpha=1.0), S), nw_a0=sd_mean(kernel_nw(ds, alpha=0.0), S))
        rows.append(row); print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
    A = pd.DataFrame(rows)
    dec = {e: dict(pos=int((A.obs > A[e]).sum()), n=len(A), mean_resid=round(float((A.obs - A[e]).mean()), 3))
           for e in ("jeffreys", "raw", "iso", "nw_a1", "nw_a0")}
    print("decomposition", dec, flush=True)

    # (b) known-kernel simulation
    truths = [("adaption:gemma-4-31b-it", "norm100"), ("claude:haiku", "norm100"), ("adaption:glm-5.2", "ended+comp"),
              ("adaption:gemma-4-26b-a4b-it", "norm0"), ("claude:haiku", "norm0"), ("adaption:gemma-4-31b-it", "norm0")]
    rng = np.random.default_rng(11)
    kk = []
    for m, c in truths:
        d = calib[(calib.model == m) & (calib.cond == c)]
        sg = s_sign(d)
        true_k = point(d, "iso", sg)
        cells = d.groupby(["delta", "mean"]).size()
        true_sd = sd_mean(true_k, 48, seed0=5000)
        res = {"jeffreys": [], "raw": [], "iso": []}
        for rep in range(8):
            syn = [dict(delta=dl, mean=mn, lead=float(x)) for (dl, mn), n in cells.items()
                   for x in rng.random(n) < true_k(dl, mn)]
            syn = pd.DataFrame(syn)
            for e in res:
                res[e].append(sd_mean(point(syn, e, sg), 12, seed0=7000 + 100 * rep))
        row = dict(model=m, cond=c, s_sign=sg, true_sd=true_sd,
                   **{f"bias_{e}": float(np.mean(v) - true_sd) for e, v in res.items()},
                   **{f"bias_{e}_sd": float(np.std(v)) for e, v in res.items()})
        kk.append(row); print("known", {k: (round(x, 2) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)

    # (c) artificial lapse sweep
    sweep = []
    for m, c in [("adaption:gemma-4-31b-it", "norm100"), ("claude:haiku", "norm0")]:
        d = calib[(calib.model == m) & (calib.cond == c)]
        base = point(d, "iso", s_sign(d))
        for lam in (0, 0.03, 0.06, 0.1, 0.15):
            k = (lambda b, l: (lambda dl, mn: l + (1 - 2 * l) * b(dl, mn)))(base, lam)
            sweep.append(dict(model=m, cond=c, lam=lam, sd_h=sd_mean(k, 32, seed0=9000)))
            print("lapse", sweep[-1], flush=True)
    json.dump(dict(note="POST-HOC", decomposition=dec, cells=rows, known_kernel=kk, lapse_sweep=sweep),
              open(R / "posthoc_mechanism.json", "w"), indent=1)


if __name__ == "__main__":
    main()
