"""Physics fits on existing logs (no new LLM calls).

(1) In-population sensitivity: maximum-likelihood fit of P_win = sigma(eta * Delta) to every contest in the LLM
    populations (displayed round-start scores replayed exactly), vs the same fit to the win probabilities implied
    by isolated pairwise calibration (eq. 1). Bootstrap over runs / calibration rows.
(2) Bonabeau threshold: protocol mean-field rho_c from the calibrated eta, vs the rho sweeps.
(3) Upset probability u in BNR-rule populations: P(lower-scored agent wins | unequal scores).
Writes results/physics_fits.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import displayed_states
from predict import load_calib

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def contests(rec):
    """(Delta from the winner-candidate i's view, i_won) for every contest, using displayed round-start scores."""
    out = []
    for disp, x in zip(displayed_states(rec), rec["history"]):
        for i, j, ci, cj, w in x["events"]:
            out.append((disp[i] - disp[j], 1.0 if w == i else 0.0))
    return out


def fit_eta(d, y, wts=None):
    d = np.asarray(d, float); y = np.asarray(y, float); wts = np.ones_like(d) if wts is None else np.asarray(wts, float)
    def nll(eta):
        z = eta * d
        return float(np.sum(wts * (np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))))
    return minimize_scalar(nll, bounds=(-20, 20), method="bounded").x


def calib_pwin_data(c):
    """Implied P_win(Delta) at each displayed Delta>0 from pooled LEAD rates; returns (Delta, P, n) arrays."""
    g = c.groupby("delta").lead.agg(["mean", "count"])
    D = [x for x in g.index if x > 0 and -x in g.index]
    P = [0.5 + 0.5 * (g.loc[x, "mean"] - g.loc[-x, "mean"]) for x in D]
    n = [min(g.loc[x, "count"], g.loc[-x, "count"]) for x in D]
    return np.array(D, float), np.clip(np.array(P), 0, 1), np.array(n, float)


def eta_from_calib(c):
    D, P, n = calib_pwin_data(c)
    # binomial pseudo-likelihood: P successes out of n at each Delta (and mirror)
    return fit_eta(np.r_[D, D], np.r_[np.ones_like(D), np.zeros_like(D)], np.r_[n * P, n * (1 - P)])


def main(B=200, seed=0):
    rng = np.random.default_rng(seed)
    calib = load_calib()
    recs = [json.loads(l) for f in sorted(R.glob("population_v2*.jsonl")) for l in open(f)]
    rows = []
    for (m, cnd), grp in pd.DataFrame([dict(model=r["model"], cond=r["cond"], rho=r["rho"], idx=k)
                                       for k, r in enumerate(recs)]).groupby(["model", "cond"]):
        runs = [recs[k] for k in grp[grp.rho == 0.5].idx]
        if not runs:
            continue
        per_run = [contests(r) for r in runs]
        allc = [c for pr in per_run for c in pr]
        eta_pop = fit_eta([a for a, _ in allc], [b for _, b in allc])
        boots = []
        for _ in range(B):
            pick = [per_run[k] for k in rng.integers(0, len(per_run), len(per_run))]
            cc = [c for pr in pick for c in pr]
            boots.append(fit_eta([a for a, _ in cc], [b for _, b in cc]))
        cal = calib[(calib.model == m) & (calib.cond == cnd)]
        eta_cal = eta_from_calib(cal)
        cb = [eta_from_calib(cal.sample(len(cal), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(B)]
        rows.append(dict(model=m, cond=cnd, n_runs=len(runs), n_contests=len(allc),
                         eta_pop=eta_pop, eta_pop_lo=float(np.percentile(boots, 2.5)), eta_pop_hi=float(np.percentile(boots, 97.5)),
                         eta_cal=eta_cal, eta_cal_lo=float(np.percentile(cb, 2.5)), eta_cal_hi=float(np.percentile(cb, 97.5))))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    out = dict(eta_comparison=rows)

    # (2) rho sweeps vs protocol mean-field threshold rho_c = 2*mu*(N-1) / [eta*(1-mu)*N*(1+F)]  (half of Lacasa-Luque,
    # because every agent both initiates and is targeted; mu=0.1, F=1, N=8)
    from population2 import order_params
    sweep = []
    for r in recs:
        if r["model"] in ("adaption:gemma-4-31b-it", "adaption:deepseek-v4-flash"):
            o = order_params(r)
            sweep.append(dict(model=r["model"], cond=r["cond"], rho=r["rho"], sd_h=o["sd_h"], q=o["q"]))
    sw = pd.DataFrame(sweep).groupby(["model", "cond", "rho"]).agg(sd_h=("sd_h", "mean"), q=("q", "mean"), n=("q", "size")).reset_index()
    thr = []
    for (m, cnd), g in sw.groupby(["model", "cond"]):
        e = t[(t.model == m) & (t.cond == cnd)].eta_cal
        e = float(e.iloc[0]) if len(e) else np.nan
        rc = 2 * 0.1 * 7 / (e * 0.9 * 8 * 2) if e and e > 0 else np.inf
        thr.append(dict(model=m, cond=cnd, eta_cal=e, rho_c_meanfield=rc,
                        sweep={float(k): dict(sd_h=float(v.sd_h), q=float(v.q), n=int(v.n)) for k, v in g.set_index("rho").iterrows()}))
        print(m.split(":")[-1], cnd, f"eta_cal={e:.2f} rho_c={rc:.3f}",
              " ".join(f"rho={k:.2f}:sd={v.sd_h:.2f},q={v.q:.2f}" for k, v in g.set_index("rho").iterrows()))
    out["rho_sweep_threshold"] = thr

    # (3) upset probability in BNR-rule populations
    up = []
    for fname, key in (("x2_pop.jsonl", "x2"), ("bnr_pop.jsonl", "bnr")):
        for r in map(json.loads, open(R / fname)):
            if key == "x2":
                # events: [t, i, j, ci, cj, w]; replay integer scores from traj (scores after decline of round t)
                k = np.zeros(r["N"])
                byt = {}
                for e in r["events"]:
                    byt.setdefault(e[0], []).append(e)
                for t_ in range(r["rounds"]):
                    for (_, i, j, ci, cj, w) in byt.get(t_, []):
                        if k[i] != k[j]:
                            hi = i if k[i] > k[j] else j
                            up.append(dict(src=key, cond=r["cond"], r=r["r"], upset=int(w != hi), decisive=int(ci != cj)))
                    k = np.array(r["traj"][t_], float)
            else:
                H = [np.zeros(r["N"])] + [np.array(h) for h in r["history"]]
                for t_, evs in enumerate(r["events"]):
                    disp = np.round(H[t_], 1)
                    for i, j, ci, cj, w in evs:
                        if disp[i] != disp[j]:
                            hi = i if disp[i] > disp[j] else j
                            up.append(dict(src=key, cond=r["cond"], r=r["r"], upset=int(w != hi), decisive=int(ci != cj)))
    U = pd.DataFrame(up).groupby(["src", "cond"]).agg(n=("upset", "size"), upsets=("upset", "sum")).reset_index()
    U["u"] = U.upsets / U.n
    print(U.to_string(index=False))
    out["upset"] = U.to_dict("records")
    json.dump(out, open(R / "physics_fits.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
