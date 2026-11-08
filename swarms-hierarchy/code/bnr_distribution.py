"""BNR signature test: full final-score distribution (pooled over seeds) vs
  (a) the Ben-Naim-Redner master equation dF_k/dt = r(F_{k+1}-F_k) + F_k(F_{k-1}-F_k), integrated to the same time T
      (X2 protocol: one contest per agent per round = one unit of master-equation time);
  (b) the identical discrete simulator with the frozen kernel for that framing (BNR rule q=0 for w=100,
      calibrated X2 kernel for neutral wording).
Statistic: Kolmogorov-Smirnov distance between the pooled empirical CDF and the predicted CDF; Monte Carlo p-value
from simulated pools of the same size. Post-hoc analysis (not pre-registered). Writes results/bnr_distribution.json
and paper/figures/figA3_bnr_distribution.{png,pdf}.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from bnr_x2 import master, simulate
import x2

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def ecdf(vals, K):
    v = np.asarray(vals)
    return np.array([(v <= k).mean() for k in range(K + 1)])


def main(n_sim=400, seed=0):
    rng = np.random.default_rng(seed)
    recs = [json.loads(l) for l in open(R / "x2_pop.jsonl")]
    out = {}
    for cond in ("norm100", "ended"):
        dec = None if cond == "norm100" else x2.make_decider("ended")[0]
        for r in sorted({x["r"] for x in recs}):
            rr = [x for x in recs if x["cond"] == cond and x["r"] == r]
            if not rr:
                continue
            T, N = rr[0]["rounds"], rr[0]["N"]
            obs = np.concatenate([np.array(x["traj"][-1]) for x in rr])
            K = int(max(obs.max(), 2 * T))
            F_me = master(r, T, K=K + 5)[: K + 1]                 # master-equation CDF at time T
            sims = [simulate(N, r, T, q=0.0, seed=int(rng.integers(1e9)), decider=dec)[-1] for _ in range(n_sim * len(rr))]
            sims = np.array(sims)                                    # (n_sim*k, N)
            F_sim = ecdf(sims.ravel(), K)
            Fo = ecdf(obs, K)
            ks_me, ks_sim = float(np.abs(Fo - F_me).max()), float(np.abs(Fo - F_sim).max())
            # Monte Carlo null for pools of len(rr) runs drawn from the simulator
            null_sim, null_me = [], []
            for b in range(n_sim):
                pool = sims[rng.choice(len(sims), len(rr), replace=False)].ravel()
                Fp = ecdf(pool, K)
                null_sim.append(np.abs(Fp - F_sim).max()); null_me.append(np.abs(Fp - F_me).max())
            p_sim = float((np.array(null_sim) >= ks_sim).mean())
            key = f"{cond}/r={r}"
            out[key] = dict(cond=cond, r=r, T=T, N=N, seeds=len(rr), obs=obs.tolist(), F_obs=Fo.tolist(),
                            F_master=F_me.tolist(), F_sim=F_sim.tolist(), ks_master=ks_me, ks_sim=ks_sim,
                            p_vs_sim=p_sim, ks_sim_null95=float(np.quantile(null_sim, .95)),
                            lower_frac_obs=float((obs <= 0.1 * T).mean()), lower_frac_master=float(F_me[int(0.1 * T)]),
                            max_obs=int(obs.max()), front_master=float((1 - r) * T) if r < 1 else 0.0)
            print(f"{key:16s} KS vs master-eq {ks_me:.2f} | KS vs simulator {ks_sim:.2f} (p={p_sim:.2f}, null95 {np.quantile(null_sim,.95):.2f})"
                  f" | F(0.1T) obs {out[key]['lower_frac_obs']:.2f} master {out[key]['lower_frac_master']:.2f}"
                  f" | max score {obs.max()} vs front (1-r)T={out[key]['front_master']:.1f}")
    json.dump(out, open(R / "bnr_distribution.json", "w"))
    figure(out)


def figure(out):
    sys.path.insert(0, str(ROOT / "paper"))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figs import S, INK2, save
    rs = sorted({v["r"] for v in out.values()})
    fig, axs = plt.subplots(1, len(rs), figsize=(6.6, 2.4), sharey=True)
    for ax, r in zip(axs, rs):
        for cond, col, lab in (("norm100", S[7], "w = 100"), ("ended", S[0], "neutral")):
            v = out.get(f"{cond}/r={r}")
            if not v:
                continue
            k = np.arange(len(v["F_obs"]))
            ax.step(k, v["F_obs"], where="post", color=col, lw=1.8, label=f"observed, {lab}")
            ax.plot(k, v["F_sim"], color=col, lw=1, ls=":", label=f"simulator, {lab}")
        v = out[f"norm100/r={r}"]
        ax.plot(np.arange(len(v["F_master"])), v["F_master"], color="black", lw=1, ls="--", label="BNR master equation")
        ax.set_xlim(0, max(10, 1.15 * max(out[f"norm100/r={r}"]["max_obs"], 3)))
        ax.set_title(f"r = {r}", loc="left", fontsize=8.5, weight="bold"); ax.set_xlabel("final score k")
    axs[0].set_ylabel("fraction with score ≤ k")
    axs[-1].legend(fontsize=5.8, frameon=False, loc="lower right")
    fig.tight_layout(); save(fig, "figA3_bnr_distribution")


if __name__ == "__main__":
    main()
