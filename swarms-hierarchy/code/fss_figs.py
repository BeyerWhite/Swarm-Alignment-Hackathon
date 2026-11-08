"""FSS figures: (a) frozen order q vs norm weight w for N=8..700 (proportional forgetting) — curves overlap;
(b) BNR additive forgetting: lower-class fraction L vs r for N, status-sensitive vs status-blind kernels.
Also reports the off-support fraction of decisions in large-N simulations."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from figures import SERIES, MARKERS, INK2, save, NICE
from fss import VecKernel, grid_table
from predict import load_calib
ROOT = Path(__file__).resolve().parents[1]

def fig_fss(models=("adaption:gemma-4-31b-it", "adaption:glm-5.2", "claude:haiku")):
    d = pd.read_json(ROOT / "results" / "fss_w.jsonl", lines=True); d = d[d.kind == "flexible"]
    fig, axs = plt.subplots(1, len(models), figsize=(7.4, 2.6), sharey=True)
    Ns = [8, 32, 128, 700]
    for ax, m in zip(axs, models):
        for k, N in enumerate(Ns):
            g = d[(d.model == m) & (d.N == N)].groupby("w").q.agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], color=SERIES[k], marker=MARKERS[k], ms=4, lw=1.5,
                        capsize=1.5, elinewidth=.8, label=f"N={N}")
        ax.set_title(NICE.get(m, m), fontsize=8); ax.set_xlabel("norm weight w")
    axs[0].set_ylabel("frozen order q"); axs[-1].legend(frameon=False, fontsize=7)
    save(fig, "fig6_fss_proportional")

def bnr_L(k, N, r, rounds=300, seed=0):
    rng = np.random.default_rng(seed); h = np.zeros(N)
    for t in range(rounds):
        i = np.arange(N); j = rng.integers(0, N - 1, N); j = j + (j >= i); dd = np.round(h, 1)
        ci = rng.random(N) < k(dd - dd[j], 0); cj = rng.random(N) < k(dd[j] - dd, 0)
        iw = (ci & ~cj) | (~(ci ^ cj) & (rng.random(N) < .5)); w = np.where(iw, i, j); np.add.at(h, w, 1.0)
        h = np.maximum(h - r, 0)
    return float((h <= 0.05 * rounds).mean())

def fig_bnr():
    rs = np.round(np.arange(0.2, 2.61, 0.2), 2)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for k, (lab, ker) in enumerate([("status-sensitive (η=1)", VecKernel("bonabeau", (0.0, 1.0))),
                                    ("status-blind (η=0)", VecKernel("bonabeau", (0.0, 0.0)))]):
        for n_i, N in enumerate((8, 700)):
            L = [np.mean([bnr_L(ker, N, r, seed=s) for s in range(3)]) for r in rs]
            ax.plot(rs, L, color=SERIES[k], marker=MARKERS[n_i], ms=4, lw=1.5, ls="-" if N == 700 else ":",
                    label=f"{lab}, N={N}")
    ax.plot([0, 2], [0, 1], color=INK2, lw=1, ls="--"); ax.text(0.9, 0.38, "continuum guide r / r_c (r_c = 2);\nF(0.05T) is not the asymptotic lower-class mass", fontsize=6.5, color=INK2)
    ax.set_xlabel("additive forgetting r (discrete sim., T=300)"); ax.set_ylabel("F(0.05 T): fraction at or below 0.05 T")
    ax.legend(frameon=False, fontsize=6.5); save(fig, "fig7_bnr_additive")

def off_support(model="adaption:gemma-4-31b-it", cond="norm100", N=700, rho=0.5):
    from fss import simulate
    c = load_calib(); D, M, Z = grid_table(c[(c.model == model) & (c.cond == cond)])
    k = VecKernel("flexible", (D, M, Z)); seen = []
    def wrap(delta, mean):
        seen.append(((np.abs(delta) > D[-1]) | (mean < M[0]) | (mean > M[-1])).mean()); return k(delta, mean)
    simulate(wrap, N, rho, rounds=120, seed=0)
    return float(np.mean(seen))

if __name__ == "__main__":
    fig_fss(); fig_bnr()
    for m, c in [("adaption:gemma-4-31b-it", "norm100"), ("adaption:gemma-4-31b-it", "norm0"), ("claude:haiku", "ended+comp")]:
        print("off-support fraction", m, c, round(off_support(m, c), 3))
