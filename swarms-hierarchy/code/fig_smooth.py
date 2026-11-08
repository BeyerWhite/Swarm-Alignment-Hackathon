"""Figure 4 (smooth, not sudden). (a) Threshold sweep: observed q vs rho for Gemma-4-26B LLM populations (N = 8, 16)
against the frozen kernel prediction, the status-blind null and the mean-field rho_c. (b) Simulated q(rho) from the
measured Gemma-4-31B kernel (w = 100) for N = 8 ... 700 (results/fss_rho_w100.jsonl, from the companion session's
fss.py sweep, no LLM calls).
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
from figs import S, MK, INK2, save, plt

R = ROOT / "results"


def main():
    th = json.load(open(R / "threshold_eval.json"))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax = axs[0]
    rc = th["rho_c"]["8"]
    ax.axvspan(rc["q05"], 1.02, color="#efeeea", lw=0)
    ax.axvline(rc["median"], color=INK2, ls="--", lw=1)
    ax.text(rc["median"] + .015, .08, "mean-field $\\rho_c$\n(90% interval shaded)", fontsize=6, color=INK2)
    for k, N in enumerate(("8", "16")):
        rows = th["curves"][N]["rows"]
        x = [r["rho"] for r in rows]
        col = S[0] if N == "8" else S[1]
        ax.fill_between(x, [r["nw"][1] for r in rows], [r["nw"][2] for r in rows], color=col, alpha=.13, lw=0)
        ax.plot(x, [r["nw"][0] for r in rows], color=col, lw=1, ls=":")
        ax.plot(x, [r["q_obs"] for r in rows], MK[k] + "-", color=col, ms=4, lw=1.3, label=f"LLM populations, N = {N}")
    rows = th["curves"]["8"]["rows"]
    ax.fill_between([r["rho"] for r in rows], [r["null"][1] for r in rows], [r["null"][2] for r in rows],
                    color="#9a9893", alpha=.25, lw=0, label="status-blind null")
    ax.set_xlabel("contact probability $\\rho$"); ax.set_ylabel("rank stability q"); ax.set_ylim(0, 1); ax.set_xlim(0, 1.02)
    ax.set_title("(a) LLM threshold sweep (Gemma-4-26B)", loc="left", fontsize=8.5, weight="bold")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax = axs[1]
    recs = [json.loads(l) for l in open(R / "fss_rho_w100.jsonl")]
    recs = [r for r in recs if r["model"] == "adaption:gemma-4-31b-it" and r["kind"] == "flexible"]
    Ns = sorted({r["N"] for r in recs})
    cm = plt.get_cmap("viridis")
    for i, N in enumerate(Ns):
        xs = sorted({r["rho"] for r in recs if r["N"] == N})
        ys = [np.mean([r["q"] for r in recs if r["N"] == N and r["rho"] == x]) for x in xs]
        ax.plot(xs, ys, color=cm(i / max(1, len(Ns) - 1)), lw=1.3, label=f"N = {N}")
    ax.set_xlabel("contact probability $\\rho$"); ax.set_ylabel("rank stability q (simulated)"); ax.set_ylim(0, 1)
    ax.set_title("(b) No sharpening with size (simulation)", loc="left", fontsize=8.5, weight="bold")
    ax.legend(fontsize=5.5, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout(); save(fig, "fig4_smooth")


if __name__ == "__main__":
    main()
