"""Figure 3 (revised): predicted vs observed population dispersion sd(h), common design (27 status-sensitive cells +
DeepSeek control). (a) Frozen, pre-registered Jeffreys-smoothed kernel (prospective). (b) Post-hoc unshrunk monotone
(isotonic) kernel on the same calibration data. Data: results/posthoc_attenuation.csv (same seed-mean predictive
distributions and 90% intervals as the paper's evaluation).
"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
from figs import S, MK, INK2, save, plt, NICE, ORDER

D = pd.read_csv(ROOT / "results" / "posthoc_attenuation.csv")
D = D[(D.design == "common") & (D.metric == "sd_h")]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
mx = 7.2
for ax, est, title in ((axs[0], "flexible", "(a) pre-registered kernel (prospective)"),
                       (axs[1], "flex_iso", "(b) unshrunk kernel (post hoc)")):
    f = D[D.estimator == est]
    for k, m in enumerate([m for m in ORDER if m in set(f.model)]):
        d = f[f.model == m]
        ax.errorbar(d.pred, d.obs, xerr=[d.pred - d.lo, d.hi - d.pred], fmt=MK[k % 8], color=S[k % 8], ms=4.3,
                    capsize=1.5, elinewidth=.8, label=NICE[m], zorder=3)
    if est == "flexible":
        nl = D[D.estimator == "null"]
        ax.scatter(nl.pred, nl.obs, facecolors="none", edgecolors=INK2, s=15, lw=.8, label="status-blind null", zorder=2)
    sens = f[~f.model.str.contains("deepseek")]
    ax.text(.35, mx - .55, f"above diagonal: {int((sens.obs > sens.pred).sum())}/{len(sens)} cells\n"
                           f"mean obs − pred: {(sens.obs - sens.pred).mean():+.2f}", fontsize=6.6, color=INK2, va="top")
    ax.plot([0, mx], [0, mx], ls="--", lw=.9, color=INK2); ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("predicted dispersion sd(h) (90% PI)"); ax.set_title(title, loc="left", fontsize=8.3, weight="bold")
axs[0].set_ylabel("observed dispersion sd(h)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(); save(fig, "fig3_attenuation")
