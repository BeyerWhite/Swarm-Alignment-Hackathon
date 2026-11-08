"""Paper figures (static PNG + PDF). Palette: validated reference categorical slots in fixed order
(blue, orange, aqua, yellow); low-contrast slots carry secondary encoding (marker shapes + legend)."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from predict import load_calib
from population2 import kernel_logistic_delta, order_params

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2, "lines.markersize": 6})
NICE = {"claude:haiku": "Claude Haiku", "claude:sonnet": "Claude Sonnet", "gpt:gpt-6-luna": "GPT-6-luna",
        "adaption:deepseek-v4-flash": "DeepSeek-v4-flash", "adaption:qwen3.8-27b": "Qwen3.8-27B",
        "adaption:qwen3.5-397b": "Qwen3.5-397B", "adaption:gemma-4-31b-it": "Gemma-4-31B",
        "adaption:gemma-4-26b-a4b-it": "Gemma-4-26B-A4B", "adaption:glm-5.2": "GLM-5.2"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def eta_table(B=200):
    df = load_calib(); rows = []
    rng = np.random.default_rng(0)
    for (m, c), d in df.groupby(["model", "cond"]):
        pt = kernel_logistic_delta(d).params[2]
        bs = [kernel_logistic_delta(d, rng).params[2] for _ in range(B)]
        rows.append(dict(model=m, cond=c, eta=pt, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), n=len(d)))
    t = pd.DataFrame(rows); t.to_csv(ROOT / "results" / "eta_table.csv", index=False)
    return t


def fig_norm_sweep(t):
    """Fig 1: implied local eta vs organisers' norm weight w, one line per model."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    models = [m for m in NICE if m in set(t.model) and (t[t.model == m].cond.str.startswith("norm")).sum() >= 3]
    for k, m in enumerate(models[:8]):
        d = t[(t.model == m) & t.cond.str.startswith("norm")].copy()
        d["w"] = d.cond.str[4:].astype(int); d = d.sort_values("w")
        ax.errorbar(d.w + (k - len(models) / 2) * 0.8, d.eta, yerr=[d.eta - d.lo, d.hi - d.eta], color=SERIES[k],
                    marker=MARKERS[k], capsize=2, elinewidth=1, label=NICE[m])
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("norm weight w  (0 = equal turns, 100 = defer to higher scores)")
    ax.set_ylabel("finite-difference status sensitivity\n(|Δ| ≤ 1; posterior median, 95% CI)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "fig1_eta_vs_norm")


def fig_wording(t):
    """Fig 2: eta under the 2x2 wording factorial, small multiples by model."""
    conds = ["ended", "won", "ended+comp", "won+comp"]
    models = [m for m in NICE if m in set(t.model) and set(conds) <= set(t[t.model == m].cond)]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(models)); w = 0.2
    for k, c in enumerate(conds):
        d = t[t.cond == c].set_index("model").loc[models]
        ax.bar(x + (k - 1.5) * w, d.eta, w * 0.9, color=SERIES[k], label=c, yerr=[d.eta - d.lo, d.hi - d.eta],
               error_kw=dict(lw=1, capsize=1.5, ecolor=INK2))
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in models], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("finite-difference status sensitivity\n(posterior median, 95% CI)"); ax.legend(frameon=False, fontsize=7, ncol=4)
    save(fig, "fig2_wording_factorial")


def fig_pred_vs_obs(metric="sd_h"):
    """Fig 3: observed seed-mean vs flexible-kernel 90% predictive interval; null as hollow markers."""
    ev = json.load(open(ROOT / "results" / "eval_v2.json"))
    t = pd.DataFrame(ev["table"]); t = t[(t.metric == metric) & (t.rho == 0.5) & (t.N == 8) & (t.design == "grid")]
    f = t[t.kernel == "flexible"].reset_index(drop=True); nl = t[t.kernel == "null"].set_index(["model", "cond"])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    lo = [float(s.strip("[]").split(",")[0]) for s in f.pred90]; hi = [float(s.strip("[]").split(",")[1]) for s in f.pred90]
    models = [m for m in NICE if m in set(f.model)]
    for k, m in enumerate(models[:8]):
        idx = f.index[f.model == m]
        ax.errorbar(f.loc[idx, "pred_mean"], f.loc[idx, "obs_mean"],
                    xerr=[f.loc[idx, "pred_mean"] - np.array(lo)[idx], np.array(hi)[idx] - f.loc[idx, "pred_mean"]],
                    fmt=MARKERS[k], color=SERIES[k], capsize=2, elinewidth=1, label=NICE[m])
    nn = nl.loc[list(zip(f.model, f.cond))]
    ax.scatter(nn.pred_mean, f.obs_mean, facecolors="none", edgecolors=INK2, s=22, label="status-blind null", zorder=1)
    mx = max(f.obs_mean.max(), max(hi)) * 1.08
    ax.plot([0, mx], [0, mx], color=INK2, lw=1, ls="--"); ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    lab = {"sd_h": "status dispersion sd(h)", "q": "frozen-order fraction q"}[metric]
    ax.set_xlabel(f"predicted {lab}  (pairwise kernel)"); ax.set_ylabel(f"observed {lab}  (LLM population)")
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, f"fig3_pred_vs_obs_{metric}")


def fig_trajectories(model, conds=("norm0", "ended+comp", "norm100")):
    """Fig 4: status trajectories h_i(t) for one seed per condition (small multiples, shared y)."""
    recs = [json.loads(l) for p in sorted((ROOT / "results").glob("population_v2*.jsonl")) for l in open(p)]
    fig, axs = plt.subplots(1, len(conds), figsize=(7.2, 2.4), sharey=True)
    for ax, c in zip(axs, conds):
        r = [x for x in recs if x["model"] == model and x["cond"] == c]
        if not r:
            continue
        H = np.array([x["h"] for x in r[0]["history"]])
        order = np.argsort(-H[-1])
        for rank, i in enumerate(order):
            ax.plot(H[:, i], color=SERIES[0] if rank == 0 else INK2, lw=2 if rank == 0 else 1, alpha=1 if rank == 0 else .5)
        ax.set_title(c, fontsize=9); ax.set_xlabel("round")
    axs[0].set_ylabel("displayed status h")
    fig.suptitle(NICE.get(model, model), fontsize=9, y=1.02)
    save(fig, f"fig4_trajectories_{model.split(':')[-1]}")


def fig_rho_sweep(model, metric="sd_h", conds=("norm0", "ended+comp", "norm100")):
    """Fig 5: order parameter vs interaction rate rho; frozen flexible (grid) 90% bands vs observed seeds."""
    ev = pd.DataFrame(json.load(open(ROOT / "results" / "eval_v2.json"))["table"])
    recs = [json.loads(l) for p in sorted((ROOT / "results").glob("population_v2*.jsonl")) for l in open(p)]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for k, c in enumerate(conds):
        e = ev[(ev.model == model) & (ev.cond == c) & (ev.metric == metric) & (ev.kernel == "flexible")
               & (ev.design == "grid")].sort_values("rho")
        if e.empty:
            continue
        lo = [float(s.strip("[]").split(",")[0]) for s in e.pred90]; hi = [float(s.strip("[]").split(",")[1]) for s in e.pred90]
        ax.fill_between(e.rho, lo, hi, color=SERIES[k], alpha=.18, lw=0)
        ax.plot(e.rho, e.pred_mean, color=SERIES[k], lw=1, ls="--")
        obs = pd.DataFrame([(r["rho"], order_params(r)[metric]) for r in recs if r["model"] == model and r["cond"] == c],
                           columns=["rho", "v"])
        if len(obs):
            g = obs.groupby("rho").v.mean()                      # same statistic as the band: mean over seeds
            ax.scatter(g.index, g.values, color=SERIES[k], marker=MARKERS[k], s=28, label=f"{c} (seed mean)",
                       zorder=3, edgecolors=SURF, linewidths=.8)
    ax.set_xlabel("interaction rate ρ"); ax.set_ylabel({"sd_h": "status dispersion sd(h)", "q": "frozen order q"}[metric])
    ax.set_title(NICE.get(model, model) + ": observed seed means vs frozen kernel 90% PI for the seed mean", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7)
    save(fig, f"fig5_rho_sweep_{model.split(':')[-1]}_{metric}")


if __name__ == "__main__":
    eta_table()
    t = pd.read_csv(ROOT / "results" / "eta_fd_beta.csv").rename(columns={"fd_sens": "eta"})
    fig_norm_sweep(t); fig_wording(t)
    if (ROOT / "results" / "eval_v2.json").exists():
        fig_pred_vs_obs("sd_h"); fig_pred_vs_obs("q")
    for m in sys.argv[1:]:
        fig_trajectories(m)
    if (ROOT / "results" / "eval_v2.json").exists():
        for m in ("adaption:deepseek-v4-flash", "adaption:gemma-4-31b-it"):
            for met in ("sd_h", "q"):
                try:
                    fig_rho_sweep(m, met)
                except Exception as ex:
                    print("rho sweep skipped", m, ex)
    print("figures ->", FIG)
