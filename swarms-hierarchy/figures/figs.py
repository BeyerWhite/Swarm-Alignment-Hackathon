"""Paper figures (Apart template, US Letter, 6.5 in text width). Writes figures/*.png (300 dpi) + .pdf.
Palette: validated categorical slots in fixed order; diverging blue (turn-taking) <-> red (defers to status),
grey midpoint. Every figure is self-contained in its caption."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
R, OUT = ROOT / "results", ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f08f8e", "#b52f2e"])
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "lines.linewidth": 1.8,
                     "font.family": "DejaVu Sans"})
NICE = {"claude:haiku": "Claude Haiku", "claude:sonnet": "Claude Sonnet", "gpt:gpt-6-luna": "GPT-6-luna",
        "adaption:deepseek-v4-flash": "DeepSeek-v4-flash", "adaption:qwen3.8-27b": "Qwen3.8-27B",
        "adaption:qwen3.5-397b": "Qwen3.5-397B", "adaption:gemma-4-31b-it": "Gemma-4-31B",
        "adaption:gemma-4-26b-a4b-it": "Gemma-4-26B-A4B", "adaption:glm-5.2": "GLM-5.2"}
ORDER = ["claude:haiku", "claude:sonnet", "gpt:gpt-6-luna", "adaption:gemma-4-31b-it", "adaption:gemma-4-26b-a4b-it",
         "adaption:glm-5.2", "adaption:qwen3.5-397b", "adaption:qwen3.8-27b", "adaption:deepseek-v4-flash"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 1: protocol + pipeline schematic
def fig1():
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    for ax in axs:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(ax, x, y, w, h, txt, fc="#f4f3f0", fs=6.8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.3", fc=fc, ec=INK2, lw=.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
                linespacing=1.25)

    def arrow(ax, a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=INK2, lw=.9))

    a = axs[0]; a.set_title("(a) One coordinator contest", loc="left", fontsize=8.5, weight="bold")
    box(a, 0.2, 6.6, 4.3, 2.9, "Agent i sees:\ntask, partner j,\nscores h_i and h_j,\nframing text")
    box(a, 5.4, 6.6, 4.3, 2.9, "Agent j sees:\nsame task,\nmirrored scores,\nsame framing")
    box(a, 0.35, 3.9, 4.0, 1.6, "LEAD or DEFER", fc="#dbe8fa", bold=True, fs=6.6)
    box(a, 5.55, 3.9, 4.0, 1.6, "LEAD or DEFER", fc="#dbe8fa", bold=True, fs=6.6)
    arrow(a, (2.35, 6.6), (2.35, 5.55)); arrow(a, (7.55, 6.6), (7.55, 5.55))
    box(a, 0.4, 0.2, 9.2, 2.6, "Lone LEAD becomes coordinator;\nboth LEAD or both DEFER: coin flip.\n"
        "Coordinator +1, other −1; all scores ×0.9.", fc="#fbe3d8")
    arrow(a, (2.35, 3.9), (3.4, 2.85)); arrow(a, (7.55, 3.9), (6.6, 2.85))

    b = axs[1]; b.set_title("(b) Testing pairs → groups", loc="left", fontsize=8.5, weight="bold")
    steps = [("1. Calibrate: measure each model's\nP(LEAD | own score, partner score)", "#dbe8fa"),
             ("2. Freeze: simulate groups from that\nkernel; save predictions (write-once)", "#f4f3f0"),
             ("3. Run: 8 LLM agents, 40 rounds,\nidentical protocol", "#fbe3d8"),
             ("4. Score: observed vs predicted,\nand vs a status-blind null", "#f4f3f0")]
    for k, (t, c) in enumerate(steps):
        y = 7.6 - k * 2.45
        box(b, 0.3, y, 9.4, 1.85, t, fc=c)
        if k < 3:
            arrow(b, (5.0, y), (5.0, y - 0.55))
    save(fig, "fig1_protocol")


# ------------------------------------------------------------------ Fig 2: sensitivity heatmap
def fig2():
    t = pd.read_csv(R / "eta_fd_beta.csv")
    conds = ["norm0", "norm25", "norm50", "norm75", "norm100", "ended", "won", "ended+comp", "won+comp"]
    lab = ["w=0\n(equal\nturns)", "w=25", "w=50", "w=75", "w=100\n(defer to\nscores)", "neutral", "\"won\"", "neutral\n+comp.", "\"won\"\n+comp."]
    M = np.full((len(ORDER), len(conds)), np.nan); sig = np.zeros_like(M, bool)
    for i, m in enumerate(ORDER):
        for j, c in enumerate(conds):
            r = t[(t.model == m) & (t.cond == c)]
            if len(r):
                M[i, j] = r.fd_sens.iloc[0]; sig[i, j] = (r.lo.iloc[0] > 0) | (r.hi.iloc[0] < 0)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -3.5, 3.5), aspect="auto")
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "–", ha="center", va="center", fontsize=7, color=INK2); continue
            ax.text(j, i, f"{M[i, j]:+.1f}" + ("*" if sig[i, j] else ""), ha="center", va="center", fontsize=6.6,
                    color="white" if abs(M[i, j]) > 2.2 else INK)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(lab, fontsize=6.5)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([NICE[m] for m in ORDER])
    ax.axvline(4.5, color="white", lw=3)
    ax.text(2, -0.95, "explicit organiser norm weight w", ha="center", fontsize=7.2, color=INK2)
    ax.text(6.5, -0.95, "score wording (no norm)", ha="center", fontsize=7.2, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("status sensitivity\n(− turn-taking · + defers to higher score)", fontsize=7)
    save(fig, "fig2_sensitivity_heatmap")


# ------------------------------------------------------------------ Fig 3: predicted vs observed
def fig3():
    ev = pd.DataFrame(json.load(open(R / "eval_v2.json"))["table"])
    ev = ev[(ev.rho == 0.5) & (ev.N == 8) & (ev.design == "grid")]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.15))
    for ax, met, lab in zip(axs, ("sd_h", "q"), ("status dispersion sd(h)", "frozen-order fraction q")):
        f = ev[(ev.metric == met) & (ev.kernel == "flexible")]; nl = ev[(ev.metric == met) & (ev.kernel == "null")]
        nl = nl.set_index(["model", "cond"])
        for k, m in enumerate([m for m in ORDER if m in set(f.model)]):
            d = f[f.model == m]
            lo = d.pred90.map(lambda s: float(s.strip("[]").split(",")[0])); hi = d.pred90.map(lambda s: float(s.strip("[]").split(",")[1]))
            ax.errorbar(d.pred_mean, d.obs_mean, xerr=[d.pred_mean - lo, hi - d.pred_mean], fmt=MK[k % 8],
                        color=S[k % 8], ms=4.5, capsize=1.5, elinewidth=.8, label=NICE[m], zorder=3)
        nn = nl.loc[list(zip(f.model, f.cond))]
        ax.scatter(nn.pred_mean, f.obs_mean.values, facecolors="none", edgecolors=INK2, s=16, lw=.8,
                   label="status-blind null", zorder=2)
        mx = max(f.obs_mean.max(), f.pred_mean.max()) * 1.1
        ax.plot([0, mx], [0, mx], ls="--", lw=.9, color=INK2); ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.set_xlabel(f"predicted {lab}\n(frozen pairwise kernel, 90% PI)"); ax.set_ylabel(f"observed {lab}")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    axs[0].set_title("(a) dispersion", loc="left", weight="bold"); axs[1].set_title("(b) persistence", loc="left", weight="bold")
    fig.tight_layout()
    save(fig, "fig3_pred_vs_obs")


# ------------------------------------------------------------------ Fig 4: mixed populations (X1)
def fig4():
    rf = json.load(open(R / "reeval_followups.json"))["x1"]
    arms = [("ds1_gemma7", "1 DeepSeek +\n7 Gemma-31B\n(w=100)", 1 / 8), ("ds4_gemma4", "4 DeepSeek +\n4 Gemma-31B\n(w=100)", 4 / 8),
            ("ds1_glm7_w0", "1 DeepSeek +\n7 GLM-5.2\n(w=0)", 1 / 8)]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    for i, (a, lab, null) in enumerate(arms):
        k = rf[a]["rank1/nw"] if "rank1/nw" in rf[a] else rf[a]["rank1_minority/nw"]
        ax.plot([i - .25, i + .25], [null, null], color=INK2, lw=1.2, ls=":")
        ax.errorbar(i - .08, k["pred"], yerr=[[k["pred"] - k["lo"]], [k["hi"] - k["pred"]]], fmt="s", color=S[0],
                    ms=5, capsize=2.5, label="frozen kernel (90% PI)" if i == 0 else None)
        ax.plot(i + .1, rf[a]["rank1_last10"] / rf[a]["n"], "o", color=S[1], ms=6,
                label="observed (5 seeds)" if i == 0 else None)
        ax.text(i + .1, rf[a]["rank1_last10"] / rf[a]["n"] + .07, f"{rf[a]['rank1_last10']}/{rf[a]['n']}", ha="center", fontsize=7)
    ax.plot([], [], color=INK2, ls=":", label="chance (composition)")
    ax.set_xticks(range(3)); ax.set_xticklabels([a[1] for a in arms], fontsize=6.6); ax.set_ylim(0, 1.15)
    ax.set_ylabel("P(leader is a DeepSeek agent)"); ax.set_ylim(0, 1.45); ax.set_yticks([0, .25, .5, .75, 1])
    ax.legend(fontsize=6.2, frameon=False, loc="upper left", ncol=1)
    ax.set_title("(a) who ends up on top", loc="left", weight="bold")
    ax = axs[1]
    recs = [json.loads(l) for l in open(R / "x1_pop.jsonl")]
    r = [x for x in recs if x["arm"] == "ds1_gemma7"][0]
    H = np.array([x["h"] for x in r["history"]]); ty = r["types"]
    for i in range(H.shape[1]):
        ds = ty[i].endswith("deepseek-v4-flash")
        ax.plot(H[:, i], color=S[1] if ds else "#9b9a95", lw=2.2 if ds else 1.0, zorder=3 if ds else 1,
                label=("DeepSeek-v4-flash (never defers)" if ds else ("Gemma-4-31B" if i == [j for j, t in enumerate(ty) if not t.endswith("flash")][0] else None)))
    ax.set_xlabel("round"); ax.set_ylabel("score h"); ax.legend(fontsize=6.3, frameon=False, loc="lower left")
    ax.set_title("(b) one run, 1 DeepSeek + 7 Gemma", loc="left", weight="bold")
    fig.tight_layout(); save(fig, "fig4_mixed_populations")


# ------------------------------------------------------------------ Fig 5: hysteresis
def fig5():
    rf = json.load(open(R / "reeval_followups.json"))["hyst"]
    recs = [json.loads(l) for l in open(R / "hyst_pop.jsonl")]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw=dict(width_ratios=[1, 1.2]))
    ax = axs[0]
    conds = [("norm0", "w=0\n(equal turns)"), ("ended", "neutral"), ("norm50", "w=50")]
    for i, (c, lab) in enumerate(conds):
        k = rf[f"{c}/seeded/nw"]["rank1"]
        ax.errorbar(i - .1, k["pred"], yerr=[[k["pred"] - k["lo"]], [k["hi"] - k["pred"]]], fmt="s", color=S[0], ms=5,
                    capsize=2.5, label="frozen kernel, seed mean (90% PI)" if i == 0 else None)
        ps = k["per_seed"]
        ax.plot([i + .12] * len(ps), ps, "o", color=S[1], ms=4.5, alpha=.85, label="observed runs" if i == 0 else None)
        ax.plot(i + .12, np.mean(ps), "_", color=INK, ms=14, mew=2, label="observed mean" if i == 0 else None)
    ax.axhline(40, color=INK2, ls=":", lw=.9); ax.text(2.45, 40.8, "run length (censored)", fontsize=6, ha="right", color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in conds]); ax.set_ylim(0, 44)
    ax.set_ylabel("rounds seeded leader stays at rank 1"); ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    ax.set_title("(a) does an installed leader persist?", loc="left", weight="bold")
    ax = axs[1]
    for k, (c, lab) in enumerate([("norm0", "w=0"), ("norm50", "w=50")]):
        r = [x for x in recs if x["cond"] == c and x["arm"] == "seeded"][0]
        H = np.array([x["h"] for x in r["history"]]); lead = int(np.argmax(r["h0"]))
        ax.plot(H[:, lead], color=S[k * 7 if k else 0] if False else (S[0] if k == 0 else S[7]), lw=2, label=f"seeded leader, {lab}")
        others = np.delete(H, lead, axis=1)
        ax.fill_between(range(len(H)), others.min(1), others.max(1), color=S[0] if k == 0 else S[7], alpha=.12,
                        label=f"other agents (range), {lab}")
    ax.set_xlabel("round"); ax.set_ylabel("score h"); ax.legend(fontsize=6.2, frameon=False, loc="upper right")
    ax.set_title("(b) Gemma-31B, seeded start", loc="left", weight="bold")
    fig.tight_layout(); save(fig, "fig5_hysteresis")


# ------------------------------------------------------------------ Fig A1: finite-size / finite-time
def figA1():
    d = pd.read_json(R / "fss_w.jsonl", lines=True); d = d[d.kind == "flexible"]
    hz = pd.DataFrame(json.load(open(R / "fss_horizon.json")))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axs[0]
    for k, N in enumerate([8, 32, 128, 700]):
        g = d[(d.model == "adaption:gemma-4-31b-it") & (d.N == N)].groupby("w").q.agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], color=S[k], marker=MK[k], ms=3.5, lw=1.3, capsize=1.5, elinewidth=.7, label=f"N = {N}")
    ax.set_xlabel("norm weight w (kernel interpolated)"); ax.set_ylabel("q (40-round window)")
    ax.set_title("(a) Gemma-31B kernel: no sharpening with N", loc="left", weight="bold"); ax.legend(frameon=False, fontsize=6.5)
    ax = axs[1]
    h = hz[hz.N == 700]
    labs = {(120, 40): "T=120, window 40", (600, 40): "T=600, window 40", (600, 200): "T=600, window 200"}
    x = np.arange(3); wdt = .26
    for k, (key, lab) in enumerate(labs.items()):
        v = [h[(h.model == m) & (h.rounds == key[0]) & (h.window == key[1])].q_mean.iloc[0]
             for m in ("claude:haiku", "adaption:gemma-4-31b-it", "adaption:glm-5.2")]
        ax.bar(x + (k - 1) * wdt, v, wdt * .92, color=S[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(["Haiku\nw=75", "Gemma-31B\nw=50", "GLM-5.2\nw=25"], fontsize=6.8)
    ax.set_ylabel("q at N = 700"); ax.legend(frameon=False, fontsize=6.2); ax.set_ylim(0, 1)
    ax.set_title("(b) q depends on window", loc="left", weight="bold")
    fig.tight_layout(); save(fig, "figA1_finite_size")


# ------------------------------------------------------------------ Fig A2: BNR-inspired populations
def figA2():
    rf = json.load(open(R / "reeval_followups.json"))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for ax, src, ttl, rs in ((axs[0], "x2", "(a) X2: N=16, perfect matching", [0.25, 0.5, 1.25]),
                             (axs[1], "bnr", "(b) rate-2 normalisation, N=12", [0.5, 1.0, 1.5, 2.0, 2.5])):
        for k, (cond, kern, lab) in enumerate([("norm100", "bnr_q0" if src == "x2" else "grid", "w=100: kernel/BNR sim."),
                                               ("ended", "kernel" if src == "x2" else "grid", "neutral: kernel"),
                                               ("norm100", None if src == "x2" else "blind", "w=100: status-blind")]):
            if kern is None:
                continue
            pts = [(r, rf[src].get(f"{cond}/r={r}/{kern}")) for r in rs]
            pts = [(r, p) for r, p in pts if p]
            col = [S[7], S[0], INK2][k]
            ax.fill_between([r for r, _ in pts], [p["lo"] for _, p in pts], [p["hi"] for _, p in pts], color=col, alpha=.15, lw=0)
            ax.plot([r for r, _ in pts], [p["pred"] for _, p in pts], color=col, ls="--", lw=1, label=f"predicted, {lab}")
            if kern != "blind":
                ax.plot([r for r, _ in pts], [p["obs"] for _, p in pts], MK[k], color=col, ms=5, label=f"observed, {'w=100' if cond=='norm100' else 'neutral'}")
        ax.set_xlabel("forgetting rate r"); ax.set_ylabel("F(0.1T): fraction at or below 0.1T"); ax.set_ylim(-0.05, 1.08)
        ax.set_title(ttl, loc="left", weight="bold")
    axs[1].legend(frameon=False, fontsize=5.8, loc="lower right")
    fig.tight_layout(); save(fig, "figA2_bnr")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, figA1, figA2):
        f(); print("ok", f.__name__)
