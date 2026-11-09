"""Figure 5 (real-task game v2). (a) Verified throughput per round by arm and model. (b) Coordinator incumbency
(probability that a round's coordinator also coordinated the round before) vs its random-order null, arm `none`, and
the w100 - w0 difference with the direction predicted by the two-agent test. Data: results/taskgame2_runs.jsonl.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "code"))
from figs import S, INK2, save, plt, NICE
import taskgame2 as tg

R = ROOT / "results"


def main():
    recs = [json.loads(l) for l in open(R / "taskgame2_runs.jsonl")]
    P = json.load(open(R / "taskgame2_predictions.json"))
    M = {}
    for r in recs:
        M.setdefault((r["model"], r["arm"]), []).append(tg.metrics(r))
    models = tg.MODELS
    lab = [NICE[m].replace("-A4B", "") for m in models]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    arms = [("noplan", "coordination off", "#9a9893"), ("none", "free (no guidance)", S[0]), ("shuffle", "off, shuffled lists", S[2])]
    x = np.arange(len(models)); w = 0.26
    for i, (arm, name, col) in enumerate(arms):
        vals = [[v["throughput"] for v in M[(m, arm)]] for m in models]
        ax.bar(x + (i - 1) * w, [np.mean(v) for v in vals], w, color=col, label=name,
               yerr=[np.std(v, ddof=1) / np.sqrt(len(v)) for v in vals], error_kw=dict(lw=.7, capsize=1.5))
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.5, rotation=15)
    ax.set_ylabel("correct unique tasks per round (max 8)"); ax.set_ylim(0, 9.2)
    ax.set_title("(a) Planning raises output; own task order raises it most", loc="left", fontsize=8.2, weight="bold")
    ax.legend(fontsize=6, frameon=False, loc="upper left", ncol=3)
    ax = axs[1]
    for i, m in enumerate(models):
        inc = [v["incumbency"] for v in M[(m, "none")] if not np.isnan(v["incumbency"])]
        nul = [v["incumbency_null"] for v in M[(m, "none")] if not np.isnan(v["incumbency_null"])]
        d0 = [v["incumbency"] for v in M[(m, "w0")] if not np.isnan(v["incumbency"])]
        d1 = [v["incumbency"] for v in M[(m, "w100")] if not np.isnan(v["incumbency"])]
        ax.plot([i - .12], [np.mean(nul)], "_", color=INK2, ms=10, mew=1.5)
        ax.plot([i - .12], [np.mean(inc)], "o", color=S[0], ms=5)
        diff = np.mean(d1) - np.mean(d0)
        pred = P["predictions"][m]["sign"]
        col = S[1] if np.sign(diff) == pred else "#9a9893"
        ax.annotate("", xy=(i + .15, np.mean(d1)), xytext=(i + .15, np.mean(d0)),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        ax.text(i, -.09, "pred. " + ("+" if pred > 0 else "−"), fontsize=5.5, color=INK2, ha="center")
    ax.plot([], [], "o", color=S[0], label="free: recurrence"); ax.plot([], [], "_", color=INK2, mew=1.5, label="random-order null")
    ax.plot([], [], color=S[1], label="w0→w100, as predicted"); ax.plot([], [], color="#9a9893", label="w0→w100, not as predicted")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(lab, fontsize=6.5, rotation=15)
    ax.set_ylabel("P(coordinator keeps role)"); ax.set_ylim(-.12, .95)
    ax.set_title("(b) Planner recurrence; framing effect", loc="left", fontsize=8.2, weight="bold")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout(); save(fig, "fig5_taskgame")


if __name__ == "__main__":
    main()
