"""Evaluate the two signature-prediction experiments against their frozen predictions and draw Figure 6.

Threshold (Bonabeau / Lacasa-Luque): results/threshold_pop.jsonl vs results/threshold_predictions.json
X2b (Ben-Naim-Redner): results/x2b_pop.jsonl vs results/x2b_predictions.json
  - F(0.1T) vs r: observed seed mean vs frozen simulator / kernel seed-mean intervals; BNR asymptote min(r, 1)
  - front: final maximum score vs (1-r)T + sqrt(T ln N) (BNR extreme-value estimate) and simulator kmax
  - shape: KS distance of the pooled final-score CDF to the master-equation CDF and to the frozen simulator scores,
           with a Monte Carlo reference distribution from pools of simulated agents of the same size
Writes results/physics_tests_eval.json and paper/figures/fig6_physics_tests.{png,pdf}.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import seed_mean_dist
from population2 import order_params

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def ecdf(v, K):
    v = np.asarray(v)
    return np.array([(v <= k).mean() for k in range(K + 1)])


def eval_threshold():
    P = json.load(open(R / "threshold_predictions.json"))
    recs = [json.loads(l) for l in open(R / "threshold_pop.jsonl")]
    out = dict(rho_c=P["rho_c"], s_median=P["s_median"], s_90=P["s_90"], curves={})
    for N in (8, 16):
        rows = []
        for rho in P["rhos"]:
            rr = [r for r in recs if r["N"] == N and r["rho"] == rho]
            if not rr:
                continue
            qs = [order_params(r, P["window"])["q"] for r in rr]; k = len(qs)
            row = dict(rho=rho, k=k, q_obs=float(np.mean(qs)), q_runs=qs)
            for kern in ("nw", "bonabeau", "null"):
                p = [x for x in P["preds"] if x["kernel"] == kern and x["N"] == N and x["rho"] == rho][0]
                d = seed_mean_dist(p["q"], p["B"], p["S"], k)
                row[kern] = [float(d.mean()), float(np.quantile(d, .05)), float(np.quantile(d, .95))]
            row["above_null"] = row["q_obs"] > row["null"][2]
            row["in_nw"] = row["nw"][1] <= row["q_obs"] <= row["nw"][2]
            rows.append(row)
        on = [r["rho"] for r in rows if r["above_null"]]
        out["curves"][str(N)] = dict(rows=rows, onset=min(on) if on else None,
                                     coverage_nw=f"{sum(r['in_nw'] for r in rows)}/{len(rows)}")
        print(f"[threshold] N={N}: mean-field rho_c {P['rho_c'][str(N)]['median']:.2f} "
              f"[{P['rho_c'][str(N)]['q05']:.2f}, {P['rho_c'][str(N)]['q95']:.2f}]; observed onset {out['curves'][str(N)]['onset']};"
              f" NW coverage {out['curves'][str(N)]['coverage_nw']}")
        for r in rows:
            print(f"    rho={r['rho']:.2f} q={r['q_obs']:.2f}  NW {r['nw'][0]:.2f}[{r['nw'][1]:.2f},{r['nw'][2]:.2f}]"
                  f"  null p95 {r['null'][2]:.2f}{'  *above null' if r['above_null'] else ''}")
    return out


def eval_x2b(rng):
    P = json.load(open(R / "x2b_predictions.json"))
    recs = [json.loads(l) for l in open(R / "x2b_pop.jsonl")]
    T, N = P["rounds"], P["N"]
    out = dict(T=T, N=N, rows=[])
    for cond in ("norm100", "ended"):
        for r in P["rs"]:
            rr = [x for x in recs if x["cond"] == cond and x["r"] == r]
            if not rr:
                continue
            k = len(rr)
            obs = np.concatenate([np.array(x["traj"][-1]) for x in rr])
            row = dict(cond=cond, r=r, k=k, F01_obs=float(np.mean([x["L"] for x in rr])), kmax_obs=float(np.mean([x["kmax"] for x in rr])),
                       front_bnr=float(max(1 - r, 0) * T + (np.sqrt(T * np.log(N)) if r < 1 else 0)))
            K = int(max(obs.max(), 2 * T))
            Fo = ecdf(obs, K)
            for p in [p for p in P["preds"] if p["cond"] == cond and p["r"] == r]:
                d = seed_mean_dist(p["L"], len(p["L"]) // 10, 10, k)
                row[f"F01_{p['kernel']}"] = [float(d.mean()), float(np.quantile(d, .05)), float(np.quantile(d, .95))]
                sim = np.asarray(p["final_scores"]); Fs = ecdf(sim, K)
                ks = float(np.abs(Fo - Fs).max())
                null = [np.abs(ecdf(rng.choice(sim, len(obs), replace=False), K) - Fs).max() for _ in range(300)]
                row[f"ks_{p['kernel']}"] = ks; row[f"ks_{p['kernel']}_p"] = float((np.array(null) >= ks).mean())
            Fm = np.asarray(P["master_cdf"][str(r)])[: K + 1]
            if len(Fm) < K + 1:
                Fm = np.r_[Fm, np.ones(K + 1 - len(Fm))]
            row["ks_master"] = float(np.abs(Fo - Fm).max())
            row["F01_master"] = float(Fm[int(0.1 * T)])
            row["F_obs"] = Fo.tolist(); row["F_master"] = Fm.tolist()
            out["rows"].append(row)
            print(f"[X2b] {cond:7s} r={r:<5} F(0.1T) obs {row['F01_obs']:.2f} | master {row['F01_master']:.2f} | "
                  + " | ".join(f"{kk} {row[f'F01_{kk}'][0]:.2f}[{row[f'F01_{kk}'][1]:.2f},{row[f'F01_{kk}'][2]:.2f}] KS {row[f'ks_{kk}']:.2f} (p {row[f'ks_{kk}_p']:.2f})"
                               for kk in ("bnr_q0", "kernel") if f"F01_{kk}" in row)
                  + f" | KS master {row['ks_master']:.2f} | kmax {row['kmax_obs']:.1f} vs BNR front {row['front_bnr']:.1f}")
    return out


def figure(th, xb):
    sys.path.insert(0, str(ROOT / "paper"))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figs import S, MK, INK2, save
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1]))
    # (a) threshold
    ax = axs[0]
    rc = th["rho_c"]["8"]
    ax.axvspan(rc["q05"], min(rc["q95"], 1.05), color="#e6e5e1", alpha=.7, lw=0)
    ax.axvline(rc["median"], color=INK2, ls="--", lw=1)
    for k, N in enumerate(("8", "16")):
        rows = th["curves"].get(N, {}).get("rows", [])
        if not rows:
            continue
        x = [r["rho"] for r in rows]
        if N == "8":
            ax.fill_between(x, [r["nw"][1] for r in rows], [r["nw"][2] for r in rows], color=S[0], alpha=.15, lw=0)
            ax.fill_between(x, [r["null"][1] for r in rows], [r["null"][2] for r in rows], color=INK2, alpha=.12, lw=0)
        ax.plot(x, [r["q_obs"] for r in rows], MK[k] + "-", color=S[0] if N == "8" else S[1], ms=4, lw=1.3, label=f"observed, N={N}")
    ax.text(rc["median"] + .02, .12, "mean-field ρ_c", fontsize=6, color=INK2)
    ax.set_xlabel("contest probability ρ"); ax.set_ylabel("frozen order q"); ax.set_ylim(0, 1)
    ax.set_title("(a) Bonabeau threshold", loc="left", fontsize=8.5, weight="bold"); ax.legend(fontsize=6, frameon=False, loc="upper left")
    # (b) F(0.1T) vs r
    ax = axs[1]
    for cond, col, kern, lab in (("norm100", S[7], "bnr_q0", "w = 100"), ("ended", S[0], "kernel", "neutral")):
        rows = [r for r in xb["rows"] if r["cond"] == cond]
        if not rows:
            continue
        x = [r["r"] for r in rows]
        ax.fill_between(x, [r[f"F01_{kern}"][1] for r in rows], [r[f"F01_{kern}"][2] for r in rows], color=col, alpha=.15, lw=0)
        ax.plot(x, [r["F01_obs"] for r in rows], "o", color=col, ms=4, label=f"observed, {lab}")
    xs = np.linspace(0, 1.3, 50)
    ax.plot(xs, np.minimum(xs, 1), color=INK2, ls="--", lw=1, label="BNR: lower class = r")
    ax.set_xlabel("decline rate r"); ax.set_ylabel("fraction at or below 0.1T"); ax.set_ylim(-.03, 1.05)
    ax.set_title("(b) Ben-Naim–Redner classes", loc="left", fontsize=8.5, weight="bold"); ax.legend(fontsize=6, frameon=False, loc="lower right")
    # (c) distribution shape at one r
    ax = axs[2]
    rsel = 0.4
    for cond, col, lab in (("norm100", S[7], "w = 100"), ("ended", S[0], "neutral")):
        rows = [r for r in xb["rows"] if r["cond"] == cond and r["r"] == rsel]
        if rows:
            ax.step(np.arange(len(rows[0]["F_obs"])), rows[0]["F_obs"], where="post", color=col, lw=1.6, label=f"observed, {lab}")
    rows = [r for r in xb["rows"] if r["r"] == rsel]
    if rows:
        ax.plot(np.arange(len(rows[0]["F_master"])), rows[0]["F_master"], color="black", ls="--", lw=1, label="BNR master eq.")
        ax.set_xlim(0, max(r["kmax_obs"] for r in rows) * 1.15 + 2)
    ax.set_xlabel(f"final score k (r = {rsel}, T = {xb['T']})"); ax.set_ylabel("fraction with score ≤ k")
    ax.set_title("(c) score distribution", loc="left", fontsize=8.5, weight="bold"); ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(); save(fig, "fig6_physics_tests")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    th = eval_threshold() if (R / "threshold_pop.jsonl").exists() else {}
    xb = eval_x2b(rng) if (R / "x2b_pop.jsonl").exists() else {}
    json.dump(dict(threshold=th, x2b=xb), open(R / "physics_tests_eval.json", "w"), default=float)
    if th and xb:
        figure(th, xb)
