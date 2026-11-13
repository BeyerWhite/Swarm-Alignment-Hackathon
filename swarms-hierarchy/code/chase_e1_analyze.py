"""Frozen analysis of E1 (bystander kernel). Estimands and tests are defined in
refine-logs/CHASE_PREREGISTRATION.md section 5; this file is hash-stamped in results/chase_e1_freeze.json.

Per model x framing (lead = 1 if the strict parser returns LEAD):
  beta_p   bystander slope: OLS (linear probability) slope of lead on p in the LOG grid, controlling for o; HC1 SE.
           Chase predicts beta_p < 0 (lead less against partners with winning records).
  beta_o   winner/loser slope on the agent's own record: (a) in the OWN block, (b) in the LOG grid.
  beta_d   direct-history slope in the DYAD block.
  T        transitive inference, difference-in-differences of LEAD rates:
           [for_shared - against_shared] - [for_distinct - against_distinct]; Wald z. Prediction T > 0.
  S_cwin, S_closs   shared-vs-distinct contrasts where the shared third party carries no ordering information.
  R        redaction contrast: mean over o of [rate(REDACT, 3 outcome-less lines) - rate(LOG, p = 0)].
Primary tests: beta_p and T, Holm-adjusted across all model x framing cells within each estimand.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]


def lpm(df, cols):
    """OLS of lead on [1, cols] with HC1 standard errors -> {col: (beta, se, p)}."""
    X = np.c_[np.ones(len(df)), df[cols].values.astype(float)]
    y = df.lead.values.astype(float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n, k = X.shape
    V = XtX_inv @ (X.T * e ** 2) @ X @ XtX_inv * n / max(n - k, 1)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    out = {}
    for i, c in enumerate(cols, 1):
        z = b[i] / se[i] if se[i] > 0 else 0.0
        out[c] = (float(b[i]), float(se[i]), float(2 * norm.sf(abs(z))) if se[i] > 0 else 1.0)
    return out


def rate(df, cell):
    d = df[df.cell == cell]
    return (float(d.lead.mean()), len(d)) if len(d) else (float("nan"), 0)


def did(df, a, b, c, d):
    """(a - b) - (c - d) of cell LEAD rates with a Wald test."""
    rs = [rate(df, x) for x in (a, b, c, d)]
    if any(n == 0 for _, n in rs):
        return (float("nan"), float("nan"), 1.0)
    est = (rs[0][0] - rs[1][0]) - (rs[2][0] - rs[3][0])
    se = float(np.sqrt(sum(p * (1 - p) / n for p, n in rs)))
    return (float(est), se, float(2 * norm.sf(abs(est / se))) if se > 0 else 1.0)


def diff(df, a, b):
    (pa, na), (pb, nb) = rate(df, a), rate(df, b)
    if not na or not nb:
        return (float("nan"), float("nan"), 1.0)
    se = float(np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb))
    return (float(pa - pb), se, float(2 * norm.sf(abs((pa - pb) / se))) if se > 0 else 1.0)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main(path=None):
    path = path or ROOT / "results" / "chase_e1_kernel.jsonl"
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    df = df.drop_duplicates("jid", keep="last")
    invalid = df.choice.isna().groupby([df.model, df.framing]).mean()
    df = df[df.choice.notna()].copy()
    df["lead"] = (df.choice == "LEAD").astype(float)
    cells = df.groupby(["model", "framing", "cell", "arm", "block", "d", "o", "p", "tc"]).lead.agg(["mean", "sum", "count"]).reset_index()
    cells.to_csv(ROOT / "results" / "chase_e1_cells.csv", index=False)
    rows = []
    for (m, fr), g in df.groupby(["model", "framing"]):
        grid = lpm(g[g.block == "grid"], ["o", "p"])
        own = lpm(g[g.block == "own"], ["o"])
        dyad = lpm(g[g.block == "dyad"], ["d"])
        direct = lpm(g[g.block == "direct"], ["d", "p"])
        red = [diff(g, f"X_o{o:+d}", f"L_o{o:+d}_p+0") for o in (-2, 0, 2)]
        rows.append(dict(
            model=m.split(":")[1], framing=fr, n=len(g), invalid=float(invalid.get((m, fr), 0)),
            base=rate(g, "L_o+0_p+0")[0], lead_all=float(g.lead.mean()),
            beta_p=grid["p"], beta_o_log=grid["o"], beta_o_own=own["o"], beta_d=dyad["d"],
            direct_d=direct["d"], direct_p=direct["p"],
            contrast_p=diff(g[g.block == "grid"].assign(cell=np.where(g[g.block == "grid"].p == 3, "hi", np.where(g[g.block == "grid"].p == -3, "lo", "x"))), "hi", "lo"),
            T=did(g, "R_for_shared", "R_against_shared", "R_for_distinct", "R_against_distinct"),
            chain_gap=diff(g, "R_for_shared", "R_against_shared"),
            ctrl_gap=diff(g, "R_for_distinct", "R_against_distinct"),
            S_cwin=diff(g, "R_cwin_shared", "R_cwin_distinct"), S_closs=diff(g, "R_closs_shared", "R_closs_distinct"),
            R=(float(np.mean([r[0] for r in red])), float(np.sqrt(np.sum([r[1] ** 2 for r in red])) / 3)),
            conflict_dwin_against=rate(g, "LC_dwin_against")[0], conflict_dloss_for=rate(g, "LC_dloss_for")[0],
        ))
    for key in ("beta_p", "T"):
        adj = holm([r[key][2] for r in rows])
        for r, a in zip(rows, adj):
            r[key + "_holm"] = float(a)
    for r in rows:
        r["responsive"] = bool(r["beta_p_holm"] < 0.05 or r["T_holm"] < 0.05)
    (ROOT / "results" / "chase_e1_summary.json").write_text(json.dumps(rows, indent=1))
    f3 = lambda t: f"{t[0]:+.3f}±{t[1]:.3f}"
    print(f"{'model':20s} {'framing':8s} {'n':>5s} inv   base | beta_p (holm)        | beta_o own    log         | beta_d        | T (holm)             | R")
    for r in rows:
        print(f"{r['model']:20s} {r['framing']:8s} {r['n']:5d} {r['invalid']:.3f} {r['base']:.2f} | {f3(r['beta_p'])} ({r['beta_p_holm']:.3f}) | "
              f"{f3(r['beta_o_own'])} {f3(r['beta_o_log'])} | {f3(r['beta_d'])} | {f3(r['T'])} ({r['T_holm']:.3f}) | {r['R'][0]:+.3f}±{r['R'][1]:.3f}"
              f"{'  *' if r['responsive'] else ''}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
