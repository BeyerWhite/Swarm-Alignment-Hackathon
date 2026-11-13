"""Kernels for the Chase study: P(LEAD | visible record), fitted on E1 vignette rows and E2 state-matched rows.

Nested logistic kernels (the "why" ladder of the pre-registration):
  K0 constant | K1 + d (direct history) | K2 + o (own record) | K3 + p (partner's record) | K4 + tc (chain)
An arm can only use features its prompt shows:  DYAD <= K1,  OWN / REDACT <= K2,  LOG / SCORE+LOG <= K4.
SCORE+LOG kernels always include the displayed score gap `delta` as well.
FLEX is a Nadaraya-Watson smoother over the arm's full feature set (the counterpart of the main project's
flexible / nw kernels): no functional form, shrunk to the base rate away from data.

Functional form. Visited records reach |o|, |p| of 10 or more, far beyond the E1 grid, and the main project found
the strictly linear logit overconfident at large gaps (holdout M5). The ladder therefore uses the signed-log
transform s(x) = sign(x) * log2(1 + |x|) of every feature. LIN is the top ladder level with raw features, kept as
the strict-Bonabeau comparison.

A fit on rows measured at a different group size than the target (E1 rows use an 8-agent system prompt) carries
an intercept dummy `off` for those rows; predictions set it to 0.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
LADDER = {"K0": [], "K1": ["d"], "K2": ["d", "o"], "K3": ["d", "o", "p"], "K4": ["d", "o", "p", "tc"]}
ARM_MAX = {"DYAD": "K1", "OWN": "K2", "REDACT": "K2", "LOG": "K4", "SCORE+LOG": "K4"}
FEATS = ["d", "o", "p", "tc", "n_own", "n_par", "delta", "mean"]


def levels(arm):
    ks = list(LADDER)
    return ks[: ks.index(ARM_MAX[arm]) + 1]


def cols_for(arm, level):
    return (["delta"] if arm == "SCORE+LOG" else []) + LADDER[level]


def load_rows(paths=None):
    """E1 rows (design='grid', N=8) and E2 state rows (design='state', N as recorded), valid choices only."""
    paths = paths or [ROOT / "results" / "chase_e1_kernel.jsonl", ROOT / "results" / "chase_e2_statecal.jsonl"]
    rows = []
    for p in paths:
        if Path(p).exists():
            rows += [json.loads(l) for l in open(p)]
    df = pd.DataFrame(rows)
    if "design" not in df:
        df["design"] = "grid"
    df["design"] = df.design.fillna("grid")
    if "N" not in df:
        df["N"] = 8
    df["N"] = df.N.fillna(8).astype(int)
    for c in ("delta", "mean"):
        if c not in df:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)
    df = df.drop_duplicates("jid", keep="last")
    df = df[df.choice.notna()].copy()
    df["lead"] = (df.choice == "LEAD").astype(float)
    return df


def select(df, model, framing, arm, N):
    """Rows for one kernel: same model, framing and arm; `off` = 1 for rows measured at another group size."""
    d = df[(df.model == model) & (df.framing == framing) & (df.arm == arm)].copy()
    d["off"] = (d.N != N).astype(float)
    return d


def slog(x):
    """Signed log2 transform: 0 -> 0, +-1 -> +-1, +-3 -> +-2, +-7 -> +-3."""
    return np.sign(x) * np.log2(1 + np.abs(x))


def fit_logistic(d, cols, rng=None, ridge=0.05, tf=slog):
    """Ridge-penalised logistic MLE of lead on [1, tf(cols), off]. With `rng`: row bootstrap. Returns pfun(feat)."""
    use_off = bool(d.off.any()) and bool((d.off == 0).any())
    X = np.c_[np.ones(len(d)), tf(d[cols].values.astype(float).reshape(len(d), len(cols)))]
    if use_off:
        X = np.c_[X, d.off.values]
    y = d.lead.values
    if rng is not None:
        idx = rng.integers(0, len(d), len(d)); X, y = X[idx], y[idx]
    pen = np.r_[0.0, np.full(X.shape[1] - 1, ridge)]

    def nll(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z) + np.sum(pen * w ** 2)

    def grad(w):
        z = X @ w
        return X.T @ (1 / (1 + np.exp(-z)) - y) + 2 * pen * w
    w = minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS").x
    a, b = float(w[0]), [float(x) for x in w[1:1 + len(cols)]]

    def p(f):
        z = a + sum(bi * float(tf(f[c])) for bi, c in zip(b, cols))
        return 1 / (1 + np.exp(-z)) if z > -700 else 0.0
    p.params = dict(a=a, **dict(zip(cols, b)), **({"off": float(w[-1])} if use_off else {}))
    return p


def fit_flex(d, cols, rng=None, bw=0.6, alpha=1.0):
    """Nadaraya-Watson over slog(cols) on same-N rows when there are any, else on all rows. With `rng`: row bootstrap."""
    dd = d[d.off == 0] if (d.off == 0).sum() >= 50 else d
    Z = slog(dd[cols].values.astype(float).reshape(len(dd), len(cols))); y = dd.lead.values
    if rng is not None:
        idx = rng.integers(0, len(y), len(y)); Z, y = Z[idx], y[idx]
    p0 = (y.sum() + 0.5) / (len(y) + 1)
    cache = {}

    def p(f):
        key = tuple(round(float(f[c]), 2) for c in cols)
        if key not in cache:
            w = np.exp(-0.5 * (((Z - slog(np.array(key))) / bw) ** 2).sum(1)) if cols else np.ones(len(y))
            cache[key] = float((w @ y + alpha * p0) / (w.sum() + alpha))
        return cache[key]
    p.params = dict(p0=float(p0), n=int(len(y)))
    return p


def fit(d, arm, level, rng=None):
    if level == "FLEX":
        return fit_flex(d, cols_for(arm, ARM_MAX[arm]), rng)
    if level == "LIN":
        return fit_logistic(d, cols_for(arm, ARM_MAX[arm]), rng, tf=lambda x: x)
    return fit_logistic(d, cols_for(arm, level), rng)


def score_kernel(model, framing):
    """Legacy score-board kernel for the SCORE arm, from the paper's grid + state-matched calibration."""
    from predict import load_calib, load_statecal
    from population2 import KERNELS
    cond = {"comp": "ended+comp", "norm100": "norm100", "norm0": "norm0"}[framing]
    c, s = load_calib(), load_statecal()
    d = pd.concat([c, s[[x for x in s.columns if x in c.columns]]], ignore_index=True) if len(s) else c
    d = d[(d.model == model) & (d.cond == cond)]

    def make(rng=None):
        k = KERNELS["nw"](d, rng)
        return lambda f: k(f["delta"], f["mean"])
    return make, len(d)
