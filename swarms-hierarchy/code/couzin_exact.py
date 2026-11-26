"""Couzin study, post-hoc (M-A4): separation-proof tests of the uninformed-buffer trend (H5 / F2 and their reverse).

The registered tests use a Wald z from logistic regression, which fails under separation: Gemma's minority won 0 of
24 groups in 6 of 7 M4 arms, so the slope SE was about 40. This script reports, for every M4-type dataset (main M4 on
Gemma-31B, M4b on the rule-selected model, and each follow-up model):

- a permutation test of the trend: statistic sum((n_u - mean n_u) * y) over groups, y = minority wins (ties 0.5),
  with y permuted across the groups of one protocol, 20,000 permutations; one-sided p for "falls with n_u" (the H5
  direction) and for "rises with n_u";
- a Firth-penalised logistic slope (finite under separation) with its Wald 90% interval.

These are sensitivity analyses added after the data were seen. They do not replace the registered results.
Usage: couzin_exact.py   -> results/couzin_exact.json
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PERMS = 20000


def firth(X, y, it=200):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = p * (1 - p)
        info = X.T @ (X * W[:, None]); inv = np.linalg.inv(info + 1e-10 * np.eye(len(b)))
        h = W * np.einsum("ij,jk,ik->i", X, inv, X)
        step = inv @ (X.T @ (y - p + h * (0.5 - p)))
        b += np.clip(step, -2, 2)
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = p * (1 - p)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * W[:, None]) + 1e-10 * np.eye(len(b)))))
    return b, se


def trend(levels, rng):
    """levels: list of (n_u, outcomes array)."""
    nu = np.concatenate([np.full(len(v), n, float) for n, v in levels]); y = np.concatenate([v for _, v in levels]).astype(float)
    c = nu - nu.mean(); T = float(c @ y)
    perm = np.array([c @ rng.permutation(y) for _ in range(PERMS)])
    b, se = firth(np.column_stack([nu, np.ones_like(nu)]), y)
    return dict(n=len(y), rates={str(n): float(np.mean(v)) for n, v in levels},
                p_falls=float((np.sum(perm <= T) + 1) / (PERMS + 1)), p_rises=float((np.sum(perm >= T) + 1) / (PERMS + 1)),
                firth_slope=float(b[0]), firth_ci90=[float(b[0] - 1.645 * se[0]), float(b[0] + 1.645 * se[0])])


def load(path, model=None):
    G = {}
    for l in open(path):
        r = json.loads(l)
        if model and r["model"] != model:
            continue
        G.setdefault((r["model"], r["arm"]), []).append(r["out"]["majA"])
    return G


def main():
    rng = np.random.default_rng(31)
    out = {}
    sets = []
    G = load(RES / "couzin_pop.jsonl")
    models = sorted({m for m, a in G if a.startswith("M4")})
    for m in models:
        if any(a.startswith("M4_") for mm, a in G if mm == m):
            sets.append((f"main M4 ({m})", m, G, "M4_sim_nu0", "M4", (6, 12, 24)))
        if any(a.startswith("M4b_") for mm, a in G if mm == m):
            sets.append((f"main M4b ({m})", m, G, "M4b_sim_nu0", "M4b", (12, 24)))
    for d in sorted(RES.glob("couzin_*/couzin_pop.jsonl")):
        Gd = load(d)
        for m in sorted({m for m, a in Gd if a.startswith("M4b")}):
            sets.append((f"follow-up M4b ({m})", m, Gd, "M4b_sim_nu0", "M4b", (12, 24)))
    for name, m, Gx, base, pre, nus in sets:
        out[name] = {}
        for pr in ("sim", "obs"):
            ids = [base] + [f"{pre}_{pr}_nu{n}" for n in nus]
            if not all((m, a) in Gx for a in ids):
                out[name][pr] = dict(status="incomplete"); continue
            levels = [(0, np.array(Gx[(m, base)]))] + [(n, np.array(Gx[(m, f"{pre}_{pr}_nu{n}")])) for n in nus]
            out[name][pr] = trend(levels, rng)
        print(name)
        for pr, t in out[name].items():
            if "rates" in t:
                print(f"   {pr}: rates {t['rates']}  p(falls) {t['p_falls']:.4f}  p(rises) {t['p_rises']:.4f}  "
                      f"Firth slope {t['firth_slope']:+.3f} [{t['firth_ci90'][0]:+.3f}, {t['firth_ci90'][1]:+.3f}]")
            else:
                print(f"   {pr}: {t['status']}")
    (RES / "couzin_exact.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
