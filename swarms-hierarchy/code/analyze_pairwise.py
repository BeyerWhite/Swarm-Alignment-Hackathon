"""Fit micro-parameters from pairwise LLM decisions and map them onto the Bonabeau model."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, str(Path(__file__).parent))
from theory import rho_c_bonabeau

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    df["lead"] = (df.choice == "LEAD").astype(float)
    df["valid"] = df.choice.notna()
    return df


def sig(x):
    return 1 / (1 + np.exp(-x))


def fit_cell(d):
    """Logistic p(LEAD) = sig(a + b*delta + c*mean). Returns params and eta_eff.

    Contest win prob P(D) = 1/2 + (p(D) - p(-D))/2 ; Bonabeau P = sig(eta*D) has slope eta/4 at 0,
    so eta_eff = 4 * dP/dD|0 = 4 * b * sig'(a + c*mean) evaluated at the sample mean status.
    """
    d = d[d.valid]
    X = sm.add_constant(d[["delta", "mean"]].astype(float))
    try:
        m = sm.Logit(d.lead, X).fit(disp=0)
        a, b, c = m.params["const"], m.params["delta"], m.params["mean"]
    except Exception:            # perfect separation etc.
        m = sm.GLM(d.lead, X, family=sm.families.Binomial()).fit_regularized(alpha=0.05)
        a, b, c = m.params
    z = a + c * d["mean"].mean()
    eta = 4 * b * sig(z) * (1 - sig(z))
    return dict(a=a, b=b, c=c, eta=eta, base_lead=sig(z))


def eta_nonparam(d, n_grid=4001):
    """Fit Bonabeau P(D)=sig(eta*D) to the implied contest curve P(D)=1/2+(p(D)-p(-D))/2 (D>0),
    by binomial pseudo-likelihood on the pooled LEAD rates. Returns eta (inf if deterministic)."""
    d = d[d.valid]
    g = d.groupby("delta").lead.agg(["mean", "count"])
    ll_best, eta_best = -np.inf, 0.0
    D = [x for x in g.index if x > 0 and -x in g.index]
    if not D:
        return np.nan
    P = np.array([0.5 + (g.loc[x, "mean"] - g.loc[-x, "mean"]) / 2 for x in D])
    n = np.array([min(g.loc[x, "count"], g.loc[-x, "count"]) for x in D])
    for eta in np.linspace(-10, 10, n_grid):
        m = np.clip(sig(eta * np.array(D, float)), 1e-9, 1 - 1e-9)
        ll = (n * (P * np.log(m) + (1 - P) * np.log(1 - m))).sum()
        if ll > ll_best:
            ll_best, eta_best = ll, eta
    return eta_best          # grid edge +-10 means "effectively deterministic"


def summarise(df, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for (model, frame), d in df.groupby(["model", "frame"]):
        est = fit_cell(d)
        est["eta"] = eta_nonparam(d)
        boots = [eta_nonparam(d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))))
                 for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p0 = d[d.valid & (d.delta == 0)].lead.mean()
        rows.append(dict(model=model, frame=frame, n=int(d.valid.sum()), invalid=int((~d.valid).sum()),
                         p_lead=round(d[d.valid].lead.mean(), 3), p_lead_tie=round(p0, 2), slope_b=round(est["b"], 3),
                         mean_effect_c=round(est["c"], 3), eta=round(est["eta"], 3),
                         eta_ci=f"[{lo:.3f}, {hi:.3f}]",
                         rho_c_N16_mu0p1=(round(rho_c_bonabeau(0.1, est["eta"], 1, 16), 3)
                                          if est["eta"] > 0 else "none (eta<=0)")))
    return pd.DataFrame(rows)


def curve_table(df):
    t = df[df.valid].groupby(["model", "frame", "delta"]).lead.mean().unstack("delta").round(2)
    return t


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "pairwise_pilot.jsonl"
    df = load(path)
    print(curve_table(df).to_string())
    print()
    print(summarise(df).to_string(index=False))
