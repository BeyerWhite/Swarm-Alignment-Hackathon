"""Summarise v2 calibration: LEAD curves on displayed delta, local eta with bootstrap CI (Bonabeau kernel)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from predict import load_calib
from population2 import kernel_logistic_delta


def eta_ci(d, B=300, seed=0):
    rng = np.random.default_rng(seed)
    pt = kernel_logistic_delta(d).params[2]
    bs = [kernel_logistic_delta(d, rng).params[2] for _ in range(B)]
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def local_slope(d):
    """Nonparametric local P_win slope from |delta|<=1 cells: eta_loc = 4 * [P_win(delta)-1/2]/delta averaged."""
    out = []
    for x in sorted(set(d.delta[(d.delta > 0) & (d.delta <= 1.0)])):
        p, m = d[d.delta == x].lead.mean(), d[d.delta == -x].lead.mean()
        out.append(4 * ((p - m) / 2) / x)
    return float(np.mean(out)) if out else np.nan


if __name__ == "__main__":
    df = load_calib()
    pd.set_option("display.width", 250)
    print(df.groupby(["model", "cond", "delta"]).lead.mean().unstack("delta").round(2).to_string()); print()
    rows = []
    for (m, c), d in df.groupby(["model", "cond"]):
        pt, lo, hi = eta_ci(d)
        rows.append(dict(model=m, cond=c, n=len(d), p_lead=round(d.lead.mean(), 2),
                         p_tie=round(d[d.delta == 0].lead.mean(), 2), eta_bonabeau=round(pt, 3),
                         ci=f"[{lo:.3f}, {hi:.3f}]", eta_local_np=round(local_slope(d), 3),
                         mean_effect=round(d.groupby("mean").lead.mean().diff().mean(), 3)))
    print(pd.DataFrame(rows).to_string(index=False))
