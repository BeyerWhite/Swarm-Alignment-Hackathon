"""Matched-range comparison of status sensitivity in isolated pairs (calibration) vs inside LLM populations.
By eq. (1), s = 4 * d ell / d Delta at Delta = 0. We estimate the local slope by least squares of the LEAD indicator
on Delta over |Delta| <= 1 (displayed scores), separately for calibration rows and for every individual decision
made inside the common-design populations (round-start displayed scores replayed exactly). Run-level bootstrap.
Post-hoc diagnostic for the systematic under-prediction of stratification. Writes results/matched_sensitivity.csv.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import displayed_states
from predict import load_calib

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def slope(d, y):
    d, y = np.asarray(d, float), np.asarray(y, float)
    if len(d) < 10 or d.std() == 0:
        return np.nan
    return 4 * np.polyfit(d, y, 1)[0]


def decisions(rec):
    out = []
    for disp, x in zip(displayed_states(rec), rec["history"]):
        for i, j, ci, cj, w in x["events"]:
            for a_, b_, c in ((i, j, ci), (j, i, cj)):
                dd = round(disp[a_] - disp[b_], 1)
                if abs(dd) <= 1.0:
                    out.append((dd, 1.0 if c == "LEAD" else 0.0))
    return out


def main(B=500, seed=0):
    rng = np.random.default_rng(seed)
    cal = load_calib()
    recs = [json.loads(l) for f in sorted(R.glob("population_v2*.jsonl")) for l in open(f)]
    recs = [r for r in recs if r["rho"] == 0.5 and r["N"] == 8]
    rows = []
    for (m, c) in sorted({(r["model"], r["cond"]) for r in recs}):
        rr = [r for r in recs if r["model"] == m and r["cond"] == c]
        per = [decisions(r) for r in rr]
        allp = [z for p in per for z in p]
        cc = cal[(cal.model == m) & (cal.cond == c) & (cal.delta.abs() <= 1.0)]
        s_pop = slope([z[0] for z in allp], [z[1] for z in allp])
        s_cal = slope(cc.delta, cc.lead)
        bp, bc = [], []
        for _ in range(B):
            pk = [per[k] for k in rng.integers(0, len(per), len(per))]
            zz = [z for p in pk for z in p]
            bp.append(slope([z[0] for z in zz], [z[1] for z in zz]))
            cb = cc.sample(len(cc), replace=True, random_state=int(rng.integers(1e9)))
            bc.append(slope(cb.delta, cb.lead))
        diff = np.array(bp) - np.array(bc)
        rows.append(dict(model=m, cond=c, n_pop=len(allp), n_cal=len(cc), s_pop=s_pop, s_cal=s_cal,
                         diff=s_pop - s_cal, diff_lo=np.nanpercentile(diff, 2.5), diff_hi=np.nanpercentile(diff, 97.5)))
    t = pd.DataFrame(rows)
    t.to_csv(R / "matched_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    sens = t[~t.model.str.contains("deepseek")]
    print("\nstatus-sensitive models: s_pop > s_cal in", int((sens["diff"] > 0).sum()), "of", len(sens),
          "cells; credibly (CI > 0) in", int((sens.diff_lo > 0).sum()), "; credibly < 0 in", int((sens.diff_hi < 0).sum()))


if __name__ == "__main__":
    main()
