"""Addendum A7 to the loss-weight sweep (frozen before any F-sweep population data).

Curve-level fixed-eta vs measured-s comparison. The onset grid point separates the two hypotheses only at F = 3, so
we also score the full q(rho) curves by predictive density:
  fixed_eta : NW kernel from the F = 1 calibration (threshold sweep rows), simulated under F = 2, 3
  measured  : NW kernel from the F-specific calibration (already frozen in fsweep_predictions.json)
Adds rho = 0.2 (splits F = 2: fixed-eta rho_c 0.25 vs measured 0.17) with predictions for both kernels and null.
Writes results/fsweep_addendum_A7.json (write-once).
"""
import hashlib, json, sys, datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from population2 import KERNELS, KernelDecider, run, order_params
from hysteresis import calib_rows
import fsweep as fs

R = Path(__file__).resolve().parents[1] / "results"
OUT = R / "fsweep_addendum_A7.json"
EXTRA = [0.2]


def main(B=12, S=6):
    if OUT.exists():
        sys.exit("frozen; refusing to overwrite")
    assert not (R / "fsweep_pop.jsonl").exists(), "population data already exist"
    rng = np.random.default_rng(7)
    rows1 = calib_rows(fs.MODEL, fs.COND)
    res = dict(written=datetime.datetime.now().isoformat(timespec="seconds"), extra_rhos=EXTRA,
               fsweep_pred_sha=json.load(open(R / "fsweep_predictions.json"))["sha256"],
               a6_sha=json.load(open(R / "fsweep_addendum_A6.json"))["sha256"],
               note="fixed_eta = NW kernel from F=1 calibration simulated at F; scored against observed q by composite "
                    "LPD over all rho, alongside the frozen measured-s NW predictions. rho=0.2 predictions for "
                    "nw (F-specific), fixed_eta and null. Caveat: F=1 calibration covers mean scores near 0, so "
                    "fixed_eta extrapolates at the negative scores F>1 populations reach.",
               preds=[])
    for F in fs.FS:
        dfF = fs.calib_df(F)
        for kern, src in (("fixed_eta", rows1), ("nw", dfF), ("null", dfF)):
            rhos = fs.RHOS + EXTRA if kern == "fixed_eta" else EXTRA
            for rho in rhos:
                qs, sds = [], []
                for b in range(B):
                    k = KERNELS["nw" if kern == "fixed_eta" else kern](src, rng)
                    for s in range(S):
                        o = order_params(run(fs.N, rho, fs.MU, fs.T, KernelDecider(k), seed=int(rng.integers(1e9)), F=F), fs.WIN)
                        qs.append(o["q"]); sds.append(o["sd_h"])
                res["preds"].append(dict(F=F, kernel=kern, rho=rho, q=qs, sd_h=sds, B=B, S=S))
            print(F, kern, [round(float(np.mean(p["q"])), 2) for p in res["preds"] if p["F"] == F and p["kernel"] == kern], flush=True)
    res["sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    OUT.write_text(json.dumps(res)); print("frozen", res["sha256"])


if __name__ == "__main__":
    main()
