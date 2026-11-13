"""E0 (retrospective, no LLM calls): Chase triad sequences and linearity in the EXISTING score-board populations,
compared with surrogate populations driven by the paper's pairwise score kernels.

Every comparison here is RETROSPECTIVE: the populations were run and scored on 7-8 Nov 2026 for other metrics.
Kernels are fitted on pairwise calibration rows only (grid + state-matched), never on population decisions.
Output: results/chase_e0_retro.json
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from population2 import KERNELS, KernelDecider, run
from predict import load_calib, load_statecal
from chase_metrics import events_from_history, structure, pooled

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["dd_ds", "p_trans", "t_tri", "landau", "dci"]
S = int(sys.argv[1]) if len(sys.argv) > 1 else 200          # surrogate populations per kernel


def main():
    calib = load_calib()
    st = load_statecal()
    both = pd.concat([calib, st[[c for c in st.columns if c in calib.columns or c == "design"]]], ignore_index=True)
    runs = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "population_v2*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if r.get("N") == 8 and abs(r.get("rho", 0) - 0.5) < 1e-9 and r.get("rounds") == 40 and isinstance(r.get("cond"), str):
                runs.setdefault((r["model"], r["cond"]), []).append(r)
    out = []
    rng = np.random.default_rng(0)
    for (model, cond), recs in sorted(runs.items()):
        obs = [structure(events_from_history(r), 8, 40) for r in recs]
        entry = dict(model=model, cond=cond, n_runs=len(recs), observed=pooled(obs))
        d_grid = calib[(calib.model == model) & (calib.cond == cond)]
        d_both = both[(both.model == model) & (both.cond == cond)]
        for name, kern, d in (("nw_state", "nw", d_both), ("bonabeau_state", "bonabeau", d_both), ("null", "null", d_grid)):
            if len(d) < 50:
                continue
            k = KERNELS[kern](d)
            sims = [structure(events_from_history(run(8, 0.5, 0.1, 40, KernelDecider(k), seed=int(rng.integers(1e9)))), 8, 40)
                    for _ in range(S)]
            # predictive distribution of the POOLED statistic over n_runs populations (same pooling as observed)
            n = len(recs)
            pools = [pooled([sims[i] for i in rng.integers(0, S, n)]) for _ in range(400)]
            entry[name] = {key: dict(mean=float(np.mean([p[key] for p in pools])),
                                     q05=float(np.quantile([p[key] for p in pools], .05)),
                                     q95=float(np.quantile([p[key] for p in pools], .95))) for key in KEYS}
        out.append(entry)
        o = entry["observed"]
        line = f"{model.split(':')[1]:20s} {cond:11s} n={len(recs)} obs dd_ds={o['dd_ds']:.2f} t_tri={o['t_tri']:+.2f} h'={o['landau']:.2f}"
        for name in ("nw_state", "null"):
            if name in entry:
                e = entry[name]
                line += f" | {name}: dd_ds={e['dd_ds']['mean']:.2f}[{e['dd_ds']['q05']:.2f},{e['dd_ds']['q95']:.2f}] t_tri={e['t_tri']['mean']:+.2f}[{e['t_tri']['q05']:+.2f},{e['t_tri']['q95']:+.2f}]"
        print(line, flush=True)
    (ROOT / "results" / "chase_e0_retro.json").write_text(json.dumps(dict(
        label="RETROSPECTIVE", surrogates_per_kernel=S, note="pooled over runs; 90% predictive intervals", rows=out), indent=1))


if __name__ == "__main__":
    main()
