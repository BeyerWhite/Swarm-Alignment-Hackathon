"""Supplementary sizing calculation for E3, from the frozen E2 kernels only (no LLM calls, no population data).

For one model x framing cell: bootstrap-refit the OWN and LOG kernels, simulate more groups than planned, and report
the power of the pre-registered LOG-versus-OWN tests at several group counts (Fisher exact on linear counts for
groups of four; Mann-Whitney on per-group t_tri at N = 8). Done for the ladder kernels (K2 vs K4) and for FLEX.
Does not touch results/chase_e2_predictions.json.  -> results/chase_e2_power_supp.json

  chase_e2_power.py --cells "model|framing" ... [--B 200] [--procs 8]
"""
import argparse, json, multiprocessing, sys, zlib
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
sys.path.insert(0, str(Path(__file__).parent))
import chase_fit as F
from chase_pop import run, KernelDecider
from chase_metrics import events_from_history, structure
from chase_e2 import ROUNDS, RHO

ROOT = Path(__file__).resolve().parents[1]
GMAX = {4: 160, 8: 20}
SIZES = {4: [40, 60, 80, 100, 120, 160], 8: [6, 8, 10, 12, 16, 20]}
PAIRS = {"ladder": ("K2", "K4"), "FLEX": ("FLEX", "FLEX")}
_G = {}


def _boot(args):
    key, level, b = args
    model, framing, arm, N = key
    rng = np.random.default_rng([b, N, 777, zlib.crc32(f"{model}|{framing}|{arm}|{level}".encode())])
    k = F.fit(_G[key], arm, level, rng)
    st = [structure(events_from_history(run(N, RHO, ROUNDS[N], arm, KernelDecider(k, arm), seed=int(rng.integers(1e9)),
                                            keep_feats=False)), N, ROUNDS[N]) for _ in range(GMAX[N])]
    return [int(s["linear"]) for s in st], [float(np.nan_to_num(s["t_tri"])) for s in st]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    rows = F.load_rows()
    tasks = []
    for cell in a.cells:
        model, framing = cell.split("|")
        for N in (4, 8):
            for arm in ("OWN", "LOG"):
                key = (model, framing, arm, N)
                _G[key] = F.select(rows, model, framing, arm, N)
                for level in {p[0 if arm == "OWN" else 1] for p in PAIRS.values()}:
                    tasks += [(key, level, b) for b in range(a.B)]
    with multiprocessing.get_context("fork").Pool(a.procs) as pool:
        res = dict(zip(tasks, pool.map(_boot, tasks, chunksize=2)))
    out = []
    for cell in a.cells:
        model, framing = cell.split("|")
        for name, (lo, ll) in PAIRS.items():
            e = dict(model=model, framing=framing, kernels=name, N4={}, N8={})
            for G in SIZES[4]:
                ps, d = [], []
                for b in range(a.B):
                    x = sum(res[((model, framing, "OWN", 4), lo, b)][0][:G]); y = sum(res[((model, framing, "LOG", 4), ll, b)][0][:G])
                    ps.append(fisher_exact([[x, G - x], [y, G - y]])[1]); d.append((y - x) / G)
                e["N4"][G] = dict(power=float(np.mean(np.array(ps) < 0.05)), diff=float(np.mean(d)))
            for G in SIZES[8]:
                ps, d = [], []
                for b in range(a.B):
                    x = res[((model, framing, "OWN", 8), lo, b)][1][:G]; y = res[((model, framing, "LOG", 8), ll, b)][1][:G]
                    ps.append(mannwhitneyu(y, x, alternative="two-sided").pvalue if len(set(x) | set(y)) > 1 else 1.0)
                    d.append(float(np.mean(y) - np.mean(x)))
                e["N8"][G] = dict(power=float(np.mean(np.array(ps) < 0.05)), diff=float(np.mean(d)))
            out.append(e)
            print(model.split(":")[1], framing, name)
            print("  N4 groups per arm:", {G: round(v["power"], 2) for G, v in e["N4"].items()}, "diff", round(e["N4"][40]["diff"], 2))
            print("  N8 seeds per arm: ", {G: round(v["power"], 2) for G, v in e["N8"].items()}, "diff", round(e["N8"][6]["diff"], 2))
    (ROOT / "results" / "chase_e2_power_supp.json").write_text(json.dumps(dict(B=a.B, cells=a.cells, results=out)))


if __name__ == "__main__":
    main()
