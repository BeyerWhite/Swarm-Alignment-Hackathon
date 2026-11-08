"""X1 step 1: cross-pair probe. Model A vs model B under the identical prompt (condition `cond`), gaps
Delta in {-6,-3,0,3,6} (A's score minus B's), n contests per gap, positions/IDs/tasks randomised.
Tests the independence identity P_win^{A vs B}(D) = 1/2 + 1/2[l_A(D) - l_B(-D)] using single-model calibration."""
import argparse, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).parent))
from llm import call, parse_choice
from prompts_v2 import SYSTEM, TASKS, pair_prompt
from calibrate import CONDS
from predict import load_calib

ROOT = Path(__file__).resolve().parents[1]
GAPS = [-6, -3, 0, 3, 6]


def ell(d, model, cond):
    """Single-model LEAD rate at displayed gap d (pooled over mean), linear interpolation on the grid."""
    g = d[(d.model == model) & (d.cond == cond)].groupby("delta").lead.mean()
    return float(np.interp(-0.0 if False else 0, [0], [0])) if g.empty else (lambda x: float(np.interp(x, g.index, g.values)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", default="adaption:deepseek-v4-flash")
    ap.add_argument("--B", default="adaption:gemma-4-31b-it")
    ap.add_argument("--cond", default="norm100")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    rng = random.Random(11)
    jobs = []
    for D in GAPS:
        for c in range(a.n):
            ia, ib = rng.sample(range(1, 100), 2); task = rng.choice(TASKS); base = rng.choice([-2.0, 0.0, 2.0])
            hA, hB = base + D / 2, base - D / 2
            jobs.append(dict(D=D, hA=hA, hB=hB,
                             pa=(SYSTEM.format(me=ia, n=8), pair_prompt(ia, ib, hA, hB, rng, CONDS[a.cond], task=task)),
                             pb=(SYSTEM.format(me=ib, n=8), pair_prompt(ib, ia, hB, hA, rng, CONDS[a.cond], task=task))))

    def one(j):
        ca = parse_choice(call(a.A, *j["pa"], tag="x1_probe"))
        cb = parse_choice(call(a.B, *j["pb"], tag="x1_probe"))
        return ca, cb
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, jobs))
    out = ROOT / "results" / "x1_probe.jsonl"
    with open(out, "a") as f:
        for j, (ca, cb) in zip(jobs, res):
            f.write(json.dumps(dict(A=a.A, B=a.B, cond=a.cond, D=j["D"], hA=j["hA"], hB=j["hB"], cA=ca, cB=cb)) + "\n")
    calib = load_calib()
    lA, lB = ell(calib, a.A, a.cond), ell(calib, a.B, a.cond)
    tot = 0.0
    print(f"{a.A} vs {a.B} [{a.cond}]")
    for D in GAPS:
        r = [x for j, x in zip(jobs, res) if j["D"] == D and x[0] and x[1]]
        wins = sum(1.0 if (ca == "LEAD" and cb != "LEAD") else 0.5 if ca == cb else 0.0 for ca, cb in r)
        n = len(r); pred = 0.5 + 0.5 * (lA(D) - lB(-D)); obs = wins / max(n, 1)
        x2 = (obs - pred) ** 2 * n / max(pred * (1 - pred), 1e-3); tot += x2
        print(f"  D={D:+d}: observed P_win(A)={obs:.2f} (n={n})  identity predicts {pred:.2f}  "
              f"[l_A={lA(D):.2f}, l_B(-D)={lB(-D):.2f}]  chi2={x2:.2f}")
    print(f"  total chi2={tot:.2f} on {len(GAPS)} df, p={1 - chi2.cdf(tot, len(GAPS)):.3f}")


if __name__ == "__main__":
    main()
