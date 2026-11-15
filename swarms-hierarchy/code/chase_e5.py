"""E5 (exploratory, H5): Gemma-4-26B under the equal-turns framing `norm0`. Amendment C6 in refine-logs/CHASE_LOG.md.

Runs the E2 machinery of chase_e2.py unchanged, with this stage's own arms and files, so that no frozen E2 file is
touched:
  chase_e5.py calib   --cells "adaption:gemma-4-26b-a4b-it|norm0"   state-matched calibration, pass 1 (as in E2)
  chase_e5.py calib2  --cells ...                                    pass 2: records visited by surrogates driven by
                                                                     the kernels refitted on E1 + pass-1 rows
  chase_e5.py predict --cells ...                                    frozen forecasts, write-once
  -> results/chase_e5_statecal.jsonl, results/chase_e5_predictions.json

Differences from E2, all fixed before any call: no score arms (the legacy score kernel does not cover this model),
and the second calibration pass, added because the E3 post-hoc diagnosis found that surrogates driven by E1 kernels
under-sample the records real populations reach.
"""
import random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import chase_e2 as E
import chase_fit as F
from chase_prompts import SYSTEM, TASKS, log_prompt

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "results" / "chase_e1_kernel.jsonl"
STATECAL = ROOT / "results" / "chase_e5_statecal.jsonl"
PRED = ROOT / "results" / "chase_e5_predictions.json"
CAL1 = {4: {"DYAD": 150, "OWN": 300, "LOG": 450, "REDACT": 150}, 8: {"OWN": 300, "LOG": 450}}
CAL2 = {4: {"OWN": 150, "LOG": 225, "REDACT": 75}, 8: {"OWN": 150, "LOG": 225}}
PLAN = {4: {"OWN": 40, "LOG": 40, "DYAD": 20, "REDACT": 20}, 8: {"OWN": 6, "LOG": 6}}


def calib_jobs2(_e1, cells, controls, seed):
    """Pass 2: like chase_e2.calib_jobs, but the surrogate groups are driven by kernels fitted on E1 + pass-1 rows
    at the population's own group size (top two ladder levels of the arm)."""
    rows = F.load_rows([E1, STATECAL])
    jobs = []
    for cell in cells:
        model, framing = cell.split("|")
        for N, arms in CAL2.items():
            for arm, k in arms.items():
                d = F.select(rows, model, framing, arm, N)
                if not (d.design == "state").any():
                    sys.exit(f"no pass-1 rows for {(model, framing, arm, N)}; run calib first")
                lv = F.levels(arm)
                rng_sim = np.random.default_rng([seed, 2, N, len(arm)])
                views = []
                for kern in (F.fit(d, arm, lv[-1]), F.fit(d, arm, lv[-2])):
                    for _ in range(30 if N == 4 else 8):
                        dec = E.Tracer(kern, arm)
                        E.run(N, E.RHO, E.ROUNDS[N], arm, dec, seed=int(rng_sim.integers(1e9)), keep_feats=False)
                        views += dec.views
                rng = random.Random(f"{seed}|{cell}|{arm}|{N}|pass2")
                for s in range(k):
                    v = rng.choice(views)
                    prompt = log_prompt(arm, framing, v["me"], v["other"], v["t_now"], v["own"], v["partner"],
                                        rng.choice(TASKS), scores=v["scores"], rng=rng)
                    jobs.append(dict(jid=f"{model}|{framing}|{arm}|N{N}|r2s{s}", model=model, framing=framing, arm=arm, N=N,
                                     design="state", system=SYSTEM.format(me=v["me"], n=N), prompt=prompt,
                                     **{c: v[c] for c in F.FEATS}))
    return jobs


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("calib", "calib2", "predict"):
        sys.exit(__doc__)
    if any(x in sys.argv for x in ("--controls", "--statecal", "--out", "--e1")):
        sys.exit("chase_e5 fixes its own files and has no controls")
    E.PLAN, E.PLAN_CONTROL, E.CAL_CONTROL = PLAN, {}, {}
    if cmd == "calib2":
        E.CAL, E.calib_jobs = CAL2, calib_jobs2
        sys.argv[1] = "calib"
    else:
        E.CAL = CAL1
    sys.argv += ["--e1", str(E1), "--statecal", str(STATECAL), "--out", str(PRED)]
    E.main()


if __name__ == "__main__":
    main()
