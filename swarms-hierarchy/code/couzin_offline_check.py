"""Couzin study offline check (no API calls): does the simulator reproduce the engine?

Scripted agents (couzin.FAKE_PARAMS) have known decision probabilities, so their exact probability tables can be
written down. For a set of arms, the engine (couzin.run_group with scripted agents) and the simulator with the exact
tables must agree within Monte Carlo error. A disagreement is a bug in one of them, not an estimation problem.
Usage: COUZIN_RESULTS=<scratch dir> couzin_offline_check.py [--groups 200]
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np

os.environ["COUZIN_FAKE"] = "1"
sys.path.insert(0, str(Path(__file__).parent))
import couzin as cz
import couzin_forecast as cf

sig = lambda z: 1 / (1 + np.exp(-z))


def exact_tables(model):
    p = cz.fake_params(model)
    L = np.array([math.log((j + .5) / (cz.K + .5 - j)) for j in range(cz.K + 1)])
    a, pers, lab = p["a"], p["pers"], p["lab"]
    E = dict(U0=float(sig(lab)), Uph=np.stack([sig(a * L + pers)] * 2),
             Uent=0.5 * (sig(a * L + lab) + sig(a * L - lab)), I0={}, I1={}, I2={},
             CL=np.stack([sig(a * L + pers + 1.5), sig(a * L - pers + 1.5)]))
    for w in cz.OMEGAS:
        b = cz.FAKE_OMEGA_BIAS(w)
        E["I0"][w] = float(0.5 * (sig(b + lab) + sig(b - lab)))
        E["I1"][w] = sig(a * L + pers + b)
        E["I2"][w] = sig(a * L - pers + b)
    return E


ARMS = ["M3_N10_p0.2", "M3_N40_p0.1", "M3_N20_p0.1_w", "M4_sim_nu0", "M4_obs_nu12", "M5_N10_p0.6",
        "M6_N10_n1", "M7_N10"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--groups", type=int, default=200)
    a = ap.parse_args()
    base, extra = cz.PRIMARY, "adaption:deepseek-v4-flash"
    E = {m: exact_tables(m) for m in (base, extra)}
    omegas = {m: cz.frozen_omegas(m) for m in (base, extra)}
    rng = np.random.default_rng(1)
    bad = []
    for aid in ARMS:
        ex = extra if aid.startswith("M7") else None
        recs = [cz.run_group(aid, base, s, extra_model=ex) for s in range(1, a.groups + 1)]
        outs = [cz.outcomes(r) for r in recs]
        ag, spec = cf.agents_for(aid, base, ex, omegas)
        sim = cf.simulate(ag, E, 20000, rng, protocol=spec["protocol"], start=spec["start"], claims=spec["claims"])
        line = [aid]
        for key in ("consensusA", "majA", "cohesion"):
            e = float(np.mean([o[key] for o in outs])); s = float(sim[key].mean())
            se = float(np.std([o[key] for o in outs]) / math.sqrt(len(outs))) + 1e-3
            z = (e - s) / se
            line.append(f"{key} engine {e:.3f} sim {s:.3f} z {z:+.1f}")
            if abs(z) > 3.5:
                bad.append((aid, key, round(z, 1)))
        print(" | ".join(line), flush=True)
    print("OFFLINE CHECK", "PASSED" if not bad else f"FAILED {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
