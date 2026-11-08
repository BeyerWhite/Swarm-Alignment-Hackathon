"""X2 reference: Ben-Naim-Redner master equation vs a discrete-round agent simulator (no LLM calls).

Master equation (time unit: each agent takes part in one contest on average):
    dF_k/dt = r (F_{k+1} - F_k) + F_k (F_{k-1} - F_k),   F_k(0) = 1.
Discrete simulator: each round = random perfect matching (every agent in exactly one contest); the fitter agent
wins (+1, loser unchanged; ties coin flip); with prob q the less fit agent wins instead (upset, BVR 2006);
then each agent loses Poisson(r) points, floored at 0. One round = one unit of master-equation time.

Lower-class fraction is measured as L(x) = F(k <= x t); BNR scaling gives Phi(x) = r + x for x < 1 - r (r < 1).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def master(r, T, K=None, dt=0.01):
    K = K or int(1.5 * T) + 50
    F = np.ones(K + 2)                        # F[k] for k = 0..K+1 ; F_{-1} = 0
    out = {}
    for step in range(int(T / dt)):
        Fm1 = np.r_[0.0, F[:-1]]
        Fp1 = np.r_[F[1:], 1.0]
        dF = r * (Fp1 - F) + F * (Fm1 - F)
        F = np.clip(F + dt * dF, 0, 1); F[-1] = 1.0
    return F[:-1]


def simulate(N, r, T, q=0.0, seed=0, decider=None):
    """decider(k_i, k_j, rng) -> bool array 'i wins' (overrides fitter-wins + q); used for LLM-kernel surrogates."""
    rng = np.random.default_rng(seed)
    k = np.zeros(N, int)
    traj = []
    for t in range(T):
        perm = rng.permutation(N)
        i, j = perm[0::2][: N // 2], perm[1::2][: N // 2]
        if decider is None:
            iw = np.where(k[i] == k[j], rng.random(len(i)) < .5, k[i] > k[j])
            upset = (k[i] != k[j]) & (rng.random(len(i)) < q)
            iw = iw ^ upset
        else:
            iw = decider(k[i], k[j], rng)
        w = np.where(iw, i, j)
        k[w] += 1
        k = np.maximum(k - rng.poisson(r, N), 0)
        traj.append(k.copy())
    return np.array(traj)


def L_of(k, t, x=0.1):
    return float((k <= max(1, x * t)).mean())


if __name__ == "__main__":
    T = 200; res = []
    print("r     master F(0.1t)  theory r+0.1   sim N=700 L(0.1)   sim kmax/t (theory (1-r)+sqrt(ln N/t))")
    for r in (0.25, 0.5, 0.75, 0.9, 1.25):
        F = master(r, T)
        sim = simulate(700, r, T, seed=1)
        th = min(r + 0.1, 1.0)
        kmax_th = max(1 - r, 0) + np.sqrt(np.log(700) / T) if r < 1 else 0
        print(f"{r:<5} {F[int(0.1*T)]:.3f}           {th:.3f}          {L_of(sim[-1], T):.3f}            "
              f"{sim[-1].max()/T:.3f} ({kmax_th:.3f})")
        res.append(dict(r=r, master_F01=float(F[int(0.1 * T)]), theory=th, sim=L_of(sim[-1], T)))
    print("\nupset probability q shifts L (N=700, T=200):")
    for q in (0.0, 0.05, 0.1, 0.2):
        print(" q", q, [(r, round(L_of(simulate(700, r, T, q=q, seed=2)[-1], T), 2)) for r in (0.25, 0.5, 0.75, 1.0)])
    print("\nN=16 quantisation (T=30, 200 seeds) — the LLM-scale reference:")
    for r in (0.25, 0.5, 1.25):
        Ls = [L_of(simulate(16, r, 30, seed=s)[-1], 30) for s in range(200)]
        print(f" r={r}: L mean {np.mean(Ls):.2f}, 90% [{np.quantile(Ls,.05):.2f}, {np.quantile(Ls,.95):.2f}]")
    json.dump(res, open(ROOT / "results" / "x2_reference.json", "w"))
