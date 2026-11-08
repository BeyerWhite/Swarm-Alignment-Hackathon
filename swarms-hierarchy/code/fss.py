"""Finite-size scaling with frozen pairwise kernels (no LLM calls).

Vectorised version of the population2 protocol (synchronous rounds, 0.1-rounded display, lone-LEAD wins,
coin-flip ties, +1/-1, decay mu). Sweeps rho and the norm weight w (kernel interpolated linearly between
calibrated w levels) for N in {8..700}. Order parameters over the last `window` rounds:
  sd_h  status dispersion;  q  frozen-order fraction;  top  leader's share of wins.
Susceptibility chi = N * Var_seeds(q). Output: results/fss.parquet-like JSONL.
"""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]


def grid_table(d):
    """Jeffreys-smoothed LEAD-rate table on the (delta, mean) calibration grid (same as kernel_flexible)."""
    c = d.groupby(["delta", "mean"]).lead.agg(["sum", "count"])
    rate = ((c["sum"] + 0.5) / (c["count"] + 1.0)).unstack("mean").sort_index()
    rate = rate.interpolate(method="index", axis=0, limit_direction="both")
    rate = rate.T.sort_index().interpolate(method="index", axis=0, limit_direction="both").T
    return np.array(rate.index, float), np.array(rate.columns, float), rate.values


def bonabeau_params(d):
    from population2 import kernel_logistic_delta
    return kernel_logistic_delta(d).params[:2]


class VecKernel:
    """kind='flexible': bilinear on grid (clamped); kind='bonabeau': sig(a + b delta)."""
    def __init__(self, kind, spec):
        self.kind = kind
        if kind == "flexible":
            D, M, Z = spec
            self.D, self.M = D, M
            self.f = RegularGridInterpolator((D, M), Z, bounds_error=False, fill_value=None)
        else:
            self.a, self.b = spec

    def __call__(self, delta, mean):
        if self.kind == "flexible":
            pts = np.c_[np.clip(delta, self.D[0], self.D[-1]), np.clip(mean, self.M[0], self.M[-1])]
            return np.clip(self.f(pts), 0, 1)
        return 1 / (1 + np.exp(-(self.a + self.b * delta)))


def simulate(kernel, N, rho, mu=0.1, rounds=120, window=40, seed=0, h0=None):
    rng = np.random.default_rng(seed)
    h = np.zeros(N) if h0 is None else np.array(h0, float)
    H, wins = [], np.zeros(N)
    for t in range(rounds):
        init = np.nonzero(rng.random(N) < rho)[0]
        if len(init):
            j = rng.integers(0, N - 1, len(init)); j = j + (j >= init)
            disp = np.round(h, 1)
            hi, hj = disp[init], disp[j]
            m = (hi + hj) / 2
            ci = rng.random(len(init)) < kernel(hi - hj, m)
            cj = rng.random(len(init)) < kernel(hj - hi, m)
            coin = rng.random(len(init)) < 0.5
            i_wins = (ci & ~cj) | (~(ci ^ cj) & coin)
            w = np.where(i_wins, init, j); l = np.where(i_wins, j, init)
            dh = np.zeros(N); np.add.at(dh, w, 1.0); np.add.at(dh, l, -1.0)
            h = (h + dh) * (1 - mu)
            if t >= rounds - window:
                np.add.at(wins, w, 1.0)
        else:
            h = h * (1 - mu)
        if t >= rounds - window:
            H.append(h.copy())
    H = np.array(H)
    tot = H.var(axis=1).mean()
    return dict(sd_h=float(H.std(axis=1).mean()), q=float(H.mean(0).var() / tot) if tot > 0 else 0.0,
                top=float(wins.max() / max(wins.sum(), 1)), final_h=None)


def _job(args):
    kind, spec, N, rho, seed, mu, rounds, window = args
    o = simulate(VecKernel(kind, spec), N, rho, mu, rounds, window, seed)
    o.pop("final_h")
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--Ns", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256, 700])
    ap.add_argument("--rhos", nargs="+", type=float, default=list(np.round(np.linspace(0.05, 1.0, 20), 3)))
    ap.add_argument("--ws", nargs="+", type=float, default=[0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--kinds", nargs="+", default=["flexible", "bonabeau"])
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--sweep", choices=["rho_at_w", "w_at_rho"], default="w_at_rho")
    ap.add_argument("--fixed", type=float, default=0.5, help="rho (w_at_rho) or w (rho_at_w)")
    ap.add_argument("--out", default=str(ROOT / "results" / "fss.jsonl"))
    a = ap.parse_args()
    import pandas as pd
    from predict import load_calib
    calib = load_calib()
    jobs, meta = [], []
    for model in a.models:
        dm = calib[calib.model == model]
        levels = sorted(int(c[4:]) for c in dm.cond.unique() if c.startswith("norm"))
        tabs = {w: grid_table(dm[dm.cond == f"norm{w}"]) for w in levels}
        bps = {w: bonabeau_params(dm[dm.cond == f"norm{w}"]) for w in levels}
        ws = a.ws if a.sweep == "w_at_rho" else [a.fixed]
        rhos = [a.fixed] if a.sweep == "w_at_rho" else a.rhos
        for w in ws:
            if w < levels[0] or w > levels[-1]:
                continue
            lo = max(x for x in levels if x <= w); hi = min(x for x in levels if x >= w)
            lam = 0.0 if hi == lo else (w - lo) / (hi - lo)
            D, M, Zlo = tabs[lo]; D2, M2, Zhi = tabs[hi]
            assert np.allclose(D, D2) and np.allclose(M, M2)
            specs = {"flexible": (D, M, (1 - lam) * Zlo + lam * Zhi),
                     "bonabeau": tuple((1 - lam) * np.array(bps[lo]) + lam * np.array(bps[hi]))}
            for kind in a.kinds:
                for N in a.Ns:
                    for rho in rhos:
                        for s in range(a.seeds):
                            jobs.append((kind, specs[kind], N, rho, 1000 * N + s, a.mu, a.rounds, a.window))
                            meta.append(dict(model=model, kind=kind, w=w, N=N, rho=rho, seed=s, mu=a.mu,
                                             rounds=a.rounds, window=a.window,
                                             bonabeau_ab=list(map(float, specs["bonabeau"]))))
    print(f"{len(jobs)} simulations", flush=True)
    with ProcessPoolExecutor(8) as ex, open(a.out, "a") as f:
        for m, o in zip(meta, ex.map(_job, jobs, chunksize=4)):
            f.write(json.dumps({**m, **o}) + "\n")


if __name__ == "__main__":
    main()
