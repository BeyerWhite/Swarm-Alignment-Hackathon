"""v2 population engine: Bonabeau-type coordinator contests with LLM agents or frozen kernels.

Protocol (identical for LLM and kernel runs):
  each round every agent initiates a contest with prob rho against a uniform random partner;
  all contests in a round are decided from the round-start scores (synchronous);
  both answer LEAD/DEFER; a lone LEAD wins, otherwise coin flip; coordinator +Q, other -F;
  then all scores shrink by (1 - mu). Displayed scores are rounded to 0.1.
Separate RNG streams: `sched` (pairings, coin flips, kernel draws) and `prompt` (names, tasks, order).
"""
import json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from llm import call, parse_choice
from prompts_v2 import SYSTEM, TASKS, pair_prompt

ROOT = Path(__file__).resolve().parents[1]


# ---------------- deciders ----------------
class LLMDecider:
    def __init__(self, model, cond, retries=2):
        self.model, self.cond, self.retries = model, cond, retries
        self.invalid = 0

    def __call__(self, pairs, h, names, N, rng_prompt, rng_sched, mu, Q, F):
        jobs = []
        for i, j in pairs:
            task = rng_prompt.choice(TASKS)                      # one shared task per contest
            for a, b in ((i, j), (j, i)):
                jobs.append((SYSTEM.format(me=names[a], n=N),
                             pair_prompt(names[a], names[b], round(h[a], 1), round(h[b], 1), rng_prompt,
                                         self.cond, q=Q, f=F, mu=mu, task=task)))

        def one(job):
            for _ in range(self.retries + 1):
                c = parse_choice(call(self.model, job[0], job[1], tag="population2"))
                if c:
                    return c
            return None

        with ThreadPoolExecutor(len(jobs) or 1) as ex:
            ch = list(ex.map(one, jobs))
        out = []
        for k in range(len(pairs)):
            ci, cj = ch[2 * k], ch[2 * k + 1]
            if ci is None:
                self.invalid += 1; ci = "LEAD" if rng_sched.random() < 0.5 else "DEFER"
            if cj is None:
                self.invalid += 1; cj = "LEAD" if rng_sched.random() < 0.5 else "DEFER"
            out.append((ci, cj))
        return out


class KernelDecider:
    """p(LEAD | h_self, h_other) given as a vectorisable function pfun(delta, mean)."""
    def __init__(self, pfun):
        self.pfun, self.invalid = pfun, 0

    def __call__(self, pairs, h, names, N, rng_prompt, rng_sched, mu, Q, F):
        out = []
        for i, j in pairs:
            hi, hj = round(h[i], 1), round(h[j], 1)
            m = (hi + hj) / 2
            pi, pj = self.pfun(hi - hj, m), self.pfun(hj - hi, m)
            out.append(("LEAD" if rng_sched.random() < pi else "DEFER",
                        "LEAD" if rng_sched.random() < pj else "DEFER"))
        return out


# ---------------- kernels fitted on calibration data ----------------
def kernel_flexible(df, rng=None):
    """Bilinear interpolation of Jeffreys-smoothed LEAD rates on the (displayed delta, mean) grid; clamped
    outside the grid. With `rng`, each cell rate is drawn from its Beta(k+.5, n-k+.5) posterior."""
    c = df.groupby(["delta", "mean"]).lead.agg(["sum", "count"])
    if rng is None:
        rate = (c["sum"] + 0.5) / (c["count"] + 1.0)
    else:
        rate = rng.beta(c["sum"] + 0.5, c["count"] - c["sum"] + 0.5)
        rate = c["sum"].copy().astype(float) * 0 + rate
    g = rate.unstack("mean")
    g = g.sort_index().interpolate(method="index", axis=0, limit_direction="both")
    g = g.T.sort_index().interpolate(method="index", axis=0, limit_direction="both").T
    if not np.isfinite(g.values).all():
        raise ValueError("calibration grid has empty/nonfinite cells")
    D, M, Z = np.array(g.index, float), np.array(g.columns, float), g.values

    def p(delta, mean):
        d = min(max(delta, D[0]), D[-1]); m = min(max(mean, M[0]), M[-1])
        col = np.array([np.interp(d, D, Z[:, k]) for k in range(len(M))])
        return float(np.interp(m, M, col))
    return p


def kernel_logistic(df, rng=None):
    """One-eta Bonabeau-type kernel: p = sig(a + b*delta + c*mean) (ridge-penalised logistic MLE)."""
    from scipy.optimize import minimize
    X = np.c_[np.ones(len(df)), df.delta.values, df["mean"].values]
    y = df.lead.values

    def nll(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.01 * np.sum(w[1:] ** 2)
    if rng is not None:                                    # row bootstrap
        idx = rng.integers(0, len(df), len(df)); X, y = X[idx], y[idx]
    w = minimize(nll, np.zeros(3), method="BFGS").x
    a, b, c = w
    f = lambda delta, mean: float(1 / (1 + np.exp(-(a + b * delta + c * mean))))
    f.params = (float(a), float(b), float(c))
    return f


def kernel_logistic_delta(df, rng=None):
    """Strict Bonabeau-family kernel: p(LEAD) = sig(a + b*delta), no dependence on mean status."""
    from scipy.optimize import minimize
    X = np.c_[np.ones(len(df)), df.delta.values]
    y = df.lead.values
    if rng is not None:
        idx = rng.integers(0, len(df), len(df)); X, y = X[idx], y[idx]

    def nll(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.01 * w[1] ** 2
    a, b = minimize(nll, np.zeros(2), method="BFGS").x
    f = lambda delta, mean: float(1 / (1 + np.exp(-(a + b * delta))))
    s = 1 / (1 + np.exp(-a))
    f.params = (float(a), float(b), float(4 * b * s * (1 - s)))     # last = implied local eta = 4 P'_win(0)
    return f


def kernel_null(df, rng=None):
    k, n = df.lead.sum(), len(df)
    p0 = float((k + 0.5) / (n + 1)) if rng is None else float(rng.beta(k + 0.5, n - k + 0.5))
    return lambda delta, mean: p0


def kernel_nw(df, rng=None, bw_delta=0.6, bw_mean=1.5, alpha=1.0):
    """Nadaraya-Watson smoother of LEAD over (displayed delta, mean) for scattered (grid + state-matched) data,
    shrunk toward the base rate with `alpha` pseudo-observations. With `rng`: row bootstrap."""
    D, M, Y = df.delta.values.astype(float), df["mean"].values.astype(float), df.lead.values.astype(float)
    if rng is not None:
        idx = rng.integers(0, len(D), len(D)); D, M, Y = D[idx], M[idx], Y[idx]
    p0 = (Y.sum() + 0.5) / (len(Y) + 1)
    cache = {}

    def p(delta, mean):
        key = (round(delta, 2), round(mean, 2))
        if key not in cache:
            w = np.exp(-0.5 * (((D - delta) / bw_delta) ** 2 + ((M - mean) / bw_mean) ** 2))
            cache[key] = float((w @ Y + alpha * p0) / (w.sum() + alpha))
        return cache[key]
    return p


KERNELS = {"flexible": kernel_flexible, "bonabeau": kernel_logistic_delta, "logistic3": kernel_logistic,
           "null": kernel_null, "nw": kernel_nw}


# ---------------- simulation ----------------
def run(N, rho, mu, rounds, decider, seed=0, Q=1.0, F=1.0, h0=None, meta=None, out=None):
    rng_sched, rng_prompt = random.Random(seed), random.Random(10 ** 6 + seed)
    names = rng_prompt.sample(range(1, 100), N)
    h = np.zeros(N) if h0 is None else np.array(h0, float)
    hist = []
    for t in range(rounds):
        pairs = []
        for i in range(N):
            if rng_sched.random() < rho:
                j = rng_sched.randrange(N - 1); j += j >= i
                pairs.append((i, j))
        choices = decider(pairs, h, names, N, rng_prompt, rng_sched, mu, Q, F) if pairs else []
        dh, events = np.zeros(N), []
        for (i, j), (ci, cj) in zip(pairs, choices):
            if ci == "LEAD" and cj != "LEAD":
                w = i
            elif cj == "LEAD" and ci != "LEAD":
                w = j
            else:
                w = i if rng_sched.random() < 0.5 else j
            l = j if w == i else i
            dh[w] += Q; dh[l] -= F
            events.append([i, j, ci, cj, w])
        h = (h + dh) * (1 - mu)
        hist.append(dict(t=t, h=[round(float(x), 3) for x in h], events=events))
    rec = dict(**(meta or {}), N=N, rho=rho, mu=mu, Q=Q, F=F, seed=seed, rounds=rounds, names=names,
               h0=None if h0 is None else [float(x) for x in h0], invalid=decider.invalid,
               history=hist, finished=time.time())
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


# ---------------- order parameters ----------------
def landau_h(W):
    """de Vries (1995) h' = h + 6u/(n^3-n): Landau h with observed equal-win ties as 1/2 and u unobserved
    unordered dyads contributing the expected correction."""
    n = len(W)
    off = ~np.eye(n, dtype=bool)
    seen = (W + W.T) > 0
    D = (W > W.T).astype(float) + 0.5 * ((W == W.T) & off)       # ties and unknown dyads: 1/2 each
    V = D.sum(1)
    u = int(((~seen) & off).sum() // 2)
    return 12 / (n ** 3 - n) * np.sum((V - (n - 1) / 2) ** 2) + 6 * u / (n ** 3 - n)


def order_params(rec, window=20):
    hist = rec["history"][-window:]
    N = rec["N"]
    H = np.array([x["h"] for x in hist])
    W = np.zeros((N, N)); wins = np.zeros(N)
    for x in hist:
        for i, j, ci, cj, w in x["events"]:
            l = j if w == i else i
            W[w, l] += 1; wins[w] += 1
    tot = H.var(axis=1).mean()
    return dict(
        sd_h=float(H.std(axis=1).mean()),
        q=float(H.mean(axis=0).var() / tot) if tot > 0 else 0.0,
        leader_share=float(wins.max() / max(wins.sum(), 1)),
        rank_persist=float(spearmanr(H[0], H[-1])[0]) if H[0].std() > 0 and H[-1].std() > 0 else 0.0,
        landau=float(landau_h(W)),
    )


METRICS = ["sd_h", "q", "leader_share", "rank_persist", "landau"]
