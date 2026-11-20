"""Shared helpers for the amendment C10 analyses (E11-E14). No model calls.

  load(file, model, framing, N, seed0, plan)  -> {arm: [records in seed order]}; asserts every planned group is there
  feats(arm, me, other, own, partner, t_now)  -> every feature an agent could read off its prompt:
        d, o, p, tc, n_own, n_par            chase_prompts.features (unchanged)
        hinge, last, streak, idlow, idme     chase_e6a.extras (unchanged)
        plast, pstreak                       the partner's most recent outcome (+1 won, -1 lost, 0 none) and signed run
        phinge = min(p + 3, 0)               a second slope for deep losing partner records
        cmp = sign(p - o)                    partner's third-party net above (+1), equal to (0) or below (-1) one's own
        tr                                   the round number shown in the prompt
  rows(rec)                -> per-decision rows rebuilt from a stored history (d, o, p, tc checked against stored values)
  fit(d, cols, rng=None)   -> (a, {col: b}) by chase_fit.fit_logistic (signed-log features, ridge 0.05), unchanged
  glmm(d, cols)            -> the same logistic with a normal random intercept per agent (Gauss-Hermite, 20 nodes)
  Decider(coef, arm, sigma, rng)  simulated agent: logit = a + sum b slog(f) + u, u ~ N(0, sigma^2) once per agent
  simulate(coef, arm, N, G, rng, sigma=0) -> list of chase_metrics.structure dicts
  group_boot(...)          group-bootstrap refits: resample groups, refit, simulate G groups, keep the headline
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
sys.path.insert(0, str(Path(__file__).parent))
import chase_fit as F
from chase_e2 import RHO, ROUNDS
from chase_e6a import extras
from chase_metrics import events_from_history, structure
from chase_pop import run
from chase_prompts import features, visible

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
QWEN, G31, G26 = "adaption:qwen3.5-397b", "adaption:gemma-4-31b-it", "adaption:gemma-4-26b-a4b-it"


def load(fname, model, framing, N, seed0, plan):
    out = {arm: {} for arm in plan}
    for line in open(R / fname):
        r = json.loads(line)
        if (r.get("model") == model and r["framing"] == framing and r["N"] == N and r["arm"] in plan
                and 0 <= r["seed"] - seed0 < plan[r["arm"]]):
            out[r["arm"]][r["seed"]] = r
    for arm in plan:
        assert len(out[arm]) == plan[arm], (fname, model, framing, arm, len(out[arm]), plan[arm])
    return {arm: [out[arm][s] for s in sorted(out[arm])] for arm in plan}


def _run(outs):
    n = 0
    for x in reversed(outs):
        if x != outs[-1]:
            break
        n += 1
    return (outs[-1] * n) if outs else 0


def feats(arm, me, other, own, partner, t_now):
    f = features(me, other, own, partner, arm)
    f.update(extras(me, other, own, f))
    pouts = [1 if w == other else -1 for _, w, _l in partner]
    f.update(plast=pouts[-1] if pouts else 0, pstreak=_run(pouts), phinge=min(f["p"] + 3, 0),
             cmp=float(np.sign(f["p"] - f["o"])), tr=t_now)
    return f


def rows(rec):
    names, arm, events, out, bad = rec["names"], rec["arm"], [], [], 0
    for x in rec["history"]:
        t_now = x["t"] + 1
        for k, (i, j, ci, cj, w) in enumerate(x["events"]):
            for side, (a, b, c) in enumerate(((i, j, ci), (j, i, cj))):
                own, partner = visible(arm, names[a], names[b], events, t_now)
                f = feats(arm, names[a], names[b], own, partner, t_now)
                bad += (f["d"], f["o"], f["p"], f["tc"]) != tuple(x["feats"][2 * k + side][:4])
                out.append(dict(f, lead=float(c == "LEAD"), seed=rec["seed"], agent=f"{rec['seed']}|{a}", off=0.0))
        events += [(t_now, names[e[4]], names[e[1] if e[4] == e[0] else e[0]]) for e in x["events"]]
    return out, bad


def frame(recs):
    out, bad = [], 0
    for r in recs:
        x, b = rows(r)
        out += x; bad += b
    return pd.DataFrame(out), bad


def fit(d, cols, rng=None):
    p = F.fit_logistic(d, cols, rng)
    return p.params["a"], {c: p.params[c] for c in cols}


def logit(coef, f):
    a, b = coef
    return a + sum(v * float(F.slog(f[c])) for c, v in b.items())


def loss(coef, d):
    z = coef[0] + F.slog(d[list(coef[1])].values.astype(float)) @ np.array(list(coef[1].values())) if coef[1] else np.full(len(d), coef[0])
    p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
    return float(-(d.lead.values * np.log(p) + (1 - d.lead.values) * np.log(1 - p)).sum())


def loso(d, cols):
    """Leave-one-seed-out log-loss per decision."""
    tot = 0.0
    for s in sorted(d.seed.unique()):
        tot += loss(fit(d[d.seed != s], cols), d[d.seed == s])
    return tot / len(d)


def glmm(d, cols, ridge=0.05, nodes=20):
    """Logistic with a N(0, sigma^2) intercept per agent; marginal likelihood by Gauss-Hermite. Same ridge on slopes."""
    X = np.c_[np.ones(len(d)), F.slog(d[cols].values.astype(float).reshape(len(d), len(cols)))]
    y = d.lead.values
    g = pd.factorize(d.agent)[0]
    G = g.max() + 1
    xk, wk = np.polynomial.hermite.hermgauss(nodes)
    lw = np.log(wk / np.sqrt(np.pi))
    pen = np.r_[0.0, np.full(len(cols), ridge)]
    w0 = np.r_[fit(d, cols)[0], list(fit(d, cols)[1].values())]

    def nll(th):
        w, s = th[:-1], np.exp(th[-1])
        Z = (X @ w)[:, None] + np.sqrt(2) * s * xk[None, :]
        ll = y[:, None] * Z - np.logaddexp(0, Z)
        S = np.zeros((G, nodes))
        np.add.at(S, g, ll)
        return -float(logsumexp(S + lw[None, :], axis=1).sum()) + float(np.sum(pen * w ** 2))

    best = min((minimize(nll, np.r_[w0, ls], method="L-BFGS-B") for ls in (np.log(0.3), np.log(1.0))), key=lambda r: r.fun)
    w, s = best.x[:-1], float(np.exp(best.x[-1]))
    z0 = X @ w0
    nll0 = float(np.sum(np.logaddexp(0, z0) - y * z0) + np.sum(pen * w0 ** 2))
    return dict(coef=(float(w[0]), dict(zip(cols, map(float, w[1:])))), sigma=s, nll=float(best.fun), nll0=nll0,
                lr=max(0.0, 2 * (nll0 - float(best.fun))), agents=int(G))


class Decider:
    """Simulated agent with the given logistic coefficients and, optionally, a per-agent normal intercept."""
    def __init__(self, coef, arm, sigma=0.0, rng=None):
        self.coef, self.arm, self.sigma, self.rng, self.u, self.invalid = coef, arm, sigma, rng, None, 0

    def __call__(self, pairs, names, events, t_now, h, N, rng_prompt, rng_sched, mu, Q, Fq):
        if self.u is None:
            self.u = self.rng.normal(0, self.sigma, N) if self.sigma > 0 else np.zeros(N)
        out = []
        for i, j in pairs:
            ps = []
            for a, b in ((i, j), (j, i)):
                own, partner = visible(self.arm, names[a], names[b], events, t_now)
                z = logit(self.coef, feats(self.arm, names[a], names[b], own, partner, t_now)) + self.u[a]
                ps.append(1 / (1 + np.exp(-z)))
            out.append(("LEAD" if rng_sched.random() < ps[0] else "DEFER", "LEAD" if rng_sched.random() < ps[1] else "DEFER"))
        return out, []


def simulate(coef, arm, N, G, rng, sigma=0.0):
    return [structure(events_from_history(run(N, RHO, ROUNDS[N], arm, Decider(coef, arm, sigma, rng),
                                              seed=int(rng.integers(1e9)), keep_feats=False)), N, ROUNDS[N]) for _ in range(G)]


def summ(v, obs):
    v = np.asarray(v, float)
    q05, q95 = float(np.quantile(v, .05)), float(np.quantile(v, .95))
    return dict(mean=float(v.mean()), q05=q05, q95=q95, observed=float(obs), covered=bool(q05 <= obs <= q95))


def group_boot(d, cols, arm, N, G, obs, reps, rng, mixed=False):
    """Resample groups (seeds) with replacement, refit (with the agent intercept if mixed), simulate G groups; linear share."""
    seeds = np.array(sorted(d.seed.unique()))
    by = {s: d[d.seed == s] for s in seeds}
    out = []
    for _ in range(reps):
        pick = rng.choice(seeds, len(seeds))
        parts = []
        for k, s in enumerate(pick):
            x = by[s].copy(); x["agent"] = f"b{k}|" + x.agent; parts.append(x)
        db = pd.concat(parts, ignore_index=True)
        if mixed:
            m = glmm(db, cols); coef, sig = m["coef"], m["sigma"]
        else:
            coef, sig = fit(db, cols), 0.0
        out.append(np.mean([s["linear"] for s in simulate(coef, arm, N, G, rng, sig)]))
    return summ(out, obs)
