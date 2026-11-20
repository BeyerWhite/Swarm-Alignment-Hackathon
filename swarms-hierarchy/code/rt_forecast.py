"""Response-threshold study: kernel fit and the frozen population forecast.

Kernel (primary, "full"): multinomial logit over {REST, A, B} (a type with no open task is not available), REST = 0,

  u_j = a + a0*[K=0] + n*ln s_j + kappa*h_j - omega*h_-j + rho*L + b1*[j listed first]
        + sig*(others' share of j - 1/2) + g*ln(agents still to move) + lam*(t/T)

s_j: open tasks of type j; h_j, h_-j: own turns on j and on the other type among the last L <= 8 shown (L = 0 when
K = 0). This is the Bonabeau-Theraulaz model: P(work on j alone) = s^n / (s^n + theta^n), with ln theta_j lowered by
kappa/n for each past turn on j instead of resting (practice) and raised by omega/n for each past turn on the other
type instead of resting (disuse through the other task). With a full window the Theraulaz xi (practice) and phi
(disuse, any turn not on j) enter only as kappa = n(xi + phi_rest), so kappa, omega and rho are what is identified.
Type reinforcement tau = kappa + omega: one past turn moved from the other type to j raises the log-odds of j over the
other type by 2 tau; it is what makes identical agents specialize. Fitted by maximum likelihood with a ridge of 0.01 on
all coefficients but the two intercepts. No cell-level smoothing toward 1/2 (the paper's lesson).

Variants fitted to the same rows:
  histblind  kappa = omega = 0 (the history-blind null: thresholds fixed, as the status-blind null in the paper)
  position   work/rest: logit in [1, K=0, ln(sA + sB), ln(still to move), t/T]; type: the first listed type with a
             fitted probability (foraging for work by position; Tofts & Franks 1992)
  flex       full + separate n for K = 0 + h_j^2 (sensitivity version)
  linear     full with s_j instead of ln s_j (only for H1)

Answers: a simulated DO is correct with the model's measured accuracy; a turn is invalid with its measured
invalid/stale rate. Forecast: for each of B bootstrap refits (rows resampled within cells; replay rows within states)
every registered RT3/RT4 arm is simulated through rt.run_pop on the 8 population seeds; the point is the median of the
8-seed means and the interval the 5th-95th percentile.

  rt_forecast.py fit                     print fits
  rt_forecast.py freeze --B 200 [--procs 8]   write results/rt_predictions.json once
"""
import argparse, hashlib, json, os, random, sys, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
import rt
from rt import TYPES, R

KERNEL, CALIB = R / "rt_kernel.jsonl", R / "rt_calib.jsonl"
NAMES = ["a", "a0", "n", "kappa", "omega", "rho", "b1", "sig", "g", "lam"]
RIDGE = 0.01
METRICS = ("dol", "dol_excess", "gini_work", "persist", "work_rate", "mean_waiting", "points")
RM_METRICS = ("a_share_pre", "a_share_post", "spec_pre", "spec_post", "spec_gain", "waitA_pre", "waitA_peak", "waitA_post")


# ----------------------------------------------------------------------------------------------- data

def load_rows(model):
    c = rt.chosen_c()
    rows = [r for r in map(json.loads, open(KERNEL)) if r["model"] == model]
    if CALIB.exists():
        rows += [r for r in map(json.loads, open(CALIB)) if r["model"] == model and abs(r["c"] - c) < 1e-9]
    return rows


def answer_rates(rows):
    do = [r for r in rows if r["action"] == "do"]
    acc = float(np.mean([r["correct"] for r in do])) if do else 1.0
    inv = float(np.mean([r["action"] in ("invalid", "stale") for r in rows])) if rows else 0.0
    return acc, inv


def cluster(r):
    return f"{r['set']}|{r.get('state', r['cell'])}"


def design(rows, variant="full"):
    """Arrays: X[i, j, :] features of alternative j (0 = A, 1 = B), avail[i, j], y[i] in {0 rest, 1 A, 2 B}."""
    rows = [r for r in rows if r["action"] in ("do", "rest")]
    n = len(rows)
    nf = {"full": 10, "histblind": 10, "flex": 12, "linear": 10}[variant]
    X = np.zeros((n, 2, nf)); av = np.zeros((n, 2), bool); y = np.zeros(n, int)
    for i, r in enumerate(rows):
        L = r["L"]; rem = r["n"] - r["pos"] + 1; k0 = float(r["K"] == 0)
        for j, t in enumerate(TYPES):
            s = r["s" + t]
            if s <= 0:
                continue
            av[i, j] = True
            h, ho = r["h" + t], r["h" + ("B" if t == "A" else "A")]
            share = r["othA"] if t == "A" else 1 - r["othA"]
            ls = s if variant == "linear" else np.log(s)
            f = [1.0, k0, ls, h, -ho, float(L), float(r["first"] == t), share - 0.5, np.log(rem), r["t"] / r["T"]]
            if variant == "flex":
                f += [k0 * np.log(s), h * h]
            X[i, j] = f
        y[i] = 0 if r["action"] == "rest" else 1 + TYPES.index(r["type"])
    return X, av, y


def nll(theta, X, av, y, free):
    th = np.zeros(X.shape[2]); th[free] = theta
    u = X @ th                                         # (n, 2)
    u = np.where(av, u, -np.inf)
    m = np.maximum(u.max(1), 0.0)
    ex = np.where(av, np.exp(u - m[:, None]), 0.0)
    den = np.exp(-m) + ex.sum(1)
    lp = np.log(den) + m
    uy = np.where(y == 0, 0.0, u[np.arange(len(y)), np.clip(y - 1, 0, 1)])
    ll = uy - lp
    p = ex / den[:, None]                              # (n, 2) probabilities of A, B
    ind = np.stack([(y == 1), (y == 2)], 1).astype(float)
    grad = -((ind - p)[:, :, None] * X).sum((0, 1))
    pen = np.ones(X.shape[2]) * RIDGE; pen[:2] = 0.0
    val = -ll.sum() + (pen * th * th).sum()
    grad = grad + 2 * pen * th
    return val, grad[free]


def fit(rows, variant="full"):
    X, av, y = design(rows, variant)
    free = np.arange(X.shape[2])
    if variant == "histblind":
        free = np.array([i for i in free if NAMES[i] not in ("kappa", "omega")])
    res = minimize(nll, np.zeros(len(free)), args=(X, av, y, free), jac=True, method="L-BFGS-B")
    th = np.zeros(X.shape[2]); th[free] = res.x
    k = len(free)
    return dict(variant=variant, theta=th.tolist(), nll=float(res.fun), k=k, aic=2 * k + 2 * float(res.fun),
                n_rows=int(len(y)))


def fit_position(rows):
    """Position null: logistic work/rest in [1, K0, ln(sA+sB), ln rem, t/T]; P(first listed | work, both open)."""
    rows = [r for r in rows if r["action"] in ("do", "rest") and r["sA"] + r["sB"] > 0]
    Z = np.array([[1.0, float(r["K"] == 0), np.log(r["sA"] + r["sB"]), np.log(r["n"] - r["pos"] + 1),
                   r["t"] / r["T"]] for r in rows])
    w = np.array([r["action"] == "do" for r in rows], float)

    def f(b):
        z = Z @ b
        ll = w * z - np.logaddexp(0, z)
        p = 1 / (1 + np.exp(-z))
        return -ll.sum() + RIDGE * (b[2:] ** 2).sum(), -(Z * (w - p)[:, None]).sum(0) + 2 * RIDGE * np.r_[0, 0, b[2:]]
    res = minimize(f, np.zeros(5), jac=True, method="L-BFGS-B")
    both = [r for r in rows if r["action"] == "do" and r["sA"] > 0 and r["sB"] > 0]
    pi = float(np.mean([r["type"] == r["first"] for r in both])) if both else 0.5
    return dict(variant="position", beta=res.x.tolist(), pi=pi, nll=float(res.fun))


# ----------------------------------------------------------------------------------------------- samplers

def probs(theta, view, variant):
    """P(rest), P(A), P(B) for a view under a fitted kernel."""
    if variant == "position":
        b, pi = theta["beta"], theta["pi"]
        stot = view["nA"] + view["nB"]
        if stot == 0:
            return 1.0, 0.0, 0.0
        z = (b[0] + b[1] * (view["K"] == 0) + b[2] * np.log(stot) + b[3] * np.log(view["n"] - view["pos"] + 1)
             + b[4] * view["t"] / view["T"])
        pw = 1 / (1 + np.exp(-z))
        if view["nA"] and view["nB"]:
            pa = pi if view["type_order"][0] == "A" else 1 - pi
        else:
            pa = 1.0 if view["nA"] else 0.0
        return 1 - pw, pw * pa, pw * (1 - pa)
    th = np.asarray(theta["theta"])
    me = view["me"]
    h = view["hist"][me][-rt.HWIN:] if view["K"] > 0 else []
    L = len(h)
    oth = [a for a in view["stats"] if a != me]
    dA = sum(view["stats"][a]["done"]["A"] for a in oth); dB = sum(view["stats"][a]["done"]["B"] for a in oth)
    othA = dA / (dA + dB) if dA + dB else 0.5
    if view["others"] != "shown":
        othA = 0.5
    ex = []
    for t in TYPES:
        s = view["n" + t]
        if s <= 0:
            ex.append(0.0); continue
        hj, ho = h.count(t), h.count("B" if t == "A" else "A"); share = othA if t == "A" else 1 - othA
        f = [1.0, float(view["K"] == 0), np.log(s), hj, -ho, float(L), float(view["type_order"][0] == t), share - 0.5,
             np.log(view["n"] - view["pos"] + 1), view["t"] / view["T"]]
        if len(th) == 12:
            f += [float(view["K"] == 0) * np.log(s), hj * hj]
        ex.append(float(np.exp(np.clip(np.dot(th, f), -50, 50))))
    den = 1.0 + sum(ex)
    return 1 / den, ex[0] / den, ex[1] / den


def make_sampler(theta, variant, acc, inv, seed):
    rng = random.Random(seed)

    def decide(view):
        if rng.random() < inv:
            return "??"
        pr, pa, pb = probs(theta, view, variant)
        u = rng.random()
        if u < pr:
            return "REST"
        j = "A" if u < pr + pa else "B"
        x = next(x for x in view["open"] if x["type"] == j)
        return f"DO T{x['id']}: {x['answer'] if rng.random() < acc else 'x'}"
    return decide


# ----------------------------------------------------------------------------------------------- simulation

def all_arms():
    return [(st, arm) for st in ("RT3", "RT4") for arm in rt.arms(st)]


def simulate(theta, variant, acc, inv, c, b, n_perm=50):
    """One bootstrap draw: every arm, 8 seeds -> {arm_key: {metric: 8-seed mean}}."""
    out = {}
    for st, arm in all_arms():
        rows = []
        for s in rt.POP_SEEDS:
            dec = make_sampler(theta, variant, acc, inv, seed=zlib.crc32(f"{variant}|{b}|{rt.arm_key(arm)}|{s}".encode()))
            rec = rt.run_pop("sim", c=c, seed=s, decide=dec, **arm)
            m = rt.metrics(rec, n_perm=n_perm)
            if arm.get("remove_at"):
                m.update(rt.removal_metrics(rec))
            rows.append(m)
        out[rt.arm_key(arm)] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    return out


def boot_rows(rows, rng):
    by = {}
    for r in rows:
        by.setdefault(cluster(r), []).append(r)
    out = []
    for v in by.values():
        out += [v[rng.randrange(len(v))] for _ in v]
    return out


def _job(args):
    model, b, c, variants = args
    rows = load_rows(model)
    acc, inv = answer_rates(rows)
    rr = rows if b < 0 else boot_rows(rows, random.Random(1000 + b))
    res = {}
    for v in variants:
        th = fit_position(rr) if v == "position" else fit(rr, v)
        res[v] = dict(params=th, sim=simulate(th, v, acc, inv, c, b))
    return model, b, res


def summarize(draws):
    """draws: list of {arm: {metric: value}} -> {arm: {metric: {median, lo, hi, mean, sd}}}."""
    out = {}
    for arm in draws[0]:
        out[arm] = {}
        for k in draws[0][arm]:
            v = np.array([d[arm][k] for d in draws], float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            out[arm][k] = dict(median=float(np.median(v)), lo=float(np.percentile(v, 5)),
                               hi=float(np.percentile(v, 95)), mean=float(v.mean()), sd=float(v.std(ddof=1)))
    return out


def cmd_fit(a):
    for m in a.models:
        rows = load_rows(m)
        print(f"== {m}: {len(rows)} rows, accuracy/invalid {answer_rates(rows)}")
        for v in ("full", "histblind", "flex", "linear"):
            f = fit(rows, v)
            print(f"  {v:9s} AIC {f['aic']:.1f}  " + "  ".join(f"{k} {x:+.3f}" for k, x in zip(NAMES, f["theta"])))
        print("  position ", fit_position(rows))


def cmd_freeze(a):
    out_path = rt.FORECAST
    if out_path.exists():
        sys.exit("forecast already frozen; refusing to overwrite")
    rt.stamp(rt.PREREG)
    import rt_kernel
    models = rt_kernel.gated(a.models)
    c = rt.chosen_c()
    variants = ("full", "histblind", "position", "flex")
    jobs = [(m, b, c, variants) for m in models for b in [-1] + list(range(a.B))]
    with Pool(a.procs) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    out = dict(created=__import__("time").strftime("%F %T"), c=c, B=a.B, models=models, seeds=list(rt.POP_SEEDS),
               variants=list(variants), fits={}, forecasts={}, params_boot={}, answer_rates={})
    for m in models:
        rows = load_rows(m)
        out["answer_rates"][m] = dict(zip(("acc", "inv"), answer_rates(rows)))
        point = next(r for (mm, b, r) in res if mm == m and b == -1)
        boots = [r for (mm, b, r) in res if mm == m and b >= 0]
        out["fits"][m] = {v: point[v]["params"] for v in variants}
        out["fits"][m]["linear"] = fit(rows, "linear")
        out["forecasts"][m] = {v: summarize([bb[v]["sim"] for bb in boots]) for v in variants}
        out["params_boot"][m] = {v: [bb[v]["params"] for bb in boots] for v in variants if v != "position"}
    # RT4 model: the cheapest gated model whose tau = kappa + omega bootstrap 5% quantile is above 0
    # (Qwen3.8-27B before Gemma-4-31B)
    rt4 = None
    for m in [x for x in ("adaption:qwen3.8-27b", "adaption:gemma-4-31b-it") if x in models]:
        xs = [p["theta"][NAMES.index("kappa")] + p["theta"][NAMES.index("omega")] for p in out["params_boot"][m]["full"]]
        if np.percentile(xs, 5) > 0:
            rt4 = m; break
    out["rt4_model"] = rt4
    body = json.dumps(out, sort_keys=True)
    out["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    with open(out_path, "x") as f:
        json.dump(out, f)
    print(f"frozen {out_path} sha256 {out['sha256']} file {hashlib.sha256(out_path.read_bytes()).hexdigest()}; "
          f"RT4 model {rt4}")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit"); f.add_argument("--models", nargs="+", default=list(rt.MODELS))
    z = sp.add_parser("freeze"); z.add_argument("--models", nargs="+", default=list(rt.MODELS))
    z.add_argument("--B", type=int, default=200); z.add_argument("--procs", type=int, default=8)
    a = p.parse_args()
    {"fit": cmd_fit, "freeze": cmd_freeze}[a.cmd](a)


if __name__ == "__main__":
    main()
