"""Fit the R1 decision kernel per model and freeze population forecasts for R2 (write-once, sha256-stamped).

Kernel components, all fitted on R1 single-decision data only (results/rosen_kernel.jsonl, reasoning 'none'):
  start     P(start a workbench | none held or in progress): ridge logistic per m on [1, F/16, H/16] (+ k/7, the number
            of others visibly building, fitted on m = 2 core + crowd cells and shared by every m)          PRIMARY
            alternative: per-(m, H) cell means, PAVA-decreasing in F, interpolated in F and H         SENSITIVITY
  type      which type is started: base shares from core-cell builds (+1 smoothing) times exp(-gamma * n_j / q_j), n_j =
            others visibly building or holding type j, q_j = its arrivals per round (4, 2, 2); gamma by maximum
            likelihood on the crowd cells (crowded type A or B)
  cont      P(keep building | k of F built): ridge logistic per m on cont cells; m = 2 on [1, F/16, k/16, H/16] (also
            used for m = 3), m = 1 on [1, F/16, k/16] (its cells all start in round 1, so H is not separable from k)
  second    P(build again | a workbench is ready): cont-free cell means by F (second cells), linear in F, clamped
  use       P(do an own-type task | own workbench ready and such a task open): pooled (use cells)
  do        P(pick the oldest open task) and accuracy, from DO answers in core cells; invalid rate from core cells
Hidden arm and autarky: the same kernel with nothing visible about others (k = 0, n_j = 0). Autarky is an
extrapolation from team states and is labelled so.

Forecast: B bootstrap draws (cells resampled within cell), each refitted and simulated through the real engine
(rosen.run_pop with a kernel policy) for the 8 registered seeds; point = median of the 8-seed means, interval =
5th-95th percentile. Rational benchmark (rosen.policy_rational) stored beside it.

Usage: rosen_forecast.py freeze --models ... [--B 200]      -> results/rosen_r2_predictions.json (refuses to overwrite)
       rosen_forecast.py show --models ...                   (fit and print, no file)
"""
import argparse, hashlib, json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
import rosen
from rosen import TYPES, T, run_pop, metrics

R = rosen.R
PRED = R / "rosen_r2_predictions.json"
SEEDS = list(range(41001, 41009))
ARMS = ([("team", 2.0, F) for F in (1, 3, 5, 7, 9, 11, 13)] + [("team", 1.0, F) for F in (3, 7)] +
        [("hidden", 2.0, F) for F in (3, 7)] + [("autarky", 2.0, F) for F in (1, 3, 5, 7, 9, 11, 13)] +
        [("autarky", 3.0, F) for F in (1, 3)])
METRICS = ("dol", "points", "build_rate", "ready_rate", "benches", "multi_ready", "one_bench", "top_share",
           "use_share", "coverage", "dup", "alloc_l1")
RIDGE = 0.01


def arm_key(arm, m, F):
    return f"{arm}|m={m:g}|F={F}"


def ridge_logit(X, y, w=None, lam=RIDGE):
    X, y = np.asarray(X, float), np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    def nll(b):
        z = X @ b
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)) + lam * np.sum(b[1:] ** 2))
    def grad(b):
        p = 1 / (1 + np.exp(-(X @ b)))
        g = X.T @ (w * (p - y)); g[1:] += 2 * lam * b[1:]
        return g
    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x


def sig(z):
    return 1 / (1 + np.exp(-z))


def pava_dec(y, w):
    """Weighted pool-adjacent-violators for a non-increasing fit."""
    y, w = list(map(float, y)), list(map(float, w))
    blocks = [[y[i], w[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0] - 1e-12:
            a, b = blocks[i], blocks[i + 1]
            ww = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / ww if ww else 0.0, ww, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return [b[0] for b in blocks for _ in range(b[2])]


def load(models):
    rows = [json.loads(l) for l in open(R / "rosen_kernel.jsonl")]
    return [r for r in rows if r["model"] in models and r.get("reasoning", "none") == "none"]


def resample(rows, rng):
    by = defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r)
    out = []
    for v in by.values():
        out += [v[i] for i in rng.integers(0, len(v), len(v))]
    return out


def fit(rows):
    """All kernel components from one model's rows."""
    th = {}
    core = [r for r in rows if r["set"] == "core"]
    crowd = [r for r in rows if r["set"] == "crowd"]
    # start: per m, logistic on [1, F/16, H/16]; k effect from m = 2 core + crowd
    for m in (1.0, 2.0, 3.0):
        rr = [r for r in core if r["m"] == m] + ([r for r in crowd] if m == 2.0 else [])
        X = [[1, r["F"] / 16, r["H"] / 16, r.get("k", 0) / 7] for r in rr]
        y = [r["action"] == "build" for r in rr]
        b = ridge_logit(X, y) if rr else np.zeros(4)
        th[f"start_m{m:g}"] = b.tolist()
    th["start_k"] = th["start_m2"][3]
    # isotonic alternative: per (m, H) means over F, PAVA decreasing
    iso = {}
    for m in (1.0, 2.0, 3.0):
        for H in (6, 11, 16):
            Fs = sorted({r["F"] for r in core if r["m"] == m and r["H"] == H})
            ys = [np.mean([r["action"] == "build" for r in core if r["m"] == m and r["H"] == H and r["F"] == F]) for F in Fs]
            ws = [sum(1 for r in core if r["m"] == m and r["H"] == H and r["F"] == F) for F in Fs]
            iso[f"{m:g}|{H}"] = [Fs, pava_dec(ys, ws)] if Fs else [[], []]
    th["iso"] = iso
    # type choice
    bt = [r.get("type") for r in core if r["action"] == "build"]
    base = np.array([bt.count(j) + 1.0 for j in TYPES]); base /= base.sum()
    th["type_base"] = base.tolist()
    cb = [r for r in crowd if r["action"] == "build"]
    best, bg = None, 0.0
    for g in np.arange(-5, 10.001, 0.02):
        ll = 0.0
        for r in cb:
            nj = np.array([r["k"] * (j == r.get("ctype", "A")) / rosen.ARRIVE[j] for j in TYPES])
            wts = base * np.exp(-g * nj)
            ll += np.log(wts[TYPES.index(r["type"])] / wts.sum())
        if best is None or ll > best:
            best, bg = ll, float(g)
    th["gamma"] = bg if cb else 0.0
    # continuation
    cont = [r for r in rows if r["set"] == "cont"]
    for m, feats in ((2.0, lambda r: [1, r["F"] / 16, r["k"] / 16, r["H"] / 16]), (1.0, lambda r: [1, r["F"] / 16, r["k"] / 16, 0.0])):
        cc = [r for r in cont if r["m"] == m]
        th[f"cont_m{m:g}"] = (ridge_logit([feats(r) for r in cc], [r["action"] == "build" and r.get("type") == "A" for r in cc])
                              .tolist() if cc else [0.0] * 4)
    # second workbench
    sec = [r for r in rows if r["set"] == "second"]
    Fs = sorted({r["F"] for r in sec})
    th["second"] = [Fs, [float(np.mean([r["action"] == "build" for r in sec if r["F"] == F])) for F in Fs]]
    # use
    use = [r for r in rows if r["set"] == "use" and r["action"] == "do"]
    th["use"] = float(np.mean([r.get("type") == "B" for r in use])) if use else 1.0
    # DO choice, accuracy, invalid
    dos = [r for r in core if r["action"] == "do" and r.get("type")]
    th["oldest"] = float(np.mean([r["oldest"] for r in dos])) if dos else 0.5
    th["accuracy"] = float(np.mean([r["correct"] for r in dos])) if dos else 1.0
    th["invalid"] = float(np.mean([r["action"] == "invalid" for r in core])) if core else 0.0
    return th


def p_start(th, F, H, m, k, kernel="logit"):
    if kernel == "iso":
        tab = {}
        for key, (Fs, ps) in th["iso"].items():
            mm, HH = key.split("|")
            if float(mm) == m and Fs:
                tab[int(HH)] = (Fs, ps)
        Hs = sorted(tab)
        if not Hs:
            return 0.0
        Hc = min(max(H, Hs[0]), Hs[-1])
        lo = max(h for h in Hs if h <= Hc); hi = min(h for h in Hs if h >= Hc)
        f = lambda h: float(np.interp(F, *tab[h]))
        p = f(lo) if lo == hi else f(lo) + (f(hi) - f(lo)) * (Hc - lo) / (hi - lo)
        p = min(1 - 1e-6, max(1e-6, p))
        return float(sig(np.log(p / (1 - p)) + th["start_k"] * k / 7))        # same crowding shift as the logistic
    b = th[f"start_m{m:g}"]
    return float(sig(b[0] + b[1] * F / 16 + b[2] * H / 16 + th["start_k"] * k / 7))


def make_policy(th, rng, kernel="logit"):
    def ans(x):
        return x["answer"] if rng.random() < th["accuracy"] else "0"
    def do_general(view):
        op = view["open"]
        if not op:
            return "nothing to do"
        x = op[0] if rng.random() < th["oldest"] else op[rng.integers(len(op))]
        return f"DO T{x['id']}: {ans(x)}"
    def pol(view):
        if rng.random() < th["invalid"]:
            return "?"
        F, m, me = view["F"], view["m"], view["me"]
        H = view["T"] - view["t"] + 1
        st = view["stats"][me]
        ready = [j for j in TYPES if st["bench"][j] >= F]
        prog = [j for j in TYPES if 0 < st["bench"][j] < F]
        visible = view["arm"] == "team"
        n = {j: sum(1 for a, s in view["stats"].items() if a != me and s["bench"][j] > 0) if visible else 0 for j in TYPES}
        if ready:
            Fs, ps = th["second"]
            if Fs and rng.random() < float(np.interp(F, Fs, ps)):
                cand = [j for j in TYPES if j not in ready]
                if cand:
                    return f"BUILD {cand[rng.integers(len(cand))]}"
            own = [x for x in view["open"] if x["type"] in ready]
            if own and rng.random() < th["use"]:
                return f"DO T{own[0]['id']}: {ans(own[0])}"
            return do_general(view)
        if prog:
            j = prog[0]; c = th["cont_m1"] if m == 1.0 else th["cont_m2"]
            if rng.random() < sig(c[0] + c[1] * F / 16 + c[2] * st["bench"][j] / 16 + c[3] * H / 16):
                return f"BUILD {j}"
            return do_general(view)
        if rng.random() < p_start(th, F, H, m, sum(n.values()), kernel):
            w = np.array(th["type_base"]) * np.exp(-th["gamma"] * np.array([n[j] / rosen.ARRIVE[j] for j in TYPES]))
            return f"BUILD {TYPES[rng.choice(3, p=w / w.sum())]}"
        return do_general(view)
    return pol


def simulate(th, arm, m, F, draw, kernel="logit"):
    out = []
    for s in SEEDS:
        rng = np.random.default_rng([draw, s, int(F), int(m * 10), {"team": 0, "hidden": 1, "autarky": 2}[arm]])
        out.append(metrics(run_pop("kernel", arm, F, m, s, decide=make_policy(th, rng, kernel))))
    return {k: float(np.nanmean([o[k] for o in out])) if not all(np.isnan(o[k]) for o in out) else float("nan")
            for k in METRICS}


def forecast(rows, B, kernel):
    th0 = fit(rows)
    rng = np.random.default_rng(12345)
    draws = []
    for b in range(B):
        th = th0 if b == 0 else fit(resample(rows, rng))
        draws.append({arm_key(*a): simulate(th, *a, draw=b, kernel=kernel) for a in ARMS})
    res = {}
    for a in ARMS:
        k = arm_key(*a)
        res[k] = {}
        for mt in METRICS:
            v = np.array([d[k][mt] for d in draws], float); v = v[~np.isnan(v)]
            res[k][mt] = (dict(point=float(np.median(v)), lo=float(np.percentile(v, 5)), hi=float(np.percentile(v, 95)))
                          if len(v) else None)
    return th0, res


def rational():
    out = {}
    for a in ARMS:
        ms = [metrics(run_pop("rational", a[0], a[2], a[1], s, decide=rosen.policy_rational)) for s in SEEDS]
        out[arm_key(*a)] = {k: float(np.nanmean([x[k] for x in ms])) if not all(np.isnan(x[k]) for x in ms) else None
                            for k in METRICS}
    return out


HYP = {
    "H1_kernel": "Per model, on core cells: P(build) falls with F and rises with m and H; a logistic in Rosen's net value "
                 "per remaining round, m(1 - F/H) - 1, beats a logistic in F alone by AIC > 4. Myopia index = fitted "
                 "slope dF50/dH divided by its rational value (1 - 1/m), m = 2 and 3, bootstrap 95% CI.",
    "H2_corner": "Team, m = 2, F in {1,3,5,7} pooled: of agents holding >= 1 ready workbench, >= 80% hold exactly one; "
                 "median share of their post-ready DOs in the workbench type >= 0.9.",
    "H3_jump": "Team, m = 2: population DOL against F fits a step (two levels, break at a midpoint of the F grid, 3 "
               "parameters) better than a straight line by AIC > 4. Reported with the fitted break against F* = 8 and "
               "the forecast's break.",
    "H4a_economics": "DOL at team m = 2, F = 3 exceeds DOL at team m = 1, F = 3 (one-sided permutation test over "
                     "populations, alpha 0.05).",
    "H4b_exchange": "At m = 2, F = 1 the share of autarky agents holding >= 2 ready workbenches exceeds the share of team "
                    "agents doing so (one-sided Fisher exact test). Autarky at m = 3, F in {1,3} against its rational "
                    "benchmark is descriptive.",
    "H5_coordination": "At F = 3: coverage (types with >= 1 ready workbench) higher and duplication (ready workbenches "
                       "beyond the 4/2/2 quota) lower in team than in hidden (one-sided permutation tests).",
    "forecast": "Per model and arm, observed 8-seed means of dol, build_rate, benches, points and coverage are scored "
                "against the frozen 90% intervals of the PRIMARY (logistic) kernel; reported as k of n.",
    "multiplicity": "Holm over the p-values of H4a, H4b, H5-coverage and H5-duplication within each model; H1 and H3 are AIC criteria.",
    "notes": "All populations kept. Gate before R1 populations: parse rate >= 90% in R0 pilot populations and R0 "
             "comprehension >= 60%. Who takes which type is descriptive.",
}


def cmd_freeze(a):
    if PRED.exists():
        sys.exit("frozen; refusing to overwrite")
    out = dict(stage="R2", models=a.models, seeds=SEEDS, arms=[arm_key(*x) for x in ARMS], metrics=list(METRICS),
               B=a.B, ridge=RIDGE, hypotheses=HYP, rational=rational(), kernels={}, forecast={}, forecast_iso={})
    for mod in a.models:
        rows = load([mod])
        th, fc = forecast(rows, a.B, "logit")
        _, fi = forecast(rows, max(20, a.B // 4), "iso")
        out["kernels"][mod] = th; out["forecast"][mod] = fc; out["forecast_iso"][mod] = fi
        out.setdefault("n_rows", {})[mod] = len(rows)
        print(mod, "fitted on", len(rows), "rows", flush=True)
    out["sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    PRED.write_text(json.dumps(out, indent=1))
    print("frozen", out["sha256"])


def cmd_show(a):
    for mod in a.models:
        rows = load([mod])
        th = fit(rows)
        print(mod, len(rows), "rows")
        print("  ", {k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in th.items() if k != "iso"})
        for x in ARMS:
            r = simulate(th, *x, draw=0)
            print("  ", arm_key(*x), {k: round(v, 2) for k, v in r.items() if k in ("dol", "build_rate", "benches", "points")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "show"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=200)
    a = ap.parse_args()
    {"freeze": cmd_freeze, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
