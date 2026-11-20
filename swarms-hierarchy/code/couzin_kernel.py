"""Couzin study: M0 gate, M1 kernel tables, the frozen omega rule, nonlinearity indices and the M4b model rule.

Decisions are coded relative to the counted option A (for informed agents A = the preferred site).
Count tables per model (successes s, trials n of "chose A"):
  U0[2]        uninformed, round 0: chose the first label of the frozen pair (label bias)
  Uph[2][6]    uninformed, own last = A, j of 5 visible on A, phase early (t <= 2) / late (t >= 3); folded
               (own B, j  ->  own A, 5 - j, choice flipped)
  Uent[6]      uninformed, first choice after observing (observe-first protocol), j on A; folded
  I0[w][2]     informed at omega w, round 0: chose the preferred site
  I1[w][6]     informed, own last = preferred, j of 5 visible on the preferred site
  I2[w][6]     informed, own last = other, j on the preferred site
  CL[2][6]     (primary model, robustness) uninformed with one visible claimant on A: own A / own B, j on A; folded
Point estimates are monotone in j (weighted pool-adjacent-violators, unshrunk apart from a (s+.5)/(n+1) floor).

Usage: couzin_kernel.py gate | rules | show [--model m]
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import couzin as cz

R = cz.R
CHEAP = ["adaption:gemma-4-26b-a4b-it", "adaption:qwen3.8-27b", "adaption:deepseek-v4-flash"]   # M4b candidates (owner default: cheap only)
COST_ORDER = ["adaption:deepseek-v4-flash", "adaption:gemma-4-26b-a4b-it", "adaption:qwen3.8-27b",
              "adaption:gemma-4-31b-it", "adaption:qwen3.5-397b", "adaption:glm-5.2"]
OM = list(cz.OMEGAS)


# ---------------------------------------------------------------- decisions
def decisions(model, calibration=True, pilot=False):
    """Yield dicts: role U/I, omega, t, own (A/B/None), j, k, a (chose A, relative), lab0 (chose pair[0]; t = 0)."""
    pre = "couzin_pilot_" if pilot else "couzin_"
    pair = cz.frozen_labels()
    for r in cz.rows(R / f"{pre}m1_kernel.jsonl"):
        if r["model"] != model or r["choseA"] is None:
            continue
        chose = r["A_label"] if r["choseA"] else [x for x in pair if x != r["A_label"]][0]
        yield dict(src="M1", role=r["role"], omega=r.get("omega"), t=r["t"], own=r["own"], j=r["jA"], k=cz.K,
                   a=r["choseA"], lab0=int(chose == pair[0]))
    if not calibration:
        return
    for rec in cz.rows(R / f"{pre}pop.jsonl"):
        if rec["model"] != model or not rec["arm"].startswith(("M2_", "M2c_")):
            continue
        A = rec["A_label"]
        for u in range(len(rec["traj"])):
            for i in range(rec["N"]):
                v = rec["traj"][u][i]
                if v < 0 or (u == 0 and rec.get("start") == "convention"):
                    continue
                prev = rec["traj"][u - 1][i] if u > 0 else -1
                own = None if prev < 0 else ("A" if prev == 1 else "B")
                pref = rec["pref"][i]
                d = dict(src="M2", role=rec["roles"][i], omega=rec["omega"][i], t=u, own=own, j=None, k=cz.K, a=v)
                if u == 0:
                    d["lab0"] = int((v == 1) == (A == pair[0]))
                else:
                    jA, k = rec["seen"][u][i]
                    if k != cz.K:
                        continue
                    d["j"] = jA
                if pref == "B":                                     # relative to the preferred site
                    d["a"] = 1 - v
                    d["own"] = {None: None, "A": "B", "B": "A"}[own]
                    d["j"] = None if d["j"] is None else cz.K - d["j"]
                yield d


def tables(model, calibration=True, robust_rows=True, pilot=False):
    z = lambda *s: np.zeros(s + (2,))
    T = dict(U0=np.zeros(2), Uph=z(2, 6), Uent=z(6), I0={w: np.zeros(2) for w in OM},
             I1={w: z(6) for w in OM}, I2={w: z(6) for w in OM}, CL=z(2, 6))

    def add(arr, ok):
        arr[0] += ok; arr[1] += 1
    for d in decisions(model, calibration, pilot):
        if d["role"] == "U":
            if d["t"] == 0:
                add(T["U0"], d["lab0"])
            elif d["own"] is None:
                add(T["Uent"][d["j"]], d["a"]); add(T["Uent"][cz.K - d["j"]], 1 - d["a"])   # fold (both copies, half weight below)
            else:
                ph = 0 if d["t"] <= 2 else 1
                if d["own"] == "A":
                    add(T["Uph"][ph][d["j"]], d["a"])
                else:
                    add(T["Uph"][ph][cz.K - d["j"]], 1 - d["a"])
        else:
            w = d["omega"]
            if w not in T["I0"]:
                continue
            if d["t"] == 0:
                add(T["I0"][w], d["a"])
            elif d["own"] == "A":
                add(T["I1"][w][d["j"]], d["a"])
            elif d["own"] == "B":
                add(T["I2"][w][d["j"]], d["a"])
    T["Uent"] = T["Uent"] / 2.0                                    # each entry decision was counted twice by the fold
    if robust_rows:
        for r in cz.rows(R / f"{'couzin_pilot_' if pilot else 'couzin_'}m1_kernel_robust.jsonl"):
            if r["model"] != model or r["choseA"] is None or not r["set"].startswith("claim"):
                continue
            own, j, a = r["own"], r["jA"], r["choseA"]
            if r["claim"] == "B":                                   # fold: claimant on B -> claimant on A
                own, j, a = {"A": "B", "B": "A"}[own], cz.K - j, 1 - a
            add(T["CL"][0 if own == "A" else 1][j], a)
    return T


# ---------------------------------------------------------------- isotonic estimates
def pava(y, w):
    """Weighted pool-adjacent-violators: non-decreasing fit."""
    y, w = list(map(float, y)), list(map(float, w))
    blocks = [[y[i], w[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            a, b = blocks[i], blocks[i + 1]
            ww = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / ww, ww, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, c in blocks:
        out += [v] * c
    return np.array(out)


def iso(counts, draw=None):
    """counts: (6, 2) array of (s, n). draw: numpy Generator -> one parametric-bootstrap draw (Beta(s+.5, n-s+.5))."""
    s, n = counts[:, 0], counts[:, 1]
    if draw is None:
        p = (s + 0.5) / (n + 1)
    else:
        p = draw.beta(s + 0.5, n - s + 0.5)
    return pava(p, n + 1)


def point(counts, draw=None):
    s, n = counts
    return (s + 0.5) / (n + 1) if draw is None else draw.beta(s + 0.5, n - s + 0.5)


def estimates(T, draw=None):
    """Probability tables (numpy) used by the simulator."""
    E = dict(U0=point(T["U0"], draw), Uph=np.stack([iso(T["Uph"][p], draw) for p in (0, 1)]),
             Uent=iso(T["Uent"], draw), I0={}, I1={}, I2={})
    # Uent is folded: enforce symmetry p(j) = 1 - p(5 - j) after the monotone fit
    E["Uent"] = 0.5 * (E["Uent"] + 1 - E["Uent"][::-1])
    for w in OM:
        E["I0"][w] = point(T["I0"][w], draw)
        E["I1"][w] = iso(T["I1"][w], draw)
        E["I2"][w] = iso(T["I2"][w], draw)
    E["CL"] = np.stack([iso(T["CL"][o], draw) for o in (0, 1)]) if T["CL"][:, :, 1].sum() else None
    return E


# ---------------------------------------------------------------- indices
def logit_j(j):
    return math.log((j + 0.5) / (cz.K + 0.5 - j))


def fit_family(rows, fix_a=None):
    """MLE of logit P(A) = a * logit((j+.5)/6) + beta * own_sign (+ c). rows: (j, own_sign, a)."""
    from scipy.optimize import minimize
    if not rows:
        return dict(a=float("nan"), beta=float("nan"), c=float("nan"))
    J = np.array([logit_j(r[0]) for r in rows]); S = np.array([r[1] for r in rows], float); Y = np.array([r[2] for r in rows], float)

    def nll(th):
        a = fix_a if fix_a is not None else th[0]
        z = a * J + th[1] * S + th[2]
        return np.sum(np.logaddexp(0, z) - Y * z)
    th = minimize(nll, np.array([1.0, 0.0, 0.0]), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8)).x
    return dict(a=float(fix_a if fix_a is not None else th[0]), beta=float(th[1]), c=float(th[2]))


def nonlinearity(model, calibration=True, pilot=False):
    """Uninformed index: fitted exponent a (unfolded rows, own A/B, both phases; symmetric design, intercept free),
    and the contrast P(A | 4 of 5) - P(A | 3 of 5) averaged over own last (isotonic folded table, late phase)."""
    rows = [(d["j"], 1 if d["own"] == "A" else -1, d["a"]) for d in decisions(model, calibration, pilot)
            if d["role"] == "U" and d["t"] > 0 and d["own"] is not None]
    f = fit_family(rows)
    E = estimates(tables(model, calibration, pilot=pilot))
    up = E["Uph"][1]
    pA = lambda own, j: up[j] if own == "A" else 1 - up[cz.K - j]
    contrast = 0.5 * ((pA("A", 4) - pA("A", 3)) + (pA("B", 4) - pA("B", 3)))
    return dict(a=f["a"], beta=f["beta"], c=f["c"], contrast=float(contrast), n=len(rows))


def omega_rule(model, pilot=False):
    """Frozen: strong = smallest omega with P(stay at preferred | own preferred, 0 of 5 there) >= .9;
    weak = largest omega with P(follow a 4-of-5 majority away | own preferred, 1 of 5 there) >= .5."""
    T = tables(model, calibration=False, robust_rows=False, pilot=pilot)
    E = estimates(T)
    stay0 = {w: float(E["I1"][w][0]) for w in OM}
    follow1 = {w: float(1 - E["I1"][w][1]) for w in OM}
    flags = []
    strong = next((w for w in OM if stay0[w] >= .9), None)
    if strong is None:
        strong = OM[-1]; flags.append("no omega reaches the strong criterion; strong = 6")
    weak = next((w for w in OM[::-1] if follow1[w] >= .5), None)
    if weak is None:
        weak = OM[0]; flags.append("no omega meets the weak criterion; weak = 0.5")
    if weak >= strong:
        lower = [w for w in OM if w < strong]
        weak = lower[0] if lower else OM[0]
        flags.append(f"weak >= strong; weak set to {weak}")
        if weak >= strong:
            strong = OM[-1]; weak = OM[0]; flags.append("fallback strong 6, weak 0.5")
    return dict(weak=weak, strong=strong, stay0=stay0, follow1=follow1, flags=flags)


# ---------------------------------------------------------------- M0 gate
def gate(models, pilot=False):
    pre = "couzin_pilot_" if pilot else "couzin_"
    out = R / f"{pre}m0_gate.json"
    if out.exists() and not pilot:
        sys.exit(f"{out} exists (write-once)")
    quiz, labs = cz.rows(R / f"{pre}m0_quiz.jsonl"), cz.rows(R / f"{pre}m0_labels.jsonl")
    res = {}
    for m in models:
        q = [r for r in quiz if r["model"] == m]; l = [r for r in labs if r["model"] == m]
        acc = sum(r["correct"] for r in q) / max(len(q), 1)
        qparse = sum(r["value"] is not None for r in q) / max(len(q), 1)
        lparse = sum(r["choice"] is not None for r in l) / max(len(l), 1)
        bias = {}
        for pi, pair in enumerate(cz.LABEL_PAIRS):
            per_order = []
            for o in (0, 1):
                v = [r for r in l if r["pair"] == pi and r["order"] == o and r["choice"]]
                per_order.append(sum(r["choice"] == pair[0] for r in v) / len(v) if v else .5)
            bias[pi] = dict(p_first_label=float(np.mean(per_order)), p_first_named=float(
                0.5 * (per_order[0] + 1 - per_order[1])))
        res[m] = dict(quiz_acc=acc, quiz_parse=qparse, label_parse=lparse, n_quiz=len(q), n_labels=len(l),
                      passed=bool(acc >= .8 and qparse >= .9 and lparse >= .9), bias=bias)
    passed = [m for m in models if res[m]["passed"]]
    score = {pi: max([abs(res[m]["bias"][pi]["p_first_label"] - .5) for m in passed] or [1]) for pi in range(len(cz.LABEL_PAIRS))}
    best = min(score, key=lambda pi: (score[pi], pi))
    d = dict(models=res, passed=passed, label_pair=list(cz.LABEL_PAIRS[best]), pair_scores=score,
             primary_passed=cz.PRIMARY in passed)
    out.write_text(json.dumps(d, indent=1))
    print(json.dumps(d, indent=1))
    return d


def rules(models, pilot=False):
    pre = "couzin_pilot_" if pilot else "couzin_"
    out = R / f"{pre}m1_rules.json"
    if out.exists() and not pilot:
        sys.exit(f"{out} exists (write-once)")
    g = json.loads((R / f"{pre}m0_gate.json").read_text())
    passed = [m for m in models if m in g["passed"]]
    idx = {m: nonlinearity(m, calibration=False, pilot=pilot) for m in passed}
    om = {m: omega_rule(m, pilot=pilot) for m in passed}
    a0 = idx[cz.PRIMARY]["a"]
    cands = [m for m in CHEAP if m in passed and m != cz.PRIMARY and not math.isnan(idx[m]["a"])]
    m4b = max(cands, key=lambda m: (abs(idx[m]["a"] - a0), -COST_ORDER.index(m))) if cands else None
    d = dict(nonlinearity=idx, omega=om, m4b_model=m4b, m4b_candidates=cands)
    out.write_text(json.dumps(d, indent=1))
    print(json.dumps(d, indent=1))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gate", "rules", "show"])
    ap.add_argument("--models", nargs="+", default=cz.MODELS6)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.what == "gate":
        gate(a.models, a.pilot)
    elif a.what == "rules":
        rules(a.models, a.pilot)
    else:
        for m in a.models:
            T = tables(m, pilot=a.pilot)
            E = estimates(T)
            print(m, json.dumps({k: (v.round(3).tolist() if isinstance(v, np.ndarray) else
                                     {w: x.round(3).tolist() if isinstance(x, np.ndarray) else round(float(x), 3) for w, x in v.items()} if isinstance(v, dict) else v)
                                 for k, v in E.items()}))


if __name__ == "__main__":
    main()
