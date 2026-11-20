"""Evaluation of the hierarchy-efficiency study, phase 1 (pre-registered hypotheses S1-S7).

Reads results/hier_{g1c,g2,bench,g3,g4}_pop.jsonl, hier_kernels.json, hier_freeze.json, hier_freeze_g3.json;
writes results/hier_eval.json. Holm family (one-sided unless stated): S1, S2a-aligned, S2a-misaligned, S3 (two-sided),
S4, S5-worst, S5-best, S6a, S6b.
  hier_eval.py [--draws 10000]
"""
import argparse, itertools, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, norm

sys.path.insert(0, str(Path(__file__).parent))
import office as O
from hier_sim import sorting_index

R = O.R


def load(name):
    p = R / f"hier_{name}_pop.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def signflip(d, alt="greater", draws=20000, rng=None):
    d = np.array([x for x in d if x == x], float)
    if len(d) == 0:
        return float("nan"), 0
    obs = d.mean()
    if len(d) <= 14:
        signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    else:
        signs = rng.choice([1, -1], size=(draws, len(d)))
    null = (signs * d).mean(1)
    if alt == "greater":
        p = float(np.mean(null >= obs - 1e-12))
    else:
        p = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    return p, len(d)


def holm(ps):
    keys = [k for k in ps if ps[k] == ps[k]]
    order = sorted(keys, key=lambda k: ps[k])
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    K = json.load(open(R / "hier_kernels.json"))
    fz = json.load(open(R / "hier_freeze.json"))
    fz3 = json.load(open(R / "hier_freeze_g3.json")) if (R / "hier_freeze_g3.json").exists() else {"forecasts": {}}
    gain, comps, rule = K["gain"], fz["compositions"], fz["composition_rule"]
    g2, bench, g3, g4, g1c = load("g2"), load("bench"), load("g3"), load("g4"), load("g1c")
    res, ps = {}, {}

    # ---- S1 sorting (full record, both compositions pooled)
    full = [p for p in g2 if p["record"] == "full"]

    def si(p, gmap=None):
        return sorting_index(p["metrics"], p["seats"], gmap or gain)
    obs = [si(p) for p in full]
    valid = [(p, s) for p, s in zip(full, obs) if s == s]
    stat = float(np.mean([s for _, s in valid])) if valid else float("nan")
    null = []
    for _ in range(a.draws):
        vals = []
        for p, _ in valid:
            ms = comps[p["comp"]]
            perm = dict(zip(ms, rng.permutation([gain[m] for m in ms])))
            vals.append(si(p, perm))
        null.append(np.nanmean(vals))
    ps["S1"] = float(np.mean(np.array(null) >= stat - 1e-12)) if valid else float("nan")
    res["S1"] = dict(mean_sorting_index=stat, n=len(valid), p=ps["S1"],
                     by_comp={c: float(np.nanmean([si(p) for p in full if p["comp"] == c])) for c in comps})

    # ---- S2 welfare loss per composition (full record)
    res["S2"] = {}
    for c in comps:
        W = np.array([p["metrics"]["throughput"] for p in full if p["comp"] == c])
        B = {arm: np.array([p["metrics"]["throughput"] for p in bench if p["comp"] == c and p["key"].split("|")[2] == arm])
             for arm in ("noplan", "ownorder", "oracle")}
        if len(W) == 0 or any(len(v) == 0 for v in B.values()):
            continue

        def L_of(Wm, Bm):
            ws = max(Bm.values())
            return (ws - Wm) / (ws - Bm["noplan"]) if ws > Bm["noplan"] else float("nan")
        Lobs = L_of(W.mean(), {k: v.mean() for k, v in B.items()})
        boots = np.array([L_of(rng.choice(W, len(W)).mean(), {k: rng.choice(v, len(v)).mean() for k, v in B.items()})
                          for _ in range(a.draws)])
        boots = boots[boots == boots]
        Lnull = fz["forecasts"][f"null|{c}"]["welfare_loss"]["mean"]
        ps[f"S2a-{c}"] = float(np.mean(boots <= 0))
        res["S2"][c] = dict(L=float(Lobs), ci95=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                            p_L_above_0=ps[f"S2a-{c}"], L_random_role_null=Lnull,
                            p_L_below_null=float(np.mean(boots >= Lnull)), W=float(W.mean()),
                            bench={k: float(v.mean()) for k, v in B.items()},
                            W_star_arm=max(B, key=lambda k: B[k].mean()))
        # secondary: loss against the best *hierarchical* arrangement (oracle planner), not against any arm
        Lo = lambda Wm, Bm: (Bm["oracle"] - Wm) / (Bm["oracle"] - Bm["noplan"]) if Bm["oracle"] > Bm["noplan"] \
            else float("nan")
        bo = np.array([Lo(rng.choice(W, len(W)).mean(), {k: rng.choice(v, len(v)).mean() for k, v in B.items()})
                       for _ in range(a.draws)])
        bo = bo[bo == bo]
        res["S2"][c]["L_vs_oracle"] = float(Lo(W.mean(), {k: v.mean() for k, v in B.items()}))
        res["S2"][c]["L_vs_oracle_ci95"] = [float(np.quantile(bo, 0.025)), float(np.quantile(bo, 0.975))] \
            if len(bo) else None

    # ---- S3 assertiveness vs skill (misaligned, full)
    most, best = rule["misaligned"]["most"], rule["misaligned"]["best"]
    mis = [p for p in full if p["comp"] == "misaligned"]
    D = [p["metrics"]["per_model"][most]["plan_share"] - p["metrics"]["per_model"][best]["plan_share"] for p in mis]
    ps["S3"], n3 = signflip(D, "two-sided", rng=rng)
    res["S3"] = dict(most_assertive=most, best_manager=best, mean_diff=float(np.nanmean(D)) if D else None, n=n3,
                     p_two_sided=ps["S3"], winner=(most if np.nanmean(D) > 0 else best) if D else None)

    # ---- S4 information: full vs lean, paired by composition and seed
    lean = {(p["comp"], p["seed"]): si(p) for p in g2 if p["record"] == "lean"}
    d4 = [si(p) - lean[(p["comp"], p["seed"])] for p in full if (p["comp"], p["seed"]) in lean]
    ps["S4"], n4 = signflip(d4, rng=rng)
    res["S4"] = dict(mean_diff=float(np.nanmean(d4)) if d4 else None, n=n4, p=ps["S4"],
                     lean_mean=float(np.nanmean(list(lean.values()))) if lean else None)

    # ---- S5 disposition sentence (G3 vs G2 misaligned full, paired by seed)
    base = {p["seed"]: p for p in mis}
    res["S5"] = {}
    for who, alt_sign in (("worst", -1), ("best", +1)):
        rows = [p for p in g3 if p["disposed"] == who and p["seed"] in base]
        m = fz["g3_targets"][who]
        dW = [alt_sign * (p["metrics"]["throughput"] - base[p["seed"]]["metrics"]["throughput"]) for p in rows]
        dS = [p["metrics"]["per_model"][m]["plan_share"] - base[p["seed"]]["metrics"]["per_model"][m]["plan_share"]
              for p in rows]
        ps[f"S5-{who}"], n5 = signflip(dW, rng=rng)
        res["S5"][who] = dict(model=m, n=n5, throughput_change=alt_sign * float(np.nanmean(dW)) if dW else None,
                              p=ps[f"S5-{who}"], plan_share_change=float(np.nanmean(dS)) if dS else None,
                              p_plan_share_up=signflip(dS, rng=rng)[0])

    # ---- S6 span of control
    app = [p for p in g1c if p["arm"] == "appointed" and p["manager_model"] in K["planq"]]
    q = {m: K["planq"][m]["q"] for m in K["planq"]}
    if app:
        f = [p["metrics"]["follow_rate"] for p in app]; qq = [q[p["manager_model"]] for p in app]
        rho = float(spearmanr(qq, f, nan_policy="omit").statistic)
        ms = sorted({p["manager_model"] for p in app}); null = []
        for _ in range(a.draws):
            perm = dict(zip(ms, rng.permutation([q[m] for m in ms])))
            null.append(spearmanr([perm[p["manager_model"]] for p in app], f, nan_policy="omit").statistic)
        ps["S6a"] = float(np.mean(np.array(null) >= rho - 1e-12))
        span = [p["metrics"]["mean_span"] for p in app]
        res["S6a"] = dict(spearman_follow_vs_planq=rho, p=ps["S6a"], n=len(app),
                          spearman_span_vs_planq=float(spearmanr(qq, span).statistic),
                          by_manager={m: dict(follow=float(np.nanmean([p["metrics"]["follow_rate"] for p in app
                                                                       if p["manager_model"] == m])),
                                              span=float(np.mean([p["metrics"]["mean_span"] for p in app
                                                                  if p["manager_model"] == m])),
                                              gain=gain.get(m), planq=q[m]) for m in ms})
    good = {p["seed"]: p for p in g4 if p["seeded"] == "good"}
    d6 = [good[p["seed"]]["metrics"]["follow_rate"] - p["metrics"]["follow_rate"] for p in g4
          if p["seeded"] == "poor" and p["seed"] in good]
    ps["S6b"], n6 = signflip(d6, rng=rng)
    fol = K["follow"]
    slope = {}
    for m, cells in fol.items():
        xs, ys = [], []
        for c, (k, n, _) in cells.items():
            qb = c.split("|")[0]
            if qb != "new":
                xs += [float(qb)] * n; ys += [1] * k + [0] * (n - k)
        slope[m] = float(np.polyfit(xs, ys, 1)[0]) if xs else None
    res["S6b"] = dict(follow_good_minus_poor=float(np.nanmean(d6)) if d6 else None, n=n6, p=ps["S6b"],
                      g1e_linear_slope_by_model=slope)

    # ---- S7 forecast coverage and log score against the random-role null
    hits, cells, margin = 0, [], 0.0
    groups = defaultdict(list)
    for p in g2:
        groups[f"g2|{p['comp']}|{p['record']}"].append(p)
    for p in bench:
        groups[f"bench|{p['comp']}|{p['key'].split('|')[2]}"].append(p)
    for p in g3:
        groups[f"g3|{p['disposed']}"].append(p)
    allfc = dict(fz["forecasts"], **fz3["forecasts"])
    for key, pops in groups.items():
        fc = allfc.get(key)
        if not fc:
            continue
        c = pops[0]["comp"]
        b, mo = rule[c]["best"], rule[c]["most"]
        obsv = dict(throughput=np.mean([p["metrics"]["throughput"] for p in pops]),
                    plan_rate=np.mean([p["metrics"]["plan_rate"] for p in pops]),
                    best_manager_plan_share=np.nanmean([p["metrics"]["per_model"][b]["plan_share"] for p in pops]),
                    most_assertive_plan_share=np.nanmean([p["metrics"]["per_model"][mo]["plan_share"] for p in pops]))
        for metric, v in obsv.items():
            f = fc.get(metric)
            if not f or f["mean"] != f["mean"] or v != v:
                continue
            inside = f["lo"] <= v <= f["hi"]
            hits += inside
            cells.append(dict(cell=key, metric=metric, observed=float(v), forecast=f["mean"], lo=f["lo"], hi=f["hi"],
                              inside=bool(inside)))
            nk = fz["forecasts"].get(f"null|{c}") if key.startswith("g2|") else None
            if nk and metric in ("best_manager_plan_share", "most_assertive_plan_share", "plan_rate"):
                def lp(ff):
                    sd = max(1e-3, (ff["hi"] - ff["lo"]) / (2 * 1.645))
                    return norm.logpdf(v, ff["mean"], sd)
                margin += lp(f) - lp(nk[metric])
    res["S7"] = dict(hits=int(hits), n=len(cells), cells=cells, log_score_margin_vs_null_nats=float(margin))

    # ---- exploratory: convexity of influence in manager gain (free populations, full record)
    xs, ys = [], []
    for p in full:
        for m in comps[p["comp"]]:
            s = p["metrics"]["per_model"][m]["infl_share"]
            if s == s:
                xs.append(gain[m]); ys.append(s)
    res["exploratory_convexity"] = dict(quadratic_coef=float(np.polyfit(xs, ys, 2)[0]) if len(xs) > 3 else None,
                                        n=len(xs))
    res["holm"] = holm(ps)
    res["raw_p"] = ps
    json.dump(res, open(R / "hier_eval.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("S7",)}, indent=1, default=str))
    print(f"S7 coverage {hits} of {len(cells)}; log-score margin {margin:.1f} nats")


if __name__ == "__main__":
    main()
