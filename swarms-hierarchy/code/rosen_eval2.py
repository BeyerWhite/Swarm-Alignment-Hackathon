"""Evaluate the follow-ups and phase 2 (ROSEN_PHASE2_PREREGISTRATION.md). Written and tested on simulated data
before any call of these stages. Each section is skipped if its data file is missing.

Writes results/rosen_p2_eval.json.   Usage: rosen_eval2.py [--B 2000]
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, wilcoxon, binomtest

sys.path.insert(0, str(Path(__file__).parent))
import rosen, rosen_p2
from rosen import metrics
from rosen_forecast import arm_key
from rosen_eval import perm_greater, holm
from rosen_eval import step_vs_line as _step_vs_line


def step_vs_line(F, y):
    """Phase 1's step-vs-line AIC test; None when fewer than three F values have data."""
    ok = [(f, v) for f, v in zip(F, y) if v is not None and not np.isnan(v)]
    return _step_vs_line(*zip(*ok)) if len({f for f, _ in ok}) >= 3 else None

R = rosen.R
MAIN_RESULTS = Path("/Users/beyer_white/Desktop/LocalCodeRepository/Swarm-Alignment-Hackathon/swarms-hierarchy/results")
SCORED = ("dol", "build_rate", "benches", "points", "coverage")
GEMMA = "adaption:gemma-4-31b-it"


def load(name):
    p = R / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def by_arm(recs, keyf):
    out = defaultdict(list)
    for r in recs:
        out[keyf(r)].append(r)
    return out


def mean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def strat_trend(rows, sign, B, rng):
    """rows: (stratum, x, y). Statistic: sum over strata of Spearman rho(x, y) * sign; permutation of y within stratum.
    One-sided p for sign * trend > 0."""
    st = defaultdict(list)
    for s, x, y in rows:
        if y is not None and not np.isnan(y):
            st[s].append((x, y))
    def stat(groups):
        tot = 0.0
        for g in groups.values():
            xs, ys = zip(*g)
            if len(set(ys)) > 1 and len(set(xs)) > 1:
                tot += spearmanr(xs, ys).statistic
        return sign * tot
    obs = stat(st)
    perm = []
    for _ in range(B):
        pg = {}
        for s, g in st.items():
            xs, ys = zip(*g); pg[s] = list(zip(xs, rng.permutation(ys)))
        perm.append(stat(pg))
    return dict(stat=float(obs), p=float((1 + np.sum(np.array(perm) >= obs - 1e-12)) / (B + 1)),
                n=sum(len(g) for g in st.values()))


def strat_perm_greater(rows, B, rng):
    """rows: (stratum, group in {0,1}, y). Statistic: sum over strata of mean(y|1) - mean(y|0); labels permuted within stratum."""
    st = defaultdict(list)
    for s, g, y in rows:
        if y is not None and not np.isnan(y):
            st[s].append((g, y))
    def stat(groups):
        tot = 0.0
        for v in groups.values():
            y1 = [y for g, y in v if g == 1]; y0 = [y for g, y in v if g == 0]
            if y1 and y0:
                tot += np.mean(y1) - np.mean(y0)
        return tot
    obs = stat(st)
    perm = []
    for _ in range(B):
        pg = {s: list(zip(rng.permutation([g for g, _ in v]), [y for _, y in v])) for s, v in st.items()}
        perm.append(stat(pg))
    return dict(diff=float(obs), p=float((1 + np.sum(np.array(perm) >= obs - 1e-12)) / (B + 1)))


def sig_down(x, lo, hi, x50, s):
    return lo + (hi - lo) / (1 + np.exp((x - x50) / s))


def sig_up(x, lo, hi, x50, s):
    return lo + (hi - lo) / (1 + np.exp(-(x - x50) / s))


def fit_sig(x, y, down=True):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(y); x, y = x[ok], y[ok]
    f = sig_down if down else sig_up
    span = x.max() - x.min()
    y = np.clip(y, 0.0, 1.0)                 # DOL can exceed 1 by rounding error
    p0 = [float(y.min()), float(y.max()), float(np.median(x)), span / 6]
    try:
        p, _ = curve_fit(f, x, y, p0=p0, bounds=([0, 0, x.min() - span, span / 200], [1, 1, x.max() + span, 4 * span]),
                         maxfev=20000)
    except Exception:
        return None
    lo, hi, x50, s = p
    return dict(lo=float(lo), hi=float(hi), x50=float(x50), s=float(s), w=float(s / x50) if x50 > 0 else float("nan"))


def boot_sig(groups, B, rng, down=True):
    """groups: {x: [y per population]}. Point fit plus bootstrap (populations resampled within x)."""
    xs = sorted(groups)
    def flat(gr):
        return [x for x in xs for _ in gr[x]], [y for x in xs for y in gr[x]]
    pt = fit_sig(*flat(groups), down=down)
    bs = []
    for _ in range(B):
        g2 = {x: list(rng.choice(groups[x], len(groups[x]))) for x in xs}
        f = fit_sig(*flat(g2), down=down)
        if f:
            bs.append(f)
    ci = {k: [float(np.nanpercentile([b[k] for b in bs], q)) for q in (5, 95)] for k in ("x50", "s", "w")} if bs else {}
    return dict(point=pt, ci90=ci, n_boot=len(bs)), bs


def logit_f50(recs, B, rng):
    """Agent-level P(start a workbench) = sig(a + b F); F50 = -a/b. Bootstrap populations within F."""
    from rosen_forecast import ridge_logit
    def f50(rr):
        X, y = [], []
        for r in rr:
            for a in r["stats"].values():
                X.append([1, r["F"] / 16]); y.append(any(v > 0 for v in a["bench"].values()))
        b = ridge_logit(X, y, lam=1e-4)
        return float(-b[0] / b[1] * 16) if b[1] < 0 else float("inf")
    byF = by_arm(recs, lambda r: r["F"])
    pt = f50(recs)
    bs = [f50([x for F in byF for x in rng.choice(byF[F], len(byF[F]))]) for _ in range(B)]
    return pt, np.array(bs)


def h2_corner(recs, F_max=7):
    ms = [metrics(r) for r in recs if r["F"] <= F_max]
    builders, one = 0, 0
    for r in recs:
        if r["F"] > F_max:
            continue
        for a in r["stats"].values():
            n = sum(v >= r["F"] for v in a["bench"].values())
            builders += n >= 1; one += n == 1
    us = [m["use_share"] for m in ms if not np.isnan(m["use_share"])]
    return dict(builders=builders, share_one=one / builders if builders else None,
                median_pop_use_share=float(np.median(us)) if us else None,
                supported=bool(builders and one / builders >= 0.8 and us and np.median(us) >= 0.9))


# ----------------------------------------------------------------------------------------------- sections

def sec_r2x(B, rng):
    pred_p = R / "rosen_r2x_predictions.json"
    pops = load("rosen_r2x_pop.jsonl")
    if not (pred_p.exists() and pops):
        return None
    pred = json.load(open(pred_p))
    out = dict(pred_sha=pred["sha256"], models={})
    disc = dict(v2_only=0, v1_only=0)
    for mod in pred["models"]:
        rr = [r for r in pops if r["model"] == mod]
        arms = by_arm(rr, lambda r: arm_key(r["arm"], r["m"], r["F"]))
        cov = {"v2": [0, 0], "v1": [0, 0]}; table = {}
        for k, recs in arms.items():
            ms = [metrics(r) for r in recs]
            table[k] = {}
            for mt in SCORED:
                obs = mean([m[mt] for m in ms])
                if np.isnan(obs):
                    continue
                row = dict(obs=obs)
                for v in ("v2", "v1"):
                    f = pred[v][mod].get(k, {}).get(mt)
                    if f is None:
                        continue
                    ins = f["lo"] - 1e-9 <= obs <= f["hi"] + 1e-9
                    row[v] = dict(point=f["point"], lo=f["lo"], hi=f["hi"], inside=ins)
                    cov[v][0] += ins; cov[v][1] += 1
                if "v2" in row and "v1" in row:
                    disc["v2_only"] += row["v2"]["inside"] and not row["v1"]["inside"]
                    disc["v1_only"] += row["v1"]["inside"] and not row["v2"]["inside"]
                table[k][mt] = row
        team2 = [r for r in rr if r["arm"] == "team" and r["m"] == 2.0]
        d3 = [metrics(r)["dol"] for r in rr if r["arm"] == "team" and r["m"] == 2.0 and r["F"] == 3]
        d1 = [metrics(r)["dol"] for r in rr if r["arm"] == "team" and r["m"] == 1.0 and r["F"] == 3]
        out["models"][mod] = dict(coverage={v: dict(inside=c[0], n=c[1]) for v, c in cov.items()}, arms=table,
                                  replication=dict(H2=h2_corner(team2),
                                                   H3=step_vs_line([r["F"] for r in team2], [metrics(r)["dol"] for r in team2]),
                                                   H4a=perm_greater(d3, d1, B, rng)))
    n = disc["v2_only"] + disc["v1_only"]
    out["mcnemar_pooled"] = dict(**disc, p=float(binomtest(disc["v2_only"], n, 0.5, alternative="greater").pvalue) if n else None)
    # retrospective v2 on R2 data
    r2 = load("rosen_r2_pop.jsonl"); retro = {}
    for mod in pred["models"]:
        arms = by_arm([r for r in r2 if r["model"] == mod], lambda r: arm_key(r["arm"], r["m"], r["F"]))
        k_in = n_all = 0
        for k, recs in arms.items():
            ms = [metrics(r) for r in recs]
            for mt in SCORED:
                obs = mean([m[mt] for m in ms]); f = pred["v2_retro"][mod].get(k, {}).get(mt)
                if f is None or np.isnan(obs):
                    continue
                k_in += f["lo"] - 1e-9 <= obs <= f["hi"] + 1e-9; n_all += 1
        retro[mod] = dict(inside=k_in, n=n_all)
    out["v2_retro_on_R2"] = retro
    return out


def sec_r2r(B, rng):
    rr = [r for r in load("rosen_r2r_pop.jsonl") if r["model"] == GEMMA]
    base = [r for r in load("rosen_r2_pop.jsonl") if r["model"] == GEMMA]
    if not rr:
        return None
    key = lambda r: (r["arm"], r["m"], r["F"], r["seed"])
    bmap = {key(r): r for r in base}
    on57 = [metrics(r)["build_rate"] for r in rr if r["arm"] == "team" and r["m"] == 2.0 and r["F"] in (5, 7)]
    off57 = [metrics(r)["build_rate"] for r in base if r["arm"] == "team" and r["m"] == 2.0 and r["F"] in (5, 7)]
    pairs = [(metrics(r)["points"], metrics(bmap[key(r)])["points"]) for r in rr
             if r["arm"] == "team" and r["m"] == 2.0 and key(r) in bmap]
    d = [a - b for a, b in pairs]
    team = [r for r in rr if r["arm"] == "team" and r["m"] == 2.0]
    res = dict(
        a_build_F57=perm_greater(on57, off57, B, rng),
        b_step=step_vs_line([r["F"] for r in team], [metrics(r)["dol"] for r in team]),
        c_points=dict(mean_diff=mean(d), n=len(d),
                      p=float(wilcoxon(d, alternative="greater").pvalue) if len(d) > 5 and any(d) else None),
        d_m1_dol=mean([metrics(r)["dol"] for r in rr if r["arm"] == "team" and r["m"] == 1.0]),
        arm_means={arm_key(r0["arm"], r0["m"], r0["F"]): {k: mean([metrics(x)[k] for x in v])
                                                          for k in ("dol", "build_rate", "benches", "points", "invalid_rate")}
                   for _, v in by_arm(rr, lambda r: arm_key(r["arm"], r["m"], r["F"])).items() for r0 in v[:1]},
        invalid_rate=mean([metrics(r)["invalid_rate"] for r in rr]))
    res["holm"] = holm(dict(a=res["a_build_F57"]["p"] if res["a_build_F57"] else None, c=res["c_points"]["p"]))
    return res


def sec_r2b(B, rng):
    m3 = [r for r in load("rosen_r2b_pop.jsonl") if r["model"] == GEMMA and r["m"] == 3.0]
    m2 = [r for r in load("rosen_r2_pop.jsonl") if r["model"] == GEMMA and r["arm"] == "team" and r["m"] == 2.0]
    if not m3:
        return None
    f3, b3 = logit_f50(m3, B, rng); f2, b2 = logit_f50(m2, B, rng)
    sh = b3 - b2; sh = sh[np.isfinite(sh)]
    return dict(F50_m2=f2, F50_m3=f3, shift=f3 - f2, shift_ci95=[float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))],
                supported=bool(len(sh) and np.percentile(sh, 2.5) > 0), rational_shift=16 * (1 - 1 / 3) - 16 * (1 - 1 / 2),
                step_m3=step_vs_line([r["F"] for r in m3], [metrics(r)["dol"] for r in m3]),
                arm_means={F: {k: mean([metrics(r)[k] for r in v]) for k in ("dol", "build_rate", "benches", "points")}
                           for F, v in sorted(by_arm(m3, lambda r: r["F"]).items())})


def sec_r5(B, rng):
    r5 = [r for r in load("rosen_r5_pop.jsonl") if r["model"] == GEMMA]
    if not r5:
        return None
    r8 = [r for r in load("rosen_r2_pop.jsonl") if r["model"] == GEMMA and r["arm"] == "team" and r["m"] == 2.0
          and r["F"] in (1, 3, 5, 7, 9)]
    fits, boots = {}, {}
    for Nn, recs in ((4, [r for r in r5 if r["N"] == 4]), (8, r8), (16, [r for r in r5 if r["N"] == 16])):
        g = defaultdict(list)
        for r in recs:
            g[r["F"]].append(metrics(r)["dol"])
        if g:
            fits[Nn], boots[Nn] = boot_sig(dict(g), B // 4, rng)
            fits[Nn]["means"] = {F: mean(v) for F, v in sorted(g.items())}
    out = dict(rosen=fits)
    if 4 in boots and 16 in boots and boots[4] and boots[16]:
        k = min(len(boots[4]), len(boots[16]))
        lr = np.log(np.array([b["w"] for b in boots[16][:k]]) / np.array([b["w"] for b in boots[4][:k]]))
        lr = lr[np.isfinite(lr)]
        ci = [float(np.percentile(lr, 5)), float(np.percentile(lr, 95))]
        pt = (np.log(fits[16]["point"]["w"] / fits[4]["point"]["w"]) if fits[16]["point"] and fits[4]["point"] else None)
        out["log_w16_over_w4"] = dict(point=pt, ci90=ci,
                                      n_independent=bool(pt is not None and ci[0] <= 0 <= ci[1] and abs(pt) < np.log(2)))
    # hierarchy contrast (descriptive): order parameter q against rho, N = 8 and 16
    te = MAIN_RESULTS / "threshold_eval.json"
    if te.exists():
        cur = json.load(open(te))["curves"]; hier = {}
        for Nn, c in cur.items():
            g = {row["rho"]: list(row["q_runs"]) for row in c["rows"]}
            hier[int(Nn)], _ = boot_sig(g, B // 4, rng, down=False)
        out["hierarchy"] = hier
    return out


def sec_r6(B, rng):
    # R-A3: the codebook battery is primary; the registered multiplication battery is reported as battery_mul
    acc = load("rosen_r6_acc_code.jsonl"); pops = [r for r in load("rosen_r6_pop.jsonl") if r["model"] == GEMMA]
    if not acc and not pops:
        return None
    out = dict(battery={}, battery_mul={})
    for key, rows in (("battery", acc), ("battery_mul", load("rosen_r6_acc.jsonl"))):
        for mod in sorted({r["model"] for r in rows}):
            p = {c: mean([r["correct"] for r in rows if r["model"] == mod and r["cond"] == c]) for c in ("table", "none")}
            mh = p["table"] / p["none"] if p["none"] > 0 else float("inf")
            out[key][mod] = dict(p_table=p["table"], p_none=p["none"], m_hat=mh,
                                 F_star=16 * (1 - p["none"] / p["table"]) if p["table"] > 0 else None)
    if pops:
        tooled, untooled = [], []
        for r in pops:
            for h in r["history"]:
                for act in h["acts"].values():
                    if act[0] == "do":
                        (tooled if act[4] else untooled).append(act[3])
        out["pop_accuracy"] = dict(tooled=mean(tooled), n_tooled=len(tooled), untooled=mean(untooled), n_untooled=len(untooled))
        out["H2"] = h2_corner(pops)
        out["H3"] = step_vs_line([r["F"] for r in pops], [metrics(r)["dol"] for r in pops])
        out["arm_means"] = {F: {k: mean([metrics(r)[k] for r in v]) for k in ("dol", "build_rate", "benches", "points", "use_share")}
                            for F, v in sorted(by_arm(pops, lambda r: r["F"]).items())}
    return out


def sec_r3(B, rng):
    pops = load("rosen_r3_pop.jsonl")
    if not pops:
        return None
    out = {}
    for mod in sorted({r["model"] for r in pops}):
        rr = [(r, rosen_p2.bm_metrics(r)) for r in pops if r["model"] == mod]
        a_dol = strat_trend([(r["J"], r["c"], m["dol"]) for r, m in rr], -1, B, rng)
        a_hand = strat_trend([(r["J"], r["c"], m["handoff_share"]) for r, m in rr], -1, B, rng)
        b_J = strat_perm_greater([(r["c"], int(r["J"] == 4), m["dol"]) for r, m in rr], B, rng)
        cells = {f"J={J}|c={c}": {k: mean([m[k] for r, m in rr if r["J"] == J and r["c"] == c])
                                  for k in ("dol", "handoff_share", "types_per_agent", "benches", "points", "invalid_rate")}
                 for J in (2, 4) for c in (0.0, 0.5, 1.0, 2.0)}
        out[mod] = dict(a_dol_falls_with_c=a_dol, a_handoff_falls_with_c=a_hand, b_dol_higher_J4=b_J, cells=cells,
                        holm=holm(dict(a_dol=a_dol["p"], a_hand=a_hand["p"], b_J=b_J["p"])))
    return out


def sec_r4(B, rng):
    pops = load("rosen_r4_pop.jsonl")
    if not pops:
        return None
    out = {}
    for mod in sorted({r["model"] for r in pops}):
        rr = [(r, rosen_p2.yb_metrics(r)) for r in pops if r["model"] == mod]
        a = strat_trend([(r["horizon"], r["lam"], m["dol"]) for r, m in rr], 1, B, rng)
        b = strat_perm_greater([(r["lam"], int(r["horizon"] == 24), m["dol"]) for r, m in rr if r["lam"] > 0], B, rng)
        z0 = [m["dol"] for r, m in rr if r["lam"] == 0 and r["horizon"] == 24]
        z8 = [m["dol"] for r, m in rr if r["lam"] == 0 and r["horizon"] == 8]
        c = perm_greater(z0, z8, B, rng)
        cells = {f"lam={l}|T={h}": {k: mean([m[k] for r, m in rr if r["lam"] == l and r["horizon"] == h])
                                    for k in ("dol", "dol_late", "top_share", "points", "invalid_rate")}
                 for l in (0.0, 0.25, 0.5) for h in (8, 24)}
        out[mod] = dict(a_dol_rises_with_lam=a, b_horizon_at_lam_pos=b,
                        c_horizon_at_lam0=dict(c, p_two_sided=min(1.0, 2 * min(c["p"], 1 - c["p"])) if c else None),
                        cells=cells, holm=holm(dict(a=a["p"], b=b["p"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(11)
    out = {}
    for name, f in (("R2x", sec_r2x), ("R2r", sec_r2r), ("R2b", sec_r2b), ("R5", sec_r5), ("R6", sec_r6),
                    ("R3", sec_r3), ("R4", sec_r4)):
        res = f(a.B, rng)
        if res is not None:
            out[name] = res
            print("==", name, json.dumps(res, default=float)[:900], flush=True)
    json.dump(out, open(R / "rosen_p2_eval.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
