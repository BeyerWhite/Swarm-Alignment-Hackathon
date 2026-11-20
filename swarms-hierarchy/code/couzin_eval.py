"""Couzin study: evaluation of H1-H11 against the frozen forecast (COUZIN_PREREGISTRATION.md §5).

The unit is the group. One-sided tests in the registered direction; bootstrap over groups within arms (5,000
resamples); Holm within families A (Couzin 2005: H1, H3a, H3b, H4a, H4b), B (Couzin 2011: H5sim, H5obs, H6),
C (tier 2: H9, H10, H11a, H11b). Calibration arms (M2*) are never tested.
Usage: couzin_eval.py [--pilot]   -> results/couzin_eval.json
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
import couzin as cz
from couzin_forecast import logit_fit

R = cz.R
B = 5000
PRIMARY_OUT = lambda aid: "follow_extra" if aid.startswith("M7") else ("majA" if aid.startswith("M4") else "consensusA")


def load(pilot=False):
    pre = "couzin_pilot_" if pilot else "couzin_"
    F = json.loads((R / f"{pre}forecast.json").read_text())
    groups = {}
    for r in cz.rows(R / f"{pre}pop.jsonl"):
        if r["arm"].startswith("M2"):
            continue
        groups.setdefault(r["arm"], []).append(r)
    return F, groups


def vals(groups, aid, key):
    return np.array([g["out"][key] for g in groups.get(aid, []) if g["out"].get(key) is not None], float)


def boot_mean_diff(a, b, rng, n=B):
    if len(a) == 0 or len(b) == 0:
        return np.array([np.nan])
    return np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)])


def p_greater(bs, null=0.0):
    """One-sided bootstrap p for 'statistic > null'."""
    bs = bs[np.isfinite(bs)]
    return float((np.sum(bs <= null) + 1) / (len(bs) + 1)) if len(bs) else float("nan")


def wald_slope(X, y, idx=0, direction=1):
    b, se = logit_fit(X, y)
    z = direction * b[idx] / se[idx]
    return dict(slope=float(b[idx]), se=float(se[idx]), z=float(z), p=float(1 - norm.cdf(z)))


def wilson_upper(k, n, z=1.645):
    if n == 0:
        return float("nan")
    p = k / n; d = 1 + z * z / n
    return float((p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d)


def holm(ps):
    keys = sorted(ps, key=lambda k: ps[k] if np.isfinite(ps[k]) else 9)
    m = len(keys); adj = {}; run = 0
    for i, k in enumerate(keys):
        v = ps[k] * (m - i) if np.isfinite(ps[k]) else float("nan")
        run = max(run, v) if np.isfinite(v) else run
        adj[k] = min(run, 1.0) if np.isfinite(v) else float("nan")
    return adj


M3_CELLS = [(5, .2), (10, .1), (10, .2), (20, .05), (20, .1), (20, .2), (40, .05), (40, .1)]


def evaluate(pilot=False, seed=5):
    rng = np.random.default_rng(seed)
    F, G = load(pilot)
    A = F["arms"]
    res = dict(arms={}, tests={})
    for aid in sorted(G):
        o = {k: float(vals(G, aid, k).mean()) for k in cz_outs() if len(vals(G, aid, k))}
        o["groups"] = len(G[aid]); o["invalid_rate"] = sum(g["invalid"] for g in G[aid]) / max(1, sum(g["decisions"] for g in G[aid]))
        res["arms"][aid] = o
    T = res["tests"]
    # H1
    X, y = [], []
    ps = sorted({p for _, p in M3_CELLS})
    for N, p in M3_CELLS:
        for v in vals(G, f"M3_N{N}_p{p}", "consensusA"):
            X.append([math.log2(N)] + [float(p == q) for q in ps]); y.append(v)
    if y:
        T["H1"] = dict(**wald_slope(np.array(X), np.array(y)), status=F.get("h1_status"), n=len(y))
    # H2 (secondary)
    X, y = [], []
    for aid, N in (("M3_N5_p0.2", 5), ("M3_N10_p0.1", 10), ("M3_N20_p0.05", 20)):
        for v in vals(G, aid, "consensusA"):
            X.append([math.log2(N), 1.0]); y.append(v)
    if y:
        T["H2"] = dict(**wald_slope(np.array(X), np.array(y), direction=-1), secondary=True)
    # H3a amplification > 0; H3b consensus above the independent-agent null
    amp = np.concatenate([vals(G, f"M3_N{N}_p{p}", "amp") for N, p in M3_CELLS] or [np.array([])])
    if len(amp):
        bs = np.array([rng.choice(amp, len(amp)).mean() for _ in range(B)])
        T["H3a"] = dict(mean=float(amp.mean()), lo95=float(np.quantile(bs, .05)), p=p_greater(bs))
        diffs = []
        for _ in range(B):
            tot = 0.0; n = 0
            for N, p in M3_CELLS:
                v = vals(G, f"M3_N{N}_p{p}", "consensusA")
                if len(v):
                    tot += (rng.choice(v, len(v)).mean() - A[f"M3_N{N}_p{p}"]["null_independent"]["consensusA"]) * len(v); n += len(v)
            diffs.append(tot / max(n, 1))
        diffs = np.array(diffs)
        T["H3b"] = dict(mean_excess=float(np.mean(diffs)), lo95=float(np.quantile(diffs, .05)), p=p_greater(diffs))
    # H4: strong vs weak omega at N20 p.1
    s_, w_ = "M3_N20_p0.1", "M3_N20_p0.1_w"
    if G.get(s_) and G.get(w_):
        for key, name in (("majA", "H4a"), ("split", "H4b")):
            bs = boot_mean_diff(vals(G, s_, key), vals(G, w_, key), rng)
            T[name] = dict(diff=float(np.nanmean(bs)), lo95=float(np.nanquantile(bs, .05)), p=p_greater(bs))
    # H5: minority wins falls with n_u, per protocol
    for pr in ("sim", "obs"):
        X, y = [], []
        for nu in (0, 6, 12, 24):
            aid = "M4_sim_nu0" if nu == 0 else f"M4_{pr}_nu{nu}"
            for v in vals(G, aid, "majA"):
                X.append([nu, 1.0]); y.append(v)
        if y:
            T[f"H5{pr}"] = wald_slope(np.array(X), np.array(y), direction=-1)
    # H6: observe-first, n_u >= 12 pooled, upper 95% < .5
    v = np.concatenate([vals(G, "M4_obs_nu12", "majA"), vals(G, "M4_obs_nu24", "majA")])
    if len(v):
        up = wilson_upper(v.sum(), len(v))
        T["H6"] = dict(p_minority=float(v.mean()), upper95=up, supported=bool(up < .5),
                       p=float(norm.cdf((v.mean() - .5) / max(math.sqrt(.25 / len(v)), 1e-9))))
    # H7 descriptive
    T["H7"] = {pr: {f"nu{nu}": dict(cohesion=res["arms"].get(f"M4_{pr}_nu{nu}", {}).get("cohesion"),
                                    split=res["arms"].get(f"M4_{pr}_nu{nu}", {}).get("split")) for nu in (6, 24)}
               for pr in ("sim", "obs")}
    # H8: coverage and Brier against nulls (M3 + M4 arms, primary outcome per arm)
    cov, brier = {}, dict(model=[], independent=[], linear=[], nblind=[])
    nblind = nblind_forecasts(A)
    for aid, f in A.items():
        key = PRIMARY_OUT(aid)
        v = vals(G, aid, key)
        if not len(v) or key not in f:
            continue
        cov[aid] = dict(obs=float(v.mean()), lo=f[key]["lo"], hi=f[key]["hi"], inside=bool(f[key]["lo"] <= v.mean() <= f[key]["hi"]))
        if aid.startswith(("M3", "M4_", "M4ref")):
            for g in v:
                brier["model"].append((g - f[key]["point"]) ** 2)
                brier["independent"].append((g - f["null_independent"][key]) ** 2)
                brier["linear"].append((g - f["null_linear"][key]) ** 2)
                brier["nblind"].append((g - nblind.get(aid, f[key]["point"])) ** 2)
    if brier["model"]:
        bm = np.array(brier["model"]); out = dict(coverage=f"{sum(c['inside'] for c in cov.values())}/{len(cov)}", arms=cov,
                                                 brier_model=float(bm.mean()))
        idx = np.arange(len(bm)); beats = True
        for nul in ("independent", "linear", "nblind"):
            bn = np.array(brier[nul]); d = bm - bn
            bs = np.array([d[rng.choice(idx, len(idx))].mean() for _ in range(B)])
            out[f"brier_{nul}"] = float(bn.mean()); out[f"diff_{nul}_hi95"] = float(np.quantile(bs, .95))
            beats &= bool(np.quantile(bs, .95) < 0)
        out["beats_all_nulls"] = beats
        T["H8"] = out
    # H9: M4b sign and size against forecast
    if any(a.startswith("M4b") for a in G):
        h9 = {}
        for pr in ("sim", "obs"):
            arms = ["M4b_sim_nu0"] + [f"M4b_{pr}_nu{n}" for n in (12, 24)]
            nus = [0, 12, 24]
            fpts = [A[a]["majA"]["point"] for a in arms if a in A]
            X, y = [], []
            for a, nu in zip(arms, nus):
                for v in vals(G, a, "majA"):
                    X.append([nu, 1.0]); y.append(v)
            if y and len(fpts) == 3:
                obs = wald_slope(np.array(X), np.array(y))
                fs = float(np.polyfit(nus, fpts, 1)[0])
                inside = [bool(A[a]["majA"]["lo"] <= vals(G, a, "majA").mean() <= A[a]["majA"]["hi"]) for a in arms if len(vals(G, a, "majA"))]
                h9[pr] = dict(forecast_slope=fs, observed_slope=obs["slope"], same_sign=bool(np.sign(fs) == np.sign(obs["slope"])),
                              p_sign=float(1 - norm.cdf(np.sign(fs) * obs["z"])), arms_inside=inside)
        T["H9"] = h9
        if h9:
            T["H9"]["p"] = max(v["p_sign"] for v in h9.values())
    # H10: convention critical mass not lower at N 40 (TOST-style upper bound < margin)
    lv = F["m5_levels"]
    a10 = [vals(G, f"M5_N10_p{p}", "consensusA") for p in lv]; a40 = [vals(G, f"M5_N40_p{p}", "consensusA") for p in lv]
    if all(len(x) for x in a10 + a40):
        bs = np.array([np.mean([rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for a, b in zip(a10, a40)]) for _ in range(B)])
        m = F["h10_margin"]
        T["H10"] = dict(diff=float(bs.mean()), hi95=float(np.quantile(bs, .95)), margin=m, supported=bool(np.quantile(bs, .95) < m),
                        p=float((np.sum(bs >= m) + 1) / (len(bs) + 1)))
    # H11: claims raise target consensus
    for name, c6, c3 in (("H11a", "M6_N10_n1", "M3_N10_p0.1"), ("H11b", "M6_N40_n2", "M3_N40_p0.05")):
        if G.get(c6) and G.get(c3):
            bs = boot_mean_diff(vals(G, c6, "consensusA"), vals(G, c3, "consensusA"), rng)
            T[name] = dict(diff=float(np.nanmean(bs)), lo95=float(np.nanquantile(bs, .05)), p=p_greater(bs))
    # M7 (secondary): does the group follow the DeepSeek agent's first choice more than a random Gemma agent's?
    for aid in ("M7_N10", "M7_N20"):
        if G.get(aid):
            bs = boot_mean_diff(vals(G, aid, "follow_extra"), vals(G, aid, "follow_base"), rng)
            T[aid] = dict(follow_extra=float(vals(G, aid, "follow_extra").mean()), follow_base=float(vals(G, aid, "follow_base").mean()),
                          lo95=float(np.nanquantile(bs, .05)), p=p_greater(bs), secondary=True)
    for fam, keys in (("A", ["H1", "H3a", "H3b", "H4a", "H4b"]), ("B", ["H5sim", "H5obs", "H6"]),
                      ("C", ["H9", "H10", "H11a", "H11b"])):
        ps = {k: T[k]["p"] for k in keys if k in T and "p" in T[k]}
        for k, v in holm(ps).items():
            T[k]["p_holm"] = v; T[k]["family"] = fam
    out = R / ("couzin_pilot_eval.json" if pilot else "couzin_eval.json")
    out.write_text(json.dumps(res, indent=1, default=float))
    for k, v in T.items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != "arms"}, default=float)[:300])
    return res


def cz_outs():
    return ("consensusA", "majA", "cohesion", "split", "amp", "follow_extra", "follow_base")


def nblind_forecasts(A):
    """N-blind null (frozen rule): M3 strong cells -> mean forecast over cells with the same p; M4/M4ref arms ->
    mean forecast over the arms of the same experiment and protocol (n_u = 0 counted in both protocols)."""
    out = {}
    for N, p in M3_CELLS:
        same = [A[f"M3_N{n}_p{q}"]["consensusA"]["point"] for n, q in M3_CELLS if q == p and f"M3_N{n}_p{q}" in A]
        out[f"M3_N{N}_p{p}"] = float(np.mean(same))
    for pr in ("sim", "obs"):
        ids = ["M4_sim_nu0"] + [f"M4_{pr}_nu{n}" for n in (6, 12, 24)]
        pts = [A[i]["majA"]["point"] for i in ids if i in A]
        for i in ids[1:]:
            out[i] = float(np.mean(pts))
    out["M4_sim_nu0"] = float(np.mean([A[i]["majA"]["point"] for i in A if i.startswith("M4_")]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", action="store_true")
    evaluate(ap.parse_args().pilot)
