"""Couzin study: population simulator, Hartnett et al. (2016) check, frozen forecast, nulls and the M3 power rule.

The simulator plays the same game as couzin.py with agents drawn from probability tables (couzin_kernel.estimates):
each round, agent i sees k = min(5, active others) agents drawn without replacement, so j (number on A) is
hypergeometric given the counts; j is mapped to the 5-neighbour scale (j * 5 / k) and the table is interpolated.

Modes: kernel (measured, isotonic), independent (round-0 probabilities every round: no social information),
linear (linear-voter family a = 1 fitted to the same decisions), hartnett (the Hartnett et al. 2016 local rule).

Usage:
  couzin_forecast.py hartnett                 the simulator check (must pass before any freeze)
  couzin_forecast.py freeze [--draws 200 --per 100]   write-once results/couzin_forecast.json (after M1 + M2)
  couzin_forecast.py plan                     print the planned arms and group counts
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import couzin as cz
import couzin_kernel as ck

R = cz.R
OUTS = ("consensusA", "majA", "cohesion", "split", "amp", "follow_extra", "follow_base")
H10_MARGIN = 0.10


# ---------------------------------------------------------------- planned arms
def planned(m3_groups=40, m5_levels=(0.2, 0.4)):
    """[(arm, model_key, groups)]; model_key 'base' (primary), 'm4b' (rule-selected second model)."""
    P = []
    for N, p in [(5, .2), (10, .1), (10, .2), (20, .05), (20, .1), (20, .2), (40, .05), (40, .1)]:
        P.append((f"M3_N{N}_p{p}", "base", m3_groups))
    P += [("M3_N10_p0", "base", 12), ("M3_N40_p0", "base", 12), ("M3_N20_p0.1_w", "base", m3_groups)]
    P += [("M4_sim_nu0", "base", 24)] + [(f"M4_{pr}_nu{n}", "base", 24) for pr in ("sim", "obs") for n in (6, 12, 24)]
    P += [("M4ref_sim_nu0", "base", 24), ("M4ref_obs_nu12", "base", 24)]
    P += [("M4b_sim_nu0", "m4b", 24)] + [(f"M4b_{pr}_nu{n}", "m4b", 24) for pr in ("sim", "obs") for n in (12, 24)]
    P += [(f"M5_N{N}_p{p}", "base", 20) for N in (10, 40) for p in m5_levels]
    P += [("M6_N10_n1", "base", 30), ("M6_N40_n2", "base", 30)]
    P += [("M7_N10", "base", 30), ("M7_N20", "base", 30)]
    return P


CALIBRATION = [("M2_N10_p0.1", "base", 24), ("M2c_obs_nu6", "base", 8), ("M2c_obs_nu6", "m4b", 8)]


# ---------------------------------------------------------------- tables
def linear_tables(model):
    """Linear-voter null: logit P = logit((j+.5)/6) + beta * own (+ c), fitted by MLE per table family."""
    D = list(ck.decisions(model))
    lg = lambda z: 1 / (1 + np.exp(-z))
    L = np.array([ck.logit_j(j) for j in range(cz.K + 1)])
    E = ck.estimates(ck.tables(model))                    # keep round-0 entries and claims as measured
    for ph, sel in ((0, lambda t: t <= 2), (1, lambda t: t >= 3)):
        f = ck.fit_family([(d["j"], 1 if d["own"] == "A" else -1, d["a"]) for d in D
                           if d["role"] == "U" and d["t"] > 0 and d["own"] and sel(d["t"])], fix_a=1.0)
        E["Uph"][ph] = lg(L + f["beta"] + f["c"])
    f = ck.fit_family([(d["j"], 0, d["a"]) for d in D if d["role"] == "U" and d["t"] > 0 and d["own"] is None], fix_a=1.0)
    E["Uent"] = lg(L + f["c"])
    for w in ck.OM:
        rows = [(d["j"], 1 if d["own"] == "A" else -1, d["a"]) for d in D
                if d["role"] == "I" and d["omega"] == w and d["t"] > 0 and d["own"]]
        if rows:
            f = ck.fit_family(rows, fix_a=1.0)
            E["I1"][w] = lg(L + f["beta"] + f["c"]); E["I2"][w] = lg(L - f["beta"] + f["c"])
    return E


def hartnett_tables(b, k=4, omegas=(1.5, 2.0)):
    """Hartnett rule on k neighbours: P(+1) = (1 + tanh(b h) / tanh(b)) / 2, h = (w n+ - n-) / (w n+ + n-).
    Relative to the preferred option (w' = 1 / w for agents biased toward -1); own state does not matter."""
    j = np.arange(k + 1)

    def G(w):
        h = (w * j - (k - j)) / np.maximum(w * j + (k - j), 1e-12)
        return 0.5 * (1 + np.tanh(b * h) / np.tanh(b))
    E = dict(U0=0.5, Uph=np.stack([G(1.0), G(1.0)]), Uent=G(1.0), I0={}, I1={}, I2={}, CL=None)
    for w in omegas:
        E["I0"][w] = 0.5; E["I1"][w] = G(w); E["I2"][w] = G(w)
    return E


# ---------------------------------------------------------------- simulator
def interp(tab, x):
    """tab: (L,) over the table's own neighbour scale 0..L-1; x: float array on that scale."""
    L = len(tab) - 1
    x = np.clip(x, 0, L); lo = np.floor(x).astype(int); hi = np.minimum(lo + 1, L); fr = x - lo
    return tab[lo] * (1 - fr) + tab[hi] * fr


def simulate(agents, E_by_model, G, rng, protocol="sim", start="fresh", claims=False, mode="kernel", T=cz.T,
             k=cz.K, init=None):
    """agents: list of (model, pref in {None,'A','B'}, omega). Returns dict of outcome arrays (G,)."""
    N = len(agents)
    pref = np.array([0 if p is None else (1 if p == "A" else -1) for _, p, _ in agents])
    a_is_l0 = rng.random(G) < .5
    s = np.full((G, N), -1, np.int8)

    def p_round0(i):
        m, p, w = agents[i]; E = E_by_model[m]
        if p is None:
            q = E["U0"]; return np.where(a_is_l0, q, 1 - q)
        q = E["I0"][w]; return np.full(G, q if p == "A" else 1 - q)
    if start == "convention":
        s[:] = 0; t0 = 1
    else:
        t0 = 0
        for i in range(N):
            if protocol == "obs" and pref[i] == 0:
                continue
            pr = init if init is not None else p_round0(i)
            s[:, i] = rng.random(G) < pr
    first = s.copy()
    claimant = (pref != 0) & claims
    for t in range(max(t0, 1), T + 1):
        act = s >= 0
        nA = (s == 1).sum(1); nB = (s == 0).sum(1)
        cA = ((s == 1) & claimant).sum(1); cB = ((s == 0) & claimant).sum(1)
        new = s.copy()
        for i in range(N):
            m, p, w = agents[i]; E = E_by_model[m]
            own = s[:, i]
            nAo = nA - (own == 1); nBo = nB - (own == 0)
            kk = np.minimum(k, nAo + nBo)
            ok = kk > 0
            j = np.zeros(G, int)
            if ok.any():
                j[ok] = rng.hypergeometric(nAo[ok], nBo[ok], kk[ok])
            kt_u = len(E["Uph"][0]) - 1
            x = np.where(ok, j * kt_u / np.maximum(kk, 1), kt_u / 2)
            if mode == "independent":
                pr = p_round0(i) if not (protocol == "obs" and p is None) else np.where(a_is_l0, E["U0"], 1 - E["U0"])
            elif p is None:
                ph = 0 if t <= 2 else 1
                f = E["Uph"][ph]
                pr = np.where(own == 1, interp(f, x), np.where(own == 0, 1 - interp(f, kt_u - x), interp(E["Uent"], x)))
                if claims and E.get("CL") is not None:
                    cAo = cA - ((own == 1) & claimant[i]); cBo = cB - ((own == 0) & claimant[i])
                    vA = np.zeros(G, int); vB = np.zeros(G, int)
                    mA = ok & (j > 0) & (cAo > 0)
                    vA[mA] = rng.hypergeometric(cAo[mA], (nAo - cAo)[mA], j[mA])
                    mB = ok & (kk - j > 0) & (cBo > 0)
                    vB[mB] = rng.hypergeometric(cBo[mB], (nBo - cBo)[mB], (kk - j)[mB])
                    CL = E["CL"]
                    onA = (vA > 0) & (vB == 0) & (own >= 0)
                    onB = (vB > 0) & (vA == 0) & (own >= 0)
                    pa = np.where(own == 1, interp(CL[0], x), interp(CL[1], x))           # claimant on A
                    pb = 1 - np.where(own == 0, interp(CL[0], kt_u - x), interp(CL[1], kt_u - x))   # fold
                    pr = np.where(onA, pa, np.where(onB, pb, pr))
            else:
                xr = x if p == "A" else kt_u - x
                ownr = own if p == "A" else np.where(own >= 0, 1 - own, -1)
                q = np.where(ownr == 1, interp(E["I1"][w], xr), interp(E["I2"][w], xr))
                pr = q if p == "A" else 1 - q
            mv = np.ones(G, bool)
            new[:, i] = np.where(mv, rng.random(G) < pr, own)
        s = new
    shareA = (s == 1).mean(1)
    act0 = (first >= 0).sum(1)
    s0 = np.where(act0 > 0, (first == 1).sum(1) / np.maximum(act0, 1), .5)
    maj = lambda v: np.where(v > .5, 1.0, np.where(v == .5, .5, 0.0))
    out = dict(consensusA=(shareA >= .8).astype(float), majA=maj(shareA), cohesion=np.maximum(shareA, 1 - shareA),
               split=(np.maximum(shareA, 1 - shareA) < .8).astype(float), amp=maj(shareA) - maj(s0), shareA=shareA)
    models = [a[0] for a in agents]
    if len(set(models)) > 1:                                  # M7: extra model is agents[0]
        fm = maj(shareA)
        out["follow_extra"] = np.where(fm == .5, .5, (fm == first[:, 0]).astype(float))
        jx = rng.integers(1, N, G)
        out["follow_base"] = np.where(fm == .5, .5, (fm == first[np.arange(G), jx]).astype(float))
    return out


def agents_for(aid, base, extra=None, omegas=None):
    spec = cz.arm(aid)
    out = []
    for kind, p, wk in spec["agents"]:
        m = extra if kind == "extra" else base
        w = None if p is None else omegas[m][0 if wk == "weak" else 1]
        out.append((m, p, w))
    return out, spec


# ---------------------------------------------------------------- Hartnett check
def hartnett_check(G=300, N=200, T=150, seed=7):
    """Pass iff with Hartnett's rule (k = 4 random neighbours, weak majority 60% of biased at w+ = 1.5, strong
    minority at 1/w- = 2) the slope of P(majority wins) over rho0 in {0, .3, .6, .9} is > 0 at b = 0.3 (near-linear:
    unbiased individuals favour the majority) and < 0 at b = 10 (highly nonlinear: they favour the minority)."""
    rng = np.random.default_rng(seed)
    res = {}
    for b in (0.3, 10.0):
        E = {"h": hartnett_tables(b, omegas=(1.5, 2.0))}
        ps = []
        for rho0 in (0, .3, .6, .9):
            nb = round(N * (1 - rho0)); nmaj = round(nb * .6)
            ag = [("h", "A", 1.5)] * nmaj + [("h", "B", 2.0)] * (nb - nmaj) + [("h", None, None)] * (N - nb)
            o = simulate(ag, E, G, rng, T=T, k=4, init=0.5)
            ps.append(float(o["majA"].mean()))
        slope = float(np.polyfit([0, .3, .6, .9], ps, 1)[0])
        res[b] = dict(p_majority=ps, slope=slope)
    ok = res[0.3]["slope"] > 0 and res[10.0]["slope"] < 0
    return dict(passed=ok, results={str(k): v for k, v in res.items()},
                rule="slope of P(majority) over rho0 > 0 at b=0.3 and < 0 at b=10")


# ---------------------------------------------------------------- forecast
def forecast_arm(aid, base, extra, E_draws, E_point, omegas, groups, rng, per):
    ag, spec = agents_for(aid, base, extra, omegas)
    kw = dict(protocol=spec["protocol"], start=spec["start"], claims=spec["claims"])
    draws = {o: [] for o in OUTS}
    pooled = {o: [] for o in OUTS}
    for Ed in E_draws:
        o = simulate(ag, Ed, per, rng, **kw)
        for key in OUTS:
            if key in o:
                draws[key].append(float(o[key].mean())); pooled[key].append(o[key])
    res = dict(groups=groups)
    for key in OUTS:
        if not draws[key]:
            continue
        arr = np.array(draws[key]); allg = np.concatenate(pooled[key])
        # arm mean over `groups` groups: kernel draw first, then groups from that draw (uncertainty + sampling)
        bs = rng.integers(0, len(arr), 2000)
        sims = np.array([rng.choice(pooled[key][b], groups).mean() for b in bs])
        res[key] = dict(point=float(allg.mean()), kernel_lo=float(np.quantile(arr, .05)), kernel_hi=float(np.quantile(arr, .95)),
                        lo=float(np.quantile(sims, .05)), hi=float(np.quantile(sims, .95)))
    for mode, E in (("independent", E_point), ("linear", None)):
        if mode == "linear":
            E = {m: linear_tables(m) for m in E_point}
        o = simulate(ag, E, per * 20, rng, mode="independent" if mode == "independent" else "kernel", **kw)
        res[f"null_{mode}"] = {key: float(o[key].mean()) for key in OUTS if key in o}
    return res


def h1_power(q, groups, reps=1000, seed=11):
    """Power of the H1 test (one-sided Wald z of the log2 N slope, p fixed effects, alpha .05) given cell rates q."""
    rng = np.random.default_rng(seed)
    cells = [(5, .2), (10, .1), (10, .2), (20, .05), (20, .1), (20, .2), (40, .05), (40, .1)]
    ps = sorted({p for _, p in cells})
    X = np.array([[np.log2(N)] + [float(p == pp) for pp in ps] for N, p in cells for _ in range(groups)])
    hits = 0
    for _ in range(reps):
        y = np.concatenate([rng.random(groups) < q[c] for c in cells]).astype(float)
        b, se = logit_fit(X, y)
        hits += (b[0] / se[0] > 1.645) if np.isfinite(se[0]) else 0
    return hits / reps


def logit_fit(X, y, it=50):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p)); b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = p * (1 - p) + 1e-9
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


def freeze(draws=200, per=100, seed=2026, pilot=False):
    out = R / ("couzin_pilot_forecast.json" if pilot else "couzin_forecast.json")
    if out.exists() and not pilot:
        sys.exit(f"{out} exists (write-once)")
    hc = hartnett_check()
    print("Hartnett check:", json.dumps(hc))
    if not hc["passed"]:
        sys.exit("Hartnett check failed: simulator not validated; no forecast written")
    rules = json.loads((R / ("couzin_pilot_m1_rules.json" if pilot else "couzin_m1_rules.json")).read_text())
    base, m4b, extra = cz.PRIMARY, rules["m4b_model"], "adaption:deepseek-v4-flash"
    rng = np.random.default_rng(seed)
    models = [m for m in (base, m4b, extra) if m]
    T = {m: ck.tables(m, pilot=pilot) for m in models}
    E_point = {m: ck.estimates(T[m]) for m in models}
    E_draws = [{m: ck.estimates(T[m], rng) for m in models} for _ in range(draws)]
    omegas = {m: (rules["omega"][m]["weak"], rules["omega"][m]["strong"]) if m in rules["omega"] else (0.5, 6.0) for m in models}
    # M5 levels: forecast critical mass at N = 10 under the convention start
    crit = {}
    for p in [round(x, 1) for x in np.arange(.1, 1.0, .1)]:
        ag, spec = agents_for(f"M5_N10_p{p}", base, None, omegas)
        o = simulate(ag, E_point, 4000, rng, start="convention")
        crit[p] = float(o["consensusA"].mean())
    pc = next((p for p, v in crit.items() if v >= .5), None)
    if pc is None:
        m5 = (0.8, 0.9); m5_flag = "no p reaches 0.5 consensus under the convention start; levels 0.8 / 0.9"
    else:
        lo = max(round(pc - .1, 1), .1); hi = min(round(lo + .2, 1), .9)
        m5 = (lo, hi); m5_flag = None
    # M3 power rule
    q3 = {}
    for N, p in [(5, .2), (10, .1), (10, .2), (20, .05), (20, .1), (20, .2), (40, .05), (40, .1)]:
        ag, spec = agents_for(f"M3_N{N}_p{p}", base, None, omegas)
        q3[(N, p)] = float(simulate(ag, E_point, 20000, rng)["consensusA"].mean())
    power = {g: h1_power(q3, g) for g in (30, 35, 40)}
    g3 = next((g for g in (30, 35, 40) if power[g] >= .8), 40)
    h1_status = "verdict" if power[40] >= .7 else "estimate (power at 40 groups < 0.7)"
    arms = {}
    for aid, mk, g in planned(g3, m5):
        mdl = base if mk == "base" else m4b
        if mdl is None:
            continue
        t1 = time.time()
        arms[aid] = dict(model=mdl, **forecast_arm(aid, mdl, extra if aid.startswith("M7") else None,
                                                 E_draws, E_point, omegas, g, rng, per))
        print(f"{aid}: consensusA {arms[aid].get('consensusA', {}).get('point', float('nan')):.3f} "
              f"majA {arms[aid]['majA']['point']:.3f} [{arms[aid]['majA']['lo']:.2f}, {arms[aid]['majA']['hi']:.2f}] "
              f"({time.time() - t1:.0f}s)", flush=True)
    counts_hash = hashlib.sha256(json.dumps({m: {k: (v.tolist() if isinstance(v, np.ndarray) else {str(w): x.tolist() for w, x in v.items()})
                                                  for k, v in T[m].items()} for m in models}, sort_keys=True).encode()).hexdigest()
    d = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), hartnett=hc, rules=rules, omegas=omegas, m4b_model=m4b,
             kernel_counts_sha256=counts_hash, m3_groups=g3, h1_power=power, h1_status=h1_status,
             m3_point={f"N{N}_p{p}": v for (N, p), v in q3.items()}, m5_levels=list(m5), m5_flag=m5_flag,
             m5_critical_curve=crit, h10_margin=H10_MARGIN, draws=draws, per=per, arms=arms)
    out.write_text(json.dumps(d, indent=1))
    print(f"forecast written: {out}  (M3 groups {g3}, power {power}, M5 levels {m5})")
    return d


def in_stage(aid, stage):
    if stage is None:
        return True
    if stage == "M4":
        return aid.startswith(("M4_", "M4ref_"))
    return aid.startswith(stage + "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["hartnett", "freeze", "plan"])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--per", type=int, default=100)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--stage", default=None, help="plan: only arms of this stage (M3, M4, M4b, M5, M6, M7)")
    a = ap.parse_args()
    if a.what == "hartnett":
        r = hartnett_check(); print(json.dumps(r, indent=1)); sys.exit(0 if r["passed"] else 1)
    if a.what == "plan":
        f = R / "couzin_forecast.json"
        if f.exists():
            d = json.loads(f.read_text())
            for aid, v in d["arms"].items():
                if in_stage(aid, a.stage):
                    print(aid, v["model"], v["groups"])
        else:
            for x in planned():
                print(*x)
        return
    freeze(a.draws, a.per, pilot=a.pilot)


if __name__ == "__main__":
    main()
