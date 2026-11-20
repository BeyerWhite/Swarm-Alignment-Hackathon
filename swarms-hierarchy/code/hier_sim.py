"""Forecast simulator and frozen choices for the hierarchy-efficiency study.

The simulator plays the office game (office.run_pop, the same engine as the LLM runs) with every LLM seat replaced
by a stochastic policy built from that model's measured kernels (results/hier_kernels.json):
  role      P(PLAN | own rank, own past planning, other planner status)      G1d, raw cell frequencies (unshrunk)
  plan      valid with P = G1b valid rate; optimal assignment with P = G1b mean q of valid plans, else random;
            number of agents assigned drawn from the model's G1b spans
  follow    P(do the assigned task | planner record bin, fit)               G1e, raw cell frequencies
  choice    otherwise: the first listed open task with P = first_listed (G1c 7-worker arms), else uniform
  answer    correct with P = skill[model][kind] (G1a); invalid reply with P = invalid rate (G1c)
Records: the simulator computes the role features from the true stats in both record versions, so the forecast
for the lean record equals that for the full record (S4 then tests a difference the kernels do not predict).

  hier_sim.py freeze      composition rule, G3 targets, G4 manager, forecasts for G2 + benchmarks  (write-once)
  hier_sim.py freeze_g3   forecasts for G3 from the disposition re-check kernels                  (write-once)
  hier_sim.py selftest    offline: synthetic kernels, engine + policies + forecast pipeline
"""
import argparse, hashlib, itertools, json, random, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import office as O
import hier_measure as HM

R = O.R
SIM_N = 400                  # simulated populations per cell
DRAWS = 2000                 # resamples for intervals of the registered-seed-count mean
SEEDS_G2, SEEDS_BENCH, SEEDS_G3 = 12, 6, 12
SIM_SEED0 = 900000


# ----------------------------------------------------------------------------------------------- policies
def sim_policy(model, K, role_override=None):
    sk, pq = K["skill"][model], K["planq"].get(model, dict(valid=1.0, q_valid=0.0, spans=[7]))
    role = role_override if role_override is not None else K["role"].get(model, {})
    fol = K["follow"].get(model, {})
    first = K.get("first_listed", {}).get(model, 0.5)
    inval = K.get("invalid", {}).get(model, 0.0)
    rmean = (sum(k for k, n in role.values()) / max(1, sum(n for k, n in role.values()))) if isinstance(role, dict) \
        else role

    def p_plan(st):
        if not isinstance(role, dict):
            return role                                   # random-role null: one rate for everyone
        a = st["me"]
        cell = f"{HM.rank_bin(st, a)}|{int(st['stats'][a]['coord'] > 0)}|{HM.other_status(st, a)}"
        k, n = role.get(cell, (0, 0))
        return k / n if n else rmean

    def plan(st, r):
        if r.random() >= pq["valid"]:
            return "I will coordinate."                    # an invalid reply
        others = [a for a in st["active_ids"] if a != st["me"]]
        tasks = st["open_tasks"]
        span = min(len(others), r.choice(pq["spans"]) if pq["spans"] else len(others))
        if r.random() < max(0.0, min(1.0, pq["q_valid"])):
            line = O.oracle_plan(dict(st, active_ids=[st["me"]] + others), K["skill"])
            pairs = [p for p in line[6:].split(", ") if p]
        else:
            pairs = [f"T{x['id']}->{a}" for x, a in zip(r.sample(tasks, min(len(tasks), len(others))),
                                                     r.sample(others, len(others)))]
        r.shuffle(pairs)
        return "PLAN: " + ", ".join(pairs[:max(1, span)])

    def act(st):
        r = random.Random(st["prompt_seed"] * 1e9 + 7)
        if r.random() < inval:
            return ""
        can_plan = st["arm"] == "free" or st["is_manager"]
        if st["is_manager"] or (can_plan and r.random() < p_plan(st)):
            return plan(st, r)
        by = {x["id"]: x for x in st["open_tasks"]}
        target = None
        if st["mine"]:
            p, tid = st["mine"][0]
            cell = f"{HM.qbin(st['stats'][p])}|{HM.fit_of(st, st['me'], by[tid]['kind'])}"
            k, n, _ = fol.get(cell, (1, 2, 0))
            if r.random() < (k / n if n else 0.5):
                target = tid
        if target is None:
            target = st["shown"][0]["id"] if r.random() < first else r.choice(st["shown"])["id"]
        task = by[target]
        return f"DO {target}: " + (task["answer"] if r.random() < sk.get(task["kind"], 0.5) else "999999")
    return act


def policies_for(seats, K, overrides=None):
    pol = {}
    for i, s in enumerate(seats):
        if s == "oracle":
            pol[i] = O.oracle_policy(K["skill"])
        elif s != "absent":
            pol[i] = sim_policy(s, K, (overrides or {}).get(s))
    return pol


def simulate(seats, arm, n, K, kinds, record="full", seed0=SIM_SEED0, overrides=None, disposition=()):
    out = []
    for i in range(n):
        rec = O.run_pop(seats, arm, seed0 + i, record=record, kinds=kinds, policies=policies_for(seats, K, overrides),
                        disposition=disposition)
        out.append(O.metrics(rec) | dict(seats=rec["seats"]))
    return out


# ----------------------------------------------------------------------------------------------- statistics
def sorting_index(m, seats, gain):
    """Spearman over active LLM seats between share of followed assignments and the seat model's G1c gain."""
    a = [x for x, s in seats.items() if s in gain]
    infl = [m["infl"].get(x, 0) for x in a]
    if sum(infl) == 0 or len({gain[seats[x]] for x in a}) < 2:
        return float("nan")
    return float(spearmanr(infl, [gain[seats[x]] for x in a]).statistic)


def summary(vals, n_mean, rng):
    v = np.array([x for x in vals if x == x], float)
    if len(v) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    means = [rng.choice(v, n_mean).mean() for _ in range(DRAWS)]
    return dict(mean=float(v.mean()), lo=float(np.quantile(means, 0.05)), hi=float(np.quantile(means, 0.95)),
                pop_lo=float(np.quantile(v, 0.05)), pop_hi=float(np.quantile(v, 0.95)), n=int(len(v)))


def cell_forecast(sims, comp, gain, assertive, n_mean, rng):
    best, most = max(comp, key=lambda m: gain[m]), max(comp, key=lambda m: assertive[m])
    f = {k: summary([s[k] for s in sims], n_mean, rng) for k in ("throughput", "plan_rate", "follow_rate",
                                                                 "mean_span", "collide")}
    f["best_manager_plan_share"] = summary([s["per_model"][best]["plan_share"] for s in sims], n_mean, rng)
    f["most_assertive_plan_share"] = summary([s["per_model"][most]["plan_share"] for s in sims], n_mean, rng)
    f["best_manager_infl_share"] = summary([s["per_model"][best]["infl_share"] for s in sims], n_mean, rng)
    f["sorting_index"] = summary([sorting_index(s, s["seats"], gain) for s in sims], n_mean, rng)
    return f, [s["throughput"] for s in sims]


# ----------------------------------------------------------------------------------------------- frozen rules
def choose_compositions(assertive, gain, models):
    """Frozen composition rule (pre-registration section G1 freeze).
    For every 4-model subset: rho = Spearman(assertiveness, gain) over its 4 models.
      aligned:    most assertive == best manager; maximize rho; ties -> larger gain gap between best and second
                  manager; then alphabetical. Fallback if no subset qualifies: maximize rho.
      misaligned: most assertive != best manager; minimize rho; ties -> larger assertiveness gap between the most
                  assertive model and the best manager; then alphabetical. Fallback: minimize rho."""
    rows = []
    for S in itertools.combinations(sorted(models), 4):
        a, g = [assertive[m] for m in S], [gain[m] for m in S]
        rho = float(spearmanr(a, g).statistic) if len(set(a)) > 1 and len(set(g)) > 1 else 0.0
        most, best = max(S, key=lambda m: (assertive[m], m)), max(S, key=lambda m: (gain[m], m))
        gs = sorted(g, reverse=True)
        rows.append(dict(S=list(S), rho=rho, most=most, best=best, ggap=gs[0] - gs[1],
                         agap=assertive[most] - assertive[best]))
    al = [r for r in rows if r["most"] == r["best"]] or rows
    mis = [r for r in rows if r["most"] != r["best"]] or rows
    aligned = sorted(al, key=lambda r: (-r["rho"], -r["ggap"], r["S"]))[0]
    misaligned = sorted(mis, key=lambda r: (r["rho"], -r["agap"], r["S"]))[0]
    return dict(aligned=aligned, misaligned=misaligned, table=rows)


def freeze():
    p = R / "hier_freeze.json"
    if p.exists():
        sys.exit(f"{p.name} exists (write-once)")
    K = json.load(open(R / "hier_kernels.json"))
    kinds = [tuple(k) for k in K["kinds"]]
    gate = json.load(open(R / "hier_g0_gate.json"))
    models = [m for m in O.MODELS if m in gate["passed_models"] and m in K.get("gain", {})]
    gain, assertive = K["gain"], K["assertiveness"]
    comp = choose_compositions(assertive, gain, models)
    comps = {"aligned": comp["aligned"]["S"], "misaligned": comp["misaligned"]["S"]}
    mis = comps["misaligned"]
    targets = dict(best=max(mis, key=lambda m: (gain[m], m)), worst=min(mis, key=lambda m: (gain[m], m)))
    g4 = max(models, key=lambda m: (gain[m], m))
    rng = np.random.default_rng(20261120)
    fc, Wsim = {}, {}
    for cname, c in comps.items():
        seats = O.seat_list(c)
        for record in ("full", "lean"):
            sims = simulate(seats, "free", SIM_N, K, kinds, record=record)
            fc[f"g2|{cname}|{record}"], Wsim[f"g2|{cname}|{record}"] = cell_forecast(sims, c, gain, assertive,
                                                                                    SEEDS_G2, rng)
        pooled = float(np.mean([s["plan_rate"] for s in simulate(seats, "free", SIM_N // 4, K, kinds)]))
        sims = simulate(seats, "free", SIM_N, K, kinds, overrides={m: pooled for m in c})
        fc[f"null|{cname}"], Wsim[f"null|{cname}"] = cell_forecast(sims, c, gain, assertive, SEEDS_G2, rng)
        for arm in ("noplan", "ownorder"):
            sims = simulate(seats, arm, SIM_N, K, kinds)
            fc[f"bench|{cname}|{arm}"], Wsim[f"bench|{cname}|{arm}"] = cell_forecast(sims, c, gain, assertive,
                                                                                   SEEDS_BENCH, rng)
        s2 = O.seat_list(c); s2.remove(O.worst_worker(c, K["skill"]))
        sims = simulate(["oracle"] + s2, "appointed", SIM_N, K, kinds)
        fc[f"bench|{cname}|oracle"], Wsim[f"bench|{cname}|oracle"] = cell_forecast(sims, c, gain, assertive,
                                                                                 SEEDS_BENCH, rng)
        # welfare loss forecast: L = (W* - W) / (W* - W_noplan), W* = best benchmark arm mean
        Wstar = max(np.mean(Wsim[f"bench|{cname}|{a}"]) for a in ("noplan", "ownorder", "oracle"))
        Wno = np.mean(Wsim[f"bench|{cname}|noplan"])
        for key in (f"g2|{cname}|full", f"g2|{cname}|lean", f"null|{cname}"):
            L = [(Wstar - w) / (Wstar - Wno) if Wstar > Wno else float("nan") for w in Wsim[key]]
            fc[key]["welfare_loss"] = summary(L, SEEDS_G2, rng)
    out = dict(kernels_sha256=hashlib.sha256(open(R / "hier_kernels.json", "rb").read()).hexdigest(),
               models=models, compositions=comps, composition_rule=comp, g3_targets=targets, g4_manager=g4,
               forecasts=fc, sim_n=SIM_N, code_sha256={f: hashlib.sha256(open(Path(__file__).parent / f, "rb").read())
                                                       .hexdigest() for f in ("office.py", "hier_sim.py",
                                                                              "hier_measure.py", "hier_tasks.py")})
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(dict(compositions=comps, g3_targets=targets, g4_manager=g4,
                          sha=hashlib.sha256(open(p, "rb").read()).hexdigest()), indent=1))


def freeze_g3():
    p = R / "hier_freeze_g3.json"
    if p.exists():
        sys.exit(f"{p.name} exists (write-once)")
    K = json.load(open(R / "hier_kernels.json"))
    kinds = [tuple(k) for k in K["kinds"]]
    fz = O.frozen()
    rows = [json.loads(l) for l in open(HM.KER)]
    over = {}
    for m in set(fz["g3_targets"].values()):
        cells = {}
        for r in rows:
            if r["part"] == "g3d" and r["model"] == m:
                c = "|".join(map(str, r["cell"]))
                k, n = cells.get(c, (0, 0)); cells[c] = (k + r["plan"], n + 1)
        over[m] = cells
    comp = fz["compositions"]["misaligned"]; seats = O.seat_list(comp)
    rng = np.random.default_rng(20261121)
    fc = {}
    for who, m in fz["g3_targets"].items():
        disp = tuple(i for i, s in enumerate(seats) if s == m)
        sims = simulate(seats, "free", SIM_N, K, kinds, overrides={m: over[m]}, disposition=disp)
        fc[f"g3|{who}"], _ = cell_forecast(sims, comp, K["gain"], K["assertiveness"], SEEDS_G3, rng)
    json.dump(dict(freeze_sha256=hashlib.sha256(open(R / "hier_freeze.json", "rb").read()).hexdigest(),
                   overrides=over, forecasts=fc), open(p, "w"), indent=1)
    print("G3 forecasts frozen:", hashlib.sha256(open(p, "rb").read()).hexdigest())


# ----------------------------------------------------------------------------------------------- self-test
def synthetic_kernels(kinds):
    rng = random.Random(3)
    K = dict(kinds=kinds, skill={}, planq={}, role={}, follow={}, first_listed={}, invalid={})
    for i, m in enumerate(O.MODELS):
        K["skill"][m] = {k: min(0.98, 0.35 + 0.1 * i + rng.uniform(-0.1, 0.1)) for k, _ in kinds}
        K["planq"][m] = dict(valid=0.9, q_valid=0.15 * i, spans=[3, 5, 7])
        base = [0.6, 0.05, 0.3, 0.1, 0.0, 0.2][i]
        K["role"][m] = {"|".join(map(str, c)): [int(round(16 * base)), 16] for c in HM.g1d_cells()}
        K["follow"][m] = {f"{q}|{f}": [14, 20, 0] for q in HM.QBINS for f in ("good", "poor")}
        K["first_listed"][m], K["invalid"][m] = 0.6, 0.02
    K["assertiveness"] = {m: K["role"][m]["top|0|none"][0] / 16 for m in O.MODELS}
    K["gain"] = {m: 0.2 * i for i, m in enumerate(O.MODELS)}
    return K


def selftest():
    import time
    kinds = [("multiply", 1), ("count", 2), ("weekday", 2), ("sort", 2)]
    K = synthetic_kernels(kinds)
    comp = choose_compositions(K["assertiveness"], K["gain"], O.MODELS)
    print("aligned", comp["aligned"]["S"], round(comp["aligned"]["rho"], 2))
    print("misaligned", comp["misaligned"]["S"], round(comp["misaligned"]["rho"], 2))
    c = comp["misaligned"]["S"]
    t0 = time.time()
    rng = np.random.default_rng(1)
    for arm in ("noplan", "ownorder", "free"):
        sims = simulate(O.seat_list(c), arm, 20, K, kinds)
        f, W = cell_forecast(sims, c, K["gain"], K["assertiveness"], 12, rng)
        print(arm, {k: round(v["mean"], 3) for k, v in f.items()})
    s2 = O.seat_list(c); s2.remove(O.worst_worker(c, K["skill"]))
    sims = simulate(["oracle"] + s2, "appointed", 20, K, kinds)
    print("oracle", round(np.mean([s["throughput"] for s in sims]), 3))
    print(f"{(time.time() - t0) / 80:.3f} s per simulated population")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["freeze", "freeze_g3", "selftest"])
    a = ap.parse_args()
    {"freeze": freeze, "freeze_g3": freeze_g3, "selftest": selftest}[a.what]()


if __name__ == "__main__":
    main()
