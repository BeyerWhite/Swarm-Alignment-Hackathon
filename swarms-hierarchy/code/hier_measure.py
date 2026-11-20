"""Single-agent measurements for the hierarchy-efficiency study (stages G0 and G1), on synthetic office-game states.

  battery    G0: accuracy of every model on every (kind, level) cell, DO format     -> results/hier_g0_battery.jsonl
  select     G0: frozen task-selection rule (hier_tasks.select), write-once         -> results/hier_g0_selection.json
  comp       G0: rules comprehension quiz, options shuffled per call               -> results/hier_g0_comprehension.jsonl
  gate       G0: parse and comprehension gates per model                            -> results/hier_g0_gate.json
  g1a        worker skill A[model][kind] on the chosen kinds                        -> results/hier_g1_kernel.jsonl
  g1b        manager skill offline: plan quality on synthetic records (scored against the true skills)
  g1d        role-taking: P(PLAN | own rank, own planning, other planner)
  g1e        following:   P(follow | planner record quality, fit of the assigned kind to own record)
  g3check    G1a and G1d re-measured with the disposition sentence for the two G3 target models (needs the freeze)
  summarize  kernels + G1c population results                                      -> results/hier_kernels.json
  reliability  G1c top-up rule: exit 3 if P(best manager stays best) < 0.6           -> results/hier_g1c_reliability.json

Synthetic states are built with the same prompt code as office.py (office.build_prompt / office.system_for).
"""
import argparse, json, os, random, re, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hier_tasks as HT
import office as O

R = O.R
KER = R / "hier_g1_kernel.jsonl"
_wlock = threading.Lock()
QBINS = ("new", 0.1, 0.3, 0.5, 0.7, 0.9)
RANKS = ("top", "middle", "bottom")
OTHER = ("none", "past", "live")


def _write(path, rows):
    with _wlock, open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _keys(path):
    return {json.loads(l)["key"] for l in open(path)} if path.exists() else set()


def _call(model, system, prompt, tag):
    from llm import call
    return call(model, system, prompt, tag=tag, max_tokens=O.MAX_TOKENS)


# ------------------------------------------------------------------------------------------ synthetic states
def base_state(rng, kinds, arm="free", record="full", t=9, wave=1):
    ids = rng.sample(range(10, 100), O.N)
    st = dict(me=ids[0], t=t, T=O.T, wave=wave, arm=arm, record=record, kinds=kinds,
              stats={a: O.new_stats(kinds) for a in ids}, plans_on=arm in ("free", "appointed"), live_plans={},
              mine=[], wave1_log=[], is_manager=False, manager_id=None, n_active=O.N, active_ids=ids,
              seat_model={}, prompt_seed=rng.random(), disposition=False)
    return st, ids


def fill_records(st, ids, rng, skill, models, skip=()):
    """Random records for agents not in skip: 0-4 attempts per kind, successes ~ Binomial(att, true skill)."""
    for a in ids:
        m = rng.choice(models)
        st["seat_model"][a] = m
        if a in skip:
            continue
        s = st["stats"][a]
        for k in st["kinds"]:
            n = rng.randint(0, 4)
            s["att"][k] = n
            s["ok"][k] = sum(rng.random() < skill[m][k] for _ in range(n))
        s["done"] = sum(s["ok"].values())


def open_tasks(rng, kinds_levels, start=40, only=None, per=O.PER_KIND):
    out, tid = [], start
    for k, lev in kinds_levels:
        if only and k != only:
            continue
        for _ in range(per):
            out.append(HT.make_task(tid, k, lev, rng)); tid += 1
    rng.shuffle(out)
    return out


def acc_of(s):
    att = sum(s["att"].values())
    return sum(s["ok"].values()) / att if att else 0.5


def rank_bin(st, a):
    """top = 1st-2nd of 8 by record accuracy, bottom = 7th-8th, else middle (ties broken against the agent)."""
    mine = acc_of(st["stats"][a])
    above = sum(acc_of(s) >= mine for b, s in st["stats"].items() if b != a)
    below = sum(acc_of(s) <= mine for b, s in st["stats"].items() if b != a)
    if above <= 1:
        return "top"
    if below <= 1:
        return "bottom"
    return "middle"


def other_status(st, a):
    if any(b in st["live_plans"] and any(x == a for _, x in st["live_plans"][b]) for b in st["live_plans"] if b != a):
        return "live"
    if any(s["coord"] > 0 for b, s in st["stats"].items() if b != a):
        return "past"
    return "none"


def qbin(s):
    if s["followed"] == 0:
        return "new"
    q = s["fcorrect"] / s["followed"]
    return min(QBINS[1:], key=lambda b: abs(b - q))


def fit_of(st, a, kind):
    s = st["stats"][a]
    if s["att"][kind] == 0:
        return "good"
    return "good" if s["ok"][kind] / s["att"][kind] >= acc_of(s) else "poor"


# ------------------------------------------------------------------------------------------ G0 battery
BATTERY_N = 20


def battery_jobs(models):
    J = []
    for m in models:
        for kind in HT.KINDS:
            for level in (1, 2):
                for i in range(BATTERY_N):
                    J.append(dict(key=f"bat|{m}|{kind}|{level}|{i}", model=m, kind=kind, level=level, i=i))
    return J


def battery_one(j):
    rng = random.Random(j["key"])
    st, ids = base_state(rng, ["tasks"], arm="noplan", record="lean")
    for a in ids:
        st["stats"][a]["done"] = rng.randint(2, 8)
    tasks = open_tasks(rng, [(j["kind"], j["level"])], per=3)
    st["shown"] = st["open_tasks"] = tasks
    resp = _call(j["model"], O.system_for(st), O.build_prompt(st), "hier:g0bat")
    act = O.parse(resp, set(ids), False)
    by = {x["id"]: x for x in tasks}
    ok = act[0] == "do" and act[1] in by and HT.correct(by[act[1]], act[2])
    ans = act[0] == "do" and act[1] in by and HT.answerable(by[act[1]], act[2])
    return dict(key=j["key"], model=j["model"], kind=j["kind"], level=j["level"], ok=bool(ok), parsed=bool(ans),
                resp=(resp or "")[:120])


def battery_tables(path=None):
    acc, parse = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))
    rows = [json.loads(l) for l in open(path or R / "hier_g0_battery.jsonl")]
    g = defaultdict(list)
    for r in rows:
        g[(r["kind"], r["level"], r["model"])].append(r)
    for (k, lev, m), rs in g.items():
        acc[k][lev][m] = float(np.mean([r["ok"] for r in rs]))
        parse[k][lev][m] = float(np.mean([r["parsed"] for r in rs]))
    return acc, parse


def do_select():
    p = R / "hier_g0_selection.json"
    if p.exists():
        sys.exit(f"{p.name} exists (write-once); not overwriting")
    acc, parse = battery_tables()
    sel = HT.select(acc, parse)
    sel["accuracy"] = {k: {str(l): v for l, v in d.items()} for k, d in acc.items()}
    sel["parse"] = {k: {str(l): v for l, v in d.items()} for k, d in parse.items()}
    json.dump(sel, open(p, "w"), indent=1)
    print(json.dumps(dict(chosen=sel["chosen"], swapped=sel["swapped"], gate_passed=sel["gate_passed"]), indent=1))


# ------------------------------------------------------------------------------------------ G0 comprehension
QUIZ = [
    ("Agent 31 has done multiply-type tasks correctly 9 of 10 times; Agent 47, 3 of 10 times. One multiply task is "
     "open and both are free. To maximize the team's correct tasks, who should be assigned it?",
     ["Agent 31", "Agent 47", "It makes no difference"], 0),
    ("Two agents in the same wave do the same task correctly. How many count for the team?", ["Both", "One", "None"], 1),
    ("If you post a plan on your turn, how many tasks do you do yourself on that turn?", ["0", "1", "2"], 0),
    ("Agent 22's plans: assignments followed 10, of which correct 2. Agent 58's plans: assignments followed 10, of "
     "which correct 9. Whose plans have worked better?", ["Agent 22", "Agent 58", "Neither"], 1),
    ("An agent answers a task wrongly. What happens?", ["It counts as done", "The turn is wasted and the task stays open",
                                                         "The task is removed"], 1),
    ("A plan posted in wave 1 of round 5 is visible during which waves?",
     ["Only wave 1 of round 5", "Wave 2 of round 5 and wave 1 of round 6", "All remaining rounds"], 1),
    ("You are right on reverse-type tasks 2 of 8 times and on weekday-type tasks 8 of 8 times. Open: T3 (reverse) "
     "and T4 (weekday); nobody else will take either. Which should you do to help the team most?", ["T3", "T4"], 1),
    ("What is the team's goal?", ["Each agent finishes as many tasks as it can for itself",
                                  "As many tasks completed correctly as possible across the team",
                                  "As many plans posted as possible"], 1),
]
COMP_N = 15


def comp_jobs(models):
    return [dict(key=f"comp|{m}|{q}|{i}", model=m, q=q, i=i) for m in models for q in range(len(QUIZ))
            for i in range(COMP_N)]


def comp_one(j):
    rng = random.Random(j["key"])
    text, opts, right = QUIZ[j["q"]]
    order = rng.sample(range(len(opts)), len(opts))
    letters = "ABC"[: len(opts)]
    body = "\n".join(f"{letters[i]}) {opts[o]}" for i, o in enumerate(order))
    system = O.SYS_FREE.format(me=36, n=O.N) + "\n\nBefore the game starts, answer a question about the rules."
    prompt = f"{text}\n{body}\n\nReply with one line: ANSWER: <letter>"
    resp = _call(j["model"], system, prompt, "hier:g0comp")
    m = re.search(r"ANSWER\s*:?\s*\**\s*([A-C])\b", resp or "", re.I) or re.match(r"^\W*([A-C])\b", resp or "")
    got = m.group(1).upper() if m else None
    return dict(key=j["key"], model=j["model"], q=j["q"], ok=got == letters[order.index(right)], parsed=got is not None,
                resp=(resp or "")[:80])


def do_gate(models):
    """Registered gates: parse >= 0.90 in the battery cells of the chosen kinds and in the pilot; comprehension >= 0.60."""
    sel = json.load(open(R / "hier_g0_selection.json"))
    chosen = [(c["kind"], c["level"]) for c in sel["chosen"]]
    bat = [json.loads(l) for l in open(R / "hier_g0_battery.jsonl")]
    comp = [json.loads(l) for l in open(R / "hier_g0_comprehension.jsonl")]
    pilot = [json.loads(l) for l in open(R / "hier_pilot.jsonl")] if (R / "hier_pilot.jsonl").exists() else []
    out = dict(task_gate=sel["gate_passed"], models={})
    for m in models:
        pb = np.mean([r["parsed"] for r in bat if r["model"] == m and (r["kind"], r["level"]) in chosen])
        pc = np.mean([r["ok"] for r in comp if r["model"] == m])
        inval = [sum(1 for h in p["history"] for a, act in h["acts"].items()
                     if p["seats"][a] == m and act[0] == "invalid") /
                 max(1, sum(1 for h in p["history"] for a in h["acts"] if p["seats"][a] == m)) for p in pilot
                 if m in p["seats"].values()]
        pp = 1 - float(np.mean(inval)) if inval else None
        ok = pb >= 0.9 and pc >= 0.6 and (pp is None or pp >= 0.9)
        out["models"][m] = dict(parse_battery=round(float(pb), 3), comprehension=round(float(pc), 3),
                                parse_pilot=None if pp is None else round(pp, 3), passed=bool(ok))
    out["passed_models"] = [m for m in models if out["models"][m]["passed"]]
    json.dump(out, open(R / "hier_g0_gate.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ G1 kernels
G1A_N, G1B_STATES, G1B_N, G1D_N, G1E_N = 40, 30, 10, 16, 20


def g1a_jobs(models, kinds, tag="g1a", n=G1A_N, disp=False):
    return [dict(key=f"{tag}|{m}|{k}|{i}", part=tag, model=m, kind=k, i=i, disp=disp) for m in models for k, _ in kinds
            for i in range(n)]


def g1a_one(j, kinds, skill):
    rng = random.Random(j["key"])
    names = [k for k, _ in kinds]
    st, ids = base_state(rng, names, arm="noplan")
    fill_records(st, ids, rng, skill, list(skill))
    st["disposition"] = j["disp"]
    tasks = open_tasks(rng, kinds, only=j["kind"])
    st["shown"] = st["open_tasks"] = tasks
    resp = _call(j["model"], O.system_for(st), O.build_prompt(st), "hier:" + j["part"])
    act = O.parse(resp, set(ids), False)
    by = {x["id"]: x for x in tasks}
    ok = act[0] == "do" and act[1] in by and HT.correct(by[act[1]], act[2])
    return dict(key=j["key"], part=j["part"], model=j["model"], kind=j["kind"], ok=bool(ok), valid=act[0] == "do",
                resp=(resp or "")[:100])


def plan_value(pairs, others, tasks, truth):
    """Expected correct tasks if every agent follows: assigned agents do their task (distinct agents and tasks),
    unassigned agents contribute their mean accuracy over kinds. Returns (value, random, optimal)."""
    from scipy.optimize import linear_sum_assignment
    by = {x["id"]: x for x in tasks}
    mean = {a: float(np.mean([truth[a][x["kind"]] for x in tasks])) for a in others}
    seen_a, seen_t, v = set(), set(), 0.0
    for tid, a in pairs:
        if a in others and a not in seen_a and tid in by and tid not in seen_t:
            seen_a.add(a); seen_t.add(tid); v += truth[a][by[tid]["kind"]]
    v += sum(mean[a] for a in others if a not in seen_a)
    M = np.array([[truth[a][x["kind"]] for x in tasks] for a in others])
    r, c = linear_sum_assignment(-M)
    return v, sum(mean.values()), float(M[r, c].sum())


def g1b_jobs(models):
    return [dict(key=f"g1b|{m}|{s}|{i}", part="g1b", model=m, s=s, i=i) for m in models for s in range(G1B_STATES)
            for i in range(G1B_N)]


def g1b_state(s, kinds, skill):
    rng = random.Random(f"g1b-state|{s}")
    names = [k for k, _ in kinds]
    st, ids = base_state(rng, names, arm="appointed", wave=1)
    st["is_manager"], st["manager_id"] = True, ids[0]
    fill_records(st, ids, rng, skill, list(skill), skip=(ids[0],))
    st["stats"][ids[0]].update(coord=8, followed=rng.randint(20, 40))
    st["stats"][ids[0]]["fcorrect"] = int(st["stats"][ids[0]]["followed"] * rng.uniform(0.4, 0.8))
    tasks = open_tasks(rng, kinds)
    st["shown"] = st["open_tasks"] = tasks
    return st, ids


def g1b_one(j, kinds, skill):
    st, ids = g1b_state(j["s"], kinds, skill)
    st["prompt_seed"] = random.Random(j["key"]).random()
    resp = _call(j["model"], O.system_for(st), O.build_prompt(st), "hier:g1b")
    act = O.parse(resp, set(ids) - {ids[0]}, True)
    others = ids[1:]
    truth = {a: skill[st["seat_model"][a]] for a in others}
    pairs = act[1] if act[0] == "plan" else []
    v, vr, vo = plan_value(pairs, others, st["open_tasks"], truth)
    span = len({a for _, a in pairs})
    return dict(key=j["key"], part="g1b", model=j["model"], s=j["s"], valid=act[0] == "plan", span=span,
                q=(v - vr) / (vo - vr) if vo > vr else float("nan"), value=v, v_random=vr, v_opt=vo,
                resp=(resp or "")[:160])


def g1d_cells():
    return [(r, own, oth) for r in RANKS for own in (0, 1) for oth in OTHER]


def g1d_jobs(models, tag="g1d", n=G1D_N, disp=False):
    return [dict(key=f"{tag}|{m}|{r}|{own}|{oth}|{i}", part=tag, model=m, cell=(r, own, oth), i=i, disp=disp)
            for m in models for (r, own, oth) in g1d_cells() for i in range(n)]


def g1d_state(key, cell, kinds, skill):
    """Free-arm state realizing the cell (own rank, own past planning, other planner status); retried until the
    computed features match the cell."""
    r_t, own, oth = cell
    names = [k for k, _ in kinds]
    for attempt in range(200):
        rng = random.Random(f"{key}|{attempt}")
        st, ids = base_state(rng, names, arm="free", wave=2)
        me = ids[0]
        fill_records(st, ids, rng, skill, list(skill), skip=(me,))
        s = st["stats"][me]
        frac = {"top": 1.0, "middle": 0.6, "bottom": 0.1}[r_t]
        for k in names:
            n = rng.randint(2, 3); s["att"][k] = n; s["ok"][k] = int(round(frac * n))
        s["done"] = sum(s["ok"].values())
        if own:
            s.update(coord=2, followed=6, fcorrect=4)
        tasks = open_tasks(rng, kinds)
        st["shown"] = st["open_tasks"] = tasks
        if oth != "none":
            p = ids[3]
            st["stats"][p].update(coord=3, followed=9, fcorrect=6)
            if oth == "live":
                others = [a for a in ids if a != p]
                st["live_plans"] = {p: [(x["id"], a) for x, a in zip(rng.sample(tasks, len(others)), others)]}
                st["mine"] = [(p, tid) for tid, a in st["live_plans"][p] if a == me]
        st["wave1_log"] = [f"- Agent {a} did T{rng.randint(20, 39)}" for a in ids[4:8]]
        if (rank_bin(st, me), int(s["coord"] > 0), other_status(st, me)) == cell:
            return st, ids
    raise RuntimeError(f"cannot realize cell {cell}")


def g1d_one(j, kinds, skill):
    st, ids = g1d_state(j["key"], tuple(j["cell"]), kinds, skill)
    st["disposition"] = j["disp"]
    resp = _call(j["model"], O.system_for(st), O.build_prompt(st), "hier:" + j["part"])
    act = O.parse(resp, set(ids) - {ids[0]}, True)
    return dict(key=j["key"], part=j["part"], model=j["model"], cell=list(j["cell"]), plan=act[0] == "plan",
                valid=act[0] != "invalid", resp=(resp or "")[:100])


def g1e_jobs(models):
    return [dict(key=f"g1e|{m}|{q}|{fit}|{i}", part="g1e", model=m, q=q, fit=fit, i=i) for m in models
            for q in QBINS for fit in ("good", "poor") for i in range(G1E_N)]


def g1e_one(j, kinds, skill):
    rng = random.Random(j["key"])
    names = [k for k, _ in kinds]
    st, ids = base_state(rng, names, arm="free", wave=2)
    me, p = ids[0], ids[3]
    fill_records(st, ids, rng, skill, list(skill), skip=(me,))
    tasks = open_tasks(rng, kinds)
    st["shown"] = st["open_tasks"] = tasks
    target = rng.choice(tasks)
    s = st["stats"][me]
    for k in names:
        good = (k == target["kind"]) == (j["fit"] == "good")
        s["att"][k] = 3; s["ok"][k] = 3 if good else rng.choice([0, 1])
    s["done"] = sum(s["ok"].values())
    if j["q"] == "new":
        st["stats"][p].update(coord=1, followed=0, fcorrect=0)
    else:
        st["stats"][p].update(coord=3, followed=10, fcorrect=int(round(10 * j["q"])))
    others = [a for a in ids if a not in (p, me)]
    rest = [x for x in tasks if x["id"] != target["id"]]
    st["live_plans"] = {p: [(target["id"], me)] + [(x["id"], a) for x, a in zip(rng.sample(rest, len(others)), others)]}
    st["mine"] = [(p, target["id"])]
    st["wave1_log"] = [f"- Agent {p} posted a plan"] + [f"- Agent {a} did T{rng.randint(20, 39)}" for a in ids[5:8]]
    assert fit_of(st, me, target["kind"]) == j["fit"] and qbin(st["stats"][p]) == j["q"]
    resp = _call(j["model"], O.system_for(st), O.build_prompt(st), "hier:g1e")
    act = O.parse(resp, set(ids) - {me}, True)
    return dict(key=j["key"], part="g1e", model=j["model"], q=j["q"], fit=j["fit"],
                follow=act[0] == "do" and act[1] == target["id"], plan=act[0] == "plan", valid=act[0] != "invalid",
                resp=(resp or "")[:100])


# ------------------------------------------------------------------------------------------ summarize
def summarize():
    rows = [json.loads(l) for l in open(KER)] if KER.exists() else []
    kinds = O.load_selection()
    by = defaultdict(list)
    for r in rows:
        by[r["part"]].append(r)
    out = dict(kinds=kinds)
    sk = defaultdict(dict)
    for (m, k), rs in _group(by["g1a"], ("model", "kind")).items():
        sk[m][k] = float(np.mean([r["ok"] for r in rs]))
    out["skill"] = sk
    out["skill_n"] = {f"{m}|{k}": len(rs) for (m, k), rs in _group(by["g1a"], ("model", "kind")).items()}
    pq = {}
    for (m,), rs in _group(by["g1b"], ("model",)).items():
        qs = [r["q"] if r["valid"] else 0.0 for r in rs]
        pq[m] = dict(q=float(np.nanmean(qs)), valid=float(np.mean([r["valid"] for r in rs])),
                     spans=[r["span"] for r in rs if r["valid"]],
                     q_valid=float(np.nanmean([r["q"] for r in rs if r["valid"]])) if any(r["valid"] for r in rs)
                     else float("nan"))
    out["planq"] = pq
    role = defaultdict(dict)
    for (m, c), rs in _group(by["g1d"], ("model", "cell")).items():
        role[m]["|".join(map(str, c))] = [int(sum(r["plan"] for r in rs)), len(rs)]
    out["role"] = role
    fol = defaultdict(dict)
    for (m, q, fit), rs in _group(by["g1e"], ("model", "q", "fit")).items():
        fol[m][f"{q}|{fit}"] = [int(sum(r["follow"] for r in rs)), len(rs), int(sum(r["plan"] for r in rs))]
    out["follow"] = fol
    out["assertiveness"] = {m: float(np.mean([k / n for k, n in cells.values()])) for m, cells in role.items()}
    pops = [json.loads(l) for l in open(R / "hier_g1c_pop.jsonl")] if (R / "hier_g1c_pop.jsonl").exists() else []
    if pops:
        tp = defaultdict(list)
        for p in pops:
            lab = p["manager_model"] if p["arm"] == "appointed" else p["arm"]
            tp[lab].append(p["metrics"]["throughput"])
        base = float(np.mean(tp["noplan"]))
        out["g1c"] = {lab: dict(throughput=float(np.mean(v)), n=len(v), gain=float(np.mean(v)) - base)
                      for lab, v in tp.items()}
        out["gain"] = {m: out["g1c"][m]["gain"] for m in O.MODELS if m in out["g1c"]}
        # behaviour of unassigned DO choices (list position) and invalid rates, per model, from the 7-worker arms
        first, inval = defaultdict(list), defaultdict(list)
        for p in pops:
            if p["arm"] not in ("noplan", "ownorder"):
                continue
            for h in p["history"]:
                for a, act in h["acts"].items():
                    m = p["seats"][a]
                    inval[m].append(act[0] == "invalid")
                    if act[0] == "do" and len(act) > 5:
                        first[m].append(act[5] == 0)
        out["first_listed"] = {m: float(np.mean(v)) for m, v in first.items()}
        out["invalid"] = {m: float(np.mean(v)) for m, v in inval.items()}
    json.dump(out, open(R / "hier_kernels.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ("skill", "assertiveness", "gain")}, indent=1))


RELIABILITY_MIN = 0.6     # registered top-up rule for G1c


def reliability(draws=2000):
    """Bootstrap over G1c seeds (paired: all arms share seeds and tasks): probability that the model with the
    highest mean gain is still the highest when seeds are resampled. Exit code 3 asks the queue for the
    registered G1c top-up (8 more seeds), done once."""
    pops = [json.loads(l) for l in open(R / "hier_g1c_pop.jsonl")]
    tp = defaultdict(dict)
    for p in pops:
        lab = p["manager_model"] if p["arm"] == "appointed" else p["arm"]
        tp[lab][p["seed"]] = p["metrics"]["throughput"]
    seeds = sorted(set.intersection(*[set(v) for v in tp.values()]))
    ms = [m for m in O.MODELS if m in tp]
    G = np.array([[tp[m][s] - tp["noplan"][s] for s in seeds] for m in ms])
    best = int(np.argmax(G.mean(1)))
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(draws):
        idx = rng.integers(0, len(seeds), len(seeds))
        hits += int(np.argmax(G[:, idx].mean(1)) == best)
    rel = hits / draws
    out = dict(best=ms[best], p_best_stays_best=rel, n_seeds=len(seeds),
               gain={m: float(G[i].mean()) for i, m in enumerate(ms)},
               gain_se={m: float(G[i].std(ddof=1) / np.sqrt(len(seeds))) for i, m in enumerate(ms)})
    json.dump(out, open(R / "hier_g1c_reliability.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    if rel < RELIABILITY_MIN:
        sys.exit(3)


def _group(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(tuple(r[k]) if isinstance(r[k], list) else r[k] for k in keys)].append(r)
    return g


# ------------------------------------------------------------------------------------------ main
def run(jobs, fn, path, parallel):
    done = _keys(path)
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{len(todo)} calls to make ({len(jobs) - len(todo)} done) -> {path.name}", flush=True)
    buf, n = [], 0
    with ThreadPoolExecutor(parallel) as ex:
        for r in ex.map(fn, todo):
            buf.append(r); n += 1
            if len(buf) >= 50:
                _write(path, buf); buf = []
                print(f"  {n}/{len(todo)}", flush=True)
    _write(path, buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["battery", "select", "comp", "gate", "g1a", "g1b", "g1d", "g1e", "g3check",
                                     "summarize", "reliability"])
    ap.add_argument("--models", nargs="+", default=O.MODELS)
    ap.add_argument("--parallel", type=int, default=12)
    a = ap.parse_args()
    if a.part == "battery":
        return run(battery_jobs(a.models), battery_one, R / "hier_g0_battery.jsonl", a.parallel)
    if a.part == "select":
        return do_select()
    if a.part == "comp":
        return run(comp_jobs(a.models), comp_one, R / "hier_g0_comprehension.jsonl", a.parallel)
    if a.part == "gate":
        return do_gate(a.models)
    if a.part == "summarize":
        return summarize()
    if a.part == "reliability":
        return reliability()
    kinds = O.load_selection()
    if a.part == "g1a":
        sel = json.load(open(R / "hier_g0_selection.json"))
        prior = {m: {k: sel["accuracy"][k][str(l)][m] for k, l in kinds} for m in O.MODELS}   # records only
        return run(g1a_jobs(a.models, kinds), lambda j: g1a_one(j, kinds, prior), KER, a.parallel)
    skill = O.load_skill()
    if a.part == "g1b":
        return run(g1b_jobs(a.models), lambda j: g1b_one(j, kinds, skill), KER, a.parallel)
    if a.part == "g1d":
        return run(g1d_jobs(a.models), lambda j: g1d_one(j, kinds, skill), KER, a.parallel)
    if a.part == "g1e":
        return run(g1e_jobs(a.models), lambda j: g1e_one(j, kinds, skill), KER, a.parallel)
    if a.part == "g3check":
        fz = O.frozen()
        ms = sorted(set(fz["g3_targets"].values()))
        jobs = g1a_jobs(ms, kinds, tag="g3a", n=20, disp=True) + g1d_jobs(ms, tag="g3d", n=8, disp=True)
        fn = lambda j: g1a_one(j, kinds, skill) if j["part"] == "g3a" else g1d_one(j, kinds, skill)
        return run(jobs, fn, KER, a.parallel)


if __name__ == "__main__":
    main()
