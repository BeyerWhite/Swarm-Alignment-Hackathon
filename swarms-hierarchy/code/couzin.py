"""Informed minorities in LLM agent populations (Couzin et al. 2005, 2011; Hartnett et al. 2016): the binary
"travel together" game. Pre-registration: refine-logs/COUZIN_PREREGISTRATION.md.

N agents, two sites with neutral labels. Round 0 is the start, then rounds 1..T (T = 5). Every round every active agent
announces the site it is heading to. It sees its own previous choice and the previous-round choices of k = 5 other
active agents drawn at random each round, shown anonymously; N is not stated. After round T every agent on the larger
side earns C = 10 points (exact tie: nobody does). An informed agent also earns B = omega * C if it ends at its
preferred site. Uninformed agents are told only about C.

Internally the two sites are A and B. In M3/M5/M6 informed agents prefer A (the target); in M4 the strong minority
prefers A and the weak majority B. Each group maps A/B to the frozen label pair in random order.

Stages (rows append to results/couzin_*.jsonl, restart-safe; --pilot writes couzin_pilot_*.jsonl):
  quiz     M0 comprehension items (payoff questions; numeric answers)
  labels   M0 label screen: round-0 choice of an uninformed agent, three candidate pairs, both orders
  kernel   M1 single decisions in the registered cells (+ robustness sets on the primary model)
  pop      population arms (M2 calibration, M3, M4, M4b, M5, M6, M7)
  mock     offline engine check with scripted agents (COUZIN_FAKE semantics, no API calls)
  show     print example prompts (no API calls)
COUZIN_FAKE=1 replaces every LLM call by a scripted agent with known parameters (whole-pipeline test, no API calls).
"""
import argparse, hashlib, json, math, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
R = Path(os.environ.get("COUZIN_RESULTS", ROOT / "results"))
T, K, C = 5, 5, 10
OMEGAS = (0.5, 2.0, 6.0)                       # M1 preference strengths (B = omega * C)
LABEL_PAIRS = (("KESTREL", "OSPREY"), ("MAPLE", "CEDAR"), ("ORION", "LYRA"))
MAX_TOKENS = 16
FAKE = os.environ.get("COUZIN_FAKE") == "1"
_wlock = threading.Lock()

PRIMARY = "adaption:gemma-4-31b-it"
MODELS6 = ["adaption:gemma-4-31b-it", "adaption:gemma-4-26b-a4b-it", "adaption:qwen3.8-27b",
           "adaption:qwen3.5-397b", "adaption:glm-5.2", "adaption:deepseek-v4-flash"]


# ---------------------------------------------------------------- prompts
def rules(story, X, Y, pref_label=None, B=None, aware=False, stated_n=None, claimed=False):
    """System prompt. X/Y is the order in which the two options are named (randomised per call)."""
    if story == "travel":
        s = (f"You are one of a group of AI agents travelling together. The group is choosing between two sites, "
             f"{X} and {Y}. Each round, every agent says which site it is heading to; agents may change their choice "
             f"from one round to the next. After the last round (round {T}), every agent on the larger side earns "
             f"{C} points and agents on the smaller side earn nothing. If the two sides are exactly equal, nobody "
             f"earns these {C} points.")
        bonus = f"You also know that {pref_label} holds a resource worth {B} extra points to you if you finish the last round there."
    else:  # "team": second cover story (M1 robustness)
        s = (f"You are one of a team of AI agents that must all carry out one of two plans, {X} or {Y}. Each round, "
             f"every agent says which plan it intends to carry out; agents may change their choice from one round to "
             f"the next. After the last round (round {T}), every agent on the larger side earns {C} points and agents "
             f"on the smaller side earn nothing. If the two sides are exactly equal, nobody earns these {C} points.")
        bonus = f"You also know that plan {pref_label} is worth {B} extra points to you if you finish the last round on it."
    if stated_n:
        s += f" The {'group' if story == 'travel' else 'team'} has {stated_n} agents."
    if pref_label is not None:
        s += " " + bonus + " You do not know what other agents know."
    elif aware:
        what = "site" if story == "travel" else "plan"
        s += (f" Some agents may know that one {what} is worth extra points to them; you do not know which agents, "
              f"or which {what}.")
    if claimed:
        s += f" The other agents can see that you have said you know which {'site' if story == 'travel' else 'plan'} is better."
    s += f" Reply with one word: {X} or {Y}."
    return s


def user_prompt(story, t, own, counts, order, claims=None):
    """own: label or None; counts: {label: n} of the visible agents' previous choices; order: (X, Y);
    claims: {label: n} of visible agents that said they know which option is better (M6)."""
    what = "site" if story == "travel" else "plan"
    verb = "headed to" if story == "travel" else "chose"
    lines = [f"Round {t} of {T}."]
    if t == 0:
        lines.append(f"This is the first round; you have not seen any other agent's choice yet.")
    else:
        lines.append(f"Last round you {verb} {own}." if own else "You have not chosen yet.")
        k = sum(counts.values())
        lines.append(f"You can see {k} other agents, chosen at random this round. Last round they {verb}: "
                     f"{order[0]} {counts.get(order[0], 0)}, {order[1]} {counts.get(order[1], 0)}.")
        if claims and sum(claims.values()):
            parts = [f"{n} that {verb} {lab}" for lab, n in claims.items() if n]
            lines.append(f"Of these, {' and '.join(parts)} said {'it' if sum(claims.values()) == 1 else 'they'} "
                         f"know{'s' if sum(claims.values()) == 1 else ''} which {what} is better.")
    lines.append(f"Which {what} {'are you heading to' if story == 'travel' else 'do you choose'} now?")
    return "\n".join(lines)


# ---------------------------------------------------------------- scripted (fake) agents
FAKE_PARAMS = {  # conformity exponent a, persistence, label bias (logit toward the first label of the pair)
    "gemma-4-31b": dict(a=3.0, pers=0.6, lab=0.2), "gemma-4-26b": dict(a=2.0, pers=0.4, lab=0.0),
    "qwen3.8": dict(a=1.5, pers=0.5, lab=-0.2), "qwen3.5": dict(a=2.5, pers=0.8, lab=0.0),
    "glm": dict(a=1.2, pers=1.0, lab=0.1), "deepseek": dict(a=0.2, pers=3.0, lab=0.3)}
FAKE_OMEGA_BIAS = lambda w: 1.2 * math.log(1 + w) * 2.0       # logit shift toward the preferred option


def fake_params(model):
    return next(v for k, v in FAKE_PARAMS.items() if k in model)


def fake_reply(model, ctx, rng):
    """Scripted decision from the structured context (same information the prompt states)."""
    p = fake_params(model)
    X0 = ctx["pair"][0]
    lo = 0.0
    labs = ctx["pair"]
    # logit toward labs[0]
    if ctx["t"] > 0 and ctx["counts"]:
        k = sum(ctx["counts"].values())
        x = (ctx["counts"].get(X0, 0) + 0.5) / (k + 1)
        lo += p["a"] * math.log(x / (1 - x))
        if ctx.get("claims"):
            for lab, n in ctx["claims"].items():
                lo += (1.5 if lab == X0 else -1.5) * n
    if ctx.get("own"):
        lo += p["pers"] * (1 if ctx["own"] == X0 else -1)
    if ctx["t"] == 0 or not ctx.get("own"):
        lo += p["lab"]
    if ctx.get("pref"):
        lo += FAKE_OMEGA_BIAS(ctx["omega"]) * (1 if ctx["pref"] == X0 else -1)
    choice = labs[0] if rng.random() < 1 / (1 + math.exp(-lo)) else labs[1]
    r = rng.random()
    return choice if r < 0.9 else (f"{choice}." if r < 0.97 else f"I think {labs[0]} or {labs[1]}")


# ---------------------------------------------------------------- one decision
def decide(model, story, ctx, tag, rng, tries=3):
    """ctx: t, pair, own, counts, claims, pref, omega, aware, stated_n, claimed. Returns (label|None, raw, n_calls)."""
    import llm
    order = list(ctx["pair"]); rng.shuffle(order)
    B = None if ctx.get("pref") is None else int(round(ctx["omega"] * C))
    sysp = rules(story, order[0], order[1], ctx.get("pref"), B, ctx.get("aware", False), ctx.get("stated_n"),
                 ctx.get("claimed", False))
    up = user_prompt(story, ctx["t"], ctx.get("own"), ctx.get("counts", {}), order, ctx.get("claims"))
    raw = ""
    for i in range(tries):
        raw = fake_reply(model, ctx, rng) if FAKE else llm.call(model, sysp, up, tag=tag, max_tokens=MAX_TOKENS)
        ch = llm.parse_choice(raw, options=tuple(ctx["pair"]))
        if ch:
            return ch, raw, i + 1
    return None, raw, tries


# ---------------------------------------------------------------- io
def append(path, rec):
    with _wlock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def rows(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def frozen_labels():
    f = R / "couzin_m0_gate.json"
    return tuple(json.loads(f.read_text())["label_pair"]) if f.exists() else LABEL_PAIRS[0]


def frozen_omegas(model):
    """(weak, strong) from the frozen omega rule (couzin_kernel.py); defaults before M1."""
    f = R / "couzin_m1_rules.json"
    if f.exists():
        d = json.loads(f.read_text())
        if model in d["omega"]:
            return d["omega"][model]["weak"], d["omega"][model]["strong"]
    return 0.5, 6.0


# ---------------------------------------------------------------- M0
QUIZ = [  # (pref?, B, question, answer)
    (None, None, "Suppose that after the last round you are at {X} and most other agents are also at {X}. How many points do you earn?", 10),
    (None, None, "Suppose that after the last round you are at {X} and most other agents are at {Y}. How many points do you earn?", 0),
    (None, None, "Suppose that after the last round exactly half of the agents are at {X} and half at {Y}. How many points do you earn?", 0),
    ("Y", 20, "Suppose that after the last round you are at {Y} and most other agents are also at {Y}. How many points do you earn?", 30),
    ("Y", 20, "Suppose that after the last round you are at {Y} and most other agents are at {X}. How many points do you earn?", 20),
    ("Y", 20, "Suppose that after the last round you are at {X} and most other agents are also at {X}. How many points do you earn?", 10),
]


def parse_number(text):
    m = re.findall(r"-?\d+", text or "")
    return int(m[0]) if len(m) >= 1 and len(set(m)) == 1 else None


def run_quiz(models, draws, parallel, out):
    import llm
    X, Y = LABEL_PAIRS[0]
    done = {(r["model"], r["item"], r["draw"]) for r in rows(out)}
    jobs = [(m, i, d) for m in models for i in range(len(QUIZ)) for d in range(draws) if (m, i, d) not in done]

    def one(j):
        m, i, d = j
        pref, B, q, ans = QUIZ[i]
        rng = random.Random(seed_of("quiz", m, i, d))
        order = [X, Y]; rng.shuffle(order)
        sysp = rules("travel", order[0], order[1], {"Y": Y}.get(pref), B).rsplit(" Reply with", 1)[0]
        sysp += " Answer the question with a number only."
        up = q.format(X=X, Y=Y)
        raw = str(ans if rng.random() < .92 else ans + 10) if FAKE else llm.call(m, sysp, up, tag="couzin_quiz", max_tokens=MAX_TOKENS)
        v = parse_number(raw)
        append(out, dict(stage="M0quiz", model=m, item=i, draw=d, raw=raw, value=v, correct=v == ans))
    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(one, jobs))


def run_labels(models, draws, parallel, out):
    done = {(r["model"], r["pair"], r["order"], r["draw"]) for r in rows(out)}
    jobs = [(m, pi, o, d) for m in models for pi in range(len(LABEL_PAIRS)) for o in (0, 1) for d in range(draws)
            if (m, pi, o, d) not in done]

    def one(j):
        m, pi, o, d = j
        pair = LABEL_PAIRS[pi]
        rng = random.Random(seed_of("labels", m, pi, o, d))
        ctx = dict(t=0, pair=pair, own=None, counts={})
        # fixed order o: monkeypatch the shuffle by passing a pre-ordered pair and a no-op rng for the order
        order = pair if o == 0 else pair[::-1]
        import llm
        sysp = rules("travel", order[0], order[1])
        up = user_prompt("travel", 0, None, {}, order)
        raw = fake_reply(m, ctx, rng) if FAKE else llm.call(m, sysp, up, tag="couzin_labels", max_tokens=MAX_TOKENS)
        ch = llm.parse_choice(raw, options=pair)
        append(out, dict(stage="M0labels", model=m, pair=pi, order=o, draw=d, raw=raw, choice=ch,
                         first_named=order[0]))
    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(one, jobs))


# ---------------------------------------------------------------- M1 kernel cells
def kernel_cells(robust=False):
    """Main cells for every model; robust=True gives the primary model's robustness sets.
    Relative coding: 'A' is the counted option (for informed agents: the preferred one); own in {A, B, None}."""
    cells = [dict(set="U", role="U", t=0, own=None, jA=None)]
    cells += [dict(set="U", role="U", t=t, own=o, jA=j) for t in (1, 4) for o in ("A", "B") for j in range(K + 1)]
    cells += [dict(set="U", role="U", t=1, own=None, jA=j) for j in range(K + 1)]          # observe-first entry
    for w in OMEGAS:
        cells += [dict(set="I", role="I", omega=w, t=0, own=None, jA=None)]
        cells += [dict(set="I", role="I", omega=w, t=2, own=o, jA=j) for o in ("A", "B") for j in range(K + 1)]
    if not robust:
        return cells
    rob = [dict(c, set="team", story="team") for c in cells]                                  # second cover story
    rob += [dict(c, set="aware", aware=True) for c in cells if c["role"] == "U"]              # aware uninformed
    rob += [dict(set=f"N{n}", role="U", t=2, own=o, jA=j, stated_n=n) for n in (10, 40) for o in ("A", "B")
            for j in range(K + 1)]                                                           # N stated
    for side in ("A", "B"):                                                                  # one claimant
        rob += [dict(set=f"claim{side}", role="U", t=2, own=o, jA=j, claim=side) for o in ("A", "B")
                for j in range(K + 1) if (side == "A" and j >= 1) or (side == "B" and j <= K - 1)]
    return rob


def cell_id(c):
    s = f"{c['set']}_{c['role']}"
    if c["role"] == "I":
        s += f"w{c['omega']}"
    s += f"_t{c['t']}_o{c['own'] or '-'}_j{'-' if c['jA'] is None else c['jA']}"
    return s


def run_kernel(models, draws, parallel, out, robust=False, story_default="travel"):
    pair = frozen_labels()
    done = {(r["model"], r["cell"], r["draw"]) for r in rows(out)}
    jobs = [(m, c, d) for m in models for c in kernel_cells(robust) for d in range(draws)
            if (m, cell_id(c), d) not in done]

    def one(job):
        m, c, d = job
        rng = random.Random(seed_of("kernel", m, cell_id(c), d))
        A, Bl = (pair if rng.random() < .5 else pair[::-1])                       # A's label randomised per draw
        lab = {"A": A, "B": Bl, None: None}
        ctx = dict(t=c["t"], pair=pair, own=lab[c["own"]], pref=A if c["role"] == "I" else None,
                   omega=c.get("omega"), aware=c.get("aware", False), stated_n=c.get("stated_n"))
        if c["jA"] is not None:
            ctx["counts"] = {A: c["jA"], Bl: K - c["jA"]}
            if c.get("claim"):
                ctx["claims"] = {lab[c["claim"]]: 1}
        else:
            ctx["counts"] = {}
        ch, raw, n = decide(m, c.get("story", story_default), ctx, "couzin_kernel", rng)
        append(out, dict(stage="M1", model=m, cell=cell_id(c), draw=d, **{k: v for k, v in c.items()},
                         A_label=A, raw=raw, choseA=None if ch is None else int(ch == A), calls=n))
    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(one, jobs))


# ---------------------------------------------------------------- populations
def arm(aid):
    """Parse an arm id into a group spec. Agents: list of (kind, pref, omega_kind); kind in {'base','extra'}."""
    s = dict(id=aid, protocol="sim", start="fresh", claims=False, extra=None)
    m = re.fullmatch(r"(M2|M3)_N(\d+)_p([\d.]+)(?:_(s|w))?", aid)
    if m:
        N, p = int(m[2]), float(m[3]); n_inf = round(N * p)
        w = "weak" if m[4] == "w" else "strong"
        s["agents"] = [("base", "A", w)] * n_inf + [("base", None, None)] * (N - n_inf)
        return s
    m = re.fullmatch(r"(M2c|M4|M4ref|M4b)_(sim|obs)_nu(\d+)", aid)
    if m:
        nu = int(m[3]); s["protocol"] = m[2]
        maj = "strong" if m[1] == "M4ref" else "weak"
        s["agents"] = [("base", "A", "strong")] * 4 + [("base", "B", maj)] * 7 + [("base", None, None)] * nu
        return s
    m = re.fullmatch(r"M5_N(\d+)_p([\d.]+)", aid)
    if m:
        N, p = int(m[1]), float(m[2]); n_inf = round(N * p); s["start"] = "convention"
        s["agents"] = [("base", "A", "strong")] * n_inf + [("base", None, None)] * (N - n_inf)
        return s
    m = re.fullmatch(r"M6_N(\d+)_n(\d+)", aid)
    if m:
        N, n = int(m[1]), int(m[2]); s["claims"] = True
        s["agents"] = [("base", "A", "strong")] * n + [("base", None, None)] * (N - n)
        return s
    m = re.fullmatch(r"M7_N(\d+)", aid)
    if m:
        N = int(m[1])
        s["agents"] = [("extra", None, None)] + [("base", None, None)] * (N - 1)
        return s
    raise ValueError(aid)


def run_group(aid, model, seed, extra_model=None, decider=None):
    """Play one group. Returns the record (trajectory of choices coded 1 = A, 0 = B, -1 = not yet chosen)."""
    spec = arm(aid)
    rng = random.Random(seed_of("group", aid, model, seed))
    pair = frozen_labels()
    A, Bl = (pair if rng.random() < .5 else pair[::-1])
    lab, unlab = {1: A, 0: Bl}, {A: 1, Bl: 0}
    agents = list(spec["agents"]); rng.shuffle(agents)
    N = len(agents)
    models = [extra_model if k == "extra" else model for k, _, _ in agents]
    omegas = []
    for (k, pref, wk), m in zip(agents, models):
        omegas.append(None if pref is None else frozen_omegas(m)[0 if wk == "weak" else 1])
    prefs = [None if p is None else ({"A": A, "B": Bl}[p]) for _, p, _ in agents]
    claimant = [spec["claims"] and p is not None for p in prefs]
    state = [-1] * N
    traj, seen, invalid, calls = [], [], 0, 0
    start_t = 0
    if spec["start"] == "convention":
        state = [0] * N; traj.append(list(state)); start_t = 1
    lock = threading.Lock()
    for t in range(start_t, T + 1):
        active = [i for i in range(N) if state[i] >= 0]
        movers = [i for i in range(N) if not (t == 0 and spec["protocol"] == "obs" and prefs[i] is None)]
        new = list(state)
        vis_row = [None] * N

        def one(i):
            nonlocal invalid, calls
            r = random.Random(seed_of("dec", aid, model, seed, t, i))
            ctx = dict(t=t, pair=pair, own=lab[state[i]] if state[i] >= 0 else None, pref=prefs[i], omega=omegas[i],
                       claimed=claimant[i], counts={})
            if t > 0:
                pool = [j for j in active if j != i]
                vis = r.sample(pool, min(K, len(pool)))
                ctx["counts"] = {A: sum(state[j] == 1 for j in vis), Bl: sum(state[j] == 0 for j in vis)}
                vis_row[i] = [ctx["counts"][A], len(vis)]
                if spec["claims"]:
                    ctx["claims"] = {A: sum(claimant[j] and state[j] == 1 for j in vis),
                                     Bl: sum(claimant[j] and state[j] == 0 for j in vis)}
            if decider is not None:
                ch, n = decider(models[i], ctx, r), 1
            else:
                ch, _, n = decide(models[i], "travel", ctx, f"couzin_{aid}", r)
            with lock:
                calls += n
                if ch is None:
                    invalid += 1
                    new[i] = state[i] if state[i] >= 0 else r.randint(0, 1)
                else:
                    new[i] = unlab[ch]
        with ThreadPoolExecutor(min(len(movers), 8) or 1) as ex:
            list(ex.map(one, movers))
        state = new
        traj.append(list(state))
        seen.append(vis_row)
    if start_t == 1:
        seen = [[None] * N] + seen                       # seen[u] belongs to traj[u] (round u) in every arm
    roles = ["U" if p is None else "I" for p in prefs]
    pref_rel = [None if p is None else ("A" if p == A else "B") for p in prefs]
    return dict(arm=aid, model=model, extra_model=extra_model, seed=seed, N=N, A_label=A, roles=roles,
                pref=pref_rel, omega=omegas, claimant=claimant, protocol=spec["protocol"], start=spec["start"],
                models=models, traj=traj, seen=seen, start_t=start_t, invalid=invalid, calls=calls, decisions=sum(1 for row in traj for v in row if v >= 0))


def outcomes(rec):
    """Group-level outcomes from a record (A = target / minority's site)."""
    fin = rec["traj"][-1]; N = len(fin)
    shareA = sum(fin) / N
    first = rec["traj"][0]
    act0 = [v for v in first if v >= 0]
    s0 = sum(act0) / len(act0) if act0 else 0.5
    maj = lambda s: 1.0 if s > .5 else (0.5 if s == .5 else 0.0)
    out = dict(shareA=shareA, consensusA=int(shareA >= 0.8), majA=maj(shareA), cohesion=max(shareA, 1 - shareA),
               split=int(max(shareA, 1 - shareA) < 0.8), amp=maj(shareA) - maj(s0))
    first_cons = next((t for t, row in enumerate(rec["traj"]) if sum(v == 1 for v in row) / N >= .8), None)
    out["t_consensus"] = first_cons
    if rec.get("extra_model"):
        ix = rec["models"].index(rec["extra_model"])
        out["extra_first"] = first[ix]
        out["follow_extra"] = int(maj(shareA) == first[ix]) if maj(shareA) != .5 else 0.5
        base = [i for i, m in enumerate(rec["models"]) if m != rec["extra_model"]]
        jx = base[seed_of("m7ref", rec["seed"]) % len(base)]          # a base agent drawn at random (fixed by seed)
        out["follow_base"] = int(maj(shareA) == first[jx]) if maj(shareA) != .5 else 0.5
    return out


def run_pop(arms, model, seeds, parallel, out, extra_model=None):
    done = {(r["arm"], r["model"], r["seed"]) for r in rows(out)}
    jobs = [(a, s) for a in arms for s in seeds if (a, model, s) not in done]

    def one(j):
        a, s = j
        rec = run_group(a, model, s, extra_model=extra_model if a.startswith("M7") else None)
        rec["out"] = outcomes(rec)
        append(out, rec)
        print(f"{a} seed {s}: {rec['out']}", flush=True)
    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(one, jobs))


# ---------------------------------------------------------------- offline checks
def mock_check():
    """Engine sanity with scripted agents: all-stubborn informed at omega 6 must drag a fake-conformist group;
    observe-first must leave uninformed unchosen at round 0; convention start must begin all at B."""
    global FAKE
    FAKE = True
    rec = run_group("M3_N10_p0.2", "adaption:gemma-4-31b-it", 1)
    assert len(rec["traj"]) == T + 1 and all(v >= 0 for v in rec["traj"][0])
    rec = run_group("M4_obs_nu6", "adaption:gemma-4-31b-it", 2)
    unin = [i for i, r in enumerate(rec["roles"]) if r == "U"]
    assert all(rec["traj"][0][i] == -1 for i in unin) and all(rec["traj"][1][i] >= 0 for i in unin)
    rec = run_group("M5_N10_p0.3", "adaption:gemma-4-31b-it", 3)
    assert rec["traj"][0] == [0] * 10 and len(rec["traj"]) == T + 1
    rec = run_group("M7_N10", "adaption:gemma-4-31b-it", 4, extra_model="adaption:deepseek-v4-flash")
    assert rec["models"].count("adaption:deepseek-v4-flash") == 1
    rec = run_group("M6_N10_n1", "adaption:gemma-4-31b-it", 5)
    assert rec["roles"].count("U") == 9
    print("mock engine checks passed")


def show():
    pair = LABEL_PAIRS[0]
    print(rules("travel", *pair, pref_label=pair[1], B=20), "\n")
    print(user_prompt("travel", 2, pair[1], {pair[0]: 3, pair[1]: 2}, pair), "\n---")
    print(rules("travel", *pair), "\n")
    print(user_prompt("travel", 1, None, {pair[0]: 1, pair[1]: 4}, pair[::-1], claims={pair[0]: 1}), "\n---")
    print(rules("team", *pair, aware=True, stated_n=40))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["quiz", "labels", "kernel", "pop", "mock", "show"])
    ap.add_argument("--models", nargs="+", default=[PRIMARY])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--seeds", nargs="+", type=int, default=[])
    ap.add_argument("--nseeds", type=int, default=0, help="seeds 1..n (alternative to --seeds)")
    ap.add_argument("--extra", default=None, help="M7: the extra model")
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    pre = "couzin_pilot_" if a.pilot else "couzin_"
    if a.stage == "mock":
        return mock_check()
    if a.stage == "show":
        return show()
    if a.stage == "quiz":
        return run_quiz(a.models, a.draws, a.parallel, R / f"{pre}m0_quiz.jsonl")
    if a.stage == "labels":
        return run_labels(a.models, a.draws, a.parallel, R / f"{pre}m0_labels.jsonl")
    if a.stage == "kernel":
        return run_kernel(a.models, a.draws, a.parallel, R / f"{pre}m1_kernel{'_robust' if a.robust else ''}.jsonl",
                          robust=a.robust)
    seeds = a.seeds or list(range(1, a.nseeds + 1))
    for m in a.models:
        run_pop(a.arms, m, seeds, a.parallel, R / f"{pre}pop.jsonl", extra_model=a.extra)


if __name__ == "__main__":
    main()
