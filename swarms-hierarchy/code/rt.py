"""Response-threshold division of labour (Bonabeau, Theraulaz & Deneubourg 1996; Theraulaz et al. 1998) in LLM agent
populations: the two-type workshop game.

N agents share queues of easy, exactly checkable tasks of two types, A and B (equal arrivals; task kinds rotated over
the type letters by seed). Every round each agent takes one turn, in a fresh random order, and sees the queue as it is
(sequential: no collisions). On its turn an agent DOes one open task or RESTs.

Effort-cost rule (team points): a correct task earns 1 point and costs c points of effort (net 1 - c); a wrong answer
costs c and leaves the task open; resting costs nothing; at the end of every round each task still waiting costs the
team 1 point. Nothing is gained by experience: doing more A never makes A easier or worth more, so any specialization
is behavioural, not economic (contrast: Rosen study).

Benchmark (team-optimal, `policy_rational`): work iff the open tasks are at least the agents still to move this round
(you included); otherwise working and resting give the team the same points, provided later movers clear the queue.
c (< 1) therefore never changes what is optimal; it only makes resting a stated, costless option.

History channels (the experimental lever)
  K = 8   own row in the record (tasks done per type) and "your last 8 turns"
  K = 0   neither: no information about one's own past (fixed-threshold condition)
  others  "shown": other agents' rows and this round's earlier moves; "hidden": neither

Usage
  rt.py mock --policy rational|always|hill [--K 0 8] [--D ...] [--c ...]     offline engine check, no API calls
  rt.py show --K 8 --D 0.5 --c 0.5                                           print example prompts, no API calls
"""
import argparse, hashlib, json, os, random, re, sys, zlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
R = Path(os.environ.get("RT_RESULTS", ROOT / "results"))
N, T, CAP, WARM, HWIN = 8, 24, 12, 4, 8        # agents, rounds, max waiting per type, warm-up rounds dropped, history window
TYPES = ("A", "B")
# RT-R (rest-reward framing, RT_REST_PREREGISTRATION.md): RT_FRAME=rest. There is no effort cost and the parameter
# c is the rest reward r: a correct task earns 1, a wrong answer 0, resting earns r, each waiting task costs 1 per
# round. The team-optimal benchmark is unchanged (work iff open tasks >= agents still to move, while rounds remain).
FRAME = os.environ.get("RT_FRAME", "effort")
KINDS = ("add", "upper", "next")
MAX_TOKENS = 60
WORDS = ["apple", "river", "stone", "cloud", "tiger", "maple", "piano", "lemon", "candle", "garden", "rocket", "silver",
         "forest", "planet", "bridge", "window", "orange", "pencil", "harbor", "meadow"]


def plural(k, word):
    return f"{k} {word}" + ("" if k == 1 else "s")


def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def arrivals(D, n=N):
    """Per-type arrivals per round at demand D (tasks per agent per round, split equally over the two types)."""
    a = int(round(D * n / 2))
    return {j: a for j in TYPES}


def kind_map(seed):
    r = seed % 3
    return {t: KINDS[(i + r) % 3] for i, t in enumerate(TYPES)}


def make_task(tid, typ, kind, rng):
    if kind == "add":
        a, b = rng.randint(10, 49), rng.randint(10, 49)
        return dict(id=tid, type=typ, kind=kind, text=f"What is {a} + {b}?", answer=str(a + b))
    if kind == "upper":
        w = rng.choice(WORDS)
        return dict(id=tid, type=typ, kind=kind, text=f"Write '{w}' in capital letters.", answer=w.upper())
    x = rng.randint(100, 999)
    return dict(id=tid, type=typ, kind=kind, text=f"What number comes after {x}?", answer=str(x + 1))


def correct(task, ans):
    a = ans.strip().strip(".'\"` ")
    if task["kind"] == "upper":
        return bool(a) and a.split()[0].strip(".'\"`") == task["answer"]
    m = re.search(r"-?\d+", a)
    return bool(m) and m.group(0) == task["answer"]


def parse(text):
    lines = [l.strip().strip("*`> ") for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return ("invalid",)
    l = lines[0]
    m = re.match(r"^DO\s+T?(\d+)\s*(?:\(\s*(?:type\s+)?[AB]\s*\))?\s*[:\-]\s*(.+)$", l, re.I)
    if m:
        return ("do", int(m.group(1)), m.group(2))
    if re.match(r"^REST\b", l, re.I):
        return ("rest",)
    return ("invalid",)


# ----------------------------------------------------------------------------------------------- prompts

def system_prompt(me, c, D, n=N, rounds=T, K=8, others="shown"):
    arr = arrivals(D, n)
    each = arr["A"]
    if K > 0:
        info = (f"You see your own record (tasks done of each type) and your last {HWIN} turns"
                + (", the other agents' records and what was done earlier this round." if others == "shown" else "."))
    else:
        info = ("You see the other agents' records and what was done earlier this round." if others == "shown"
                else "You see only the open tasks.")
    return (f"You are Agent {me}, one of {n} AI agents working as a team in a simulated workshop. The team's goal is to "
            f"end the game with as many points as possible after {rounds} rounds. Each round every agent takes one "
            f"turn, in a random order; you see the open tasks as they are when your turn comes. {info}\n\n"
            f"There are two task types, A and B. At the start of every round {plural(each, 'new task')} of type A and "
            f"{plural(each, 'new task')} of type B arrive (at most {CAP} tasks of one type can wait at once; extra "
            f"arrivals are lost). The two types are worth the same.\n\n"
            f"On your turn you either do one open task or rest.\n"
            + (f"- A correct task earns the team 1 point and costs the team {fmt(c)} points of effort (net "
               f"{fmt(round(1 - c, 4))}). A wrong answer costs the effort and leaves the task open.\n"
               f"- Resting costs nothing.\n" if FRAME == "effort" else
               f"- A correct task earns the team 1 point. A wrong answer earns nothing and leaves the task open.\n"
               f"- Resting earns the team {plural(int(c), 'point') if float(c).is_integer() else fmt(c) + ' points'} "
               f"(an agent who rests does routine upkeep instead).\n")
            + f"- At the end of every round, each task still waiting costs the team 1 point.\n\n"
            "Reply with one line in exactly one of these formats:\nDO <task id>: <answer>\nREST")


def hist_str(h):
    return ", ".join(h) if h else "none yet"


def render(view):
    """User prompt from a structured view (shared by populations and the single-decision battery)."""
    me, K, others = view["me"], view["K"], view["others"]
    s = f"Round {view['t']} of {view['T']}. Your turn ({view['pos']} of {view['n']} this round).\n"
    s += f"Team points so far: {fmt(round(view['points'], 2))}.\n"
    s += f"Waiting now: A {view['nA']}, B {view['nB']}.\n"
    if K > 0:
        st = view["stats"][me]
        s += (f"\nYour record: tasks done A {st['done']['A']}, B {st['done']['B']}.\n"
              f"Your last {HWIN} turns (oldest first): {hist_str(view['hist'][me][-HWIN:])}.\n")
    if others == "shown":
        rows = [f"- Agent {a}: tasks done A {view['stats'][a]['done']['A']}, B {view['stats'][a]['done']['B']}"
                for a in view["order_rows"] if a != me]
        s += "\nOther agents:\n" + "\n".join(rows) + "\n"
        if view["pos"] > 1:
            s += "\nEarlier this round:\n" + ("\n".join(view["earlier"]) or "- nothing") + "\n"
    blocks = []
    for j in view["type_order"]:
        xs = [x for x in view["open"] if x["type"] == j]
        blocks += [f"- T{x['id']} ({j}): {x['text']}" for x in xs]
    s += "\nOpen tasks:\n" + ("\n".join(blocks) or "- none")
    return s + "\n\nYour action?"


# ----------------------------------------------------------------------------------------------- engine

def snapshot(view):
    """Everything render() needs, copied (for replay cells of the kernel battery)."""
    me = view["me"]
    return dict(me=me, K=view["K"], others=view["others"], t=view["t"], T=view["T"], n=view["n"], pos=view["pos"],
                points=view["points"], nA=view["nA"], nB=view["nB"], c=view["c"], D=view["D"],
                stats={a: dict(done=dict(st["done"])) for a, st in view["stats"].items()},
                hist={me: list(view["hist"][me])}, order_rows=list(view["order_rows"]), earlier=list(view["earlier"]),
                open=[dict(x) for x in view["open"]], type_order=list(view["type_order"]))


def run_pop(model, K, D, c, seed, others="shown", decide=None, rounds=T, n_agents=N, remove_at=None, n_remove=2,
            reasoning="none", tag="rt", log_views=False):
    """One population. `decide(view) -> reply text` replaces the LLM if given (it must use only what render() shows).
    remove_at: after that round the n_remove agents with the highest share of A among their work turns leave.
    log_views: keep a snapshot of every decision's view (pilots; used for the replay cells)."""
    views = []
    rng = random.Random(f"rt|{seed}|{K}|{D}|{c}|{others}|{remove_at}")
    task_rng = random.Random(seed)                      # identical task stream across arms with the same seed
    kmap = kind_map(seed)
    ids = random.Random(seed + 77).sample(range(10, 100), n_agents)
    active = ids[:]
    stats = {a: dict(done={j: 0 for j in TYPES}, work=0, rest=0, wrong=0) for a in ids}
    hist = {a: [] for a in ids}
    queues = {j: [] for j in TYPES}
    nxt, points, lost, rec_rounds, removed = 1, 0.0, 0, [], []
    for t in range(1, rounds + 1):
        arr = arrivals(D, n_agents)                     # demand stays at the original team's level after removal
        for j in TYPES:
            for _ in range(arr[j]):
                task = make_task(nxt, j, kmap[j], task_rng); nxt += 1
                if len(queues[j]) < CAP:
                    queues[j].append(task)
                else:
                    lost += 1
        order = active[:]; rng.shuffle(order)
        rec = dict(t=t, order=order, acts={}, open_start={j: len(queues[j]) for j in TYPES})
        earlier = []
        for pos, a in enumerate(order, 1):
            open_tasks = sorted([x for j in TYPES for x in queues[j]], key=lambda x: x["id"])
            rows = [x for x in active]; rng.shuffle(rows)
            type_order = list(TYPES); rng.shuffle(type_order)
            view = dict(me=a, K=K, others=others, t=t, T=rounds, n=len(active), pos=pos, points=points,
                        nA=len(queues["A"]), nB=len(queues["B"]), stats=stats, hist=hist, order_rows=rows,
                        earlier=earlier, open=open_tasks, type_order=type_order, c=c, D=D)
            if log_views:
                views.append(snapshot(view))
            if decide:
                text = decide(view)
            else:
                text = reply(model, system_prompt(a, c, D, len(active), rounds, K, others), render(view), view, tag,
                             reasoning)
            act = parse(text)
            st = stats[a]
            listed = [x["id"] for j in type_order for x in open_tasks if x["type"] == j]
            if act[0] == "do":
                task = next((x for x in open_tasks if x["id"] == act[1]), None)
                if task is None:
                    rec["acts"][str(a)] = ["stale", act[1]]
                    hist[a].append("nothing"); earlier.append(f"- Agent {a} named a task that was not open")
                    continue
                ok = correct(task, act[2])
                points -= c if FRAME == "effort" else 0.0; st["work"] += 1
                if ok:
                    points += 1; st["done"][task["type"]] += 1
                    queues[task["type"]] = [x for x in queues[task["type"]] if x["id"] != task["id"]]
                else:
                    st["wrong"] += 1
                hist[a].append(task["type"])
                rec["acts"][str(a)] = ["do", task["type"], task["id"], bool(ok), listed.index(task["id"]),
                                       type_order[0], view["nA"], view["nB"]]
                earlier.append(f"- Agent {a} did T{task['id']} ({task['type']})" + ("" if ok else ", wrong answer"))
            elif act[0] == "rest":
                st["rest"] += 1; hist[a].append("rest")
                if FRAME == "rest":
                    points += c
                rec["acts"][str(a)] = ["rest", None, None, None, None, type_order[0], view["nA"], view["nB"]]
                earlier.append(f"- Agent {a} rested")
            else:
                hist[a].append("nothing")
                rec["acts"][str(a)] = ["invalid", (text or "")[:60]]
                earlier.append(f"- Agent {a} did nothing")
        waiting = sum(len(queues[j]) for j in TYPES)
        points -= waiting
        rec.update(points=points, waiting={j: len(queues[j]) for j in TYPES})
        rec_rounds.append(rec)
        if remove_at and t == remove_at:
            def a_share(a):
                w = stats[a]["done"]["A"] + stats[a]["done"]["B"]
                return (stats[a]["done"]["A"] / w if w else 0.0, stats[a]["done"]["A"])
            gone = sorted(active, key=a_share, reverse=True)[:n_remove]
            removed = gone
            active = [a for a in active if a not in gone]
    return dict(model=model, K=K, D=D, c=c, others=others, seed=seed, N=n_agents, T=rounds, ids=ids, kinds=kmap,
                remove_at=remove_at, removed=removed, stats={str(k): v for k, v in stats.items()}, points=points,
                lost=lost, history=rec_rounds, **({"reasoning": reasoning} if reasoning != "none" else {}),
                **({"views": views} if log_views else {}))


# ----------------------------------------------------------------------------------------------- metrics

def dol_index(counts):
    """Gorelick et al. (2004) division-of-labour index: I(agent; task type) / H(task type), from a count matrix."""
    M = np.asarray(counts, float)
    tot = M.sum()
    if tot == 0:
        return float("nan")
    P = M / tot
    pa, pt = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    I = float((P[nz] * np.log(P[nz] / (pa @ pt)[nz])).sum())
    Ht = float(-(pt[pt > 0] * np.log(pt[pt > 0])).sum())
    return I / Ht if Ht > 0 else float("nan")


def _work_acts(rec, t0, t1):
    out = []
    for r in rec["history"]:
        if t0 <= r["t"] <= t1:
            for a, x in r["acts"].items():
                if x[0] == "do":
                    out.append((r["t"], a, x[1]))
    return out


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def metrics(rec, warm=WARM, n_perm=200, t1=None):
    """Population summaries over rounds warm+1 .. t1 (default: last round, or the removal round)."""
    t1 = t1 or (rec["remove_at"] or rec["T"])
    acts = _work_acts(rec, warm + 1, t1)
    agents = sorted({a for r in rec["history"] if warm < r["t"] <= t1 for a in r["order"]}, key=str)
    agents = [str(a) for a in agents]
    idx = {a: i for i, a in enumerate(agents)}
    M = np.zeros((len(agents), 2))
    for _, a, j in acts:
        M[idx[a], TYPES.index(j)] += 1
    dol = dol_index(M)
    # permutation null: shuffle which agent did each work act within its round (keeps per-round composition)
    prng = np.random.default_rng(0)
    by_t = {}
    for t, a, j in acts:
        by_t.setdefault(t, []).append((a, j))
    null = []
    for _ in range(n_perm):
        Mp = np.zeros_like(M)
        for t, lst in by_t.items():
            ag = [a for a, _ in lst]; prng.shuffle(ag)
            for a, (_, j) in zip(ag, lst):
                Mp[idx[a], TYPES.index(j)] += 1
        null.append(dol_index(Mp))
    work = [sum(1 for _, a, _ in acts if a == ag) for ag in agents]
    # persistence: same type as the agent's own previous work turn (all rounds up to t1), vs the population type mix
    same = tot = 0
    last = {}
    for r in rec["history"]:
        if r["t"] > t1:
            break
        for a, x in r["acts"].items():
            if x[0] == "do":
                if a in last and r["t"] > warm:
                    tot += 1; same += int(last[a] == x[1])
                last[a] = x[1]
    pA = M[:, 0].sum() / max(M.sum(), 1)
    turns = sum(1 for r in rec["history"] if warm < r["t"] <= t1 for _ in r["order"])
    rests = sum(1 for r in rec["history"] if warm < r["t"] <= t1 for x in r["acts"].values() if x[0] == "rest")
    inval = sum(1 for r in rec["history"] if warm < r["t"] <= t1 for x in r["acts"].values() if x[0] in ("invalid", "stale"))
    wait = [sum(r["waiting"].values()) for r in rec["history"] if warm < r["t"] <= t1]
    return dict(dol=dol, dol_null=float(np.nanmean(null)), dol_excess=dol - float(np.nanmean(null)),
                gini_work=gini(work), persist=same / tot if tot else float("nan"),
                persist_null=float(pA ** 2 + (1 - pA) ** 2), work_rate=len(acts) / max(turns, 1),
                rest_rate=rests / max(turns, 1), invalid_rate=inval / max(turns, 1), mean_waiting=float(np.mean(wait)),
                points=rec["points"])


def removal_metrics(rec, pre=(9, 16), post=(25, 32), spec_min=4, spec_share=0.75):
    """Removal arm: the remaining agents before (rounds 9-16) and after (25-32) the A specialists left at round 16.
    spec_*: remaining agents with >= spec_min work turns in the window, >= spec_share of them on A."""
    gone = {str(a) for a in rec["removed"]}
    rem = [str(a) for a in rec["ids"] if str(a) not in gone]

    def window(t0, t1):
        cnt = {a: {"A": 0, "B": 0} for a in rem}
        for r in rec["history"]:
            if t0 <= r["t"] <= t1:
                for a, x in r["acts"].items():
                    if x[0] == "do" and a in cnt:
                        cnt[a][x[1]] += 1
        tot = sum(c["A"] + c["B"] for c in cnt.values())
        share = sum(c["A"] for c in cnt.values()) / tot if tot else float("nan")
        spec = sum(1 for c in cnt.values() if c["A"] + c["B"] >= spec_min and c["A"] >= spec_share * (c["A"] + c["B"]))
        return share, spec
    sp, kp = window(*pre); so, ko = window(*post)
    wA = {r["t"]: r["waiting"]["A"] for r in rec["history"]}
    return dict(a_share_pre=sp, a_share_post=so, spec_pre=kp, spec_post=ko, spec_gain=ko - kp,
                waitA_pre=float(np.mean([wA[t] for t in range(13, 17)])),
                waitA_peak=float(max(wA[t] for t in range(17, 25))),
                waitA_post=float(np.mean([wA[t] for t in range(29, 33)])))


# ----------------------------------------------------------------------------------------------- scripted agents

def _remaining(view):
    return view["n"] - view["pos"] + 1


def policy_rational(view, rng=None):
    """Team-optimal: work iff open tasks >= agents still to move (incl. me); type with the larger backlog."""
    if not view["open"] or len(view["open"]) < _remaining(view):
        return "REST"
    j = "A" if view["nA"] >= view["nB"] else "B"
    x = next(x for x in view["open"] if x["type"] == j)
    return f"DO T{x['id']}: {x['answer']}"


def policy_always(view, rng=None):
    """Work whenever anything is open; take the first listed task (foraging for work by position)."""
    if not view["open"]:
        return "REST"
    j = view["type_order"][0] if any(x["type"] == view["type_order"][0] for x in view["open"]) else view["type_order"][1]
    x = next(x for x in view["open"] if x["type"] == j)
    return f"DO T{x['id']}: {x['answer']}"


def make_hill(n=2.0, theta=4.0, xi=0.25, phi=0.05, beta_pos=0.0, g_rem=0.0, k0_shift=0.0, acc=1.0, seed=0):
    """Scripted response-threshold agent (multinomial Hill-logit). Uses only what the prompt shows: own history only
    if K > 0. ln theta_j = ln theta - xi * (#j in last 8 turns) + phi * (#turns in last 8 not spent on j).
    beta_pos: bonus for the first listed type; g_rem: utility of work per ln(agents still to move); k0_shift: utility
    shift of work when no own history is shown; acc: probability a DO answer is correct.
    In the fitted parametrization (rt_forecast.py) this is kappa = n*(xi + phi), omega = 0, rho = -n*phi,
    a = -n ln theta, a0 = k0_shift, b1 = beta_pos, g = g_rem."""
    rng = random.Random(seed)

    def decide(view):
        h = view["hist"][view["me"]][-HWIN:] if view["K"] > 0 else []
        L = len(h)
        u = {}
        for j in TYPES:
            s = view["n" + j]
            if s == 0:
                continue
            lt = np.log(theta) - xi * h.count(j) + phi * (L - h.count(j))
            u[j] = (n * (np.log(s) - lt) + beta_pos * (view["type_order"][0] == j)
                    + g_rem * np.log(view["n"] - view["pos"] + 1) + (k0_shift if view["K"] == 0 else 0.0))
        if not u:
            return "REST"
        w = {j: np.exp(v) for j, v in u.items()}
        tot = 1.0 + sum(w.values())
        r = rng.random() * tot
        for j, v in w.items():
            if r < v:
                x = next(x for x in view["open"] if x["type"] == j)
                return f"DO T{x['id']}: {x['answer'] if rng.random() < acc else 'x'}"
            r -= v
        return "REST"
    return decide


# ----------------------------------------------------------------------------------------------- LLM or test double

# RT_FAKE=1 replaces every call by a scripted agent with known parameters (offline pipeline test; no API calls).
FAKE = os.environ.get("RT_FAKE") == "1"
FAKE_PARAMS = {"gemma": dict(n=1.5, theta=3.0, xi=0.0, phi=0.0, beta_pos=1.5, g_rem=-0.3, acc=0.97),
               "qwen": dict(n=2.0, theta=4.0, xi=0.3, phi=0.05, beta_pos=0.3, g_rem=-0.2, k0_shift=0.5, acc=0.97),
               "glm": dict(n=2.0, theta=3.0, xi=0.4, phi=0.05, beta_pos=0.2, g_rem=-0.2, acc=0.97)}
_fake_n = [0]


def fake_params(model):
    return next(v for k, v in FAKE_PARAMS.items() if k in model)


def reply(model, sysp, prompt, view, tag, reasoning="none"):
    if FAKE:
        _fake_n[0] += 1
        return make_hill(**fake_params(model), seed=zlib.crc32(f"{prompt}|{_fake_n[0]}".encode()))(view)
    from llm import call
    return call(model, sysp, prompt, tag=tag, max_tokens=MAX_TOKENS if reasoning == "none" else 2000,
                reasoning=reasoning)


# ----------------------------------------------------------------------------------------------- stages (calls)

MODELS = ("adaption:gemma-4-31b-it", "adaption:qwen3.8-27b")
PILOT_SEEDS = (7, 8)
POP_SEEDS = tuple(range(51001, 51009))
PREREG = R / "rt_prereg_freeze.json"
CCHOICE = R / "rt_c_choice.json"
FORECAST = Path(os.environ.get("RT_FORECAST", R / "rt_predictions.json"))   # RT-A1: GLM-5.2 has its own file


def arms(stage):
    """Registered population arms: dicts of run_pop keyword arguments (without model, seed, c)."""
    if stage == "pilot":
        return [dict(K=8, D=0.5)]
    if stage == "RT3":
        out = [dict(K=K, D=D) for K in (0, 8) for D in (0.25, 0.5, 0.75, 1.0)]
        out.append(dict(K=8, D=0.5, others="hidden"))
        out.append(dict(K=8, D=0.5, rounds=32, remove_at=16))
        return out
    if stage == "RT4":
        return [dict(K=8, D=0.5, n_agents=4), dict(K=8, D=0.5, n_agents=16)]
    raise ValueError(stage)


def arm_key(arm):
    a = dict(K=8, D=0.5, others="shown", rounds=T, n_agents=N, remove_at=None); a.update(arm)
    return f"K={a['K']}|D={a['D']}|others={a['others']}|T={a['rounds']}|N={a['n_agents']}|rm={a['remove_at']}"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(path):
    if not Path(path).exists():
        sys.exit(f"missing {path}; refusing to start")
    return json.load(open(path))["sha256"]


def chosen_c():
    if not CCHOICE.exists():
        sys.exit("effort cost not chosen yet (results/rt_c_choice.json missing)")
    return float(json.load(open(CCHOICE))["c"])


def cmd_run(a):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    pre = stamp(PREREG)
    fc = stamp(FORECAST) if a.stage != "pilot" else None
    c = chosen_c()
    out = R / f"rt_{a.stage.lower()}_pop.jsonl"
    seeds = a.seeds or (PILOT_SEEDS if a.stage == "pilot" else POP_SEEDS)
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l); done.add((r["model"], r["arm"], r["seed"]))
    jobs = [(m, arm, s) for m in a.models for arm in arms(a.stage) for s in seeds
            if (m, arm_key(arm), s) not in done]
    print(f"{a.stage}: {len(jobs)} populations to run, c = {c}", flush=True)
    lock = threading.Lock()

    def go(job):
        m, arm, s = job
        rec = run_pop(m, c=c, seed=s, tag=f"rt:{a.stage}", log_views=(a.stage == "pilot"), **arm)
        rec.update(arm=arm_key(arm), stage=a.stage, prereg_sha=pre, **({"forecast_sha": fc} if fc else {}))
        with lock, open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  done {m} {arm_key(arm)} seed {s}: points {rec['points']:.1f}", flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, jobs))


# ----------------------------------------------------------------------------------------------- commands

def cmd_mock(a):
    for K in a.K:
        for D in a.D:
            for c in a.c:
                rows = []
                for seed in a.seeds:
                    pol = (make_hill(xi=a.xi, seed=seed) if a.policy == "hill" else
                           {"rational": policy_rational, "always": policy_always}[a.policy])
                    rec = run_pop("mock", K, D, c, seed, others=a.others, decide=pol, remove_at=a.remove_at)
                    rows.append(metrics(rec))
                keys = ("dol", "dol_null", "dol_excess", "gini_work", "persist", "work_rate", "mean_waiting", "points")
                print(f"{a.policy:8s} K={K} D={D} c={c}: " +
                      "  ".join(f"{k} {np.nanmean([r[k] for r in rows]):.3f}" for k in keys))


def cmd_show(a):
    def grab(view):
        if view["t"] == a.round and view["pos"] == a.pos:
            print("=== SYSTEM\n" + system_prompt(view["me"], view["c"], view["D"], view["n"], view["T"], view["K"],
                                                  view["others"]))
            print("=== USER\n" + render(view))
        return make_hill(seed=1)(view)
    run_pop("mock", a.K, a.D, a.c, 41001, others=a.others, decide=grab)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("mock")
    m.add_argument("--policy", default="rational", choices=["rational", "always", "hill"])
    m.add_argument("--K", type=int, nargs="+", default=[8])
    m.add_argument("--D", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    m.add_argument("--c", type=float, nargs="+", default=[0.5])
    m.add_argument("--xi", type=float, default=0.25)
    m.add_argument("--others", default="shown")
    m.add_argument("--remove-at", type=int, default=None)
    m.add_argument("--seeds", type=int, nargs="+", default=list(range(41001, 41009)))
    s = sp.add_parser("show")
    s.add_argument("--K", type=int, default=8); s.add_argument("--D", type=float, default=0.5)
    s.add_argument("--c", type=float, default=0.5); s.add_argument("--others", default="shown")
    s.add_argument("--round", type=int, default=6); s.add_argument("--pos", type=int, default=3)
    r = sp.add_parser("run")
    r.add_argument("--stage", required=True, choices=["pilot", "RT3", "RT4"])
    r.add_argument("--models", nargs="+", required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=None)
    r.add_argument("--parallel", type=int, default=10)
    a = p.parse_args()
    {"mock": cmd_mock, "show": cmd_show, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
