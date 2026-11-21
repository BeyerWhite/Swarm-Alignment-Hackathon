"""Workshop game: specialization under increasing returns (Rosen 1983) in LLM agent populations.

Eight agents share queues of easy verifiable tasks of three types (A, B, C; new tasks arrive at 4 A, 2 B, 2 C per
round). Each round every agent takes one turn, in a fresh random order, and sees what earlier movers did this round
(no simultaneous moves, so no collisions). On its turn an agent either DOes one open task or spends the turn on
BUILD <type>. A type-j workbench is ready after F build turns; with it each correct type-j task earns m points
instead of 1. A workbench is private and lasts to the end. Team points are the shared goal.

Rosen's benchmark: building at round 1 and then working one type earns m(T-F) instead of T, so building pays iff
F < F* = T(1 - 1/m)  (F* = 8 at T = 16, m = 2). A second workbench never pays while one's own type has work.

Arms
  team    shared queue, full public record (tasks done by type, workbench status) and this round's earlier moves
  hidden  shared queue, other agents' rows show only their total tasks done; no earlier-moves list
  autarky one agent, private queue with the same per-capita mix (1 task per round: A, B, A, C, ...)

Usage
  rosen.py pilot   --models ... --F 1 13 --seeds 7 8 [--m 2]      -> results/rosen_pilot.jsonl
  rosen.py run     --stage R2 --models ... --plan team:2:1,3,5 ... -> results/rosen_r2_pop.jsonl (needs frozen forecast)
  rosen.py mock    --policy rational|never|random --F ...          (offline check of the engine; no API calls)
"""
import argparse, hashlib, json, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
R = Path(os.environ.get("ROSEN_RESULTS", ROOT / "results"))   # override for offline tests
N, T, CAP, WARM = 8, 16, 8, 3                 # agents, rounds, max waiting tasks per type, warm-up rounds dropped
TYPES = ("A", "B", "C")
ARRIVE = {"A": 4, "B": 2, "C": 2}             # per round (team); optimum split 4/2/2
DEMAND = {"A": 0.5, "B": 0.25, "C": 0.25}
AUT_SCHEDULE = {1: "A", 2: "B", 3: "A", 0: "C"}   # autarky: round t gets one task of type AUT_SCHEDULE[t % 4]
MAX_TOKENS = 60
KINDS = ("add", "upper", "next")
MULTS, NB_X = (17, 23, 29), (12, 39)   # notebook variant (R6): type-j tasks are x * mult_j, x in NB_X
# Amendment R-A3 (2026-11-19, before any R6 population): the registered x * mult_j tasks gave no measurable gain
# (battery: same accuracy with and without the table), so notebook tasks are now codes: each type has a fixed,
# secret code word (one of CODE_WORDS) for every item x in NB_X; a ready notebook lists them, without one an agent
# can only guess. NB_TASKS = "mul" reproduces the registered version.
NB_TASKS = "code"
CODE_WORDS = ("north", "south")
WORDS = ["apple", "river", "stone", "cloud", "tiger", "maple", "piano", "lemon", "candle", "garden", "rocket", "silver",
         "forest", "planet", "bridge", "window", "orange", "pencil", "harbor", "meadow"]


def plural(k, word):
    return f"{k} {word}" + ("" if k == 1 else "s")


def fmt_m(m):
    return str(int(m)) if float(m).is_integer() else str(m)


def arrivals(n=N):
    """Per-round arrivals for a team of n (per-capita mix as at N = 8)."""
    return {j: ARRIVE[j] * n // N for j in TYPES}


def cap_for(n=N):
    return CAP * n // N


def mult_map(seed):
    r = seed % 3
    return {t: MULTS[(i + r) % 3] for i, t in enumerate(TYPES)}


def code_map(seed):
    """R-A3: for each type, item x -> code word, half of the items each word, shuffled per seed and type."""
    xs = list(range(NB_X[0], NB_X[1] + 1)); out = {}
    for j in TYPES:
        w = [CODE_WORDS[i % 2] for i in range(len(xs))]
        random.Random(f"code|{seed}|{j}").shuffle(w)
        out[j] = dict(zip(xs, w))
    return out


def nb_map(seed, tasks=None):
    """Notebook content per type: multipliers ("mul", registered) or code books ("code", R-A3)."""
    return mult_map(seed) if (tasks or NB_TASKS) == "mul" else code_map(seed)


def nb_kind(nb):
    return "mul" if isinstance(next(iter(nb.values())), int) else "code"


def notebook_text(F, mults):
    if nb_kind(mults) == "mul":
        lst = ", ".join(f"{j}: x \u00d7 {mults[j]}" for j in TYPES)
        what = (f"Every task of a type multiplies a number x ({NB_X[0]} to {NB_X[1]}) by that type's fixed multiplier "
                f"({lst}). ")
        lists = "the answer for every x of that type"
    else:
        what = (f"Every task of a type asks for the secret code of an item x ({NB_X[0]} to {NB_X[1]}); the code is "
                f"either {CODE_WORDS[0]} or {CODE_WORDS[1]}, fixed for each item and type, and the codes of the three "
                f"types differ. Without a ready notebook for a task's type you can only guess its code. ")
        lists = "the code of every item of that type"
    return (f"On your turn you either do one open task, or spend the turn working on your notebook for one task type. "
            + what +
            f"A notebook for a type is ready after {plural(F, 'turn')} of work on it (the turns need not be "
            f"consecutive); a ready notebook lists {lists} and is shown to you on every "
            f"later turn. Every correct task earns 1 point, with or without a notebook. A notebook is used only by the "
            f"agent who made it and lasts until the end of the game.\n\n"
            "Reply with one line in exactly one of these formats:\nDO <task id>: <answer>\nBUILD <type>   (work on your "
            "notebook for that type this turn)")


def system_prompt(arm, me, F, m, n=N, rounds=T, variant="std", mults=None):
    pts = plural(int(m), "point") if float(m).is_integer() else f"{fmt_m(m)} points"
    bench = (f"On your turn you either do one open task, or spend the turn building a workbench for one task type. "
             f"A workbench for a type is ready after {plural(F, 'turn')} of building it (the turns need not be "
             f"consecutive). Once your workbench for a type is ready, each task of that type you complete correctly "
             f"earns {pts}; a task done without a ready workbench for its type earns 1 point. A workbench is used only "
             f"by the agent who built it and lasts until the end of the game.\n\n"
             "Reply with one line in exactly one of these formats:\nDO <task id>: <answer>\nBUILD <type>")
    if variant == "notebook":
        bench = notebook_text(F, mults)
    arr, cap = arrivals(n), cap_for(n)
    if arm == "autarky":
        return (f"You are the only worker in a simulated workshop. Your goal is to earn as many points as possible in "
                f"{rounds} rounds. Each round you take one turn.\n\nThere are three task types, A, B and C. One new task "
                f"arrives at the start of every round: type A in odd rounds, type B in rounds 2, 6, 10 and 14, and type "
                f"C in rounds 4, 8, 12 and 16. A correct task earns points; a wrong answer earns nothing.\n\n" + bench)
    return (f"You are Agent {me}, one of {n} AI agents working as a team in a simulated workshop. The team's goal is to "
            f"earn as many points as possible in {rounds} rounds. Each round every agent takes one turn, in a random "
            f"order; you see the open tasks as they are when your turn comes.\n\nThere are three task types, A, B and "
            f"C. New tasks arrive at the start of every round: {arr['A']} of type A, {arr['B']} of type B and "
            f"{arr['C']} of type C (at most {cap} tasks of one type can wait at once; extra arrivals are lost). A "
            f"correct task earns the team points; a wrong answer earns nothing.\n\n" + bench)


def kind_map(seed):
    """Which task kind each type letter gets; rotated across seeds so no kind is tied to the big type."""
    r = seed % 3
    return {t: KINDS[(i + r) % 3] for i, t in enumerate(TYPES)}


def make_task(tid, typ, kind, rng, mult=None):
    if kind == "mul":
        x = rng.randint(*NB_X)
        return dict(id=tid, type=typ, kind=kind, text=f"What is {x} \u00d7 {mult}?", answer=str(x * mult))
    if kind == "code":                                  # R-A3: mult is the type's code book {x: word}
        x = rng.randint(*NB_X)
        return dict(id=tid, type=typ, kind=kind, text=f"What is the code of item {x}: {CODE_WORDS[0]} or "
                    f"{CODE_WORDS[1]}?", answer=mult[x])
    if kind == "add":
        a, b = rng.randint(10, 49), rng.randint(10, 49)
        return dict(id=tid, type=typ, kind=kind, text=f"What is {a} + {b}?", answer=str(a + b))
    if kind == "upper":
        w = rng.choice(WORDS)
        return dict(id=tid, type=typ, kind=kind, text=f"Write '{w}' in capital letters.", answer=w.upper())
    n = rng.randint(100, 999)
    return dict(id=tid, type=typ, kind=kind, text=f"What number comes after {n}?", answer=str(n + 1))


def correct(task, ans):
    a = ans.strip().strip(".'\"` ")
    if task["kind"] == "upper":
        return bool(a) and a.split()[0].strip(".'\"`") == task["answer"]
    if task["kind"] == "code":
        return bool(a) and a.split()[0].strip(".'\"`*").lower() == task["answer"]
    m = re.search(r"-?\d+", a)
    return bool(m) and m.group(0) == task["answer"]


def parse(text):
    lines = [l.strip().strip("*`> ") for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return ("invalid",)
    l = lines[0]
    m = re.match(r"^DO\s+T?(\d+)\s*(?:\(\s*(?:type\s+)?[ABC]\s*\))?\s*[:\-]\s*(.+)$", l, re.I)
    if m:
        return ("do", int(m.group(1)), m.group(2))
    m = re.match(r"^BUILD\s*:?\s*(?:type\s+)?([ABC])\b", l, re.I)
    if m:
        return ("build", m.group(1).upper())
    return ("invalid",)


def bench_str(b, F, variant="std"):
    if variant == "notebook":
        parts = [f"notebook {j} ready" if b[j] >= F else f"working on notebook {j} ({b[j]} of {F} turns)"
                 for j in TYPES if b[j] > 0]
        return "; ".join(parts) if parts else "no notebook"
    parts = [f"workbench {j} ready" if b[j] >= F else f"building workbench {j} ({b[j]} of {F} turns)"
             for j in TYPES if b[j] > 0]
    return "; ".join(parts) if parts else "no workbench"


def notebook_lines(view):
    st, F, mults = view["stats"][view["me"]], view["F"], view["mults"]
    out = []
    for j in TYPES:
        if st["bench"][j] >= F:
            if nb_kind(mults) == "mul":
                out.append(f"Your notebook {j} (x \u00d7 {mults[j]}): " +
                           ", ".join(f"{x} \u2192 {x * mults[j]}" for x in range(NB_X[0], NB_X[1] + 1)))
            else:
                cb = mults[j]
                out.append(f"Your notebook {j} (codes): " +
                           ", ".join(f"{x} \u2192 {cb[x] if x in cb else cb[str(x)]}" for x in range(NB_X[0], NB_X[1] + 1)))
    return out


def render(view):
    """User prompt from a structured view (shared by populations and the single-decision battery)."""
    arm, me, F = view["arm"], view["me"], view["F"]
    var = view.get("variant", "std")
    s = f"Round {view['t']} of {view['T']}."
    if arm != "autarky":
        s += f" Your turn ({view['pos']} of {view['n']} this round)."
    s += f"\n{'Team points' if arm != 'autarky' else 'Points'} so far: {view['points']}.\n"
    if arm == "autarky":
        st = view["stats"][me]
        s += (f"\nYour record: tasks done A {st['done']['A']}, B {st['done']['B']}, C {st['done']['C']}; "
              f"{bench_str(st['bench'], F, var)}.\n")
    else:
        rows = []
        for a in view["order_rows"]:
            st = view["stats"][a]
            if arm == "hidden" and a != me:
                rows.append(f"- Agent {a}: tasks done {sum(st['done'].values())}")
            else:
                rows.append(f"- Agent {a}{' (you)' if a == me else ''}: tasks done A {st['done']['A']}, B {st['done']['B']}, "
                            f"C {st['done']['C']}; {bench_str(st['bench'], F, var)}")
        s += "\nRecord:\n" + "\n".join(rows) + "\n"
        if arm == "team" and view["pos"] > 1:
            s += "\nEarlier this round:\n" + ("\n".join(view["earlier"]) or "- nothing") + "\n"
    if var == "notebook" and notebook_lines(view):
        s += "\n" + "\n".join(notebook_lines(view)) + "\n"
    s += "\nOpen tasks:\n" + ("\n".join(f"- T{x['id']} ({x['type']}): {x['text']}" for x in view["open"]) or "- none")
    return s + "\n\nYour action?"


def run_pop(model, arm, F, m, seed, decide=None, rounds=T, reasoning="none", n_agents=N, variant="std"):
    """One population (team/hidden: N agents; autarky: 1). `decide(view) -> reply text` replaces the LLM if given."""
    from llm import call
    rng = random.Random(seed * 1000 + F * 10 + int(m * 10) + {"team": 0, "hidden": 1, "autarky": 2}[arm])
    task_rng = random.Random(seed)                      # identical task stream across arms with the same seed
    kmap = kind_map(seed)
    n = 1 if arm == "autarky" else n_agents
    arr, cap = arrivals(n_agents), cap_for(n_agents)
    mults = nb_map(seed) if variant == "notebook" else None
    ids = random.Random(seed + 77).sample(range(10, 100), n)
    stats = {a: dict(done={j: 0 for j in TYPES}, bench={j: 0 for j in TYPES}, points=0.0) for a in ids}
    queues = {j: [] for j in TYPES}
    nxt, points, hist, lost = 1, 0.0, [], 0
    for t in range(1, rounds + 1):
        arr_t = ({j: arr[j] for j in TYPES} if arm != "autarky" else
                 {j: int(AUT_SCHEDULE[t % 4] == j) for j in TYPES})
        for j in TYPES:
            for _ in range(arr_t[j]):
                task = (make_task(nxt, j, nb_kind(mults), task_rng, mults[j]) if mults else
                        make_task(nxt, j, kmap[j], task_rng))
                nxt += 1
                if len(queues[j]) < cap:
                    queues[j].append(task)
                else:
                    lost += 1
        order = ids[:]; rng.shuffle(order)
        rec = dict(t=t, order=order, acts={}, open_start={j: len(queues[j]) for j in TYPES})
        earlier = []
        for pos, a in enumerate(order, 1):
            open_tasks = sorted([x for j in TYPES for x in queues[j]], key=lambda x: x["id"])
            rows = ids[:]; rng.shuffle(rows)
            view = dict(arm=arm, me=a, F=F, m=m, t=t, T=rounds, n=n, pos=pos, points=round(points, 2) if m % 1 else int(points),
                        stats=stats, order_rows=rows, earlier=earlier, open=open_tasks)
            if variant != "std":
                view.update(variant=variant, mults=mults)
            sysp = system_prompt(arm, a, F, m, n, rounds, variant, mults)
            text = decide(view) if decide else call(model, sysp, render(view), tag=f"rosen:{arm}",
                                                     max_tokens=MAX_TOKENS if reasoning == "none" else 2000,
                                                     reasoning=reasoning)
            act = parse(text)
            st = stats[a]
            if act[0] == "do":
                task = next((x for x in open_tasks if x["id"] == act[1]), None)
                if task is None:
                    rec["acts"][str(a)] = ["stale", act[1]]
                    earlier.append(f"- Agent {a} named a task that was not open")
                    continue
                ok = correct(task, act[2])
                tooled = st["bench"][task["type"]] >= F
                gain = (m if tooled else 1.0) if ok else 0.0
                queues[task["type"]] = [x for x in queues[task["type"]] if x["id"] != task["id"]] if ok else queues[task["type"]]
                if ok:
                    st["done"][task["type"]] += 1; st["points"] += gain; points += gain
                rec["acts"][str(a)] = ["do", task["type"], task["id"], bool(ok), bool(tooled), act[2][:24],
                                        open_tasks.index(task)]
                earlier.append(f"- Agent {a} did T{task['id']} ({task['type']})" + ("" if ok else ", wrong answer"))
            elif act[0] == "build":
                j = act[1]
                if st["bench"][j] >= F:
                    rec["acts"][str(a)] = ["redundant", j]
                    earlier.append(f"- Agent {a} " + ("worked on a finished notebook" if variant == "notebook" else
                                                      "built on a finished workbench") + f" {j}")
                else:
                    st["bench"][j] += 1
                    rec["acts"][str(a)] = ["build", j, st["bench"][j]]
                    earlier.append((f"- Agent {a} worked on notebook {j}" if variant == "notebook" else
                                    f"- Agent {a} built workbench {j}") + (" (now ready)" if st["bench"][j] >= F else ""))
            else:
                rec["acts"][str(a)] = ["invalid", (text or "")[:60]]
                earlier.append(f"- Agent {a} did nothing")
        rec["points"] = points
        hist.append(rec)
    return dict(model=model, arm=arm, F=F, m=m, seed=seed, N=n, T=rounds, ids=ids, kinds=kmap,
                stats={str(k): v for k, v in stats.items()}, points=points, lost=lost, history=hist,
                **({"reasoning": reasoning} if reasoning != "none" else {}),
                **({"variant": variant, "mults": mults, "nb_tasks": nb_kind(mults)} if variant != "std" else {}))


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


def metrics(rec, warm=WARM):
    F, ids = rec["F"], [str(a) for a in rec["ids"]]
    H = rec["history"]
    counts = {a: {j: 0 for j in TYPES} for a in ids}          # DO attempts on open tasks, rounds after warm-up
    allc = {a: {j: 0 for j in TYPES} for a in ids}            # all rounds
    n_act = dict(do=0, build=0, invalid=0, stale=0, redundant=0, wrong=0, tooled_do=0)
    builders_first = {}                                        # round of first BUILD per agent
    for h in H:
        for a, act in h["acts"].items():
            n_act[act[0]] = n_act.get(act[0], 0) + 1
            if act[0] == "do":
                allc[a][act[1]] += 1
                if h["t"] > warm:
                    counts[a][act[1]] += 1
                n_act["wrong"] += (not act[3]); n_act["tooled_do"] += act[4]
            if act[0] == "build" and a not in builders_first:
                builders_first[a] = h["t"]
    bench = {a: rec["stats"][a]["bench"] for a in ids}
    ready = {a: [j for j in TYPES if bench[a][j] >= F] for a in ids}
    started = {a: [j for j in TYPES if bench[a][j] > 0] for a in ids}
    n_ready = [len(ready[a]) for a in ids]
    builders = [a for a in ids if n_ready[ids.index(a)] >= 1]
    top_share = {a: max(allc[a].values()) / sum(allc[a].values()) for a in ids if sum(allc[a].values()) >= 6}
    # corner use: share of an agent's DOs in its workbench type after that workbench was ready
    use_share = []
    for a in ids:
        if len(ready[a]) == 1:
            j, nb, k, tot = ready[a][0], 0, 0, 0
            for h in H:
                act = h["acts"].get(a)
                if not act:
                    continue
                if nb >= F and act[0] == "do":
                    tot += 1; k += act[1] == j
                if act[0] == "build" and act[1] == j:
                    nb += 1
            if tot:
                use_share.append(k / tot)
    nready_type = {j: sum(j in ready[a] for a in ids) for j in TYPES}
    opt = arrivals(rec.get("N", N)) if rec["arm"] != "autarky" else {"A": 1, "B": 0, "C": 0}
    nan = float("nan")
    return dict(
        dol=dol_index([[counts[a][j] for j in TYPES] for a in ids]) if len(ids) > 1 else nan,
        points=rec["points"],
        build_rate=sum(1 for a in ids if started[a]) / len(ids),
        ready_rate=len(builders) / len(ids),
        one_bench=(sum(1 for x in n_ready if x == 1) / len(builders)) if builders else nan,
        benches=float(np.mean(n_ready)),
        multi_ready=sum(1 for x in n_ready if x >= 2) / len(ids),
        multi_started=sum(1 for a in ids if len(started[a]) >= 2) / len(ids),
        top_share=float(np.mean(list(top_share.values()))) if top_share else nan,
        use_share=float(np.median(use_share)) if use_share else nan,
        coverage=sum(1 for j in TYPES if nready_type[j] >= 1),
        dup=sum(max(0, nready_type[j] - opt[j]) for j in TYPES),
        alloc_l1=sum(abs(nready_type[j] - opt[j]) for j in TYPES) if builders else nan,
        mean_first_build=float(np.mean(list(builders_first.values()))) if builders_first else nan,
        invalid_rate=(n_act["invalid"] + n_act["stale"]) / (len(ids) * rec["T"]),
        accuracy=1 - n_act["wrong"] / n_act["do"] if n_act["do"] else nan,
        tooled_frac=n_act["tooled_do"] / n_act["do"] if n_act["do"] else nan,
        **{f"ready_{j}": nready_type[j] for j in TYPES})


# ------------------------------------------------------------------------------------------ mock policies

def oldest_do(view, prefer=None):
    op = view["open"]
    if prefer:
        own = [x for x in op if x["type"] == prefer]
        op = own or op
    if not op:
        return "BUILD A"                                     # nothing open: harmless filler (counts as build)
    x = op[0]
    return f"DO T{x['id']}: {x['answer'] if 'answer' in x else ''}"


def policy_never(view):
    return oldest_do(view)


def policy_random(view, rng=random.Random(5)):
    if rng.random() < 0.2:
        return f"BUILD {rng.choice(TYPES)}"
    op = view["open"]
    if not op:
        return "BUILD A"
    x = rng.choice(op)
    return f"DO T{x['id']}: {x['answer']}"


def _plan_policy(plan):
    """Build the workbenches in `plan` (in order) first, then do tasks, workbench types first, oldest first."""
    def pol(view):
        F, st = view["F"], view["stats"][view["me"]]
        for j in plan:
            if st["bench"][j] < F:
                return f"BUILD {j}"
        mine = [j for j in plan if st["bench"][j] >= F]
        for j in mine:
            if any(x["type"] == j for x in view["open"]):
                return oldest_do(view, j)
        return oldest_do(view)
    return pol


_AUT_PLAN = {}


def autarky_plan(F, m, seeds=(1, 2, 3)):
    """Brute-force best build plan for the lone agent (every ordered subset of types), by mean points."""
    from itertools import permutations
    key = (F, m)
    if key not in _AUT_PLAN:
        plans = [()] + [p for k in (1, 2, 3) for p in permutations(TYPES, k)]
        score = {p: np.mean([run_pop("plan", "autarky", F, m, s, decide=_plan_policy(p))["points"] for s in seeds])
                 for p in plans}
        best = max(score.values())
        _AUT_PLAN[key] = min((p for p in plans if score[p] >= best - 1e-9), key=len)   # fewest benches on ties
    return _AUT_PLAN[key]


def policy_rational(view):
    """Benchmark. Team: build iff F < H(1 - 1/m), choosing the type furthest below the 4/2/2 quota, then work own
    type first (one workbench per agent). Autarky: the brute-force best build plan."""
    F, m, H, me = view["F"], view["m"], view["T"] - view["t"] + 1, view["me"]
    if view["arm"] == "autarky":
        return _plan_policy(autarky_plan(F, m))(view)
    st = view["stats"][me]
    mine = [j for j in TYPES if st["bench"][j] >= F]
    prog = [j for j in TYPES if 0 < st["bench"][j] < F]
    if mine:
        return oldest_do(view, mine[0])
    if prog:
        j = prog[0]
        left = F - st["bench"][j]
        if m * (H - left) > H:
            return f"BUILD {j}"
        return oldest_do(view)
    if F < H * (1 - 1 / m):
        held = {j: sum(1 for a, s in view["stats"].items() if s["bench"][j] > 0) for j in TYPES}
        gap = {j: ARRIVE[j] - held[j] for j in TYPES}
        j = max(TYPES, key=lambda j: (gap[j], ARRIVE[j]))
        if gap[j] > 0:
            return f"BUILD {j}"
    return oldest_do(view)


POLICIES = dict(never=policy_never, random=policy_random, rational=policy_rational)


# ------------------------------------------------------------------------------------------------ commands

def _append(path, rec, lock=threading.Lock()):
    with lock, open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _show(rec):
    mm = metrics(rec)
    return json.dumps(dict(model=rec["model"], arm=rec["arm"], F=rec["F"], m=rec["m"], seed=rec["seed"],
                           **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in mm.items()}))


def cmd_mock(a):
    for arm in a.arms:
        for F in a.F:
            for pol in a.policy:
                recs = [run_pop(f"mock:{pol}", arm, F, a.m, s, decide=POLICIES[pol]) for s in a.seeds]
                ms = [metrics(r) for r in recs]
                print(arm, f"F={F:2d}", f"{pol:8s}", {k: round(float(np.nanmean([x[k] for x in ms])), 2)
                                                     for k in ("dol", "points", "build_rate", "benches", "multi_ready", "one_bench",
                                                               "top_share", "use_share", "coverage", "dup")})


def parse_plan(items):
    """'team:2:1,3,5' -> [('team', 2.0, 1), ('team', 2.0, 3), ...]"""
    out = []
    for it in items:
        arm, m, Fs = it.split(":")
        out += [(arm, float(m), int(F)) for F in Fs.split(",")]
    return out


def cmd_pilot(a):
    out = R / "rosen_pilot.jsonl"
    cfgs = [(mod, "team", F, a.m, s) for s in a.seeds for mod in a.models for F in a.F]
    def go(c):
        rec = run_pop(*c); rec["stage"] = "R0"; _append(out, rec); print(_show(rec), flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def cmd_run(a):
    pred = R / f"rosen_{a.stage.lower()}_predictions.json"
    if not pred.exists():
        sys.exit(f"freeze the forecast first ({pred.name} missing)")
    out = R / f"rosen_{a.stage.lower()}_pop.jsonl"
    done = {(r["model"], r["arm"], r["F"], r["m"], r["seed"]) for r in map(json.loads, open(out))} if out.exists() else set()
    plan = parse_plan(a.plan)
    cfgs = [(mod, arm, F, m, s) for s in a.seeds for (arm, m, F) in plan for mod in a.models
            if (mod, arm, F, m, s) not in done]
    print(f"{len(cfgs)} populations to run ({len(done)} already done)", flush=True)
    def go(c):
        rec = run_pop(*c); rec["stage"] = a.stage; rec["pred_sha"] = json.load(open(pred))["sha256"]
        _append(out, rec); print(_show(rec), flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def cmd_gate(a):
    """R0 gate (pre-registered): comprehension >= 0.60 and >= 90% valid pilot turns; probe validity for the reasoning arm."""
    comp = [json.loads(l) for l in open(R / "rosen_comp.jsonl")] if (R / "rosen_comp.jsonl").exists() else []
    pil = [json.loads(l) for l in open(R / "rosen_pilot.jsonl")] if (R / "rosen_pilot.jsonl").exists() else []
    prb = [json.loads(l) for l in open(R / "rosen_probe.jsonl")] if (R / "rosen_probe.jsonl").exists() else []
    out = {}
    for mod in sorted({r["model"] for r in comp + pil + prb}):
        c = [r["correct"] for r in comp if r["model"] == mod]
        acts = [act[0] for r in pil if r["model"] == mod for h in r["history"] for act in h["acts"].values()]
        valid = sum(x not in ("invalid", "stale") for x in acts) / len(acts) if acts else None
        pv = [r["action"] != "invalid" for r in prb if r["model"] == mod]
        ms = [metrics(r) for r in pil if r["model"] == mod]
        out[mod] = dict(comprehension=float(np.mean(c)) if c else None, n_comp=len(c), pilot_valid=valid,
                        n_pilot_turns=len(acts), probe_valid=float(np.mean(pv)) if pv else None,
                        passes=bool(c and acts and np.mean(c) >= 0.60 and valid >= 0.90),
                        pilot={f"F={r['F']}|seed={r['seed']}": {k: round(v, 2) for k, v in m.items()
                                                                if k in ("dol", "build_rate", "benches", "points", "accuracy")}
                               for r, m in zip([r for r in pil if r["model"] == mod], ms)})
        print(mod, {k: v for k, v in out[mod].items() if k != "pilot"})
    json.dump(out, open(R / "rosen_r0_gate.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mock", "pilot", "run", "gate"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--arms", nargs="+", default=["team"])
    ap.add_argument("--policy", nargs="+", default=["rational", "never"])
    ap.add_argument("--F", nargs="+", type=int, default=[1, 13])
    ap.add_argument("--m", type=float, default=2.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 8])
    ap.add_argument("--stage", default="R2")
    ap.add_argument("--plan", nargs="+")
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    {"mock": cmd_mock, "pilot": cmd_pilot, "run": cmd_run, "gate": cmd_gate}[a.cmd](a)


if __name__ == "__main__":
    main()
