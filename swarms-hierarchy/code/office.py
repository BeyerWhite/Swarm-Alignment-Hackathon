"""Office game: is emergent hierarchy efficient? (Rosen 1982; Garicano 2000). Built on taskgame2.py.

Eight seats share a queue of verifiable tasks of four kinds (chosen in G0 so that models differ in accuracy). Each
round the seats act in two simultaneous waves of four (same-wave agents cannot see each other and can collide; the
second wave sees the first). On its turn an agent DOes one open task or PLANs: posts a plan assigning open tasks to
other agents, visible for the next two waves. A wrong answer wastes the turn and leaves the task open. The team goal
is the number of tasks done correctly. Agents see numbers, not model names.

Seats are model names ("adaption:..."), "oracle" (scripted planner: optimal assignment on the measured skill
matrix) or "absent" (an empty seat that never acts, so 7-agent arms keep the 8-seat wave structure).

Arms
  free       every agent may DO or PLAN                                        (G2, G3: emergent hierarchy)
  noplan     DO only                                                             (benchmark)
  ownorder   DO only, each agent sees the open tasks in its own random order     (benchmark; the paper's best arm)
  appointed  seat 0 is the coordinator and only PLANs; the others DO only         (G1c manager trials, oracle, G4)
Record
  full       per agent: correct/attempted by kind; plans posted; assignments followed and how many were correct
  lean       per agent: tasks done; rounds as coordinator; assignments followed  (taskgame2's record)

Usage (all population stages need the frozen task selection results/hier_g0_selection.json):
  office.py pilot  --seeds ...                                  -> results/hier_pilot.jsonl
  office.py g1c    --seeds ...                                  -> results/hier_g1c_pop.jsonl
  office.py g2 | bench | g3 | g4 --seeds ...                    -> results/hier_<stage>_pop.jsonl (need the freeze)
  office.py mock   --policy ...                                  (offline engine check, no API calls)
"""
import argparse, hashlib, json, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hier_tasks as HT

ROOT = Path(__file__).resolve().parents[1]
R = Path(os.environ.get("HIER_RESULTS", ROOT / "results"))      # override for offline tests
N, T, WAVE, WARM, LIVE, PER_KIND = 8, 20, 4, 5, 2, 3            # seats, rounds, wave size, warm-up, plan life, open per kind
MAX_TOKENS = 100
MODELS = ["adaption:gemma-4-31b-it", "adaption:gemma-4-26b-a4b-it", "adaption:qwen3.8-27b",
          "adaption:deepseek-v4-flash", "adaption:qwen3.5-397b", "adaption:glm-5.2"]
PANEL = ["adaption:gemma-4-31b-it"] * 2 + ["adaption:qwen3.8-27b"] * 2 + ["adaption:gemma-4-26b-a4b-it"] * 2 + \
        ["adaption:deepseek-v4-flash"]                         # G1c worker panel (7 seats)
DISPOSITION = "You tend to take charge and coordinate others' work."
_wlock = threading.Lock()

INTRO = ("You are Agent {me}, one of {n} AI agents sharing a work queue in a simulated sandbox. Each round the agents "
         "act in two waves of four. Agents in the same wave act at the same time and cannot see each other's choices; "
         "the second wave sees what the first wave did. If two agents do the same task, only one counts and the other "
         "agent's work is wasted. A wrong answer is wasted too, and the task stays open. Agents differ in how "
         "accurate they are on each kind of task. The team's goal is to complete as many tasks correctly as "
         "possible. ")
FMT_DO = "DO <task id>: <answer>"
FMT_PLAN = "PLAN: <task id>-><agent number>, <task id>-><agent number>, ..."
SYS_FREE = INTRO + ("On your turn, either do one open task yourself, or act as coordinator: post a plan that assigns "
                    "open tasks to other agents (you do no task on that turn). A plan stays visible to all agents for "
                    "the next two waves; agents may follow or ignore it. Reply with one line in exactly one of these "
                    "formats:\n" + FMT_DO + "\n" + FMT_PLAN)
SYS_DO = INTRO + "On your turn, do one open task. Reply with one line in exactly this format:\n" + FMT_DO
SYS_WORKER = INTRO + ("Agent {mgr} is the team's coordinator and posts plans that assign open tasks to agents; a plan "
                      "stays visible for the next two waves, and you may follow or ignore it. On your turn, do one "
                      "open task. Reply with one line in exactly this format:\n" + FMT_DO)
SYS_MANAGER = INTRO + ("You are the team's coordinator. On each of your turns you post a plan that assigns open tasks "
                       "to other agents; you do no tasks yourself. A plan stays visible to all agents for the next "
                       "two waves; agents may follow or ignore it. Reply with one line in exactly this format:\n"
                       + FMT_PLAN)


def load_selection():
    sel = json.load(open(R / "hier_g0_selection.json"))
    return [(c["kind"], c["level"]) for c in sel["chosen"]]


def load_skill():
    """Measured worker skill A[model][kind] (G1a), used by the oracle planner and the simulator."""
    return json.load(open(R / "hier_kernels.json"))["skill"]


# --------------------------------------------------------------------------------------------- parsing
def parse(text, ids, can_plan):
    lines = [l.strip().strip("*`> ") for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return ("invalid",)
    l = lines[0]
    m = re.match(r"^DO\s+T?(\d+)\s*[:\-]\s*(.+)$", l, re.I)
    if m:
        return ("do", int(m.group(1)), m.group(2))
    m = re.match(r"^PLAN\s*:?\s*(.*)$", l, re.I)
    if m and can_plan:
        pairs = [(int(a), int(b)) for a, b in re.findall(r"T?(\d+)\s*->\s*(?:Agent\s*)?(\d+)", m.group(1), re.I)]
        pairs = [(a, b) for a, b in pairs if b in ids]
        return ("plan", pairs) if pairs else ("invalid",)
    return ("invalid",)


# --------------------------------------------------------------------------------------------- prompts
def record_rows(me, stats, kinds, record, can_plan, rng):
    rows = list(stats.items()); rng.shuffle(rows)
    out = []
    for a, s in rows:
        tag = f"Agent {a}{' (you)' if a == me else ''}"
        if record == "full":
            by = ", ".join(f"{k} {s['ok'][k]}/{s['att'][k]}" for k in kinds)
            line = f"- {tag}: correct/attempted by kind: {by}"
            if can_plan:
                line += (f"; plans posted {s['coord']}; assignments followed {s['followed']}, of which correct "
                         f"{s['fcorrect']}")
        else:
            line = f"- {tag}: tasks done {s['done']}"
            if can_plan:
                line += f", rounds as coordinator {s['coord']}, assignments followed {s['followed']}"
        out.append(line)
    return "\n".join(out)


def build_prompt(st):
    """st: the state dict handed to every seat's policy (see run_pop). Returns the user prompt."""
    me, rng = st["me"], random.Random(st["prompt_seed"])
    s = f"Round {st['t']} of {st['T']}, wave {st['wave']} of 2.\n"
    s += f"\nPublic record:\n{record_rows(me, st['stats'], st['kinds'], st['record'], st['plans_on'], rng)}\n"
    if st["plans_on"]:
        live = st["live_plans"]
        pl = "\n".join(f"- Agent {p}: " + ", ".join(f"T{tid}->{'you' if b == me else 'Agent ' + str(b)}"
                                                    for tid, b in prs) for p, prs in live.items()) or "- none"
        mine = [f"T{tid} (from Agent {p})" for p, tid in st["mine"]]
        s += f"\nCurrent plans (open tasks only):\n{pl}\n"
        if not st["is_manager"]:
            s += f"Assigned to you: {', '.join(mine) if mine else 'nothing'}\n"
    if st["wave"] == 2:
        s += "\nFirst wave this round:\n" + ("\n".join(st["wave1_log"]) or "- nothing") + "\n"
    s += "\nOpen tasks:\n" + "\n".join(f"- T{x['id']} [{x['kind']}]: {x['text']}" for x in st["shown"])
    s += "\n\nYour action?"
    return s


def system_for(st):
    me, arm = st["me"], st["arm"]
    if arm == "free":
        sys_ = SYS_FREE
    elif arm in ("noplan", "ownorder"):
        sys_ = SYS_DO
    elif st["is_manager"]:
        sys_ = SYS_MANAGER
    else:
        sys_ = SYS_WORKER.replace("{mgr}", str(st["manager_id"]))
    sys_ = sys_.format(me=me, n=st["n_active"])
    if st["disposition"]:
        sys_ += " " + DISPOSITION
    return sys_


# --------------------------------------------------------------------------------------------- seat policies
def llm_policy(model, tag):
    from llm import call

    def act(st):
        return call(model, system_for(st), build_prompt(st), tag=tag, max_tokens=MAX_TOKENS)
    return act


def oracle_plan(st, skill):
    """Optimal one-task-per-agent assignment of open tasks to the other active agents (Hungarian on skill)."""
    from scipy.optimize import linear_sum_assignment
    others = [a for a in st["active_ids"] if a != st["me"]]
    tasks = st["open_tasks"]
    if not others or not tasks:
        return ""
    M = np.array([[skill[st["seat_model"][a]][x["kind"]] for x in tasks] for a in others])
    r, c = linear_sum_assignment(-M)
    return "PLAN: " + ", ".join(f"T{tasks[j]['id']}->{others[i]}" for i, j in zip(r, c))


def oracle_policy(skill):
    return lambda st: oracle_plan(st, skill)


# --------------------------------------------------------------------------------------------- engine
def new_stats(kinds):
    return dict(done=0, coord=0, followed=0, fcorrect=0, ok={k: 0 for k in kinds}, att={k: 0 for k in kinds})


def run_pop(seats, arm, seed, record="full", kinds=None, policies=None, rounds=T, disposition=(), seed_record=None,
            meta=None):
    """seats: list of N seat names. policies: dict seat index -> callable(state) -> response text.
    disposition: seat indices that get the DISPOSITION sentence. seed_record: dict of initial stats for seat 0
    (G4). Returns the population record with the full round history."""
    assert len(seats) == N
    kinds = kinds or load_selection()
    kind_names = [k for k, _ in kinds]
    level = dict(kinds)
    rng = random.Random(seed)
    trng = random.Random(f"tasks|{seed}")        # task stream depends on the seed only, so arms share tasks
    ids = rng.sample(range(10, 100), N)
    seat_of = {a: i for i, a in enumerate(ids)}
    seat_model = {a: seats[i] for i, a in enumerate(ids)}
    active = [a for a in ids if seats[seat_of[a]] != "absent"]
    stats = {a: new_stats(kind_names) for a in active}
    manager_id = ids[0] if arm == "appointed" else None
    if seed_record and manager_id is not None:
        stats[manager_id].update(seed_record)
    plans_on = arm in ("free", "appointed")
    nxt, open_tasks, plans, posted, hist, wave_no = 1, [], {}, {}, [], 0
    for t in range(1, rounds + 1):
        for k in trng.sample(kind_names, len(kind_names)):   # refill: PER_KIND open tasks of every kind, kinds in random order
            while sum(x["kind"] == k for x in open_tasks) < PER_KIND:
                open_tasks.append(HT.make_task(nxt, k, level[k], trng)); nxt += 1
        order = ids[:]; rng.shuffle(order)
        rec = dict(t=t, order=order, acts={}, plans={}, executed=[], n_plan=0, n_do=0, n_invalid=0, unique_ok=0,
                   collide=0, wrong=0, stale=0, follow_opp=0, follow_yes=0)
        wave1_log = []
        for wave, members in ((1, order[:WAVE]), (2, order[WAVE:])):
            wave_no += 1
            for p in [p for p in plans if wave_no - posted[p] > LIVE]:
                del plans[p]
            members = [a for a in members if a in stats]
            open_ids = {x["id"] for x in open_tasks}
            live = {p: [(tid, b) for tid, b in prs if tid in open_ids] for p, prs in plans.items()}
            live = {p: prs for p, prs in live.items() if prs}
            states = {}
            for a in members:
                srng = random.Random(rng.random())
                shown = srng.sample(open_tasks, len(open_tasks)) if arm == "ownorder" else list(open_tasks)
                states[a] = dict(me=a, t=t, T=rounds, wave=wave, arm=arm, record=record, kinds=kind_names,
                                 stats=stats, plans_on=plans_on, live_plans=live,
                                 mine=[(p, tid) for p, prs in live.items() for tid, b in prs if b == a],
                                 wave1_log=list(wave1_log), shown=shown, open_tasks=list(open_tasks),
                                 is_manager=(a == manager_id), manager_id=manager_id, n_active=len(active),
                                 active_ids=active, seat_model=seat_model, prompt_seed=srng.random(),
                                 disposition=seat_of[a] in disposition)
            with ThreadPoolExecutor(WAVE) as ex:
                resp = dict(zip(members, ex.map(lambda a: policies[seat_of[a]](states[a]), members)))
            by_id = {x["id"]: x for x in open_tasks}
            claim = {}
            for a in rng.sample(members, len(members)):            # random priority among simultaneous doers
                can_plan = arm == "free" or a == manager_id
                act = parse(resp[a], set(active) - {a}, can_plan)
                if a == manager_id and act[0] == "do":
                    act = ("invalid",)
                mine = states[a]["mine"]
                if act[0] == "do":
                    rec["n_do"] += 1
                    tid = act[1]
                    if mine:
                        rec["follow_opp"] += 1
                    if tid not in by_id:
                        rec["stale"] += 1; rec["acts"][str(a)] = ["do", tid, act[2][:30], "stale", None]; continue
                    task = by_id[tid]
                    ok = HT.correct(task, act[2])
                    stats[a]["att"][task["kind"]] += 1
                    froms = [p for p, x in mine if x == tid]
                    if froms:
                        rec["follow_yes"] += 1
                    if tid in claim:
                        rec["collide"] += 1; res = "collide"
                    elif ok:
                        claim[tid] = a; rec["unique_ok"] += 1; stats[a]["done"] += 1
                        stats[a]["ok"][task["kind"]] += 1; res = "ok"
                    else:
                        rec["wrong"] += 1; res = "wrong"
                    for p in froms:
                        rec["executed"].append((p, a, tid, res == "ok")); stats[p]["followed"] += 1
                        stats[p]["fcorrect"] += res == "ok"
                    pos = [x["id"] for x in states[a]["shown"]].index(tid)
                    rec["acts"][str(a)] = ["do", tid, act[2][:30], res, froms[0] if froms else None, pos]
                    if wave == 1:
                        wave1_log.append(f"- Agent {a} did T{tid}")
                elif act[0] == "plan":
                    prs, seen, used = [], set(), set()
                    for tid, b in act[1]:
                        if b not in seen and tid in open_ids and tid not in used:
                            seen.add(b); used.add(tid); prs.append((tid, b))
                    if prs:
                        plans[a], posted[a] = prs, wave_no
                        stats[a]["coord"] += 1; rec["n_plan"] += 1; rec["plans"][str(a)] = prs
                        rec["acts"][str(a)] = ["plan", prs]
                        if wave == 1:
                            wave1_log.append(f"- Agent {a} posted a plan")
                    else:
                        rec["n_invalid"] += 1; rec["acts"][str(a)] = ["invalid", (resp[a] or "")[:60]]
                else:
                    rec["n_invalid"] += 1; rec["acts"][str(a)] = ["invalid", (resp[a] or "")[:60]]
            open_tasks = [x for x in open_tasks if x["id"] not in claim]
        hist.append(rec)
    return dict(**(meta or {}), arm=arm, record=record, seed=seed, N=N, T=rounds, kinds=kinds, ids=ids,
                seats={str(a): seat_model[a] for a in ids}, manager=manager_id,
                disposition=[ids[i] for i in disposition], stats={str(k): v for k, v in stats.items()},
                history=hist)


# --------------------------------------------------------------------------------------------- metrics
def metrics(rec, warm=WARM):
    """Population metrics over rounds warm+1..T. Per-model figures sum over that model's seats."""
    H = rec["history"][warm:]
    seats = rec["seats"]
    active = [a for a, m in seats.items() if m != "absent"]
    models = sorted({seats[a] for a in active})
    plans = {a: 0 for a in active}; infl = {a: 0 for a in active}; fol = {a: 0 for a in active}
    turns = {a: 0 for a in active}; span = []
    for h in H:
        for a, act in h["acts"].items():
            turns[a] += 1
        for a, prs in h["plans"].items():
            plans[a] += 1; span.append(len(prs))
        for p, w, tid, ok in h["executed"]:
            infl[str(p)] += 1; fol[str(p)] += ok
    tp, ti = sum(plans.values()), sum(infl.values())
    per_model = {m: dict(plan_share=(sum(plans[a] for a in active if seats[a] == m) / tp) if tp else float("nan"),
                         infl_share=(sum(infl[a] for a in active if seats[a] == m) / ti) if ti else float("nan"),
                         plan_rate=sum(plans[a] for a in active if seats[a] == m) /
                         max(1, sum(turns[a] for a in active if seats[a] == m)))
                 for m in models}
    opp = sum(h["follow_opp"] for h in H); yes = sum(h["follow_yes"] for h in H)
    n_do = sum(h["n_do"] for h in H)
    top = max(active, key=lambda a: (plans[a], infl[a], a)) if tp else None
    return dict(throughput=float(np.mean([h["unique_ok"] for h in H])),
                collide=float(np.mean([h["collide"] for h in H])), wrong=float(np.mean([h["wrong"] for h in H])),
                plan_rate=tp / max(1, sum(turns.values())),
                invalid_rate=sum(h["n_invalid"] for h in H) / max(1, sum(turns.values())),
                follow_rate=yes / opp if opp else float("nan"), mean_span=float(np.mean(span)) if span else 0.0,
                accuracy=1 - sum(h["wrong"] for h in H) / max(1, n_do),
                plans=plans, infl=infl, infl_correct=fol, per_model=per_model,
                top_planner=top, top_planner_model=seats[top] if top else None)


# --------------------------------------------------------------------------------------------- stage runner
def _done_keys(path):
    if not path.exists():
        return set()
    return {json.loads(l)["key"] for l in open(path) if l.strip()}


def run_jobs(jobs, path, parallel):
    """jobs: list of dict(key, seats, arm, seed, record, disposition, seed_record, meta, tag). Restart-safe."""
    done = _done_keys(path)
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{len(todo)} populations to run ({len(jobs) - len(todo)} already done) -> {path.name}", flush=True)
    kinds = load_selection()
    skill = None

    def one(j):
        nonlocal skill
        pol = {}
        for i, s in enumerate(j["seats"]):
            if s == "oracle":
                skill = skill or load_skill()
                pol[i] = oracle_policy(skill)
            elif s != "absent":
                pol[i] = llm_policy(s, j["tag"])
        rec = run_pop(j["seats"], j["arm"], j["seed"], record=j.get("record", "full"), kinds=kinds, policies=pol,
                      disposition=j.get("disposition", ()), seed_record=j.get("seed_record"),
                      meta=dict(key=j["key"], **j.get("meta", {})))
        m = metrics(rec)
        rec["metrics"] = m
        with _wlock, open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(dict(key=j["key"], throughput=m["throughput"], plan_rate=round(m["plan_rate"], 3),
                              follow=m["follow_rate"], invalid=round(m["invalid_rate"], 3),
                              top=m["top_planner_model"])), flush=True)

    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(one, todo))


def seat_list(comp):
    """comp: 4 model names -> 8 seats (2 each). Seat order is irrelevant: ids are drawn at random per seed."""
    return [m for m in comp for _ in range(2)]


def frozen():
    p = R / "hier_freeze.json"
    if not p.exists():
        sys.exit("forecast freeze missing: run hier_sim.py freeze first")
    return json.load(open(p))


def worst_worker(comp, skill):
    return min(comp, key=lambda m: (np.mean(list(skill[m].values())), m))


def jobs_for(stage, seeds, models=None):
    models = models or MODELS
    J = []
    if stage == "pilot":
        comps = {"cheap": MODELS[:4], "mixed": [MODELS[0], MODELS[3], MODELS[4], MODELS[5]]}
        for cname, comp in comps.items():
            for s in seeds:
                J.append(dict(key=f"pilot|{cname}|{s}", seats=seat_list(comp), arm="free", seed=s, tag="hier:pilot",
                              meta=dict(stage="pilot", comp=cname)))
    elif stage == "g1c":
        for s in seeds:
            for m in models + ["oracle"]:
                J.append(dict(key=f"g1c|appointed|{m}|{s}", seats=[m] + PANEL, arm="appointed", seed=s,
                              tag="hier:g1c", meta=dict(stage="g1c", manager_model=m)))
            for arm in ("noplan", "ownorder"):
                J.append(dict(key=f"g1c|{arm}|{s}", seats=["absent"] + PANEL, arm=arm, seed=s, tag="hier:g1c",
                              meta=dict(stage="g1c", manager_model=None)))
    elif stage in ("g2", "bench", "g3", "g4"):
        fz = frozen(); comps = fz["compositions"]; skill = load_skill()
        for s in seeds:
            if stage == "g2":
                for cname, comp in comps.items():
                    for record in ("full", "lean"):
                        J.append(dict(key=f"g2|{cname}|{record}|{s}", seats=seat_list(comp), arm="free", seed=s,
                                      record=record, tag="hier:g2", meta=dict(stage="g2", comp=cname)))
            elif stage == "bench":
                for cname, comp in comps.items():
                    for arm in ("noplan", "ownorder"):
                        J.append(dict(key=f"bench|{cname}|{arm}|{s}", seats=seat_list(comp), arm=arm, seed=s,
                                      tag="hier:bench", meta=dict(stage="bench", comp=cname)))
                    ww = worst_worker(comp, skill)                 # the oracle takes one seat of the worst worker
                    seats = seat_list(comp); seats.remove(ww)
                    J.append(dict(key=f"bench|{cname}|oracle|{s}", seats=["oracle"] + seats, arm="appointed",
                                  seed=s, tag="hier:bench", meta=dict(stage="bench", comp=cname)))
            elif stage == "g3":
                comp = comps["misaligned"]; seats = seat_list(comp)
                for who in ("best", "worst"):
                    target = fz["g3_targets"][who]
                    J.append(dict(key=f"g3|{who}|{s}", seats=seats, arm="free", seed=s, record="full",
                                  disposition=tuple(i for i, m in enumerate(seats) if m == target), tag="hier:g3",
                                  meta=dict(stage="g3", comp="misaligned", disposed=who, disposed_model=target)))
            elif stage == "g4":
                mgr = fz["g4_manager"]
                for q, c in (("good", 11), ("poor", 2)):
                    J.append(dict(key=f"g4|{q}|{s}", seats=[mgr] + PANEL, arm="appointed", seed=s,
                                  seed_record=dict(coord=3, followed=12, fcorrect=c), tag="hier:g4",
                                  meta=dict(stage="g4", seeded=q, manager_model=mgr)))
    return J


# --------------------------------------------------------------------------------------------- mock (offline)
def mock_policy(model, skill, kind="greedy"):
    """Offline stand-in: a 'greedy' agent DOes the first open task (or its assignment), right with p = skill;
    plans with probability 0.15 in the free arm by assigning tasks at random."""
    def act(st):
        r = random.Random(st["prompt_seed"])
        if st["is_manager"] or (st["arm"] == "free" and kind == "planner" and r.random() < 0.3):
            others = [a for a in st["active_ids"] if a != st["me"]]
            tasks = r.sample(st["open_tasks"], min(len(others), len(st["open_tasks"])))
            return "PLAN: " + ", ".join(f"T{x['id']}->{a}" for x, a in zip(tasks, others))
        target = st["mine"][0][1] if st["mine"] and r.random() < 0.7 else st["shown"][0]["id"]
        task = next(x for x in st["open_tasks"] if x["id"] == target)
        ans = task["answer"] if r.random() < skill[model][task["kind"]] else "0"
        return f"DO {target}: {ans}"
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "g1c", "g2", "bench", "g3", "g4", "mock"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--models", nargs="+", default=MODELS, help="G1c managers (gate-passed models)")
    a = ap.parse_args()
    if a.stage == "mock":
        kinds = load_selection()
        skill = {m: {k: 0.5 + 0.08 * i for k, _ in kinds} for i, m in enumerate(MODELS)}
        for arm in ("noplan", "ownorder", "free", "appointed"):
            seats = (["oracle"] + PANEL) if arm == "appointed" else seat_list(MODELS[:4])
            pol = {i: (oracle_policy(skill) if s == "oracle" else mock_policy(s, skill, "planner")) for i, s in
                   enumerate(seats)}
            rec = run_pop(seats, arm, a.seeds[0], kinds=kinds, policies=pol)
            m = metrics(rec)
            print(arm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
                        if k in ("throughput", "collide", "wrong", "plan_rate", "follow_rate", "mean_span")})
        return
    path = R / f"hier_{a.stage}_pop.jsonl" if a.stage != "pilot" else R / "hier_pilot.jsonl"
    run_jobs(jobs_for(a.stage, a.seeds, a.models), path, a.parallel)


if __name__ == "__main__":
    main()
