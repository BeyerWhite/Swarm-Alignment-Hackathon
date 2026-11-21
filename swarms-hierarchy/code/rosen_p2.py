"""Rosen study, follow-ups and phase 2 (approved by the owner 2026-11-19): engines, batteries and run commands.

  run2   Rosen workshop game with options (rosen.run_pop): reasoning effort (R2r), m = 3 (R2b), group size N (R5),
         notebook variant (R6). Stage outputs results/rosen_<stage>_pop.jsonl.
  nbacc  R6 accuracy battery: x * mult answered with and without the notebook table in view (measured gain).
  bm     R3 Becker-Murphy: jobs are chains A -> B -> C; a step done by a different agent from the job's previous step
         costs a handoff fee c; J jobs arrive per round (market size).                -> results/rosen_r3_pop.jsonl
  yb     R4 Yang-Borland: no workbench; a correct type-j task earns 1 + lam * (tasks of type j this agent has done
         before). Stated horizon 8 or 24 rounds; every game is stopped after round 8.  -> results/rosen_r4_pop.jsonl
  mock   offline engine check with scripted policies (no API calls)

Every run command refuses to start without results/rosen_p2_prereg_freeze.json and stamps its sha256 in each record.
"""
import argparse, hashlib, json, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rosen
from rosen import TYPES, N, T, CAP, ARRIVE, kind_map, make_task, correct, parse, dol_index, plural, fmt_m

R = rosen.R
FREEZE = R / "rosen_p2_prereg_freeze.json"
FREEZE_RA3 = R / "rosen_p2_RA3_freeze.json"      # amendment R-A3 (codebook notebook tasks), stamped before R6 runs
_lock = threading.Lock()


def _append(path, rec):
    with _lock, open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def prereg_sha():
    if not FREEZE.exists():
        sys.exit("pre-registration not frozen (results/rosen_p2_prereg_freeze.json missing)")
    return json.load(open(FREEZE))["sha256"]


def ra3_sha():
    if not FREEZE_RA3.exists():
        sys.exit("amendment R-A3 not frozen (results/rosen_p2_RA3_freeze.json missing); R6 notebook runs refuse")
    return json.load(open(FREEZE_RA3))["sha256"]


def _llm(model, sysp, prompt, tag, reasoning="none"):
    from llm import call
    return call(model, sysp, prompt, tag=tag, max_tokens=rosen.MAX_TOKENS if reasoning == "none" else 2000,
                reasoning=reasoning)


# ============================================================================== R3: Becker-Murphy handoff chains

BM_F, BM_M, BM_JCAP = 3, 2.0, 8


def bm_system(me, c, J, F=BM_F, m=BM_M, rounds=T):
    fee = (f"When a step is done by a different agent from the one who did the previous step of the same job, the team "
           f"pays a handoff fee of {plural(1, 'point') if c == 1 else fmt_m(c) + ' points'}, taken from that step's points."
           if c > 0 else "There is no fee when different agents do the steps of one job.")
    return (f"You are Agent {me}, one of {N} AI agents working as a team in a simulated workshop. The team's goal is to "
            f"earn as many points as possible in {rounds} rounds. Each round every agent takes one turn, in a random "
            f"order; you see the open steps as they are when your turn comes.\n\n"
            f"Work comes as jobs. Every job has three steps that must be done in order: step A, then step B, then step "
            f"C; a step opens only when the previous step of its job is done. {plural(J, 'new job')} arrive at the "
            f"start of every round (at most {BM_JCAP} jobs can wait to start; extra arrivals are lost). A correct step "
            f"earns the team points; a wrong answer earns nothing.\n\n"
            f"On your turn you either do one open step, or spend the turn building a workbench for one step type (A, B "
            f"or C). A workbench for a step type is ready after {plural(F, 'turn')} of building it (the turns need not be "
            f"consecutive). Once your workbench for a step type is ready, each step of that type you complete correctly "
            f"earns {plural(int(m), 'point')}; a step done without a ready workbench for its type earns 1 point. A "
            f"workbench is used only by the agent who built it and lasts until the end of the game.\n\n{fee}\n\n"
            "Reply with one line in exactly one of these formats:\nDO <step id>: <answer>\nBUILD <type>")


def bm_render(view):
    s = f"Round {view['t']} of {view['T']}. Your turn ({view['pos']} of {N} this round).\n"
    s += f"Team points so far: {fmt_m(round(view['points'], 2))}.\n\nRecord:\n"
    for a in view["order_rows"]:
        st = view["stats"][a]
        s += (f"- Agent {a}{' (you)' if a == view['me'] else ''}: steps done A {st['done']['A']}, B {st['done']['B']}, "
              f"C {st['done']['C']}; {rosen.bench_str(st['bench'], view['F'])}\n")
    if view["pos"] > 1:
        s += "\nEarlier this round:\n" + ("\n".join(view["earlier"]) or "- nothing") + "\n"
    lines = []
    for x in view["open"]:
        tag = f"{x['type']}, job {x['job']}" + (f"; step {TYPES[TYPES.index(x['type']) - 1]} done by Agent {x['prev']}"
                                                if x["prev"] is not None else "")
        lines.append(f"- T{x['id']} ({tag}): {x['text']}")
    s += "\nOpen steps:\n" + ("\n".join(lines) or "- none")
    return s + "\n\nYour action?"


def run_bm(model, c, J, seed, decide=None, rounds=T, F=BM_F, m=BM_M):
    rng = random.Random(seed * 1000 + int(c * 10) * 10 + J)
    task_rng = random.Random(seed)
    kmap = kind_map(seed)
    ids = random.Random(seed + 77).sample(range(10, 100), N)
    stats = {a: dict(done={j: 0 for j in TYPES}, bench={j: 0 for j in TYPES}, points=0.0) for a in ids}
    queues = {j: [] for j in TYPES}
    nxt, job, points, fees, lost, jobs_done, hist = 1, 1, 0.0, 0.0, 0, 0, []

    def new_step(j, g, prev):
        nonlocal nxt
        x = make_task(nxt, j, kmap[j], task_rng); nxt += 1
        x.update(job=g, prev=prev); queues[j].append(x)

    for t in range(1, rounds + 1):
        for _ in range(J):
            if len(queues["A"]) < BM_JCAP:
                new_step("A", job, None)
            else:
                lost += 1
            job += 1
        order = ids[:]; rng.shuffle(order)
        rec = dict(t=t, order=order, acts={})
        earlier = []
        for pos, a in enumerate(order, 1):
            open_tasks = sorted([x for j in TYPES for x in queues[j]], key=lambda x: x["id"])
            rows = ids[:]; rng.shuffle(rows)
            view = dict(game="bm", me=a, F=F, m=m, c=c, J=J, t=t, T=rounds, pos=pos, points=points, stats=stats,
                        order_rows=rows, earlier=earlier, open=open_tasks)
            text = decide(view) if decide else _llm(model, bm_system(a, c, J, F, m, rounds), bm_render(view), "rosen_bm")
            act = parse(text)
            st = stats[a]
            if act[0] == "do":
                x = next((y for y in open_tasks if y["id"] == act[1]), None)
                if x is None:
                    rec["acts"][str(a)] = ["stale", act[1]]; earlier.append(f"- Agent {a} named a step that was not open")
                    continue
                ok = correct(x, act[2])
                tooled = st["bench"][x["type"]] >= F
                hand = x["prev"] is not None and x["prev"] != a
                gain = ((m if tooled else 1.0) - (c if hand else 0.0)) if ok else 0.0
                if ok:
                    queues[x["type"]] = [y for y in queues[x["type"]] if y["id"] != x["id"]]
                    st["done"][x["type"]] += 1; st["points"] += gain; points += gain; fees += c * hand
                    if x["type"] == "C":
                        jobs_done += 1
                    else:
                        new_step(TYPES[TYPES.index(x["type"]) + 1], x["job"], a)
                rec["acts"][str(a)] = ["do", x["type"], x["id"], bool(ok), bool(tooled), bool(hand), x["job"]]
                earlier.append(f"- Agent {a} did T{x['id']} ({x['type']}, job {x['job']})" + ("" if ok else ", wrong answer"))
            elif act[0] == "build":
                j = act[1]
                if st["bench"][j] >= F:
                    rec["acts"][str(a)] = ["redundant", j]; earlier.append(f"- Agent {a} built on a finished workbench {j}")
                else:
                    st["bench"][j] += 1; rec["acts"][str(a)] = ["build", j, st["bench"][j]]
                    earlier.append(f"- Agent {a} built workbench {j}" + (" (now ready)" if st["bench"][j] >= F else ""))
            else:
                rec["acts"][str(a)] = ["invalid", (text or "")[:60]]; earlier.append(f"- Agent {a} did nothing")
        rec["points"] = points
        hist.append(rec)
    return dict(game="bm", model=model, c=c, J=J, F=F, m=m, seed=seed, N=N, T=rounds, ids=ids, kinds=kmap,
                stats={str(k): v for k, v in stats.items()}, points=points, fees=fees, lost=lost, jobs_done=jobs_done,
                history=hist)


def bm_metrics(rec, warm=rosen.WARM):
    ids = [str(a) for a in rec["ids"]]
    counts = {a: {j: 0 for j in TYPES} for a in ids}
    okc = {a: {j: 0 for j in TYPES} for a in ids}
    hand, bc, inval, n = 0, 0, 0, 0
    for h in rec["history"]:
        for a, act in h["acts"].items():
            n += 1; inval += act[0] in ("invalid", "stale")
            if act[0] == "do":
                if h["t"] > warm:
                    counts[a][act[1]] += 1
                if act[3]:
                    okc[a][act[1]] += 1
                    if act[1] != "A":
                        bc += 1; hand += act[5]
    F = rec["F"]
    nready = [sum(rec["stats"][a]["bench"][j] >= F for j in TYPES) for a in ids]
    return dict(dol=dol_index([[counts[a][j] for j in TYPES] for a in ids]), points=rec["points"], fees=rec["fees"],
                handoff_share=hand / bc if bc else float("nan"), jobs_done=rec["jobs_done"],
                types_per_agent=float(np.mean([sum(okc[a][j] >= 2 for j in TYPES) for a in ids])),
                build_rate=sum(any(rec["stats"][a]["bench"][j] > 0 for j in TYPES) for a in ids) / len(ids),
                benches=float(np.mean(nready)), invalid_rate=inval / n)


def _bm_net(view, x):
    st = view["stats"][view["me"]]
    rate = view["m"] if st["bench"][x["type"]] >= view["F"] else 1.0
    return rate - (view["c"] if x["prev"] is not None and x["prev"] != view["me"] else 0.0)


def bm_generalist(view):
    """Benchmark policy: never build; continue own jobs first, then start new jobs (step A), then take over other
    agents' steps only if they still pay after the fee."""
    op = view["open"]
    mine = [x for x in op if x["prev"] == view["me"]]
    new = [x for x in op if x["prev"] is None]
    other = [x for x in op if x["prev"] not in (None, view["me"]) and _bm_net(view, x) > 0]
    pick = (mine or new or other)
    if not pick:
        return "BUILD A" if view["stats"][view["me"]]["bench"]["A"] < view["F"] else "nothing"
    x = pick[0]
    return f"DO T{x['id']}: {x['answer']}"


def bm_specialist(view):
    """Benchmark policy: one workbench per agent, step types split 3/3/2 in arrival order, then own type first."""
    F, me, st = view["F"], view["me"], view["stats"][view["me"]]
    mine = [j for j in TYPES if st["bench"][j] > 0]
    if not mine:
        held = {j: sum(1 for s in view["stats"].values() if s["bench"][j] > 0) for j in TYPES}
        quota = {"A": 3, "B": 3, "C": 2}
        j = max(TYPES, key=lambda j: (quota[j] - held[j], -TYPES.index(j)))
        return f"BUILD {j}"
    j = mine[0]
    if st["bench"][j] < F:
        return f"BUILD {j}"
    own = [x for x in view["open"] if x["type"] == j and _bm_net(view, x) > 0]
    if own:
        return f"DO T{own[0]['id']}: {own[0]['answer']}"
    rest = [x for x in view["open"] if _bm_net(view, x) > 0]
    return f"DO T{rest[0]['id']}: {rest[0]['answer']}" if rest else "nothing"


# ============================================================================= R4: Yang-Borland learning by doing

YB_PLAY = 8


def yb_rate(lam, k):
    return 1 + lam * k


def yb_system(me, lam, horizon):
    if lam > 0:
        rule = (f"Agents get better with practice: a correct task earns 1 point plus {fmt_m(lam)} points for every task "
                f"of the same type that agent has completed before (so an agent's fifth type-A task earns "
                f"{fmt_m(yb_rate(lam, 4))} points). Practice is per agent and per task type.")
    else:
        rule = "Every correct task earns 1 point."
    return (f"You are Agent {me}, one of {N} AI agents working as a team in a simulated workshop. The team's goal is to "
            f"earn as many points as possible in {horizon} rounds. Each round every agent takes one turn, in a random "
            f"order; you see the open tasks as they are when your turn comes.\n\nThere are three task types, A, B and "
            f"C. New tasks arrive at the start of every round: {ARRIVE['A']} of type A, {ARRIVE['B']} of type B and "
            f"{ARRIVE['C']} of type C (at most {CAP} tasks of one type can wait at once; extra arrivals are lost). A "
            f"wrong answer earns nothing. {rule}\n\nOn your turn you do one open task. Reply with one line in exactly "
            "this format:\nDO <task id>: <answer>")


def yb_render(view):
    lam = view["lam"]
    s = f"Round {view['t']} of {view['T']}. Your turn ({view['pos']} of {N} this round).\n"
    s += f"Team points so far: {fmt_m(round(view['points'], 2))}.\n\nRecord:\n"
    for a in view["order_rows"]:
        d = view["stats"][a]["done"]
        s += f"- Agent {a}{' (you)' if a == view['me'] else ''}: tasks done A {d['A']}, B {d['B']}, C {d['C']}"
        if lam > 0:
            s += "; next task earns " + ", ".join(f"{j} {fmt_m(yb_rate(lam, d[j]))}" for j in TYPES)
        s += "\n"
    if view["pos"] > 1:
        s += "\nEarlier this round:\n" + ("\n".join(view["earlier"]) or "- nothing") + "\n"
    s += "\nOpen tasks:\n" + ("\n".join(f"- T{x['id']} ({x['type']}): {x['text']}" for x in view["open"]) or "- none")
    return s + "\n\nYour action?"


def run_yb(model, lam, horizon, seed, decide=None, play=YB_PLAY):
    rng = random.Random(seed * 1000 + int(lam * 100) * 10 + horizon)
    task_rng = random.Random(seed)
    kmap = kind_map(seed)
    ids = random.Random(seed + 77).sample(range(10, 100), N)
    stats = {a: dict(done={j: 0 for j in TYPES}, points=0.0) for a in ids}
    queues = {j: [] for j in TYPES}
    nxt, points, lost, hist = 1, 0.0, 0, []
    for t in range(1, play + 1):
        for j in TYPES:
            for _ in range(ARRIVE[j]):
                x = make_task(nxt, j, kmap[j], task_rng); nxt += 1
                if len(queues[j]) < CAP:
                    queues[j].append(x)
                else:
                    lost += 1
        order = ids[:]; rng.shuffle(order)
        rec = dict(t=t, order=order, acts={}); earlier = []
        for pos, a in enumerate(order, 1):
            open_tasks = sorted([x for j in TYPES for x in queues[j]], key=lambda x: x["id"])
            rows = ids[:]; rng.shuffle(rows)
            view = dict(game="yb", me=a, lam=lam, t=t, T=horizon, pos=pos, points=points, stats=stats, order_rows=rows,
                        earlier=earlier, open=open_tasks)
            text = decide(view) if decide else _llm(model, yb_system(a, lam, horizon), yb_render(view), "rosen_yb")
            act = parse(text); st = stats[a]
            if act[0] == "do":
                x = next((y for y in open_tasks if y["id"] == act[1]), None)
                if x is None:
                    rec["acts"][str(a)] = ["stale", act[1]]; earlier.append(f"- Agent {a} named a task that was not open")
                    continue
                ok = correct(x, act[2])
                gain = yb_rate(lam, st["done"][x["type"]]) if ok else 0.0
                if ok:
                    queues[x["type"]] = [y for y in queues[x["type"]] if y["id"] != x["id"]]
                    st["done"][x["type"]] += 1; st["points"] += gain; points += gain
                rec["acts"][str(a)] = ["do", x["type"], x["id"], bool(ok), gain]
                earlier.append(f"- Agent {a} did T{x['id']} ({x['type']})" + ("" if ok else ", wrong answer"))
            else:
                rec["acts"][str(a)] = ["invalid", (text or "")[:60]]; earlier.append(f"- Agent {a} did nothing")
        rec["points"] = points; hist.append(rec)
    return dict(game="yb", model=model, lam=lam, horizon=horizon, seed=seed, N=N, played=play, ids=ids, kinds=kmap,
                stats={str(k): v for k, v in stats.items()}, points=points, lost=lost, history=hist)


def yb_metrics(rec, warm=1):
    ids = [str(a) for a in rec["ids"]]
    cnt = {a: {j: 0 for j in TYPES} for a in ids}
    late = {a: {j: 0 for j in TYPES} for a in ids}
    inval = n = 0
    for h in rec["history"]:
        for a, act in h["acts"].items():
            n += 1; inval += act[0] != "do"
            if act[0] == "do":
                if h["t"] > warm:
                    cnt[a][act[1]] += 1
                if h["t"] > 4:
                    late[a][act[1]] += 1
    tot = {a: sum(cnt[a].values()) for a in ids}
    return dict(dol=dol_index([[cnt[a][j] for j in TYPES] for a in ids]),
                dol_late=dol_index([[late[a][j] for j in TYPES] for a in ids]),
                top_share=float(np.mean([max(cnt[a].values()) / tot[a] for a in ids if tot[a]])),
                points=rec["points"], invalid_rate=inval / n)


def yb_specialist(view):
    """Benchmark: each agent keeps to the type of its first task (types split by the 4/2/2 quota in move order)."""
    d = view["stats"][view["me"]]["done"]
    if sum(d.values()):
        j = max(TYPES, key=lambda j: d[j])
    else:
        held = {j: sum(1 for s in view["stats"].values() if sum(s["done"].values()) and max(TYPES, key=lambda k: s["done"][k]) == j)
                for j in TYPES}
        j = max(TYPES, key=lambda j: (ARRIVE[j] - held[j], ARRIVE[j]))
    op = [x for x in view["open"] if x["type"] == j] or view["open"]
    if not op:
        return "nothing"
    return f"DO T{op[0]['id']}: {op[0]['answer']}"


def yb_oldest(view):
    op = view["open"]
    return f"DO T{op[0]['id']}: {op[0]['answer']}" if op else "nothing"


# ========================================================================================== R6: accuracy battery

NBACC_REPS = 60


def cmd_nbacc(a):
    from llm import call
    out = R / ("rosen_r6_acc.jsonl" if rosen.NB_TASKS == "mul" else f"rosen_r6_acc_{rosen.NB_TASKS}.jsonl")
    sha = prereg_sha()
    extra = {} if rosen.NB_TASKS == "mul" else dict(amend_sha=ra3_sha(), nb_tasks=rosen.NB_TASKS)
    jobs = [(mod, cond, r) for mod in a.models for cond in ("table", "none") for r in range(NBACC_REPS)]
    print(f"{len(jobs)} calls", flush=True)

    def go(j):
        mod, cond, r = j
        rng = random.Random(f"nbacc|{mod}|{cond}|{r}")
        seed = rng.randrange(10 ** 6); mults = rosen.nb_map(seed)
        ids = rng.sample(range(10, 100), N); me = ids[0]
        typ = rng.choice(TYPES)
        stats = {x: dict(done={k: rng.randint(0, 3) for k in TYPES}, bench={k: 0 for k in TYPES}, points=0.0) for x in ids}
        if cond == "table":
            stats[me]["bench"][typ] = 3
        task = make_task(33, typ, rosen.nb_kind(mults), rng, mults[typ])
        view = dict(arm="team", me=me, F=3, m=1.0, t=5, T=T, n=N, pos=1, points=24, stats=stats, order_rows=ids,
                    earlier=[], open=[task], variant="notebook", mults=mults)
        what = "number" if task["kind"] == "mul" else "code word"
        prompt = rosen.render(view).replace("\n\nYour action?", f"\n\nThis is not your turn yet. Answer with the {what} "
                                                                f"only.\nQuestion: {task['text']}")
        text = call(mod, rosen.system_prompt("team", me, 3, 1.0, N, T, "notebook", mults), prompt, tag="rosen_nbacc",
                    max_tokens=16)
        if task["kind"] == "mul":
            mm = re.search(r"-?\d+", (text or "").replace(",", ""))
            ok = bool(mm) and mm.group(0) == task["answer"]
        else:
            ok = correct(task, text or "")
        _append(out, dict(model=mod, cond=cond, rep=r, type=typ, answer=task["answer"], response=(text or "")[:60],
                          correct=ok, prereg_sha=sha, **extra))

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(go, jobs))
    rows = [json.loads(l) for l in open(out)]
    for mod in a.models:
        p = {c: np.mean([r["correct"] for r in rows if r["model"] == mod and r["cond"] == c]) for c in ("table", "none")}
        print(mod, {k: round(float(v), 3) for k, v in p.items()})


# =============================================================================================== run commands

def _done(out, keyf):
    return {keyf(r) for r in map(json.loads, open(out))} if out.exists() else set()


def cmd_run2(a):
    sha = prereg_sha()
    amend = ra3_sha() if a.variant == "notebook" and rosen.NB_TASKS != "mul" else None
    out = R / f"rosen_{a.stage.lower()}_pop.jsonl"
    key = lambda r: (r["model"], r["arm"], r["F"], r["m"], r["seed"], r.get("N", 8), r.get("variant", "std"),
                     r.get("reasoning", "none"))
    done = _done(out, key)
    n = 1
    cfgs = []
    for s in a.seeds:
        for (arm, m, F) in rosen.parse_plan(a.plan):
            for mod in a.models:
                k = (mod, arm, F, m, s, 1 if arm == "autarky" else a.n_agents, a.variant, a.reasoning)
                if k not in done:
                    cfgs.append((mod, arm, F, m, s))
    print(f"{len(cfgs)} populations to run ({len(done)} already in {out.name})", flush=True)

    def go(c):
        rec = rosen.run_pop(*c, reasoning=a.reasoning, n_agents=a.n_agents, variant=a.variant)
        rec.update(stage=a.stage, prereg_sha=sha)
        if amend:
            rec["amend_sha"] = amend
        if a.reasoning != "none":
            rec["reasoning"] = a.reasoning
        _append(out, rec); print(rosen._show(rec), flush=True)

    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def cmd_bm(a):
    sha = prereg_sha()
    out = R / "rosen_r3_pop.jsonl"
    done = _done(out, lambda r: (r["model"], r["c"], r["J"], r["seed"]))
    cfgs = [(mod, c, J, s) for s in a.seeds for c in a.c for J in a.J for mod in a.models if (mod, c, J, s) not in done]
    print(f"{len(cfgs)} populations to run", flush=True)

    def go(cf):
        rec = run_bm(*cf); rec.update(stage="R3", prereg_sha=sha); _append(out, rec)
        print(json.dumps(dict(model=cf[0], c=cf[1], J=cf[2], seed=cf[3], **{k: round(v, 3) for k, v in bm_metrics(rec).items()})),
              flush=True)

    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def cmd_yb(a):
    sha = prereg_sha()
    out = R / "rosen_r4_pop.jsonl"
    done = _done(out, lambda r: (r["model"], r["lam"], r["horizon"], r["seed"]))
    cfgs = [(mod, l, h, s) for s in a.seeds for l in a.lam for h in a.horizon for mod in a.models if (mod, l, h, s) not in done]
    print(f"{len(cfgs)} populations to run", flush=True)

    def go(cf):
        rec = run_yb(*cf); rec.update(stage="R4", prereg_sha=sha); _append(out, rec)
        print(json.dumps(dict(model=cf[0], lam=cf[1], horizon=cf[2], seed=cf[3], **{k: round(v, 3) for k, v in yb_metrics(rec).items()})),
              flush=True)

    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def cmd_mock(a):
    seeds = range(1, 5)
    print("R3 benchmark (mean points over 4 seeds): generalist vs specialist")
    for J in (2, 4):
        for c in (0, 0.5, 1, 2):
            g = [run_bm("mock", c, J, s, decide=bm_generalist) for s in seeds]
            sp = [run_bm("mock", c, J, s, decide=bm_specialist) for s in seeds]
            mg = [bm_metrics(r) for r in g]; ms = [bm_metrics(r) for r in sp]
            print(f"  J={J} c={c}: generalist {np.mean([r['points'] for r in g]):6.1f} (dol {np.mean([m['dol'] for m in mg]):.2f}, "
                  f"handoff {np.nanmean([m['handoff_share'] for m in mg]):.2f})  specialist {np.mean([r['points'] for r in sp]):6.1f} "
                  f"(dol {np.mean([m['dol'] for m in ms]):.2f}, handoff {np.nanmean([m['handoff_share'] for m in ms]):.2f})")
    print("R4 benchmark (8 rounds): oldest-first vs keep-own-type")
    for lam in (0, 0.25, 0.5):
        o = [run_yb("mock", lam, 24, s, decide=yb_oldest) for s in seeds]
        sp = [run_yb("mock", lam, 24, s, decide=yb_specialist) for s in seeds]
        print(f"  lam={lam}: oldest {np.mean([r['points'] for r in o]):6.1f} (dol {np.mean([yb_metrics(r)['dol'] for r in o]):.2f})"
              f"  specialist {np.mean([r['points'] for r in sp]):6.1f} (dol {np.mean([yb_metrics(r)['dol'] for r in sp]):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run2", "nbacc", "bm", "yb", "mock"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--stage", default="R2r")
    ap.add_argument("--plan", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(41001, 41009)))
    ap.add_argument("--n-agents", dest="n_agents", type=int, default=N)
    ap.add_argument("--variant", default="std", choices=["std", "notebook"])
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--c", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--J", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--lam", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    ap.add_argument("--horizon", nargs="+", type=int, default=[8, 24])
    ap.add_argument("--parallel", type=int, default=10)
    a = ap.parse_args()
    dict(run2=cmd_run2, nbacc=cmd_nbacc, bm=cmd_bm, yb=cmd_yb, mock=cmd_mock)[a.cmd](a)


if __name__ == "__main__":
    main()
