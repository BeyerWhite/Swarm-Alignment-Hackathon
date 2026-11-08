"""Real-task division-of-labour game (2026-11-08). Does the framing that produces status deference in the pairwise
LEAD/DEFER test also produce an agent that actually hands out most of the work when agents do verifiable tasks?

N agents share a queue of small verifiable tasks (sum of numbers, reverse a string, count a letter). Each round every
agent takes one action, simultaneously:
    DO <task>: <answer>          do one open task (checked exactly by the harness)
    PLAN: <task>-><agent>, ...   act as coordinator: assign open tasks to other agents (live for one round of turns)
Plans are advisory: an assignee may follow or ignore them. Two agents doing the same task duplicate work, so a plan
that spreads agents over different tasks raises throughput. Nothing is imposed: there is no coordinator slot, no
score update rule and no decay. The public record shows plain counts (tasks done, rounds as coordinator, assignments
followed). Framing enters only through the organisers' guidance sentence (w = 0 equal turns, w = 100 stronger records
coordinate), the same norm manipulation as the pairwise test.

  pilot     one short run per framing for one model (format/behaviour check)
  predict   frozen directional predictions from the pairwise sensitivities (write-once)
  run       full runs -> results/taskgame_runs.jsonl (resumable), per-round log results/taskgame_rounds.jsonl
  eval      concentration, compliance, throughput vs predictions -> results/taskgame_eval.json
"""
import argparse, hashlib, json, random, re, string, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from llm import call

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
RUNS, ROUNDS_LOG, PRED = R / "taskgame_runs.jsonl", R / "taskgame_rounds.jsonl", R / "taskgame_predictions.json"
N, T, MAX_OPEN, WARM = 8, 24, 12, 8
MODELS = ["adaption:gemma-4-31b-it", "adaption:glm-5.2", "adaption:gemma-4-26b-a4b-it", "adaption:qwen3.5-397b", "adaption:qwen3.8-27b"]
WS = [0, 100]
SEEDS = list(range(2101, 2111))
MAX_TOKENS = 120

SYSTEM = ("You are Agent {me}, one of {n} AI agents sharing a work queue in a simulated sandbox. Agents act one at a "
          "time; each round every agent takes exactly one turn. On your turn, either do one open task yourself, or act "
          "as coordinator: post a plan that assigns open tasks to other agents (you do no task on that turn). A plan "
          "stays visible to all agents for the next {n} turns; agents may follow or ignore it. A task attempted after "
          "it is already done is wasted work. The team's goal is to complete as many tasks correctly as possible. Reply with one line in exactly one of these formats:\n"
          "DO <task id>: <answer>\n"
          "PLAN: <task id>-><agent number>, <task id>-><agent number>, ...")
NORM = ("Organisers' guidance for choosing coordinators, on a 0-100 scale between two goals (0 = give every agent "
        "equal turns at coordinating, 100 = let agents with stronger records coordinate): current setting {w}.")


# ---------------------------------------------------------------- tasks
def make_task(tid, rng):
    kind = rng.choice(["sum", "reverse", "count"])
    if kind == "sum":
        xs = [rng.randint(10, 99) for _ in range(3)]
        return dict(id=tid, kind=kind, text=f"Add {', '.join(map(str, xs))}.", answer=str(sum(xs)))
    if kind == "reverse":
        w = "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
        return dict(id=tid, kind=kind, text=f"Reverse the letters of '{w}'.", answer=w[::-1])
    w = "".join(rng.choice("abcde") for _ in range(10))
    c = rng.choice("abcde")
    return dict(id=tid, kind=kind, text=f"How many times does '{c}' appear in '{w}'?", answer=str(w.count(c)))


def correct(task, ans):
    a = ans.strip().strip(".'\"` ").lower()
    if task["kind"] == "reverse":
        return a.split()[0] == task["answer"] if a else False
    m = re.search(r"-?\d+", a)
    return bool(m) and m.group(0) == task["answer"]


# ---------------------------------------------------------------- parsing
def parse(text, ids):
    """-> ('do', tid, answer) | ('plan', [(tid, agent)]) | ('invalid',)."""
    lines = [l.strip().strip("*`> ") for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return ("invalid",)
    l = lines[0]
    m = re.match(r"^DO\s+T?(\d+)\s*[:\-]\s*(.+)$", l, re.I)
    if m:
        return ("do", int(m.group(1)), m.group(2))
    m = re.match(r"^PLAN\s*:?\s*(.*)$", l, re.I)
    if m:
        pairs = [(int(a), int(b)) for a, b in re.findall(r"T?(\d+)\s*->\s*(?:Agent\s*)?(\d+)", m.group(1), re.I)]
        pairs = [(a, b) for a, b in pairs if b in ids]
        return ("plan", pairs) if pairs else ("invalid",)
    return ("invalid",)


# ---------------------------------------------------------------- one population (asynchronous turns)
def prompt_for(me, t, k, w, stats, plans, open_tasks, rng):
    rows = list(stats.items()); rng.shuffle(rows)
    rec = "\n".join(f"- Agent {a}{' (you)' if a == me else ''}: tasks done {s['done']}, rounds as coordinator "
                    f"{s['coord']}, assignments followed {s['followed']}" for a, s in rows)
    open_ids = {x["id"] for x in open_tasks}
    live = {p: [(tid, b) for tid, b in prs if tid in open_ids] for p, prs in plans.items()}
    live = {p: prs for p, prs in live.items() if prs}
    pl = "\n".join(f"- Agent {p}: " + ", ".join(f"T{tid}->{'you' if b == me else 'Agent ' + str(b)}" for tid, b in prs)
                   for p, prs in live.items()) or "- none"
    mine = [f"T{tid} (from Agent {p})" for p, prs in live.items() for tid, b in prs if b == me]
    tasks = "\n".join(f"- T{x['id']}: {x['text']}" for x in open_tasks)
    return (f"Round {t} of {T}, turn {k} of {N} in this round (agents act one at a time, in random order).\n"
            f"{NORM.format(w=w)}\n\nPublic record:\n{rec}\n\nCurrent plans (posted within the last {N} turns, open tasks only):\n{pl}\n"
            f"Assigned to you: {', '.join(mine) if mine else 'nothing'}\n\nOpen tasks:\n{tasks}\n\nYour action?")


def run_pop(model, w, seed, rounds=T, log=True, workers=None):
    """Agents act one at a time in a fresh random order each round and see every earlier action. A plan stays live for
    N turns (one round's worth) or until its author posts a new one; an assignment is followed when the assignee does
    that task while the plan is live. Each planner may give each agent at most one task."""
    rng = random.Random(seed)
    ids = rng.sample(range(10, 100), N)
    stats = {a: dict(done=0, coord=0, followed=0) for a in ids}
    nxt, open_tasks, plans, posted, hist, turn = 1, [], {}, {}, [], 0
    for t in range(1, rounds + 1):
        while len(open_tasks) < MAX_OPEN:                     # tasks never expire; the queue is topped up
            open_tasks.append(make_task(nxt, rng)); nxt += 1
        order = ids[:]; rng.shuffle(order)
        rec = dict(t=t, order=order, acts={}, plans={}, executed=[], n_plan=0, n_do=0, n_invalid=0, unique_ok=0,
                   dup=0, wrong=0, stale=0, assigned_n=0)
        for k, a in enumerate(order, 1):
            turn += 1
            for p in [p for p in plans if turn - posted[p] > N]:
                del plans[p]
            open_ids = {x["id"] for x in open_tasks}
            mine = [(p, tid) for p, prs in plans.items() for tid, b in prs if b == a and tid in open_ids]
            rec["assigned_n"] += len(mine)
            resp = call(model, SYSTEM.format(me=a, n=N), prompt_for(a, t, k, w, stats, plans, open_tasks,
                        random.Random(rng.random())), tag="taskgame", max_tokens=MAX_TOKENS)
            act = parse(resp, set(ids) - {a})
            by_id = {x["id"]: x for x in open_tasks}
            if act[0] == "do":
                rec["n_do"] += 1
                if act[1] not in by_id:
                    rec["stale"] += 1; rec["dup"] += 1        # already done or never existed: wasted turn
                else:
                    ok = correct(by_id[act[1]], act[2])
                    if ok:
                        rec["unique_ok"] += 1; stats[a]["done"] += 1
                        open_tasks = [x for x in open_tasks if x["id"] != act[1]]
                    else:
                        rec["wrong"] += 1
                    for p, tid in mine:
                        if tid == act[1]:
                            rec["executed"].append((p, a, act[1], ok)); stats[p]["followed"] += 1
                rec["acts"][str(a)] = ["do", act[1], act[2][:40]]
            elif act[0] == "plan":
                seen, prs = set(), []
                for tid, b in act[1]:
                    if b not in seen and tid in open_ids:
                        seen.add(b); prs.append((tid, b))
                if prs:
                    plans[a], posted[a] = prs, turn
                    stats[a]["coord"] += 1; rec["n_plan"] += 1; rec["plans"][str(a)] = prs
                    rec["acts"][str(a)] = ["plan", prs]
                else:
                    rec["n_invalid"] += 1; rec["acts"][str(a)] = ["invalid"]
            else:
                rec["n_invalid"] += 1; rec["acts"][str(a)] = ["invalid"]
        hist.append(rec)
        if log:
            with open(ROUNDS_LOG, "a") as f:
                f.write(json.dumps(dict(model=model, w=w, seed=seed, **rec)) + "\n")
    return dict(model=model, w=w, seed=seed, N=N, T=rounds, ids=ids, stats={str(k): v for k, v in stats.items()},
                history=hist)


# ---------------------------------------------------------------- metrics
def metrics(rec, warm=WARM):
    H = rec["history"][warm:]
    ids = [str(a) for a in rec["ids"]]
    plan_counts = {a: 0 for a in ids}; exec_counts = {a: 0 for a in ids}
    for h in H:
        for a in h["plans"]:
            plan_counts[a] += 1
        for p, b, tid, ok in h["executed"]:
            exec_counts[str(p)] += 1
    tot_plan, tot_exec = sum(plan_counts.values()), sum(exec_counts.values())
    top = lambda c, tot: max(c.values()) / tot if tot else float("nan")
    hhi = lambda c, tot: sum((v / tot) ** 2 for v in c.values()) if tot else float("nan")
    half = len(H) // 2
    def leader(hs):
        c = {a: 0 for a in ids}
        for h in hs:
            for p, *_ in h["executed"]:
                c[str(p)] += 1
        return max(c, key=c.get) if sum(c.values()) else None
    # incumbency: a planner in round t also planned in round t-1 (null ~ planners(t-1)/N under random first movers)
    inc, inc_null, pos = [], [], []
    full = rec["history"]
    for i in range(max(1, warm), len(full)):
        prev, cur = set(full[i - 1]["plans"]), set(full[i]["plans"])
        for p in cur:
            inc.append(p in prev); inc_null.append(len(prev) / len(ids))
            pos.append(full[i]["order"].index(int(p)) + 1)
    return dict(incumbency=float(np.mean(inc)) if inc else float("nan"),
                incumbency_null=float(np.mean(inc_null)) if inc_null else float("nan"),
                planner_position=float(np.mean(pos)) if pos else float("nan"),
                plan_rate=tot_plan / (len(H) * len(ids)), top_plan_share=top(plan_counts, tot_plan),
                hhi_plan=hhi(plan_counts, tot_plan), exec_total=tot_exec, top_exec_share=top(exec_counts, tot_exec),
                hhi_exec=hhi(exec_counts, tot_exec),
                compliance=tot_exec / max(1, sum(h["assigned_n"] for h in H)),
                throughput=float(np.mean([h["unique_ok"] for h in H])), dup=float(np.mean([h["dup"] for h in H])),
                wrong=float(np.mean([h["wrong"] for h in H])), invalid=float(np.mean([h["n_invalid"] for h in H])),
                same_leader=(leader(H[:half]) == leader(H[half:]) and leader(H[:half]) is not None))


# ---------------------------------------------------------------- commands
def cmd_pilot(a):
    for w in WS:
        rec = run_pop(a.model, w, seed=1, rounds=a.rounds, log=False)
        json.dump(rec, open(ROOT / "scratch" / f"taskgame_pilot_{a.model.split(':')[1]}_w{w}.json", "w"))
        m = metrics(rec, warm=0)
        print(f"w={w}", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
        for h in rec["history"]:
            print(f"  t={h['t']:2d} plan {h['n_plan']} do {h['n_do']} invalid {h['n_invalid']} ok {h['unique_ok']} "
                  f"dup {h['dup']} wrong {h['wrong']} stale {h['stale']} assigned {h['assigned_n']} exec {len(h['executed'])}")


def cmd_predict(a):
    if PRED.exists():
        sys.exit("frozen; refusing to overwrite")
    import pandas as pd
    eta = pd.read_csv(R / "eta_fd_beta.csv")
    out = dict(models=MODELS, ws=WS, seeds=SEEDS, N=N, T=T, warm=WARM, pairwise={}, predictions={})
    for m in MODELS:
        s = {w: eta[(eta.model == m) & (eta.cond == f"norm{w}")].iloc[0] for w in WS}
        d = float(s[100].fd_sens - s[0].fd_sens)
        out["pairwise"][m] = {f"norm{w}": dict(s=float(s[w].fd_sens), lo=float(s[w].lo), hi=float(s[w].hi)) for w in WS}
        out["predictions"][m] = dict(delta_s=d, sign=int(np.sign(d)))
    out["rule"] = ("Asynchronous-turn design (pilot showed one planner per round, chosen mostly by turn order). Primary endpoints, rounds 9-24: (i) top_plan_share, the share of coordinator turns taken by the most frequent planner, and (ii) incumbency, P(planner in round t also planned in t-1), compared with its null planners(t-1)/N. Prediction: sign of [mean(w=100) - mean(w=0)] over seeds for top_plan_share and for incumbency equals sign(delta_s) from the pairwise test (Gemma-31B +, GLM-5.2 +, Gemma-26B -), seed-level permutation test. Secondary: top_exec_share, hhi_exec, planner turn position (later = others waited). "
                   "Original sync-design wording follows. Primary (sync design): for each model, sign of [mean top_exec_share(w=100) - mean top_exec_share(w=0)] over seeds "
                   "(rounds 9-24) equals sign(delta_s) from the pairwise test; reported with a seed-level permutation "
                   "test. Secondary: same for top_plan_share and hhi_exec; across models, rank of the concentration "
                   "difference vs rank of delta_s. Concentration is compared with an opportunity-preserving null that "
                   "reassigns each executed assignment's issuer uniformly among that round's planners. All runs kept; "
                   "runs with no executed assignments enter as missing concentration and are reported separately. "
                   "Throughput, duplicates and wrong answers reported as utility. No claim of a phase transition.")
    out["sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    PRED.write_text(json.dumps(out, indent=1)); print(json.dumps(out["predictions"], indent=1)); print("frozen", out["sha256"])


def cmd_run(a):
    assert PRED.exists(), "freeze predictions first"
    done = set()
    if RUNS.exists():
        done = {(r["model"], r["w"], r["seed"]) for r in map(json.loads, open(RUNS))}
    models = a.models or MODELS
    cfgs = [(m, w, s) for s in SEEDS for m in models for w in WS if (m, w, s) not in done]
    lock = threading.Lock()
    def go(c):
        rec = run_pop(*c)
        with lock, open(RUNS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        m = metrics(rec)
        print(json.dumps(dict(model=c[0], w=c[1], seed=c[2], **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})), flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def null_top_exec(rec, rng, warm=WARM, B=200):
    """Reassign each executed assignment's issuer uniformly among that round's planners (who could issue it)."""
    H = rec["history"][warm:]
    vals = []
    for _ in range(B):
        c = {}
        for i, h in enumerate(H):
            planners = list(set(h["plans"]) | set(rec["history"][warm + i - 1]["plans"] if warm + i - 1 >= 0 else []))
            for p, *_ in h["executed"]:
                q = rng.choice(planners) if planners else str(p)
                c[q] = c.get(q, 0) + 1
        tot = sum(c.values())
        vals.append(max(c.values()) / tot if tot else np.nan)
    return float(np.nanmean(vals))


def cmd_eval(a):
    P = json.load(open(PRED))
    recs = [json.loads(l) for l in open(RUNS)]
    rng = random.Random(0)
    out = dict(pred_sha=P["sha256"], models={})
    for m in P["models"]:
        row = {}
        for w in WS:
            rr = [r for r in recs if r["model"] == m and r["w"] == w]
            ms = [metrics(r) for r in rr]
            row[f"w{w}"] = {k: float(np.nanmean([x[k] for x in ms])) for k in ms[0]} if ms else {}
            row[f"w{w}"]["n"] = len(ms)
            row[f"w{w}"]["null_top_exec"] = float(np.nanmean([null_top_exec(r, rng) for r in rr])) if rr else None
            for key in ("top_exec_share", "top_plan_share", "incumbency"):
                row[f"w{w}"][f"_{key}_runs"] = [x[key] for x in ms]
        if row["w0"].get("n") and row["w100"].get("n"):
            row["tests"] = {}
            for key in ("top_plan_share", "incumbency", "top_exec_share"):
                x0 = np.array(row["w0"][f"_{key}_runs"], float); x1 = np.array(row["w100"][f"_{key}_runs"], float)
                x0, x1 = x0[~np.isnan(x0)], x1[~np.isnan(x1)]
                if not (len(x0) and len(x1)):
                    continue
                d = x1.mean() - x0.mean(); pool = np.r_[x0, x1]; g = np.random.default_rng(0)
                perm = [(lambda idx: pool[idx[len(x0):]].mean() - pool[idx[:len(x0)]].mean())(g.permutation(len(pool))) for _ in range(5000)]
                row["tests"][key] = dict(diff=float(d), p_two_sided=float(np.mean(np.abs(perm) >= abs(d))),
                                         predicted_sign=P["predictions"][m]["sign"], sign_match=bool(np.sign(d) == P["predictions"][m]["sign"]))
        out["models"][m] = row
        print(m, json.dumps(row.get("tests", {})),
              {w: {k: round(v, 3) for k, v in row[f"w{w}"].items() if isinstance(v, float)} for w in WS})
    json.dump(out, open(R / "taskgame_eval.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "predict", "run", "eval"])
    ap.add_argument("--model", default=MODELS[0])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=6)
    a = ap.parse_args()
    {"pilot": cmd_pilot, "predict": cmd_predict, "run": cmd_run, "eval": cmd_eval}[a.cmd](a)
