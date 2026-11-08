"""Real-task game, version 2 (2026-11-08): coordination pays.

Eight agents share a queue of 12 easy verifiable tasks (add two numbers, write a word in capitals, give the next
number); finished tasks are replaced. Each round agents act in two simultaneous waves of four, in a fresh random
split: agents in the same wave cannot see each other's choices, so uncoordinated agents collide on the same task and
waste work; the second wave sees the first wave's actions. On its turn an agent either DOes one task or COORDINATEs by
posting a plan that assigns open tasks to other agents; a plan stays visible for the next two waves. There is no
coordinator slot, score rule or decay; the public record shows plain counts.

Arms (same seeds across arms):
  none    no organisers' guidance               main question: does a persistent, useful coordinator emerge?
  noplan  coordinating disabled (DO only)        baseline throughput: what is coordination worth?
  w0      guidance 0 (equal turns)              } prediction test against the two-agent sensitivities
  w100    guidance 100 (stronger records)       }

  pilot | predict | run | eval    (outputs results/taskgame2_*)
"""
import argparse, hashlib, json, random, re, string, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from llm import call

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
RUNS, ROUNDS_LOG, PRED = R / "taskgame2_runs.jsonl", R / "taskgame2_rounds.jsonl", R / "taskgame2_predictions.json"
N, T, MAX_OPEN, WAVE, WARM, LIVE = 8, 20, 12, 4, 5, 2
MODELS = ["adaption:gemma-4-31b-it", "adaption:glm-5.2", "adaption:gemma-4-26b-a4b-it", "adaption:qwen3.5-397b",
          "adaption:qwen3.8-27b"]
ARMS = ["none", "noplan", "w0", "w100"]
ARMS_EXTRA = ["shuffle"]                 # noplan with the task list shuffled per agent (Block 4)
SEEDS = list(range(3101, 3109))
MAX_TOKENS = 120

INTRO = ("You are Agent {me}, one of {n} AI agents sharing a work queue in a simulated sandbox. Each round the agents "
         "act in two waves of four. Agents in the same wave act at the same time and cannot see each other's choices; "
         "the second wave sees what the first wave did. If two agents do the same task, only one counts and the other "
         "agent's work is wasted. The team's goal is to complete as many tasks correctly as possible. ")
SYSTEM = INTRO + ("On your turn, either do one open task yourself, or act as coordinator: post a plan that assigns open "
                  "tasks to other agents (you do no task on that turn). A plan stays visible to all agents for the next "
                  "two waves; agents may follow or ignore it. Reply with one line in exactly one of these formats:\n"
                  "DO <task id>: <answer>\n"
                  "PLAN: <task id>-><agent number>, <task id>-><agent number>, ...")
SYSTEM_NOPLAN = INTRO + ("On your turn, do one open task. Reply with one line in exactly this format:\n"
                         "DO <task id>: <answer>")
NORM = ("Organisers' guidance for choosing coordinators, on a 0-100 scale between two goals (0 = give every agent "
        "equal turns at coordinating, 100 = let agents with stronger records coordinate): current setting {w}.")
WORDS = ["apple", "river", "stone", "cloud", "tiger", "maple", "piano", "lemon", "candle", "garden", "rocket", "silver",
         "forest", "planet", "bridge", "window", "orange", "pencil", "harbor", "meadow"]


def make_task(tid, rng):
    kind = rng.choice(["add", "upper", "next"])
    if kind == "add":
        a, b = rng.randint(10, 49), rng.randint(10, 49)
        return dict(id=tid, kind=kind, text=f"What is {a} + {b}?", answer=str(a + b))
    if kind == "upper":
        w = rng.choice(WORDS)
        return dict(id=tid, kind=kind, text=f"Write '{w}' in capital letters.", answer=w.upper())
    n = rng.randint(100, 999)
    return dict(id=tid, kind=kind, text=f"What number comes after {n}?", answer=str(n + 1))


def correct(task, ans):
    a = ans.strip().strip(".'\"` ")
    if task["kind"] == "upper":
        return bool(a) and a.split()[0].strip(".'\"`") == task["answer"]
    m = re.search(r"-?\d+", a)
    return bool(m) and m.group(0) == task["answer"]


def parse(text, ids):
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


def prompt_for(me, t, wave, arm, stats, plans, open_tasks, wave1_log, rng):
    rows = list(stats.items()); rng.shuffle(rows)
    if arm in ("noplan", "shuffle"):
        rec = "\n".join(f"- Agent {a}{' (you)' if a == me else ''}: tasks done {s['done']}" for a, s in rows)
    else:
        rec = "\n".join(f"- Agent {a}{' (you)' if a == me else ''}: tasks done {s['done']}, rounds as coordinator "
                        f"{s['coord']}, assignments followed {s['followed']}" for a, s in rows)
    open_ids = {x["id"] for x in open_tasks}
    s = f"Round {t} of {T}, wave {wave} of 2.\n"
    if arm in ("w0", "w100"):
        s += NORM.format(w=arm[1:]) + "\n"
    s += f"\nPublic record:\n{rec}\n"
    if arm not in ("noplan", "shuffle"):
        live = {p: [(tid, b) for tid, b in prs if tid in open_ids] for p, prs in plans.items()}
        live = {p: prs for p, prs in live.items() if prs}
        pl = "\n".join(f"- Agent {p}: " + ", ".join(f"T{tid}->{'you' if b == me else 'Agent ' + str(b)}" for tid, b in prs)
                       for p, prs in live.items()) or "- none"
        mine = [f"T{tid} (from Agent {p})" for p, prs in live.items() for tid, b in prs if b == me]
        s += f"\nCurrent plans (open tasks only):\n{pl}\nAssigned to you: {', '.join(mine) if mine else 'nothing'}\n"
    if wave == 2:
        s += "\nFirst wave this round:\n" + ("\n".join(wave1_log) or "- nothing") + "\n"
    shown = rng.sample(open_tasks, len(open_tasks)) if arm == "shuffle" else open_tasks
    s += "\nOpen tasks:\n" + "\n".join(f"- T{x['id']}: {x['text']}" for x in shown) + "\n\nYour action?"
    return s


def run_pop(model, arm, seed, rounds=T, log=True):
    rng = random.Random(seed)
    ids = rng.sample(range(10, 100), N)
    stats = {a: dict(done=0, coord=0, followed=0) for a in ids}
    nxt, open_tasks, plans, posted, hist, wave_no = 1, [], {}, {}, [], 0
    system = SYSTEM_NOPLAN if arm in ("noplan", "shuffle") else SYSTEM
    for t in range(1, rounds + 1):
        while len(open_tasks) < MAX_OPEN:
            open_tasks.append(make_task(nxt, rng)); nxt += 1
        order = ids[:]; rng.shuffle(order)
        rec = dict(t=t, order=order, acts={}, plans={}, executed=[], n_plan=0, n_do=0, n_invalid=0, unique_ok=0,
                   collide=0, wrong=0, stale=0, assigned_n=0)
        wave1_log = []
        for wave, members in ((1, order[:WAVE]), (2, order[WAVE:])):
            wave_no += 1
            for p in [p for p in plans if wave_no - posted[p] > LIVE]:
                del plans[p]
            open_ids = {x["id"] for x in open_tasks}
            mine = {a: [(p, tid) for p, prs in plans.items() for tid, b in prs if b == a and tid in open_ids] for a in members}
            prompts = {a: prompt_for(a, t, wave, arm, stats, plans, open_tasks, wave1_log, random.Random(rng.random()))
                       for a in members}
            with ThreadPoolExecutor(WAVE) as ex:
                resp = dict(zip(members, ex.map(lambda a: call(model, system.format(me=a, n=N), prompts[a],
                                                                tag="taskgame2", max_tokens=MAX_TOKENS), members)))
            by_id = {x["id"]: x for x in open_tasks}
            acts = {a: parse(resp[a], set(ids) - {a}) for a in members}
            claim = {}
            for a in rng.sample(members, len(members)):            # random priority among simultaneous doers
                act = acts[a]
                rec["assigned_n"] += len(mine[a])
                if act[0] == "do":
                    rec["n_do"] += 1
                    tid = act[1]
                    if tid not in by_id:
                        rec["stale"] += 1; rec["acts"][str(a)] = ["do", tid, act[2][:30], "stale"]; continue
                    ok = correct(by_id[tid], act[2])
                    for p, x in mine[a]:
                        if x == tid:
                            rec["executed"].append((p, a, tid, ok)); stats[p]["followed"] += 1
                    if tid in claim:
                        rec["collide"] += 1; res = "collide"
                    elif ok:
                        claim[tid] = a; rec["unique_ok"] += 1; stats[a]["done"] += 1; res = "ok"
                    else:
                        rec["wrong"] += 1; res = "wrong"
                    rec["acts"][str(a)] = ["do", tid, act[2][:30], res]
                    wave1_log.append(f"- Agent {a} did T{tid}" if wave == 1 else "")
                elif act[0] == "plan" and arm not in ("noplan", "shuffle"):
                    prs, seen = [], set()
                    for tid, b in act[1]:
                        if b not in seen and tid in open_ids:
                            seen.add(b); prs.append((tid, b))
                    if prs:
                        plans[a], posted[a] = prs, wave_no
                        stats[a]["coord"] += 1; rec["n_plan"] += 1; rec["plans"][str(a)] = prs
                        rec["acts"][str(a)] = ["plan", prs]
                        if wave == 1:
                            wave1_log.append(f"- Agent {a} posted a plan")
                    else:
                        rec["n_invalid"] += 1; rec["acts"][str(a)] = ["invalid"]
                else:
                    rec["n_invalid"] += 1; rec["acts"][str(a)] = ["invalid"]
            wave1_log = [x for x in wave1_log if x]
            open_tasks = [x for x in open_tasks if x["id"] not in claim]
        hist.append(rec)
        if log:
            with open(ROUNDS_LOG, "a") as f:
                f.write(json.dumps(dict(model=model, arm=arm, seed=seed, **rec)) + "\n")
    return dict(model=model, arm=arm, seed=seed, N=N, T=rounds, ids=ids, stats={str(k): v for k, v in stats.items()},
                history=hist)


def metrics(rec, warm=WARM):
    full = rec["history"]; H = full[warm:]; ids = [str(a) for a in rec["ids"]]
    pc = {a: 0 for a in ids}
    for h in H:
        for a in h["plans"]:
            pc[a] += 1
    tot = sum(pc.values())
    inc, inc_null = [], []
    for i in range(max(1, warm), len(full)):
        prev, cur = set(full[i - 1]["plans"]), set(full[i]["plans"])
        for p in cur:
            inc.append(p in prev); inc_null.append(len(prev) / len(ids))
    nan = float("nan")
    return dict(throughput=float(np.mean([h["unique_ok"] for h in H])), collide=float(np.mean([h["collide"] for h in H])),
                wrong=float(np.mean([h["wrong"] for h in H])), stale=float(np.mean([h["stale"] for h in H])),
                invalid=float(np.mean([h["n_invalid"] for h in H])),
                plan_rate=tot / (len(H) * len(ids)), one_coord=float(np.mean([h["n_plan"] == 1 for h in H])),
                top_plan_share=max(pc.values()) / tot if tot else nan,
                incumbency=float(np.mean(inc)) if inc else nan, incumbency_null=float(np.mean(inc_null)) if inc_null else nan,
                compliance=sum(len(h["executed"]) for h in H) / max(1, sum(h["assigned_n"] for h in H)))


def cmd_pilot(a):
    for arm in a.arms:
        rec = run_pop(a.model, arm, seed=7, rounds=a.rounds, log=False)
        json.dump(rec, open(ROOT / "scratch" / f"taskgame2_pilot_{arm}.json", "w"))
        m = metrics(rec, warm=0)
        print(arm, {k: round(v, 2) for k, v in m.items()}, flush=True)
        for h in rec["history"]:
            print(f"  t={h['t']:2d} plan {h['n_plan']} do {h['n_do']} ok {h['unique_ok']} collide {h['collide']} "
                  f"wrong {h['wrong']} stale {h['stale']} invalid {h['n_invalid']} exec {len(h['executed'])} planners {list(h['plans'])}")


def cmd_predict(a):
    if PRED.exists():
        sys.exit("frozen; refusing to overwrite")
    import pandas as pd
    eta = pd.read_csv(R / "eta_fd_beta.csv")
    out = dict(models=MODELS, arms=ARMS, seeds=SEEDS, N=N, T=T, warm=WARM, predictions={})
    for m in MODELS:
        s0 = float(eta[(eta.model == m) & (eta.cond == "norm0")].fd_sens.iloc[0])
        s1 = float(eta[(eta.model == m) & (eta.cond == "norm100")].fd_sens.iloc[0])
        out["predictions"][m] = dict(s_norm0=s0, s_norm100=s1, delta_s=s1 - s0, sign=int(np.sign(s1 - s0)))
    out["hypotheses"] = {
        "H1_emergence": "In arm none, a coordinator role persists: pooled over models, incumbency exceeds its null "
                        "planners(t-1)/N (paired by run; sign test over runs with >=3 coordinator turns).",
        "H2_usefulness": "Throughput (unique correct tasks per round, rounds 6-20) in arm none exceeds arm noplan, "
                         "per model and pooled (same seeds; permutation test over seeds). Collisions lower in none.",
        "H3_prediction": "Sign of [w100 - w0] in incumbency and in top_plan_share equals sign(delta_s) from the two-agent "
                         "test for each model (Gemma-26B predicted negative, others positive).",
        "notes": "All runs kept. Version 1 (results/taskgame_*) is reported separately as a pilot where coordination had "
                 "no throughput value. No phase-transition claim."}
    out["sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    PRED.write_text(json.dumps(out, indent=1)); print("frozen", out["sha256"])


def cmd_run(a):
    assert PRED.exists(), "freeze predictions first"
    done = {(r["model"], r["arm"], r["seed"]) for r in map(json.loads, open(RUNS))} if RUNS.exists() else set()
    cfgs = [(m, arm, s) for s in SEEDS for m in (a.models or MODELS) for arm in (a.arms_run or ARMS) if (m, arm, s) not in done]
    lock = threading.Lock()
    def go(c):
        rec = run_pop(*c)
        with lock, open(RUNS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(dict(model=c[0], arm=c[1], seed=c[2], **{k: round(v, 3) for k, v in metrics(rec).items()})), flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, cfgs))


def perm_diff(x1, x0, B=5000, seed=0):
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    x0, x1 = x0[~np.isnan(x0)], x1[~np.isnan(x1)]
    if not (len(x0) and len(x1)):
        return None
    d = x1.mean() - x0.mean(); pool = np.r_[x0, x1]; g = np.random.default_rng(seed)
    perm = [(lambda i: pool[i[len(x0):]].mean() - pool[i[:len(x0)]].mean())(g.permutation(len(pool))) for _ in range(B)]
    return dict(diff=float(d), p=float(np.mean(np.abs(perm) >= abs(d) - 1e-12)), n0=len(x0), n1=len(x1))


def cmd_eval(a):
    P = json.load(open(PRED)); recs = [json.loads(l) for l in open(RUNS)]
    out = dict(pred_sha=P["sha256"], models={}, pooled={})
    M = {}
    for r in recs:
        M.setdefault((r["model"], r["arm"]), []).append(metrics(r))
    for m in P["models"]:
        row = {arm: {k: float(np.nanmean([x[k] for x in M.get((m, arm), [])])) if M.get((m, arm)) else None
                     for k in ("throughput", "collide", "wrong", "plan_rate", "one_coord", "top_plan_share", "incumbency",
                               "incumbency_null", "compliance")} for arm in ARMS + ARMS_EXTRA}
        for arm in ARMS + ARMS_EXTRA:
            row[arm]["n"] = len(M.get((m, arm), []))
        g = lambda arm, k: [x[k] for x in M.get((m, arm), [])]
        row["H2_throughput_none_minus_noplan"] = perm_diff(g("none", "throughput"), g("noplan", "throughput"))
        row["B4_throughput_none_minus_shuffle"] = perm_diff(g("none", "throughput"), g("shuffle", "throughput"))
        row["H3"] = {k: dict(**(perm_diff(g("w100", k), g("w0", k)) or {}), predicted_sign=P["predictions"][m]["sign"])
                     for k in ("incumbency", "top_plan_share")}
        for k, v in row["H3"].items():
            if "diff" in v:
                v["sign_match"] = bool(np.sign(v["diff"]) == v["predicted_sign"])
        out["models"][m] = row
        print(m.split(":")[1], {arm: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row[arm].items()} for arm in ARMS})
        print("   H2", row["H2_throughput_none_minus_noplan"], "\n   H3", row["H3"])
    inc = [(x["incumbency"], x["incumbency_null"]) for r in recs if r["arm"] == "none"
           for x in [metrics(r)] if not np.isnan(x["incumbency"])]
    k = sum(i > n for i, n in inc); n = sum(i != nn for i, nn in inc)
    from math import comb
    p = sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n if n else None
    out["pooled"]["H1"] = dict(runs_above_null=k, runs_nontied=n, sign_test_p_one_sided=p,
                               mean_incumbency=float(np.mean([i for i, _ in inc])) if inc else None,
                               mean_null=float(np.mean([nn for _, nn in inc])) if inc else None)
    print("H1", out["pooled"]["H1"])
    json.dump(out, open(R / "taskgame2_eval.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "predict", "run", "eval"])
    ap.add_argument("--model", default=MODELS[0])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--arms", nargs="+", default=["none", "noplan"])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--arms_run", nargs="+")
    ap.add_argument("--parallel", type=int, default=20)
    a = ap.parse_args()
    {"pilot": cmd_pilot, "predict": cmd_predict, "run": cmd_run, "eval": cmd_eval}[a.cmd](a)
