"""v2 pairwise calibration of the LLM decision kernel p(LEAD | h_self, h_other).

Matches the population protocol: N=8 agents, IDs drawn from 1..99, one shared task per pair,
identical score-update wording; framing enters only via prompts_v2 conditions.
Also runs the inequality-feedback probe (full leaderboard with low vs high background dispersion).
"""
import argparse, hashlib, json, random, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from llm import call, parse_choice, LOG, _is_quota_msg
from prompts_v2 import SYSTEM, TASKS, pair_prompt, cond_name

ROOT = Path(__file__).resolve().parents[1]
N_POP = 8
DELTAS_NEAR = [-1, -0.5, 0, 0.5, 1]
DELTAS_FAR = [-16, -8, -4, -2, 2, 4, 8, 16]
MEANS = [-4.0, 0.0, 4.0]

CONDS = {
    "norm0": dict(kind="norm", w=0), "norm25": dict(kind="norm", w=25), "norm50": dict(kind="norm", w=50),
    "norm75": dict(kind="norm", w=75), "norm100": dict(kind="norm", w=100),
    "ended": dict(kind="wording", verb="ended", comp=False), "won": dict(kind="wording", verb="won", comp=False),
    "ended+comp": dict(kind="wording", verb="ended", comp=True), "won+comp": dict(kind="wording", verb="won", comp=True),
}

# inequality probe: focal pair sits at ranks 4-5 of 8 in both boards; only background spread differs
BOARDS = {"low": [2.5, 2.0, 1.5, -1.5, -2.0, -2.5], "high": [11.0, 8.0, 5.0, -5.0, -8.0, -11.0]}
PROBE_DELTAS = [-1.0, -0.5, 0.5, 1.0]


def kernel_jobs(model, cname, reps_near, reps_far, rng):
    jobs, meta = [], []
    cond = CONDS[cname]
    for d, reps in [(d, reps_near) for d in DELTAS_NEAR] + [(d, reps_far) for d in DELTAS_FAR]:
        for m in MEANS:
            for _ in range(reps):
                me, other = rng.sample(range(1, 100), 2)
                h_me, h_ot = m + d / 2, m - d / 2
                jobs.append(dict(model=model, system=SYSTEM.format(me=me, n=N_POP), tag="calib",
                                 prompt=pair_prompt(me, other, h_me, h_ot, rng, cond)))
                meta.append(dict(exp="kernel", model=model, cond=cname, delta=d, mean=m, h_me=h_me, h_other=h_ot))
    return jobs, meta


def probe_jobs(model, cname, reps, rng):
    jobs, meta = [], []
    cond = CONDS[cname]
    for disp, bg in BOARDS.items():
        for d in PROBE_DELTAS:
            for _ in range(reps):
                ids = rng.sample(range(1, 100), 8)
                me, other, rest = ids[0], ids[1], ids[2:]
                board = list(zip(rest, bg))
                jobs.append(dict(model=model, system=SYSTEM.format(me=me, n=N_POP), tag="probe",
                                 prompt=pair_prompt(me, other, d / 2, -d / 2, rng, cond, board=board)))
                meta.append(dict(exp="probe", model=model, cond=cname, disp=disp, delta=d, mean=0.0,
                                 h_me=d / 2, h_other=-d / 2))
    return jobs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude:haiku")
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--reps_near", type=int, default=8)
    ap.add_argument("--reps_far", type=int, default=4)
    ap.add_argument("--probe_conds", nargs="*", default=[])
    ap.add_argument("--probe_reps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "results" / "calib_v2.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    jobs, meta = [], []
    for c in a.conds:
        j, m = kernel_jobs(a.model, c, a.reps_near, a.reps_far, rng); jobs += j; meta += m
    for c in a.probe_conds:
        j, m = probe_jobs(a.model, c, a.probe_reps, rng); jobs += j; meta += m
    order = list(range(len(jobs))); rng.shuffle(order)
    # resume: recover responses already logged for identical (model, system+prompt) calls
    logged = defaultdict(list)
    if LOG.exists():
        for l in open(LOG):
            r = json.loads(l)
            if r["model"] == a.model and r["tag"] in ("calib", "probe") and r["response"] and not _is_quota_msg(r["response"]):
                logged[r["phash"]].append(r["response"])
    lock = threading.Lock()
    f = open(a.out, "w")          # rebuilt from the call log on every (re)start; nothing is lost

    def write(i, resp, recovered):
        with lock:
            f.write(json.dumps({**meta[i], "response": resp, "choice": parse_choice(resp),
                                "recovered": recovered}) + "\n"); f.flush()

    todo = []
    for i in order:
        ph = hashlib.sha1((jobs[i]["system"] + jobs[i]["prompt"]).encode()).hexdigest()[:12]
        if logged.get(ph):
            write(i, logged[ph].pop(0), True)
        else:
            todo.append(i)
    print(f"{len(jobs)} jobs: {len(jobs) - len(todo)} recovered from log, {len(todo)} to run", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(call, jobs[i]["model"], jobs[i]["system"], jobs[i]["prompt"], jobs[i]["tag"]): i for i in todo}
        for fu in as_completed(futs):
            write(futs[fu], fu.result(), False)
    f.close()


if __name__ == "__main__":
    main()
