"""State-matched calibration: query the LLM at the (displayed) score pairs that populations actually visit.

Visited states come from surrogate populations simulated with the grid-calibrated flexible and Bonabeau kernels
(no LLM population data is used). Output rows have the same schema as grid calibration plus design="state",
written to results/statecal_<model>.jsonl (kept apart from calib_v2*.jsonl so grid kernels are unchanged).
Resumable from the call log; rows are streamed as calls complete.
"""
import argparse, hashlib, json, random, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from llm import call, parse_choice, LOG, _is_quota_msg
from prompts_v2 import SYSTEM, pair_prompt
from calibrate import CONDS, N_POP
from population2 import KERNELS, KernelDecider, run
from predict import load_calib

ROOT = Path(__file__).resolve().parents[1]


def visited_states(d, n_sims=20, N=8, rho=0.5, mu=0.1, rounds=40, seed=0):
    """Displayed (h_self, h_other) for every decision in surrogate populations under flexible + Bonabeau kernels."""
    rng = np.random.default_rng(seed)
    out = []
    for kern in ("flexible", "bonabeau"):
        k = KERNELS[kern](d)
        for _ in range(n_sims):
            rec = run(N, rho, mu, rounds, KernelDecider(k), seed=int(rng.integers(1e9)))
            h = np.zeros(N)
            for x in rec["history"]:
                disp = [round(v, 1) for v in h]
                for i, j, *_ in x["events"]:
                    out.append((disp[i], disp[j])); out.append((disp[j], disp[i]))
                h = np.array(x["h"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conds", nargs="+", default=["norm0", "ended", "ended+comp", "norm100"])
    ap.add_argument("--k", type=int, default=400)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = ROOT / "results" / f"statecal_{a.model.replace(':', '_')}.jsonl"
    calib = load_calib()
    rng = random.Random(a.seed)
    jobs, meta = [], []
    for c in a.conds:
        d = calib[(calib.model == a.model) & (calib.cond == c)]
        states = visited_states(d, seed=a.seed)
        for _ in range(a.k):
            hs, ho = rng.choice(states)
            me, other = rng.sample(range(1, 100), 2)
            jobs.append(dict(model=a.model, system=SYSTEM.format(me=me, n=N_POP), tag="statecal",
                             prompt=pair_prompt(me, other, hs, ho, rng, CONDS[c])))
            meta.append(dict(exp="kernel", design="state", model=a.model, cond=c,
                             delta=round(hs - ho, 2), mean=round((hs + ho) / 2, 2), h_me=hs, h_other=ho))
    logged = defaultdict(list)
    if LOG.exists():
        for l in open(LOG):
            r = json.loads(l)
            if r["model"] == a.model and r["tag"] == "statecal" and r["response"] and not _is_quota_msg(r["response"]):
                logged[r["phash"]].append(r["response"])
    lock = threading.Lock()
    f = open(out, "w")

    def write(i, resp, rec):
        with lock:
            f.write(json.dumps({**meta[i], "response": resp, "choice": parse_choice(resp), "recovered": rec}) + "\n")
            f.flush()

    todo = []
    for i in range(len(jobs)):
        ph = hashlib.sha1((jobs[i]["system"] + jobs[i]["prompt"]).encode()).hexdigest()[:12]
        if logged.get(ph):
            write(i, logged[ph].pop(0), True)
        else:
            todo.append(i)
    print(f"{len(jobs)} jobs: {len(jobs) - len(todo)} recovered, {len(todo)} to run", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(call, jobs[i]["model"], jobs[i]["system"], jobs[i]["prompt"], "statecal"): i for i in todo}
        for fu in as_completed(futs):
            write(futs[fu], fu.result(), False)
    f.close()


if __name__ == "__main__":
    main()
