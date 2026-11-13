"""Population engine for the Chase experiments: the population2 protocol with an event log as the visible state.

Protocol (same scheduling and tie rule as population2.run):
  each round every agent initiates a contest with prob rho against a uniform random partner;
  all contests in a round are decided from the round-start record (synchronous);
  both answer LEAD/DEFER; a lone LEAD wins, otherwise coin flip; the outcome is appended to the event log.
  Score arms (SCORE, SCORE+LOG) additionally keep population2 scores: winner +Q, loser -F, then shrink by (1 - mu).
What each agent is shown is set by the information arm (chase_prompts.ARMS).

Deciders
  LLMDecider(models, arm, framing)   models: one model id, or a list of N ids for mixed groups
  KernelDecider(pfun)                pfun(feat) -> P(LEAD), feat = dict(d, o, p, tc, n_own, n_par, delta, mean)
"""
import json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from chase_prompts import SYSTEM, TASKS, log_prompt, features, visible

ROOT = Path(__file__).resolve().parents[1]
SCORE_ARMS = ("SCORE", "SCORE+LOG")


def _feat(arm, a, b, names, events, t_now, h):
    own, partner = ([], []) if arm == "SCORE" else visible(arm, names[a], names[b], events, t_now)
    f = features(names[a], names[b], own, partner, arm) if arm != "SCORE" else dict(d=0, o=0, p=0, tc=0, n_own=0, n_par=0)
    ha, hb = round(float(h[a]), 1), round(float(h[b]), 1)
    f.update(delta=round(ha - hb, 2) if arm in SCORE_ARMS else 0.0, mean=round((ha + hb) / 2, 2) if arm in SCORE_ARMS else 0.0)
    return f, own, partner, (ha, hb)


class LLMDecider:
    def __init__(self, models, arm, framing, retries=2, tag="chase_pop"):
        self.models, self.arm, self.framing, self.retries, self.tag = models, arm, framing, retries, tag
        self.invalid = 0

    def __call__(self, pairs, names, events, t_now, h, N, rng_prompt, rng_sched, mu, Q, F):
        from llm import call, parse_choice
        jobs, feats = [], []
        for i, j in pairs:
            task = rng_prompt.choice(TASKS)
            for a, b in ((i, j), (j, i)):
                f, own, partner, sc = _feat(self.arm, a, b, names, events, t_now, h)
                model = self.models if isinstance(self.models, str) else self.models[a]
                jobs.append((model, SYSTEM.format(me=names[a], n=N),
                             log_prompt(self.arm, self.framing, names[a], names[b], t_now, own, partner, task,
                                        scores=sc, rng=rng_prompt, q=Q, f=F, mu=mu)))
                feats.append(f)

        def one(job):
            for _ in range(self.retries + 1):
                c = parse_choice(call(job[0], job[1], job[2], tag=self.tag))
                if c:
                    return c
            return None

        with ThreadPoolExecutor(len(jobs) or 1) as ex:
            ch = list(ex.map(one, jobs))
        for k, c in enumerate(ch):
            if c is None:
                self.invalid += 1
                ch[k] = "LEAD" if rng_sched.random() < 0.5 else "DEFER"
        return [(ch[2 * k], ch[2 * k + 1]) for k in range(len(pairs))], feats


class KernelDecider:
    def __init__(self, pfun, arm):
        self.pfun, self.arm, self.invalid = pfun, arm, 0

    def __call__(self, pairs, names, events, t_now, h, N, rng_prompt, rng_sched, mu, Q, F):
        out, feats = [], []
        for i, j in pairs:
            fi = _feat(self.arm, i, j, names, events, t_now, h)[0]
            fj = _feat(self.arm, j, i, names, events, t_now, h)[0]
            out.append(("LEAD" if rng_sched.random() < self.pfun(fi) else "DEFER",
                        "LEAD" if rng_sched.random() < self.pfun(fj) else "DEFER"))
            feats += [fi, fj]
        return out, feats


def run(N, rho, rounds, arm, decider, seed=0, mu=0.1, Q=1.0, F=1.0, meta=None, out=None, keep_feats=True):
    """One group. History rows match population2 ({t, h, events: [i, j, ci, cj, winner]}) plus per-decision features."""
    rng_sched, rng_prompt = random.Random(seed), random.Random(10 ** 6 + seed)
    names = rng_prompt.sample(range(1, 100), N)
    h = np.zeros(N)
    events, hist = [], []                       # events: (display round, winner name, loser name)
    for t in range(rounds):
        pairs = []
        for i in range(N):
            if rng_sched.random() < rho:
                j = rng_sched.randrange(N - 1); j += j >= i
                pairs.append((i, j))
        choices, feats = decider(pairs, names, events, t + 1, h, N, rng_prompt, rng_sched, mu, Q, F) if pairs else ([], [])
        dh, evs, new = np.zeros(N), [], []
        for (i, j), (ci, cj) in zip(pairs, choices):
            if ci == "LEAD" and cj != "LEAD":
                w = i
            elif cj == "LEAD" and ci != "LEAD":
                w = j
            else:
                w = i if rng_sched.random() < 0.5 else j
            l = j if w == i else i
            dh[w] += Q; dh[l] -= F
            evs.append([i, j, ci, cj, w]); new.append((t + 1, names[w], names[l]))
        events += new
        h = (h + dh) * (1 - mu)
        row = dict(t=t, h=[round(float(x), 3) for x in h], events=evs)
        if keep_feats:
            row["feats"] = [[f["d"], f["o"], f["p"], f["tc"], f["n_own"], f["n_par"], f["delta"], f["mean"]] for f in feats]
        hist.append(row)
    rec = dict(**(meta or {}), arm=arm, N=N, rho=rho, mu=mu, Q=Q, F=F, seed=seed, rounds=rounds, names=names,
               invalid=decider.invalid, history=hist, finished=time.time())
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


FEAT_COLS = ["d", "o", "p", "tc", "n_own", "n_par", "delta", "mean"]


def decisions(rec):
    """Per-decision rows (features + lead) from a run, for kernel fitting / leave-one-seed-out bounds."""
    rows = []
    for x in rec["history"]:
        for k, ev in enumerate(x["events"]):
            for side in (0, 1):
                rows.append(dict(zip(FEAT_COLS, x["feats"][2 * k + side]), lead=float(ev[2 + side] == "LEAD"),
                                 seed=rec["seed"], t=x["t"]))
    return rows
