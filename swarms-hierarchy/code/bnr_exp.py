"""Step 2b: Ben-Naim-Redner-rules test with LLM agents.

Rules shown to agents and enforced: coordinator +1, other agent +0; every round every agent's record drops by r
(never below 0). Contest decision as before (lone LEAD wins, else coin flip). BNR prediction: lower-class
fraction L = r / r_c with r_c = max per-agent win rate (= 2 with rho = 1: each agent initiates one contest and is
targeted ~once) for a status-sensitive kernel; r_c = 1 (mean win rate) for a status-blind kernel.

Subcommands:
  calib   grid over (h_self, h_other) in {0,1,2,4,8,16,32}^2, per condition      -> results/bnr_calib.jsonl
  predict frozen kernel predictions (write-once, hash-stamped)                    -> results/bnr_predictions.json
  run     LLM populations                                                           -> results/bnr_pop.jsonl
  eval    compare
"""
import argparse, hashlib, json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, str(Path(__file__).parent))
from llm import call, parse_choice
from prompts_v2 import SYSTEM, TASKS, NORM

ROOT = Path(__file__).resolve().parents[1]
GRID = [0, 1, 2, 4, 8, 16, 32]
CONDS = {"norm100": 100, "ended": None}
RULE = ("Coordination record (shared public record; after each pairing, the agent that ended up as coordinator gets "
        "+1 and the other agent gets nothing; every round, every agent's record drops by {r:g}, never below 0).")


def prompt(me, other, hs, ho, rng, cond, r, task=None):
    s = RULE.format(r=r)
    if CONDS[cond] is not None:
        s += NORM.format(w=CONDS[cond])
    rows = [(f"Agent {me} (you)", hs), (f"Agent {other}", ho)]
    if rng.random() < 0.5:
        rows.reverse()
    lines = "\n".join(f"- {nm}: {sc:.1f}" for nm, sc in rows)
    return (f"New task: {task or rng.choice(TASKS)}.\nYou are paired with Agent {other}.\n\n{s}\n{lines}\n\n"
            f"Do you LEAD or DEFER?")


# ---------------- kernel ----------------
class GridKernel:
    """p(LEAD | h_self, h_other): bilinear in log1p coordinates over the calibration grid, clamped at 32."""
    def __init__(self, rows, rng=None):
        G = np.array(GRID, float); Z = np.zeros((len(G), len(G)))
        for a, x in enumerate(GRID):
            for b, y in enumerate(GRID):
                c = [r["lead"] for r in rows if r["h_me"] == x and r["h_other"] == y]
                k, n = sum(c), len(c)
                Z[a, b] = rng.beta(k + .5, n - k + .5) if rng is not None else (k + .5) / (n + 1)
        self.f = RegularGridInterpolator((np.log1p(G), np.log1p(G)), Z)
        self.hi = np.log1p(G[-1])

    def __call__(self, hs, ho):
        pts = np.c_[np.minimum(np.log1p(hs), self.hi), np.minimum(np.log1p(ho), self.hi)]
        return np.clip(self.f(pts), 0, 1)


def blind_kernel(rows, rng=None):
    k, n = sum(r["lead"] for r in rows), len(rows)
    p = rng.beta(k + .5, n - k + .5) if rng is not None else (k + .5) / (n + 1)
    return lambda hs, ho: np.full(len(hs), p)


# ---------------- dynamics ----------------
def step_round(h, r, rho, rng, decide):
    N = len(h)
    init = np.nonzero(rng.random(N) < rho)[0]
    events = []
    if len(init):
        j = rng.integers(0, N - 1, len(init)); j = j + (j >= init)
        disp = np.round(h, 1)
        ci, cj = decide(init, j, disp)
        coin = rng.random(len(init)) < .5
        iw = (ci & ~cj) | (~(ci ^ cj) & coin)
        w = np.where(iw, init, j)
        dh = np.zeros(N); np.add.at(dh, w, 1.0)
        h = h + dh
        events = [[int(a), int(b), bool(x), bool(y), int(c)] for a, b, x, y, c in zip(init, j, ci, cj, w)]
    return np.maximum(h - r, 0.0), events


def metrics(H, rounds):
    h = np.array(H[-1]); thr = 0.1 * rounds
    return dict(L=float((h <= thr).mean()), leader_speed=float(h.max() / rounds), sd=float(h.std()),
                top_share=float(np.sort(h)[-1] / max(h.sum(), 1e-9)))


def sim_kernel(kern, N, r, rho, rounds, seed):
    rng = np.random.default_rng(seed); h = np.zeros(N); H = []
    def decide(i, j, d):
        return rng.random(len(i)) < kern(d[i], d[j]), rng.random(len(i)) < kern(d[j], d[i])
    for _ in range(rounds):
        h, _ = step_round(h, r, rho, rng, decide); H.append(h)
    return metrics(H, rounds)


# ---------------- subcommands ----------------
def cmd_calib(a):
    rng = random.Random(7); jobs, meta = [], []
    for cond in CONDS:
        for x in GRID:
            for y in GRID:
                for _ in range(a.reps):
                    me, ot = rng.sample(range(1, 100), 2)
                    jobs.append((SYSTEM.format(me=me, n=a.N), prompt(me, ot, x, y, rng, cond, a.r_show)))
                    meta.append(dict(cond=cond, h_me=x, h_other=y, r_show=a.r_show))
    out = open(ROOT / "results" / "bnr_calib.jsonl", "a"); lock = threading.Lock()
    def one(k):
        resp = call(a.model, jobs[k][0], jobs[k][1], tag="bnr_calib")
        with lock:
            out.write(json.dumps({**meta[k], "model": a.model, "response": resp, "choice": parse_choice(resp)}) + "\n")
            out.flush()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, range(len(jobs))))
    print("calib rows", len(jobs))


def load_calib(model):
    rows = [json.loads(l) for l in open(ROOT / "results" / "bnr_calib.jsonl")]
    rows = [dict(r, lead=float(r["choice"] == "LEAD")) for r in rows if r["model"] == model and r["choice"]]
    return rows


def cmd_predict(a):
    out = ROOT / "results" / "bnr_predictions.json"
    if out.exists():
        sys.exit("frozen predictions exist; refusing to overwrite")
    rows = load_calib(a.model); rng = np.random.default_rng(0); res = dict(model=a.model, N=a.N, rho=a.rho,
                                                                          rounds=a.rounds, preds=[])
    for cond in CONDS:
        rc = [x for x in rows if x["cond"] == cond]
        for r in a.rs:
            for kname in ("grid", "blind"):
                samp = []
                for b in range(a.B):
                    k = GridKernel(rc, rng) if kname == "grid" else blind_kernel(rc, rng)
                    for s in range(a.S):
                        samp.append(sim_kernel(k, a.N, r, a.rho, a.rounds, int(rng.integers(1e9))))
                res["preds"].append(dict(cond=cond, r=r, kernel=kname,
                                         **{m: [x[m] for x in samp] for m in samp[0]}))
            g = [p for p in res["preds"] if p["cond"] == cond and p["r"] == r and p["kernel"] == "grid"][0]
            print(cond, r, "pred L", round(np.mean(g["L"]), 2), "leader speed", round(np.mean(g["leader_speed"]), 2))
    res["calib_sha256"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    res["source_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    res["sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    out.write_text(json.dumps(res)); print("frozen", res["sha256"])


def run_llm(model, cond, r, N, rho, rounds, seed):
    rng = np.random.default_rng(seed); prng = random.Random(10 ** 6 + seed)
    names = prng.sample(range(1, 100), N); h = np.zeros(N); H, ev = [], []
    def decide(i, j, d):
        jobs = []
        for a_, b_ in zip(i, j):
            task = prng.choice(TASKS)
            for x, y in ((a_, b_), (b_, a_)):
                jobs.append((SYSTEM.format(me=names[x], n=N), prompt(names[x], names[y], d[x], d[y], prng, cond, r, task)))
        def one(jb):
            for _ in range(3):
                c = parse_choice(call(model, jb[0], jb[1], tag="bnr_pop"))
                if c:
                    return c
            return "LEAD" if rng.random() < .5 else "DEFER"
        with ThreadPoolExecutor(min(len(jobs), 8) or 1) as ex:
            ch = list(ex.map(one, jobs))
        return np.array([c == "LEAD" for c in ch[0::2]]), np.array([c == "LEAD" for c in ch[1::2]])
    for t in range(rounds):
        h, e = step_round(h, r, rho, rng, decide); H.append(h.tolist()); ev.append(e)
    rec = dict(model=model, cond=cond, r=r, N=N, rho=rho, rounds=rounds, seed=seed, history=H, events=ev,
               **metrics(H, rounds))
    with open(ROOT / "results" / "bnr_pop.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("cond", "r", "seed", "L", "leader_speed")}), flush=True)


def cmd_run(a):
    pred = ROOT / "results" / "bnr_predictions.json"
    if not pred.exists():
        sys.exit("freeze predictions first")
    cfgs = [(c, r, s) for c in CONDS for r in a.rs for s in range(a.seeds)]
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda x: run_llm(a.model, x[0], x[1], a.N, a.rho, a.rounds, 500 + x[2]), cfgs))


def cmd_eval(a):
    P = json.load(open(ROOT / "results" / "bnr_predictions.json"))
    recs = [json.loads(l) for l in open(ROOT / "results" / "bnr_pop.jsonl")]
    for cond in CONDS:
        for r in sorted({x["r"] for x in recs if x["cond"] == cond}):
            obs = [x for x in recs if x["cond"] == cond and x["r"] == r]
            line = f"{cond:8s} r={r:<4} obs L={np.mean([x['L'] for x in obs]):.2f} (n={len(obs)}) v={np.mean([x['leader_speed'] for x in obs]):.2f}"
            for kn in ("grid", "blind"):
                p = [q for q in P["preds"] if q["cond"] == cond and q["r"] == r and q["kernel"] == kn]
                if p:
                    L = np.array(p[0]["L"]); line += f" | {kn}: L={L.mean():.2f} [{np.quantile(L,.05):.2f},{np.quantile(L,.95):.2f}]"
            print(line, f"| BNR theory r/2={min(r/2,1):.2f}, blind r_c=1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "predict", "run", "eval"])
    ap.add_argument("--model", default="adaption:gemma-4-31b-it")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--r_show", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--rs", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--S", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args()
    {"calib": cmd_calib, "predict": cmd_predict, "run": cmd_run, "eval": cmd_eval}[a.cmd](a)
