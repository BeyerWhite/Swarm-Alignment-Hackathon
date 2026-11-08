"""X2: Ben-Naim-Redner test with LLM agents (handoff spec). Integer scores k >= 0, random perfect matching each
round (every agent in exactly one contest), coordinator +1, other unchanged, then each agent loses Poisson(r)
points (floor 0). One round = one unit of master-equation time, so BNR predicts r_c = 1 and L = r below it.

  calib   X2-wording calibration incl. high scores (up to 30) -> results/x2_calib.jsonl
  predict frozen predictions (BNR-q simulator + calibrated grid kernel) -> results/x2_predictions.json
  run     LLM populations -> results/x2_pop.jsonl
  eval
L is defined (frozen) as the fraction of agents with k <= max(1, 0.1 t) at the final round (t = 30 -> k <= 3).
"""
import argparse, hashlib, json, random, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from llm import call, parse_choice
from prompts_v2 import SYSTEM, TASKS, NORM
from bnr_x2 import simulate, L_of

ROOT = Path(__file__).resolve().parents[1]
MODEL = "adaption:gemma-4-31b-it"
N, T, RS = 16, 30, [0.25, 0.5, 1.25]
CONDS = {"norm100": 100, "ended": None}
LEVELS = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
GAPS = [0, 1, 2, 5]
RULE = ("Coordination record (shared public record of whole points; after each pairing, the agent that ended up as "
        "coordinator gets +1 and the other agent's record is unchanged; every round, each agent's record drops by a "
        "random number of points averaging {r:g}, never below 0).")


def prompt(me, other, ks, ko, rng, cond, r, task=None):
    s = RULE.format(r=r) + (NORM.format(w=CONDS[cond]) if CONDS[cond] is not None else "")
    rows = [(f"Agent {me} (you)", ks), (f"Agent {other}", ko)]
    if rng.random() < .5:
        rows.reverse()
    lines = "\n".join(f"- {nm}: {sc:d}" for nm, sc in rows)
    return (f"New task: {task or rng.choice(TASKS)}.\nYou are paired with Agent {other}.\n\n{s}\n{lines}\n\n"
            f"Do you LEAD or DEFER?")


def cmd_calib(a):
    rng = random.Random(21); jobs = []
    for cond in CONDS:
        for lo in LEVELS:
            for g in GAPS:
                hi = lo + g
                if hi > 30:
                    continue
                for ks, ko in ({(hi, lo), (lo, hi)}):
                    for _ in range(a.reps):
                        me, ot = rng.sample(range(1, 100), 2); r = rng.choice(RS)
                        jobs.append(dict(cond=cond, ks=ks, ko=ko, r=r,
                                         sys=SYSTEM.format(me=me, n=N), p=prompt(me, ot, ks, ko, rng, cond, r)))
    out = open(ROOT / "results" / "x2_calib.jsonl", "a"); lock = threading.Lock()
    def one(j):
        resp = call(MODEL, j["sys"], j["p"], tag="x2_calib")
        with lock:
            out.write(json.dumps({k: j[k] for k in ("cond", "ks", "ko", "r")} | dict(response=resp, choice=parse_choice(resp))) + "\n"); out.flush()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs))
    print("calib calls", len(jobs))


def lead_table(cond):
    t = defaultdict(list)
    for l in open(ROOT / "results" / "x2_calib.jsonl"):
        x = json.loads(l)
        if x["cond"] == cond and x["choice"]:
            t[(x["ks"], x["ko"])].append(x["choice"] == "LEAD")
    return t


def make_decider(cond, rng_post=None):
    """Kernel decider from the X2 calibration: p(LEAD | ks, ko) looked up at the nearest calibrated (level, gap)
    cell (gap clamped to 5, levels to 30); Jeffreys posterior draw if rng_post."""
    t = lead_table(cond)
    cells = {}
    for (ks, ko), v in t.items():
        k_, n_ = sum(v), len(v)
        cells[(ks, ko)] = rng_post.beta(k_ + .5, n_ - k_ + .5) if rng_post is not None else (k_ + .5) / (n_ + 1)
    keys = np.array(list(cells)); vals = np.array([cells[tuple(k)] for k in keys])
    def p(ks, ko):
        g = np.clip(ks - ko, -5, 5); lo = np.minimum(ks, ko).clip(0, 30)
        cand_lo = np.minimum(keys[:, 0], keys[:, 1]); cand_g = keys[:, 0] - keys[:, 1]
        d = np.abs(cand_g[None, :] - g[:, None]) * 10 + np.abs(cand_lo[None, :] - lo[:, None])
        return vals[d.argmin(1)]
    def decider(ki, kj, rng):
        ci = rng.random(len(ki)) < p(ki, kj); cj = rng.random(len(ki)) < p(kj, ki)
        return (ci & ~cj) | (~(ci ^ cj) & (rng.random(len(ki)) < .5))
    return decider, cells


def coverage(cond):
    t = lead_table(cond)
    rows = []
    for (ks, ko), v in sorted(t.items()):
        if ks > ko:
            loser = t.get((ko, ks), [])
            pw = .5 + .5 * (np.mean(v) - (np.mean(loser) if loser else np.nan))
            rows.append((ks, ko, round(float(pw), 2), len(v)))
    return rows


def cmd_predict(a):
    out = ROOT / "results" / "x2_predictions.json"
    if out.exists():
        sys.exit("frozen; refusing to overwrite")
    rng = np.random.default_rng(0)
    res = dict(model=MODEL, N=N, rounds=T, rs=RS, L_definition="fraction with k <= max(1, 0.1*t) at final round",
               note=("Under norm100 the calibration gives q_hat ~ 0 (agents behave as the BNR rule), so a match of the "
                     "norm100 arm to the BNR simulator is close to automatic. The informative tests are (i) whether "
                     "agents stay decisive at the high scores populations reach (calibration covers levels up to 30), "
                     "and (ii) the 'ended' control arm, which the kernel predicts breaks the BNR pattern."),
               coverage={c: coverage(c) for c in CONDS}, preds=[])
    for cond in CONDS:
        for r in RS:
            for kname in ("bnr_q0", "kernel"):
                if kname == "bnr_q0" and cond == "ended":
                    continue
                Ls, ks = [], []
                for b in range(a.B):
                    dec = None if kname == "bnr_q0" else make_decider(cond, rng)[0]
                    for s in range(a.S):
                        tr = simulate(N, r, T, q=0.0, seed=int(rng.integers(1e9)), decider=dec)
                        Ls.append(L_of(tr[-1], T)); ks.append(int(tr[-1].max()))
                res["preds"].append(dict(cond=cond, r=r, kernel=kname, L=Ls, kmax=ks))
                print(f"{cond:8s} r={r:<5} {kname:7s} L={np.mean(Ls):.2f} [{np.quantile(Ls,.05):.2f},{np.quantile(Ls,.95):.2f}]"
                      f"  kmax={np.mean(ks):.1f}")
    res["sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    out.write_text(json.dumps(res)); print("frozen", res["sha256"])


def run_llm(cond, r, seed):
    rng = np.random.default_rng(seed); prng = random.Random(10 ** 6 + seed)
    names = prng.sample(range(1, 100), N); k = np.zeros(N, int); traj, ev = [], []
    for t in range(T):
        perm = rng.permutation(N); I, J = perm[0::2], perm[1::2]
        jobs = []
        for i, j in zip(I, J):
            task = prng.choice(TASKS)
            for x, y in ((i, j), (j, i)):
                jobs.append((SYSTEM.format(me=names[x], n=N), prompt(names[x], names[y], int(k[x]), int(k[y]), prng, cond, r, task)))
        def one(jb):
            for _ in range(3):
                c = parse_choice(call(MODEL, jb[0], jb[1], tag="x2_pop"))
                if c:
                    return c
            return None
        with ThreadPoolExecutor(8) as ex:
            ch = list(ex.map(one, jobs))
        for n_, (i, j) in enumerate(zip(I, J)):
            ci, cj = ch[2 * n_], ch[2 * n_ + 1]
            ci = ci or ("LEAD" if rng.random() < .5 else "DEFER"); cj = cj or ("LEAD" if rng.random() < .5 else "DEFER")
            w = i if (ci == "LEAD" and cj != "LEAD") else j if (cj == "LEAD" and ci != "LEAD") else (i if rng.random() < .5 else j)
            k[w] += 1; ev.append([t, int(i), int(j), ci, cj, int(w)])
        k = np.maximum(k - rng.poisson(r, N), 0); traj.append(k.tolist())
    rec = dict(model=MODEL, cond=cond, r=r, seed=seed, N=N, rounds=T, traj=traj, events=ev,
               L=L_of(np.array(traj[-1]), T), kmax=int(max(traj[-1])))
    with open(ROOT / "results" / "x2_pop.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(dict(cond=cond, r=r, seed=seed, L=rec["L"], kmax=rec["kmax"])), flush=True)


def cmd_run(a):
    if not (ROOT / "results" / "x2_predictions.json").exists():
        sys.exit("freeze predictions first")
    cfgs = [(c, r, s) for c in CONDS for r in RS for s in (901, 902, 903)]
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda x: run_llm(*x), cfgs))


def cmd_eval(a):
    P = json.load(open(ROOT / "results" / "x2_predictions.json"))
    recs = [json.loads(l) for l in open(ROOT / "results" / "x2_pop.jsonl")]
    for cond in CONDS:
        for r in RS:
            o = [x for x in recs if x["cond"] == cond and x["r"] == r]
            if not o:
                continue
            line = f"{cond:8s} r={r:<5} obs L={np.mean([x['L'] for x in o]):.2f} kmax={np.mean([x['kmax'] for x in o]):.1f} (n={len(o)})"
            for p in [p for p in P["preds"] if p["cond"] == cond and p["r"] == r]:
                L = np.array(p["L"]); d = np.random.default_rng(0).choice(L, (4000, len(o))).mean(1)
                line += f" | {p['kernel']}: L={L.mean():.2f} 90%[{np.quantile(d,.05):.2f},{np.quantile(d,.95):.2f}]"
            print(line)
    # high-score decisiveness check inside populations
    for cond in CONDS:
        ev = [e for x in recs if x["cond"] == cond for e in x["events"]]
        hi = [e for e in ev if e[3] != e[4]]
        print(cond, "decisive contests:", len(hi), "of", len(ev))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "predict", "run", "eval"])
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--S", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    {"calib": cmd_calib, "predict": cmd_predict, "run": cmd_run, "eval": cmd_eval}[a.cmd](a)
