"""Incident-scale check: real LLM populations at N = 700 (Gemma-4-31B), same protocol as the N = 8 runs
(population2: rho = 0.5, mu = 0.1, Q = F = 1, 40 rounds, synchronous contests, displayed scores rounded to 0.1).
Tests the large-N simulations (A5 / Figure A1) directly, including the ~9.5% of decision states that fall
outside the calibrated mean-score range.

Only change from population2.run: agent IDs are drawn from 1..999 (population2 draws from 1..99, which cannot
name 700 agents). Each round is checkpointed so a crash keeps the data.

  predict -> results/pop700_predictions.json   (write-once, hash-stamped; kernels: grid, state-matched, Bonabeau, null)
  run     -> results/pop700_rounds.jsonl (per round) + results/pop700_pop.jsonl (per run)
  eval
"""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from population2 import KERNELS, KernelDecider, LLMDecider, order_params
from predict import load_calib, load_statecal
from calibrate import CONDS

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PRED, ROUNDS_OUT, POP_OUT = RES / "pop700_predictions.json", RES / "pop700_rounds.jsonl", RES / "pop700_pop.jsonl"
N, RHO, MU, T, Q, F = 700, 0.5, 0.1, 40, 1.0, 1.0
MEAN_RANGE, DELTA_MAX = (-4.0, 4.0), 16.0          # calibrated support of the grid kernel


def grid_rows(model, cond):
    c = load_calib()
    return c[(c.model == model) & (c.cond == cond)]


def all_rows(model, cond):
    c, s = load_calib(), load_statecal()
    if len(s):
        s = s[(s.model == model) & (s.cond == cond)]
        c = c[(c.model == model) & (c.cond == cond)]
        import pandas as pd
        return pd.concat([c, s[[x for x in s.columns if x in c.columns]]], ignore_index=True)
    return c[(c.model == model) & (c.cond == cond)]


def simulate(decider, seed, n=N, rounds=T, meta=None, checkpoint=None):
    """population2.run with IDs from 1..999 and optional per-round checkpointing."""
    rng_sched, rng_prompt = random.Random(seed), random.Random(10 ** 6 + seed)
    names = rng_prompt.sample(range(1, 1000), n)
    h = np.zeros(n)
    hist, off = [], [0, 0]
    for t in range(rounds):
        t0 = time.time()
        pairs = []
        for i in range(n):
            if rng_sched.random() < RHO:
                j = rng_sched.randrange(n - 1); j += j >= i
                pairs.append((i, j))
        for i, j in pairs:                                   # support bookkeeping on displayed scores
            hi, hj = round(h[i], 1), round(h[j], 1)
            off[0] += 1
            off[1] += not (MEAN_RANGE[0] <= (hi + hj) / 2 <= MEAN_RANGE[1] and abs(hi - hj) <= DELTA_MAX)
        choices = decider(pairs, h, names, n, rng_prompt, rng_sched, MU, Q, F) if pairs else []
        dh, events = np.zeros(n), []
        for (i, j), (ci, cj) in zip(pairs, choices):
            if ci == "LEAD" and cj != "LEAD":
                w = i
            elif cj == "LEAD" and ci != "LEAD":
                w = j
            else:
                w = i if rng_sched.random() < 0.5 else j
            l = j if w == i else i
            dh[w] += Q; dh[l] -= F
            events.append([i, j, ci, cj, w])
        h = (h + dh) * (1 - MU)
        hist.append(dict(t=t, h=[round(float(x), 3) for x in h], events=events))
        if checkpoint:
            with open(checkpoint, "a") as f:
                f.write(json.dumps(dict(**(meta or {}), seed=seed, **hist[-1])) + "\n")
            print(f"{(meta or {}).get('cond')} round {t + 1}/{rounds}: {len(pairs)} contests, "
                  f"sd_h={np.std(h):.2f}, max={h.max():.1f}, {time.time() - t0:.0f}s", flush=True)
    return dict(**(meta or {}), N=n, rho=RHO, mu=MU, Q=Q, F=F, seed=seed, rounds=rounds, names=names,
                h0=None, invalid=decider.invalid, history=hist, off_support=off[1] / max(off[0], 1),
                finished=time.time())


def metrics(rec):
    m = order_params(rec)
    H = np.array([x["h"] for x in rec["history"][-20:]])
    m["max_h"] = float(H[-1].max())
    m["off_support"] = rec.get("off_support")
    return m


def cmd_predict(a):
    if PRED.exists():
        sys.exit("frozen; refusing to overwrite")
    rng = np.random.default_rng(700)
    res = dict(model=a.model, N=N, rho=RHO, mu=MU, rounds=T, B=a.B, S=a.S, created=time.time(), preds=[])
    for cond in a.conds:
        g, al = grid_rows(a.model, cond), all_rows(a.model, cond)
        for kern, rows in (("flexible", g), ("nw", al), ("bonabeau", g), ("null", g)):
            samp = []
            for b in range(a.B):
                k = KERNELS[kern](rows, rng)
                for s in range(a.S):
                    samp.append(metrics(simulate(KernelDecider(k), int(rng.integers(1e9)))))
            res["preds"].append(dict(cond=cond, kernel=kern, n_rows=len(rows),
                                     **{m: [x[m] for x in samp] for m in samp[0]}))
            p = res["preds"][-1]
            print(cond, kern, "sd_h", round(np.mean(p["sd_h"]), 2), "q", round(np.mean(p["q"]), 3),
                  "off_support", round(np.mean(p["off_support"]), 3), flush=True)
    res["sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    PRED.write_text(json.dumps(res)); print("frozen", res["sha256"])


def cmd_run(a):
    if not PRED.exists():
        sys.exit("freeze predictions first")
    for cond in a.conds:
        meta = dict(model=a.model, cond=cond, tag="pop700")
        rec = simulate(LLMDecider(a.model, CONDS[cond]), a.seed, n=a.N, rounds=a.rounds, meta=meta,
                       checkpoint=ROUNDS_OUT if a.N == N else None)
        if a.N != N:                                         # smoke test: report only, write nothing
            print("smoke", cond, json.dumps(metrics(rec))); continue
        with open(POP_OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("done", cond, json.dumps(metrics(rec)), flush=True)


def cmd_eval(a):
    P = json.load(open(PRED))
    recs = [json.loads(l) for l in open(POP_OUT)] if POP_OUT.exists() else []
    for r in recs:
        o = metrics(r)
        print(f"\n{r['cond']} seed {r['seed']}: observed sd_h={o['sd_h']:.2f} q={o['q']:.3f} "
              f"leader_share={o['leader_share']:.4f} off_support={o['off_support']:.3f} invalid={r['invalid']}")
        for p in P["preds"]:
            if p["cond"] != r["cond"]:
                continue
            line = f"  {p['kernel']:9s}"
            for m in ("sd_h", "q"):
                s = np.array(p[m]); lo, hi = np.quantile(s, [0.05, 0.95])
                line += f" {m}={s.mean():.2f} [{lo:.2f},{hi:.2f}] {'IN ' if lo <= o[m] <= hi else 'OUT'}"
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "run", "eval"])
    ap.add_argument("--model", default="adaption:gemma-4-31b-it")
    ap.add_argument("--conds", nargs="+", default=["norm100", "norm0"])
    ap.add_argument("--B", type=int, default=10)
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--N", type=int, default=N)
    ap.add_argument("--rounds", type=int, default=T)
    a = ap.parse_args()
    {"predict": cmd_predict, "run": cmd_run, "eval": cmd_eval}[a.cmd](a)
