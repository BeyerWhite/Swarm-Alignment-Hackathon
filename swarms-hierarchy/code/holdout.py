"""Prospective off-grid holdout: choose the displayed score pair (off the calibration grid, inside support)
where frozen flexible and Bonabeau kernels disagree most on P_win, then run 48 role-swapped contests."""
import argparse, hashlib, json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from llm import batch, parse_choice
from prompts_v2 import SYSTEM, TASKS, pair_prompt
from calibrate import CONDS
from population2 import KERNELS
from predict import load_calib, calib_sha

ROOT = Path(__file__).resolve().parents[1]


def pwin(f, hi, hj):
    m = (hi + hj) / 2
    pi, pj = f(hi - hj, m), f(hj - hi, m)
    return pi * (1 - pj) + 0.5 * (pi * pj + (1 - pi) * (1 - pj))


def choose(d):
    kf, kb = KERNELS["flexible"](d), KERNELS["bonabeau"](d)
    best = None
    for delta in [0.2, 0.6, 1.4, 3.0, 6.0, 12.0]:
        for mean in [-2.0, 2.0]:
            hi, hj = round(mean + delta / 2, 1), round(mean - delta / 2, 1)
            diff = abs(pwin(kf, hi, hj) - pwin(kb, hi, hj))
            if best is None or diff > best[0]:
                best = (diff, hi, hj, pwin(kf, hi, hj), pwin(kb, hi, hj))
    return best


def score(choice, path):
    """Log score of the realised high-score winner under each frozen P_win."""
    import math
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("stage") == "contest" and r["model"] == choice["model"] and r["cond"] == choice["cond"]]
    ll = {"flexible": 0.0, "bonabeau": 0.0}; n = dropped = hi_wins = ties = 0
    for r in rows:
        if r["cA"] is None or r["cB"] is None:
            dropped += 1; continue
        a_hi = r["hA"] > r["hB"]
        c_hi, c_lo = (r["cA"], r["cB"]) if a_hi else (r["cB"], r["cA"])
        n += 1
        for k in ll:
            p = choice[f"pwin_{k}"]
            if c_hi == "LEAD" and c_lo != "LEAD":
                ll[k] += math.log(p); hi_wins += k == "flexible"
            elif c_lo == "LEAD" and c_hi != "LEAD":
                ll[k] += math.log(1 - p)
            else:
                ll[k] += 0.5 * math.log(p) + 0.5 * math.log(1 - p); ties += k == "flexible"
    return dict(n=n, dropped=dropped, decisive_hi_wins=hi_wins, ties=ties,
                empirical_pwin=(hi_wins + 0.5 * ties) / max(n, 1), logscore=ll,
                pwin_flexible=choice["pwin_flexible"], pwin_bonabeau=choice["pwin_bonabeau"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude:haiku")
    ap.add_argument("--cond", default="ended+comp")
    ap.add_argument("--contests", type=int, default=48)
    ap.add_argument("--out", default=str(ROOT / "results" / "holdout.jsonl"))
    a = ap.parse_args()
    frozen = ROOT / "results" / f"holdout_choice_{a.model.replace(':', '_')}_{a.cond}.json"
    if frozen.exists():
        sys.exit(f"holdout choice already frozen at {frozen}; refusing to re-select")
    d = load_calib(); d = d[(d.model == a.model) & (d.cond == a.cond)]
    diff, hi, hj, pf, pb = choose(d)
    choice = dict(model=a.model, cond=a.cond, h_hi=hi, h_lo=hj, pwin_flexible=pf, pwin_bonabeau=pb, diff=diff,
                  calib_sha256=calib_sha(),
                  source_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                  scoring="lone LEAD wins; tie -> expected log score 0.5*log p + 0.5*log(1-p); invalid -> dropped")
    frozen.write_text(json.dumps(choice, indent=1))           # durable freeze BEFORE any call
    print(json.dumps(choice), flush=True)
    rng = random.Random(77)
    jobs, meta = [], []
    for c in range(a.contests):
        ids = rng.sample(range(1, 100), 2)
        task = rng.choice(TASKS)
        # role swap: half the contests give the higher score to the first-sampled ID
        hA, hB = (hi, hj) if c % 2 == 0 else (hj, hi)
        for me, other, hm, ho in ((ids[0], ids[1], hA, hB), (ids[1], ids[0], hB, hA)):
            jobs.append(dict(model=a.model, system=SYSTEM.format(me=me, n=8), tag="holdout",
                             prompt=pair_prompt(me, other, hm, ho, rng, CONDS[a.cond], task=task)))
        meta.append(dict(contest=c, hA=hA, hB=hB))
    resp = batch(jobs, workers=16)
    ch = [parse_choice(r) for r in resp]
    with open(a.out, "a") as f:
        for k, m in enumerate(meta):
            f.write(json.dumps(dict(stage="contest", model=a.model, cond=a.cond, **m,
                                    cA=ch[2 * k], cB=ch[2 * k + 1])) + "\n")
    print(json.dumps(score(choice, a.out)), flush=True)


if __name__ == "__main__":
    main()
