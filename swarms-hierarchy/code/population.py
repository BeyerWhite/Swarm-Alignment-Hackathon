"""Population experiment: Bonabeau dynamics with LLM agents in the loop (or a fitted surrogate).

Each round: every agent initiates a contest with probability rho against a uniformly random partner
(well-mixed / mean-field, as in Lacasa & Luque).  All contests in a round are decided in parallel from
the scores at the start of the round (synchronous update).  Both agents answer LEAD/DEFER; a single
LEAD wins, otherwise a coin flip.  Coordinator gets +Q, the other -F; then all scores shrink by (1-mu).
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from llm import batch, parse_choice
from pairwise import SYSTEM, scalar_prompt

ROOT = Path(__file__).resolve().parents[1]


def decide_llm(model, frame, pairs, h, names, N, rng, mu, Q, F):
    jobs = []
    for i, j in pairs:
        for a, b in ((i, j), (j, i)):
            jobs.append(dict(model=model, system=SYSTEM.format(me=names[a], n=N),
                             prompt=scalar_prompt(names[a], names[b], round(h[a], 1), round(h[b], 1), N, rng,
                                                  q=Q, f=F, mu=mu, frame=frame), tag="population"))
    resp = batch(jobs, workers=16)
    ch = [parse_choice(r) for r in resp]
    return [(ch[2 * k], ch[2 * k + 1]) for k in range(len(pairs))], resp


def decide_surrogate(params, pairs, h, rng_np):
    """params: (a, b, c) logistic in (delta, mean) or a callable p(h_self, h_other)."""
    if callable(params):
        pfun = params
    else:
        a, b, c = params
        pfun = lambda x, y: 1 / (1 + np.exp(-(a + b * (x - y) + c * (x + y) / 2)))
    out = []
    for i, j in pairs:
        hi, hj = round(h[i], 1), round(h[j], 1)
        pi, pj = pfun(hi, hj), pfun(hj, hi)
        out.append(("LEAD" if rng_np.random() < pi else "DEFER", "LEAD" if rng_np.random() < pj else "DEFER"))
    return out, None


def run(N, rho, mu, rounds, model=None, frame="reward", Q=1.0, F=1.0, seed=0, h0=None,
        surrogate=None, out=None, tag=""):
    rng, rng_np = random.Random(seed), np.random.default_rng(seed)
    names = rng.sample(range(1, 100), N)
    h = np.zeros(N) if h0 is None else np.array(h0, float)
    hist = []
    for t in range(rounds):
        pairs = []
        for i in range(N):
            if rng.random() < rho:
                j = rng.randrange(N - 1); j += j >= i
                pairs.append((i, j))
        if pairs:
            if surrogate is not None:
                choices, _ = decide_surrogate(surrogate, pairs, h, rng_np)
            else:
                choices, _ = decide_llm(model, frame, pairs, h, names, N, rng, mu, Q, F)
        else:
            choices = []
        dh = np.zeros(N)
        events = []
        for (i, j), (ci, cj) in zip(pairs, choices):
            if ci == "LEAD" and cj != "LEAD":
                w = i
            elif cj == "LEAD" and ci != "LEAD":
                w = j
            else:
                w = i if rng.random() < 0.5 else j
            l = j if w == i else i
            dh[w] += Q; dh[l] -= F
            events.append([int(i), int(j), ci, cj, int(w)])
        h = (h + dh) * (1 - mu)
        hist.append(dict(t=t, h=[round(x, 3) for x in h], events=events))
    rec = dict(tag=tag, N=N, rho=rho, mu=mu, Q=Q, F=F, model=model, frame=frame, seed=seed,
               surrogate=None if surrogate is None or callable(surrogate) else list(surrogate), h0=None if h0 is None else list(map(float, h0)), names=names,
               rounds=rounds, history=hist, finished=time.time())
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


# ---------- model-free order parameters ----------
def order_params(rec, window=None):
    hist = rec["history"]
    window = window or max(5, len(hist) // 3)
    H = np.array([x["h"] for x in hist[-window:]])
    N = rec["N"]
    wins = np.zeros(N)
    games = np.zeros(N)
    for x in hist[-window:]:
        for i, j, ci, cj, w in x["events"]:
            games[i] += 1; games[j] += 1; wins[w] += 1
    share = wins / max(wins.sum(), 1)
    sd = H.std(axis=1).mean()                        # status dispersion
    # persistence of rank: correlation of h between first and last time in window
    rk = np.corrcoef(H[0], H[-1])[0, 1] if H[0].std() > 0 and H[-1].std() > 0 else 0.0
    winrate = np.where(games > 0, wins / np.maximum(games, 1), np.nan)
    # frozen-order fraction (Edwards-Anderson-like): dispersion of time-averaged status / total dispersion
    tot = H.var(axis=1).mean()
    q = float(H.mean(axis=0).var() / tot) if tot > 0 else 0.0
    return dict(sd_h=float(sd), q=q, top_share=float(share.max()), rank_persist=float(rk),
                winrate_sd=float(np.nanstd(winrate)), mean_h=float(H.mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.5])
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--model", default="claude:haiku")
    ap.add_argument("--frame", default="reward")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--seeded_hierarchy", type=float, default=0.0,
                    help="if >0, start from a linear hierarchy with this spread (hysteresis test)")
    ap.add_argument("--out", default=str(ROOT / "results" / "population.jsonl"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    for rho in a.rho:
        for s in a.seeds:
            h0 = np.linspace(a.seeded_hierarchy / 2, -a.seeded_hierarchy / 2, a.N) if a.seeded_hierarchy else None
            rec = run(a.N, rho, a.mu, a.rounds, model=a.model, frame=a.frame, seed=s, h0=h0, out=a.out, tag=a.tag)
            print(json.dumps(dict(rho=rho, seed=s, **order_params(rec))), flush=True)


def empirical_surrogate(pairwise_df, model, frame):
    """p(LEAD | delta) by linear interpolation of the measured curve (pooled over mean status)."""
    d = pairwise_df[(pairwise_df.model == model) & (pairwise_df.frame == frame) & pairwise_df.valid]
    g = d.groupby("delta").lead.mean()
    xs, ys = np.array(g.index, float), g.values
    return lambda x, y: float(np.interp(x - y, xs, ys))
