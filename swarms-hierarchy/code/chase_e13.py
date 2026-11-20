"""E13 (amendment C10): why the Gemma-4-26B `norm0` own-record arm stays above its forecasts. No calls.

  chase_e13.py [--reps 100] [--test] [--out FILE]      -> results/chase_e13.json

Data: OWN decisions of the Gemma-4-26B `norm0` groups of four in E5 (results/chase_e3_pop.jsonl, seeds 4000+, 40
groups) and E8a (results/chase_e8_pop.jsonl, seeds 34000+, 140 groups), 180 groups, rebuilt from the stored
histories. Reported alongside without a test (forecasts covered there): Gemma-4-31B `norm100` OWN (E3 4000+ 40, E9
14000+ 40) and Qwen3.5-397B `comp` OWN (E7 24000+ 40, E10 44000+ 80). --test swaps in Qwen3.5-397B `norm0` OWN
(E8b, 34000+, 80 groups) and is the only mode run before the hash.

Models (logistic on signed-log features, ridge as chase_fit):
  C0  X of E6: d, o, hinge, n_own, last, streak, idlow, idme
  C1  C0 + tr, the round number shown in the prompt
  C2  C0 with a normal random intercept per agent (an agent is one member of one group), fitted by marginal
      likelihood; tested against C0 by the likelihood ratio, p from the 50:50 mixture of chi2(0) and chi2(1).
Adequacy: group bootstrap (resample the groups, refit, simulate as many groups; `reps` times; for C2 every simulated
agent draws its own intercept once); 90% interval against the observed linear share.
Reading, fixed now:
  "no gap"         C0 covers the observed share: on all 180 groups the miss is gone, so the earlier misses came
                   from the calibration records and from E5's high draw;
  "agent differences"  C0 does not cover; C2 has LR p < 0.05 and covers;
  "round number"   C0 does not cover; C1 improves the leave-one-seed-out log-loss on C0 by at least 0.005 and covers;
  "both" if the last two both hold; "neither" otherwise.
Descriptive: the spread of per-agent LEAD rates (agents with at least 8 decisions) against the spread in groups
simulated from C0 at the same agents' decision counts (90% interval over 200 draws); the C2 sigma in the two
comparison cells.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).parent))
import chase_c10 as C

C0 = ["d", "o", "hinge", "n_own", "last", "streak", "idlow", "idme"]
C1 = C0 + ["tr"]
MAIN = [("chase_e3_pop.jsonl", C.G26, "norm0", 4000, 40), ("chase_e8_pop.jsonl", C.G26, "norm0", 34000, 140)]
COMPARE = {"gemma31_norm100": [("chase_e3_pop.jsonl", C.G31, "norm100", 4000, 40), ("chase_e9_pop.jsonl", C.G31, "norm100", 14000, 40)],
           "qwen_comp": [("chase_e7_pop.jsonl", C.QWEN, "comp", 24000, 40), ("chase_e10_pop.jsonl", C.QWEN, "comp", 44000, 80)]}
TEST = [("chase_e8_pop.jsonl", C.QWEN, "norm0", 34000, 80)]


def get(parts):
    recs = []
    for fname, model, framing, seed0, G in parts:
        recs += C.load(fname, model, framing, 4, seed0, {"OWN": G})["OWN"]
    d, bad = C.frame(recs)
    obs = float(np.mean([C.structure(C.events_from_history(r), 4, r["rounds"])["linear"] for r in recs]))
    return recs, d, bad, obs


def spread(d, coef, rng, draws=200):
    """SD of per-agent LEAD rates (agents with >= 8 decisions), observed and under C0 (Bernoulli at the same rows)."""
    keep = d.groupby("agent").lead.transform("size") >= 8
    dd = d[keep]
    z = coef[0] + C.F.slog(dd[list(coef[1])].values.astype(float)) @ np.array(list(coef[1].values()))
    p = 1 / (1 + np.exp(-z))
    obs = float(dd.groupby("agent").lead.mean().std())
    sim = [float(dd.assign(lead=(rng.random(len(p)) < p).astype(float)).groupby("agent").lead.mean().std()) for _ in range(draws)]
    return dict(agents=int(dd.agent.nunique()), observed=obs, null_q05=float(np.quantile(sim, .05)), null_q95=float(np.quantile(sim, .95)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out", default=str(C.R / "chase_e13.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(20261120)
    recs, d, bad, obs = get(TEST if a.test else MAIN)
    G = len(recs)
    res = dict(groups=G, decisions=len(d), mismatch=int(bad), observed=obs)
    res["loso_logloss"] = dict(C0=C.loso(d, C0), C1=C.loso(d, C1))
    m = C.glmm(d, C0)
    res["C2"] = dict(sigma=m["sigma"], lr=m["lr"], p=float(0.5 * chi2.sf(m["lr"], 1)) if m["lr"] > 0 else 1.0, agents=m["agents"],
                     coef=m["coef"])
    res["adequacy"] = dict(C0=C.group_boot(d, C0, "OWN", 4, G, obs, a.reps, rng),
                           C1=C.group_boot(d, C1, "OWN", 4, G, obs, a.reps, rng),
                           C2=C.group_boot(d, C0, "OWN", 4, G, obs, a.reps, rng, mixed=True))
    res["spread"] = spread(d, C.fit(d, C0), rng)
    ad = res["adequacy"]
    het = res["C2"]["p"] < 0.05 and ad["C2"]["covered"]
    rnd = res["loso_logloss"]["C0"] - res["loso_logloss"]["C1"] >= 0.005 and ad["C1"]["covered"]
    res["reading"] = ("no gap" if ad["C0"]["covered"] else "both" if het and rnd else "agent differences" if het
                      else "round number" if rnd else "neither")
    print(f"== groups {G} decisions {len(d)} mismatch {bad} observed linear {obs:.3f}")
    print("   LOSO log-loss", {k: round(v, 4) for k, v in res["loso_logloss"].items()})
    print(f"   C2 sigma {m['sigma']:.3f} LR {m['lr']:.2f} p {res['C2']['p']:.3g} agents {m['agents']}")
    for k, v in ad.items():
        print(f"   {k}: {v['mean']:.2f} [{v['q05']:.2f}, {v['q95']:.2f}] covered {v['covered']}")
    print("   per-agent LEAD-rate spread", res["spread"])
    print("   reading:", res["reading"])
    if not a.test:
        res["compare"] = {}
        for name, parts in COMPARE.items():
            _, dc, _, oc = get(parts)
            mc = C.glmm(dc, C0)
            res["compare"][name] = dict(observed=oc, sigma=mc["sigma"], lr=mc["lr"],
                                        p=float(0.5 * chi2.sf(mc["lr"], 1)) if mc["lr"] > 0 else 1.0,
                                        spread=spread(dc, C.fit(dc, C0), rng))
            print(f"   compare {name}: sigma {mc['sigma']:.3f} LR {mc['lr']:.2f}", res["compare"][name]["spread"])
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("written", a.out)


if __name__ == "__main__":
    main()
