"""E4 (rewind) evaluation: do re-runs of the same mixed-model group reproduce the same hierarchy?

  chase_e4_eval.py [--pop results/chase_e4_rewind.jsonl] [--out results/chase_e4_eval.json] [--groups 40]

Pre-registered outcomes (section 8, H6): share of linear groups, and share of re-runs reproducing the modal
hierarchy, per arm (OWN, LOG); H6 expects linearity higher under LOG and reproducibility far below 100% and not
higher under LOG. Fixed here before any E4 outcome is read:
- A re-run's hierarchy is its set of final relationships over the six dyads of the fixed members (frozen majority
  rule of chase_metrics.final_relations); an undecided dyad is part of the pattern.
- modal share = share of re-runs whose pattern equals the most frequent pattern in that arm.
- pair-identical = probability that two distinct re-runs of an arm have the same pattern (the analogue of Chase's
  two-meeting design). Random rank orders would give 1/24 = 0.042 among linear groups.
- LOG versus OWN: Fisher exact (two-sided) on linear counts and on modal counts; 95% bootstrap intervals over
  re-runs (seeds resampled jointly in both arms, 4,000 draws) for every share and difference. In bootstrap samples
  the modal pattern is held at the arm's observed one, and pairs of copies of the same re-run are left out of
  pair-identical; both would otherwise be biased upward.
- Descriptive: mean number of members each model dominates at the end (0-3), and how often each model is on top.
Nothing is analysed unless both arms hold all planned re-runs.
"""
import argparse, hashlib, json, sys
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
sys.path.insert(0, str(Path(__file__).parent))
from chase_metrics import events_from_history, final_relations, structure

ROOT = Path(__file__).resolve().parent.parent
SEED0, ARMS = 9000, ("OWN", "LOG")


def describe(rec):
    N, rounds = rec["N"], rec["rounds"]
    ev = events_from_history(rec)
    rel = final_relations(ev, rounds, rounds // 2)
    pattern = tuple(rel.get(frozenset(p), -1) for p in combinations(range(N), 2))
    dom = [sum(1 for v in rel.values() if v == i) for i in range(N)]
    st = structure(ev, N, rounds)
    return dict(pattern=pattern, linear=st["linear"], dom=dom, top=int(np.argmax(dom)) if st["linear"] else -1,
                dd_ds=st["dd_ds"], seq=st["seq"])


def pair_identical(patterns, idx=None):
    """Share of pairs of distinct re-runs with the same pattern. In a bootstrap sample (idx = original index of each
    draw) pairs of copies of one re-run are left out; counting them would bias the share upward."""
    idx = list(range(len(patterns))) if idx is None else list(idx)
    mult = Counter(idx)
    by = Counter()
    for i, m in mult.items():
        by[patterns[idx.index(i)]] += m
    own = sum(m * m for m in mult.values())
    same = sum(k * k for k in by.values()) - own
    return same / (len(idx) ** 2 - own)


def arm_stats(rows, mode=None, idx=None):
    """mode and idx are given for bootstrap samples: the modal pattern is held at the arm's observed one (taking the
    maximum again in each sample would bias the share upward)."""
    pats = [r["pattern"] for r in rows]
    if mode is None:
        mode, k = Counter(pats).most_common(1)[0]
    else:
        k = sum(p == mode for p in pats)
    seq = Counter()
    for r in rows:
        seq.update(r["seq"])
    tot = sum(seq.values())
    return dict(linear=float(np.mean([r["linear"] for r in rows])), modal=k / len(rows), modal_n=k,
                linear_n=int(sum(r["linear"] for r in rows)), n=len(rows), distinct=len(set(pats)),
                pair_identical=pair_identical(pats, idx), modal_pattern=list(mode),
                modal_is_linear=bool(next((r["linear"] for r in rows if r["pattern"] == mode), False)),
                dd_ds=(seq["DD"] + seq["DS"]) / tot if tot else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default=str(ROOT / "results" / "chase_e4_rewind.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "chase_e4_eval.json"))
    ap.add_argument("--groups", type=int, default=40)
    ap.add_argument("--framing", default="comp")
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    recs, members = {arm: {} for arm in ARMS}, None
    for l in open(a.pop):
        r = json.loads(l)
        if r["framing"] == a.framing and r["arm"] in recs and 0 <= r["seed"] - SEED0 < a.groups:
            recs[r["arm"]][r["seed"]] = r
            members = members or r["members"]
            assert r["members"] == members, "membership differs between re-runs"
    have = {arm: len(recs[arm]) for arm in ARMS}
    if any(n < a.groups for n in have.values()):
        sys.exit(f"incomplete: {have} of {a.groups} per arm; nothing analysed")
    seeds = [SEED0 + g for g in range(a.groups)]
    D = {arm: [describe(recs[arm][s]) for s in seeds] for arm in ARMS}
    res = dict(pop_sha256=hashlib.sha256(Path(a.pop).read_bytes()).hexdigest(), framing=a.framing, members=members,
               groups=a.groups, arms={arm: arm_stats(D[arm]) for arm in ARMS})
    for arm in ARMS:
        inv, dec = sum(recs[arm][s]["invalid"] for s in seeds), 2 * sum(len(x["events"]) for s in seeds for x in recs[arm][s]["history"])
        res["arms"][arm].update(invalid=inv, decisions=dec,
                                mean_dominated={m: float(np.mean([r["dom"][i] for r in D[arm]])) for i, m in enumerate(members)},
                                top_share={m: float(np.mean([r["top"] == i for r in D[arm]])) for i, m in enumerate(members)})
    # bootstrap over re-runs, seeds resampled jointly
    rng, keys = np.random.default_rng(20261114), ("linear", "modal", "pair_identical")
    draws = {arm: {k: [] for k in keys} for arm in ARMS}
    for _ in range(a.boot):
        idx = rng.integers(0, a.groups, a.groups)
        for arm in ARMS:
            s = arm_stats([D[arm][i] for i in idx], mode=tuple(res["arms"][arm]["modal_pattern"]), idx=idx)
            for k in keys:
                draws[arm][k].append(s[k])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res["contrast"] = {}
    for k in keys:
        for arm in ARMS:
            res["arms"][arm][k + "_ci95"] = ci(draws[arm][k])
        res["contrast"][k] = dict(diff=res["arms"]["LOG"][k] - res["arms"]["OWN"][k],
                                  ci95=ci(np.array(draws["LOG"][k]) - np.array(draws["OWN"][k])))
    for k in ("linear", "modal"):
        o, l = res["arms"]["OWN"], res["arms"]["LOG"]
        res["contrast"][k]["fisher_p"] = float(fisher_exact([[l[k + "_n"], l["n"] - l[k + "_n"]], [o[k + "_n"], o["n"] - o[k + "_n"]]])[1])
    res["same_modal_pattern_in_both_arms"] = res["arms"]["OWN"]["modal_pattern"] == res["arms"]["LOG"]["modal_pattern"]
    Path(a.out).write_text(json.dumps(res, indent=1))
    short = [m.split(":")[-1] for m in members]
    print("members (index order):", short)
    for arm in ARMS:
        s = res["arms"][arm]
        print(f"{arm:4s} linear {s['linear']:.2f} {s['linear_ci95']}  modal {s['modal']:.2f} ({s['modal_n']}/{s['n']}, linear={s['modal_is_linear']})"
              f" {s['modal_ci95']}  pair-identical {s['pair_identical']:.2f} {s['pair_identical_ci95']}  distinct {s['distinct']}"
              f"  DD+DS {s['dd_ds']:.2f}  invalid {s['invalid']}/{s['decisions']}")
        print("     dominated:", {m: round(v, 2) for m, v in zip(short, s["mean_dominated"].values())},
              " top:", {m: round(v, 2) for m, v in zip(short, s["top_share"].values())})
    for k, c in res["contrast"].items():
        print(f"LOG - OWN {k}: {c['diff']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]" + (f"  Fisher p={c['fisher_p']:.4f}" if "fisher_p" in c else ""))
    print("same modal pattern in both arms:", res["same_modal_pattern_in_both_arms"])
    print("written", a.out)


if __name__ == "__main__":
    main()
