"""E11 (amendment C10): is the extra linearity from partner records built when relationships form, or later? No calls.

  chase_e11_eval.py [--boot 5000] [--test] [--out FILE]      -> results/chase_e11_eval.json

Data (all groups of four, `comp`, existing groups only):
  Qwen3.5-397B  E7 (seeds 24000+: OWN 40, LOG 40, REDACT 20) + E10 (44000+: OWN 80, LOG 80, REDACT 80)
  Gemma-4-31B   E3 (4000+: OWN 80, LOG 80) + E9 (14000+: OWN 40, LOG 40)
--test swaps in Qwen3.5-397B `norm100` (E7) for both, writes to the scratch path given by --out, and is the only
mode run before the hash.

Per group:
  initial relationship of a dyad: the winner of its first contest with one LEAD and one DEFER (the E9 rule);
      a dyad with no such contest takes the winner of its first contest; a dyad that never met has none.
  initially linear: all six dyads have an initial relationship and no triad of them is cyclic.
  finally linear: the frozen structure rule (chase_metrics.structure), as in every earlier test.
  reversal: a contest with one LEAD and one DEFER, in a dyad already formed by the E9 rule, won by the agent that
      currently stands below; it flips the current relationship.
Decomposition of the LOG - OWN linear share. Per arm a = share initially linear, s = share finally linear among
initially linear groups, r = the same among the rest. With bars for the mean of the two arms, exactly
  L_LOG - L_OWN = formation + later,  formation = (a_LOG - a_OWN)(s_bar - r_bar),
                                       later = a_bar (s_LOG - s_OWN) + (1 - a_bar)(r_LOG - r_OWN)   (keep + repair).
Intervals: group bootstrap within arm, 95%.
  P1 (Qwen3.5-397B): later - formation > 0, interval excluding zero.
  P2 (Gemma-4-31B): formation - later > 0, interval excluding zero.
  Reading per model: "later" / "formation" if that difference is positive with its interval excluding zero, else "mixed".
Secondary, per model, LOG against OWN (difference of pooled shares, group-bootstrap interval):
  S1 share of reversals in which the new upper agent had the higher net record against third parties at that
     contest (stored o of both agents; ties excluded). Predicted LOG > OWN in Qwen3.5-397B.
Descriptive: reversals per formed dyad (LOG against OWN); keep and repair terms separately; among triads whose three relationships
all formed by the E9 rule, the share whose initial relationships are transitive; the same for Qwen REDACT.
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from chase_c10 import G31, QWEN, R, load
import chase_e3_eval as V

SETS = {
    "qwen": [("chase_e7_pop.jsonl", QWEN, "comp", 24000, {"OWN": 40, "LOG": 40, "REDACT": 20}),
             ("chase_e10_pop.jsonl", QWEN, "comp", 44000, {"OWN": 80, "LOG": 80, "REDACT": 80})],
    "gemma31": [("chase_e3_pop.jsonl", G31, "comp", 4000, {"OWN": 80, "LOG": 80}),
                ("chase_e9_pop.jsonl", G31, "comp", 14000, {"OWN": 40, "LOG": 40})],
}
TEST = {k: [("chase_e7_pop.jsonl", QWEN, "norm100", 24000, {"OWN": 40, "LOG": 40})] for k in SETS}
TRIADS = list(itertools.combinations(range(4), 3))


def cyclic(rel, tri):
    """rel: {frozenset(i, j): winner}. A triad with all three relationships is cyclic if every agent wins once."""
    ks = [frozenset(p) for p in itertools.combinations(tri, 2)]
    if not all(k in rel for k in ks):
        return False
    return len({rel[k] for k in ks}) == 3


def n_cyclic(rel):
    return sum(cyclic(rel, t) for t in TRIADS)


def trace(rec):
    first_any, init, cur, revs = {}, {}, {}, []
    for x in rec["history"]:
        for k, (i, j, ci, cj, w) in enumerate(x["events"]):
            key = frozenset((i, j))
            first_any.setdefault(key, w)
            if ci == cj:
                continue
            if key not in cur:
                cur[key] = w; init[key] = w
                continue
            if cur[key] != w:
                f = x["feats"]
                o_w = f[2 * k][1] if w == i else f[2 * k + 1][1]
                o_l = f[2 * k + 1][1] if w == i else f[2 * k][1]
                cur[key] = w
                revs.append(dict(t=x["t"], toward_better=None if o_w == o_l else bool(o_w > o_l)))
    rel0 = {k: init.get(k, first_any[k]) for k in first_any}
    init_lin = len(rel0) == 6 and n_cyclic(rel0) == 0
    tri_all = [t for t in TRIADS if all(frozenset(p) in init for p in itertools.combinations(t, 2))]
    return dict(init_linear=bool(init_lin), final_linear=bool(V.struct(rec)["linear"]), formed=len(init), revs=revs,
                tri_closed=len(tri_all), tri_trans=sum(not cyclic(init, t) for t in tri_all))


def decompose(own, log):
    def arm(g):
        a = np.mean([x["init_linear"] for x in g])
        s = [x["final_linear"] for x in g if x["init_linear"]]
        r = [x["final_linear"] for x in g if not x["init_linear"]]
        return a, (np.mean(s) if s else np.nan), (np.mean(r) if r else np.nan), np.mean([x["final_linear"] for x in g])
    (a0, s0, r0, L0), (a1, s1, r1, L1) = arm(own), arm(log)
    ab, sb, rb = (a0 + a1) / 2, (s0 + s1) / 2, (r0 + r1) / 2
    formation, keep, repair = (a1 - a0) * (sb - rb), ab * (s1 - s0), (1 - ab) * (r1 - r0)
    return dict(diff=L1 - L0, formation=formation, keep=keep, repair=repair, later=keep + repair,
                later_minus_formation=keep + repair - formation, own=dict(a=a0, s=s0, r=r0, L=L0), log=dict(a=a1, s=s1, r=r1, L=L1))


def shares(g):
    rv = [v for x in g for v in x["revs"]]
    tb = [v["toward_better"] for v in rv if v["toward_better"] is not None]
    return dict(S1=np.mean(tb) if tb else np.nan,
                rev_per_dyad=len(rv) / max(1, sum(x["formed"] for x in g)),
                init_trans=sum(x["tri_trans"] for x in g) / max(1, sum(x["tri_closed"] for x in g)),
                n_rev=len(rv), n_S1=len(tb))


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def analyse(tr, B, rng):
    own, log = tr["OWN"], tr["LOG"]
    dec = decompose(own, log)
    sh = {arm: shares(tr[arm]) for arm in tr}
    bd, bs = {k: [] for k in ("formation", "keep", "repair", "later", "later_minus_formation", "diff")}, {"S1": []}
    for _ in range(B):
        o = [own[i] for i in rng.integers(0, len(own), len(own))]
        l = [log[i] for i in rng.integers(0, len(log), len(log))]
        d = decompose(o, l)
        for k in bd:
            bd[k].append(d[k])
        so, sl = shares(o), shares(l)
        for k in bs:
            bs[k].append(sl[k] - so[k])
    out = dict(decomposition={k: (float(v) if not isinstance(v, dict) else {a: float(b) for a, b in v.items()}) for k, v in dec.items()},
               ci95={k: ci(v) for k, v in bd.items()}, arms={a: {k: float(v) for k, v in s.items()} for a, s in sh.items()})
    lo, hi = out["ci95"]["later_minus_formation"]
    out["reading"] = "later" if lo > 0 else "formation" if hi < 0 else "mixed"
    out["S"] = {k: dict(diff=float(sh["LOG"][k] - sh["OWN"][k]), ci95=ci(bs[k])) for k in bs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out", default=str(R / "chase_e11_eval.json"))
    a = ap.parse_args()
    sets = TEST if a.test else SETS
    rng, res = np.random.default_rng(20261120), {}
    for name, parts in sets.items():
        tr = {}
        for fname, model, framing, seed0, plan in parts:
            for arm, recs in load(fname, model, framing, 4, seed0, plan).items():
                tr.setdefault(arm, []).extend(trace(r) for r in recs)
        res[name] = dict(groups={k: len(v) for k, v in tr.items()}, **analyse(tr, a.boot, rng))
        x = res[name]
        print(f"== {name}: groups {x['groups']}  LOG-OWN {x['decomposition']['diff']:+.2f} = formation "
              f"{x['decomposition']['formation']:+.3f} + later {x['decomposition']['later']:+.3f} "
              f"(keep {x['decomposition']['keep']:+.3f}, repair {x['decomposition']['repair']:+.3f}); "
              f"later - formation {x['decomposition']['later_minus_formation']:+.3f} {np.round(x['ci95']['later_minus_formation'], 3).tolist()} "
              f"-> {x['reading']}")
        for k in ("S1",):
            print(f"   {k} LOG-OWN {x['S'][k]['diff']:+.3f} {np.round(x['S'][k]['ci95'], 3).tolist()}  "
                  + "  ".join(f"{arm} {s[k]:.3f} (n {int(s['n_' + k])})" for arm, s in x["arms"].items()))
        print("   reversals per formed dyad", {arm: round(s["rev_per_dyad"], 3) for arm, s in x["arms"].items()},
              " initial triads transitive", {arm: round(s["init_trans"], 3) for arm, s in x["arms"].items()})
    if not a.test:
        res["P1_qwen_later"] = res["qwen"]["reading"] == "later"
        res["P2_gemma31_formation"] = res["gemma31"]["reading"] == "formation"
        print("P1 (Qwen: later) supported:", res["P1_qwen_later"], " P2 (Gemma-31B: formation) supported:", res["P2_gemma31_formation"])
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("written", a.out)


if __name__ == "__main__":
    main()
