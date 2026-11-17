"""E9 evaluation (amendment C8): the registered sequence test on fresh Gemma-31B groups of four.

  chase_e9_eval.py [--pop results/chase_e9_pop.jsonl] [--out results/chase_e9_eval.json]

Formation rule, fixed: a relationship forms at the pair's first contest in which one agent leads and the other defers
("decided" in chase_seq_posthoc.py, imported unchanged). Unit of analysis: the group. A group's share is taken over
its triads with two formed relationships; a group with no such triad is left out and counted.

Primary predictions (two-sided Mann-Whitney on per-group shares, Holm over the four; supported if adjusted p < 0.05
and the direction is as predicted):
  P1 comp     DD + DS share: LOG > OWN
  P2 comp     DD share:      LOG > OWN
  P3 norm100  DD share:      LOG > OWN
  P4 norm100  DS share:      OWN > LOG
P5 (criterion, no test): pooled DD + DS in the DYAD arm lies in [0.40, 0.60] in each framing.
Estimate only: norm100 DD + DS, LOG - OWN.
Secondary: S1 triads formed by DD or DS end transitive more often than triads formed by BDID or ISDB (pooled over
arms within a cell, final relationships by the frozen majority rule, group-bootstrap interval); S2 DD + DS as a share
of all triads (unformed triads counted against); S3 linear share, LOG versus OWN, Fisher exact, unadjusted.
All intervals: 95%, group bootstrap, 4,000 draws, seeds resampled jointly for OWN and LOG.
Nothing is analysed unless every planned group is present.
"""
import argparse, collections, hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
sys.path.insert(0, str(Path(__file__).parent))
from chase_metrics import events_from_history, final_relations, structure
from chase_seq_posthoc import formed

ROOT = Path(__file__).resolve().parents[1]
MODEL, SEED0 = "adaption:gemma-4-31b-it", 14000
PLAN = {"OWN": 40, "LOG": 40, "DYAD": 20}
FRAMINGS = ("comp", "norm100")
SEQ = ("DD", "DS", "BDID", "ISDB")


def describe(rec):
    """Per group: counts of each sequence type, triads possible, linear, and [type, transitive?] per classified triad."""
    N, rounds = rec["N"], rec["rounds"]
    first = {frozenset(p): (n, p[0], p[1]) for n, p in enumerate(formed(rec, "decided"))}
    ev = events_from_history(rec)
    rel = final_relations(ev, rounds, rounds // 2)
    cnt, tri_out = collections.Counter(), []
    for tri in itertools.combinations(range(N), 3):
        pairs = [frozenset(p) for p in itertools.combinations(tri, 2)]
        ds = sorted(first[k] for k in pairs if k in first)
        if len(ds) < 2:
            continue
        (_, w1, l1), (_, w2, l2) = ds[0], ds[1]
        typ = "DD" if w2 == w1 else "DS" if l2 == l1 else "BDID" if l2 == w1 else "ISDB"
        cnt[typ] += 1
        if all(k in rel for k in pairs):
            tri_out.append((typ in ("DD", "DS"), len({rel[k] for k in pairs}) < 3))
    return dict(cnt=[cnt[k] for k in SEQ], possible=len(list(itertools.combinations(range(N), 3))),
                linear=bool(structure(ev, N, rounds)["linear"]), tri=tri_out)


def share(groups, num):
    """Per-group shares of the sequence types in `num` among classified triads (groups without any are dropped)."""
    out = []
    for g in groups:
        n = sum(g["cnt"])
        if n:
            out.append(sum(g["cnt"][SEQ.index(k)] for k in num) / n)
    return np.array(out)


def pooled(groups, num, all_triads=False):
    n = sum(g["possible"] if all_triads else sum(g["cnt"]) for g in groups)
    return sum(g["cnt"][SEQ.index(k)] for g in groups for k in num) / n if n else float("nan")


def trans_gap(groups):
    a = [t for g in groups for (good, t) in g["tri"] if good]
    b = [t for g in groups for (good, t) in g["tri"] if not good]
    return (np.mean(a) if a else float("nan")), (np.mean(b) if b else float("nan")), len(a), len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default=str(ROOT / "results" / "chase_e9_pop.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "chase_e9_eval.json"))
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    recs = {(f, arm): {} for f in FRAMINGS for arm in PLAN}
    for l in open(a.pop):
        r = json.loads(l)
        k = (r["framing"], r["arm"])
        if r["model"] == MODEL and r["N"] == 4 and k in recs and 0 <= r["seed"] - SEED0 < PLAN[r["arm"]]:
            recs[k][r["seed"]] = r
    have = {f"{f}|{arm}": len(v) for (f, arm), v in recs.items()}
    if any(len(v) < PLAN[arm] for (f, arm), v in recs.items()):
        sys.exit(f"incomplete: {have}; nothing analysed")
    D = {k: [describe(v[SEED0 + g]) for g in range(PLAN[k[1]])] for k, v in recs.items()}
    rng = np.random.default_rng(20261116)
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    res = dict(pop_sha256=hashlib.sha256(Path(a.pop).read_bytes()).hexdigest(), have=have, cells={})
    tests = {}
    for f in FRAMINGS:
        own, log, dyad = D[(f, "OWN")], D[(f, "LOG")], D[(f, "DYAD")]
        cell = dict(arms={}, contrasts={})
        for arm, G in (("OWN", own), ("LOG", log), ("DYAD", dyad)):
            boot = [[G[i] for i in rng.integers(0, len(G), len(G))] for _ in range(a.boot)]
            cell["arms"][arm] = dict(
                groups=len(G), groups_with_triads=int(len(share(G, SEQ))), triads=int(sum(sum(g["cnt"]) for g in G)),
                triads_possible=int(sum(g["possible"] for g in G)),
                **{k: pooled(G, [k]) for k in SEQ}, dd_ds=pooled(G, ["DD", "DS"]),
                dd_ds_ci95=ci([pooled(b, ["DD", "DS"]) for b in boot]),
                dd_ds_of_all_triads=pooled(G, ["DD", "DS"], all_triads=True),
                linear=float(np.mean([g["linear"] for g in G])), linear_n=int(sum(g["linear"] for g in G)),
                invalid=int(sum(r["invalid"] for r in recs[(f, arm)].values())))
        idx = [rng.integers(0, 40, 40) for _ in range(a.boot)]
        for name, num in (("dd_ds", ["DD", "DS"]), ("dd", ["DD"]), ("ds", ["DS"])):
            x, y = share(own, num), share(log, num)
            c = dict(own_mean=float(x.mean()), log_mean=float(y.mean()), diff=float(y.mean() - x.mean()),
                     mw_p=float(mannwhitneyu(x, y).pvalue),
                     pooled_diff=pooled(log, num) - pooled(own, num),
                     pooled_diff_ci95=ci([pooled([log[i] for i in ix], num) - pooled([own[i] for i in ix], num) for ix in idx]))
            cell["contrasts"][name] = c
        cell["contrasts"]["dd_ds_of_all_triads_diff"] = pooled(log, ["DD", "DS"], True) - pooled(own, ["DD", "DS"], True)
        o, l = cell["arms"]["OWN"], cell["arms"]["LOG"]
        cell["contrasts"]["linear"] = dict(diff=l["linear"] - o["linear"],
                                           fisher_p=float(fisher_exact([[l["linear_n"], 40 - l["linear_n"]], [o["linear_n"], 40 - o["linear_n"]]])[1]),
                                           ci95=ci([np.mean([log[i]["linear"] for i in ix]) - np.mean([own[i]["linear"] for i in ix]) for ix in idx]))
        allg = own + log + dyad
        ta, tb, na, nb = trans_gap(allg)
        gaps = []
        for _ in range(a.boot):
            b = [allg[i] for i in rng.integers(0, len(allg), len(allg))]
            x = trans_gap(b)
            gaps.append(x[0] - x[1])
        cell["S1"] = dict(transitive_after_dd_ds=float(ta), transitive_after_other=float(tb), n_dd_ds=na, n_other=nb,
                          diff=float(ta - tb), ci95=ci(gaps))
        cell["P5_dyad_in_band"] = bool(0.40 <= cell["arms"]["DYAD"]["dd_ds"] <= 0.60)
        res["cells"][f] = cell
    C = res["cells"]
    tests = {"P1 comp DD+DS LOG>OWN": (C["comp"]["contrasts"]["dd_ds"], +1), "P2 comp DD LOG>OWN": (C["comp"]["contrasts"]["dd"], +1),
             "P3 norm100 DD LOG>OWN": (C["norm100"]["contrasts"]["dd"], +1), "P4 norm100 DS OWN>LOG": (C["norm100"]["contrasts"]["ds"], -1)}
    order = sorted(tests, key=lambda k: tests[k][0]["mw_p"])
    adj, run_max = {}, 0.0
    for rank, k in enumerate(order):
        run_max = max(run_max, min(1.0, (len(order) - rank) * tests[k][0]["mw_p"]))
        adj[k] = run_max
    res["primary"] = {k: dict(diff=c["diff"], mw_p=c["mw_p"], holm_p=adj[k], direction_ok=bool(np.sign(c["diff"]) == s),
                              supported=bool(adj[k] < 0.05 and np.sign(c["diff"]) == s)) for k, (c, s) in tests.items()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for f in FRAMINGS:
        print(f"== {f}")
        for arm, s in C[f]["arms"].items():
            print(f"  {arm:5s} G={s['groups']} (with triads {s['groups_with_triads']})  triads {s['triads']}/{s['triads_possible']}  DD {s['DD']:.2f} DS {s['DS']:.2f} "
                  f"BDID {s['BDID']:.2f} ISDB {s['ISDB']:.2f}  DD+DS {s['dd_ds']:.2f} {[round(x, 2) for x in s['dd_ds_ci95']]}  of all triads {s['dd_ds_of_all_triads']:.2f}  "
                  f"linear {s['linear']:.2f}  invalid {s['invalid']}")
        for k, c in C[f]["contrasts"].items():
            if isinstance(c, dict) and "mw_p" in c:
                print(f"  LOG - OWN {k}: per-group {c['diff']:+.2f} (MW p={c['mw_p']:.4g}); pooled {c['pooled_diff']:+.2f} {[round(x, 2) for x in c['pooled_diff_ci95']]}")
        c = C[f]["contrasts"]["linear"]
        print(f"  LOG - OWN linear: {c['diff']:+.2f} {[round(x, 2) for x in c['ci95']]} Fisher p={c['fisher_p']:.4f}")
        s = C[f]["S1"]
        print(f"  S1 transitive after DD/DS {s['transitive_after_dd_ds']:.2f} (n={s['n_dd_ds']}), after BDID/ISDB {s['transitive_after_other']:.2f} (n={s['n_other']}), "
              f"diff {s['diff']:+.2f} {[round(x, 2) for x in s['ci95']]};  P5 DYAD in band: {C[f]['P5_dyad_in_band']}")
    for k, v in res["primary"].items():
        print(f"{k}: diff {v['diff']:+.2f}  MW p={v['mw_p']:.4g}  Holm {v['holm_p']:.4g}  supported={v['supported']}")
    print("written", a.out)


if __name__ == "__main__":
    main()
