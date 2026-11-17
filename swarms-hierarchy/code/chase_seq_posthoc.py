"""POST HOC (written 2026-11-16, after all data were seen and after reading Chase 1982, Behaviour 80: 218-240).

Re-counts Chase's four triad sequences under formation rules closer to his than the pre-registered one. No model
calls; reads the population files only.

  chase_seq_posthoc.py   -> results/chase_seq_posthoc.json

Chase's criterion: one hen dominates another once she has delivered three aggressive acts and the receiver has not
attacked back for 30 minutes. The pre-registered rule here ("first": a relationship forms at the dyad's first
contest) counts coin-flip outcomes as relationships. Two alternatives, fixed before this script was first run:
  decided  a relationship forms at the dyad's first contest settled by the answers (one LEAD, one DEFER): the
           loser yielded. Coin-flip contests are ignored.
  three    a relationship forms when one agent has won three contests in a row against the other, coin flips
           included (the nearest analogue of "three acts and no counter-attack").
For each rule: shares of DD, DS, BDID, ISDB over triads with at least two formed relationships, the number of such
triads out of all triads, and a 95% group-bootstrap interval for DD + DS (2,000 draws, seed fixed).
Also, by the frozen final-relationship rule: the share of complete triads that are transitive, for comparison with
Chase's real (54 of 54) and constructed (54 of 63) triads; chance 0.75.
"""
import collections, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from chase_metrics import SEQ, final_relations, triad_census

ROOT = Path(__file__).resolve().parents[1]
RULES = ("first", "decided", "three")


def formed(rec, rule):
    """[(winner, loser)] in order of formation, one entry per dyad."""
    out, seen, run = [], set(), {}
    for h in rec["history"]:
        for i, j, ci, cj, w in h["events"]:
            k, l = frozenset((i, j)), (j if w == i else i)
            if k in seen:
                continue
            if rule == "first" or (rule == "decided" and ci != cj):
                seen.add(k); out.append((w, l))
            elif rule == "three":
                who, n = run.get(k, (None, 0))
                run[k] = (w, n + 1) if who == w else (w, 1)
                if run[k][1] == 3:
                    seen.add(k); out.append((w, l))
    return out


def sequences(rec, rule):
    first = {frozenset(p): (n, p[0], p[1]) for n, p in enumerate(formed(rec, rule))}
    seq = collections.Counter()
    for tri in itertools.combinations(range(rec["N"]), 3):
        ds = sorted(first[frozenset(p)] for p in itertools.combinations(tri, 2) if frozenset(p) in first)
        if len(ds) >= 2:
            (_, w1, l1), (_, w2, l2) = ds[0], ds[1]
            seq["DD" if w2 == w1 else "DS" if l2 == l1 else "BDID" if l2 == w1 else "ISDB"] += 1
    return seq


def census(rec):
    ev = [(h["t"], e[4], e[1] if e[4] == e[0] else e[0]) for h in rec["history"] for e in h["events"]]
    return triad_census(final_relations(ev, rec["rounds"], rec["rounds"] // 2), rec["N"])


def main():
    groups = collections.defaultdict(list)
    for name in ("chase_e3_pop.jsonl", "chase_e4_rewind.jsonl"):
        for line in open(ROOT / "results" / name):
            r = json.loads(line)
            model = "mixed (E4)" if "members" in r else r["model"].split(":")[-1]
            groups[(model, r["framing"], r["arm"], r["N"])].append(r)
    rng, res = np.random.default_rng(20261116), []
    for (model, framing, arm, N), recs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][3], x[0][2])):
        row = dict(model=model, framing=framing, arm=arm, N=N, groups=len(recs), rules={})
        n_tri = len(recs) * len(list(itertools.combinations(range(N), 3)))
        for rule in RULES:
            per = [sequences(r, rule) for r in recs]
            tot = sum(per, collections.Counter()); n = sum(tot.values())
            mat = np.array([[c[k] for k in SEQ] for c in per], float)
            boot = []
            for _ in range(2000):
                s = mat[rng.integers(0, len(recs), len(recs))].sum(0)
                if s.sum():
                    boot.append((s[0] + s[1]) / s.sum())
            row["rules"][rule] = dict(triads=n, triads_possible=n_tri, **{k: (tot[k] / n if n else None) for k in SEQ},
                                      dd_ds=((tot["DD"] + tot["DS"]) / n if n else None),
                                      dd_ds_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None)
        tr, cy, inc = map(sum, zip(*(census(r) for r in recs)))
        row["final"] = dict(transitive=tr, cyclic=cy, incomplete=inc, p_trans=tr / (tr + cy) if tr + cy else None)
        res.append(row)
    (ROOT / "results" / "chase_seq_posthoc.json").write_text(json.dumps(dict(label="POST HOC", rules=RULES, cells=res), indent=1))
    f = lambda x: "  -- " if x is None else f"{x:5.2f}"
    for row in res:
        print(f"{row['model'][:20]:20s} {row['framing']:8s} {row['arm']:9s} N{row['N']} G={row['groups']:3d}  transitive {row['final']['transitive']}/{row['final']['transitive'] + row['final']['cyclic']} = {f(row['final']['p_trans'])}")
        for rule in RULES:
            s = row["rules"][rule]
            ci = s["dd_ds_ci95"]
            print(f"    {rule:8s} triads {s['triads']:4d}/{s['triads_possible']:4d}  DD {f(s['DD'])} DS {f(s['DS'])} BDID {f(s['BDID'])} ISDB {f(s['ISDB'])}  DD+DS {f(s['dd_ds'])}"
                  + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""))


if __name__ == "__main__":
    main()
