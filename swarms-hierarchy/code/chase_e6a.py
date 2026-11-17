"""E6 part A (amendment C8): which decision model fits the own-record decisions that existing groups made? No calls.

  chase_e6a.py [--reps 100]   -> results/chase_e6a.json

Data: OWN-arm groups in the first 640 lines of results/chase_e3_pop.jsonl (E3 and E5). Extra features are rebuilt
from each stored history with the same `visible` and `features` functions the prompts used; the rebuilt d and o are
checked against the stored ones.
Candidates (logistic on signed-log features, ridge as chase_fit.fit_logistic):
  A0  d, o                              the frozen K2
  A1  A0 + hinge = min(o + 3, 0)        a second slope below -3
  A2  A0 + n_own                        number of own lines shown
  A3  A0 + last, streak                 most recent own outcome (+1, -1, 0); signed length of the current run
  A4  A0 + idlow, idme                  own ID lower than the partner's; own ID / 100
  A5  all of the above
  FLEX  chase_fit.fit_flex on d, o
Criterion: leave-one-seed-out log-loss per decision, groups of four and eight together, in the two selection cells.
Rule: among forms whose log-loss averaged over the two cells is within 0.002 of the lowest, the one with the fewest
added features; selected only if it improves on A0 by at least 0.005 in both cells, otherwise A0.
Adequacy: groups simulated from A0, A5 and the selected form, fitted leaving one seed out, against the observed
linear share (four) and t_tri (eight).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import chase_fit as F
from chase_e2 import RHO, ROUNDS
from chase_pop import KernelDecider, _feat, run
from chase_prompts import features, visible
from chase_metrics import events_from_history, pooled, structure

ROOT = Path(__file__).resolve().parents[1]
SELECT = [("adaption:gemma-4-31b-it", "norm100"), ("adaption:gemma-4-26b-a4b-it", "norm0")]
REPORT = [("adaption:gemma-4-31b-it", "comp"), ("adaption:glm-5.2", "comp")]
FORMS = {"A0": ["d", "o"], "A1": ["d", "o", "hinge"], "A2": ["d", "o", "n_own"], "A3": ["d", "o", "last", "streak"],
         "A4": ["d", "o", "idlow", "idme"], "A5": ["d", "o", "hinge", "n_own", "last", "streak", "idlow", "idme"],
         "FLEX": ["d", "o"]}
ADDED = {k: len(v) - 2 for k, v in FORMS.items()}
ADDED["FLEX"] = 1


def extras(me, other, own, f):
    """Features an agent could read off its own record and the two IDs. `own` is chronological."""
    outs = [1 if w == me else -1 for _, w, _l in own]
    run_len = 0
    for x in reversed(outs):
        if x != outs[-1]:
            break
        run_len += 1
    return dict(hinge=min(f["o"] + 3, 0), last=outs[-1] if outs else 0, streak=(outs[-1] * run_len) if outs else 0,
                idlow=float(me < other), idme=me / 100)


def decisions(rec):
    names, arm, events, rows, bad = rec["names"], rec["arm"], [], [], 0
    for x in rec["history"]:
        t_now = x["t"] + 1
        for k, (i, j, ci, cj, w) in enumerate(x["events"]):
            for side, (a, b, c) in enumerate(((i, j, ci), (j, i, cj))):
                own, partner = visible(arm, names[a], names[b], events, t_now)
                f = features(names[a], names[b], own, partner, arm)
                st = x["feats"][2 * k + side]
                bad += (f["d"], f["o"]) != (st[0], st[1])
                rows.append(dict(d=f["d"], o=f["o"], n_own=f["n_own"], **extras(names[a], names[b], own, f),
                                 lead=float(c == "LEAD"), seed=rec["seed"], off=0.0))
        events += [(t_now, names[e[4]], names[e[1] if e[4] == e[0] else e[0]]) for e in x["events"]]
    return rows, bad


class XDecider(KernelDecider):
    """KernelDecider whose features include the extras."""
    def __call__(self, pairs, names, events, t_now, h, N, rng_prompt, rng_sched, mu, Q, Fq):
        out = []
        for i, j in pairs:
            ps = []
            for a, b in ((i, j), (j, i)):
                f, own, _partner, _sc = _feat(self.arm, a, b, names, events, t_now, h)
                f.update(extras(names[a], names[b], own, f))
                ps.append(self.pfun(f))
            out.append(("LEAD" if rng_sched.random() < ps[0] else "DEFER", "LEAD" if rng_sched.random() < ps[1] else "DEFER"))
        return out, []


def fit(d, form):
    return F.fit_flex(d, FORMS[form]) if form == "FLEX" else F.fit_logistic(d, FORMS[form])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    recs, mismatch = {}, 0
    for n, line in enumerate(open(ROOT / "results" / "chase_e3_pop.jsonl")):
        if n >= 640:
            break
        r = json.loads(line)
        if r["arm"] == "OWN" and (r["model"], r["framing"]) in SELECT + REPORT:
            recs.setdefault((r["model"], r["framing"], r["N"]), []).append(r)
    data = {}
    for k, rs in recs.items():
        rows = []
        for r in rs:
            x, bad = decisions(r)
            rows += x; mismatch += bad
        data[k] = pd.DataFrame(rows)
    print("rebuilt d or o differing from the stored value:", mismatch)
    res = dict(mismatch=int(mismatch), cells={}, forms=FORMS)
    loss = {}
    for cell in SELECT + REPORT:
        tot, cnt = {f: 0.0 for f in FORMS}, 0
        for N in (4, 8):
            d = data.get((*cell, N))
            if d is None:
                continue
            for s in sorted(d.seed.unique()):
                tr, te = d[d.seed != s], d[d.seed == s]
                cnt += len(te)
                for form in FORMS:
                    p = np.clip([fit_p(r) for fit_p in [fit(tr, form)] for r in te.to_dict("records")], 1e-6, 1 - 1e-6)
                    tot[form] += float(-(te.lead.values * np.log(p) + (1 - te.lead.values) * np.log(1 - p)).sum())
        loss[cell] = {f: tot[f] / cnt for f in FORMS}
        res["cells"]["|".join(cell)] = dict(decisions=cnt, loso_logloss=loss[cell])
        print("|".join(cell), cnt, "decisions:", {f: round(v, 4) for f, v in loss[cell].items()})
    avg = {f: float(np.mean([loss[c][f] for c in SELECT])) for f in FORMS}
    near = [f for f in FORMS if avg[f] <= min(avg.values()) + 0.002]
    cand = min(near, key=lambda f: (ADDED[f], avg[f]))
    ok = cand != "A0" and all(loss[c]["A0"] - loss[c][cand] >= 0.005 for c in SELECT)
    selected = cand if ok else "A0"
    res.update(average=avg, within_0002_of_best=near, candidate=cand, selected=selected)
    print("average over selection cells:", {f: round(v, 4) for f, v in avg.items()})
    print("within 0.002 of best:", near, " candidate:", cand, " selected:", selected)
    # adequacy: leave-one-seed-out simulations
    rng = np.random.default_rng(20261116)
    res["adequacy"] = {}
    for cell in SELECT + REPORT:
        for N in (4, 8):
            d, rs = data.get((*cell, N)), recs.get((*cell, N))
            if d is None:
                continue
            obs = pooled([structure(events_from_history(r), N, r["rounds"]) for r in rs])
            key, G, stat = "|".join(cell) + f"|N{N}", len(rs), "linear" if N == 4 else "t_tri"
            out = dict(groups=G, observed=obs[stat])
            seeds = sorted(d.seed.unique())
            for form in dict.fromkeys(["A0", selected, "A5"]):
                kern = {s: fit(d[d.seed != s], form) for s in seeds}
                v = []
                for b in range(a.reps):
                    k = kern[seeds[b % len(seeds)]]
                    st = [structure(events_from_history(run(N, RHO, ROUNDS[N], "OWN", XDecider(k, "OWN"),
                                                            seed=int(rng.integers(1e9)), keep_feats=False)), N, ROUNDS[N]) for _ in range(G)]
                    v.append(pooled(st)[stat])
                out[form] = dict(mean=float(np.mean(v)), q05=float(np.quantile(v, .05)), q95=float(np.quantile(v, .95)),
                                 covered=bool(np.quantile(v, .05) <= obs[stat] <= np.quantile(v, .95)))
            res["adequacy"][key] = out
            print(key, stat, "observed", round(obs[stat], 2), {f: (round(x["mean"], 2), [round(x["q05"], 2), round(x["q95"], 2)], x["covered"])
                                                              for f, x in out.items() if isinstance(x, dict)})
    (ROOT / "results" / "chase_e6a.json").write_text(json.dumps(res, indent=1))
    print("written results/chase_e6a.json")


if __name__ == "__main__":
    main()
