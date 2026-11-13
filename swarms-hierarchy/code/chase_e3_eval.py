"""E3 evaluation: population runs against the frozen E2 forecasts (no LLM calls).

Written and hash-logged on 2026-11-13 while the first E3 cell was running, before any population outcome was read;
tested on synthetic kernel-driven groups only (scratch/e3test/).

A model x framing cell is analysed only when every group of amendment C5 for that cell is present.
For each complete cell:
  * per arm: observed structure (all groups, and the first `groups` seeds the frozen interval was computed for),
    the frozen forecast (top ladder level, LIN, FLEX, and K0 as the kernel-null reference), whether the observed
    value falls in the frozen 5-95% interval, and the invalid-reply rate;
  * LOG versus OWN: Fisher exact on linear counts (groups of four), Mann-Whitney on per-group t_tri (N = 8),
    group-bootstrap interval for the difference, the frozen contrast interval, and the forecast for LOG with the
    bystander features withheld (H4);
  * adequacy: the leave-one-seed-out upper bound of section 7 (top-level kernel refitted on the population's own
    decisions from the other seeds, one simulated group per held-out seed, REPS replicates).
Holm adjustment over the groups-of-four LOG-versus-OWN contrasts of the non-control cells (amendment C5).

  chase_e3_eval.py [--pop results/chase_e3_pop.jsonl] [--out results/chase_e3_eval.json] [--reps 200]
"""
import argparse, hashlib, json, sys, zlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
sys.path.insert(0, str(Path(__file__).parent))
import chase_fit as F
from chase_pop import run, KernelDecider, FEAT_COLS
from chase_metrics import events_from_history, structure, pooled
from chase_e2 import ROUNDS, RHO

ROOT = Path(__file__).resolve().parents[1]
GEMMA, GLM, DEEPSEEK = "adaption:gemma-4-31b-it", "adaption:glm-5.2", "adaption:deepseek-v4-flash"
SEED0 = {4: 4000, 8: 8000}
# groups per arm, amendment C5
TARGET = {
    (GEMMA, "norm100"): {4: {"OWN": 40, "LOG": 40, "DYAD": 20, "REDACT": 20}, 8: {"OWN": 6, "LOG": 6, "SCORE+LOG": 6, "SCORE": 6}},
    (GEMMA, "comp"): {4: {"OWN": 80, "LOG": 80, "DYAD": 20, "REDACT": 20}, 8: {"OWN": 12, "LOG": 12, "SCORE+LOG": 6, "SCORE": 6}},
    (GLM, "comp"): {4: {"OWN": 40, "LOG": 40}},
    (DEEPSEEK, "comp"): {4: {"OWN": 20, "LOG": 20}, 8: {"OWN": 4, "LOG": 4}},
}
CONTROL = {(DEEPSEEK, "comp")}
STAT = {4: "linear", 8: "t_tri"}          # headline statistic by group size
EXTRA = ["dd_ds", "landau", "p_trans"]


def load_pop(path):
    """{(model, framing, arm, N): {seed: record}} restricted to the seeds of amendment C5."""
    recs = defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        tgt = TARGET.get((r["model"], r["framing"]), {}).get(r["N"], {}).get(r["arm"])
        if tgt and 0 <= r["seed"] - SEED0[r["N"]] < tgt:
            recs[(r["model"], r["framing"], r["arm"], r["N"])][r["seed"]] = r
    return recs


def struct(rec):
    return structure(events_from_history(rec), rec["N"], rec["rounds"])


def headline(structs, N):
    """Linear share over all groups (an incomplete group counts as not linear, as in the frozen contrasts) or pooled t_tri."""
    if N == 4:
        return float(np.mean([s["linear"] for s in structs]))
    return float(pooled(structs)["t_tri"])


def qsum(samples, obs):
    v = np.asarray([x for x in samples if x is not None and np.isfinite(x)], float)
    lo, hi = float(np.quantile(v, .05)), float(np.quantile(v, .95))
    return dict(mean=float(v.mean()), q05=lo, q95=hi, covered=bool(lo <= obs <= hi), quantile_of_observed=float(np.mean(v <= obs)))


def forecast(entry, level, N, obs):
    lv = entry[level]
    samples = [x / entry["groups"] for x in lv["linear_n"]] if N == 4 else lv["t_tri"]["samples"]
    out = dict(headline=qsum(samples, obs[STAT[N]]))
    for k in EXTRA:
        if lv[k].get("mean") is not None and np.isfinite(obs[k]):
            out[k] = qsum(lv[k]["samples"], obs[k])
    return out


def decisions(recs):
    """Population decisions as kernel rows: one per agent per contest, features as the agent saw them."""
    rows = []
    for seed, r in recs.items():
        for x in r["history"]:
            for k, ev in enumerate(x["events"]):
                for side in (0, 1):
                    rows.append(dict(zip(FEAT_COLS, x["feats"][2 * k + side]), lead=float(ev[2 + side] == "LEAD"), seed=seed, off=0.0))
    return pd.DataFrame(rows)


def upper_bound(recs, arm, N, reps, key):
    """Leave-one-seed-out: refit the arm's top kernel on the other seeds, simulate `reps` groups for the held-out seed."""
    d = decisions(recs)
    rng = np.random.default_rng([N, 4242, zlib.crc32("|".join(map(str, key)).encode())])
    sims = []                                           # sims[s][r]
    for seed in sorted(recs):
        k = F.fit(d[d.seed != seed], arm, F.ARM_MAX[arm])
        sims.append([structure(events_from_history(run(N, RHO, ROUNDS[N], arm, KernelDecider(k, arm), seed=int(rng.integers(1e9)),
                                                       keep_feats=False)), N, ROUNDS[N]) for _ in range(reps)])
    reps_structs = [[s[r] for s in sims] for r in range(reps)]
    return reps_structs


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, (len(ps) - rank) * ps[i]); adj[i] = min(1.0, run_max)
    return [float(x) for x in adj]


def boot_diff(a, b, N, rng, B=5000):
    """Group-bootstrap 95% interval for headline(b) - headline(a)."""
    out = []
    for _ in range(B):
        ra = [a[i] for i in rng.integers(0, len(a), len(a))]; rb = [b[i] for i in rng.integers(0, len(b), len(b))]
        out.append(headline(rb, N) - headline(ra, N))
    out = np.asarray(out); out = out[np.isfinite(out)]
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default=str(ROOT / "results" / "chase_e3_pop.jsonl"))
    ap.add_argument("--predictions", default=str(ROOT / "results" / "chase_e2_predictions.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "chase_e3_eval.json"))
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    pred = json.loads(Path(a.predictions).read_text())
    fz = {(e["model"], e["framing"], e["arm"], e["N"]): e for e in pred["predictions"]}
    fc = {(c["model"], c["framing"]): c for c in pred["contrasts"]}
    recs = load_pop(a.pop)
    res = dict(predictions_sha256=pred["sha256"], pop_sha256=hashlib.sha256(Path(a.pop).read_bytes()).hexdigest(), cells=[], incomplete=[])
    rng = np.random.default_rng(20261113)

    for (model, framing), plan in TARGET.items():
        have = {(N, arm): len(recs.get((model, framing, arm, N), {})) for N, arms in plan.items() for arm in arms}
        want = {(N, arm): G for N, arms in plan.items() for arm, G in arms.items()}
        if have != want:
            res["incomplete"].append(dict(model=model, framing=framing, have={f"{arm}@{N}": v for (N, arm), v in have.items()},
                                          want={f"{arm}@{N}": v for (N, arm), v in want.items()}))
            print(f"-- {model.split(':')[1]} {framing}: incomplete, not analysed "
                  f"({sum(have.values())} of {sum(want.values())} groups)")
            continue
        cell = dict(model=model, framing=framing, control=(model, framing) in CONTROL, arms=[], contrasts={})
        S = {}
        print(f"\n== {model.split(':')[1]} {framing}")
        for N, arms in plan.items():
            for arm, G in arms.items():
                key = (model, framing, arm, N)
                rs = recs[key]; seeds = sorted(rs); st = [struct(rs[s]) for s in seeds]; S[(arm, N)] = st
                e = fz[key]; gp = e["groups"]                      # size the frozen interval was computed for
                def obs_of(structs):
                    p = pooled(structs)
                    return dict(linear=headline(structs, 4), t_tri=float(p["t_tri"]), dd_ds=float(p["dd_ds"]),
                                landau=float(p["landau"]), p_trans=float(p["p_trans"]),
                                seq={k: float(p[k]) for k in ("DD", "DS", "BDID", "ISDB")}, n_complete=p["n_complete"])
                full, planned = obs_of(st), obs_of(st[:gp])
                top = "LEGACY" if arm == "SCORE" else F.ARM_MAX[arm]
                row = dict(arm=arm, N=N, groups=len(st), groups_planned=gp, observed=full, observed_planned=planned,
                           invalid=int(sum(rs[s]["invalid"] for s in seeds)),
                           decisions=int(sum(2 * len(x["events"]) for s in seeds for x in rs[s]["history"])),
                           forecast={lv: forecast(e, lv, N, planned) for lv in dict.fromkeys([top, "LIN", "FLEX", "K0"]) if lv in e})
                row["top_level"] = top
                if arm != "SCORE":
                    ub = upper_bound(rs, arm, N, a.reps, key)
                    row["upper_bound"] = dict(headline=qsum([headline(x, N) for x in ub], full[STAT[N]]),
                                              dd_ds=qsum([pooled(x)["dd_ds"] for x in ub], full["dd_ds"]))
                cell["arms"].append(row)
                f = row["forecast"][top]["headline"]
                print(f"  {arm:9s} N={N} G={len(st):3d}  {STAT[N]} obs {full[STAT[N]]:+.2f} (planned subset {planned[STAT[N]]:+.2f})"
                      f"  forecast {f['mean']:+.2f} [{f['q05']:+.2f}, {f['q95']:+.2f}] {'in' if f['covered'] else 'OUT'}"
                      + (f"  LOSO {row['upper_bound']['headline']['mean']:+.2f}" if 'upper_bound' in row else "")
                      + f"  DD+DS {full['dd_ds']:.2f}  invalid {row['invalid']}/{row['decisions']}")
        for N in plan:
            if ("OWN", N) not in S or ("LOG", N) not in S:
                continue
            o, l = S[("OWN", N)], S[("LOG", N)]
            gp = fz[(model, framing, "OWN", N)]["groups"]
            c = dict(own=headline(o, N), log=headline(l, N), diff=headline(l, N) - headline(o, N), ci95=boot_diff(o, l, N, rng),
                     diff_planned=headline(l[:gp], N) - headline(o[:gp], N))
            if N == 4:
                x, y = sum(s["linear"] for s in o), sum(s["linear"] for s in l)
                c.update(test="fisher", own_linear_n=int(x), log_linear_n=int(y),
                         p=float(fisher_exact([[x, len(o) - x], [y, len(l) - y]])[1]))
            else:
                xa, ya = np.nan_to_num([s["t_tri"] for s in o]), np.nan_to_num([s["t_tri"] for s in l])
                c.update(test="mannwhitney", p=float(mannwhitneyu(ya, xa, alternative="two-sided").pvalue) if len(set(xa) | set(ya)) > 1 else 1.0)
            fr = fc[(model, framing)].get(f"N{N}")
            if fr:
                d = fr["diff"]
                c["frozen"] = dict(diff=d["mean"], q05=d["q05"], q95=d["q95"], covered=bool(d["q05"] <= c["diff_planned"] <= d["q95"]),
                                   power=fr.get("power_fisher_05", fr.get("power_mannwhitney_05")),
                                   log_without_bystander=fr.get("log_linear_without_bystander", fr.get("log_t_tri_without_bystander")))
            cell["contrasts"][f"N{N}"] = c
            print(f"  LOG - OWN N={N}: {c['diff']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] p={c['p']:.4f}"
                  + (f"  frozen {c['frozen']['diff']:+.2f} [{c['frozen']['q05']:+.2f}, {c['frozen']['q95']:+.2f}] "
                     f"{'in' if c['frozen']['covered'] else 'OUT'} (planned subset {c['diff_planned']:+.2f});"
                     f" LOG without bystander features forecast {c['frozen']['log_without_bystander']:.2f}" if fr else ""))
        res["cells"].append(cell)

    fam = [c for c in res["cells"] if not c["control"] and "N4" in c["contrasts"]]
    if fam:
        for c, p in zip(fam, holm([c["contrasts"]["N4"]["p"] for c in fam])):
            c["contrasts"]["N4"]["p_holm"] = p
            c["contrasts"]["N4"]["holm_family"] = len(fam)
        print("\nHolm (groups of four, LOG vs OWN, cells analysed so far):",
              {f"{c['model'].split(':')[1]}|{c['framing']}": round(c["contrasts"]["N4"]["p_holm"], 4) for c in fam})
    Path(a.out).write_text(json.dumps(res))
    print("written", a.out)


if __name__ == "__main__":
    main()
