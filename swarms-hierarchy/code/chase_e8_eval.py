"""E8 evaluation (amendment C8): equal turns (`norm0`), fresh groups of four.

  chase_e8_eval.py [--boot 5000] [--pop FILE] [--out FILE]   -> results/chase_e8_eval.json

Written and hashed in CHASE_LOG.md after the E8 runs finished and before any E8 outcome was read.
Data: results/chase_e8_pop.jsonl. E8a Gemma-4-26B `norm0`, OWN and LOG, seeds 34000-34139; E8b Qwen3.5-397B
`norm0`, OWN and LOG, seeds 34000-34079. Nothing is analysed unless every planned group is present.

Per cell (all registered in C8 unless marked descriptive):
  H5 reading of amendment C6 item 6 on the fresh groups only. LOG - OWN linear share with the group-bootstrap 95%
     interval of chase_e3_eval.boot_diff (unchanged) and Fisher exact p:
       "reverses"     diff < 0 and the interval excludes zero;
       "contradicted" diff > 0 and p < 0.05;
       "vanishes"     neither, and the interval excludes +0.17 (upper bound below 0.17);
       otherwise      "unresolved", reported as the interval.
  Decided-contest share: per group, the share of contests with one LEAD and one DEFER. E8a co-primary: LOG < OWN,
     two-sided Mann-Whitney, supported if p < 0.05 with that direction. E8b: estimate only.
  E8a secondary: fresh + E5 groups pooled (Gemma-26B `norm0` N4 OWN and LOG, seeds 4000-4039 in
     results/chase_e3_pop.jsonl), 180 per arm, same reading, descriptive; the first 40 fresh groups per arm
     (seeds 34000-34039) against the frozen E5 forecast (top kernel level, 90% interval) and its contrast interval.
  E8a descriptive: the first 40 fresh OWN groups against the E6 part B forecast (form X, 40 groups) and the
     E6 part A leave-one-seed-out interval 0.34 [0.22, 0.47].
  E8b: both arms and the contrast against the frozen E8 forecast (form X, 80 groups); LOG forecast without bystander
     features (form Xnb).
  Both, descriptive: DD + DS share by the E9 rule ("decided"), LEAD rate per arm.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
sys.path.insert(0, str(Path(__file__).parent))
import chase_e3_eval as V
import chase_fit as F
from chase_seq_posthoc import sequences

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
G26, QWEN = "adaption:gemma-4-26b-a4b-it", "adaption:qwen3.5-397b"
CELLS = {"E8a": (G26, 140), "E8b": (QWEN, 80)}
SEED0 = 34000
LOSO_E6A = (0.34, 0.22, 0.47)


def load(path, model, seed0, G):
    out = {"OWN": {}, "LOG": {}}
    for l in open(path):
        r = json.loads(l)
        if r["model"] == model and r["framing"] == "norm0" and r["N"] == 4 and r["arm"] in out and 0 <= r["seed"] - seed0 < G:
            out[r["arm"]][r["seed"]] = r
    return out


def decided(rec):
    ev = [e for h in rec["history"] for e in h["events"]]
    return sum(e[2] != e[3] for e in ev) / len(ev)


def lead_rate(recs):
    ev = [e for r in recs for h in r["history"] for e in h["events"]]
    return sum((e[2] == "LEAD") + (e[3] == "LEAD") for e in ev) / (2 * len(ev))


def dd_ds(recs):
    tot = sum((sequences(r, "decided") for r in recs), __import__("collections").Counter())
    n = sum(tot.values())
    return (tot["DD"] + tot["DS"]) / n if n else float("nan")


def reading(own, log, rng, B):
    so, sl = [V.struct(r) for r in own], [V.struct(r) for r in log]
    xo, xl = sum(s["linear"] for s in so), sum(s["linear"] for s in sl)
    d = V.headline(sl, 4) - V.headline(so, 4)
    ci = V.boot_diff(so, sl, 4, rng, B)
    p = float(fisher_exact([[xl, len(sl) - xl], [xo, len(so) - xo]])[1])
    verdict = ("reverses" if d < 0 and ci[1] < 0 else "contradicted" if d > 0 and p < 0.05
               else "vanishes" if ci[1] < 0.17 else "unresolved")
    return dict(groups=[len(so), len(sl)], own=V.headline(so, 4), log=V.headline(sl, 4), own_n=int(xo), log_n=int(xl),
                diff=d, ci95=ci, fisher_p=p, verdict=verdict)


def q(samples, obs):
    v = np.asarray(samples, float)
    lo, hi = float(np.quantile(v, .05)), float(np.quantile(v, .95))
    return dict(observed=obs, mean=float(v.mean()), q05=lo, q95=hi, covered=bool(lo <= obs <= hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--pop", default=str(R / "chase_e8_pop.jsonl"))
    ap.add_argument("--out", default=str(R / "chase_e8_eval.json"))
    a = ap.parse_args()
    pop = Path(a.pop)
    data = {k: load(pop, m, SEED0, G) for k, (m, G) in CELLS.items()}
    have = {k: {arm: len(v) for arm, v in d.items()} for k, d in data.items()}
    if any(have[k][arm] < CELLS[k][1] for k in CELLS for arm in ("OWN", "LOG")):
        sys.exit(f"incomplete: {have}; nothing analysed")
    rng = np.random.default_rng(20261118)
    res = dict(pop_sha256=hashlib.sha256(pop.read_bytes()).hexdigest(), have=have, cells={})
    for k, (model, G) in CELLS.items():
        own = [data[k]["OWN"][SEED0 + g] for g in range(G)]
        log = [data[k]["LOG"][SEED0 + g] for g in range(G)]
        c = dict(model=model, H5=reading(own, log, rng, a.boot))
        do, dl = np.array([decided(r) for r in own]), np.array([decided(r) for r in log])
        c["decided"] = dict(own=float(do.mean()), log=float(dl.mean()), diff=float(dl.mean() - do.mean()),
                            mw_p=float(mannwhitneyu(do, dl).pvalue))
        if k == "E8a":
            c["decided"]["supported"] = bool(c["decided"]["mw_p"] < 0.05 and c["decided"]["diff"] < 0)
        c["descriptive"] = dict(lead_rate=dict(OWN=lead_rate(own), LOG=lead_rate(log)), dd_ds=dict(OWN=dd_ds(own), LOG=dd_ds(log)),
                                invalid=dict(OWN=int(sum(r["invalid"] for r in own)), LOG=int(sum(r["invalid"] for r in log))))
        if k == "E8a":
            old = load(R / "chase_e3_pop.jsonl", model, 4000, 40)
            assert len(old["OWN"]) == len(old["LOG"]) == 40
            c["pooled_with_E5"] = reading(own + list(old["OWN"].values()), log + list(old["LOG"].values()), rng, a.boot)
            p5 = json.loads((R / "chase_e5_predictions.json").read_text())
            first = {"OWN": own[:40], "LOG": log[:40]}
            obs40 = {arm: V.headline([V.struct(r) for r in first[arm]], 4) for arm in first}
            c["first40_vs_E5_forecast"] = {}
            for e in p5["predictions"]:
                if e["model"] == model and e["framing"] == "norm0" and e["N"] == 4 and e["arm"] in first:
                    top = F.ARM_MAX[e["arm"]]
                    c["first40_vs_E5_forecast"][e["arm"]] = dict(level=top, **q([x / e["groups"] for x in e[top]["linear_n"]], obs40[e["arm"]]))
            fc = [x for x in p5["contrasts"] if x["model"] == model and x["framing"] == "norm0"][0]["N4"]["diff"]
            c["first40_vs_E5_forecast"]["diff"] = q(fc["samples"], obs40["LOG"] - obs40["OWN"])
            p6 = json.loads((R / "chase_e6_predictions.json").read_text())
            e6 = [e for e in p6["predictions"] if e["model"] == model and e["framing"] == "norm0" and e["N"] == 4 and e["arm"] == "OWN"][0]
            h = e6["X"]["40"]["headline"]
            c["first40_OWN_vs_E6"] = dict(observed=obs40["OWN"], E6B_X=dict(mean=h["mean"], q05=h["q05"], q95=h["q95"],
                                          covered=bool(h["q05"] <= obs40["OWN"] <= h["q95"])),
                                          E6A_loso=dict(mean=LOSO_E6A[0], q05=LOSO_E6A[1], q95=LOSO_E6A[2],
                                                        covered=bool(LOSO_E6A[1] <= obs40["OWN"] <= LOSO_E6A[2])))
            c["all140_OWN"] = c["H5"]["own"]
        else:
            p8 = json.loads((R / "chase_e8_predictions.json").read_text())
            c["vs_E8_forecast"] = {}
            for e in p8["predictions"]:
                obs = c["H5"]["own" if e["arm"] == "OWN" else "log"]
                for form in e["forms"]:
                    h = e[form][str(G)]["headline"]
                    c["vs_E8_forecast"][f"{e['arm']}|{form}"] = dict(observed=obs, mean=h["mean"], q05=h["q05"], q95=h["q95"],
                                                                     covered=bool(h["q05"] <= obs <= h["q95"]))
            d = p8["contrasts"][0]["N4"][str(G)]["diff"]
            c["vs_E8_forecast"]["diff"] = dict(observed=c["H5"]["diff"], mean=d["mean"], q05=d["q05"], q95=d["q95"],
                                               covered=bool(d["q05"] <= c["H5"]["diff"] <= d["q95"]))
        res["cells"][k] = c
    Path(a.out).write_text(json.dumps(res, indent=1))
    rd = lambda x: {k_: rd(v) for k_, v in x.items()} if isinstance(x, dict) else [rd(v) for v in x] if isinstance(x, list) else round(x, 3) if isinstance(x, float) else x
    for k, c in res["cells"].items():
        h = c["H5"]
        print(f"== {k} {c['model']}  OWN {h['own']:.2f} ({h['own_n']}/{h['groups'][0]})  LOG {h['log']:.2f} ({h['log_n']}/{h['groups'][1]})  "
              f"diff {h['diff']:+.2f} [{h['ci95'][0]:+.2f}, {h['ci95'][1]:+.2f}]  Fisher p={h['fisher_p']:.3f}  H5: {h['verdict']}")
        d = c["decided"]
        print(f"   decided share OWN {d['own']:.2f} LOG {d['log']:.2f} diff {d['diff']:+.3f} MW p={d['mw_p']:.3g}" + (f" supported={d['supported']}" if "supported" in d else " (estimate)"))
        print("   descriptive", json.dumps({kk: {a_: round(v, 3) for a_, v in vv.items()} for kk, vv in c["descriptive"].items()}))
        for kk in ("pooled_with_E5", "first40_vs_E5_forecast", "first40_OWN_vs_E6", "vs_E8_forecast"):
            if kk in c:
                print("  ", kk, json.dumps(rd(c[kk])))
    print("written", a.out)


if __name__ == "__main__":
    main()
