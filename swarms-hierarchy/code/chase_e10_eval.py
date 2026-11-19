"""E10 evaluation (amendment C9): registered replication of the E7 `comp` result in Qwen3.5-397B.

  chase_e10_eval.py [--boot 5000] [--pop FILE] [--plan FILE] [--out FILE]   -> results/chase_e10_eval.json

Written and hashed in CHASE_LOG.md (amendment C9) before any E10 call. Groups of four, `comp`, fresh seeds 44000+:
OWN 80, LOG 80, REDACT 80 (plan A) or 40 (plan B). The queue script chooses the plan from the credit headroom before
the first call and writes it to results/chase_e10_plan.json; this script reads the REDACT size from there. Nothing
is analysed unless every planned group is present.

  P1 (primary, the only confirmatory test): LOG linear share > OWN, one-sided Fisher exact, alpha 0.05.
     Reported with the group-bootstrap 95% interval of chase_e3_eval.boot_diff (unchanged).
  S1 attribution: LOG - REDACT and REDACT - OWN, intervals as above, two-sided Fisher. Reading:
       "outcomes"   LOG - REDACT > 0 with its interval excluding zero;
       "extra text" REDACT - OWN > 0 with its interval excluding zero, and the LOG - REDACT interval including zero;
       otherwise    "unresolved".
  S2 LEAD-rate account: per group, LEAD rate LOG < OWN and share of contests with one LEAD and one DEFER
     LOG > OWN (chase_e8_eval.lead_rate and decided, unchanged), two-sided Mann-Whitney each; holds if both have
     the predicted direction with p < 0.05.
  S3 DD share by the E9 rule (chase_e9_eval.describe and share, unchanged), LOG > OWN, two-sided Mann-Whitney.
  Descriptive: E10 + E7 pooled (E7 `comp` N4 OWN 40, LOG 40, REDACT 20, seeds 24000+), same contrasts; OWN and LOG
     against the frozen E7 forecast (form X) at 80 groups; invalid answers per arm.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
sys.path.insert(0, str(Path(__file__).parent))
import chase_e3_eval as V
import chase_e8_eval as E8
import chase_e9_eval as E9

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
MODEL, FRAMING, N = "adaption:qwen3.5-397b", "comp", 4
SEED0, E7_SEED0 = 44000, 24000
E7_PLAN = {"OWN": 40, "LOG": 40, "REDACT": 20}


def load(path, seed0, plan):
    out = {arm: {} for arm in plan}
    for l in open(path):
        r = json.loads(l)
        if (r["model"] == MODEL and r["framing"] == FRAMING and r["N"] == N and r["arm"] in plan
                and 0 <= r["seed"] - seed0 < plan[r["arm"]]):
            out[r["arm"]][r["seed"]] = r
    return out


def contrast(a, b, rng, B, alternative="two-sided"):
    """Linear share of b minus a; a and b are lists of group records."""
    sa, sb = [V.struct(r) for r in a], [V.struct(r) for r in b]
    xa, xb = sum(s["linear"] for s in sa), sum(s["linear"] for s in sb)
    p = float(fisher_exact([[xb, len(sb) - xb], [xa, len(sa) - xa]], alternative=alternative)[1])
    return dict(n=[len(sa), len(sb)], a=V.headline(sa, N), b=V.headline(sb, N), a_n=int(xa), b_n=int(xb),
                diff=V.headline(sb, N) - V.headline(sa, N), ci95=V.boot_diff(sa, sb, N, rng, B), p=p, alternative=alternative)


def mw(x_own, x_log):
    x_own, x_log = np.asarray(x_own, float), np.asarray(x_log, float)
    p = float(mannwhitneyu(x_log, x_own, alternative="two-sided").pvalue) if len(set(x_own) | set(x_log)) > 1 else 1.0
    return dict(own=float(x_own.mean()), log=float(x_log.mean()), diff=float(x_log.mean() - x_own.mean()), mw_p=p)


def analyse(d, rng, B):
    own, log, red = d["OWN"], d["LOG"], d["REDACT"]
    out = dict(P1=contrast(own, log, rng, B, "greater"))
    out["P1"]["supported"] = bool(out["P1"]["p"] < 0.05)
    lr, ro = contrast(red, log, rng, B), contrast(own, red, rng, B)
    reading = ("outcomes" if lr["diff"] > 0 and lr["ci95"][0] > 0 else
               "extra text" if ro["diff"] > 0 and ro["ci95"][0] > 0 and lr["ci95"][0] <= 0 <= lr["ci95"][1] else "unresolved")
    out["S1"] = dict(log_minus_redact=lr, redact_minus_own=ro, reading=reading)
    lead = mw([E8.lead_rate([r]) for r in own], [E8.lead_rate([r]) for r in log])
    dec = mw([E8.decided(r) for r in own], [E8.decided(r) for r in log])
    out["S2"] = dict(lead_rate=lead, decided=dec, holds=bool(lead["diff"] < 0 and lead["mw_p"] < 0.05
                                                              and dec["diff"] > 0 and dec["mw_p"] < 0.05))
    Do, Dl = [E9.describe(r) for r in own], [E9.describe(r) for r in log]
    dd = mw(E9.share(Do, ["DD"]), E9.share(Dl, ["DD"]))
    dd["direction_ok"] = bool(dd["diff"] > 0)
    out["S3"] = dict(dd_share=dd, dd_ds_pooled=dict(OWN=E9.pooled(Do, ["DD", "DS"]), LOG=E9.pooled(Dl, ["DD", "DS"])))
    out["invalid"] = {arm: int(sum(r["invalid"] for r in d[arm])) for arm in d}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--pop", default=str(R / "chase_e10_pop.jsonl"))
    ap.add_argument("--plan", default=str(R / "chase_e10_plan.json"))
    ap.add_argument("--e7pop", default=str(R / "chase_e7_pop.jsonl"))
    ap.add_argument("--out", default=str(R / "chase_e10_eval.json"))
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())["groups"]
    assert plan["OWN"] == plan["LOG"] == 80 and plan["REDACT"] in (40, 80), plan
    data = load(a.pop, SEED0, plan)
    have = {arm: len(v) for arm, v in data.items()}
    if any(have[arm] < plan[arm] for arm in plan):
        sys.exit(f"incomplete: {have}; nothing analysed")
    d = {arm: [data[arm][SEED0 + g] for g in range(plan[arm])] for arm in plan}
    rng = np.random.default_rng(20261119)
    res = dict(pop_sha256=hashlib.sha256(Path(a.pop).read_bytes()).hexdigest(), plan=plan, have=have,
               fresh=analyse(d, rng, a.boot))
    old = load(a.e7pop, E7_SEED0, E7_PLAN)
    assert all(len(old[arm]) == E7_PLAN[arm] for arm in E7_PLAN)
    pooled = {arm: d[arm] + [old[arm][E7_SEED0 + g] for g in range(E7_PLAN[arm])] for arm in plan}
    res["pooled_with_E7"] = analyse(pooled, rng, a.boot)
    pred = json.loads((R / "chase_e7_predictions.json").read_text())
    res["vs_E7_forecast"] = {}
    for e in pred["predictions"]:
        if e["model"] == MODEL and e["framing"] == FRAMING and e["N"] == N and e["arm"] in ("OWN", "LOG"):
            h = e["X"]["80"]["headline"]
            obs = V.headline([V.struct(r) for r in d[e["arm"]]], N)
            res["vs_E7_forecast"][e["arm"]] = dict(observed=obs, mean=h["mean"], q05=h["q05"], q95=h["q95"],
                                                   covered=bool(h["q05"] <= obs <= h["q95"]))
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k in ("fresh", "pooled_with_E7"):
        f = res[k]; p1 = f["P1"]
        print(f"== {k}: OWN {p1['a']:.2f} ({p1['a_n']}/{p1['n'][0]})  LOG {p1['b']:.2f} ({p1['b_n']}/{p1['n'][1]})  "
              f"diff {p1['diff']:+.2f} [{p1['ci95'][0]:+.2f}, {p1['ci95'][1]:+.2f}]  one-sided Fisher p={p1['p']:.4f}"
              + (f"  P1 supported={p1['supported']}" if k == "fresh" else ""))
        for nm, c in (("LOG - REDACT", f["S1"]["log_minus_redact"]), ("REDACT - OWN", f["S1"]["redact_minus_own"])):
            print(f"   {nm}: {c['diff']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] p={c['p']:.3f}")
        print(f"   S1 reading: {f['S1']['reading']}")
        s2 = f["S2"]
        print(f"   S2 LEAD {s2['lead_rate']['own']:.2f} -> {s2['lead_rate']['log']:.2f} (p={s2['lead_rate']['mw_p']:.3g}); "
              f"decided {s2['decided']['own']:.2f} -> {s2['decided']['log']:.2f} (p={s2['decided']['mw_p']:.3g}); holds={s2['holds']}")
        s3 = f["S3"]["dd_share"]
        print(f"   S3 DD share {s3['own']:.2f} -> {s3['log']:.2f} p={s3['mw_p']:.3g}; invalid {f['invalid']}")
    print("   vs E7 forecast:", json.dumps({k: {x: round(v, 3) if isinstance(v, float) else v for x, v in c.items()}
                                           for k, c in res["vs_E7_forecast"].items()}))
    print("written", a.out)


if __name__ == "__main__":
    main()
