"""E7 evaluation (amendment C8): a second model, Qwen3.5-397B, under `norm100` and `comp`.

  chase_e7_eval.py [--boot 5000] [--pop FILE] [--out FILE]   -> results/chase_e7_eval.json

Written and hashed in CHASE_LOG.md before any E7 population outcome existed. Nothing is analysed unless every planned
group is present: groups of four OWN 40, LOG 40, DYAD 20, REDACT 20 (seeds 24000+); groups of eight OWN 8, LOG 8
(seeds 28000+). By the sizing rule of C8 (frozen forecast, results/chase_e7_predictions.json) neither cell is a test,
so every p below is reported as an estimate, unadjusted; the Holm family is empty.

Per cell:
  N4  LOG - OWN linear share, group-bootstrap 95% interval (chase_e3_eval.boot_diff, unchanged), Fisher exact p.
  N8  LOG - OWN pooled t_tri, interval as above, Mann-Whitney on per-group t_tri (as chase_e3_eval).
  Every arm against its frozen 90% interval (headline form: X; DYAD has only K), and K for comparison.
  Each contrast against its frozen interval; the observed LOG arm against the LOG forecast without bystander
  features (form Xnb).
  Secondary, by the E9 rule ("decided", chase_e9_eval.describe and share, unchanged): per-group DD share,
  LOG > OWN, two-sided Mann-Whitney with the direction reported; DD + DS per arm.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
sys.path.insert(0, str(Path(__file__).parent))
import chase_e3_eval as V
import chase_e9_eval as E9

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
MODEL = "adaption:qwen3.5-397b"
FRAMINGS = ("norm100", "comp")
PLAN = {4: {"OWN": 40, "LOG": 40, "DYAD": 20, "REDACT": 20}, 8: {"OWN": 8, "LOG": 8}}
SEED0 = {4: 24000, 8: 28000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--pop", default=str(R / "chase_e7_pop.jsonl"))
    ap.add_argument("--out", default=str(R / "chase_e7_eval.json"))
    a = ap.parse_args()
    recs = {(f, arm, N): {} for f in FRAMINGS for N in PLAN for arm in PLAN[N]}
    for l in open(a.pop):
        r = json.loads(l)
        k = (r["framing"], r["arm"], r["N"])
        if r["model"] == MODEL and k in recs and 0 <= r["seed"] - SEED0[r["N"]] < PLAN[r["N"]][r["arm"]]:
            recs[k][r["seed"]] = r
    have = {"|".join(map(str, k)): len(v) for k, v in recs.items()}
    if any(len(v) < PLAN[k[2]][k[1]] for k, v in recs.items()):
        sys.exit(f"incomplete: {have}; nothing analysed")
    pred = json.loads((R / "chase_e7_predictions.json").read_text())
    fz = {(e["framing"], e["arm"], e["N"]): e for e in pred["predictions"] if e["model"] == MODEL}
    fc = {c["framing"]: c for c in pred["contrasts"] if c["model"] == MODEL}
    rng = np.random.default_rng(20261118)
    res = dict(pop_sha256=hashlib.sha256(Path(a.pop).read_bytes()).hexdigest(),
               predictions_sha256=pred["sha256"], have=have, cells={})
    for f in FRAMINGS:
        cell = dict(arms={}, contrasts={}, sizing=fc[f].get("sizing"))
        S = {}
        for N in PLAN:
            for arm, G in PLAN[N].items():
                rs = [recs[(f, arm, N)][SEED0[N] + g] for g in range(G)]
                st = [V.struct(r) for r in rs]
                S[(arm, N)] = (rs, st)
                obs = V.headline(st, N)
                e = fz[(f, arm, N)]
                row = dict(groups=G, observed=obs, linear_n=int(sum(s["linear"] for s in st)) if N == 4 else None,
                           invalid=int(sum(r["invalid"] for r in rs)), forecast={})
                for form in e["forms"]:
                    h = e[form][str(G)]["headline"]
                    row["forecast"][form] = dict(mean=h["mean"], q05=h["q05"], q95=h["q95"], covered=bool(h["q05"] <= obs <= h["q95"]))
                if N == 4:
                    D = [E9.describe(r) for r in rs]
                    row["dd_ds"] = E9.pooled(D, ["DD", "DS"]); row["dd"] = E9.pooled(D, ["DD"])
                    S[(arm, N)] += (D,)
                cell["arms"][f"{arm}|N{N}"] = row
        for N in PLAN:
            (ro, so, *do), (rl, sl, *dl) = S[("OWN", N)], S[("LOG", N)]
            c = dict(own=V.headline(so, N), log=V.headline(sl, N), diff=V.headline(sl, N) - V.headline(so, N),
                     ci95=V.boot_diff(so, sl, N, rng, a.boot))
            if N == 4:
                x, y = sum(s["linear"] for s in so), sum(s["linear"] for s in sl)
                c.update(test="fisher", p=float(fisher_exact([[x, len(so) - x], [y, len(sl) - y]])[1]))
                xo, xl = E9.share(do[0], ["DD"]), E9.share(dl[0], ["DD"])
                c["dd_share"] = dict(own=float(xo.mean()), log=float(xl.mean()), diff=float(xl.mean() - xo.mean()),
                                     mw_p=float(mannwhitneyu(xo, xl).pvalue), direction_ok=bool(xl.mean() > xo.mean()))
            else:
                xa, ya = np.nan_to_num([s["t_tri"] for s in so]), np.nan_to_num([s["t_tri"] for s in sl])
                c.update(test="mannwhitney", p=float(mannwhitneyu(ya, xa, alternative="two-sided").pvalue) if len(set(xa) | set(ya)) > 1 else 1.0)
            G = PLAN[N]["OWN"]
            fr = fc[f][f"N{N}"][str(G)]
            d = fr["diff"]
            c["frozen"] = dict(diff=d["mean"], q05=d["q05"], q95=d["q95"], covered=bool(d["q05"] <= c["diff"] <= d["q95"]),
                               power_05=fr.get("power_05"), log_without_bystander=fr.get("log_without_bystander"))
            cell["contrasts"][f"N{N}"] = c
        res["cells"][f] = cell
    Path(a.out).write_text(json.dumps(res, indent=1))
    for f, cell in res["cells"].items():
        print(f"== Qwen3.5-397B {f}")
        for k, row in cell["arms"].items():
            fx = row["forecast"].get("X", row["forecast"].get("K"))
            extra = "".join(f"  {fm} {v['mean']:+.2f} [{v['q05']:+.2f}, {v['q95']:+.2f}] {'in' if v['covered'] else 'OUT'}"
                            for fm, v in row["forecast"].items())
            print(f"  {k:10s} G={row['groups']:3d} observed {row['observed']:+.2f}{extra}  invalid {row['invalid']}"
                  + (f"  DD+DS {row['dd_ds']:.2f} DD {row['dd']:.2f}" if "dd_ds" in row else ""))
        for k, c in cell["contrasts"].items():
            fr = c["frozen"]
            print(f"  LOG - OWN {k}: {c['diff']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] {c['test']} p={c['p']:.4f}  "
                  f"frozen {fr['diff']:+.2f} [{fr['q05']:+.2f}, {fr['q95']:+.2f}] {'in' if fr['covered'] else 'OUT'}"
                  + (f"  DD share {c['dd_share']['own']:.2f} -> {c['dd_share']['log']:.2f} MW p={c['dd_share']['mw_p']:.4g}" if "dd_share" in c else ""))
    print("written", a.out)


if __name__ == "__main__":
    main()
