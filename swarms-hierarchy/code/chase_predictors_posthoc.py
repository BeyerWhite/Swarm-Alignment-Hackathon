"""Post hoc, not registered (24 Nov 2026, after E14): four candidate predictors of which cells show the bystander
effect, scored on the 13 cells with both calibration rows and OWN/LOG groups of four. No calls.

  chase_predictors_posthoc.py [--boot 100] [--groups 80] [--point 1000]   -> results/chase_predictors_posthoc.json

Calibration rows per cell are those E14 used (chase_e14.cal_rows): E2 all N = 4 rows, E5 pass 1, E7/E8/E14 round 1.
Every cell is fitted with the frozen K form (OWN d, o; LOG d, o, p, tc), because E2 and E5 rows lack the extras.
  1. partner sensitivity: -b_p, the K coefficient on slog(p) in the LOG rows (larger = yields more to strong partners),
     with a row-bootstrap 90% interval; also the empirical LEAD rate at -2..2 minus that at p >= 3 (yielding to a strong partner).
  2. full forecast: simulate groups of four from the K fits of each arm (point: `point` groups per arm; interval:
     `boot` row-bootstrap refits x `groups` groups) and take LOG - OWN in linear share.
  3. room to respond: SD of the fitted K LEAD probabilities over the LOG rows, and their mean p(1 - p).
  4. shape: empirical LEAD rate in the LOG rows by p bin (<= -3, -2..2, >= 3); range = LEAD(p <= -3) - LEAD(p >= 3).
Outcome: observed LOG - OWN linear share (E14 rule file for the known cells, E14 eval for the fresh ones).
Responsive: Gemma-4-31B comp and norm100, Qwen3.5-397B comp. Reported per predictor: Spearman rho with the observed
difference over the 13 cells, and whether the three responsive cells are the top three.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
import chase_cal as CC
import chase_e14 as T
import chase_fit as F
from chase_metrics import pooled

R = T.R
RESPONSIVE = {"adaption:gemma-4-31b-it|comp", "adaption:gemma-4-31b-it|norm100", "adaption:qwen3.5-397b|comp"}
K = {"OWN": ["d", "o"], "LOG": ["d", "o", "p", "tc"]}


def tri(x):
    return np.where(x <= -3, 0, np.where(x >= 3, 2, 1))


def probs(k, d, cols):
    z = k.params["a"] + F.slog(d[cols].values.astype(float)) @ np.array([k.params[c] for c in cols])
    return 1 / (1 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--groups", type=int, default=80)
    ap.add_argument("--point", type=int, default=1000)
    a = ap.parse_args()
    rule, ev = json.loads(T.RULE.read_text()), json.loads(T.EVAL.read_text())
    src = dict(T.KNOWN, **{c: "e14" for c in T.FRESH}, **{T.GLM_N100: "e2"})
    obs = {c: x["outcome"]["diff"] for c, x in rule["known"].items()} | {c: x["diff"] for c, x in ev["cells"].items()}
    rows, res = {}, {}
    rng = np.random.default_rng(20261124)
    for cell, s in src.items():
        rows.setdefault(s, T.cal_rows(s))
        model, framing = cell.split("|")
        d = rows[s]; d = d[(d.model == model) & (d.framing == framing)].copy()
        d["lead"], d["off"] = (d.choice == "LEAD").astype(float), 0.0
        dl, do = d[d.arm == "LOG"], d[d.arm == "OWN"]
        kl, ko = F.fit_logistic(dl, K["LOG"]), F.fit_logistic(do, K["OWN"])
        bp = [-F.fit_logistic(dl, K["LOG"], rng).params["p"] for _ in range(200)]
        bins = tri(dl.p.values)
        lead_bin = {str(b): dict(n=int((bins == b).sum()), lead=float(dl.lead[bins == b].mean()) if (bins == b).any() else None)
                    for b in (0, 1, 2)}
        pl = probs(kl, dl, K["LOG"])
        sims = []
        for b in range(a.boot):
            r2 = np.random.default_rng([b, 24])
            o = pooled(CC.simulate(F.fit_logistic(do, K["OWN"], r2), "OWN", 4, a.groups, int(r2.integers(1e9))))["linear"]
            l = pooled(CC.simulate(F.fit_logistic(dl, K["LOG"], r2), "LOG", 4, a.groups, int(r2.integers(1e9))))["linear"]
            sims.append(l - o)
        po = pooled(CC.simulate(ko, "OWN", 4, a.point, 7))["linear"]
        pll = pooled(CC.simulate(kl, "LOG", 4, a.point, 8))["linear"]
        x = dict(source=s, rows=dict(OWN=len(do), LOG=len(dl)), observed_diff=obs[cell], responsive=cell in RESPONSIVE,
                 lead=dict(OWN=float(do.lead.mean()), LOG=float(dl.lead.mean())),
                 i1_partner_sensitivity=dict(neg_b_p=-kl.params["p"], ci90=[float(np.quantile(bp, .05)), float(np.quantile(bp, .95))],
                                             b_o=kl.params["o"], b_tc=kl.params["tc"],
                                             mid_minus_strong=(lead_bin["1"]["lead"] - lead_bin["2"]["lead"])
                                             if lead_bin["2"]["lead"] is not None and lead_bin["1"]["lead"] is not None else None),
                 i2_forecast=dict(own=po, log=pll, diff=pll - po, boot_mean=float(np.mean(sims)),
                                  q05=float(np.quantile(sims, .05)), q95=float(np.quantile(sims, .95)), p_pos=float(np.mean(np.array(sims) > 0))),
                 i3_room=dict(sd_fitted=float(pl.std()), mean_pq=float(np.mean(pl * (1 - pl)))),
                 i4_shape=dict(bins=lead_bin, range=(lead_bin["0"]["lead"] - lead_bin["2"]["lead"])
                               if lead_bin["0"]["lead"] is not None and lead_bin["2"]["lead"] is not None else None))
        res[cell] = x
        print(f"{cell:38s} obs {x['observed_diff']:+.2f} | 1: -b_p {x['i1_partner_sensitivity']['neg_b_p']:+.2f} "
              f"[{x['i1_partner_sensitivity']['ci90'][0]:+.2f},{x['i1_partner_sensitivity']['ci90'][1]:+.2f}] | 2: fc {x['i2_forecast']['diff']:+.2f} "
              f"[{x['i2_forecast']['q05']:+.2f},{x['i2_forecast']['q95']:+.2f}] | 3: sd {x['i3_room']['sd_fitted']:.3f} pq {x['i3_room']['mean_pq']:.3f} "
              f"| 4: range {x['i4_shape']['range']} bins {[(v['n'], None if v['lead'] is None else round(v['lead'], 2)) for v in lead_bin.values()]}",
              flush=True)
    preds = {"1 -b_p": lambda x: x["i1_partner_sensitivity"]["neg_b_p"],
             "1 mid-strong": lambda x: x["i1_partner_sensitivity"]["mid_minus_strong"],
             "2 forecast diff": lambda x: x["i2_forecast"]["diff"],
             "3 sd fitted": lambda x: x["i3_room"]["sd_fitted"],
             "3 mean pq": lambda x: x["i3_room"]["mean_pq"],
             "4 range": lambda x: x["i4_shape"]["range"]}
    summ = {}
    for name, f in preds.items():
        cs = [c for c in res if f(res[c]) is not None]
        v, y = [f(res[c]) for c in cs], [res[c]["observed_diff"] for c in cs]
        rho, p = spearmanr(v, y)
        top = sorted(cs, key=lambda c: -f(res[c]))[:3]
        rank = {c: i + 1 for i, c in enumerate(sorted(cs, key=lambda c: -f(res[c])))}
        summ[name] = dict(cells=len(cs), spearman=float(rho), p=float(p), top3_are_responsive=set(top) == RESPONSIVE,
                          ranks_of_responsive=sorted(rank[c] for c in RESPONSIVE if c in rank))
        print(f"{name:16s} rho {rho:+.2f} (p {p:.3f}, {len(cs)} cells)  responsive ranks {summ[name]['ranks_of_responsive']}")
    out = R / "chase_predictors_posthoc.json"
    out.write_text(json.dumps(dict(args=vars(a), cells=res, summary=summ), indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
