"""E14 (amendment C10): does the LEAD-rate account predict, from calibration rows alone, which cells gain?

  chase_e14.py cal                 round-1 calibration in the fresh cells (calls; chase_cal unchanged, round 1 only)
  chase_e14.py rule [--with-glm]   write-once rule file results/chase_e14_rule.json (no calls); refuses to overwrite
  chase_e14.py eval [--boot 5000]  after every planned group exists: results/chase_e14_eval.json (no calls)

The rule, fixed before any fresh calibration row exists. From calibration rows at N = 4 drawn as round 1 of E7
(records visited by groups driven by kernels fitted on E1 rows, uniform sampling), take the LEAD rates l_OWN and
l_LOG, and the decided share they imply if the two answers in a contest were independent, q(l) = 2 l (1 - l).
  predicted gain  <=>  dq = q(l_LOG) - q(l_OWN) > 0
(partner records move the LEAD rate toward one half, so more contests are settled by the answers). A row bootstrap
(2,000) gives the share of resamples with dq > 0 and a 90% interval; they are reported, the point value decides.
Known cells, calibration rows already on disk: E2 (all N = 4 rows), E5 pass 1 (jid without "|r2"), E7 and E8 round 1.
Their outcomes were known when the rule was written, so agreement there is a consistency check, not evidence.
Fresh cells, never run as groups: Gemma-4-26B `comp` and `norm100`, Gemma-4-31B `norm0`, Qwen3.8-27B `comp`;
with --with-glm also GLM-5.2 `norm100` (its rows are in E2). OWN 80 and LOG 80 groups of four per cell, seeds
54000-54079, run whatever the prediction.
Outcome per fresh cell: "gain" if LOG > OWN in linear share with one-sided Fisher p < 0.05, otherwise "no gain".
  P1 over the cells predicted to gain: mean LOG - OWN difference, group-bootstrap 95% interval above zero.
  P2 over the cells predicted not to gain: the same interval below +0.17 (the margin of H5 in amendment C6).
  The rule passes if P1 and P2 both hold (a part with no cells is not tested). Reported: correct cells of all fresh.
Descriptive: per cell and arm, the LEAD rate and the decided share in the groups (chase_e8_eval, unchanged).
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
sys.path.insert(0, str(Path(__file__).parent))
import chase_e3_eval as V
import chase_e8_eval as E8

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = Path(os.environ.get("CHASE_CAL_DIR", R))                     # tests redirect the E14 files, as chase_cal does
RULE, POP, EVAL = OUT / "chase_e14_rule.json", OUT / "chase_e14_pop.jsonl", OUT / "chase_e14_eval.json"
SEED0, PLAN = 54000, {"OWN": 80, "LOG": 80}
FRESH = ["adaption:gemma-4-26b-a4b-it|comp", "adaption:gemma-4-26b-a4b-it|norm100", "adaption:gemma-4-31b-it|norm0",
         "adaption:qwen3.8-27b|comp"]
GLM_N100 = "adaption:glm-5.2|norm100"
KNOWN = {  # cell: (calibration file, row filter)
    "adaption:gemma-4-31b-it|comp": "e2", "adaption:gemma-4-31b-it|norm100": "e2", "adaption:glm-5.2|comp": "e2",
    "adaption:deepseek-v4-flash|comp": "e2", "adaption:gemma-4-26b-a4b-it|norm0": "e5",
    "adaption:qwen3.5-397b|comp": "e7", "adaption:qwen3.5-397b|norm100": "e7", "adaption:qwen3.5-397b|norm0": "e8"}
POPFILES = ["chase_e3_pop.jsonl", "chase_e7_pop.jsonl", "chase_e8_pop.jsonl", "chase_e9_pop.jsonl", "chase_e10_pop.jsonl"]


def cal_rows(src):
    f = {"e2": "chase_e2_statecal.jsonl", "e5": "chase_e5_statecal.jsonl", "e7": "chase_e7_statecal.jsonl",
         "e8": "chase_e8_statecal.jsonl", "e14": "chase_e14_statecal.jsonl"}[src]
    d = pd.DataFrame([json.loads(l) for l in open((OUT if src == "e14" else R) / f)])
    d = d[(d.N == 4) & d.arm.isin(["OWN", "LOG"]) & d.choice.isin(["LEAD", "DEFER"])]
    if src == "e5":
        d = d[~d.jid.str.contains(r"\|r2")]
    if src in ("e7", "e8", "e14"):
        d = d[d["round"] == 1]
    return d


def dq(l_own, l_log):
    return 2 * l_log * (1 - l_log) - 2 * l_own * (1 - l_own)


def rule_for(d, cell, rng, B=2000):
    model, framing = cell.split("|")
    d = d[(d.model == model) & (d.framing == framing)]
    o = (d[d.arm == "OWN"].choice == "LEAD").values.astype(float)
    l = (d[d.arm == "LOG"].choice == "LEAD").values.astype(float)
    assert len(o) and len(l), cell
    bs = [dq(rng.choice(o, len(o)).mean(), rng.choice(l, len(l)).mean()) for _ in range(B)]
    v = dq(o.mean(), l.mean())
    return dict(rows=dict(OWN=len(o), LOG=len(l)), l_own=float(o.mean()), l_log=float(l.mean()), dq=float(v),
                p_dq_pos=float(np.mean(np.array(bs) > 0)), dq_ci90=[float(np.quantile(bs, .05)), float(np.quantile(bs, .95))],
                prediction="gain" if v > 0 else "no gain")


def groups(files, cell, plan=None, seed0=None):
    model, framing = cell.split("|")
    out = {"OWN": {}, "LOG": {}}
    for fn in files:
        if not ((OUT if fn == POP.name else R) / fn).exists():
            continue
        for line in open((OUT if fn == POP.name else R) / fn):
            r = json.loads(line)
            if r.get("model") == model and r["framing"] == framing and r["N"] == 4 and r["arm"] in out and "members" not in r:
                if plan is None or 0 <= r["seed"] - seed0 < plan[r["arm"]]:
                    out[r["arm"]][(fn, r["seed"])] = r
    return {a: list(v.values()) for a, v in out.items()}


def outcome(g):
    so, sl = [V.struct(r) for r in g["OWN"]], [V.struct(r) for r in g["LOG"]]
    xo, xl = sum(s["linear"] for s in so), sum(s["linear"] for s in sl)
    p = float(fisher_exact([[xl, len(sl) - xl], [xo, len(so) - xo]], alternative="greater")[1])
    return dict(n=[len(so), len(sl)], own=xo / len(so), log=xl / len(sl), diff=xl / len(sl) - xo / len(so), p_one_sided=p,
                observed="gain" if p < 0.05 and xl / len(sl) > xo / len(so) else "no gain")


def cmd_cal():
    import chase_cal as CC
    CC.EXP["E14"] = dict(cells=FRESH, e1_round1=True, round1={4: {"OWN": 300, "LOG": 450}}, later={},
                         plan={4: {"OWN": [80], "LOG": [80]}})
    jobs = CC.round_jobs("E14", 1, FRESH, CC.load("E14"))
    CC.run_jobs(jobs, CC.paths("E14")[0], "chase_e14")


def cmd_rule(with_glm):
    if RULE.exists():
        sys.exit(f"{RULE} exists; the rule file is write-once")
    rng = np.random.default_rng(20261120)
    res = dict(time=time.strftime("%F %T"), rule="gain iff q(l_LOG) - q(l_OWN) > 0, q(l) = 2 l (1 - l)", known={}, fresh={})
    src = {}
    for cell, s in KNOWN.items():
        src.setdefault(s, cal_rows(s))
        res["known"][cell] = dict(rule_for(src[s], cell, rng), outcome=outcome(groups(POPFILES, cell)))
        res["known"][cell]["agree"] = res["known"][cell]["prediction"] == res["known"][cell]["outcome"]["observed"]
    e14 = cal_rows("e14")
    for cell in FRESH:
        res["fresh"][cell] = rule_for(e14, cell, rng)
    if with_glm:
        res["fresh"][GLM_N100] = rule_for(cal_rows("e2"), GLM_N100, rng)
    res["run"] = dict(cells=list(res["fresh"]), plan=PLAN, seed0=SEED0, N=4)
    body = json.dumps(res, indent=1, sort_keys=True)
    res["stamp"] = hashlib.sha256(body.encode()).hexdigest()
    RULE.write_text(json.dumps(res, indent=1))
    for k in ("known", "fresh"):
        for cell, x in res[k].items():
            print(f"{k:5s} {cell:40s} l_OWN {x['l_own']:.3f} l_LOG {x['l_log']:.3f} dq {x['dq']:+.3f} "
                  f"P(dq>0) {x['p_dq_pos']:.2f} -> {x['prediction']}"
                  + (f" | observed {x['outcome']['observed']} ({x['outcome']['own']:.2f} -> {x['outcome']['log']:.2f}) agree {x['agree']}" if k == "known" else ""))
    print("stamp", res["stamp"])


def mean_diff_ci(cells, rng, B):
    pts, bs = [], np.zeros(B)
    for g in cells:
        lo, ll = np.array([V.struct(r)["linear"] for r in g["OWN"]], float), np.array([V.struct(r)["linear"] for r in g["LOG"]], float)
        pts.append(ll.mean() - lo.mean())
        bs += np.array([rng.choice(ll, len(ll)).mean() - rng.choice(lo, len(lo)).mean() for _ in range(B)])
    bs /= len(cells)
    return dict(cells=len(cells), mean_diff=float(np.mean(pts)), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])


def cmd_eval(B):
    rule = json.loads(RULE.read_text())
    run, rng = rule["run"], np.random.default_rng(20261121)
    res = dict(rule_stamp=rule["stamp"], pop_sha256=hashlib.sha256(POP.read_bytes()).hexdigest(), cells={})
    g = {}
    for cell in run["cells"]:
        g[cell] = groups([POP.name], cell, run["plan"], run["seed0"])
        have = {a: len(v) for a, v in g[cell].items()}
        if any(have[a] < run["plan"][a] for a in run["plan"]):
            sys.exit(f"incomplete {cell}: {have}; nothing analysed")
    for cell in run["cells"]:
        o = outcome(g[cell])
        o["ci95"] = V.boot_diff([V.struct(r) for r in g[cell]["OWN"]], [V.struct(r) for r in g[cell]["LOG"]], 4, rng, B)
        o["predicted"] = rule["fresh"][cell]["prediction"]
        o["correct"] = o["predicted"] == o["observed"]
        o["lead_rate"] = {a: float(E8.lead_rate(g[cell][a])) for a in ("OWN", "LOG")}
        o["decided"] = {a: float(np.mean([E8.decided(r) for r in g[cell][a]])) for a in ("OWN", "LOG")}
        o["invalid"] = {a: int(sum(r["invalid"] for r in g[cell][a])) for a in ("OWN", "LOG")}
        res["cells"][cell] = o
    gain = [g[c] for c in run["cells"] if rule["fresh"][c]["prediction"] == "gain"]
    null = [g[c] for c in run["cells"] if rule["fresh"][c]["prediction"] == "no gain"]
    res["P1"] = dict(mean_diff_ci(gain, rng, B), holds=None) if gain else None
    if gain:
        res["P1"]["holds"] = bool(res["P1"]["ci95"][0] > 0)
    res["P2"] = dict(mean_diff_ci(null, rng, B), holds=None) if null else None
    if null:
        res["P2"]["holds"] = bool(res["P2"]["ci95"][1] < 0.17)
    res["passes"] = all(x["holds"] for x in (res["P1"], res["P2"]) if x)
    res["correct"] = f"{sum(o['correct'] for o in res['cells'].values())} of {len(res['cells'])}"
    EVAL.write_text(json.dumps(res, indent=1))
    for cell, o in res["cells"].items():
        print(f"{cell:40s} OWN {o['own']:.2f} LOG {o['log']:.2f} diff {o['diff']:+.2f} [{o['ci95'][0]:+.2f}, {o['ci95'][1]:+.2f}] "
              f"p {o['p_one_sided']:.3g} observed {o['observed']} predicted {o['predicted']} correct {o['correct']} "
              f"LEAD {o['lead_rate']} decided {o['decided']}")
    print("P1", res["P1"], "\nP2", res["P2"], "\npasses", res["passes"], " correct", res["correct"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cal", "rule", "eval"])
    ap.add_argument("--with-glm", action="store_true")
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    {"cal": cmd_cal, "rule": lambda: cmd_rule(a.with_glm), "eval": lambda: cmd_eval(a.boot)}[a.cmd]()


if __name__ == "__main__":
    main()
