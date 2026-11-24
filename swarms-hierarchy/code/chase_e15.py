"""E15 (amendment C11): do two rules, fixed from the 13-cell screen, predict which fresh cells show the bystander effect?

  chase_e15.py cal --cells CELL ... [--workers 6]   round-1 calibration rows for those cells (calls)
  chase_e15.py rule                                 write-once rule file results/chase_e15_rule.json (no calls)
  chase_e15.py pop --cells CELL ... [--parallel 6]  OWN 80 / LOG 80 groups of four per cell (calls; chase_f_run)
  chase_e15.py eval [--boot 5000]                   results/chase_e15_eval.json (no calls)
  chase_e15.py check                                Y and F on the 13 known cells against the post hoc screen
  chase_e15.py prompt                               print the norm50 framing sentence and one full LOG prompt
  chase_e15.py smoke --cells CELL ...               20 real calls per cell; reports only parse rate and time

Run `cal` and `pop` under chase_paced.py (Adaption cells) or chase_cli.py (claude:, gpt: cells).

The new framing `norm50` is the existing guidance sentence (chase_prompts.NORM) with w = 50. It is added here, at
import, by extending chase_prompts.FRAMINGS and wrapping chase_prompts.framing_text, which log_prompt looks up at
call time. The frozen chase_prompts.py is not edited; for the three existing framings the text is unchanged.

Round-1 calibration as in E7/E14 (chase_cal, e1_round1): records come from groups driven by surrogate kernels
fitted on E1 rows of the same model and framing, and OWN 300 / LOG 450 of them are queried uniformly. For `norm50`
there are no E1 rows. One kernel is fitted on the model's E1 `norm100` rows and one on its E1 `norm0` rows (each
at the last level of chase_fit.levels, the one E1-based round 1 uses first), and each drives half the groups.

The rules, from the LOG and OWN calibration rows of a cell alone:
  Y = LEAD rate at partner net -2..2 minus LEAD rate at partner net >= 3 (LOG rows)       gain iff Y >= 0.20
  F = LOG - OWN linear share in groups of four simulated from K-form fits (chase_cal.simulate;
      OWN d, o; LOG d, o, p, tc), 1,000 groups per arm at seeds 7 (OWN) and 8 (LOG)       gain iff F >= +0.15
      with a row-bootstrap interval (100 refits x 80 groups, rng [b, 24]) reported, not used.
These are the computations of chase_predictors_posthoc.py; `check` confirms equal values on the 13 known cells.
Stop rule: if each rule predicts the same for all seven cells, stage 2 is not run ("untestable on this pool").

Tests (eval). Outcome per cell: LOG - OWN linear share (chase_e14.outcome; "gain" iff one-sided Fisher p < 0.05).
  H-Y, H-F: D = mean outcome over the rule's predicted-gain cells minus mean over its predicted-null cells; 95%
  interval from resampling groups within each cell and arm; supported iff its lower bound > 0, Holm over the two
  (p = share of resamples with D <= 0, doubled for the larger). A rule with all cells on one side is not tested.
  Secondary: correct cells per rule and for "Y and F"; Spearman of Y and of F with the outcome over the 7 fresh
  cells and over all 20 (the 13 known from results/chase_predictors_posthoc.json).
  Descriptive: LEAD rate and decided share per cell and arm in the groups; E1 Y against calibration Y.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import chase_prompts as P

if "norm50" not in P.FRAMINGS:
    P.FRAMINGS = P.FRAMINGS + ("norm50",)
    _framing_text = P.framing_text
    P.framing_text = lambda framing: P.NORM.format(w=50) if framing == "norm50" else _framing_text(framing)

import chase_cal as CC
import chase_e2 as E
import chase_e14 as T
import chase_e8_eval as E8
import chase_fit as F
from chase_metrics import pooled

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = Path(os.environ.get("CHASE_CAL_DIR", R))
CAL, RULE, POP, EVAL = (OUT / f"chase_e15_{s}" for s in ("statecal.jsonl", "rule.json", "pop.jsonl", "eval.json"))
CELLS = ["adaption:gemma-4-31b-it|norm50", "adaption:gemma-4-26b-a4b-it|norm50", "adaption:qwen3.5-397b|norm50",
         "gpt:gpt-6-luna|norm100", "claude:haiku|norm100",
         "adaption:qwen3.8-27b|norm100", "adaption:deepseek-v4-flash|norm100"]
SEED0, PLAN = 64000, {"OWN": 80, "LOG": 80}
Y_CUT, F_CUT = 0.20, 0.15
K = {"OWN": ["d", "o"], "LOG": ["d", "o", "p", "tc"]}
CC.EXP["E15"] = dict(cells=CELLS, e1_round1=True, round1={4: {"OWN": 300, "LOG": 450}}, later={},
                     plan={4: {"OWN": [80], "LOG": [80]}})
_surrogates = CC.surrogates


def surrogates(exp, cell, arm, N, rnd, df):
    model, framing = cell.split("|")
    if exp == "E15" and framing == "norm50":
        e1 = F.load_rows([CC.E1])
        return [E.surrogate_kernels(e1, model, "norm100", arm)[0], E.surrogate_kernels(e1, model, "norm0", arm)[0]]
    return _surrogates(exp, cell, arm, N, rnd, df)


CC.surrogates = surrogates


def cal_rows(path):
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    d = d.drop_duplicates("jid", keep="last")
    d = d[(d.N == 4) & d.arm.isin(["OWN", "LOG"]) & d.choice.isin(["LEAD", "DEFER"]) & (d["round"] == 1)]
    return d


def scores(d, cell, boot=100, groups=80, point=1000):
    """Y and F for one cell from its calibration rows (the computation of chase_predictors_posthoc.py)."""
    model, framing = cell.split("|")
    d = d[(d.model == model) & (d.framing == framing)].copy()
    d["lead"], d["off"] = (d.choice == "LEAD").astype(float), 0.0
    dl, do = d[d.arm == "LOG"], d[d.arm == "OWN"]
    bins = np.where(dl.p.values <= -3, 0, np.where(dl.p.values >= 3, 2, 1))
    lead = {b: float(dl.lead[bins == b].mean()) if (bins == b).any() else None for b in (0, 1, 2)}
    y = lead[1] - lead[2] if lead[1] is not None and lead[2] is not None else None
    sims = []
    for b in range(boot):
        r2 = np.random.default_rng([b, 24])
        o = pooled(CC.simulate(F.fit_logistic(do, K["OWN"], r2), "OWN", 4, groups, int(r2.integers(1e9))))["linear"]
        l = pooled(CC.simulate(F.fit_logistic(dl, K["LOG"], r2), "LOG", 4, groups, int(r2.integers(1e9))))["linear"]
        sims.append(l - o)
    po = pooled(CC.simulate(F.fit_logistic(do, K["OWN"]), "OWN", 4, point, 7))["linear"]
    pl = pooled(CC.simulate(F.fit_logistic(dl, K["LOG"]), "LOG", 4, point, 8))["linear"]
    return dict(rows=dict(OWN=len(do), LOG=len(dl)), lead=dict(OWN=float(do.lead.mean()), LOG=float(dl.lead.mean())),
                lead_by_p_bin={str(b): dict(n=int((bins == b).sum()), lead=v) for b, v in lead.items()},
                Y=y, F=pl - po, F_own=po, F_log=pl,
                F_boot=dict(mean=float(np.mean(sims)), q05=float(np.quantile(sims, .05)), q95=float(np.quantile(sims, .95))) if sims else None,
                dq=T.dq(float(do.lead.mean()), float(dl.lead.mean())))


def e1_y(cell):
    model, framing = cell.split("|")
    d = pd.DataFrame([json.loads(l) for l in open(CC.E1)])
    d = d[(d.model == model) & (d.framing == framing) & (d.arm == "LOG") & d.choice.isin(["LEAD", "DEFER"])]
    if not len(d):
        return None
    L = (d.choice == "LEAD").astype(float)
    return float(L[d.p.abs() <= 2].mean() - L[d.p >= 3].mean())


def cmd_cal(cells, workers):
    bad = [c for c in cells if c not in CELLS]
    if bad:
        sys.exit(f"not E15 cells: {bad}")
    jobs = CC.round_jobs("E15", 1, cells, pd.DataFrame())
    CC.run_jobs(jobs, CAL, "chase_e15", workers=workers)


def cmd_rule(plan):
    if RULE.exists():
        sys.exit(f"{RULE} exists; the rule file is write-once")
    d = cal_rows(CAL)
    have = d.groupby(["model", "framing", "arm"]).size().to_dict()
    for c in CELLS:
        m, f = c.split("|")
        for arm, need in (("OWN", 300), ("LOG", 450)):
            if have.get((m, f, arm), 0) < 0.95 * need:
                sys.exit(f"calibration incomplete for {c} {arm}: {have.get((m, f, arm), 0)} of {need} valid rows")
    res = dict(time=time.strftime("%F %T"), rules=dict(Y=f"gain iff Y >= {Y_CUT}", F=f"gain iff F >= +{F_CUT}"),
               cal_sha256=hashlib.sha256(CAL.read_bytes()).hexdigest(), cells={})
    for c in CELLS:
        s = scores(d, c)
        s["e1_Y"] = e1_y(c)
        s["pred_Y"] = "gain" if s["Y"] is not None and s["Y"] >= Y_CUT else "no gain"
        s["pred_F"] = "gain" if s["F"] >= F_CUT else "no gain"
        res["cells"][c] = s
        print(f"{c:38s} rows {s['rows']} Y {s['Y']:+.3f} -> {s['pred_Y']:7s} F {s['F']:+.3f} "
              f"[{s['F_boot']['q05']:+.2f}, {s['F_boot']['q95']:+.2f}] -> {s['pred_F']:7s} (E1 Y {s['e1_Y']}) dq {s['dq']:+.3f}", flush=True)
    tested = {r: len({x[f"pred_{r}"] for x in res["cells"].values()}) == 2 for r in ("Y", "F")}
    res["tested"], res["stage2"] = tested, any(tested.values())
    res["run"] = dict(cells=CELLS, plan=plan, seed0=SEED0, N=4)
    res["stamp"] = hashlib.sha256(json.dumps(res, indent=1, sort_keys=True).encode()).hexdigest()
    RULE.write_text(json.dumps(res, indent=1))
    print("tested", tested, "stage 2", res["stage2"], "stamp", res["stamp"])


def cmd_pop(cells, parallel):
    rule = json.loads(RULE.read_text())
    if not rule["stage2"]:
        sys.exit("rule file says stage 2 is not run (untestable on this pool)")
    if os.environ.get("CHASE_CAL_FAKE") == "1":                      # test double; never in a real run
        import llm, random
        llm.call = lambda model, system, prompt, tag="", **k: random.Random(hash((system, prompt))).choice(["LEAD", "DEFER"])
    import chase_f_run
    plan = rule["run"]["plan"]
    sys.argv = ["chase_f_run.py", "--stage", "E15", "--cells", *cells, "--size", "4",
                "--plan", *[f"{a}={g}" for a, g in plan.items()], "--seed0", str(SEED0), "--out", str(POP),
                "--parallel", str(parallel)]
    chase_f_run.main()


def load_groups(cell, plan):
    model, framing = cell.split("|")
    out = {"OWN": {}, "LOG": {}}
    for line in open(POP):
        r = json.loads(line)
        if r["model"] == model and r["framing"] == framing and r["N"] == 4 and r["arm"] in out \
                and 0 <= r["seed"] - SEED0 < plan[r["arm"]]:
            out[r["arm"]][r["seed"]] = r
    return {a: list(v.values()) for a, v in out.items()}


def cmd_eval(B):
    from scipy.stats import spearmanr
    rule = json.loads(RULE.read_text())
    plan, rng = rule["run"]["plan"], np.random.default_rng(20261125)
    g = {c: load_groups(c, plan) for c in CELLS}
    for c, x in g.items():
        if any(len(x[a]) < plan[a] for a in plan):
            sys.exit(f"incomplete {c}: { {a: len(v) for a, v in x.items()} }; nothing analysed")
    lin = {c: {a: np.array([T.V.struct(r)["linear"] for r in x[a]], float) for a in plan} for c, x in g.items()}
    res = dict(rule_stamp=rule["stamp"], pop_sha256=hashlib.sha256(POP.read_bytes()).hexdigest(), cells={})
    for c in CELLS:
        o = T.outcome(g[c])
        o.update(pred_Y=rule["cells"][c]["pred_Y"], pred_F=rule["cells"][c]["pred_F"], Y=rule["cells"][c]["Y"], F=rule["cells"][c]["F"],
                 lead_rate={a: float(E8.lead_rate(g[c][a])) for a in plan},
                 decided={a: float(np.mean([E8.decided(r) for r in g[c][a]])) for a in plan},
                 invalid={a: int(sum(r["invalid"] for r in g[c][a])) for a in plan})
        res["cells"][c] = o
    boots = {c: np.array([rng.choice(lin[c]["LOG"], len(lin[c]["LOG"])).mean() - rng.choice(lin[c]["OWN"], len(lin[c]["OWN"])).mean()
                          for _ in range(B)]) for c in CELLS}
    tests = {}
    for r in ("Y", "F"):
        gain = [c for c in CELLS if res["cells"][c][f"pred_{r}"] == "gain"]
        null = [c for c in CELLS if c not in gain]
        if not gain or not null:
            tests[r] = dict(tested=False, gain=gain, null=null); continue
        pt = np.mean([res["cells"][c]["diff"] for c in gain]) - np.mean([res["cells"][c]["diff"] for c in null])
        bs = np.mean([boots[c] for c in gain], 0) - np.mean([boots[c] for c in null], 0)
        tests[r] = dict(tested=True, gain=gain, null=null, D=float(pt), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                        p_one_sided=float(np.mean(bs <= 0)))
    live = sorted([r for r in tests if tests[r]["tested"]], key=lambda r: tests[r]["p_one_sided"])
    for k, r in enumerate(live):                                       # Holm over the tested rules
        tests[r]["p_holm"] = float(min(1.0, max(tests[q]["p_one_sided"] * (len(live) - j) for j, q in enumerate(live[:k + 1]))))
        tests[r]["supported"] = bool(tests[r]["ci95"][0] > 0 and tests[r]["p_holm"] < 0.05)
    res["tests"] = tests
    obs = {c: res["cells"][c]["observed"] for c in CELLS}
    both = {c: "gain" if res["cells"][c]["pred_Y"] == res["cells"][c]["pred_F"] == "gain" else "no gain" for c in CELLS}
    res["correct"] = {r: sum(res["cells"][c][f"pred_{r}"] == obs[c] for c in CELLS) for r in ("Y", "F")} | {"Y and F": sum(both[c] == obs[c] for c in CELLS)}
    known = json.loads((R / "chase_predictors_posthoc.json").read_text())["cells"]
    ally = [(res["cells"][c]["Y"], res["cells"][c]["diff"]) for c in CELLS] + [(x["i1_partner_sensitivity"]["mid_minus_strong"], x["observed_diff"]) for x in known.values()]
    allf = [(res["cells"][c]["F"], res["cells"][c]["diff"]) for c in CELLS] + [(x["i2_forecast"]["diff"], x["observed_diff"]) for x in known.values()]
    sp = lambda v: dict(zip(("rho", "p"), map(float, spearmanr([a for a, _ in v], [b for _, b in v]))), cells=len(v))
    res["spearman"] = dict(Y_fresh=sp(ally[:7]), F_fresh=sp(allf[:7]), Y_all=sp(ally), F_all=sp(allf))
    EVAL.write_text(json.dumps(res, indent=1))
    for c, o in res["cells"].items():
        print(f"{c:38s} OWN {o['own']:.2f} LOG {o['log']:.2f} diff {o['diff']:+.2f} p {o['p_one_sided']:.3g} observed {o['observed']:7s} "
              f"pred Y {o['pred_Y']:7s} F {o['pred_F']:7s} LEAD {o['lead_rate']} decided {o['decided']}")
    for r, t in tests.items():
        print(r, t)
    print("correct", res["correct"], "\nspearman", res["spearman"])


def cmd_check():
    known = json.loads((R / "chase_predictors_posthoc.json").read_text())["cells"]
    src, worst = {}, 0.0
    for c, x in known.items():
        s = x["source"]; src.setdefault(s, T.cal_rows(s))
        v = scores(src[s], c)
        dy, df = abs(v["Y"] - x["i1_partner_sensitivity"]["mid_minus_strong"]), abs(v["F"] - x["i2_forecast"]["diff"])
        db = abs(v["F_boot"]["q05"] - x["i2_forecast"]["q05"]) + abs(v["F_boot"]["q95"] - x["i2_forecast"]["q95"])
        worst = max(worst, dy, df, db)
        print(f"{c:38s} Y {v['Y']:+.4f} (screen {x['i1_partner_sensitivity']['mid_minus_strong']:+.4f})  F {v['F']:+.4f} "
              f"(screen {x['i2_forecast']['diff']:+.4f})  boot q05/q95 diff {db:.1e}")
    print("largest difference", worst)
    sys.exit(0 if worst < 1e-12 else 1)


def cmd_prompt():
    print("norm50 sentence:", repr(P.framing_text("norm50")))
    for f in ("comp", "norm100", "norm0"):
        print(f"{f:8s} unchanged:", P.framing_text(f) == _framing_text(f))
    print(P.log_prompt("LOG", "norm50", "B", "C", 5, [(2, "B", "D"), (3, "C", "B")], [(1, "C", "A"), (4, "A", "C")], P.TASKS[0]))


def cmd_smoke(cells, n=20):
    """n real calls per cell on its last n LOG calibration prompts; only whether the reply parses is reported."""
    from llm import call, parse_choice
    for c in cells:
        jobs = [j for j in CC.round_jobs("E15", 1, [c], pd.DataFrame()) if j["arm"] == "LOG"][-n:]
        t0, ok = time.time(), 0
        for j in jobs:
            ch = parse_choice(call(j["model"], j["system"], j["prompt"], tag="chase_e15_smoke"))
            ok += ch is not None
        print(f"{c}: {ok}/{len(jobs)} parsed in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cal", "rule", "pop", "eval", "check", "prompt", "smoke"])
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--plan", nargs="+", default=None, help="tests only (needs CHASE_CAL_DIR): ARM=groups")
    a = ap.parse_args()
    plan = PLAN
    if a.plan:
        if "CHASE_CAL_DIR" not in os.environ:
            sys.exit("--plan is for tests only (set CHASE_CAL_DIR)")
        plan = {x.split("=")[0]: int(x.split("=")[1]) for x in a.plan}
    {"cal": lambda: cmd_cal(a.cells, a.workers), "rule": lambda: cmd_rule(plan), "pop": lambda: cmd_pop(a.cells, a.parallel),
     "eval": lambda: cmd_eval(a.boot), "check": cmd_check, "prompt": cmd_prompt, "smoke": lambda: cmd_smoke(a.cells)}[a.cmd]()


if __name__ == "__main__":
    main()
