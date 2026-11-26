"""Couzin study, amendment M-A1: the M4b design run on Claude models through the `claude` CLI
(refine-logs/COUZIN_CLAUDE_FOLLOWUP.md). Uses the frozen engine (couzin.py) unchanged; results go to
results/couzin_claude/ (COUZIN_RESULTS must point there; the queue sets it).

Stop-on-quota rule. llm.py turns a CLI usage-limit / login message into an empty reply, and the game would then keep
the agent's previous choice: a quota hit would silently bias groups toward staying put. Here every quota message
llm.py detects sets a process-wide flag; from then on every model call raises QuotaStop before or after the call,
so no row containing a quota-affected decision is written (a group is appended only after all of its decisions
succeed). The process exits with code 75 and the queue stops.

Usage (one model per call except gate):
  couzin_followup.py gate     --models m1 m2      M0 items + label screen for the Claude models, write-once gate
  couzin_followup.py kernel   --model m           M1 main cells, 20 draws
  couzin_followup.py rules    --model m           omega rule + nonlinearity index (adds the model, write-once per model)
  couzin_followup.py calib    --model m           M2c_obs_nu6 calibration, 8 groups
  couzin_followup.py forecast --model m           Hartnett check + write-once forecast for the model's M4b arms
  couzin_followup.py pop      --model m           the M4b arms, 24 groups each
  couzin_followup.py eval                         F1-F3 for every model with complete arms -> couzin_followup_eval.json
                                                  (M-A3: no test from incomplete n_u levels; Holm only over computed F1s)
  couzin_followup.py stamp | check                hash stamp of the amendment files (write-once) / check
"""
import argparse, hashlib, json, math, os, sys, threading, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import llm
import couzin as cz

QUOTA = threading.Event()
QUOTA_EXIT = 75


class QuotaStop(RuntimeError):
    pass


# ---------------------------------------------------------------- stop-on-quota guard
_orig_is_quota, _orig_call, _orig_fake = llm._is_quota_msg, llm.call, cz.fake_reply
_fake_n = [0]; _fake_lock = threading.Lock()


def _is_quota_msg(s):
    hit = _orig_is_quota(s)
    if hit:
        QUOTA.set()
    return hit


def _guarded_call(*a, **k):
    if QUOTA.is_set():
        raise QuotaStop("usage limit reached earlier in this run")
    out = _orig_call(*a, **k)
    if QUOTA.is_set():
        raise QuotaStop("usage limit reached")
    model = a[0] if a else k.get("model", "")
    if not out and str(model).startswith(("claude:", "gpt:")):   # M-A4: a CLI that fails silently (codex writes no
        QUOTA.set()                                              # output on a usage limit) also stops the run
        raise QuotaStop(f"empty reply from the {model.split(':')[0]} CLI after retries (possible usage limit)")
    return out


def _guarded_fake(*a, **k):        # offline test of the guard: COUZIN_FAKE_QUOTA_AFTER=n
    lim = int(os.environ.get("COUZIN_FAKE_QUOTA_AFTER", "0") or 0)
    if QUOTA.is_set():
        raise QuotaStop("usage limit reached earlier in this run (fake)")
    with _fake_lock:
        _fake_n[0] += 1
        if lim and _fake_n[0] > lim:
            QUOTA.set()
            raise QuotaStop("usage limit reached (fake)")
    return _orig_fake(*a, **k)


cz.FAKE_PARAMS.setdefault("haiku", dict(a=2.2, pers=0.7, lab=0.1))     # scripted stand-ins (offline test only)
cz.FAKE_PARAMS.setdefault("sonnet", dict(a=1.4, pers=1.2, lab=-0.1))
cz.FAKE_PARAMS.setdefault("gpt-", dict(a=1.8, pers=0.9, lab=0.05))
llm._is_quota_msg = _is_quota_msg
llm.call = _guarded_call
cz.fake_reply = _guarded_fake

R = cz.R
ARMS = ["M4b_sim_nu0", "M4b_sim_nu12", "M4b_sim_nu24", "M4b_obs_nu12", "M4b_obs_nu24"]
GROUPS, CAL_GROUPS, DRAWS = 24, 8, 20
FLAT = 0.05                       # |forecast change in P(minority wins) from n_u 0 to 24| below this = flat forecast
FILES = ["refine-logs/COUZIN_CLAUDE_FOLLOWUP.md", "refine-logs/COUZIN_FOLLOWUP2.md", "code/couzin_followup.py",
         "code/couzin_claude_run.sh", "code/couzin_cli_run.sh", "code/couzin_queue3.sh", "code/couzin_exact.py",
         "code/couzin.py", "code/couzin_kernel.py", "code/couzin_forecast.py", "code/couzin_eval.py", "code/llm.py"]
ROOT = Path(__file__).resolve().parents[1]
STAMP = ROOT / "results" / "couzin_followup_freeze.json"
short = lambda m: m.split(":", 1)[1].replace(".", "_")


def _scale():
    return max(1, int(os.environ.get("DRYF", "1")))


# ---------------------------------------------------------------- stages
def gate(models):
    import couzin_kernel as ck
    cz.run_quiz(models, DRAWS, 8, R / "couzin_m0_quiz.jsonl")
    cz.run_labels(models, DRAWS, 8, R / "couzin_m0_labels.jsonl")
    if not (R / "couzin_m0_gate.json").exists():
        ck.gate(models)


def seed(model):
    """M-A4 (Qwen): reuse the main study's frozen M0 gate (label pair ORION/LYRA) and its M1 kernel rows for this
    model (20 draws per cell, same cells and labels) instead of measuring them again. Write-once."""
    main = ROOT / "results"
    g = R / "couzin_m0_gate.json"
    if not g.exists():
        d = json.loads((main / "couzin_m0_gate.json").read_text())
        if model not in d["passed"]:
            sys.exit(f"{model} did not pass the main M0 gate")
        g.write_text(json.dumps(d, indent=1))
    k = R / "couzin_m1_kernel.jsonl"
    if not k.exists():
        rows = [l for l in open(main / "couzin_m1_kernel.jsonl") if json.loads(l)["model"] == model]
        k.write_text("".join(rows))
        print(f"seeded {len(rows)} M1 rows for {model} from the main study")


def rules(model):
    import couzin_kernel as ck
    f = R / "couzin_m1_rules.json"
    d = json.loads(f.read_text()) if f.exists() else dict(omega={}, nonlinearity={})
    if model in d["omega"]:
        print(f"rules for {model} exist (write-once)"); return
    d["omega"][model] = ck.omega_rule(model)
    d["nonlinearity"][model] = ck.nonlinearity(model, calibration=False)
    f.write_text(json.dumps(d, indent=1))
    print(json.dumps({k: v[model] for k, v in d.items()}, indent=1))


def forecast(model, draws=200, per=100, seed=2027):
    import couzin_kernel as ck, couzin_forecast as cf
    out = R / f"couzin_forecast_{short(model)}.json"
    if out.exists():
        print(f"{out} exists (write-once)"); return
    hc = cf.hartnett_check()
    if not hc["passed"]:
        sys.exit("Hartnett check failed; no forecast")
    rng = np.random.default_rng(seed)
    T = {model: ck.tables(model, robust_rows=False)}
    Ep = {model: ck.estimates(T[model])}
    Ed = [{model: ck.estimates(T[model], rng)} for _ in range(draws)]
    om = {model: cz.frozen_omegas(model)}
    arms = {}
    for aid in ARMS:
        arms[aid] = dict(model=model, **cf.forecast_arm(aid, model, None, Ed, Ep, om, GROUPS, rng, per))
        print(f"{aid}: majA {arms[aid]['majA']['point']:.3f} [{arms[aid]['majA']['lo']:.2f}, {arms[aid]['majA']['hi']:.2f}]", flush=True)
    nus = [0, 12, 24]
    slopes = {pr: float(np.polyfit(nus, [arms[a]["majA"]["point"] for a in ["M4b_sim_nu0", f"M4b_{pr}_nu12", f"M4b_{pr}_nu24"]], 1)[0])
              for pr in ("sim", "obs")}
    h = hashlib.sha256(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else {str(w): x.tolist() for w, x in v.items()})
                                   for k, v in T[model].items()}, sort_keys=True).encode()).hexdigest()
    d = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), model=model, hartnett=hc, omegas=om[model],
             kernel_counts_sha256=h, forecast_slopes=slopes, arms=arms, draws=draws, per=per)
    out.write_text(json.dumps(d, indent=1))
    print(f"forecast written: {out}; slopes {slopes}")


def pop(model):
    n = math.ceil(GROUPS / _scale())
    for aid in ARMS:
        cz.run_pop([aid], model, list(range(1, n + 1)), 6, R / "couzin_pop.jsonl")


def evaluate():
    from couzin_eval import wald_slope, wilson_upper, holm, vals
    from scipy.stats import norm
    groups = {}
    for r in cz.rows(R / "couzin_pop.jsonl"):
        if r["arm"].startswith("M4b"):
            groups.setdefault((r["model"], r["arm"]), []).append(r)
    res = {}
    for f in sorted(R.glob("couzin_forecast_*.json")):
        F = json.loads(f.read_text()); m = F["model"]
        G = {a: g for (mm, a), g in groups.items() if mm == m}
        if not G:
            continue
        o = dict(arms={}, tests={})
        for a in ARMS:
            v = vals(G, a, "majA")
            if len(v):
                fa = F["arms"][a]["majA"]
                o["arms"][a] = dict(groups=len(v), majA=float(v.mean()), lo=fa["lo"], hi=fa["hi"], point=fa["point"],
                                    inside=bool(fa["lo"] <= v.mean() <= fa["hi"]),
                                    cohesion=float(vals(G, a, "cohesion").mean()),
                                    invalid_rate=sum(g["invalid"] for g in G[a]) / max(1, sum(g["decisions"] for g in G[a])))
        need = math.ceil(GROUPS / _scale())
        n_of = lambda a: len(vals(G, a, "majA"))
        for pr in ("sim", "obs"):           # M-A3: tests only when all three n_u levels are complete
            arms = ["M4b_sim_nu0", f"M4b_{pr}_nu12", f"M4b_{pr}_nu24"]
            short_arms = {a: n_of(a) for a in arms if n_of(a) < need}
            if short_arms:
                for k in ("F1", "F2"):
                    o["tests"][f"{k}{pr}"] = dict(status=f"incomplete: {short_arms} of {need} groups; no test")
                continue
            X, y = [], []
            for a, nu in zip(arms, (0, 12, 24)):
                for v in vals(G, a, "majA"):
                    X.append([nu, 1.0]); y.append(v)
            fs = F["forecast_slopes"][pr]
            if len(set(y)) < 2:
                msg = f"no variation: every group scored {y[0]}; slope not estimable"
                o["tests"][f"F1{pr}"] = dict(forecast_slope=fs, forecast_change_0_24=24 * fs, status=msg)
                o["tests"][f"F2{pr}"] = dict(status=msg + "; not supported")
                continue
            w = wald_slope(np.array(X), np.array(y))
            t = dict(forecast_slope=fs, forecast_change_0_24=24 * fs, observed_slope=w["slope"], z=w["z"],
                     observed_slope_ci90=[w["slope"] - 1.645 * w["se"], w["slope"] + 1.645 * w["se"]])
            if abs(24 * fs) >= FLAT:      # sign test only when the forecast predicts a change of >= 5 points
                t.update(same_sign=bool(np.sign(fs) == np.sign(w["slope"])), p=float(1 - norm.cdf(np.sign(fs) * w["z"])))
            else:
                t.update(status="flat forecast: no sign test; coverage of the arm intervals is the test")
            o["tests"][f"F1{pr}"] = t
            o["tests"][f"F2{pr}"] = dict(**wald_slope(np.array(X), np.array(y), direction=-1))
        if n_of("M4b_obs_nu12") >= need and n_of("M4b_obs_nu24") >= need:
            v = np.concatenate([vals(G, "M4b_obs_nu12", "majA"), vals(G, "M4b_obs_nu24", "majA")])
            up = wilson_upper(v.sum(), len(v))
            o["tests"]["F3"] = dict(p_minority=float(v.mean()), upper95=up, supported=bool(up < .5))
        else:
            o["tests"]["F3"] = dict(status="incomplete: observe-first arms not complete; no test")
        o["coverage"] = f"{sum(a['inside'] for a in o['arms'].values())}/{len(o['arms'])}"
        res[m] = o
    ps = {f"{m}|{k}": t["p"] for m, o in res.items() for k, t in o["tests"].items() if k.startswith("F1") and "p" in t}
    for k, v in holm(ps).items():
        m, t = k.split("|"); res[m]["tests"][t]["p_holm"] = v
    (R / "couzin_followup_eval.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:3000])


def stamp(check):
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    if not check:
        if STAMP.exists():
            sys.exit(f"{STAMP} exists (write-once)")
        STAMP.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "sha256": {f: sha(f) for f in FILES}}, indent=1))
        print(STAMP.read_text()); return
    d = json.loads(STAMP.read_text())["sha256"]
    bad = [f for f, h in d.items() if sha(f) != h]
    print("amendment files unchanged" if not bad else f"CHANGED: {bad}")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gate", "seed", "kernel", "rules", "calib", "forecast", "pop", "eval", "stamp", "check"])
    ap.add_argument("--model"); ap.add_argument("--models", nargs="+")
    a = ap.parse_args()
    if a.stage in ("stamp", "check"):
        return stamp(a.stage == "check")
    if a.stage != "eval" and R.resolve() == (ROOT / "results").resolve():
        sys.exit("COUZIN_RESULTS must point at a follow-up folder (results/couzin_<family>), not the main results")
    try:
        if a.stage == "gate":
            gate(a.models)
        elif a.stage == "seed":
            seed(a.model)
        elif a.stage == "kernel":
            cz.run_kernel([a.model], DRAWS, 8, R / "couzin_m1_kernel.jsonl")
        elif a.stage == "rules":
            rules(a.model)
        elif a.stage == "calib":
            cz.run_pop(["M2c_obs_nu6"], a.model, list(range(1, math.ceil(CAL_GROUPS / _scale()) + 1)), 4, R / "couzin_pop.jsonl")
        elif a.stage == "forecast":
            forecast(a.model, *((40, 50) if cz.FAKE else ()))
        elif a.stage == "pop":
            pop(a.model)
        else:
            evaluate()
    except QuotaStop as e:
        print(f"=== QUOTA STOP: {e}; no row with an affected decision was written", flush=True)
        os._exit(QUOTA_EXIT)
    if QUOTA.is_set():
        print("=== QUOTA STOP after the stage", flush=True)
        os._exit(QUOTA_EXIT)


if __name__ == "__main__":
    main()
