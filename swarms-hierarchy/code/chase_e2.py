"""E2: state-matched calibration and frozen forecasts for the Chase populations.

  chase_e2.py calib   --cells "adaption:gemma-4-31b-it|norm100" ... [--controls adaption:deepseek-v4-flash|comp]
      Simulates surrogate groups with kernels fitted on E1 rows only, samples the records those groups visit, and
      queries the LLM there, at the group size and in the arm the population will use.
      -> results/chase_e2_statecal.jsonl   (append; finished job ids are skipped)
  chase_e2.py predict --cells ... [--controls ...]
      Refits every kernel on E1 + state rows with a row bootstrap, simulates the planned number of groups per
      bootstrap draw, and writes the predictive distributions once, SHA-256 stamped.
      -> results/chase_e2_predictions.json

Planned E3 sizes follow amendment C2 in refine-logs/CHASE_LOG.md.
"""
import argparse, hashlib, json, math, multiprocessing, random, sys, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
sys.path.insert(0, str(Path(__file__).parent))
from chase_prompts import SYSTEM, TASKS, log_prompt
from chase_pop import run, KernelDecider, _feat, SCORE_ARMS
from chase_metrics import events_from_history, structure, pooled
import chase_fit as F

ROOT = Path(__file__).resolve().parents[1]
RHO = 0.5
ROUNDS = {4: 24, 8: 40}
# state-matched calls per arm, by group size
CAL = {4: {"DYAD": 150, "OWN": 300, "LOG": 450, "REDACT": 150}, 8: {"OWN": 300, "LOG": 450, "SCORE+LOG": 450}}
CAL_CONTROL = {4: {"OWN": 100, "LOG": 100}, 8: {"OWN": 100, "LOG": 100}}
# planned E3 groups per arm (amendment C2)
PLAN = {4: {"OWN": 40, "LOG": 40, "DYAD": 20, "REDACT": 20}, 8: {"OWN": 6, "LOG": 6, "SCORE+LOG": 6, "SCORE": 6}}
PLAN_CONTROL = {4: {"OWN": 20, "LOG": 20}, 8: {"OWN": 4, "LOG": 4}}
CEILING = 0.85
KEYS = ["linear", "dd_ds", "p_trans", "t_tri", "landau", "dci", "DD", "DS", "BDID", "ISDB"]


class Tracer(KernelDecider):
    """KernelDecider that also records what each deciding agent was shown."""
    def __init__(self, pfun, arm):
        super().__init__(pfun, arm); self.views = []

    def __call__(self, pairs, names, events, t_now, h, N, rng_prompt, rng_sched, mu, Q, Fq):
        for i, j in pairs:
            for a, b in ((i, j), (j, i)):
                f, own, partner, sc = _feat(self.arm, a, b, names, events, t_now, h)
                self.views.append(dict(me=names[a], other=names[b], t_now=t_now, own=own, partner=partner, scores=sc, **f))
        return super().__call__(pairs, names, events, t_now, h, N, rng_prompt, rng_sched, mu, Q, Fq)


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def surrogate_kernels(e1, model, framing, arm):
    """Kernels used only to generate plausible visited records (fitted on E1 rows, never on population data)."""
    if arm == "SCORE+LOG":
        make, _ = F.score_kernel(model, framing)
        ks = make()
        kl = F.fit(F.select(e1, model, framing, "LOG", 8), "LOG", "K4")
        zero = dict(d=0, o=0, p=0, tc=0)
        both = lambda f: 1 / (1 + math.exp(-(_logit(ks(f)) + _logit(kl(f)) - _logit(kl(zero)))))
        return [ks, both]
    # REDACT has no direct-history cells in E1; its surrogate records are generated with the OWN kernel
    d = F.select(e1, model, framing, "OWN" if arm == "REDACT" else arm, 8)
    lv = F.levels(arm)
    return [F.fit(d, arm, lv[-1]), F.fit(d, arm, lv[-2])]


def visited(e1, model, framing, arm, N, seed):
    rng = np.random.default_rng(seed)
    views = []
    for k in surrogate_kernels(e1, model, framing, arm):
        for _ in range(30 if N == 4 else 8):
            dec = Tracer(k, arm)
            run(N, RHO, ROUNDS[N], arm, dec, seed=int(rng.integers(1e9)), keep_feats=False)
            views += dec.views
    return views


def calib_jobs(e1, cells, controls, seed):
    jobs = []
    for group, plan in ((cells, CAL), (controls, CAL_CONTROL)):
        for cell in group:
            model, framing = cell.split("|")
            for N, arms in plan.items():
                for arm, k in arms.items():
                    views = visited(e1, model, framing, arm, N, seed)
                    rng = random.Random(f"{seed}|{cell}|{arm}|{N}")
                    for s in range(k):
                        v = rng.choice(views)
                        jid = f"{model}|{framing}|{arm}|N{N}|s{s}"
                        prompt = log_prompt(arm, framing, v["me"], v["other"], v["t_now"], v["own"], v["partner"],
                                            rng.choice(TASKS), scores=v["scores"], rng=rng)
                        jobs.append(dict(jid=jid, model=model, framing=framing, arm=arm, N=N, design="state",
                                         system=SYSTEM.format(me=v["me"], n=N), prompt=prompt,
                                         **{c: v[c] for c in F.FEATS}))
    return jobs


def cmd_calib(a):
    from llm import call, parse_choice
    e1 = F.load_rows([Path(a.e1)])
    jobs = calib_jobs(e1, a.cells, a.controls, a.seed)
    out = Path(a.statecal)
    done = set()
    if out.exists():
        done = {json.loads(l)["jid"] for l in open(out) if json.loads(l).get("choice")}
    todo = [j for j in jobs if j["jid"] not in done]
    random.Random(a.seed).shuffle(todo)
    print(f"{len(jobs)} jobs, {len(done)} done, {len(todo)} to run; "
          f"{sum(len(j['system']) + len(j['prompt']) for j in jobs) / max(len(jobs), 1):.0f} chars/call", flush=True)
    if a.dry:
        for j in random.Random(1).sample(jobs, min(a.dry, len(jobs))):
            print("=" * 90, "\n", j["jid"], {c: j[c] for c in F.FEATS}, "\n" + j["prompt"])
        return
    lock, f, n = threading.Lock(), open(out, "a"), 0

    def one(j):
        resp, ch, tries = "", None, 0
        for tries in range(1, 4):
            resp = call(j["model"], j["system"], j["prompt"], tag="chase_e2")
            ch = parse_choice(resp)
            if ch:
                break
        return j, resp, ch, tries

    with ThreadPoolExecutor(a.workers) as ex:
        for fu in as_completed([ex.submit(one, j) for j in todo]):
            j, resp, ch, tries = fu.result()
            row = {k: v for k, v in j.items() if k not in ("system", "prompt")}
            with lock:
                f.write(json.dumps({**row, "response": resp, "choice": ch, "tries": tries}) + "\n"); f.flush()
                n += 1
                if n % 1000 == 0:
                    print(n, "done", flush=True)
    f.close(); print("finished", n, flush=True)


# ---------------- frozen forecasts ----------------
_G = {}


def _boot(args):
    """One bootstrap draw: refit the kernel, simulate the planned groups, return per-group structures (compact)."""
    key, level, G, b = args
    model, framing, arm, N = key
    rng = np.random.default_rng([b, N, zlib.crc32(f"{model}|{framing}|{arm}|{level}".encode())])
    if arm == "SCORE":
        k = _G["score"][(model, framing)](rng)
    else:
        k = F.fit(_G["rows"][key], arm, level, rng)
    st = [structure(events_from_history(run(N, RHO, ROUNDS[N], arm, KernelDecider(k, arm),
                                            seed=int(rng.integers(1e9)), keep_feats=False)), N, ROUNDS[N]) for _ in range(G)]
    p = pooled(st)
    return dict(pooled={m: p[m] for m in KEYS}, linear_n=int(sum(s["linear"] for s in st)), complete_n=p["n_complete"],
                t_tri_groups=[s["t_tri"] for s in st], params=getattr(k, "params", None))


def _summ(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return dict(mean=None)
    return dict(mean=float(v.mean()), sd=float(v.std()), q05=float(np.quantile(v, .05)), q95=float(np.quantile(v, .95)),
                samples=[round(float(x), 4) for x in v])


def cmd_predict(a):
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing to overwrite frozen predictions {out}")
    rows = F.load_rows([Path(a.e1), Path(a.statecal)])
    tasks, meta = [], {}
    _G["rows"], _G["score"] = {}, {}
    for group, plan in ((a.cells, PLAN), (a.controls, PLAN_CONTROL)):
        for cell in group:
            model, framing = cell.split("|")
            for N, arms in plan.items():
                for arm, G in arms.items():
                    key = (model, framing, arm, N)
                    if arm == "SCORE":
                        make, n = F.score_kernel(model, framing)
                        _G["score"][(model, framing)] = make
                        lv, nrow, nstate = ["LEGACY"], n, 0
                    else:
                        d = F.select(rows, model, framing, arm, N)
                        _G["rows"][key] = d
                        lv, nrow, nstate = F.levels(arm) + ["LIN", "FLEX"], len(d), int((d.design == "state").sum())
                    meta[key] = dict(model=model, framing=framing, arm=arm, N=N, groups=G, rounds=ROUNDS[N],
                                     n_rows=nrow, n_state_rows=nstate, levels=lv)
                    tasks += [(key, level, G, b) for level in lv for b in range(a.B)]
    print(len(tasks), "bootstrap simulations", flush=True)
    with multiprocessing.get_context("fork").Pool(a.procs) as pool:      # fork: workers inherit _G
        res = pool.map(_boot, tasks, chunksize=4)
    by = {}
    for (key, level, G, b), r in zip(tasks, res):
        by.setdefault((key, level), []).append(r)
    preds = []
    for key, m in meta.items():
        e = dict(m)
        for level in m["levels"]:
            rs = by[(key, level)]
            e[level] = {k: _summ([r["pooled"][k] for r in rs]) for k in KEYS}
            e[level]["linear_n"] = [r["linear_n"] for r in rs]
            e[level]["complete_n"] = [r["complete_n"] for r in rs]
            if rs[0]["params"]:
                ks = rs[0]["params"].keys()
                e[level]["params"] = {k: dict(mean=float(np.mean([r["params"][k] for r in rs])),
                                              sd=float(np.std([r["params"][k] for r in rs]))) for k in ks}
        preds.append(e)
    # LOG-versus-OWN contrasts, power and the ceiling rule, per cell
    contrasts = []
    top = lambda arm: F.ARM_MAX[arm]
    for cell in a.cells + a.controls:
        model, framing = cell.split("|")
        c = dict(model=model, framing=framing)
        k4o, k4l = (model, framing, "OWN", 4), (model, framing, "LOG", 4)
        if (k4o, top("OWN")) in by:
            o, l = by[(k4o, top("OWN"))], by[(k4l, top("LOG"))]
            G = meta[k4o]["groups"]
            ps = [fisher_exact([[x["linear_n"], G - x["linear_n"]], [y["linear_n"], G - y["linear_n"]]])[1] for x, y in zip(o, l)]
            c["N4"] = dict(own_linear=float(np.mean([x["linear_n"] for x in o]) / G), log_linear=float(np.mean([y["linear_n"] for y in l]) / G),
                           diff=_summ([(y["linear_n"] - x["linear_n"]) / G for x, y in zip(o, l)]),
                           power_fisher_05=float(np.mean(np.array(ps) < 0.05)),
                           # what LOG would look like if agents ignored bystander features (H4)
                           log_linear_without_bystander=float(np.mean([y["linear_n"] for y in by[(k4l, "K2")]]) / G),
                           ceiling=bool(np.mean([x["linear_n"] for x in o]) / G > CEILING))
        k8o, k8l = (model, framing, "OWN", 8), (model, framing, "LOG", 8)
        if (k8o, top("OWN")) in by:
            o, l = by[(k8o, top("OWN"))], by[(k8l, top("LOG"))]
            ps = []
            for x, y in zip(o, l):
                xa, ya = np.nan_to_num(x["t_tri_groups"]), np.nan_to_num(y["t_tri_groups"])
                ps.append(mannwhitneyu(ya, xa, alternative="two-sided").pvalue if len(set(xa) | set(ya)) > 1 else 1.0)
            c["N8"] = dict(own_t_tri=float(np.mean([x["pooled"]["t_tri"] for x in o])), log_t_tri=float(np.mean([y["pooled"]["t_tri"] for y in l])),
                           diff=_summ([y["pooled"]["t_tri"] - x["pooled"]["t_tri"] for x, y in zip(o, l)]),
                           power_mannwhitney_05=float(np.mean(np.array(ps) < 0.05)),
                           log_t_tri_without_bystander=float(np.mean([y["pooled"]["t_tri"] for y in by[(k8l, "K2")]])))
        contrasts.append(c)
    res = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), stage="E2 frozen forecasts", B=a.B, rho=RHO, rounds=ROUNDS,
               plan=dict(cells=PLAN, controls=PLAN_CONTROL), ceiling_rule=CEILING, cells=a.cells, controls=a.controls,
               predictions=preds, contrasts=contrasts)
    files = ["chase_prompts.py", "chase_pop.py", "chase_fit.py", "chase_e2.py", "chase_metrics.py", "llm.py", "prompts_v2.py"]
    res["source_sha256"] = {f: hashlib.sha256((Path(__file__).parent / f).read_bytes()).hexdigest() for f in files}
    res["data_sha256"] = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in (Path(a.e1), Path(a.statecal)) if Path(p).exists()}
    res["sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    out.write_text(json.dumps(res))
    print("written", out, res["sha256"])
    for c in contrasts:
        print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (vv if not isinstance(vv, dict) else {x: vv[x] for x in ("mean", "q05", "q95") if x in vv}) for kk, vv in v.items()}) for k, v in c.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "predict"])
    ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--controls", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry", type=int, default=0)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--e1", default=str(ROOT / "results" / "chase_e1_kernel.jsonl"))
    ap.add_argument("--statecal", default=str(ROOT / "results" / "chase_e2_statecal.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "chase_e2_predictions.json"))
    a = ap.parse_args()
    (cmd_calib if a.cmd == "calib" else cmd_predict)(a)


if __name__ == "__main__":
    main()
