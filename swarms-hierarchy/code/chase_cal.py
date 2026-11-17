"""Calibration in rounds and frozen forecasts for the follow-up experiments (amendment C8: E6 part B, E7, E8b).

  chase_cal.py round   E6|E7|E8     one calibration round for every cell not yet stopped, then point forecasts
                                    exit 0: another round is due; exit 3: every cell has stopped
  chase_cal.py predict E6|E7|E8     write-once forecast file (row-bootstrap refits)
  chase_cal.py status  E6|E7|E8     print the trace

A round, per cell, arm and group size:
  1. fit decision models on this experiment's rows from earlier rounds;
  2. simulate groups with them and collect every record a deciding agent was shown;
  3. query the model at k of those records: half drawn uniformly, half in equal numbers from record bins
     (round 1 of E7 and E8 is uniform and uses kernels fitted on E1 rows, as in E2).
After the round, point forecasts (linear share at N = 4, t_tri at N = 8; fixed simulation seeds) are stored. A cell
stops when all of its tracked forecasts moved by less than 0.03 since the previous round, or after round 4.

Decision models. "K": the frozen form of chase_fit (DYAD d; OWN, REDACT d, o; LOG d, o, p, tc). "X": K plus the six
features selected in E6 part A (hinge, n_own, last, streak, idlow, idme; chase_e6a.extras), for every arm that shows
the own record (OWN, REDACT, LOG). X is the headline for those arms, K for DYAD. Fits use only this experiment's
state-matched rows at the population's own group size; rows without the six features cannot be used.
"""
import hashlib, json, multiprocessing, os, random, sys, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
sys.path.insert(0, str(Path(__file__).parent))
import chase_e2 as E
import chase_fit as F
from chase_e6a import XDecider, extras
from chase_metrics import events_from_history, pooled, structure
from chase_pop import _feat, run
from chase_prompts import SYSTEM, TASKS, log_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("CHASE_CAL_DIR", ROOT / "results"))          # tests redirect output
E1 = ROOT / "results" / "chase_e1_kernel.jsonl"
QWEN, G31, G26 = "adaption:qwen3.5-397b", "adaption:gemma-4-31b-it", "adaption:gemma-4-26b-a4b-it"
EXP = {
    "E6": dict(cells=[f"{G31}|norm100", f"{G26}|norm0"], e1_round1=False,
               round1={4: {"OWN": 300}, 8: {"OWN": 300}}, later={4: {"OWN": 300}, 8: {"OWN": 300}},
               plan={4: {"OWN": [40]}, 8: {"OWN": [6]}}),
    "E7": dict(cells=[f"{QWEN}|norm100", f"{QWEN}|comp"], e1_round1=True,
               round1={4: {"DYAD": 150, "OWN": 300, "LOG": 450, "REDACT": 150}, 8: {"OWN": 300, "LOG": 450}},
               later={4: {"OWN": 150, "LOG": 225, "REDACT": 75}, 8: {"OWN": 150, "LOG": 225}},
               plan={4: {"OWN": [40, 80, 120], "LOG": [40, 80, 120], "DYAD": [20], "REDACT": [20]}, 8: {"OWN": [8], "LOG": [8]}}),
    "E8": dict(cells=[f"{QWEN}|norm0"], e1_round1=True,
               round1={4: {"OWN": 300, "LOG": 450}}, later={4: {"OWN": 150, "LOG": 225}},
               plan={4: {"OWN": [80], "LOG": [80]}}),
}
BASE = {"DYAD": ["d"], "OWN": ["d", "o"], "REDACT": ["d", "o"], "LOG": ["d", "o", "p", "tc"]}
EXTRA = ["hinge", "n_own", "last", "streak", "idlow", "idme"]
MAX_ROUNDS, TOL = 4, 0.03
POINT_G = {4: 2000, 8: 150}
KEYS = ["linear", "dd_ds", "p_trans", "t_tri", "landau", "DD", "DS", "BDID", "ISDB"]


def cols(arm, form):
    if form == "K" or arm == "DYAD":
        return BASE[arm]
    return (["d", "o"] if form == "Xnb" else BASE[arm]) + EXTRA        # Xnb: LOG without the bystander features


def forms(arm):
    return ["K"] if arm == "DYAD" else ["X", "K", "Xnb"] if arm == "LOG" else ["X", "K"]


def paths(exp):
    return OUT / f"chase_{exp.lower()}_statecal.jsonl", OUT / f"chase_{exp.lower()}_rounds.json", OUT / f"chase_{exp.lower()}_predictions.json"


def load(exp):
    p = paths(exp)[0]
    rows = [json.loads(l) for l in open(p)] if p.exists() else []
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df = df.drop_duplicates("jid", keep="last")
    df = df[df.choice.notna()].copy()
    df["lead"] = (df.choice == "LEAD").astype(float); df["off"] = 0.0
    return df


def sel(df, cell, arm, N, upto=99):
    model, framing = cell.split("|")
    if not len(df):
        return df
    return df[(df.model == model) & (df.framing == framing) & (df.arm == arm) & (df.N == N) & (df["round"] <= upto)]


class XTracer(XDecider):
    """XDecider that records what each deciding agent was shown, with the extra features."""
    def __init__(self, pfun, arm):
        super().__init__(pfun, arm); self.views = []

    def __call__(self, pairs, names, events, t_now, h, N, rng_prompt, rng_sched, mu, Q, Fq):
        out = []
        for i, j in pairs:
            ps = []
            for a, b in ((i, j), (j, i)):
                f, own, partner, sc = _feat(self.arm, a, b, names, events, t_now, h)
                f.update(extras(names[a], names[b], own, f))
                self.views.append(dict(me=names[a], other=names[b], t_now=t_now, own=own, partner=partner, scores=sc, **f))
                ps.append(self.pfun(f))
            out.append(("LEAD" if rng_sched.random() < ps[0] else "DEFER", "LEAD" if rng_sched.random() < ps[1] else "DEFER"))
        return out, []


def obin(o):
    return 0 if o <= -9 else 1 if o <= -6 else 2 if o <= -3 else 3 if o <= -1 else 4 if o == 0 else 5 if o <= 2 else 6 if o <= 5 else 7


def tri(x):
    return 0 if x <= -3 else 2 if x >= 3 else 1


def surrogates(exp, cell, arm, N, rnd, df):
    """Kernels that generate the records to query in round `rnd`."""
    model, framing = cell.split("|")
    if rnd == 1 and EXP[exp]["e1_round1"]:
        return E.surrogate_kernels(F.load_rows([E1]), model, framing, arm)
    if rnd == 1:                                                         # E6: frozen form on every existing row
        rows = F.load_rows([E1, ROOT / "results" / "chase_e2_statecal.jsonl", ROOT / "results" / "chase_e5_statecal.jsonl"])
        d, lv = F.select(rows, model, framing, arm, N), F.levels(arm)
        return [F.fit(d, arm, lv[-1]), F.fit(d, arm, lv[-2])]
    d = sel(df, cell, arm, N, rnd - 1)
    return [F.fit_logistic(d, cols(arm, f)) for f in forms(arm)[:2]]


def round_jobs(exp, rnd, cells, df):
    cfg, jobs = EXP[exp], []
    plan = cfg["round1"] if rnd == 1 else cfg["later"]
    uniform_only = rnd == 1 and cfg["e1_round1"]
    for cell in cells:
        model, framing = cell.split("|")
        for N, arms in plan.items():
            for arm, k in arms.items():
                rng_sim = np.random.default_rng([rnd, N, zlib.crc32(f"{exp}|{cell}|{arm}".encode())])
                views = []
                for kern in surrogates(exp, cell, arm, N, rnd, df):
                    for _ in range((30 if N == 4 else 8) if uniform_only else (60 if N == 4 else 12)):
                        dec = XTracer(kern, arm)
                        run(N, E.RHO, E.ROUNDS[N], arm, dec, seed=int(rng_sim.integers(1e9)), keep_feats=False)
                        views += dec.views
                rng = random.Random(f"{exp}|{cell}|{arm}|{N}|r{rnd}")
                picks = [(rng.choice(views), False) for _ in range(k if uniform_only else k - k // 2)]
                if not uniform_only:
                    bins = {}
                    for v in views:
                        bins.setdefault((tri(v["o"]), tri(v["p"])) if arm == "LOG" else obin(v["o"]), []).append(v)
                    keys = sorted(bins)
                    for n in range(k // 2):
                        picks.append((rng.choice(bins[keys[n % len(keys)]]), True))
                for s, (v, strat) in enumerate(picks):
                    prompt = log_prompt(arm, framing, v["me"], v["other"], v["t_now"], v["own"], v["partner"],
                                        rng.choice(TASKS), scores=v["scores"], rng=rng)
                    jobs.append(dict(jid=f"{exp}|{model}|{framing}|{arm}|N{N}|r{rnd}s{s}", exp=exp, round=rnd, strat=strat,
                                     model=model, framing=framing, arm=arm, N=N, design="state",
                                     system=SYSTEM.format(me=v["me"], n=N), prompt=prompt,
                                     **{c: v[c] for c in F.FEATS}, **{c: v[c] for c in EXTRA if c not in F.FEATS}))
    return jobs


def fake_call(job):
    """Test double (CHASE_CAL_FAKE=1): a fixed stochastic rule in place of the model. Never used in a real run."""
    z = 1.2 + 0.9 * F.slog(job["d"]) + 0.8 * F.slog(job["o"]) - 0.7 * F.slog(job["p"]) + 0.5 * (job["streak"] > 1) - 1.5 * (job["hinge"] < 0)
    r = random.Random(job["jid"]).random()
    return "LEAD" if r < 1 / (1 + np.exp(-z)) else "DEFER"


def run_jobs(jobs, out, tag, workers=6):
    done = set()
    if out.exists():
        done = {json.loads(l)["jid"] for l in open(out) if json.loads(l).get("choice")}
    todo = [j for j in jobs if j["jid"] not in done]
    random.Random(1).shuffle(todo)
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run; "
          f"{sum(len(j['system']) + len(j['prompt']) for j in jobs) / max(len(jobs), 1):.0f} chars/call", flush=True)
    fake = os.environ.get("CHASE_CAL_FAKE") == "1"
    if not fake:
        from llm import call, parse_choice
    lock, f, n = threading.Lock(), open(out, "a"), 0

    def one(j):
        if fake:
            c = fake_call(j)
            return j, c, c, 1
        resp, ch, tries = "", None, 0
        for tries in range(1, 4):
            resp = call(j["model"], j["system"], j["prompt"], tag=tag)
            ch = parse_choice(resp)
            if ch:
                break
        return j, resp, ch, tries

    with ThreadPoolExecutor(workers) as ex:
        for fu in as_completed([ex.submit(one, j) for j in todo]):
            j, resp, ch, tries = fu.result()
            row = {k: v for k, v in j.items() if k not in ("system", "prompt")}
            with lock:
                f.write(json.dumps({**row, "response": resp, "choice": ch, "tries": tries}) + "\n"); f.flush()
                n += 1
    f.close(); print("finished", n, flush=True)


def simulate(kern, arm, N, G, seed):
    rng = np.random.default_rng(seed)
    return [structure(events_from_history(run(N, E.RHO, E.ROUNDS[N], arm, XDecider(kern, arm), seed=int(rng.integers(1e9)),
                                              keep_feats=False)), N, E.ROUNDS[N]) for _ in range(G)]


def tracked(exp):
    return [(N, arm) for N, arms in EXP[exp]["plan"].items() for arm in arms if arm in ("OWN", "LOG")]


def point(exp, cell, df, rnd):
    out = {}
    for N, arm in tracked(exp):
        d = sel(df, cell, arm, N, rnd)
        p = pooled(simulate(F.fit_logistic(d, cols(arm, "X")), arm, N, POINT_G[N], [7, N, len(arm)]))   # same seeds every round
        out[f"{arm}|N{N}"] = p["linear"] if N == 4 else p["t_tri"]
    return out


def cmd_round(exp):
    statecal, tracef, _ = paths(exp)
    trace = json.loads(tracef.read_text()) if tracef.exists() else {c: dict(rounds_done=0, stopped=False, forecasts=[]) for c in EXP[exp]["cells"]}
    live = [c for c in EXP[exp]["cells"] if not trace[c]["stopped"]]
    if not live:
        print("every cell has stopped"); return 3
    rnd = min(trace[c]["rounds_done"] for c in live) + 1
    cells = [c for c in live if trace[c]["rounds_done"] == rnd - 1]
    print(f"=== {exp} round {rnd}: {cells}", flush=True)
    run_jobs(round_jobs(exp, rnd, cells, load(exp)), statecal, "chase_" + exp.lower())
    df = load(exp)
    for c in cells:
        f = point(exp, c, df, rnd)
        prev = trace[c]["forecasts"][-1]["point"] if trace[c]["forecasts"] else None
        move = max(abs(f[k] - prev[k]) for k in f) if prev else None
        trace[c]["forecasts"].append(dict(round=rnd, point=f, largest_move=move))
        trace[c]["rounds_done"] = rnd
        trace[c]["stopped"] = bool(rnd >= MAX_ROUNDS or (move is not None and move < TOL))
        print(c, "round", rnd, {k: round(v, 3) for k, v in f.items()}, "largest move", None if move is None else round(move, 3),
              "stopped" if trace[c]["stopped"] else "continues", flush=True)
    tracef.write_text(json.dumps(trace, indent=1))
    return 3 if all(trace[c]["stopped"] for c in EXP[exp]["cells"]) else 0


_G = {}


def _boot(args):
    key, form, G, b = args
    cell, arm, N = key
    rng = np.random.default_rng([b, N, zlib.crc32(f"{cell}|{arm}|{form}".encode())])
    k = F.fit_logistic(_G[key], cols(arm, form), rng)
    st = simulate(k, arm, N, G, int(rng.integers(1e9)))
    return dict(linear=[bool(s["linear"]) for s in st], t_tri=[s["t_tri"] for s in st], st=[{m: s[m] for m in ("seq", "transitive", "cyclic", "complete", "linear", "landau", "dci")} for s in st])


def _summ(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return dict(mean=float(v.mean()), q05=float(np.quantile(v, .05)), q95=float(np.quantile(v, .95))) if len(v) else dict(mean=None)


def cmd_predict(exp, B=200, procs=None):
    statecal, tracef, out = paths(exp)
    if out.exists():
        sys.exit(f"refusing to overwrite frozen predictions {out}")
    trace = json.loads(tracef.read_text())
    if not all(trace[c]["stopped"] for c in EXP[exp]["cells"]):
        sys.exit("calibration rounds are not finished")
    df, tasks, meta = load(exp), [], {}
    for cell in EXP[exp]["cells"]:
        for N, arms in EXP[exp]["plan"].items():
            for arm, sizes in arms.items():
                key = (cell, arm, N)
                _G[key] = sel(df, cell, arm, N)
                meta[key] = dict(model=cell.split("|")[0], framing=cell.split("|")[1], arm=arm, N=N, sizes=sizes, rounds=E.ROUNDS[N],
                                 n_rows=int(len(_G[key])), headline=forms(arm)[0], forms=forms(arm))
                tasks += [(key, form, max(sizes), b) for form in forms(arm) for b in range(B)]
    print(len(tasks), "bootstrap simulations", flush=True)
    with multiprocessing.get_context("fork").Pool(procs or max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(_boot, tasks, chunksize=4)
    by = {}
    for (key, form, G, b), r in zip(tasks, res):
        by.setdefault((key, form), []).append(r)

    def stat(r, n, N):
        if N == 4:
            return float(np.mean(r["linear"][:n]))
        tr, cy = sum(s["transitive"] for s in r["st"][:n]), sum(s["cyclic"] for s in r["st"][:n])
        return 4 * (tr / (tr + cy) - 0.75) if tr + cy else float("nan")

    def ddds(r, n):
        seq = np.sum([[s["seq"][k] for k in ("DD", "DS", "BDID", "ISDB")] for s in r["st"][:n]], 0)
        return float((seq[0] + seq[1]) / seq.sum()) if seq.sum() else float("nan")
    preds, contrasts = [], []
    for key, m in meta.items():
        e = dict(m)
        for form in m["forms"]:
            e[form] = {str(n): dict(headline=_summ([stat(r, n, m["N"]) for r in by[(key, form)]]),
                                    dd_ds=_summ([ddds(r, n) for r in by[(key, form)]])) for n in m["sizes"]}
        preds.append(e)
    for cell in EXP[exp]["cells"]:
        c = dict(model=cell.split("|")[0], framing=cell.split("|")[1])
        for N in EXP[exp]["plan"]:
            ko, kl = (cell, "OWN", N), (cell, "LOG", N)
            if (kl, "X") not in by:
                continue
            o, l, nb = by[(ko, "X")], by[(kl, "X")], by[(kl, "Xnb")]
            c[f"N{N}"] = {}
            for n in meta[ko]["sizes"]:
                if N == 4:
                    ps = [fisher_exact([[sum(y["linear"][:n]), n - sum(y["linear"][:n])], [sum(x["linear"][:n]), n - sum(x["linear"][:n])]])[1] for x, y in zip(o, l)]
                else:
                    ps = []
                    for x, y in zip(o, l):
                        xa, ya = np.nan_to_num(x["t_tri"][:n]), np.nan_to_num(y["t_tri"][:n])
                        ps.append(mannwhitneyu(ya, xa).pvalue if len(set(xa) | set(ya)) > 1 else 1.0)
                c[f"N{N}"][str(n)] = dict(own=float(np.mean([stat(x, n, N) for x in o])), log=float(np.mean([stat(y, n, N) for y in l])),
                                          diff=_summ([stat(y, n, N) - stat(x, n, N) for x, y in zip(o, l)]),
                                          power_05=float(np.mean(np.array(ps) < 0.05)),
                                          log_without_bystander=float(np.mean([stat(y, n, N) for y in nb])))
        if exp == "E7" and "N4" in c:                                    # sizing rule of amendment C8
            pw = {n: c["N4"][str(n)]["power_05"] for n in (40, 80, 120)}
            ok = [n for n in (40, 80, 120) if pw[n] >= 0.80]
            c["sizing"] = dict(power=pw, groups_per_arm=ok[0] if ok else (120 if pw[120] >= 0.50 else 40),
                               is_test=bool(ok or pw[120] >= 0.50))
        contrasts.append(c)
    res = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), stage=f"{exp} frozen forecasts (amendment C8)", B=B, rho=E.RHO,
               rounds=E.ROUNDS, plan=EXP[exp]["plan"], cells=EXP[exp]["cells"], trace=trace, predictions=preds, contrasts=contrasts)
    files = ["chase_prompts.py", "chase_pop.py", "chase_fit.py", "chase_e2.py", "chase_metrics.py", "chase_e6a.py", "chase_cal.py", "llm.py"]
    res["source_sha256"] = {f: hashlib.sha256((Path(__file__).parent / f).read_bytes()).hexdigest() for f in files}
    res["data_sha256"] = {statecal.name: hashlib.sha256(statecal.read_bytes()).hexdigest()}
    res["sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    out.write_text(json.dumps(res))
    print("written", out, res["sha256"])
    for e in preds:
        for n in e["sizes"]:
            print(e["model"].split(":")[-1], e["framing"], e["arm"], f"N{e['N']} G={n}", {f: {k: (None if v is None else round(v, 2)) for k, v in e[f][str(n)]["headline"].items()} for f in e["forms"]})
    for c in contrasts:
        print(json.dumps({k: v for k, v in c.items() if k in ("model", "framing", "sizing")}),
              {k: {n: (round(x["diff"]["mean"], 2), round(x["power_05"], 2)) for n, x in v.items()} for k, v in c.items() if k.startswith("N")})


def main():
    cmd, exp = (sys.argv + ["", ""])[1:3]
    if cmd not in ("round", "predict", "status") or exp not in EXP:
        sys.exit(__doc__)
    if cmd == "round":
        sys.exit(cmd_round(exp))
    if cmd == "predict":
        cmd_predict(exp, B=int(os.environ.get("CHASE_CAL_B", "200")))
    else:
        tracef = paths(exp)[1]
        print(tracef.read_text() if tracef.exists() else "no rounds yet")


if __name__ == "__main__":
    main()
