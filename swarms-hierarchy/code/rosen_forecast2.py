"""Follow-up 1 (written 2026-11-19, after R2): a richer single-decision kernel and a prospective re-test.

The R2 forecast (rosen_forecast.py, v1) missed for three visible reasons: autarky was extrapolated from team states,
task-type choice was memoryless, and crowding/turn order in round 1 were not measured. v2 adds, from the R1x cells
(rosen_kernel.py extend) only:
  aut_start  P(start | lone agent): ridge logistic per m in {2, 3} on [1, F/16, H/16]; type from aut-cell builds
  start2     P(start | team): m = 2 ridge logistic on core + crowd + done + pos cells, features
             [1, F/16, H/16, kb/7, kr/7, (pos-1)/7 * [round 1]]  (kb others visibly building, kr others visibly
             holding a READY workbench, pos own turn position, used in round 1 only, as measured);
             m = 1 and m = 3 use their own [1, F/16, H/16] fit on core cells plus the m = 2 shifts for kb, kr, pos
  rho        persistence of task-type choice: rho_X = (P(DO X | own record all X) - P(DO X | mixed record)) /
             (1 - P(DO X | mixed)), averaged over X in A, B, C and m in {1, 2}, clipped to [0, 1]; in the policy an
             agent with >= 2 tasks done picks the oldest open task of its most-done type with probability rho
  gamma      type crowding as in v1, with n_j counting others building OR holding type j
Everything else (continuation, second workbench, use, accuracy, invalid, oldest) is v1's.

Frozen file results/rosen_r2x_predictions.json (write-once, sha256):
  v2           prospective forecast for R2x (8 fresh seeds 41101-41108, arms ARMS2)        PRIMARY
  v1           v1's kernel (refitted from R1 rows only, same bootstrap) for the same R2x seeds and arms (baseline)
  v2_retro     v2 on the R2 seeds and arms (RETROSPECTIVE: R1x cells were chosen after seeing R2)
Usage: rosen_forecast2.py freeze --models ... [--B 200]     |     rosen_forecast2.py show --models ...
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rosen
import rosen_forecast as v1
from rosen import TYPES, run_pop, metrics
from rosen_forecast import ridge_logit, sig, resample, arm_key, METRICS

R = rosen.R
PRED = R / "rosen_r2x_predictions.json"
SEEDS2 = list(range(41101, 41109))
ARMS2 = ([("team", 2.0, F) for F in (1, 3, 5, 7, 9, 13)] + [("team", 1.0, 3)] +
         [("autarky", 2.0, F) for F in (1, 3, 7)])
B_RETRO = 100


def load_all(models):
    rows = [__import__("json").loads(l) for l in open(R / "rosen_kernel.jsonl")]
    return [r for r in rows if r["model"] in models and r.get("reasoning", "none") == "none"]


def fit2(rows):
    th = v1.fit([r for r in rows if r["set"] not in ("aut", "hist", "done", "pos")])
    # lone agent
    aut = [r for r in rows if r["set"] == "aut"]
    for m in (2.0, 3.0):
        rr = [r for r in aut if r["m"] == m]
        th[f"aut_m{m:g}"] = (ridge_logit([[1, r["F"] / 16, r["H"] / 16] for r in rr], [r["action"] == "build" for r in rr])
                             .tolist() if rr else th[f"start_m{m:g}"][:3])
    bt = [r.get("type") for r in aut if r["action"] == "build"]
    base = np.array([bt.count(j) + 1.0 for j in TYPES]); th["aut_type"] = (base / base.sum()).tolist()
    # team start with crowding by building/ready others and round-1 position
    def feats(r):
        kb = r.get("k", 0) if r["set"] == "crowd" else 0
        kr = r.get("k", 0) if r["set"] == "done" else 0
        pos = r.get("p", 1)
        if r["set"] == "pos" and r.get("cond") == "build":
            kb = pos - 1
        return [1, r["F"] / 16, r["H"] / 16, kb / 7, kr / 7, (pos - 1) / 7]
    rr = [r for r in rows if r["m"] == 2.0 and r["set"] in ("core", "crowd", "done", "pos")]
    b2 = ridge_logit([feats(r) for r in rr], [r["action"] == "build" for r in rr])
    th["start2_m2"] = b2.tolist()
    th["shift"] = dict(kb=float(b2[3]), kr=float(b2[4]), pos=float(b2[5]))
    for m in (1.0, 3.0):
        cc = [r for r in rows if r["set"] == "core" and r["m"] == m]
        th[f"start2_m{m:g}"] = ridge_logit([[1, r["F"] / 16, r["H"] / 16] for r in cc], [r["action"] == "build" for r in cc]).tolist()
    # type crowding counting building and ready others
    core = [r for r in rows if r["set"] == "core"]
    cb = [r for r in rows if r["set"] in ("crowd", "done") and r["action"] == "build" and r.get("type")]
    tb = np.array(th["type_base"])
    best, bg = None, 0.0
    for g in np.arange(-5, 10.001, 0.02):
        ll = 0.0
        for r in cb:
            nj = np.array([r["k"] * (j == r.get("ctype", "A")) / rosen.ARRIVE[j] for j in TYPES])
            w = tb * np.exp(-g * nj); ll += np.log(w[TYPES.index(r["type"])] / w.sum())
        if best is None or ll > best:
            best, bg = ll, float(g)
    th["gamma2"] = bg if cb else th["gamma"]
    # persistence
    hist = [r for r in rows if r["set"] == "hist" and r["action"] == "do" and r.get("type")]
    rhos = []
    for m in (1.0, 2.0):
        mix = [r for r in hist if r["m"] == m and r["X"] == "mix"]
        for X in TYPES:
            hx = [r for r in hist if r["m"] == m and r["X"] == X]
            if mix and hx:
                p0 = np.mean([r["type"] == X for r in mix]); p1 = np.mean([r["type"] == X for r in hx])
                rhos.append((p1 - p0) / (1 - p0) if p0 < 1 else 0.0)
    th["rho"] = float(np.clip(np.mean(rhos), 0, 1)) if rhos else 0.0
    return th


def make_policy2(th, rng):
    base = v1.make_policy(th, rng)          # used for continuation / ready / second-workbench logic

    def ans(x):
        return x["answer"] if rng.random() < th["accuracy"] else "0"

    def do_general(view):
        op = view["open"]
        if not op:
            return "nothing to do"
        done = view["stats"][view["me"]]["done"]
        if sum(done.values()) >= 2 and rng.random() < th["rho"]:
            fav = max(TYPES, key=lambda j: (done[j], -TYPES.index(j)))
            own = [x for x in op if x["type"] == fav]
            if own:
                return f"DO T{own[0]['id']}: {ans(own[0])}"
        x = op[0] if rng.random() < th["oldest"] else op[rng.integers(len(op))]
        return f"DO T{x['id']}: {ans(x)}"

    def pol(view):
        F, m, me = view["F"], view["m"], view["me"]
        H = view["T"] - view["t"] + 1
        st = view["stats"][me]
        ready = [j for j in TYPES if st["bench"][j] >= F]
        prog = [j for j in TYPES if 0 < st["bench"][j] < F]
        if ready or prog:
            out = base(view)                 # v1 logic (incl. its invalid draw) for agents that already invest
            if out.startswith("DO") and not ready:
                return do_general(view)
            return out
        if rng.random() < th["invalid"]:
            return "?"
        if view["arm"] == "autarky":
            b = th[f"aut_m{m:g}"] if f"aut_m{m:g}" in th else th[f"start2_m{m:g}"]
            if rng.random() < sig(b[0] + b[1] * F / 16 + b[2] * H / 16):
                w = np.array(th["aut_type"])
                return f"BUILD {TYPES[rng.choice(3, p=w / w.sum())]}"
            return do_general(view)
        vis = view["arm"] == "team"
        kb = sum(1 for a, s in view["stats"].items() if a != me and any(0 < s["bench"][j] < F for j in TYPES)) if vis else 0
        kr = sum(1 for a, s in view["stats"].items() if a != me and any(s["bench"][j] >= F for j in TYPES)) if vis else 0
        pos = view["pos"] if view["t"] == 1 else 1
        sh = th["shift"]
        b = th[f"start2_m{m:g}"]
        z = b[0] + b[1] * F / 16 + b[2] * H / 16 + sh["kb"] * kb / 7 + sh["kr"] * kr / 7 + sh["pos"] * (pos - 1) / 7
        if rng.random() < sig(z):
            n = {j: sum(1 for a, s in view["stats"].items() if a != me and s["bench"][j] > 0) if vis else 0 for j in TYPES}
            w = np.array(th["type_base"]) * np.exp(-th["gamma2"] * np.array([n[j] / rosen.ARRIVE[j] for j in TYPES]))
            return f"BUILD {TYPES[rng.choice(3, p=w / w.sum())]}"
        return do_general(view)
    return pol


def simulate(th, arm, m, F, draw, seeds, policy):
    out = []
    for s in seeds:
        rng = np.random.default_rng([draw, s, int(F), int(m * 10), {"team": 0, "hidden": 1, "autarky": 2}[arm]])
        out.append(metrics(run_pop("kernel", arm, F, m, s, decide=policy(th, rng))))
    return {k: float(np.nanmean([o[k] for o in out])) if not all(np.isnan(o[k]) for o in out) else float("nan")
            for k in METRICS}


def forecast(rows, B, fitter, policy, arms, seeds, seed0=12345):
    th0 = fitter(rows)
    rng = np.random.default_rng(seed0)
    draws = []
    for b in range(B):
        th = th0 if b == 0 else fitter(resample(rows, rng))
        draws.append({arm_key(*a): simulate(th, *a, draw=b, seeds=seeds, policy=policy) for a in arms})
    res = {}
    for a in arms:
        k = arm_key(*a); res[k] = {}
        for mt in METRICS:
            v = np.array([d[k][mt] for d in draws], float); v = v[~np.isnan(v)]
            res[k][mt] = (dict(point=float(np.median(v)), lo=float(np.percentile(v, 5)), hi=float(np.percentile(v, 95)))
                          if len(v) else None)
    return th0, res


def rational(arms, seeds):
    out = {}
    for a in arms:
        ms = [metrics(run_pop("rational", a[0], a[2], a[1], s, decide=rosen.policy_rational)) for s in seeds]
        out[arm_key(*a)] = {k: float(np.nanmean([x[k] for x in ms])) if not all(np.isnan(x[k]) for x in ms) else None
                            for k in METRICS}
    return out


def cmd_freeze(a):
    if PRED.exists():
        sys.exit("frozen; refusing to overwrite")
    out = dict(stage="R2x", models=a.models, seeds=SEEDS2, arms=[arm_key(*x) for x in ARMS2], metrics=list(METRICS),
               B=a.B, B_retro=B_RETRO, rational=rational(ARMS2, SEEDS2), kernels={}, v2={}, v1={}, v2_retro={},
               note="v2 is primary; v1 is the R2 kernel re-simulated for the R2x seeds; v2_retro is retrospective.")
    for mod in a.models:
        rows = load_all([mod])
        r1 = [r for r in rows if r["set"] not in ("aut", "hist", "done", "pos")]
        th, out["v2"][mod] = forecast(rows, a.B, fit2, make_policy2, ARMS2, SEEDS2)
        _, out["v1"][mod] = forecast(r1, a.B, v1.fit, lambda t, g: v1.make_policy(t, g), ARMS2, SEEDS2)
        _, out["v2_retro"][mod] = forecast(rows, B_RETRO, fit2, make_policy2, v1.ARMS, v1.SEEDS)
        out["kernels"][mod] = th
        out.setdefault("n_rows", {})[mod] = dict(all=len(rows), r1=len(r1))
        print(mod, "fitted;", len(rows), "rows", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in th.items()
                                                  if k in ("rho", "gamma2", "shift")}, flush=True)
    out["sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    PRED.write_text(json.dumps(out, indent=1))
    print("frozen", out["sha256"])


def cmd_show(a):
    for mod in a.models:
        rows = load_all([mod])
        th = fit2(rows)
        print(mod, len(rows), "rows", {k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in th.items()
                                       if k in ("aut_m2", "aut_m3", "start2_m2", "shift", "gamma2", "rho", "aut_type")})
        for x in ARMS2:
            r = simulate(th, *x, draw=0, seeds=SEEDS2, policy=make_policy2)
            print("  ", arm_key(*x), {k: round(v, 2) for k, v in r.items() if k in ("dol", "build_rate", "benches", "points")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "show"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=200)
    a = ap.parse_args()
    {"freeze": cmd_freeze, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
