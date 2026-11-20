"""E12 (amendment C10): why the forecast misses the Qwen3.5-397B `comp` partner-record arm. No calls.

  chase_e12.py [--reps 100] [--test] [--out FILE]      -> results/chase_e12.json

Data: LOG decisions of the Qwen3.5-397B `comp` groups of four in E7 (seeds 24000+, 40 groups) and E10 (44000+, 80),
rebuilt from the stored histories (chase_c10.rows). Reported alongside without a test: the same for Gemma-4-31B
`comp` LOG (E3 4000+ 80 groups, E9 14000+ 40), where the forecast was right. --test swaps in Qwen3.5-397B `norm100`
LOG (E7, 40 groups) and is the only mode run before the hash.

Candidate decision models for LOG (logistic on signed-log features, ridge as chase_fit):
  B0  X of the E7 forecast: d, o, p, tc, hinge, n_own, last, streak, idlow, idme
  B1  B0 + n_par, plast, pstreak       (how much of the partner's record is shown, and its recency)
  B2  B0 + phinge                      (a second slope for deep losing partner records)
  B3  B0 + cmp                         (partner's third-party record above, equal to or below one's own)
  B4  all of these
Criterion: leave-one-seed-out log-loss per decision. Rule (as E6 part A): among forms within 0.002 of the lowest,
the one with the fewest added features; selected only if it improves on B0 by at least 0.005, otherwise B0.
Adequacy: group bootstrap (resample the 120 groups, refit, simulate 120 groups; `reps` times) for B0 and for the
selected form; 90% interval against the observed LOG linear share of the 120 groups. The same for the OWN arm with
X (d, o + the six extras), against the observed OWN share.
Reading, fixed now:
  "calibration coverage"  B0 fitted on the groups' own decisions covers the observed LOG share: the form was
                          adequate and the forecast missed because of the records the calibration rows sampled;
  "missing feature"       B0 does not cover, a selected form other than B0 does;
  "neither"               otherwise.
Descriptive, to locate the miss: K (d, o, p, tc) fitted on the E7 calibration rows for this cell (all rounds) and
on the groups' own decisions, both evaluated at the groups' decisions; mean predicted against observed LEAD rate by
level of p and of o (<= -3, -2..2, >= 3); share of the groups' decisions in (o, p, d-sign) cells holding fewer than
five calibration rows.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import chase_c10 as C
import chase_fit as F

BASE = ["d", "o", "p", "tc", "hinge", "n_own", "last", "streak", "idlow", "idme"]
FORMS = {"B0": BASE, "B1": BASE + ["n_par", "plast", "pstreak"], "B2": BASE + ["phinge"], "B3": BASE + ["cmp"],
         "B4": BASE + ["n_par", "plast", "pstreak", "phinge", "cmp"]}
OWNX = ["d", "o", "hinge", "n_own", "last", "streak", "idlow", "idme"]
SETS = {
    "qwen": [("chase_e7_pop.jsonl", C.QWEN, "comp", 24000, 40), ("chase_e10_pop.jsonl", C.QWEN, "comp", 44000, 80)],
    "gemma31": [("chase_e3_pop.jsonl", C.G31, "comp", 4000, 80), ("chase_e9_pop.jsonl", C.G31, "comp", 14000, 40)],
}
TEST = {"qwen": [("chase_e7_pop.jsonl", C.QWEN, "norm100", 24000, 40)]}
CAL = {"qwen": ("chase_e7_statecal.jsonl", C.QWEN, "comp"), "gemma31": ("chase_e2_statecal.jsonl", C.G31, "comp")}
TEST_CAL = {"qwen": ("chase_e7_statecal.jsonl", C.QWEN, "norm100")}


def tri(x):
    return np.where(x <= -3, 0, np.where(x >= 3, 2, 1))


def select(loss):
    near = [f for f in FORMS if loss[f] <= min(loss.values()) + 0.002]
    cand = min(near, key=lambda f: (len(FORMS[f]) - len(BASE), loss[f]))
    return near, cand, (cand if cand != "B0" and loss["B0"] - loss[cand] >= 0.005 else "B0")


def locate(dpop, cal):
    out = {}
    kp, kc = C.fit(dpop, ["d", "o", "p", "tc"]), C.fit(cal, ["d", "o", "p", "tc"])
    for name, k in (("fit_on_groups", kp), ("fit_on_calibration", kc)):
        z = k[0] + F.slog(dpop[list(k[1])].values.astype(float)) @ np.array(list(k[1].values()))
        dpop = dpop.assign(**{name: 1 / (1 + np.exp(-z))})
    for var in ("p", "o"):
        lv = tri(dpop[var].values)
        out[var] = {str(l): dict(n=int((lv == l).sum()), observed=float(dpop.lead[lv == l].mean()),
                                 fit_on_groups=float(dpop.fit_on_groups[lv == l].mean()),
                                 fit_on_calibration=float(dpop.fit_on_calibration[lv == l].mean()))
                    for l in (0, 1, 2) if (lv == l).any()}
    key = lambda d: list(zip(tri(d.o.values), tri(d.p.values), np.sign(d.d.values)))
    cnt = pd.Series(key(cal)).value_counts().to_dict()
    out["share_in_thin_cells"] = float(np.mean([cnt.get(k, 0) < 5 for k in key(dpop)]))
    out["calibration_rows"] = int(len(cal))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out", default=str(C.R / "chase_e12.json"))
    a = ap.parse_args()
    sets, cals = (TEST, TEST_CAL) if a.test else (SETS, CAL)
    rng, res = np.random.default_rng(20261120), {}
    for name, parts in sets.items():
        recs = {"OWN": [], "LOG": []}
        for fname, model, framing, seed0, G in parts:
            for arm, rs in C.load(fname, model, framing, 4, seed0, {"OWN": G, "LOG": G}).items():
                recs[arm] += rs
        dl, bad_l = C.frame(recs["LOG"]); do, bad_o = C.frame(recs["OWN"])
        x = dict(groups={k: len(v) for k, v in recs.items()}, decisions=dict(LOG=len(dl), OWN=len(do)), mismatch=int(bad_l + bad_o))
        x["observed"] = obs = {arm: float(np.mean([s["linear"] for s in [C.structure(C.events_from_history(r), 4, r["rounds"]) for r in rs]]))
                               for arm, rs in recs.items()}
        x["loso_logloss"] = {f: C.loso(dl, c) for f, c in FORMS.items()}
        x["within_0002"], x["candidate"], x["selected"] = select(x["loso_logloss"])
        G = len(recs["LOG"])
        x["adequacy"] = {"LOG|B0": C.group_boot(dl, FORMS["B0"], "LOG", 4, G, obs["LOG"], a.reps, rng)}
        if x["selected"] != "B0":
            x["adequacy"]["LOG|" + x["selected"]] = C.group_boot(dl, FORMS[x["selected"]], "LOG", 4, G, obs["LOG"], a.reps, rng)
        x["adequacy"]["OWN|X"] = C.group_boot(do, OWNX, "OWN", 4, len(recs["OWN"]), obs["OWN"], a.reps, rng)
        b0, sel = x["adequacy"]["LOG|B0"]["covered"], x["adequacy"].get("LOG|" + x["selected"], {}).get("covered", False)
        x["reading"] = "calibration coverage" if b0 else "missing feature" if (x["selected"] != "B0" and sel) else "neither"
        fname, model, framing = cals[name]
        cal = pd.DataFrame([json.loads(l) for l in open(C.R / fname)])
        cal = cal[(cal.model == model) & (cal.framing == framing) & (cal.arm == "LOG") & (cal.N == 4) & cal.choice.notna()]
        cal = cal.assign(lead=(cal.choice == "LEAD").astype(float), off=0.0)
        x["locate"] = locate(dl, cal)
        res[name] = x
        print(f"== {name}: groups {x['groups']} decisions {x['decisions']} mismatch {x['mismatch']}  observed {obs}")
        print("   LOSO log-loss", {f: round(v, 4) for f, v in x["loso_logloss"].items()}, " within 0.002:", x["within_0002"],
              " selected:", x["selected"])
        for k, v in x["adequacy"].items():
            print(f"   {k}: {v['mean']:.2f} [{v['q05']:.2f}, {v['q95']:.2f}] observed {v['observed']:.2f} covered {v['covered']}")
        print("   reading:", x["reading"])
        print("   locate:", json.dumps(x["locate"]))
    if not a.test:
        res["primary_reading_qwen"] = res["qwen"]["reading"]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("written", a.out)


if __name__ == "__main__":
    main()
