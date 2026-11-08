"""E0 / E1a analyses on existing calibration logs (no new LLM calls).

E0  presentation biases: does an agent claim LEAD more when its own score row is listed first, or depending on its
    name (agent ID)? Logistic regression of LEAD on score difference + own-row-first + ID features, per model,
    pooled over framings (framing fixed effects). Uses the logged prompt text, so it is exact.
E1a contest-rule shape: for each model x framing, compare by AIC three shapes for p(LEAD | own h, partner h):
      bonabeau   : sigma(a + b*Delta)                               (logistic in the difference)
      step       : p_lo below 0, p_0 at 0, p_hi above 0             (BNR: higher score decides, plus noise)
      separate   : sigma(a + b1*h_own + b2*h_partner)               (scores enter separately)
Writes results/e0_bias.csv and results/e1a_shapes.csv.
"""
import json, re, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
sys.path.insert(0, str(Path(__file__).parent))
from llm import parse_choice

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def load_with_prompts():
    rows = []
    for f in glob.glob(str(R / "calib_v2*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("exp") != "kernel":
                continue
            rows.append(r)
    df = pd.DataFrame(rows)
    df["choice"] = df.response.map(parse_choice)
    df = df[df.choice.notna()].copy()
    df["lead"] = (df.choice == "LEAD").astype(int)
    disp = lambda x: float(f"{x:+.1f}")
    df["h_own"] = df.h_me.map(disp); df["h_oth"] = df.h_other.map(disp)
    df["delta"] = df.h_own - df.h_oth
    return df


def attach_prompt_features(df):
    """Recover prompt text from the call log by (model, displayed own score) is ambiguous; instead re-identify via
    the call log hash of system+prompt is not stored in calib rows. We therefore read the call log and match on
    (model, response) is also ambiguous. Use the call log directly: every calib call's prompt contains the rows."""
    feats = []
    with open(R / "calls.jsonl") as f:
        for l in f:
            r = json.loads(l)
            if r.get("tag") != "calib" or not r.get("response"):
                continue
            c = parse_choice(r["response"])
            if c is None:
                continue
            m_me = re.search(r"Agent (\d+) \(you\): ([+-][\d.]+)", r["prompt"])
            lines = [x for x in r["prompt"].splitlines() if x.startswith("- Agent ")]
            if not m_me or len(lines) != 2:
                continue
            m_ot = re.search(r"- Agent (\d+): ([+-][\d.]+)", r["prompt"])
            own_first = "(you)" in lines[0]
            norm = re.search(r"current setting (\d+)", r["prompt"])
            verb = "won" if "won the coordinator role" in r["prompt"] else "ended"
            comp = "coordinate well" in r["prompt"]
            cond = f"norm{norm.group(1)}" if norm else (verb + ("+comp" if comp else ""))
            feats.append(dict(model=r["model"], cond=cond, lead=int(c == "LEAD"), own_first=int(own_first),
                              my_id=int(m_me.group(1)), other_id=int(m_ot.group(1)) if m_ot else -1,
                              delta=float(m_me.group(2)) - float(m_ot.group(2)) if m_ot else np.nan))
    return pd.DataFrame(feats).dropna()


def e0(feat):
    out = []
    for m, g in feat.groupby("model"):
        g = g.copy(); g["id_hi"] = (g.my_id > g.other_id).astype(int)
        try:
            fit = smf.logit("lead ~ delta + own_first + id_hi + C(cond)", g).fit(disp=0)
            out.append(dict(model=m, n=len(g),
                            own_first_coef=fit.params["own_first"], own_first_p=fit.pvalues["own_first"],
                            id_hi_coef=fit.params["id_hi"], id_hi_p=fit.pvalues["id_hi"],
                            p_lead_first=g[g.own_first == 1].lead.mean(), p_lead_second=g[g.own_first == 0].lead.mean()))
        except Exception as ex:
            out.append(dict(model=m, n=len(g), error=str(ex)[:80]))
    t = pd.DataFrame(out); t.to_csv(R / "e0_bias.csv", index=False); return t


def nll_bernoulli(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def e1a(df):
    out = []
    for (m, c), g in df.groupby(["model", "cond"]):
        y = g.lead.values.astype(float)
        res = {}
        for name, form in (("bonabeau", "lead ~ delta"), ("separate", "lead ~ h_own + h_oth")):
            try:
                fit = smf.logit(form, g).fit(disp=0, method="bfgs", maxiter=200)
                res[name] = (2 * len(fit.params) + 2 * nll_bernoulli(fit.predict(g), y))
            except Exception:
                res[name] = np.nan
        # step: three rates by sign of delta (BNR: higher score decides, plus noise)
        sgn = np.sign(g.delta.values)
        p = np.zeros(len(g))
        for s_ in (-1, 0, 1):
            msk = sgn == s_
            if msk.any():
                p[msk] = (y[msk].sum() + .5) / (msk.sum() + 1)
        res["step"] = 2 * 3 + 2 * nll_bernoulli(p, y)
        best = min((v, k) for k, v in res.items() if np.isfinite(v))[1]
        out.append(dict(model=m, cond=c, n=len(g), **{f"AIC_{k}": round(v, 1) for k, v in res.items()}, best=best))
    t = pd.DataFrame(out); t.to_csv(R / "e1a_shapes.csv", index=False); return t


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    feat = attach_prompt_features(None)
    print("E0: presentation biases (logistic, controlling for score difference and framing)")
    print(e0(feat).round(3).to_string(index=False))
    df = load_with_prompts()
    t = e1a(df)
    print("\nE1a: best shape by AIC per model x framing")
    print(t.groupby("model").best.value_counts().unstack(fill_value=0).to_string())
    print("\noverall:", t.best.value_counts().to_dict())
