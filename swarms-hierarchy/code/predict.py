"""Pre-registered macro predictions from frozen calibration kernels.

For each (model, cond, rho, N) config and each kernel family, bootstrap the calibration rows B times,
refit the kernel, simulate S populations per refit under the exact population protocol, and store the
predictive distribution of every order parameter. Output is written once and SHA256-stamped.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from population2 import KERNELS, KernelDecider, run, order_params, METRICS

ROOT = Path(__file__).resolve().parents[1]


def calib_files():
    return sorted((ROOT / "results").glob("calib_v2*.jsonl"))


def calib_sha():
    h = hashlib.sha256()
    for f in calib_files():
        h.update(f.name.encode()); h.update(open(f, "rb").read())
    return h.hexdigest()


def rows_sha(d):
    cols = ["model", "cond", "delta", "mean", "choice"]
    blob = d[cols].sort_values(cols, na_position="first").to_json(orient="values")
    return hashlib.sha256(blob.encode()).hexdigest()


def load_statecal():
    """State-matched calibration rows (same schema, design='state'), strictly re-parsed."""
    from llm import parse_choice
    fs = sorted((ROOT / "results").glob("statecal_*.jsonl"))
    if not fs:
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(l) for f in fs for l in open(f)])
    df["choice"] = df.response.map(parse_choice)
    df = df[df.choice.notna()].copy()
    df["lead"] = (df.choice == "LEAD").astype(float)
    return df


def load_calib():
    df = pd.DataFrame([json.loads(l) for f in calib_files() for l in open(f)])
    df = df[df.exp == "kernel"].copy()
    from llm import parse_choice
    df["choice"] = df.response.map(parse_choice)        # strict re-parse of raw responses (amendment 1)
    df["valid"] = df.choice.notna()
    df = df[df.valid]
    df["lead"] = (df.choice == "LEAD").astype(float)
    # agents saw scores formatted to one decimal: fit on the *displayed* values
    disp = lambda x: float(f"{x:+.1f}")
    df["h_me_d"] = df.h_me.map(disp); df["h_other_d"] = df.h_other.map(disp)
    df["delta_nominal"] = df.delta
    df["delta"] = (df.h_me_d - df.h_other_d).round(2)
    df["mean"] = ((df.h_me_d + df.h_other_d) / 2).round(2)
    return df


def predictive(df, kernel, N, rho, mu, rounds, B, S, window, seed=0):
    rng = np.random.default_rng(seed)
    samples = {m: [] for m in METRICS}
    params = []
    for b in range(B):
        k = KERNELS[kernel](df, rng)                       # every replicate is a posterior/bootstrap draw
        if hasattr(k, "params"):
            params.append(k.params)
        for s in range(S):
            o = order_params(run(N, rho, mu, rounds, KernelDecider(k), seed=int(rng.integers(1e9))), window)
            for m in METRICS:
                samples[m].append(o[m])
    summ = {m: dict(mean=float(np.mean(v)), sd=float(np.std(v)),
                    q05=float(np.quantile(v, .05)), q95=float(np.quantile(v, .95)), samples=[round(x, 4) for x in v])
            for m, v in samples.items()}
    if params:
        P = np.array(params)
        summ["logistic_params"] = dict(mean=P.mean(0).tolist(), sd=P.std(0).tolist())
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="JSON list of {model, cond, rho, N}")
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--B", type=int, default=40)
    ap.add_argument("--S", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results" / "predictions_v2.json"))
    ap.add_argument("--kernels", nargs="+", default=["flexible", "bonabeau", "logistic3", "null"])
    ap.add_argument("--with_state", action="store_true", help="fit on grid + state-matched calibration rows")
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"refusing to overwrite frozen predictions {a.out}")
    df_all = load_calib()
    if a.with_state:
        st = load_statecal()
        df_all = pd.concat([df_all, st[[c for c in st.columns if c in df_all.columns or c == "design"]]],
                           ignore_index=True)
    configs = json.loads(a.configs)
    res = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), mu=a.mu, rounds=a.rounds, window=a.window,
               B=a.B, S=a.S, calib_rows=len(df_all), predictions=[])
    for c in configs:
        d = df_all[(df_all.model == c["model"]) & (df_all.cond == c["cond"])]
        entry = dict(**c, n_calib=len(d), calib_rows_sha256=rows_sha(d))
        entry["kernels"] = a.kernels
        entry["n_state_rows"] = int((d.get("design") == "state").sum()) if "design" in d else 0
        for kern in a.kernels:
            entry[kern] = predictive(d, kern, c["N"], c["rho"], a.mu, a.rounds, a.B, a.S, a.window)
        res["predictions"].append(entry)
        print(c, {k: round(entry[k]["sd_h"]["mean"], 2) for k in a.kernels}, flush=True)
    src = {f: hashlib.sha256(open(Path(__file__).parent / f, "rb").read()).hexdigest()
           for f in ("population2.py", "predict.py", "prompts_v2.py", "llm.py")}
    res["source_sha256"] = src
    res["calib_sha256"] = calib_sha()
    blob = json.dumps(res, sort_keys=True)
    res["sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    Path(a.out).write_text(json.dumps(res))
    print("written", a.out, res["sha256"])


if __name__ == "__main__":
    main()
