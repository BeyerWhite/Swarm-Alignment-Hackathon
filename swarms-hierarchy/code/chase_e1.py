"""E1: the bystander kernel, measured by intervention in single-agent calls.

Each call shows one agent a constructed contest record and asks LEAD / DEFER. Cells vary
  d  direct history with the partner,
  o  the agent's own record against third parties (winner/loser information),
  p  the partner's record against third parties (bystander information),
  whether own and partner records share a third party (transitive chain) or not (matched control),
in each arm's display format (DYAD / OWN / LOG / REDACT). 50 cells; see refine-logs/CHASE_PREREGISTRATION.md.

Usage:
  chase_e1.py --dry 3                       # print sample prompts, no calls
  chase_e1.py --models adaption:gemma-4-31b-it ... --workers 24
Rows are appended to results/chase_e1_kernel.jsonl; finished job ids are skipped on restart.
"""
import argparse, json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from chase_prompts import SYSTEM, TASKS, FRAMINGS, WINDOW, log_prompt, features

ROOT = Path(__file__).resolve().parents[1]
N_POP = 8
MODELS = ["adaption:gemma-4-31b-it", "adaption:glm-5.2", "adaption:deepseek-v4-flash",
          "adaption:gemma-4-26b-a4b-it", "adaption:qwen3.8-27b", "adaption:qwen3.5-397b"]
REPS, REPS_REL = 32, 96


def _thirds(n, sign, keys):
    return [(sign, k) for k in keys[:n]]


def cells():
    """name -> dict(arm, d, o=[(sign, key)], p=[(sign, key)], block, reps). Keys label third parties."""
    C = {}
    sg = lambda v: 1 if v > 0 else -1
    for o in (-2, 0, 2):
        for p in (-3, -1, 0, 1, 3):
            C[f"L_o{o:+d}_p{p:+d}"] = dict(arm="LOG", d=0, o=_thirds(abs(o), sg(o), "ab"), p=_thirds(abs(p), sg(p), "cde"),
                                           block="grid", reps=REPS)
    for nm, so, sp in (("against", -1, 1), ("for", 1, -1), ("cwin", 1, 1), ("closs", -1, -1)):
        C[f"R_{nm}_shared"] = dict(arm="LOG", d=0, o=[(so, "a")], p=[(sp, "a")], block="rel", reps=REPS_REL)
        C[f"R_{nm}_distinct"] = dict(arm="LOG", d=0, o=[(so, "a")], p=[(sp, "c")], block="rel", reps=REPS_REL)
    for d in (-1, 1):
        for p in (-2, 0, 2):
            C[f"LD_d{d:+d}_p{p:+d}"] = dict(arm="LOG", d=d, o=[], p=_thirds(abs(p), sg(p), "cde"), block="direct", reps=REPS)
    C["LC_dwin_against"] = dict(arm="LOG", d=1, o=[(-1, "a")], p=[(1, "a")], block="conflict", reps=REPS)
    C["LC_dloss_for"] = dict(arm="LOG", d=-1, o=[(1, "a")], p=[(-1, "a")], block="conflict", reps=REPS)
    for o in (-2, 0, 2):
        C[f"X_o{o:+d}"] = dict(arm="REDACT", d=0, o=_thirds(abs(o), sg(o), "ab"), p=_thirds(3, 1, "cde"),
                               block="redact", reps=REPS)
    for o in (-2, -1, 0, 1, 2):
        C[f"O_o{o:+d}"] = dict(arm="OWN", d=0, o=_thirds(abs(o), sg(o), "ab"), p=[], block="own", reps=REPS)
    for d in (-1, 1):
        for o in (-2, 0, 2):
            C[f"OD_d{d:+d}_o{o:+d}"] = dict(arm="OWN", d=d, o=_thirds(abs(o), sg(o), "ab"), p=[], block="own_direct", reps=REPS)
    for d in (-2, -1, 0, 1, 2):
        C[f"D_d{d:+d}"] = dict(arm="DYAD", d=d, o=[], p=[], block="dyad", reps=REPS)
    return C


def build(cell, framing, rng):
    """Construct one vignette: random IDs, task, current round, and event rounds inside the window."""
    ids = rng.sample(range(1, 100), 8)
    me, other = ids[0], ids[1]
    third = dict(zip("abcde", ids[2:7]))
    t_now = rng.randint(WINDOW + 1, 30)
    n_ev = abs(cell["d"]) + len(cell["o"]) + len(cell["p"])
    rounds = rng.sample(range(t_now - WINDOW, t_now), n_ev)
    own, partner = [], []
    for _ in range(abs(cell["d"])):
        own.append((rounds.pop(), me, other) if cell["d"] > 0 else (rounds.pop(), other, me))
    for s, k in cell["o"]:
        own.append((rounds.pop(), me, third[k]) if s > 0 else (rounds.pop(), third[k], me))
    for s, k in cell["p"]:
        partner.append((rounds.pop(), other, third[k]) if s > 0 else (rounds.pop(), third[k], other))
    task = rng.choice(TASKS)
    prompt = log_prompt(cell["arm"], framing, me, other, t_now, own, partner, task)
    return SYSTEM.format(me=me, n=N_POP), prompt, features(me, other, own, partner, cell["arm"])


def jobs_for(models, framings, seed, scale=1.0):
    out = []
    for m in models:
        for fr in framings:
            for cn, c in cells().items():
                for r in range(max(1, int(round(c["reps"] * scale)))):
                    jid = f"{m}|{fr}|{cn}|{r}"
                    system, prompt, feat = build(c, fr, random.Random(f"{seed}|{jid}"))
                    out.append(dict(jid=jid, model=m, framing=fr, cell=cn, arm=c["arm"], block=c["block"], rep=r,
                                    system=system, prompt=prompt, **feat))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--framings", nargs="+", default=list(FRAMINGS))
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply reps (smoke tests)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--dry", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "chase_e1_kernel.jsonl"))
    a = ap.parse_args()
    jobs = jobs_for(a.models, a.framings, a.seed, a.scale)
    if a.dry:
        rng = random.Random(0)
        print(len(jobs), "jobs;", len(cells()), "cells;", sum(len(j["system"]) + len(j["prompt"]) for j in jobs) / len(jobs), "chars/call")
        for j in rng.sample(jobs, a.dry):
            print("=" * 100, "\n", j["jid"], {k: j[k] for k in ("d", "o", "p", "tc", "n_own", "n_par")})
            print("--- SYSTEM\n" + j["system"] + "\n--- PROMPT\n" + j["prompt"])
        return
    from llm import call, parse_choice
    done = set()
    if Path(a.out).exists():
        for l in open(a.out):
            r = json.loads(l)
            if r.get("choice"):
                done.add(r["jid"])
    todo = [j for j in jobs if j["jid"] not in done]
    random.Random(a.seed).shuffle(todo)
    print(f"{len(jobs)} jobs, {len(done)} done, {len(todo)} to run", flush=True)
    lock = threading.Lock()
    f = open(a.out, "a")

    def one(j):
        resp, ch, tries = "", None, 0
        for tries in range(1, 4):
            resp = call(j["model"], j["system"], j["prompt"], tag="chase_e1")
            ch = parse_choice(resp)
            if ch:
                break
        return j, resp, ch, tries

    n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for fu in as_completed([ex.submit(one, j) for j in todo]):
            j, resp, ch, tries = fu.result()
            row = {k: v for k, v in j.items() if k not in ("system", "prompt")}
            with lock:
                f.write(json.dumps({**row, "response": resp, "choice": ch, "tries": tries}) + "\n"); f.flush()
                n += 1
                if n % 2000 == 0:
                    print(n, "done", flush=True)
    f.close()
    print("finished", n, flush=True)


if __name__ == "__main__":
    main()
