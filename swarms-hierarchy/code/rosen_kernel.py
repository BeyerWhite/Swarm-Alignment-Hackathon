"""Single-decision batteries for the workshop game (R0 comprehension and reasoning probe, R1 investment kernel).

Every item is one agent's turn in a synthetic but consistent game state, rendered with the same prompt code as the
populations (rosen.system_prompt / rosen.render). Each call gets a fresh random state (agent ids, record order, task
contents, kind rotation), so the cell estimate averages over incidental detail.

Cells (R1; 24 calls each unless topped up)
  core      F in {1,3,5,7,9,11,13} x m in {1,2,3} x rounds left H in {6,11,16}; nobody holds or builds a workbench
  crowd     round 2 (H=15), m=2, F in {3,7,11}; k in {2,5} other agents show 'building workbench X (1 of F turns)' for
            X in {A, B}, plus k = 0
  second    own workbench B ready since last round, m=2, F in {1,3,5}; others none
  cont      own 'building workbench A (k of F turns)' after building every turn since round r0, F in {3,7,11},
            k in {1,F-1}: m=2 with r0 in {1,5}, and m=1 with r0 = 1
  use       own workbench B ready, F=3, round 5, m in {2,3}, B tasks open in {1,3} (plus 4 A, 2 C)
  reason    core cells F in {3,7,11} x m in {2,3} x H in {6,16} with reasoning effort 'low' (2 models only)

Usage
  rosen_kernel.py comp    --models ...            R0 comprehension, 12 items x 20 calls  -> results/rosen_comp.jsonl
  rosen_kernel.py probe   --models ...            R0 reasoning probe, 25 calls/model     -> results/rosen_probe.jsonl
  rosen_kernel.py kernel  --models ... [--sets core crowd second cont use]               -> results/rosen_kernel.jsonl
  rosen_kernel.py reason  --models ...                                                   -> results/rosen_kernel.jsonl
  rosen_kernel.py topup   --models ...            one round: core cells with 0.15 <= p <= 0.85 get 16 more calls
  rosen_kernel.py summary                         cell table
"""
import argparse, hashlib, json, random, re, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rosen
from rosen import TYPES, ARRIVE, T, N, kind_map, make_task, parse, render, system_prompt

R = rosen.R
KFILE, CFILE, PFILE = R / "rosen_kernel.jsonl", R / "rosen_comp.jsonl", R / "rosen_probe.jsonl"
REPS, TOPUP_REPS = 24, 16
FS, MS, HS = (1, 3, 5, 7, 9, 11, 13), (1.0, 2.0, 3.0), (6, 11, 16)


def seed_of(*parts):
    """Process-independent seed (Python's hash() of strings changes between runs)."""
    return int(hashlib.sha1(repr(parts).encode()).hexdigest()[:8], 16)


def cells(sets):
    out = []
    if "core" in sets:
        out += [dict(set="core", F=F, m=m, H=H) for F in FS for m in MS for H in HS]
    if "crowd" in sets:
        out += [dict(set="crowd", F=F, m=2.0, H=15, k=0, ctype="A") for F in (3, 7, 11)]
        out += [dict(set="crowd", F=F, m=2.0, H=15, k=k, ctype=x) for F in (3, 7, 11) for x in ("A", "B") for k in (2, 5)]
    if "second" in sets:
        out += [dict(set="second", F=F, m=2.0, H=T - F - 1) for F in (1, 3, 5)]
    if "cont" in sets:
        out += [dict(set="cont", F=F, m=m, H=T - r0 - k + 1, k=k, r0=r0) for F in (3, 7, 11) for k in sorted({1, F - 1})
                for (m, r0) in ((2.0, 1), (2.0, 5), (1.0, 1))]
    if "use" in sets:
        out += [dict(set="use", F=3, m=m, H=T - 4, nb=nb) for m in (2.0, 3.0) for nb in (1, 3)]
    if "reason" in sets:
        out += [dict(set="reason", F=F, m=m, H=H) for F in (3, 7, 11) for m in (2.0, 3.0) for H in (6, 16)]
    return out


def cell_key(c):
    return "|".join(f"{k}={c[k]}" for k in ("set", "F", "m", "H", "k", "ctype", "r0", "nb") if k in c)


def _record(rng, t, n_turns_free):
    """Task counts by type for an agent that did a task on each of `n_turns_free` turns (demand-weighted)."""
    d = {j: 0 for j in TYPES}
    for _ in range(n_turns_free):
        d[rng.choices(TYPES, weights=[ARRIVE[j] for j in TYPES])[0]] += 1
    return d


def synth(c, rng):
    """A consistent team state in which agent `me` moves first in round t = T - H + 1."""
    F, m, H = c["F"], c["m"], c["H"]
    t = T - H + 1
    seed = rng.randrange(10 ** 6)
    kmap = kind_map(seed)
    ids = rng.sample(range(10, 100), N)
    me = ids[0]
    stats = {a: dict(done=_record(rng, t, t - 1), bench={j: 0 for j in TYPES}, points=0.0) for a in ids}
    others = ids[1:]
    if c["set"] == "crowd":
        for a in others[:c["k"]]:
            stats[a]["bench"][c["ctype"]] = 1; stats[a]["done"] = _record(rng, t, t - 2)
    if c["set"] in ("second", "use"):
        stats[me]["bench"]["B"] = F
        d = {j: 0 for j in TYPES}; d["B"] = t - 1 - F; stats[me]["done"] = d
    if c["set"] == "cont":
        stats[me]["bench"]["A"] = c["k"]; stats[me]["done"] = _record(rng, t, c["r0"] - 1)
    # open tasks at the start of round t: this round's arrivals plus a little backlog of the big type
    trng = random.Random(seed)
    nxt = 1 + 8 * (t - 1)
    want = dict(ARRIVE)
    if c["set"] == "use":
        want["B"] = c["nb"]
    opened = []
    for j in TYPES:
        for _ in range(want[j]):
            opened.append(make_task(nxt, j, kmap[j], trng)); nxt += 1
    opened.sort(key=lambda x: x["id"])
    pts = sum(sum(s["done"].values()) for a, s in stats.items())
    if c["set"] in ("second", "use"):
        pts += (m - 1) * stats[me]["done"]["B"]
    rows = ids[:]; rng.shuffle(rows)
    view = dict(arm="team", me=me, F=F, m=m, t=t, T=T, n=N, pos=1, points=int(pts) if float(pts).is_integer() else pts,
                stats=stats, order_rows=rows, earlier=[], open=opened)
    return system_prompt("team", me, F, m), render(view), view


def outcome(c, text, view):
    act = parse(text)
    o = dict(action=act[0])
    if act[0] == "build":
        o["type"] = act[1]
    if act[0] == "do":
        task = next((x for x in view["open"] if x["id"] == act[1]), None)
        o["type"] = task["type"] if task else None
        o["oldest"] = bool(task) and task["id"] == view["open"][0]["id"]
        o["correct"] = bool(task) and rosen.correct(task, act[2])
    return o


_lock = threading.Lock()


def _append(path, rec):
    with _lock, open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_cells(models, cl, reps, base_seed, reasoning_sets=("reason",)):
    from llm import call
    jobs = []
    for mod in models:
        for c in cl:
            for r in range(reps):
                jobs.append((mod, c, r))
    print(f"{len(jobs)} calls", flush=True)
    def go(j):
        mod, c, r = j
        rng = random.Random(seed_of(base_seed, mod, cell_key(c), r))
        sysp, prompt, view = synth(c, rng)
        reasoning = "low" if c["set"] in reasoning_sets else "none"
        text = call(mod, sysp, prompt, tag=f"rosen_kernel:{c['set']}", reasoning=reasoning,
                    max_tokens=rosen.MAX_TOKENS if reasoning == "none" else 2000)
        _append(KFILE, dict(model=mod, cell=cell_key(c), **c, rep=r, base_seed=base_seed, reasoning=reasoning,
                            response=text[:300], **outcome(c, text, view)))
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(go, jobs))


def load_kernel():
    return [json.loads(l) for l in open(KFILE)] if KFILE.exists() else []


def cmd_kernel(a):
    run_cells(a.models, cells(a.sets), REPS, base_seed=1)


def cmd_reason(a):
    run_cells(a.models, cells(["reason"]), REPS, base_seed=2)


def cmd_topup(a):
    rows = [r for r in load_kernel() if r["set"] == "core"]
    for mod in a.models:
        by = defaultdict(list)
        for r in rows:
            if r["model"] == mod:
                by[r["cell"]].append(r["action"] == "build")
        sel = [c for c in cells(["core"]) if by.get(cell_key(c)) and 0.15 <= np.mean(by[cell_key(c)]) <= 0.85]
        print(mod, f"top-up {len(sel)} cells", flush=True)
        if sel:
            run_cells([mod], sel, TOPUP_REPS, base_seed=3)


# ------------------------------------------------------------------------------------- R0: comprehension

COMP = [  # (F, m, question, answer)
    (5, 2, "If you spend your first 5 turns building a type-A workbench and then do one type-A task correctly on each "
           "of your remaining turns, how many points do you earn over the 16 rounds?", 22),
    (5, 2, "If you never build and do one task correctly on each of your 16 turns, how many points do you earn?", 16),
    (6, 3, "If you spend your first 6 turns building a type-A workbench and then do one type-A task correctly on each "
           "of your remaining turns, how many points do you earn over the 16 rounds?", 30),
    (12, 2, "If you spend your first 12 turns building a type-A workbench and then do one type-A task correctly on "
            "each of your remaining turns, how many points do you earn over the 16 rounds?", 8),
    (4, 2, "How many turns of building does a workbench need before it is ready?", 4),
    (4, 3, "Once your type-B workbench is ready, how many points does a correct type-B task earn?", 3),
    (4, 3, "If you have no type-A workbench, how many points does a correct type-A task earn?", 1),
    (4, 2, "How many new type-A tasks arrive each round?", 4),
    (4, 2, "How many new type-C tasks arrive each round?", 2),
    (3, 2, "Suppose your record said 'building workbench C (2 of 3 turns)'. How many more turns of building would "
           "the workbench need?", 1),
    (3, 2, "Which earns you more points over the 16 rounds: (1) building a type-A workbench on your first 3 turns and "
           "then doing type-A tasks, or (2) never building and doing a task every turn? Answer 1 or 2.", 1),
    (12, 2, "Which earns you more points over the 16 rounds: (1) building a type-A workbench on your first 12 turns "
            "and then doing type-A tasks, or (2) never building and doing a task every turn? Answer 1 or 2.", 2),
]
COMP_REPS = 20


def cmd_comp(a):
    from llm import call
    jobs = [(mod, i, r) for mod in a.models for i in range(len(COMP)) for r in range(COMP_REPS)]
    print(f"{len(jobs)} calls", flush=True)
    def go(j):
        mod, i, r = j
        F, m, q, ans = COMP[i]
        rng = random.Random(seed_of("comp", mod, i, r))
        sysp, prompt, _ = synth(dict(set="core", F=F, m=float(m), H=16), rng)
        prompt = prompt.replace("\n\nYour action?", f"\n\nThis is not your turn yet. Answer a question about the rules "
                                                    f"with a number only.\nQuestion: {q}")
        text = call(mod, sysp, prompt, tag="rosen_comp", max_tokens=24)
        mnum = re.search(r"-?\d+", text or "")
        _append(CFILE, dict(model=mod, item=i, rep=r, answer=ans, response=(text or "")[:120],
                            correct=bool(mnum) and int(mnum.group(0)) == ans))
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(go, jobs))
    cmd_comp_summary(a)


def cmd_comp_summary(a=None):
    rows = [json.loads(l) for l in open(CFILE)]
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r["correct"])
    for mod, v in by.items():
        items = [np.mean([r["correct"] for r in rows if r["model"] == mod and r["item"] == i]) for i in range(len(COMP))]
        print(f"{mod:32s} comprehension {np.mean(v):.2f}  by item {[round(float(x), 2) for x in items]}")


def cmd_probe(a):
    """Reasoning probe: does 'reasoning: low' work on this model, what does it change, what does it cost."""
    from llm import call
    c = dict(set="core", F=5, m=2.0, H=16)
    jobs = [(mod, r) for mod in a.models for r in range(25)]
    def go(j):
        mod, r = j
        rng = random.Random(seed_of("probe", mod, r))
        sysp, prompt, view = synth(c, rng)
        text = call(mod, sysp, prompt, tag="rosen_probe", reasoning="low", max_tokens=2000)
        _append(PFILE, dict(model=mod, rep=r, response=(text or "")[:400], **outcome(c, text, view)))
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(go, jobs))


def cmd_summary(a):
    rows = load_kernel()
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["cell"])].append(r)
    for (mod, cell), v in sorted(by.items()):
        n = len(v)
        pb = np.mean([x["action"] == "build" for x in v]); pi = np.mean([x["action"] == "invalid" for x in v])
        print(f"{mod.split(':')[1]:22s} {cell:34s} n={n:3d} build={pb:.2f} invalid={pi:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["comp", "probe", "kernel", "reason", "topup", "summary", "comp_summary"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--sets", nargs="+", default=["core", "crowd", "second", "cont", "use"])
    a = ap.parse_args()
    dict(comp=cmd_comp, probe=cmd_probe, kernel=cmd_kernel, reason=cmd_reason, topup=cmd_topup, summary=cmd_summary,
         comp_summary=cmd_comp_summary)[a.cmd](a)


if __name__ == "__main__":
    main()
