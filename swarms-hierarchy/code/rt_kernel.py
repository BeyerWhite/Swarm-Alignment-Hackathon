"""Response-threshold study: single-decision batteries (RT1 comprehension and effort-cost calibration, RT2 kernel).

Every call is one agent's turn in a fresh synthetic state built with rt.render (the population prompt), so the kernel
is measured on exactly the prompt format the populations use.

  comp     RT1: 10 rule questions x 20 calls per model                         -> results/rt_comp.jsonl
  calib    RT1: c in C_GRID (effort {0.25, 0.5, 0.75}; RT-R rest reward {0.5, 1, 2}) x backlog {1, 1+1, 2+2, 4+4} x position {1, 8} x 20  -> results/rt_calib.jsonl
  choose   RT1: write-once choice of c (rule below)                           -> results/rt_c_choice.json
  gate     RT1: comprehension >= 0.6 and >= 90% valid pilot turns            -> results/rt_gate.json
  grid     RT2a: backlog (sA, sB) in {0,1,2,4,7,11}^2 minus (0,0) x 7 own-history profiles x 8   -> rt_kernel.jsonl
  replay   RT2b: 150 pilot states x (actual, A<->B swapped own history) x 8
  social   RT2c: others' records A-specialists / B-specialists / mixed x 12 backlogs x 12
  topup    RT2: grid cells with 0.15 <= p(work) <= 0.85 get 8 more calls
  summary  print cell tables

c-choice rule (frozen in the pre-registration): over the models that passed comprehension, the pooled share of DO
answers in the calibration cell backlog A 2, B 2, position 1 of 8 (balanced own history); choose the c whose share is
closest to 0.5 (ties: the smaller c). If every c gives a share above 0.9 the study is flagged "saturated" (largest c):
the uptake half of the kernel is then reported as saturated and the type-choice half is primary.
"""
import argparse, json, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rt
from rt import TYPES, HWIN, R, kind_map, make_task, parse, render, system_prompt, reply

COMP, CALIB, KERNEL, GATE = R / "rt_comp.jsonl", R / "rt_calib.jsonl", R / "rt_kernel.jsonl", R / "rt_gate.json"
C_GRID = (0.25, 0.5, 0.75) if rt.FRAME == "effort" else (0.5, 1.0, 2.0)   # RT-R: rest reward r
S_GRID = (0, 1, 2, 4, 7, 11)
PROFILES = {"none": None, "8A": "AAAAAAAA", "5A1B2R": "AAAAABRR", "3A3B2R": "AAABBBRR", "1A5B2R": "ABBBBBRR",
            "8B": "BBBBBBBB", "8R": "RRRRRRRR"}
SOCIAL_S = ((1, 1), (2, 2), (4, 4), (7, 7), (1, 4), (4, 1), (2, 7), (7, 2), (1, 7), (7, 1), (2, 4), (4, 2))
T_SYN = 9                                    # synthetic states: round 9 of 24, so the last 8 turns are the whole past
_lock = threading.Lock()


def seed_of(*parts):
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode())


def _append(path, rec):
    with _lock, open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _done_keys(path):
    if not path.exists():
        return set()
    return {(r["model"], r["set"], r["cell"], r["rep"]) for r in map(json.loads, open(path))}


# ----------------------------------------------------------------------------------------------- synthetic states

def synth(rng, sA, sB, profile, pos, c, others_prof="mixed", t=T_SYN, D=0.5, n=rt.N):
    """A consistent team state in round t: agent `me` moves at position pos, has done what `profile` says in its
    last 8 turns (profile None: K = 0, no own information shown), and sees sA, sB open tasks."""
    seed = rng.randrange(10 ** 6)
    kmap = kind_map(seed)
    ids = rng.sample(range(10, 100), n)
    me, others = ids[0], ids[1:]
    K = 0 if profile is None else HWIN
    h = [] if profile is None else ["rest" if x == "R" else x for x in profile]
    rng.shuffle(h)
    stats = {me: dict(done={j: h.count(j) for j in TYPES})}
    for a in others:
        if others_prof == "Aspec":
            d = {"A": sum(rng.random() < 0.5 for _ in range(t - 1)), "B": 0}
        elif others_prof == "Bspec":
            d = {"A": 0, "B": sum(rng.random() < 0.5 for _ in range(t - 1))}
        else:
            d = {j: sum(rng.random() < 0.25 for _ in range(t - 1)) for j in TYPES}
        stats[a] = dict(done=d)
    trng = random.Random(seed)
    base = 1 + int(D * n) * (t - 1)
    types = ["A"] * sA + ["B"] * sB
    rng.shuffle(types)
    opened = [make_task(base + i, j, kmap[j], trng) for i, j in enumerate(types)]
    earlier = []
    for i, a in enumerate(others[:pos - 1]):
        u = rng.random()
        if u < 0.5:
            earlier.append(f"- Agent {a} rested")
        else:
            j = "A" if u < 0.75 else "B"
            earlier.append(f"- Agent {a} did T{base - 1 - i} ({j})")
    rows = ids[:]; rng.shuffle(rows)
    order = list(TYPES); rng.shuffle(order)
    return dict(me=me, K=K, others="shown", t=t, T=rt.T, n=n, pos=pos, points=rng.randint(0, 12), nA=sA, nB=sB,
                stats=stats, hist={me: h}, order_rows=rows, earlier=earlier, open=opened, type_order=order, c=c, D=D)


def covariates(view, others_prof=None):
    me = view["me"]
    h = view["hist"][me][-HWIN:] if view["K"] > 0 else []
    oth = [a for a in view["stats"] if a != me]
    dA = sum(view["stats"][a]["done"]["A"] for a in oth); dB = sum(view["stats"][a]["done"]["B"] for a in oth)
    return dict(sA=view["nA"], sB=view["nB"], hA=h.count("A"), hB=h.count("B"), L=len(h), K=view["K"],
                pos=view["pos"], n=view["n"], t=view["t"], T=view["T"], first=view["type_order"][0],
                othA=dA / (dA + dB) if dA + dB else 0.5, others=view["others"], c=view["c"],
                **({"others_prof": others_prof} if others_prof else {}))


def outcome(text, view):
    act = parse(text)
    o = dict(action=act[0])
    if act[0] == "do":
        task = next((x for x in view["open"] if x["id"] == act[1]), None)
        if task is None:
            o["action"] = "stale"
        else:
            o["type"] = task["type"]; o["correct"] = rt.correct(task, act[2])
    return o


def ask(model, view, tag):
    sysp = system_prompt(view["me"], view["c"], view["D"], view["n"], view["T"], view["K"], view["others"])
    return reply(model, sysp, render(view), view, tag)


def run_cells(models, cells, path, tag, parallel):
    """cells: list of (set, cell key, rep, builder(rng) -> (view, extra)) per model."""
    done = _done_keys(path)
    jobs = [(m, s, ck, rep, b) for m in models for (s, ck, rep, b) in cells if (m, s, ck, rep) not in done]
    print(f"{tag}: {len(jobs)} calls", flush=True)

    def go(job):
        m, s, ck, rep, build = job
        view, extra = build(random.Random(seed_of(m, s, ck, rep)))
        text = ask(m, view, tag)
        _append(path, dict(model=m, set=s, cell=ck, rep=rep, response=(text or "")[:200],
                           **covariates(view), **extra, **outcome(text, view)))
    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(go, jobs))


# ----------------------------------------------------------------------------------------------- RT1

COMP_ITEMS = [
    ("net", "After its effort, how many points does one correct task add to the team?", lambda a: _num(a) == 0.5),
    ("rest", "How many points does it cost the team when you rest?", lambda a: _num(a) == 0),
    ("wait3", "At the end of a round 3 tasks are waiting. How many points does the team lose for them?",
     lambda a: _num(a) == 3),
    ("worth", "Is a type A task worth more than a type B task? Answer yes or no.", lambda a: _word(a) == "no"),
    ("wrong", "How many points does a wrong answer cost the team?", lambda a: _num(a) == 0.5),
    ("arrB", "How many new tasks of type B arrive at the start of each round?", lambda a: _num(a) == 2),
    ("later", "It is your turn; 2 tasks are open and 5 agents, you included, still have to move this round. If you "
              "rest, can the open tasks still be done this round? Answer yes or no.", lambda a: _word(a) == "yes"),
    ("last", "It is your turn; 4 tasks are open and you are the last agent to move this round. If you rest and no task "
             "arrives before the round ends, how many points does the team lose for waiting tasks at the end of this "
             "round?", lambda a: _num(a) == 4),
    ("cap", "At most how many tasks of one type can wait at once?", lambda a: _num(a) == 12),
    ("rounds", "How many rounds does the game last?", lambda a: _num(a) == 24),
]


def _num(a):
    m = re.search(r"-?\d+(?:\.\d+)?", a or "")
    return float(m.group(0)) if m else None


def _word(a):
    m = re.search(r"[a-z]+", (a or "").lower())
    return m.group(0) if m else None


def cmd_comp(a):
    done = _done_keys(COMP)
    answers = {"net": "0.5", "rest": "0", "wait3": "3", "worth": "no", "wrong": "0.5", "arrB": "2", "later": "yes",
               "last": "4", "cap": "12", "rounds": "24"}
    items, c_comp = COMP_ITEMS, 0.5
    if rt.FRAME == "rest":     # RT-R: the three cost items change; comprehension prompt uses r = 2
        c_comp = 2.0
        alt = {"net": ("How many points does one correct task earn the team?", lambda x: _num(x) == 1),
               "rest": ("How many points does the team get when you rest?", lambda x: _num(x) == 2),
               "wrong": ("How many points does a wrong answer earn the team?", lambda x: _num(x) == 0)}
        items = [(k,) + alt[k] if k in alt else (k, q, chk) for (k, q, chk) in COMP_ITEMS]
        answers.update(net="1", rest="2", wrong="0")
    jobs = [(m, k, q, chk, r) for m in a.models for (k, q, chk) in items for r in range(a.reps)
            if (m, "comp", k, r) not in done]
    print(f"comp: {len(jobs)} calls", flush=True)

    def go(job):
        m, k, q, chk, r = job
        sysp = system_prompt(10 + r, c_comp, 0.5)
        prompt = f"Question about the rules (answer with a single number or word, nothing else): {q}"
        if rt.FAKE:
            text = answers[k]
        else:
            from llm import call
            text = call(m, sysp, prompt, tag="rt:comp", max_tokens=rt.MAX_TOKENS)
        _append(COMP, dict(model=m, set="comp", cell=k, rep=r, response=(text or "")[:120], ok=bool(chk(text))))
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, jobs))


def calib_cells():
    out = []
    for c in C_GRID:
        for (sA, sB) in ((1, 0), (1, 1), (2, 2), (4, 4)):
            for pos in (1, 8):
                for r in range(20):
                    a_, b_ = (sA, sB) if (sA == sB or r % 2 == 0) else (sB, sA)
                    ck = f"c={c}|s={sA}+{sB}|pos={pos}"
                    out.append(("calib", ck, r, lambda rng, c=c, a_=a_, b_=b_, pos=pos:
                                (synth(rng, a_, b_, PROFILES["3A3B2R"], pos, c), {})))
    return out


def cmd_calib(a):
    run_cells(a.models, calib_cells(), CALIB, "rt:calib", a.parallel)


def comp_scores():
    sc = {}
    for r in map(json.loads, open(COMP)):
        sc.setdefault(r["model"], []).append(r["ok"])
    return {m: float(np.mean(v)) for m, v in sc.items()}


def cmd_choose(a):
    if rt.CCHOICE.exists():
        print("c already chosen:", json.load(open(rt.CCHOICE))); return
    comp = comp_scores()
    passing = [m for m in a.models if comp.get(m, 0) >= 0.6]
    if not passing:
        sys.exit("no model passed comprehension; stopping")
    rows = [r for r in map(json.loads, open(CALIB)) if r["model"] in passing]
    share = {}
    for c in C_GRID:
        per = []
        for m in passing:
            v = [r["action"] == "do" for r in rows if r["model"] == m and r["cell"] == f"c={c}|s=2+2|pos=1"
                 and r["action"] in ("do", "rest")]
            per.append(float(np.mean(v)) if v else float("nan"))
        share[c] = float(np.nanmean(per))
    saturated = all(v > 0.9 for v in share.values())
    c = max(C_GRID) if saturated else min(C_GRID, key=lambda x: (abs(share[x] - 0.5), x))
    out = dict(c=c, saturated=saturated, share_do_at_2plus2_pos1=share, models=passing, comprehension=comp)
    json.dump(out, open(rt.CCHOICE, "x"), indent=1)       # write once
    print("chosen:", out)


def cmd_gate(a):
    comp = comp_scores()
    pil = {}
    for r in map(json.loads, open(R / "rt_pilot_pop.jsonl")):
        for rr in r["history"]:
            for x in rr["acts"].values():
                pil.setdefault(r["model"], []).append(x[0] in ("do", "rest"))
    valid = {m: float(np.mean(v)) for m, v in pil.items()}
    passing = [m for m in a.models if comp.get(m, 0) >= 0.6 and valid.get(m, 0) >= 0.9]
    if GATE.exists():          # RT-A1: a later model is added to the earlier gate, not instead of it
        passing = [m for m in json.load(open(GATE))["passing"] if m not in a.models] + passing
    out = dict(passing=passing, comprehension=comp, pilot_valid=valid)
    json.dump(out, open(GATE, "w"), indent=1)
    print(out)
    if not [m for m in a.models if m in passing]:
        sys.exit(3)


def gated(models):
    g = json.load(open(GATE))["passing"]
    return [m for m in models if m in g]


# ----------------------------------------------------------------------------------------------- RT2

def _grid_builder(sA, sB, pk, c):
    def b(rng):
        return synth(rng, sA, sB, PROFILES[pk], rng.randint(1, rt.N), c), {"profile": pk}
    return b


def grid_cells(c, reps=range(8), only=None):
    out = []
    for sA in S_GRID:
        for sB in S_GRID:
            if sA == sB == 0:
                continue
            for pk in PROFILES:
                ck = f"s={sA}+{sB}|h={pk}"
                if only is not None and ck not in only:
                    continue
                out += [("grid", ck, r, _grid_builder(sA, sB, pk, c)) for r in reps]
    return out


def cmd_grid(a):
    c = rt.chosen_c()
    run_cells(gated(a.models), grid_cells(c), KERNEL, "rt:grid", a.parallel)


def swap_view(v):
    v = json.loads(json.dumps(v))
    me = v["me"]
    key = str(me) if str(me) in v["stats"] else me
    d = v["stats"][key]["done"]; d["A"], d["B"] = d["B"], d["A"]
    hk = str(me) if str(me) in v["hist"] else me
    v["hist"][hk] = [{"A": "B", "B": "A"}.get(x, x) for x in v["hist"][hk]]
    return v


def _norm(v):
    """JSON round-trip turns int dict keys into strings; restore them so render() finds the agent."""
    v = json.loads(json.dumps(v))
    v["stats"] = {int(k): s for k, s in v["stats"].items()}
    v["hist"] = {int(k): h for k, h in v["hist"].items()}
    return v


def replay_states(model, k=150):
    pool = []
    for r in map(json.loads, open(R / "rt_pilot_pop.jsonl")):
        if r["model"] != model:
            continue
        for i, v in enumerate(r["views"]):
            v = _norm(v)
            h = v["hist"][v["me"]][-HWIN:]
            if v["t"] >= 3 and v["open"] and any(x in TYPES for x in h) and h.count("A") != h.count("B"):
                pool.append((f"seed={r['seed']}|i={i}", v))
    rng = random.Random(seed_of("replay", model))
    rng.shuffle(pool)
    return pool[:k]


def cmd_replay(a):
    for m in gated(a.models):
        cells = []
        for sid, v in replay_states(m, a.k):
            for cond, vv in (("actual", v), ("swapped", _norm(swap_view(v)))):
                for r in range(8):
                    cells.append(("replay", f"{sid}|{cond}", r, lambda rng, vv=vv, cond=cond, sid=sid:
                                  (_norm(vv), {"cond": cond, "state": sid})))
        run_cells([m], cells, KERNEL, "rt:replay", a.parallel)


def cmd_social(a):
    c = rt.chosen_c()
    cells = []
    for op in ("Aspec", "Bspec", "mixed"):
        for (sA, sB) in SOCIAL_S:
            for r in range(12):
                cells.append(("social", f"o={op}|s={sA}+{sB}", r, lambda rng, op=op, sA=sA, sB=sB:
                              (synth(rng, sA, sB, PROFILES["3A3B2R"], rng.randint(1, rt.N), c, others_prof=op),
                               {"others_prof": op})))
    run_cells(gated(a.models), cells, KERNEL, "rt:social", a.parallel)


def cmd_topup(a):
    c = rt.chosen_c()
    rows = [r for r in map(json.loads, open(KERNEL)) if r["set"] == "grid" and r["rep"] < 8]
    for m in gated(a.models):
        p = {}
        for r in rows:
            if r["model"] == m and r["action"] in ("do", "rest"):
                p.setdefault(r["cell"], []).append(r["action"] == "do")
        sel = {ck for ck, v in p.items() if 0.15 <= np.mean(v) <= 0.85}
        print(f"topup {m}: {len(sel)} cells", flush=True)
        run_cells([m], grid_cells(c, reps=range(8, 16), only=sel), KERNEL, "rt:topup", a.parallel)


def cmd_summary(a):
    rows = list(map(json.loads, open(KERNEL)))
    for m in sorted({r["model"] for r in rows}):
        print(f"== {m}")
        g = [r for r in rows if r["model"] == m and r["set"] == "grid"]
        for pk in PROFILES:
            sub = [r for r in g if r["profile"] == pk and r["action"] in ("do", "rest")]
            w = np.mean([r["action"] == "do" for r in sub]) if sub else float("nan")
            dA = [r["type"] == "A" for r in sub if r["action"] == "do" and r["sA"] > 0 and r["sB"] > 0]
            print(f"  {pk:7s} n={len(sub):4d}  P(work) {w:.2f}  P(A | work, both open) {np.mean(dA) if dA else float('nan'):.2f}")
        inv = np.mean([r["action"] in ("invalid", "stale") for r in rows if r["model"] == m])
        print(f"  invalid/stale {inv:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["comp", "calib", "choose", "gate", "grid", "replay", "social", "topup", "summary"])
    p.add_argument("--models", nargs="+", default=list(rt.MODELS))
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--k", type=int, default=150)
    p.add_argument("--parallel", type=int, default=10)
    a = p.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
