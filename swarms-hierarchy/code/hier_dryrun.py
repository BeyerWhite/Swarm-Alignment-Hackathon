"""Offline dry run of the whole phase-1 pipeline with a fake LLM backend (no API calls, no credits).

The fake agent solves each task kind from its text and is right with a model-specific probability; it plans with a
model-specific propensity. Every stage runs in order, into a scratch results folder:
  battery -> select -> comp -> pilot -> gate -> g1a -> summarize -> g1b/g1d/g1e -> g1c -> summarize -> freeze
  -> g2 -> bench -> g3check -> freeze_g3 -> g3 -> g4 -> eval
Usage: HIER_RESULTS=<empty scratch dir> python hier_dryrun.py [--small]
"""
import os, random, re, sys
from pathlib import Path

assert os.environ.get("HIER_RESULTS"), "set HIER_RESULTS to a scratch folder"
sys.path.insert(0, str(Path(__file__).parent))
import llm
import hier_tasks as HT

DAYS = HT.DAYS
ACC = {"gemma-4-31b-it": 0.8, "gemma-4-26b-a4b-it": 0.45, "qwen3.8-27b": 0.65, "deepseek-v4-flash": 0.55,
       "qwen3.5-397b": 0.9, "glm-5.2": 0.85}
PLAN = {"gemma-4-31b-it": 0.25, "gemma-4-26b-a4b-it": 0.2, "qwen3.8-27b": 0.1, "deepseek-v4-flash": 0.5,
        "qwen3.5-397b": 0.02, "glm-5.2": 0.15}


def solve(text):
    m = re.match(r"What is (\d+) × (\d+)\?", text)
    if m: return str(int(m[1]) * int(m[2]))
    m = re.match(r"What is (\d+) − (\d+)\?", text)
    if m: return str(int(m[1]) - int(m[2]))
    m = re.match(r"What is ([\d + ]+)\?", text)
    if m: return str(sum(map(int, m[1].split("+"))))
    m = re.match(r"How many times does the letter '(.)' appear in '(\w+)'", text)
    if m: return str(m[2].count(m[1]))
    m = re.match(r"Write '(\w+)' backwards", text)
    if m: return m[1][::-1]
    m = re.match(r"Today is (\w+)\. What day of the week will it be in (\d+) days", text)
    if m: return DAYS[(DAYS.index(m[1]) + int(m[2])) % 7]
    m = re.match(r"Sort these numbers from smallest to largest: ([\d, ]+)\.", text)
    if m: return ", ".join(map(str, sorted(map(int, m[1].split(",")))))
    m = re.match(r"Replace each letter of '(\w+)' by the letter (\d+) place", text)
    if m: return "".join(chr((ord(c) - 97 + int(m[2])) % 26 + 97) for c in m[1])
    m = re.match(r"What is the remainder when (\d+) is divided by (\d+)", text)
    if m: return str(int(m[1]) % int(m[2]))
    m = re.match(r"Which letter comes (\d+) letters after '(.)'", text)
    if m: return chr(ord(m[2]) + int(m[1]))
    m = re.match(r"What is the sum of the digits of (\d+)", text)
    if m: return str(sum(map(int, m[1])))
    return "?"


def fake_call(model, system, prompt, tag="", timeout=180, retries=2, max_tokens=16):
    name = model.split(":", 1)[1]
    r = random.Random(hash((name, system, prompt, tag)) & 0xFFFFFFFF)
    if "ANSWER: <letter>" in prompt:                 # quiz: right 80% of the time
        from hier_measure import QUIZ
        q = next(q for q in QUIZ if prompt.startswith(q[0]))
        opts = dict(re.findall(r"^([A-C])\) (.+)$", prompt, re.M))
        right = next(l for l, o in opts.items() if o == q[1][q[2]])
        return "ANSWER: " + (right if r.random() < 0.8 else r.choice(list(opts)))
    me = int(re.search(r"You are Agent (\d+)", system)[1])
    tasks = re.findall(r"- T(\d+) \[(\w+)\]: (.+)", prompt)
    agents = [int(a) for a in re.findall(r"- Agent (\d+)(?: \(you\))?:", prompt.split("Current plans")[0])
              if int(a) != me]
    manager = "You are the team's coordinator" in system
    if manager or ("act as coordinator" in system and r.random() < PLAN[name] * (2 if "take charge" in system else 1)):
        rec = {int(a): dict((k, (int(o), int(n))) for k, o, n in re.findall(r"(\w+) (\d+)/(\d+)", body))
               for a, body in re.findall(r"- Agent (\d+)[^:]*: correct/attempted by kind: ([^;\n]+)", prompt)}
        left, pairs = list(tasks), []
        for a in agents:                             # good planners match kinds to records, bad ones guess
            if not left:
                break
            if r.random() < ACC[name] - 0.3 and a in rec:
                score = lambda t: (rec[a].get(t[1], (0, 0))[0] + 1) / (rec[a].get(t[1], (0, 0))[1] + 2)
                t = max(left, key=score)
            else:
                t = r.choice(left)
            left.remove(t); pairs.append(f"T{t[0]}->{a}")
        return "PLAN: " + ", ".join(pairs)
    mine = re.search(r"Assigned to you: T(\d+)", prompt)
    tid = mine[1] if mine and r.random() < 0.7 else (tasks[0][0] if r.random() < 0.6 else r.choice(tasks)[0])
    text = next(t[2] for t in tasks if t[0] == tid)
    return f"DO {tid}: " + (solve(text) if r.random() < ACC[name] else "12345")


def run(mod, *args):
    import importlib
    m = importlib.import_module(mod)
    sys.argv = [mod + ".py", *map(str, args)]
    print(f"\n### {mod} {' '.join(map(str, args))}", flush=True)
    m.main()


if __name__ == "__main__":
    llm.call = fake_call
    rng = random.Random(1)
    for k in HT.KINDS:                     # the fake solver must agree with the generators
        for lev in (1, 2):
            t = HT.make_task(0, k, lev, rng)
            assert HT.correct(t, solve(t["text"])), (t, solve(t["text"]))
    small = "--small" in sys.argv
    s = lambda a, b: list(range(a, a + (2 if small else b)))
    run("hier_measure", "battery"); run("hier_measure", "select"); run("hier_measure", "comp")
    run("office", "pilot", "--seeds", 1, 2); run("hier_measure", "gate")
    run("hier_measure", "g1a"); run("hier_measure", "summarize")
    for part in ("g1b", "g1d", "g1e"):
        run("hier_measure", part)
    run("office", "g1c", "--seeds", *s(61001, 8)); run("hier_measure", "summarize")
    run("hier_sim", "freeze")
    run("office", "g2", "--seeds", *s(62001, 12)); run("office", "bench", "--seeds", *s(62001, 6))
    run("hier_measure", "g3check"); run("hier_sim", "freeze_g3")
    run("office", "g3", "--seeds", *s(62001, 12)); run("office", "g4", "--seeds", *s(64001, 8))
    run("hier_eval", "--draws", 500)
