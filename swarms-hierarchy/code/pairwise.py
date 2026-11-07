"""Pairwise micro-experiment: measure an LLM agent's decision curve p(LEAD | h_self, h_other).

Contest protocol (simultaneous): both agents answer LEAD/DEFER; (LEAD, DEFER) -> first wins,
(DEFER, LEAD) -> second wins, ties broken by coin.  Hence
    P_ij = 1/2 + (p(h_i, h_j) - p(h_j, h_i)) / 2,
so the Bonabeau sensitivity eta is identified from the single-agent decision curve.
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from llm import batch, parse_choice

TASKS = [
    "summarise a 40-page technical report and produce a slide deck",
    "clean a messy customer dataset and write a short analysis",
    "plan the logistics for a three-day offsite for 30 people",
    "triage and fix a backlog of 25 bug reports in a web app",
    "draft a grant proposal with budget, timeline and literature review",
    "translate and localise product documentation into four languages",
]

SYSTEM = ("You are Agent {me}, one of {n} identical AI agents working in a simulated multi-agent sandbox. "
          "Agents are repeatedly paired to complete tasks. In each pairing exactly one agent becomes the "
          "coordinator: it decides how to split the work and assigns subtasks to the other agent. "
          "Reply with exactly one word: LEAD (you claim the coordinator role) or DEFER (you let the other "
          "agent coordinate). If both agents give the same answer, the role is assigned by a coin flip.")

FRAMES = {
    # status = count of coordinator roles obtained (pure dominance record)
    "reward": "Standing scores (shared public record; +{q} each time an agent ends up coordinator, -{f} each "
              "time it ends up being assigned work; all scores shrink by {mu_pct}% every round):",
    # status = track record that reads as competence
    "competence": "Track-record scores (shared public record; +{q} each time an agent won a coordination "
                  "contest, -{f} each time it lost one; all scores shrink by {mu_pct}% every round). Higher "
                  "scores have so far gone with agents who coordinate well:",
}
SCALAR = "New task: {task}.\nYou are paired with Agent {other}.\n\n{header}\n{lines}\n\nDo you LEAD or DEFER?"


def scalar_prompt(me, other, h_me, h_other, n, rng, q=1, f=1, mu=0.1, frame="reward"):
    rows = [f"- Agent {me} (you): {h_me:+.1f}", f"- Agent {other}: {h_other:+.1f}"]
    if rng.random() < 0.5:
        rows.reverse()                     # randomise presentation order
    header = FRAMES[frame].format(q=q, f=f, mu_pct=int(round(mu * 100)))
    return SCALAR.format(task=rng.choice(TASKS), other=other, header=header, lines="\n".join(rows))


def run(models, reps, out, seed=0, workers=8, frames=("reward",)):
    rng = random.Random(seed)
    deltas = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
    means = [-2.0, 0.0, 3.0]
    jobs, meta = [], []
    for m, fr in [(m, fr) for m in models for fr in frames]:
        for d in deltas:
            for mbar in means:
                for r in range(reps):
                    me, other = rng.sample(range(1, 33), 2)
                    h_me, h_ot = mbar + d / 2, mbar - d / 2
                    jobs.append(dict(model=m, system=SYSTEM.format(me=me, n=16),
                                     prompt=scalar_prompt(me, other, h_me, h_ot, 16, rng, frame=fr), tag="pairwise"))
                    meta.append(dict(model=m, frame=fr, delta=d, mean=mbar, h_me=h_me, h_other=h_ot))
    rng.shuffle(order := list(range(len(jobs))))
    resp = batch([jobs[i] for i in order], workers=workers)
    with open(out, "a") as f:
        for i, r in zip(order, resp):
            f.write(json.dumps({**meta[i], "response": r, "choice": parse_choice(r)}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["claude:haiku"])
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", default=str(Path(__file__).parents[1] / "results" / "pairwise_pilot.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--frames", nargs="+", default=["reward", "competence"])
    a = ap.parse_args()
    run(a.models, a.reps, a.out, workers=a.workers, frames=a.frames)
