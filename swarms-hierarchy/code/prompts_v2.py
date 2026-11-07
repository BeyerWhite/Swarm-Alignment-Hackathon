"""v2 prompts: identical score-update wording in every condition; framing enters only through
(a) a numeric norm weight w in [0, 100] (equal turns vs. deference to records), or
(b) a 2x2 wording factorial (verb 'won' vs 'ended up as' x competence sentence present/absent)."""
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

RULE = {
    "ended": "after each pairing, the agent that ended up as coordinator gets +{q} and the other agent gets -{f}",
    "won": "after each pairing, the agent that won the coordinator role gets +{q} and the agent that lost it gets -{f}",
}
COMPETENCE = " So far, higher scores have gone with agents who coordinate well."
NORM = (" Organisers' guidance for choosing coordinators, on a 0-100 scale between two goals "
        "(0 = give every agent equal turns at coordinating, 100 = let agents with higher scores coordinate): "
        "current setting {w}.")


def header(cond, q=1, f=1, mu=0.1):
    """cond: dict(kind='norm', w=int) or dict(kind='wording', verb='won'|'ended', comp=bool)."""
    verb = cond.get("verb", "ended")
    s = "Scores (shared public record; " + RULE[verb].format(q=f"{q:g}", f=f"{f:g}") + \
        f"; all scores shrink by {int(round(mu * 100))}% every round)."
    if cond.get("comp"):
        s += COMPETENCE
    if cond["kind"] == "norm":
        s += NORM.format(w=cond["w"])
    return s


def cond_name(cond):
    if cond["kind"] == "norm":
        return f"norm{cond['w']}"
    return f"{cond['verb']}{'+comp' if cond.get('comp') else ''}"


def pair_prompt(me, other, h_me, h_other, rng, cond, q=1, f=1, mu=0.1, board=None, task=None):
    """board: optional list of (name, score) for other agents, shown as a full leaderboard."""
    rows = [(f"Agent {me} (you)", h_me), (f"Agent {other}", h_other)]
    if board:
        rows += [(f"Agent {nm}", sc) for nm, sc in board]
        rows.sort(key=lambda r: -r[1])
    elif rng.random() < 0.5:
        rows.reverse()
    lines = "\n".join(f"- {nm}: {sc:+.1f}" for nm, sc in rows)
    return (f"New task: {task or rng.choice(TASKS)}.\nYou are paired with Agent {other}.\n\n"
            f"{header(cond, q, f, mu)}\n{lines}\n\nDo you LEAD or DEFER?")
