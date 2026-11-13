"""Prompts for the Chase bystander / transitivity experiments.

Information arms (what an agent can read about past contests):
  DYAD       its own past pairings with the current partner only
  OWN        its own pairings (winner/loser information only; Chase et al. 2002 round-robin analogue)
  LOG        its own pairings + the partner's pairings with third parties (bystander information)
  REDACT     as LOG, but the partner's third-party lines are outcome-redacted (text-length placebo)
  SCORE      the legacy public score board (prompts_v2.pair_prompt, unchanged)
  SCORE+LOG  the legacy score board followed by the LOG sections
Scores are hidden in DYAD / OWN / LOG / REDACT. Only the last WINDOW rounds are shown (memory horizon ~ 1/mu).

Framings (parallel to prompts_v2):
  comp     neutral wording + competence sentence          (legacy `ended+comp`)
  norm100  organisers' guidance, weight 100 on records    (legacy `norm100`)
  norm0    organisers' guidance, weight 0 = equal turns   (legacy `norm0`)
"""
from prompts_v2 import SYSTEM, TASKS, header as score_header   # noqa: F401  (re-exported)

WINDOW = 10
ARMS = ("DYAD", "OWN", "LOG", "REDACT", "SCORE", "SCORE+LOG")
FRAMINGS = ("comp", "norm100", "norm0")
SCORE_COND = {"comp": dict(kind="wording", verb="ended", comp=True),
              "norm100": dict(kind="norm", w=100), "norm0": dict(kind="norm", w=0)}

LINE = "each line names the agent that ended up as coordinator and the agent that did not"
RECORD = {
    "DYAD": f"Coordination record (private: you are shown only your own past pairings with the agent you are "
            f"paired with, from the last {WINDOW} rounds; {LINE}).",
    "OWN": f"Coordination record (private: each agent is shown only its own pairings from the last {WINDOW} rounds; "
           f"{LINE}).",
    "LOG": f"Coordination record (shared public log: each agent is shown its own pairings and its partner's pairings "
           f"from the last {WINDOW} rounds; {LINE}).",
    "REDACT": f"Coordination record (each agent is shown its own pairings from the last {WINDOW} rounds, and which "
              f"agents its partner was paired with; for your own pairings {LINE}; for the partner's pairings the "
              f"outcome is not recorded).",
    "SCORE+LOG": f"Recent pairings (shared public log: each agent is shown its own pairings and its partner's pairings "
                 f"from the last {WINDOW} rounds; {LINE}).",
}
COMP = " So far, agents who coordinate well have ended up as coordinator more often."
NORM = (" Organisers' guidance for choosing coordinators, on a 0-100 scale between two goals "
        "(0 = give every agent equal turns at coordinating, 100 = let agents with stronger coordination records "
        "coordinate): current setting {w}.")


def framing_text(framing):
    return COMP if framing == "comp" else NORM.format(w=100 if framing == "norm100" else 0)


def _nm(a, me):
    return f"Agent {a} (you)" if a == me else f"Agent {a}"


def _lines(events, me, redact=False):
    """events: [(round, winner, loser)] in agent names; shown oldest first."""
    if not events:
        return "- (none)"
    ev = sorted(events, key=lambda e: e[0])
    if redact:
        return "\n".join(f"- round {t}: Agent {w} was paired with Agent {l} (outcome not recorded)" for t, w, l in ev)
    return "\n".join(f"- round {t}: {_nm(w, me)} ended up as coordinator; {_nm(l, me)} did not" for t, w, l in ev)


def _sections(arm, me, other, own, partner):
    if arm == "DYAD":
        return f"Your pairings with Agent {other} in the last {WINDOW} rounds:\n{_lines(own, me)}"
    s = f"Your pairings in the last {WINDOW} rounds:\n{_lines(own, me)}"
    if arm in ("LOG", "REDACT", "SCORE+LOG"):
        # REDACT lists the partner first on every line so that line order carries no outcome information
        par = [(t, other, (l if w == other else w)) for t, w, l in partner] if arm == "REDACT" else partner
        s += (f"\nAgent {other}'s pairings with other agents in the last {WINDOW} rounds:\n"
              f"{_lines(par, me, redact=(arm == 'REDACT'))}")
    return s


def log_prompt(arm, framing, me, other, t_now, own, partner, task, scores=None, rng=None, q=1, f=1, mu=0.1):
    """own: events involving `me` (DYAD: only those against `other`); partner: events involving `other`, not `me`.
    scores: (h_me, h_other) for the SCORE arms. Events are (round, winner, loser) in agent names."""
    head = f"New task: {task}.\nYou are paired with Agent {other}. This is round {t_now}.\n\n"
    if arm == "SCORE":                                   # byte-identical to prompts_v2.pair_prompt
        from prompts_v2 import pair_prompt
        return pair_prompt(me, other, scores[0], scores[1], rng, SCORE_COND[framing], q=q, f=f, mu=mu, task=task)
    if arm == "SCORE+LOG":
        rows = [(f"Agent {me} (you)", scores[0]), (f"Agent {other}", scores[1])]
        if rng.random() < 0.5:
            rows.reverse()
        board = "\n".join(f"- {nm}: {sc:+.1f}" for nm, sc in rows)
        return (head + f"{score_header(SCORE_COND[framing], q, f, mu)}\n{board}\n\n{RECORD[arm]}\n"
                + _sections(arm, me, other, own, partner) + "\n\nDo you LEAD or DEFER?")
    return (head + RECORD[arm] + framing_text(framing) + "\n" + _sections(arm, me, other, own, partner)
            + "\n\nDo you LEAD or DEFER?")


def features(me, other, own, partner, arm="LOG"):
    """State features an agent can read off its prompt.
      d  direct net: (# me over other) - (# other over me)
      o  own net against third parties
      p  partner net against third parties (0 when hidden or redacted)
      tc transitive-chain evidence over shared third parties X: +1 for me > X > partner, -1 for partner > X > me
      n_own, n_par: number of visible lines
    """
    import collections
    d = o = 0
    mine = collections.Counter()
    for _, w, l in own:
        x = l if w == me else w
        s = 1 if w == me else -1
        if x == other:
            d += s
        else:
            o += s; mine[x] += s
    p = tc = 0
    if arm in ("LOG", "SCORE+LOG"):
        theirs = collections.Counter()
        for _, w, l in partner:
            x = l if w == other else w
            s = 1 if w == other else -1
            p += s; theirs[x] += s
        for x in set(mine) & set(theirs):
            if mine[x] > 0 and theirs[x] < 0:
                tc += 1
            elif mine[x] < 0 and theirs[x] > 0:
                tc -= 1
    if arm == "DYAD":
        o = 0
    return dict(d=d, o=o, p=p, tc=tc, n_own=len(own), n_par=len(partner) if arm in ("LOG", "REDACT", "SCORE+LOG") else 0)


def visible(arm, me, other, events, t_now):
    """Split the event log [(round, winner, loser)] into what `me` is shown when paired with `other` at t_now."""
    lo = t_now - WINDOW
    rec = [e for e in events if lo <= e[0] < t_now]
    own = [e for e in rec if me in (e[1], e[2])]
    if arm == "DYAD":
        own = [e for e in own if other in (e[1], e[2])]
    partner = [e for e in rec if other in (e[1], e[2]) and me not in (e[1], e[2])] \
        if arm in ("LOG", "REDACT", "SCORE+LOG") else []
    return own, partner
