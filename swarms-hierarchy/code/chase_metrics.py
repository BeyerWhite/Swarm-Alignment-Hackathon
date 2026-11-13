"""Chase-style structure metrics for contest logs (Chase 1982 triad sequences; Chase et al. 2002 linearity).

All functions take `events`: a list of (round, winner, loser) with agents as 0..N-1 indices, in the order the
contests were decided (round order, then within-round event order).

Definitions (frozen in refine-logs/CHASE_PREREGISTRATION.md):
  * A dyad's relationship FORMS at its first contest; the winner of that contest is the initial dominant.
  * A dyad's FINAL relationship is the majority winner over the last `window` rounds; ties are broken by the
    most recent contest in the window; a dyad with no contest in the window falls back to the whole run;
    a dyad that never met is undecided.
  * Triad sequence (Chase 1982): among the three dyads of a triad, take the first two to form.
      DD   double dominance      - the same agent dominates in both
      DS   double subordinance   - the same agent is subordinate in both
      BDID bystander dominates the initial dominant
      ISDB initial subordinate dominates the bystander
    DD and DS guarantee a transitive triad whatever the third relationship is.
"""
import collections, itertools
import numpy as np

SEQ = ("DD", "DS", "BDID", "ISDB")


def events_from_history(rec):
    """Flatten a population record's history to [(t, winner, loser)]."""
    out = []
    for x in rec["history"]:
        for ev in x["events"]:
            i, j, w = ev[0], ev[1], ev[4]
            out.append((x["t"], w, j if w == i else i))
    return out


def first_relations(events):
    """{frozenset(i,j): (order_index, winner, loser)} for each dyad's first contest."""
    first = {}
    for k, (t, w, l) in enumerate(events):
        first.setdefault(frozenset((w, l)), (k, w, l))
    return first


def final_relations(events, rounds, window):
    """{frozenset(i,j): dominant} by the frozen majority rule; undecided dyads are absent."""
    lo = rounds - window
    win, allw, last_w, last_a = collections.Counter(), collections.Counter(), {}, {}
    for t, w, l in events:
        k = frozenset((w, l))
        allw[(w, l)] += 1; last_a[k] = w
        if t >= lo:
            win[(w, l)] += 1; last_w[k] = w
    rel = {}
    for k in last_a:
        i, j = tuple(k)
        src, last = (win, last_w) if k in last_w else (allw, last_a)
        if src[(i, j)] != src[(j, i)]:
            rel[k] = i if src[(i, j)] > src[(j, i)] else j
        else:
            rel[k] = last[k]
    return rel


def triad_sequences(events, N):
    """Counts of Chase's four sequence types over all triads with at least two formed dyads."""
    first = first_relations(events)
    seq = collections.Counter()
    for tri in itertools.combinations(range(N), 3):
        ds = sorted(first[frozenset(p)] for p in itertools.combinations(tri, 2) if frozenset(p) in first)
        if len(ds) < 2:
            continue
        (_, w1, l1), (_, w2, l2) = ds[0], ds[1]
        seq["DD" if w2 == w1 else "DS" if l2 == l1 else "BDID" if l2 == w1 else "ISDB"] += 1
    return seq


def triad_census(rel, N):
    """(transitive, cyclic, incomplete) triads under the final relations."""
    tr = cy = inc = 0
    for a, b, c in itertools.combinations(range(N), 3):
        ks = [frozenset(p) for p in ((a, b), (a, c), (b, c))]
        if any(k not in rel for k in ks):
            inc += 1; continue
        dom = collections.Counter(rel[k] for k in ks)
        if len(dom) == 3:
            cy += 1
        else:
            tr += 1
    return tr, cy, inc


def landau_h(W):
    """de Vries (1995) h' (same estimator as population2.landau_h)."""
    n = len(W)
    off = ~np.eye(n, dtype=bool)
    seen = (W + W.T) > 0
    D = (W > W.T).astype(float) + 0.5 * ((W == W.T) & off)
    V = D.sum(1)
    u = int(((~seen) & off).sum() // 2)
    return float(12 / (n ** 3 - n) * np.sum((V - (n - 1) / 2) ** 2) + 6 * u / (n ** 3 - n))


def structure(events, N, rounds, window=None):
    """All Chase metrics for one group."""
    window = window or rounds // 2
    seq = triad_sequences(events, N)
    nseq = sum(seq.values())
    rel = final_relations(events, rounds, window)
    tr, cy, inc = triad_census(rel, N)
    complete = len(rel) == N * (N - 1) // 2
    W = np.zeros((N, N))
    lo = rounds - window
    for t, w, l in events:
        if t >= lo:
            W[w, l] += 1
    tot = W.sum()
    return dict(
        n_events=len(events),
        seq={k: int(seq[k]) for k in SEQ},
        dd_ds=float((seq["DD"] + seq["DS"]) / nseq) if nseq else float("nan"),
        transitive=tr, cyclic=cy, incomplete=inc,
        p_trans=float(tr / (tr + cy)) if tr + cy else float("nan"),
        t_tri=float(4 * (tr / (tr + cy) - 0.75)) if tr + cy else float("nan"),
        linear=bool(complete and cy == 0),
        complete=bool(complete),
        landau=landau_h(W),
        dci=float(np.abs(W - W.T).sum() / 2 / tot) if tot else float("nan"),   # directional consistency
    )


def pooled(structs):
    """Pool groups: sequence shares and transitivity over all triads, plus per-group means."""
    seq = collections.Counter(); tr = cy = 0
    for s in structs:
        seq.update(s["seq"]); tr += s["transitive"]; cy += s["cyclic"]
    n = sum(seq.values())
    comp = [s for s in structs if s["complete"]]
    return dict(
        groups=len(structs), triads_seq=n,
        **{k: (seq[k] / n if n else float("nan")) for k in SEQ},
        dd_ds=(seq["DD"] + seq["DS"]) / n if n else float("nan"),
        p_trans=tr / (tr + cy) if tr + cy else float("nan"),
        t_tri=4 * (tr / (tr + cy) - 0.75) if tr + cy else float("nan"),
        linear=float(np.mean([s["linear"] for s in comp])) if comp else float("nan"),
        n_complete=len(comp),
        landau=float(np.mean([s["landau"] for s in structs])),
        dci=float(np.nanmean([s["dci"] for s in structs])),
    )
