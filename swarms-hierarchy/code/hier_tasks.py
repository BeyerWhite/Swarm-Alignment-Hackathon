"""Task battery for the hierarchy-efficiency study (Rosen 1982 / Garicano 2000), stage G0.

Eleven task types, each at two difficulty levels, all exactly checkable. G0 measures every model's accuracy on
every (type, level) cell; the frozen rule `select()` then picks the four types used in the office game.

  make_task(tid, kind, level, rng) -> dict(id, kind, level, text, answer)
  correct(task, answer_text)        -> bool  (answer_text is what follows "DO <id>:" or "ANSWER:")
  answerable(task, answer_text)     -> bool  (an answer of the right shape could be extracted)
  select(acc, parse)                -> frozen selection (see SELECTION RULE below)
"""
import random, re, string

KINDS = ("multiply", "count", "reverse", "weekday", "sort", "shift", "subtract", "remainder", "alphabet",
         "digitsum", "add3")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORDS = ["strawberry", "parallel", "committee", "bookkeeper", "mississippi", "balloon", "assessment", "necessary",
         "occurrence", "embarrass", "possession", "millennium", "tomorrow", "beginning", "successful", "accommodate",
         "programming", "letter", "referee", "pepper", "bubble", "giraffe", "corridor", "barrel", "coffee", "address",
         "tennessee", "illusion", "carrot", "mirror", "hammer", "puzzle", "rhythm", "banana", "cinnamon", "terrace"]
SHORT = ["planet", "garden", "silver", "window", "pencil", "harbor", "meadow", "candle", "rocket", "forest", "bridge",
         "orange", "basket", "copper", "dragon", "falcon", "jungle", "lantern", "marble", "napkin", "pickle", "saddle"]
SHIFT_WORDS = {1: ["cat", "dog", "sun", "map", "hat", "pen", "box", "cup", "fan", "jam", "kit", "log"],
               2: ["garden", "silver", "window", "pencil", "rocket", "forest", "bridge", "basket", "copper", "dragon"]}


def _rand_letters(rng, n, alphabet="abcdefghijklmnoprstu"):
    return "".join(rng.choice(alphabet) for _ in range(n))


def make_task(tid, kind, level, rng):
    """level 1 = easier, 2 = harder. Returns dict(id, kind, level, text, answer)."""
    hard = level == 2
    if kind == "multiply":
        a, b = (rng.randint(100, 999), rng.randint(100, 999)) if hard else (rng.randint(13, 99), rng.randint(13, 99))
        text, ans = f"What is {a} × {b}?", str(a * b)
    elif kind == "count":
        if hard:
            s = _rand_letters(rng, 18, "abcdeirst")
            ch = rng.choice(sorted(set(s)))
        else:
            s = rng.choice(WORDS)
            ch = rng.choice(sorted({c for c in s if s.count(c) >= 2}) or sorted(set(s)))
        text, ans = f"How many times does the letter '{ch}' appear in '{s}'?", str(s.count(ch))
    elif kind == "reverse":
        w = _rand_letters(rng, 9) if hard else rng.choice(SHORT)
        text, ans = f"Write '{w}' backwards, letter by letter.", w[::-1]
    elif kind == "weekday":
        d0 = rng.randrange(7)
        n = rng.randint(100, 999) if hard else rng.randint(9, 40)
        text, ans = f"Today is {DAYS[d0]}. What day of the week will it be in {n} days?", DAYS[(d0 + n) % 7]
    elif kind == "sort":
        xs = rng.sample(range(10, 100), 9 if hard else 5)
        text = "Sort these numbers from smallest to largest: " + ", ".join(map(str, xs)) + "."
        ans = ", ".join(map(str, sorted(xs)))
    elif kind == "shift":
        k = rng.randint(3, 5) if hard else 1
        w = rng.choice(SHIFT_WORDS[2 if hard else 1])
        out = "".join(chr((ord(c) - 97 + k) % 26 + 97) for c in w)
        text, ans = (f"Replace each letter of '{w}' by the letter {k} place{'s' if k > 1 else ''} later in the alphabet "
                     f"(z wraps to a). What string do you get?"), out
    elif kind == "subtract":
        a = rng.randint(100000, 999999) if hard else rng.randint(300, 999)
        b = rng.randint(10000, a - 1) if hard else rng.randint(12, 99)
        text, ans = f"What is {a} − {b}?", str(a - b)
    elif kind == "remainder":
        a, d = (rng.randint(100000, 999999), rng.choice([13, 17, 19])) if hard else (rng.randint(100, 999), 7)
        text, ans = f"What is the remainder when {a} is divided by {d}?", str(a % d)
    elif kind == "alphabet":
        k = rng.randint(9, 15) if hard else rng.randint(2, 5)
        c = rng.choice(string.ascii_lowercase[: 26 - k])
        text, ans = f"Which letter comes {k} letters after '{c}' in the alphabet?", chr(ord(c) + k)
    elif kind == "digitsum":
        n = rng.randint(10 ** 11, 10 ** 12 - 1) if hard else rng.randint(10000, 99999)
        text, ans = f"What is the sum of the digits of {n}?", str(sum(map(int, str(n))))
    elif kind == "add3":
        xs = [rng.randint(1000, 9999) for _ in range(4)] if hard else [rng.randint(10, 99) for _ in range(3)]
        text, ans = "What is " + " + ".join(map(str, xs)) + "?", str(sum(xs))
    else:
        raise ValueError(kind)
    return dict(id=tid, kind=kind, level=level, text=text, answer=ans)


NUMERIC = {"multiply", "count", "subtract", "remainder", "digitsum", "add3"}
OPS = r"([+\-−*/×%÷^]|x\b|mod\b|plus\b|minus\b|times\b|divided\b)"
WORDISH = {"reverse", "shift"}


def _clean(a):
    return (a or "").strip().strip(".'\"`*_ ")


def extract(task, ans):
    """The answer the agent gave, in canonical form, or None if none can be extracted."""
    a = _clean(ans)
    k = task["kind"]
    if k in NUMERIC:
        # Amendment H-A2: a reply that shows its working ("638 % 7 = 1", "1+7+...+3 = 66") is scored on the result,
        # not on the first number. A leading number counts only if no operator follows it; otherwise the number
        # after the first "=" counts; otherwise the first number.
        a = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", a)
        m = re.match(r"^\s*(-?\d+)(.*)$", a, re.S)
        if m and not re.match(OPS, m.group(2).lstrip(), re.I):
            return m.group(1)
        m = re.search(r"=\s*(-?\d+)", a) or re.search(r"-?\d+", a)
        return m.group(1) if m and m.re.groups else (m.group(0) if m else None)
    if k in WORDISH:
        m = re.search(r"[A-Za-z]+", a)
        return m.group(0).lower() if m else None
    if k == "weekday":
        m = re.search(r"\b(" + "|".join(DAYS) + r")\b", a, re.I)
        return m.group(1).capitalize() if m else None
    if k == "alphabet":
        m = re.match(r"^[\W_]*([A-Za-z])\b", a) or re.search(r"'([A-Za-z])'", a) or re.search(r"\b([A-Za-z])\b", a)
        return m.group(1).lower() if m else None
    if k == "sort":
        xs = re.findall(r"\d+", a)
        return ", ".join(xs) if xs else None
    raise ValueError(k)


def correct(task, ans):
    e = extract(task, ans)
    return e is not None and e == task["answer"]


def answerable(task, ans):
    return extract(task, ans) is not None


# ---------------------------------------------------------------- SELECTION RULE (frozen with the pre-registration)
SPREAD_MIN = 0.20          # gate: at least GATE_TYPES chosen types with best - worst model accuracy >= this
GATE_TYPES = 2
ACC_RANGE = (0.25, 0.95)   # a cell is eligible if its mean accuracy over models lies in this range ...
PARSE_MIN = 0.90           # ... and every model gave an extractable answer in >= 90% of its calls
N_TYPES = 4


def select(acc, parse):
    """acc[kind][level][model] and parse[kind][level][model] are proportions from the G0 battery.

    1. Eligible cells: mean accuracy over models in ACC_RANGE and parse >= PARSE_MIN for every model.
    2. For each kind, keep its eligible level with the largest spread (best minus worst model); ties -> level 2.
    3. Rank kinds by spread (ties by name). Take the top N_TYPES.
    4. Comparative advantage: if all chosen kinds share the same best model, and some unchosen kind with spread
       >= SPREAD_MIN has a different best model, it replaces the lowest-ranked chosen kind (the highest-spread such
       kind is used).
    5. Gate: N_TYPES kinds chosen and at least GATE_TYPES of them with spread >= SPREAD_MIN.
    """
    cands = []
    for kind in sorted(acc):
        best = None
        for level in (2, 1):
            a, p = acc[kind].get(level), parse[kind].get(level)
            if not a:
                continue
            vals = list(a.values())
            mean = sum(vals) / len(vals)
            if not (ACC_RANGE[0] <= mean <= ACC_RANGE[1]) or min(p.values()) < PARSE_MIN:
                continue
            spread = max(vals) - min(vals)
            top = max(sorted(a), key=lambda m: a[m])
            if best is None or spread > best["spread"] + 1e-12:
                best = dict(kind=kind, level=level, spread=round(spread, 4), mean=round(mean, 4), best_model=top)
        if best:
            cands.append(best)
    cands.sort(key=lambda c: (-c["spread"], c["kind"]))
    chosen, rest = cands[:N_TYPES], cands[N_TYPES:]
    swapped = None
    if len(chosen) == N_TYPES and len({c["best_model"] for c in chosen}) == 1:
        alt = [c for c in rest if c["spread"] >= SPREAD_MIN and c["best_model"] != chosen[0]["best_model"]]
        if alt:
            swapped = dict(out=chosen[-1]["kind"], into=alt[0]["kind"])
            chosen = chosen[:-1] + [alt[0]]
    passed = len(chosen) == N_TYPES and sum(c["spread"] >= SPREAD_MIN for c in chosen) >= GATE_TYPES
    return dict(chosen=chosen, candidates=cands, swapped=swapped, gate_passed=passed)


if __name__ == "__main__":         # self-check: every generator's answer passes its own checker
    rng = random.Random(1)
    for kind in KINDS:
        for level in (1, 2):
            for i in range(200):
                t = make_task(i, kind, level, rng)
                assert correct(t, t["answer"]), (t, extract(t, t["answer"]))
                assert correct(t, f"{t['answer']}."), t
            print(f"{kind:10s} L{level}: {make_task(0, kind, level, rng)['text']}  -> {t['answer']}")
    print("ok")
