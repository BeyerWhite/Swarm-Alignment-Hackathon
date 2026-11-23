"""Amendment H-A2: re-score the G0 battery with the corrected numeric extraction (hier_tasks.extract), from the full
responses logged in results/calls.jsonl. No API calls.

Each battery row's prompt is rebuilt exactly as hier_measure.battery_one built it (seeded by the row key), the
logged call with the same model, system prompt and prompt is found, and the full response is re-scored. The old
battery file is kept as hier_g0_battery_v0_firstint.jsonl and the old selection and gate as *_v0_superseded.json.
Then the frozen selection rule and the gate are re-applied unchanged (hier_measure select / gate).
  hier_rescore.py
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import hier_tasks as HT
import hier_measure as HM
import office as O

R = O.R


def main():
    old_bat = R / "hier_g0_battery.jsonl"
    v0 = R / "hier_g0_battery_v0_firstint.jsonl"
    if v0.exists():
        sys.exit(f"{v0.name} exists: re-scoring was already done")
    calls = {}
    for l in open(R / "calls.jsonl"):
        if '"hier:g0bat"' not in l:
            continue
        c = json.loads(l)
        calls[(c["model"], c["system"], c["prompt"])] = c["response"]
    rows, out, missing, changed = [json.loads(l) for l in open(old_bat)], [], 0, 0
    for r in rows:
        rng = random.Random(r["key"])
        st, ids = HM.base_state(rng, ["tasks"], arm="noplan", record="lean")
        for a in ids:
            st["stats"][a]["done"] = rng.randint(2, 8)
        tasks = HM.open_tasks(rng, [(r["kind"], r["level"])], per=3)
        st["shown"] = st["open_tasks"] = tasks
        resp = calls.get((r["model"], O.system_for(st), O.build_prompt(st)))
        if resp is None:
            missing += 1
            out.append(dict(r, rescored=False)); continue
        act = O.parse(resp, set(ids), False)
        by = {x["id"]: x for x in tasks}
        ok = act[0] == "do" and act[1] in by and HT.correct(by[act[1]], act[2])
        ans = act[0] == "do" and act[1] in by and HT.answerable(by[act[1]], act[2])
        changed += (bool(ok) != r["ok"])
        out.append(dict(r, ok=bool(ok), parsed=bool(ans), ok_v0=r["ok"], rescored=True))
    print(f"{len(rows)} rows, {missing} without a logged call, {changed} changed score")
    if missing:
        sys.exit("some rows could not be matched to a logged call; nothing written")
    old_bat.rename(v0)
    with open(old_bat, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    for name in ("hier_g0_selection.json", "hier_g0_gate.json"):
        p = R / name
        if p.exists():
            p.rename(R / name.replace(".json", "_v0_superseded.json"))


if __name__ == "__main__":
    main()
