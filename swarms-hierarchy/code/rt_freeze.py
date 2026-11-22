"""Hash stamps for RT amendments and follow-up pre-registrations.
  rt_freeze.py make <freeze.json> <file> ...   write once: sha256 of each file and of the listing
  rt_freeze.py check <freeze.json>             exit 0 iff the stamp exists and every file still has its hash"""
import hashlib, json, sys, time
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cmd, out = sys.argv[1], Path(sys.argv[2])
    if cmd == "make":
        files = {f: sha(f) for f in sys.argv[3:]}
        rec = dict(created=time.strftime("%F %T"), files=files,
                   sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "x") as f:
            json.dump(rec, f, indent=1)
        print(rec["sha256"])
    else:
        if not out.exists():
            print(f"missing {out}"); sys.exit(1)
        bad = [f for f, h in json.load(open(out))["files"].items() if not Path(f).exists() or sha(f) != h]
        if bad:
            print("changed since the freeze:", bad); sys.exit(1)
        print(f"freeze ok: {out}")


if __name__ == "__main__":
    main()
