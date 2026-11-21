"""Hash stamp of the Couzin pre-registration and code (write-once), and a check against it.
Usage: couzin_freeze.py stamp | check"""
import hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FILES = ["refine-logs/COUZIN_PREREGISTRATION.md", "refine-logs/COUZIN_PROPOSAL.md", "code/couzin.py",
         "code/couzin_kernel.py", "code/couzin_forecast.py", "code/couzin_eval.py", "code/couzin_budget.py",
         "code/couzin_offline_check.py", "code/couzin_queue1.sh", "code/couzin_queue2.sh", "code/couzin_common.sh", "code/couzin_run.py", "code/couzin_launch.sh", "code/couzin_freeze.py",
         "code/llm.py", "code/chase_paced.py"]
OUT = ROOT / "results" / "couzin_prereg_freeze.json"
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
if sys.argv[1] == "stamp":
    if OUT.exists():
        sys.exit(f"{OUT} exists (write-once)")
    OUT.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "sha256": {f: sha(f) for f in FILES}}, indent=1))
    print(OUT.read_text())
else:
    d = json.loads(OUT.read_text())["sha256"]
    bad = [f for f, h in d.items() if sha(f) != h]
    print("frozen files unchanged" if not bad else f"CHANGED: {bad}")
    sys.exit(1 if bad else 0)
