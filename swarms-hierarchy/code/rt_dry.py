"""Dry-run stand-in for chase_paced.py (no Adaption client, no pacing): rt_dry.py <module> [args...].
Only for queue tests with RT_FAKE=1 (scripted agents); refuses otherwise."""
import importlib, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
if os.environ.get("RT_FAKE") != "1":
    sys.exit("rt_dry.py runs only with RT_FAKE=1")
mod = importlib.import_module(sys.argv[1])
sys.argv = [sys.argv[1] + ".py"] + sys.argv[2:]
mod.main()
