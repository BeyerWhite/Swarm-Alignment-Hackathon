"""Launcher that paces Adaption requests below the account's per-user rate limit, without touching frozen files.

Measured on 2026-11-12: this account is limited to about 50 inference requests per minute
("Inference rate limit reached for this user; retry after 60 seconds"). Unpaced runs spend most of their time in
429 back-off. This wrapper spaces HTTP requests evenly (default 48 per minute) and then runs a stage's main().

Usage: chase_paced.py <module> [module args...]      e.g.  chase_paced.py chase_e1 --models ... --workers 6
Env:   ADAPTION_RPM (default 48)
"""
import importlib, os, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import llm
from adaption import Adaption

GAP = 60.0 / float(os.environ.get("ADAPTION_RPM", "48"))
_lock, _next = threading.Lock(), [0.0]


def _slot():
    with _lock:
        now = time.time()
        t = max(now, _next[0])
        _next[0] = t + GAP
    if t > now:
        time.sleep(t - now)


llm._adaption = Adaption(max_retries=0)
_create = llm._adaption.responses.create


def paced_create(*a, **k):
    _slot()
    return _create(*a, **k)


llm._adaption.responses.create = paced_create

if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1])
    sys.argv = [sys.argv[1] + ".py"] + sys.argv[2:]
    mod.main()
