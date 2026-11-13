"""Launcher for the CLI backends (`claude:<model>`, `gpt:<model>`), without touching frozen files.

Checked on 2026-11-13 with Claude Code 2.1.323: under the flags in llm.py a `claude -p` call still carries 14 tool
definitions that the old disallow list does not name (SendMessage, ListAgents, CronCreate, ...), about 4,600 tokens.
Launched from inside the desktop app it also inherits the app's CLAUDE_CODE_* variables and carries about 21,000.
The agents of this study are text-only, so this wrapper
  1. removes the desktop app's variables from the environment of every child process;
  2. adds `--tools ""` (no tool definitions) and `--no-session-persistence` to every `claude` call, and
     `--ephemeral` (no session file) to every `codex exec` call;
  3. stops the run after STOP_AFTER consecutive failed calls (sign-in lost, plan limit reached), so a dead backend
     cannot write thousands of empty rows. Unparsed rows are retried on restart by the stage runners.
What it cannot remove with a subscription sign-in: the CLI prefixes the system prompt with one sentence ("You are a
Claude agent, built on Anthropic's Claude Agent SDK.") and adds a reminder block holding the account's e-mail
address. Both are constant across cells and arms.

Usage: chase_cli.py <module> [module args...]     e.g.  chase_cli.py chase_e1 --models claude:haiku --workers 4
Env:   CLI_STOP_AFTER (default 12)
"""
import importlib, os, subprocess, sys, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

for k in list(os.environ):
    if k.startswith(("CLAUDE", "ANTHROPIC_")) or k in ("AI_AGENT", "BAGGAGE", "__CFBundleIdentifier"):
        os.environ.pop(k)

import llm

STOP_AFTER = int(os.environ.get("CLI_STOP_AFTER", "12"))
BAD = ("failed to authenticate", "not logged in", "please run /login", "invalid api key")


class _Sub:
    """Stand-in for the `subprocess` name inside llm: same behaviour, plus the extra claude flags."""
    TimeoutExpired = subprocess.TimeoutExpired

    @staticmethod
    def run(argv, **k):
        if argv and argv[0] == "claude":
            argv = list(argv) + ["--tools", "", "--no-session-persistence"]
        elif argv and argv[:2] == ["codex", "exec"]:
            argv = argv[:2] + ["--ephemeral"] + list(argv[2:])        # no session file per call; same context
            k.setdefault("stdin", subprocess.DEVNULL)                 # codex waits on an open non-terminal stdin
        return subprocess.run(argv, **k)


llm.subprocess = _Sub
_inner, _lock, _fails = llm._call, threading.Lock(), [0]


def guarded(model, *a, **k):
    out = _inner(model, *a, **k)
    if model.split(":", 1)[0] in ("claude", "gpt"):
        bad = (not out) or any(b in out.lower() for b in BAD)
        with _lock:
            _fails[0] = _fails[0] + 1 if bad else 0
            if _fails[0] >= STOP_AFTER:
                print(f"STOPPED: {STOP_AFTER} consecutive failed calls on {model}; last reply: {out[:160]!r}", flush=True)
                os._exit(3)
        if bad:
            return ""
    return out


llm._call = guarded

if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1])
    sys.argv = [sys.argv[1] + ".py"] + sys.argv[2:]
    mod.main()
