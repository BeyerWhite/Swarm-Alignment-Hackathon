"""Sandboxed LLM agent backend: every agent call is a single text-in/text-out completion
with no tools, no network access and no code execution.

Backends
  claude:<haiku|sonnet|opus>  -> `claude -p` headless, custom system prompt, all tools disallowed
  gpt:<model>                 -> `codex exec` under CODEX_HOME=~/.codex-api, read-only sandbox
  ollama:<model>              -> local Ollama HTTP API (temperature 1, thinking off, no tools)
  adaption:<model>            -> Adaption hosted open-weight models (reasoning off, 16 output tokens, hard credit cap)
"""
import hashlib, json, os, re, subprocess, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "calls.jsonl"
GPT_BUDGET = int(os.environ.get("GPT_CALL_BUDGET", "1500"))   # protect the reviewer quota
_lock = threading.Lock()
_gpt_used = 0
ADAPTION_CAP = float(os.environ.get("ADAPTION_CREDIT_CAP", "50"))       # hard project cap set by Sara: 50 of 1,500 credits
SPEND = ROOT / "results" / "adaption_spend.jsonl"            # per-call ledger (append-only)
SPEND_TOTAL = ROOT / "results" / "adaption_spend_total.json" # locked running total shared by all processes
RESERVE = 0.002            # credits reserved per in-flight call (conservative; real calls cost ~1e-5..5e-4)
MISSING_COST = 0.001       # charged when the API returns no cost (fail closed)
_adaption = None


def adaption_spent():
    if not SPEND.exists():
        return 0.0
    return sum(json.loads(l)["cost"] for l in open(SPEND))


def _ledger(delta, check_cap):
    """Atomically add `delta` to the shared running total (flock); refuse if it would exceed the cap."""
    import fcntl
    SPEND_TOTAL.parent.mkdir(exist_ok=True)
    with open(SPEND_TOTAL, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0); txt = f.read()
        total = json.loads(txt)["total"] if txt.strip() else adaption_spent()
        if check_cap and total + delta > ADAPTION_CAP:
            raise RuntimeError(f"Adaption credit cap reached ({total:.4f} + {delta} > {ADAPTION_CAP})")
        total += delta
        f.seek(0); f.truncate(); f.write(json.dumps({"total": total})); f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return total


def _adaption_call(name, system, prompt, timeout, max_tries=10, max_tokens=16):
    """One Adaption inference with 429 back-off (~60 s + jitter), transient-error retries and cap reservation."""
    global _adaption
    import random as _r
    with _lock:
        if _adaption is None:
            from adaption import Adaption
            _adaption = Adaption(max_retries=0)       # we handle retries; reads ADAPTION_API_KEY from env
    for attempt in range(max_tries):
        _ledger(RESERVE, check_cap=True)              # reserve before the request (fails closed at the cap)
        try:
            r = _adaption.responses.create(model=name, instructions=system, input=prompt, temperature=1.0,
                                           max_output_tokens=max_tokens, store=False,
                                           extra_body={"reasoning": {"effort": "none"}}, timeout=timeout)
        except Exception as e:
            _ledger(-RESERVE, check_cap=False)        # nothing was charged for a failed request
            status = getattr(e, "status_code", None)
            if status == 429 or "rate limit" in str(e).lower():
                time.sleep(60 + _r.uniform(0, 15))
            else:
                time.sleep(min(60, 2 ** attempt) + _r.uniform(0, 1))
            continue
        u = r.usage.model_dump() if getattr(r, "usage", None) is not None and hasattr(r.usage, "model_dump") else {}
        cost = u.get("cost")
        cost = float(cost) if cost is not None else MISSING_COST
        _ledger(cost - RESERVE, check_cap=False)
        with _lock, open(SPEND, "a") as f:
            f.write(json.dumps({"t": time.time(), "model": name, "cost": cost,
                                "cost_missing": u.get("cost") is None}) + "\n")
        return (r.output_text or "").strip()
    return ""                                          # exhausted retries -> logged as empty, counted invalid


_conc = threading.BoundedSemaphore(int(os.environ.get("LLM_MAX_CONC", "24")))
DISALLOWED = "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Agent,NotebookEdit,TodoWrite,Skill"


def _is_quota_msg(s):
    s = s.lower()
    return ("usage limit" in s or "session limit" in s or "rate limit" in s or "hit your" in s
            or "please run /login" in s or "not logged in" in s)


def _log(rec):
    LOG.parent.mkdir(exist_ok=True)
    with _lock, open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def call(model, system, prompt, tag="", timeout=180, retries=2, max_tokens=16):
    with _conc:
        return _call(model, system, prompt, tag, timeout, retries, max_tokens)


def _call(model, system, prompt, tag, timeout, retries, max_tokens=16):
    global _gpt_used
    fam, name = model.split(":", 1)
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            if fam == "claude":
                out = subprocess.run(
                    ["claude", "-p", "--model", name, "--system-prompt", system,
                     "--disallowedTools", DISALLOWED],
                    input=prompt, capture_output=True, text=True, timeout=timeout,
                    cwd=tempfile.gettempdir()).stdout.strip()
            elif fam == "gpt":
                with _lock:
                    if _gpt_used >= GPT_BUDGET:
                        raise RuntimeError("GPT call budget exhausted")
                    _gpt_used += 1
                with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False) as tf:
                    path = tf.name
                env = dict(os.environ, CODEX_HOME=str(Path.home() / ".codex-api"))
                subprocess.run(
                    ["codex", "exec", "--skip-git-repo-check", "-m", name,
                     "-c", "model_reasoning_effort=low", "-s", "read-only", "-o", path,
                     f"[SYSTEM]\n{system}\n[/SYSTEM]\n\n{prompt}"],
                    capture_output=True, text=True, timeout=timeout, env=env,
                    cwd=tempfile.gettempdir())
                out = Path(path).read_text().strip() if Path(path).exists() else ""
                Path(path).unlink(missing_ok=True)
            elif fam == "adaption":
                out = _adaption_call(name, system, prompt, timeout, max_tokens=max_tokens)
            elif fam == "ollama":
                import urllib.request
                body = json.dumps({"model": name, "stream": False, "think": False,
                                   "options": {"temperature": 1.0, "num_predict": 40},
                                   "messages": [{"role": "system", "content": system},
                                                {"role": "user", "content": prompt}]}).encode()
                req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    out = json.loads(resp.read())["message"]["content"].strip()
            else:
                raise ValueError(model)
        except subprocess.TimeoutExpired:
            out = ""
        dt = time.time() - t0
        if out and not _is_quota_msg(out) and not out.startswith("Error"):
            break
        time.sleep(2 * (attempt + 1))
    _log({"t": time.time(), "model": model, "tag": tag, "latency": round(dt, 2),
          "phash": hashlib.sha1((system + prompt).encode()).hexdigest()[:12],
          "system": system, "prompt": prompt, "response": "" if _is_quota_msg(out) else out,
          **({"quota_error": out[:120]} if _is_quota_msg(out) else {})})
    return "" if _is_quota_msg(out) else out


def parse_choice(text, options=("LEAD", "DEFER")):
    """Strict decision parser: a unique standalone decision line, else a leading decision word; otherwise
    None (ambiguous / missing). Never guesses from words embedded in reasoning."""
    if not text:
        return None
    lines = [l.strip().strip("*_`.\"'!:#> ").upper() for l in text.splitlines() if l.strip()]
    standalone = {l for l in lines if l in options}
    if len(standalone) == 1:
        return standalone.pop()
    if len(standalone) > 1:
        return None
    first = text.strip().splitlines()[0].upper()
    named = {o for o in options if re.search(rf"\b{o}\b", first)}
    m = re.match(r"^[\W_]*(" + "|".join(options) + r")\b", first)
    return m.group(1) if m and len(named) == 1 else None      # first line must name exactly one option


def batch(jobs, workers=8):
    """jobs: list of dicts with model/system/prompt/tag; returns responses in order."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: call(j["model"], j["system"], j["prompt"], j.get("tag", "")), jobs))
