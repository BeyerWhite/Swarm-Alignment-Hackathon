# Response-threshold study — lab log

## 2026-11-20

**Owner decisions.** Study approved for Gemma-4-31B and Qwen3.8-27B only (GLM-5.2, Claude Haiku etc. later if the
results are interesting); run before Chase E14; effort-cost rule to be briefed before any call uses it. Briefed and
approved the same day ("Approve the rule"): correct task +1 − c effort, wrong answer −c, rest 0, −1 per waiting task
per round; c chosen from {0.25, 0.5, 0.75} by the calibration rule.

**Offline build and validation (no calls).** Engine `rt.py`, batteries `rt_kernel.py`, forecast `rt_forecast.py`,
evaluation `rt_eval.py`, queue `rt_queue.sh`. Full queue run with scripted agents (`RT_FAKE=1`, scratch results
directory): see pre-registration section 7. One change came out of the offline test, before anything was frozen: the
proposal's practice (ξ) and disuse (φ) terms are not separately identified with a full 8-turn window (only their sum
is), so the kernel uses κ (own turns on j), ω (own turns on the other type) and ρ (history length), and H2 tests
τ = κ + ω. The queue's call path (`chase_paced.py` → module) was dry-run with the scripted agents and a zero credit cap.

**Pre-registration frozen.** `results/rt_prereg_freeze.json`, stamp 724e0787bf4a5d227d1b00a0a4f94e04be3fbbd5d0e1bae391ee5764ebafcc8c.

**2026-11-20 13:57 — queue launched.** `nohup caffeinate -i bash code/rt_queue.sh 150 1644` (pid 28478, log
`scratch/rt_queue.log`), waiting for the Rosen phase 2 queue (pid 1644, at R6 then R5, R4, R3). Chase E14 starts only
after this queue exits (owner's order). A reboot kills the queue; it is restart-safe (`bash code/rt_queue.sh 150`).

**2026-11-20 13:58 — queue relaunched behind the tournament study's queue 1 (no call had been made).** The tournament
study's `tour_queue1.sh` (pid 28072, ~1,600 calls) was already waiting for the same Rosen pid, so both would have
started together and shared the 50-per-minute limit. The RT queue (pid 28478) was stopped while still waiting and
relaunched under a wrapper (pid 28523, log `scratch/rt_queue.log`) that waits until both pid 1644 and pid 28072 have
exited and then runs the frozen `code/rt_queue.sh 150` unchanged. Order: Rosen phase 2 → tournament queue 1 → RT →
Chase E14; the tournament study's larger queue 2 must also wait for RT.
