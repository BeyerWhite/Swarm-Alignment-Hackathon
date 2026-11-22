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

## 2026-11-21 / 22

**Queue ran** 21 Nov 05:10 → 22:05 (stage times in `scratch/rt_queue.log`): RT1, c = 0.75 by the saturation rule
(DO share 1.00, 1.00, 0.97), both models gated in, RT2, forecast frozen 09:34 (B = 500, RT4 model Qwen3.8-27B), RT3,
RT4, evaluation. Ledger 97.6 → 99.4. No amendment.

**2026-11-22 — results written up** in `RT_REPORT.md`. Post hoc observation: all of Qwen's wrong answers are on the
"next number" task kind (30% wrong there), which the forecast's single accuracy rate did not capture.

## 2026-11-22 — follow-ups (owner: "Yes, queue GLM"; then "run another experiment on Qwen and Gemma with a framing that makes resting attractive. And push everything.")

### Amendment RT-A1 — GLM-5.2 on the frozen design (written before any GLM call of this study)

1. GLM-5.2 runs the registered design unchanged: effort frame, c = 0.75 (already chosen; its own calibration is run
   and reported but does not change c), comprehension, pilots (seeds 7, 8), gate, RT2 (grid, replay, social,
   top-up), forecast, RT3 (80 populations, seeds 51001–51008), evaluation. RT4 is not run for GLM (cost).
2. Its forecast is frozen separately, write-once, in `results/rt_predictions_glm.json` (env `RT_FORECAST`); the
   first forecast file is untouched. Evaluation to `results/rt_eval_glm.json`, same tests and Holm family.
3. Code changes, none of which alters the Gemma/Qwen run: `rt.FORECAST` can be overridden by `RT_FORECAST`;
   `rt_kernel gate` adds a newly gated model to the existing gate list instead of replacing it; a GLM test double in
   `rt.FAKE_PARAMS`; queue `code/rt_glm_queue.sh` with a 20-credit study guard. Because rt.py, rt_kernel.py and
   rt_eval.py changed (RT-R also added the rest frame and H1r, both inactive in the effort frame), their hashes no
   longer match `rt_prereg_freeze.json`; the amended files are stamped in `results/rt_A1_freeze.json`, which the GLM
   queue checks before starting.
4. Offline: the GLM queue was run end to end on a copy of the first run's data with a scripted GLM (τ = 0.9): τ
   recovered 0.905 [0.872, 0.945], H2/H3/H6a pass after Holm, full forecast coverage 67 of 78; the merged gate kept
   Gemma and Qwen. Estimated 22,400 calls, ≈ 12–16 credits.

### RT-R — rest-reward framing (new pre-registration `RT_REST_PREREGISTRATION.md`)

Resting earns the team r ∈ {0.5, 1, 2} points (chosen by the same calibration rule), no effort cost; Gemma-4-31B and
Qwen3.8-27B; results in `results/rest/`; H1r (n > 0) added to the Holm family. Queue `code/rt_rest_queue.sh`,
study guard 4 credits.

### Queue order

Both run under `code/rt_phase2_launch.sh` (name chosen to match no study's queue pattern), which waits for Chase E14
(29491), the tournament queue 1 wrapper (29645), hierarchy G0 (29071) and Couzin queue 1 (43529), then until no
other study's stage script or paced job runs, then runs GLM and then RT-R. Cap argument 200 (owner, 21 Nov).

**Frozen 2026-11-22.** RT-A1: `results/rt_A1_freeze.json` stamp 4f0d6ce1a28bbaf349e4f3e18a5acd3211af33efad32bdb353156a7be3bb77f7. RT-R: `results/rest/rt_prereg_freeze.json` stamp ef0706288db60f0c7c566e22a6c1cdc615b46d8fb3c68d518f39554eb33709fd.
