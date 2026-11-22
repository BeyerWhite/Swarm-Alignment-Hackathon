# Pre-registration: RT-R — response thresholds with a framing that makes resting attractive

Written 2026-11-22, before any API call of this experiment, at the owner's request ("run another experiment on Qwen
and Gemma with a framing that makes resting attractive"). Everything not stated here is as in
`RT_PREREGISTRATION.md` (environment, stages, metrics, tests, seeds). This file and the code it names are
hash-stamped in `results/rest/rt_prereg_freeze.json`; `code/rt_rest_queue.sh` refuses to start if any stamped file
changes. Changes go in `RT_LOG.md` as amendments RT-R1, ...

## 1. Why

In the first run (`RT_REPORT.md`) both models took an open task 97–100% of the time at every effort cost, so the
work-or-rest half of the threshold model could not be measured: there was no uptake threshold, only type choice.
The effort cost made working cheaper or dearer but left resting worth nothing. Here resting earns points directly.

## 2. The rest-reward rule (`RT_FRAME=rest`; results in `results/rest/`)

Shown to the agents (r is the rest reward):
> - A correct task earns the team 1 point. A wrong answer earns nothing and leaves the task open.
> - Resting earns the team r points (an agent who rests does routine upkeep instead).
> - At the end of every round, each task still waiting costs the team 1 point.

No effort cost. Team-optimal benchmark unchanged: work iff the open tasks are at least the agents still to move
(you included), while rounds remain; below that, working and resting give the team the same total, because some
agent must do each task and whoever does it gives up the same r. So resting at a low backlog is again never a
mistake, and now it is visibly rewarded. At r = 2 a task done now costs 1 point of upkeep, but leaving it to wait
costs 1 per round and it must still be done, so the benchmark still says work when needed.

## 3. Choice of r (RT1, write-once `results/rest/rt_c_choice.json`)

Calibration as in the first run with r ∈ {0.5, 1, 2} in place of c: backlog {1, 1+1, 2+2, 4+4} × position {1, 8}
× 20 calls per model, balanced own history. Over the models with comprehension ≥ 0.6, the pooled DO share in the cell
backlog 2+2, position 1; choose the r closest to 0.5 (ties: smaller r). If all three shares exceed 0.9 the run is
"saturated" again (r = 2), reported as such, and type choice is primary as before.

Comprehension: the 10 items of the first run with three changed to fit the rule (task earns 1; rest earns r, asked
at r = 2; wrong answer earns 0), 20 calls each.

## 4. Stages, models, seeds

Gemma-4-31B and Qwen3.8-27B; RT1 (comprehension, calibration, r rule, pilots seeds 7–8, gate), RT2 (grid, replay,
social, top-up), forecast freeze (B = 500; full, history-blind, position, flex), RT3 (80 populations per model,
seeds 51001–51008, the same task streams as the first run), RT4 by the registered τ rule, evaluation — all exactly
as registered for the first run, in a separate results directory.

## 5. Hypotheses

As in the first run (H1–H8; Holm over H2, H3, H6a), plus one confirmatory test for the question this run exists for:

- **H1r — an uptake threshold exists.** The stimulus exponent n of the full kernel is above 0: one-sided bootstrap
  p = share of the 500 frozen refits with n ≤ 0. Holm over H1r, H2, H3, H6a within each model.
  Descriptive: the fitted threshold θ against the benchmark step (open tasks = agents still to move), and n against
  the classical n = 2.

## 6. Reading

- **Threshold behaviour appears:** H1r holds (uptake rises with backlog). Then H2/H3 say whether reinforcement
  builds specialists on top of it, as in Theraulaz et al. (1998).
- **Still saturated:** no rest framing tried here moves these models off "always work"; reported with the DO shares.
- **Over-resting:** if agents rest even when tasks must wait (uptake below the benchmark at high backlog), points
  fall below the always-work benchmark; reported against `policy_rational`.
- **Comparison with the first run** (same seeds and task streams): specialization, work rate, waiting and points per
  arm, descriptive.

## 7. Offline validation and budget

The queue was run end to end with the scripted agents (`DRY=1 RT_FAKE=1`, scratch directory) before freezing; see
`RT_LOG.md`. About 48,800 calls (as the first run), ≈ 17 hours, ≈ 2 credits; study guard 4 credits; ledger guard
cap − 1 (cap 200, owner). Runs after the GLM follow-up (RT-A1), behind every queue already waiting on the account.
