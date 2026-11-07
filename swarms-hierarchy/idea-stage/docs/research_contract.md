# Research Contract: Framing sets the sign of status reinforcement in LLM agent populations

> A focused working document for the active idea. On session recovery, read this file first, then
> `refine-logs/EXPERIMENT_PLAN.md` and `refine-logs/EXPERIMENT_TRACKER.md`.

## Selected Idea
- **Description**:
  - Identical LLM agents are repeatedly paired. Each answers LEAD or DEFER for a coordinator role.
  - A public status score is injected into context: +1 to the coordinator, −1 to the other, 10% decay per round.
  - We measure the pairwise decision kernel p(LEAD | displayed own score, displayed partner score) and map it onto
    the Bonabeau model's status sensitivity η.
  - We then predict population-level hierarchy (dispersion, frozen order) under the identical protocol, and test the
    prediction against LLM-in-the-loop populations.
  - Framing manipulations (a numeric norm weight w; a wording factorial) move η through zero.
- **Source**: `idea-stage/IDEA_REPORT.md`, ideas #1–3 (bundle).
- **Selection rationale**:
  - Positive pilot: sign flip of η, and populations track surrogates.
  - Novelty: PROCEED WITH CAUTION. Review: 8/10, PROCEED. Both cross-checked by gpt-6-astra.
  - Feasible within the call budget.

## Core Claims
1. **Framing reverses status-dependent coordinator choice.**
   - Shown so far: Haiku (pilot), and Sonnet norm0 vs norm100 (η −0.19 → +2.2).
   - The wording-only test (`ended` vs `ended+comp`) is pending for Haiku.
2. **Independently calibrated pairwise kernels predict population hierarchy persistence *or* suppression.**
   - No population-level refitting.
   - Primary metrics: sd_h and q, final 20 rounds.
3. **Scope:**
   - finite-N crossover, not a sharp transition;
   - coordinator selection under imposed scores, not competence or division of labour;
   - results are per model.

## Method Summary
**Contest.** A lone LEAD wins; otherwise a coin flip. So P_win(Δ) = ½ + [p(Δ) − p(−Δ)]/2. With a symmetric logistic p,
this is exactly Bonabeau's σ(ηΔ). Both contestants update, so the mean-field threshold is ½ × the Lacasa–Luque ρ_c.

**Kernels**, fitted on calibration data:
- flexible: Jeffreys-smoothed bilinear interpolation over the (Δ, mean) grid;
- bonabeau: σ(a + bΔ), with implied η = 4P′_win(0);
- logistic3: adds a mean-status term;
- null: status-blind.

**Predictions.** Each kernel's predictive distribution is simulated with the exact `population2.py` protocol:
synchronous rounds, rounded display, separate random streams. Results go to a write-once, hash-stamped
`results/predictions_v2*.json` before any population run.

## Experiment Design
- **Data**: generated LLM decisions only, all logged in `results/calls.jsonl`. Calibration is in `results/calib_v2.jsonl`.
- **Baselines**: the status-blind null kernel (matched base LEAD rate); the strict Bonabeau kernel vs the flexible kernel.
- **Metrics**:
  - primary: sd_h and q (final 20 rounds);
  - kernel comparison: log predictive density, with a pre-declared 3-nat threshold;
  - also: 90% interval coverage and the conditional winner log-score.
- **Key parameters**: N=8, ρ=0.5, μ=0.1, F=Q=1, 40 rounds.
- **Compute budget**: subscription CLIs, about 2 calls/s across Claude. GPT-6-luna capped at 1,500 calls in total.

## Current Results
| Model / condition | η (Bonabeau kernel, 95% CI) | local η (nonparametric) | Source |
|---|---|---|---|
| haiku reward (pilot v1) | −0.27 [−0.37, −0.18] | — | pairwise_pilot |
| haiku competence (pilot v1) | 1.14 [0.81, 1.81] | — | pairwise_pilot |
| sonnet norm0 / norm50 / norm100 (v2) | −0.19 / 0.00 / 2.22 | −0.67 / 1.31 / 3.50 | calib_v2 |
| sonnet ended / ended+comp (v2) | 0.01 / 0.23 | 0.31 / 1.21 | calib_v2 |
| luna norm0 / norm100 / ended+comp (v2) | 0.02 / 0.43 / 0.30 | −0.17 / 1.67 / 1.54 | calib_v2 |

Population v2: pending.

## Key Decisions
- Sharp transitions abandoned in favour of predictive comparison with surrogates under an identical protocol.
- Kernels are fitted on *displayed* (rounded) scores.
- Sonnet population conditions {norm0, ended+comp, norm100} × 2 seeds were chosen before any v2 population data
  existed.

## Status
- [x] Idea selected
- [x] Baseline (null and surrogate) implemented
- [x] Main method implemented
- [ ] Haiku calibration v2 complete → predictions frozen → holdout → populations
- [ ] Evaluation
- [ ] Paper draft (Sunday)
