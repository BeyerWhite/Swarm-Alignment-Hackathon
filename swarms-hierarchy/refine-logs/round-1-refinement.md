# Round 1 Refinement

## Problem Anchor
- **Bottom-line problem:** Predict, before deployment and from cheap two-agent measurements, whether a population of identical LLM agents will form a coordinator hierarchy (and how strong and persistent it will be), using the Bonabeau / Ben-Naim–Redner hierarchy models as the bridge from individual response to collective outcome (Track 2: "when do leaders and roles emerge, and can this be reproduced and predicted safely").
- **Must-solve bottleneck:** The pre-registered micro→macro pipeline gets the direction right (104–224 nats over a status-blind null) but is systematically miscalibrated: populations are more unequal than predicted in 22/27 cells (all 7 status-sensitive models), the threshold sweep and the primary-kernel N = 700 forecast under-predict order at high contact, and seeded-leader persistence is mis-predicted in both directions (overthrown faster under equal turns, held longer at w = 50). Until this is explained, the screen is only "directional and optimistic".
- **Non-goals:** new hierarchy theory; tuning kernels on population outcomes; replacing the pre-registered failure with a post-hoc success; agents with memory/tools; the real-task game (reported separately).
- **Constraints:** ~9 h to submission; no GPU; LLM calls via Adaption (credits ≈ 11 of a 50 cap used; ~6–35 calls/s depending on load); all prior predictions are frozen and must not be edited; post-hoc results must be labelled post-hoc; 8-page paper.
- **Success condition:** a specific, testable explanation of the miscalibration that (i) removes most of the one-sided bias on existing data without using population outcomes to fit anything, (ii) predicts the dose–response that a sampling artefact implies, and (iii) passes at least one new *prospective* test frozen before its population runs.

## Anchor Check
- Original bottleneck: one-sided micro→macro miscalibration of the two-agent screen.
- The revision still targets it: same simulator, protocol and scoring; only the estimator and calibration allocation change.
- Drift rejected: none of the reviewer's points change the problem. The reviewer's warning that "coordinator hierarchy" must mean coordinator selection in the contest protocol is accepted (task game kept separate).

## Simplicity Check
- Dominant contribution: empirical diagnosis that sparse-cell shrinkage contributes most of the one-sided miscalibration, shown by a matched-data decomposition plus a known-kernel simulation, and a targeted prospective pilot.
- Removed/merged: `flex_mle` demoted to a diagnostic (Beta(0.05, 0.05) is boundary-concentrated, not near-flat); NW kept only as a matched α = 1 vs α = 0 control on identical data; dose–response via subsampling dropped in favour of the known-kernel simulation, which identifies estimator bias directly.
- Rejected as unnecessary complexity: ordered-binomial posterior / hierarchical models (not feasible in time); instead isotonic intervals are labelled exploratory, and the prospective pilot uses a stated clipped-resampling rule.

## Changes Made
### 1. Attenuation argument corrected
- Reviewer: Jeffreys is an exact lapse relative to the empirical kernel, λ_n = 1/(2(n+1)); observed ≠ true saturation; Beta draws add no mean attenuation; macro direction not universal.
- Action: rewritten exactly so. Direction now predicted per sign of s: weakening deference lowers dispersion, weakening anti-deference raises it — verified by an artificial-lapse sweep (below). Estimator-specific statements replace "every estimator shrinks toward ½".
### 2. Matched decomposition (done, `results/posthoc_mechanism.json`, point estimates, 24 seeds per cell, 27 status-sensitive cells)
| estimator (identical data) | positive residuals | mean sd(h) residual |
|---|---|---|
| Jeffreys (grid) | 22/27 | +0.62 |
| raw rates (grid) | 16/27 | +0.26 |
| isotonic (grid) | 15/27 | +0.13 |
| NW α = 1 (grid + state) | 21/27 | +0.48 |
| NW α = 0 (grid + state) | 13/27 | +0.07 |
Removing the pull toward ½ on identical data removes most of the bias in both estimator families.
### 3. Known-kernel simulation (done)
True kernel = isotonic fit of a real cell; synthetic calibration with the real per-cell counts (median 4); 8 replicate calibrations. Bias of predicted vs true sd(h): **Jeffreys −0.78 to −1.02** in all four deference kernels (true sd 5.0–6.1), **raw −0.04 to 0.00, isotonic −0.04 to +0.23**; in two anti-deference kernels Jeffreys +0.07, raw/isotonic ≈ 0. This identifies estimator bias independent of any population effect.
### 4. Artificial-lapse sweep (done)
λ from 0 to 0.15: Gemma-31B norm100 sd(h) 5.83 → 4.50 (monotone down); Haiku norm0 0.91 → 1.07 (monotone up). Matches the two-sided seeded-leader errors.
### 5. Prospective pilot D (frozen and running, `code/pilotD.py`)
- Conditions Gemma-26B norm100, Qwen-397B norm0. Anchors: 10 most visited positive displayed Δ (0.2 bins, mean slice 0 — populations stay at mean 0) plus mirrors = 20 per condition, 30 calls each (frozen `results/pilotD_anchors.json`).
- 2 × 2 kernels {Jeffreys, isotonic} × {old grid, old + new}, all frozen before populations; isotonic uncertainty = Binomial(n, clip(p, λ_n, 1 − λ_n)) resampling then refit (exploratory).
- Primary sd(h) of the 5-seed mean; success = LPD(iso_new) > LPD(jeffreys_new) and |obs − pred| ≤ 0.5; power = P(candidate wins | candidate true) frozen with predictions. Labelled a targeted pilot.
### 6. Claims narrowed
Title/thesis no longer say "unbiased" or "the cause"; seeded-leader neutral condition reported as not improved; threshold/N = 700/X1 regimes stated as not re-analysed (N = 700: the unshrunk Bonabeau logistic matched, flexible did not — consistent but not a test).

## Revised Proposal
**Title:** Sparse-calibration attenuation explains most of the micro→macro miscalibration of two-agent hierarchy screens.
**Thesis:** Shrinking sparse, often-saturated two-agent cells toward ½ acts as a lapse relative to the empirical kernel; through 40 rounds of reinforcement this biases predicted population dispersion down for deferential agents and up for turn-taking agents. Unshrunk, monotone estimation removes most of the bias on existing data and in known-kernel simulation; a frozen visited-state pilot tests it prospectively.
**Method:** isotonic per-mean-slice estimator (direction from calibration-only s), unchanged simulator/scoring; matched decomposition; known-kernel simulation; lapse sweep; prospective pilot D as above.
**Validation:** Claim 1 (retrospective + mechanism): matched decomposition and known-kernel bias. Claim 2 (prospective pilot): D's frozen rule. Paper statement: pre-registered criterion failed; post-hoc diagnosis with simulation evidence; pilot outcome reported whatever it is.
