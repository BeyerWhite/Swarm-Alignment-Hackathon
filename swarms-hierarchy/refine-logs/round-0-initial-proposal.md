# Research Proposal: Unbiased Two-Agent Screening of Coordinator Hierarchies in LLM Swarms

*Input critique: `/Users/saraansari/Downloads/POSTHOC kernel attenuation.md` (copied to `refine-logs/attenuation/critique.md`). Project: `/Users/saraansari/Claude/Outputs/Swarms` (canonical; GitHub clones in `scratch/repo`, `../Ideas/repo`, `../Clairfying_questions_swarm/...`). Written 2026-11-08 18:35 PST; submission deadline 2026-11-09 03:59 PST.*

## Problem Anchor
- **Bottom-line problem:** Predict, before deployment and from cheap two-agent measurements, whether a population of identical LLM agents will form a coordinator hierarchy (and how strong and persistent it will be), using the Bonabeau / Ben-Naim–Redner hierarchy models as the bridge from individual response to collective outcome (Track 2: "when do leaders and roles emerge, and can this be reproduced and predicted safely").
- **Must-solve bottleneck:** The pre-registered micro→macro pipeline gets the direction right (104–224 nats over a status-blind null) but is systematically miscalibrated: populations are more unequal than predicted in 22/27 cells (all 7 status-sensitive models), the threshold sweep and the primary-kernel N = 700 forecast under-predict order at high contact, and seeded-leader persistence is mis-predicted in both directions (overthrown faster under equal turns, held longer at w = 50). Until this is explained, the screen is only "directional and optimistic".
- **Non-goals:** new hierarchy theory; tuning kernels on population outcomes; replacing the pre-registered failure with a post-hoc success; agents with memory/tools; the real-task game (reported separately).
- **Constraints:** ~9 h to submission; no GPU; LLM calls via Adaption (credits ≈ 11 of a 50 cap used; ~6–35 calls/s depending on load); all prior predictions are frozen and must not be edited; post-hoc results must be labelled post-hoc; 8-page paper.
- **Success condition:** a specific, testable explanation of the miscalibration that (i) removes most of the one-sided bias on existing data without using population outcomes to fit anything, (ii) predicts the dose–response that a sampling artefact implies, and (iii) passes at least one new *prospective* test frozen before its population runs.

## Technical Gap
The calibration grid has a median of **4 calls per (Δ, mean) cell** (mean 5.4; 2,769 cells), and **63% of cells are saturated at 0 or 1**. Every pre-registered estimator shrinks toward ½: Jeffreys point rates (k+½)/(n+1), Beta(k+½, n−k+½) posterior draws for predictive uncertainty, and a pseudo-count toward the base rate in the NW kernel. At a saturated cell with n = 4 the point estimate is 0.9 instead of 1, and posterior draws spread further. Because errors at saturated cells can only point inward, shrinkage is a one-sided **artificial lapse rate** λ ≈ ½/(n+1) that flattens the antisymmetric part of P_win = ½ + ½[ℓ(h,h′) − ℓ(h′,h)], making simulated contests more coin-flip-like than real ones. Population metrics accumulate this over 40 rounds of reinforcement, so a small per-decision bias becomes a large under-prediction of sd(h) and q — and, symmetrically, too-slow overthrow under anti-deference. Naive fixes fail: more seeds do not help (bias, not variance); a parametric logistic removes shrinkage but imposes a functional form that E1a rejects in 57/71 cells; fitting to population data would void the prospective test.

## Method Thesis
- **One-sentence thesis:** The micro→macro miscalibration is an estimator artefact — shrinkage of sparse, saturated two-agent cells toward ½ — and an unshrunk, shape-constrained kernel calibrated where populations actually go predicts population hierarchy without the one-sided bias.
- **Smallest adequate intervention:** change only the kernel estimator (remove the prior pull at saturated cells; borrow strength across Δ through monotonicity instead of toward ½) and the allocation of calibration calls (simulation-guided, toward visited states). The simulator, protocol, metrics and scoring stay identical.
- **Why timely:** LLM agents, unlike animals, can be queried thousands of times at controlled states, so the bottleneck in "physics of agent swarms" is measurement design, not theory. Getting the measurement right is what makes pre-deployment screening possible.

## Contribution Focus
- **Dominant contribution:** a diagnosed and corrected micro→macro screening pipeline: identification of shrinkage-induced attenuation as the cause of systematic under-prediction, with a shape-constrained unshrunk estimator (`flex_iso`) and visited-state calibration, tested retrospectively and prospectively.
- **Supporting contribution:** the recommendation to deployers/evaluators: avoid shrink-to-½ priors at saturated cells, allocate calls to visited states, use monotone smoothing; reported with dose–response evidence.
- **Explicit non-contributions:** no claim that the pre-registered criterion passed; no new population dynamics; no change to framing results.

## Proposed Method
### Complexity Budget
- **Reused:** calibration data (15k calls), state-matched data, simulator `population2.run`, metrics `order_params`, scoring `seed_mean_dist` + `lpd`, frozen null and flexible predictions as references.
- **New:** two estimators (`flex_mle`: near-flat Beta(k+0.05, n−k+0.05) cell draws; `flex_iso`: per-mean weighted isotonic regression in Δ with direction from the calibration-only sign of s, cell-bootstrap uncertainty) and one variant (`nw_noshrink`, α = 0). Implemented in `code/posthoc_attenuation.py`.
- **Intentionally not used:** hierarchical Bayesian GLMs (more assumptions, no time), population-outcome fitting, lapse-rate parameters estimated from populations.

### System Overview
```
calibration calls ──► estimator (Jeffreys | MLE | isotonic) ──► kernel ℓ(Δ, mean)
                                                                   │
           frozen protocol (N, ρ, μ, rounds) ──► simulator ◄───────┘
                                                   │  B=25 draws × S=8 runs
                                                   ▼
                                    seed-mean predictive dist ──► coverage, z, LPD vs observed populations
```
### Core Mechanism
- **Input/output:** calibration rows (Δ displayed, mean displayed, LEAD) → ℓ̂(Δ, mean) and B posterior/bootstrap draws.
- **flex_iso:** for each mean level, fit ℓ non-decreasing (s > 0) or non-increasing (s < 0) in Δ by weighted PAVA on cell rates (weights = n); bilinear interpolation as in the frozen kernel; uncertainty by resampling each cell Binomial(n, k/n) and refitting. Saturated cells stay at 0/1 unless a neighbour violates monotonicity.
- **Why it is the novelty:** it targets exactly the identified failure (inward-only error at saturated cells) while keeping the nonparametric shape the data demand.

### Training Plan / Calibration Allocation (prospective test)
- Choose two high-residual, cheap cells: **Gemma-4-26B norm100** and **Qwen3.5-397B norm0**.
- Simulate 24 populations with the frozen kernel; collect visited displayed states (Δ, mean); allocate **≥ 30 calls per visited cell** (≈ 600 calls per condition).
- Fit `flex_iso` on grid + new calls; **freeze** predictions (write-once, hashed) for N = 8, ρ = 0.5, μ = 0.1, 40 rounds; then run **5 new population seeds** per condition (~1,600 calls each).

### Failure Modes and Diagnostics
- *Residual bias persists after unshrinking* → attenuation is not the main cause; check coverage/visited-state mismatch (critique §2d) and population-induced sensitivity change (existing matched-sensitivity analysis).
- *Unshrunk kernels over-predict* (overshoot in some cells, e.g. GLM ended+comp in a partial run) → report symmetric error; prefer isotonic over MLE if it reduces variance.
- *Margin over null collapses* → unshrunk kernels are too noisy; report.

### Novelty and Elegance Argument
Closest work fits winner–loser models to animals and finds misfits (Lindquist & Chase 2009) or likelihood-fits endorsement models (Kawakatsu et al. 2021); LLM-population studies report hierarchy qualitatively. None diagnoses a *measurement* source of micro→macro miscalibration or corrects it prospectively. The change is a single estimator swap plus a call-allocation rule — no new modules.

## Claim-Driven Validation Sketch
### Claim 1 (dominant): attenuation explains the one-sided miscalibration
- **Minimal experiment A (no calls, running):** re-evaluate all 31 common-design cells and the three seeded-leader conditions with flex_mle / flex_iso / nw_noshrink through the identical scoring pipeline.
- **Metric:** fraction of cells with obs > pred (sd_h), mean z, coverage, LPD margin over null.
- **Expected:** fraction drops from ~0.85 toward ~0.5 and mean z toward 0 while the margin over null is kept; seeded-leader predictions move toward observed in all three framings.
- **Dose–response B (no calls):** subsample calibration to 50% / 25% per cell; predicted sd(h) with Jeffreys falls as n falls, MLE stays flat in expectation.

### Claim 2 (prospective): visited-state, unshrunk calibration predicts new populations
- **Experiment D:** two conditions above; freeze `flex_iso` predictions; 5 new seeds each.
- **Metric:** 90% coverage of the 5-seed mean for sd(h) and q; z.
- **Expected:** both metrics covered in both conditions, with |z| < 1; the frozen Jeffreys kernel (re-simulated on the same new data for reference) shows positive residuals.

## Experiment Handoff Inputs
- Must-prove: Claim 1 (A + B), Claim 2 (D).
- Must-run: A (running), B1, D.
- Metrics: sd(h), q, coverage, z, LPD margin.
- Highest-risk assumption: that remaining residual after unshrinking is small; the partial run already shows overshoot in a few cells.

## Compute & Timeline Estimate
- A: ~30–60 min CPU (running since 18:32). B: ~20 min CPU. D: ~4.4k Adaption calls (< 0.5 credit), ~40 min wall-clock.
- Paper edits: §4.2, abstract, §5, Appendix C/D (amendment A10 post-hoc; D as prospective addendum).
