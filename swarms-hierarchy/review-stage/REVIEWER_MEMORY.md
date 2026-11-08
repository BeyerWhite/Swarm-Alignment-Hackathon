# Reviewer Memory

## Round 1 — Score: 4/10 (not ready)

### Raw Reviewer Response (verbatim)
**No. The current claims are not ready for submission.** There is a credible exploratory workshop paper here, but several conclusions exceed the evidence, and some evaluation errors change the interpretation.

I replayed all 144 main population trajectories from their events: score updates and winner assignments were consistent, with zero recorded invalid decisions. The negative status sensitivity in Haiku/Gemma-31B, DeepSeek's near-constant assertiveness, and the reported BNR/X2 cell means reproduce. The six open-weight experiments provide real prospective evidence. Those strengths survive this review.

The remaining weaknesses, ranked by severity:

1. **C3 does not establish its central physics claim.** "A sharp transition needs additive forgetting" is unsupported. The proportional-forgetting simulations use one horizon—120 rounds, with a 40-round measurement window. Their `q` measures finite-window persistence; it cannot rule out slow mixing or a stationary transition. Persistent microscopic noise does not rule out phase transitions either. The curves are not literally N-independent: Haiku/Bonabeau at w=87.5 gives q=0.9342 at N=8 versus 0.8756 at N=700. The plan's score-cap explanation is also wrong under the repeated-target encounter schedule: an actual Gemma trajectory reaches 19.141, exceeding the asserted Q/μ=10 bound. There is a separate theoretical-validation failure. Using bnr_x2.py, at r=0.9, N=700, T=200, I obtain master-equation F(20)=0.8946, versus simulator 0.9331 ± 0.0012 SEM across 20 seeds. Perfect matching, tie handling, and batched decline differ from the implemented continuous-time equation. Finally, the measured "lower-class fraction" is F(0.1T), which includes mobile agents. In the stated continuum scaling, its limit is r+0.1, or r/2+0.05 under rate-2 normalization, away from saturation. It is not the asymptotic lower-class mass. **Minimum fix:** describe discrete BNR-inspired, finite-time dynamics; compare against the identical simulator; withdraw master-equation agreement and additive-forgetting necessity. Say "no observed sharpening through N=700 over this horizon." A top venue would require time/window convergence, kernel uncertainty, and an appropriate stationary order parameter.

2. **The principal quantitative prediction criterion failed.** The declared macro criterion requires predictive coverage, not merely superiority to the status-blind null. Recomputed coverage of primary metric means: Haiku grid flexible 4/8; Sonnet flexible 4/6; Gemma-26B selected best state Bonabeau 3/8; Qwen-397B selected best state NW 5/8. Haiku dispersion misses reach z=4.63, 3.69, and 3.23. Selecting the highest-scoring frozen kernel does not make these predictions calibrated. **Minimum fix:** explicitly report failure of the original quantitative success criterion. Narrow C2 to prospective prediction of status dependence and qualitative hierarchy differences, while showing coverage and residuals. A top venue needs an independently evaluated, fixed model-selection rule and substantially better calibrated forecasts.

3. **The follow-up uncertainty calculations contain consequential errors.** hysteresis.py compares a three-seed observed mean with a single-run prediction interval. The norm50 seeded persistence durations are 20, 20, 40, averaging 26.67 rounds. Applying the existing A2 seed-mean method to the frozen samples gives an NW 90% interval of approximately [2.0, 23.2]. The observation falls outside it. C6's reassurance that it lies inside the interval is therefore incorrect. The 40-round observation is also censored. X1 and X2 independently resample flattened forecast samples when averaging seeds, incorrectly averaging away shared calibration uncertainty. BNR evaluation likewise compares means with individual-run intervals. **Minimum fix:** evaluate the same statistic on both sides and preserve the shared calibration draw across seeds. Recompute all follow-up intervals from existing samples and report the persistence underprediction. This does not by itself establish Stauffer feedback.

4. **X1's qualitative effect survives, but its significance and prediction claims need correction.** The 4+4 arm is tested against 1/8; its composition-based null is 4/8, making 4/5 successes p=0.1875, not p≈0.001. "Finishes at rank 1 in 4/5" actually means highest last-ten-round average score; literal final-round leadership is 3/5 in the 1+7 Gemma arm. All three mixed-population dispersion means exceed their reported NW intervals; two type-gap means also exceed their intervals. Correcting calibration grouping widens intervals but retains these misses. Moreover, P(A wins)=½+½[P(A=LEAD)−P(B=LEAD)] holds even for dependent choices: the joint term cancels under fair tie resolution. The probes cannot establish independence. Their p-values reproduce, but their variance calculation also treats fractional tie outcomes as Bernoulli and ignores calibration uncertainty. **Minimum fix:** correct the composition null, label the prespecified rank endpoint accurately, report the quantitative misses, and describe the probes as marginal-kernel transport checks.

5. **C1's "local η" is not an identified derivative, and saturated-cell intervals are misleading.** fit_v2.py computes an average finite-difference sensitivity using displayed gaps 0.4 and 1. Its maximum is 3.5 by construction. Claude norm100 cells saturate at 0/24 versus 24/24 LEAD, producing bootstrap intervals [3.5, 3.5]. That reflects bootstrap boundary collapse. **Minimum fix:** call this finite-difference sensitivity, distinguish it from the fitted logistic derivative, and use binomial/Beta uncertainty for saturated cells. The supported negative-sensitivity findings remain valid.

6. **Several headline comparisons overstate or misattribute the evidence.** The 572-nat Gemma-31B margin includes the interaction-rate sweep; on the common four-condition ρ=0.5 design it is 176.55 nats; across the five sensitive models comparable margins are ~118–235. The 67-nat state-matching improvement for Qwen changes both calibration design and estimator family; matched Bonabeau comparisons improve by ~55 (Qwen) and ~52 (Gemma-26B). Sonnet's amended forecast was created at 14:18:37, after its population calls ended at 14:12:14; A1 discloses this; distinguish it from forecasts frozen before execution. **Minimum fix:** separate common-design results from sweeps, report matched-estimator comparisons, label forecast chronology precisely. Describe summed marginal LPD as a composite predictive score, not a joint likelihood ratio.

7. **The defensible scientific scope is narrower than emergence of operational leadership.** Stateless responses to externally supplied scores and explicit coordination norms. X2 verifies continued obedience at high scores, but matching BNR behavior is nearly automatic once the response rule is deterministic. The board's one ASSIGN is real; task completion is supplied by the harness's random completion mechanism. Six sparse-assignment runs support a bounded negative pilot result. **Minimum fix:** frame the contribution as response-kernel estimation and finite-time population prediction under controlled prompts.

Smaller corrections: 71 calibration cells, not 81; DeepSeek's minimum condition-level LEAD rate is 97.6%, not 99%; valid inequality-probe count is 1,339, not 1,341. The 21–23% support violation is predominantly mean-status extrapolation; gap violations are about 0.7%. Refresh the figures, especially the prediction plot currently showing only Sonnet. The inequality-probe interaction p≈0.59 is correct. The raw BNR/X2 measurements hold up.

Score: 4/10
Verdict: not ready

---

## Round 2 — Score: 5/10 (almost; workshop bar) — verbatim response stored at .aris/traces/auto-review-loop/2026-11-07_run01/round2.response.md
Key points:
1. seed_mean_dist produced out-of-support intervals; use support-preserving shared-calibration resampling.
2. Residual significance overstated (cells not independent); Haiku competence contrast is only a point estimate.
3. Figures: clipped error bars, η label in Fig 2, Fig 7 L label.
4. x1.py eval still tests 4+4 against 1/8; mark superseded results; separate C6; C5 "rejected" wording.
5. Top-venue contribution limited.
---

## Round 3 — Score: 8/10 (ready; workshop bar) — top-venue score stated separately: 5/10
Remaining (minor): Monte Carlo coverage noise -> exact enumeration; legacy eval commands; stale details (DeepSeek margin, Claude norm100 CI, norm0 mean/per-seed, horizon spec, Fig 7 label); top-venue limits.
---
