# Claims — v5 (final). [R3] = round-3 fixes; [X] = fixes from the gpt-6-astra cross-check, applied after the final primary review

**Scope.** We estimate stateless LLM agents' *response kernels* to externally supplied coordination scores under
controlled prompts, and we test finite-time population predictions made from those kernels. Agents emit LEAD/DEFER
text and never perform tasks. We make no claim about operational leadership, competence, or productive delegation.
The models in question are Bonabeau/Stauffer (Lacasa & Luque 2005) and a discrete BNR-inspired variant
(Ben-Naim & Redner 2005).

**Protocol.**
- Contests: a lone LEAD wins; otherwise a coin flip.
- Scores: +1/−1, ×0.9 decay per round, displayed to 0.1.
- Populations: N=8, ρ=0.5, 40 rounds, synchronous. Each agent initiates with probability ρ, so targets repeat.

Changes from v1 are marked **[R1]**.

## C1 — Framing controls finite-difference status sensitivity
Evidence: `results/eta_fd_beta.csv`, Fig 1–2.
- **[R1]** The measure is renamed *finite-difference sensitivity*: the mean of 4·ΔP_win/Δ over displayed gaps 0.4
  and 1, with a maximum of 3.5. It is no longer called η or a derivative.
- **[R1]** CIs come from per-cell Beta(k+½, n−k+½) posteriors. The saturated Claude norm100 cells now get
  [3.10, 3.48] (maximum 3.5) [R3], not a collapsed interval.
- `results/eta_table.csv` keeps the global logistic η separately, as a different quantity.

Findings:
- (a) A significant *negative* sensitivity (turn-taking) under the "equal turns" norm (w=0): Haiku −1.92
  [CI excludes 0], Gemma-31B −1.62.
- (b) Sensitivity rises with w in most models.
- (c) **[R2]** The competence-sentence contrast (`ended+comp − ended`; posterior median and 95% interval in
  `results/competence_contrast.csv`) is credibly positive only for **GLM: +1.50 [0.67, 2.31]** and
  **Sonnet: +0.86 [0.24, 1.48]**. Haiku +0.51 [−0.10, 1.11] and Qwen-397B +0.60 [−0.15, 1.42] are positive point
  estimates only. For the Gemmas and DeepSeek the contrast is about 0, and Qwen3.8-27B is negative (−0.27).
- (d) DeepSeek-v4-flash is status-blind. Its lowest condition-level LEAD rate is 97.6% **[R1 corrected]**.

## C2 — Frozen pairwise kernels predict which populations stratify; quantitative calibration fails
Evidence: `results/coverage_common_design.csv`, `results/eval_v2.json`, Fig 3.
- **[R1]** Results are restricted to the common design (4 conditions, ρ=0.5, N=8). Sweeps are reported separately.
- **[R1] The pre-declared quantitative success criterion (90% predictive coverage for every condition) FAILED.**
  Pre-declared grid flexible kernel, primary metrics sd_h and q:

  **[R3]** These intervals enumerate exactly every calibration replicate × k-run subset:

  | model | coverage |
  |---|---|
  | Gemma-26B | 2/8 |
  | Haiku | 3/8 |
  | Sonnet | 3/6 |
  | Qwen-397B | 5/8 |
  | GLM | 6/8 |
  | Qwen3.8-27B | 6/8 |
  | Gemma-31B | 7/8 |
  | DeepSeek | 7/8 |

- What holds: on the common design, the frozen kernel beats the status-blind null by **104–224 nats [X]**. The KDE bandwidth uses a Scott factor from the B·S simulated
  runs, not the enumeration count. Across bandwidths ×0.5 to ×2 the range is 75–278 nats, positive in every model
  and setting (`results/coverage_common_design.csv`) of composite
  predictive score (a summed marginal log predictive density, not a joint likelihood ratio). This holds in all 7
  status-sensitive models. 6 are prospective; Sonnet's amended forecast (A1) was created at 14:18:37, after its
  populations finished at 14:12:14. DeepSeek is the negative control: margin ≈ 0.3 nats.
- The null predicts sd_h ≈ 2.0 for every population. Observed values span 0.9–6.3, and the kernels track that ordering
  (Fig 3).
- **[R1, new] Systematic residual.** Populations are *more* stratified than the kernel predicts. Among
  status-sensitive models, 22/27 sd_h residuals are positive (median z = +1.6), and **the model-average residual is
  positive in 7/7 models [R2]**. This is exploratory: no cell-level p-value is claimed, because cells share models
  and seeds. The same excess
  appears in all three X1 arms (C4) and in the norm50 hysteresis arm (C6). The agents are stateless and the prompt is
  a function of the displayed state, so this points to kernel misestimation at the states populations visit, not to
  agent memory. Calibration at the visited states reduces it (next bullet) but does not remove it.
- **[R1]** State-matched vs grid calibration, *same estimator* (Bonabeau): +41.1 nats (Qwen-397B), +21.7 (Gemma-26B),
  +8.1 (Haiku, retrospective). Small or no gain elsewhere (0.0 to +4.4) [X: fixed bandwidth].

## C3 — Scale: finite-time results only [R1 reframed]
Evidence: `results/fss_w.jsonl`, round-1 horizon check (below), Fig 6–7.
- **[R1] Withdrawn:** "a sharp transition needs additive forgetting", "N-independent", the score-cap explanation (a
  real trajectory reaches 19.1 > Q/μ), and agreement with the BNR master equation. At r=0.9, N=700, T=200 our discrete
  simulator gives 0.933 vs the master equation's 0.895; perfect matching, ties and batched decline differ.
- What remains:
  - Under proportional forgetting with the fitted kernels, q over a 40-round window **shows no sharpening from N=8 to
    700 over a 120-round horizon**.
  - q is a finite-window persistence measure, not a stationary order parameter. With a 200-round window it roughly
    falls by half to two-thirds: Haiku norm75 0.81 → 0.43; Gemma-31B norm50 0.60 → 0.20; GLM norm25 0.69 → 0.28
    (N=700, T=600, 6 seeds). Keeping the 40-round window at T=600 leaves q unchanged (0.81, 0.59, 0.69), so the drop
    reflects window length. Archived in `results/fss_horizon.json` [X]. Rank order is not permanently frozen.
  - No stationary transition claim is made.
- Discrete BNR-inspired dynamics:
  - The LLM populations are compared only against the **identical discrete simulator**.
  - "L" is relabelled **F(0.1T)**, the fraction of agents at or below 0.1T. It includes mobile agents and is not the
    asymptotic lower-class mass.

## C4 — Mixed-model populations (X1)
Evidence: `results/reeval_followups.json`, `results/x1_*`.

**[R1] Corrections:**
- The rank-1 endpoint is labelled correctly: highest mean score over the last 10 rounds, as frozen.
- The final-round rank 1 is also reported.
- The 4+4 arm is now tested against its composition null of 4/8.
- Intervals now preserve the shared calibration draw.

Findings:
- 1 DeepSeek + 7 Gemma-31B: the last-10 leader is DeepSeek in 4/5 runs (null 1/8, p = 0.001); final-round leader
  3/5. The frozen state-matched kernel predicted 0.61 [0.20, 1.00]: inside [R2: support-preserving interval].
- 4 + 4: 4/5 against a null of 4/8, p = 0.19, not significant.
- 1 DeepSeek + 7 GLM (w=0): 5/5, p < 0.001.
- **Quantitative misses:** sd_h is above the kernel interval in all 3 arms, and the type gap is above it in 2 of 3.
  This is consistent with the C2 residual.
- **[R1]** The cross-model probes are *marginal-kernel transport checks*: the identity P_win = ½ + ½[P_A − P_B] holds
  under fair ties even when choices are dependent. Their χ² treats fractional tie outcomes as Bernoulli and ignores
  calibration uncertainty.

## C5 — Discrete BNR-inspired populations (X2; bnr_exp)
Evidence: `results/reeval_followups.json`.
- X2 norm100 vs the identical simulator with q=0: F(0.1T) is inside the 90% seed-mean intervals at all 3 r. As the
  frozen note states, this match is near-automatic given q̂ = 0. The informative part is that agents stayed decisive
  up to scores of 25. The calibrated-kernel prediction misses at r = 0.25 (0.40 vs [0.27, 0.38]).
- The `ended` control behaves as its kernel predicts at all 3 r, with no stuck class at r = 0.25.
- bnr_exp (rate-2 normalisation): all 5 norm100 cells are inside the grid-kernel intervals. **[R2]** The observed
  values lie outside the status-blind prediction's 90% interval at r = 0.5, 1, 1.5 and 2.

## C6 — Reversal of seeded hierarchies: mixed evidence [R1 corrected]
Evidence: `results/reeval_followups.json`. Gemma-31B, 3 seeds per arm, seed-mean intervals.
- **[R2] norm0:** the seeded leader holds rank 1 for a mean of 3.0 completed rounds (per seed 2, 6, 1; first loss in rounds 3, 7, 2) [X]. That is *faster* than NW predicts (6.6
  [4.0, 9.3]) and inside the Bonabeau interval.
- **[R2] ended:** the seeded leader holds rank 1 for a mean of 5.0 completed rounds (per seed 4, 7, 4; first loss in rounds 5, 8, 5),
  inside both kernels' intervals (NW 8.7). The seeded populations end *less* stratified than predicted (sd_late 1.45 vs
  NW [1.61, 2.42]) [X].
- **norm50:** the seeded leader holds rank 1 for a mean of **26.7 completed rounds (per seed 20, 20, 40; 40 = never lost, censored), above both
  kernels' intervals** (NW [4.0, 22.7]; support-preserving, exactly enumerated, censored at 40 on both sides [R3]). Seeded sd_h is also above. This is a genuine underprediction of persistence
  at intermediate norm strength. With 3 seeds and stateless agents, it is not evidence of Stauffer feedback.
- The pooled inequality probe (7 models × 2 conditions, **1,339** valid calls **[R1 corrected]**): interaction
  p = 0.59, sign test p = 1.0. No evidence of inequality feedback was detected.

## C7 — Scoreless message board: bounded negative pilot (pre-registered)
- 6 runs produced 1 ASSIGN. Task completion is supplied by the harness at random; agents do not perform tasks.
- 80–124 duplicate-claim collisions per run did not induce delegation.

## Minor corrections [R1]
- 71 calibration cells, not 81.
- Large-N off-support decisions (21–23%) are mostly mean-status extrapolation; gap extrapolation is about 0.7%.
- Fig 3 now shows all models on the common design.

## Figures [R2]
- Fig 1–2 plot the posterior median with its 95% interval, with no clipping, labelled "finite-difference status
  sensitivity".
- Fig 7 is relabelled F(0.05T) (discrete simulator), with the continuum r/r_c shown as a guide only.
- `refine-logs/EXPERIMENT_RESULTS.md` carries a superseded-in-part notice.
- `code/x1.py eval` now uses the composition null and the shared-draw sampler.

## [R3]
- `hysteresis.py`, `x2.py` and `bnr_exp.py eval` print legacy single-run intervals and point to
  `code/reeval_followups.py`, which is authoritative.
- Coverage counts now use exact enumeration (`evaluate.seed_mean_dist`).

## [X] Cross-check fixes (gpt-6-astra, high)
- KDE bandwidth tied to independent runs, with bandwidth sensitivity reported.
- 1e-9 tolerance on interval membership.
- C6 durations labelled as completed rounds held at rank 1.
- Horizon check archived.
- Fig 5 now plots observed seed means against the seed-mean predictive band.
- The X2 legacy evaluator is noted as pooling across seeds (not single-run). `reeval_followups.py` is authoritative.
