# Experiment Results (living document; updated 2026-11-07)

> **SUPERSEDED-IN-PART NOTICE (auto-review rounds 1–2, 2026-11-07/08).** This is the chronological lab log. Where it
> conflicts with `review-stage/CLAIMS.md`, the claims file is authoritative. In particular, the following are
> superseded:
> - "all within 1 SD";
> - η from the global logistic (now finite-difference sensitivity, `results/eta_fd_beta.csv`);
> - 572-nat and 67-nat margins (now common-design and same-estimator figures, `results/coverage_common_design.csv`);
> - "N-independent / no tipping point / additive forgetting required / master-equation agreement" (C3 reframed as
>   finite-time);
> - L read as the lower-class mass (it is F(0.1T));
> - X1 4+4 tested against 1/8 (the composition null is 4/8, p = 0.19);
> - follow-up intervals built from single runs (now in `results/reeval_followups.json`);
> - hysteresis norm50 "inside" (it is outside).


All numbers come from files in `results/`. Every LLM call is logged in `results/calls.jsonl`. Predictions are
frozen in write-once, hash-stamped files before the corresponding population data is inspected (see the plan's
Amendments).

## M0 — Theory check
- The deterministic N-agent mean-field map reproduces the Lacasa–Luque pitchfork exactly: ρ_c = 0.111 at N=2 and
  0.200 at N=10, with μ=0.1, η=1, F=1.
- Our protocol updates both contestants, so its mean-field threshold is half of that.
- The stochastic finite-N model (N=8–16) is a **smooth crossover**, not a sharp transition. Sharp-ρ_c claims are
  therefore dropped, and we compare against surrogate populations run under the identical protocol.

## M1 — Pilot
See `idea-stage/pilot_results.md`. Haiku's η flips sign with framing: −0.27 under reward framing vs +1.14 under
competence framing. Pilot populations tracked their surrogates within about 1.8 SD.

## M2 — Calibration v2
Neutral wording; framing enters only through the norm weight w or the wording factorial. Fits use displayed scores
and the strict parser. η is the strict Bonabeau-kernel value 4·P′_win(0), with a bootstrap 95% CI.
`results/eta_table.csv` holds the full table.

| model | norm0 | norm25 | norm50 | norm75 | norm100 | ended | won | ended+comp | won+comp |
|---|---|---|---|---|---|---|---|---|---|
| Claude Haiku | **−1.45** [−2.34, −0.91] | −0.06 | 0.08 | 0.89 | **2.22** [1.67, 3.04] | 0.03 | 0.01 | 0.35 [0.20, 0.50] | 0.20 [0.12, 0.33] |
| Claude Sonnet | **−0.19** [−0.33, −0.12] | — | 0.00 | — | **2.22** [1.67, 3.07] | 0.01 | — | 0.23 [0.15, 0.39] | — |
| GPT-6-luna | 0.02 | — | — | — | 0.43 | — | — | 0.30 | — |
| DeepSeek-v4-flash | +0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00* | 0.00 | 0.00 |
| Gemma-4-26B-A4B | +0.21* | +0.40* | +0.40* | +0.10* | +0.13* | +0.01* | +0.01* | +0.03* | +0.03* |
| Gemma-4-31B | **−0.34*** | −0.01 | +0.05* | +1.04* | **+1.23*** | +0.01* | 0.00 | +0.02* | +0.04* |
| GLM-5.2 | +0.02* | +0.16* | +0.35* | +0.82* | +0.94* | +0.11* | +0.24* | +0.37* | +0.73* |
| Qwen3.5-397B | 0.00 | +0.08* | +0.10* | +0.12* | +0.18* | +0.05* | +0.14* | +0.11* | +0.12* |
| Qwen3.8-27B | +0.02 | +0.13* | +0.10* | +0.14* | +0.11* | +0.04 | +0.08* | +0.08* | +0.12* |

(* = 95% bootstrap CI excludes 0.) **Cross-family summary:**
- *Sign reversal* under an explicit equal-turns norm: 3 of 9 models (Haiku, Sonnet, Gemma-31B).
- *Monotone increase of η with w*: GLM, Gemma-31B and both Claude models; weak (≈0.1) for the Qwens.
- *Competence sentence raises η*: Haiku, Sonnet, GLM. Small or no effect elsewhere.
- **DeepSeek-v4-flash claims LEAD 99–100% in every condition, so η ≈ 0.** GLM at w=0 DEFERs 95% of the time.
  These are natural controls: pure assertiveness (or pure deference) without score-gap sensitivity yields coin-flip,
  egalitarian dynamics.

**Findings:**
- **(i) The norm weight moves status sensitivity monotonically through zero.** For Haiku it crosses zero between
  w=25 and w=50.
- **(ii) In the wording factorial, the competence sentence switches the winner effect on.** The verb ("won" vs
  "ended up as") has no effect.
- **(iii) With neutral wording, agents are nearly status-blind.** They claim LEAD 93–98% of the time.
- **(iv)** The *negative* (turn-taking) regime needs an explicit equal-turns norm, or the pilot's "being assigned
  work" wording.

## M3 — Inequality-feedback probe (exploratory; Haiku only so far)
The focal pair sits at ranks 4–5 of an 8-agent leaderboard; only the background spread changes.

| condition | low spread: local η | high spread: local η | low: p(LEAD) | high: p(LEAD) |
|---|---|---|---|---|
| ended+comp | 1.00 [0.43, 1.61] | 1.33 [0.79, 2.04] | .83 | .75 |
| norm50 | 0.92 [0.11, 1.69] | 1.50 [0.81, 2.21] | .71 | .60 |

**Pooled across 7 models × 2 conditions (1,341 calls): null.** The high-inequality slope exceeds the
low-inequality slope in 6 of 14 cells (sign test p = 1.0). Pooled LPM interaction Δ×high = −0.016 ± 0.030
(p = 0.59; model×condition fixed effects; SEs clustered). There is **no evidence of Stauffer σ-feedback**: background
inequality does not change status sensitivity. This favours fixed-η Bonabeau (pitchfork-type) over the
Stauffer saddle-node mechanism. Haiku's apparent effect was noise. Table: `results/probe_table.csv`.

## M5 — Pre-registered off-grid holdout (Haiku, ended+comp)
- Frozen choice: displayed (+4, −8), a gap of 12 that lies off the calibration grid. Flexible P_win = 0.875,
  Bonabeau P_win = 0.99999.
- Observed: 40 decisive high-score wins, 0 low-score wins, 8 ties (both LEAD). Empirical P_win = 0.917.
- **Log score: flexible −14.2 vs Bonabeau −47.8, Δ = 33.6 nats, which clears the pre-declared 3-nat threshold.**
  The strict one-η rule is overconfident at large gaps: low-status agents keep claiming the lead some of the time.

## M6/M7 — Populations vs frozen predictions
Setup: N=8, ρ=0.5, μ=0.1, 40 rounds. Primary metrics sd_h and q over the final 20 rounds.
`results/eval_v2.json` holds the full output.

**Claude Sonnet** (2 seeds per condition; predictions file `predictions_v2_claude_a1.json`, sha b0401def…):

| cond | obs sd_h | flexible 90% PI | Bonabeau mean | null mean (z) |
|---|---|---|---|---|
| norm0 | 0.95 | [0.95, 1.27] (z −1.5) | 1.46 | 1.99 (z −4.4) |
| ended+comp | 4.49 | [3.06, 4.82] ✓ | 4.91 ✓ | 1.97 (z +10.8) |
| norm100 | 5.79 | [4.38, 5.73] (z +1.7) | 6.05 ✓ | 2.07 (z +13.1) |

Summed log predictive density over sd_h and q:
- flexible +3.37
- logistic3 −1.33
- Bonabeau −6.36
- null −173.1

The conditional winner log-score is best for the flexible kernel in all 3 conditions. Observed populations are
slightly **more extreme than the flexible kernel predicts in both directions**. This is a candidate signature of
feedback beyond the pairwise kernel (n = 2 seeds per condition).

**Corrected evaluation (amendment A2; exact τ² + σ²/k seed-mean predictive).** Summed log predictive density over sd_h and q:

| model | flexible (grid) | Bonabeau (grid) | logistic3 | null |
|---|---|---|---|---|
| Claude Sonnet (3 cond × 2 seeds) | **+3.6** | −18.4 | −8.2 | −177.0 |
| Claude Haiku (4 cond × 3 seeds) | −40.9 | **−8.0** | −41.3 | −244.3 |

- Every kernel calibrated on pairwise data beats the status-blind null by more than 130 nats. **Pairwise behaviour
  predicts which populations become hierarchical and which stay egalitarian.**
- But Haiku populations are *more* stratified than the grid-calibrated flexible kernel predicts: `ended` sd_h is
  3.67 observed vs 2.19 predicted; `ended+comp` 6.26 vs 4.70; `norm100` 6.25 vs 5.08.
- **Diagnosis: the miss comes from coverage.** Populations spend most of their time at off-grid states (gaps of
  1.5–6 at varied mean status). The grid samples those states thinly (12 calls per cell) and interpolates them
  optimistically.
- **State-matched calibration**: 400 calls per condition at states visited by *surrogate* populations, so no LLM
  population data is used. It reproduces the population LEAD rates where the grid fails. For Haiku `ended` at
  Δ ∈ (−6, −3]: grid .83, state-matched .52, population .61. At Δ ∈ (−3, −1.5]: grid 1.00, state-matched .80,
  population .68.
- Because agents are stateless and prompts depend only on the displayed state, *population decisions are samples of
  the same kernel*. The remaining question is purely one of kernel estimation at visited states. That is the
  methodological lesson: **calibrate where the population lives.**
- Haiku state-matched predictions can only be **retrospective**: the Haiku populations were already scored, and 578
  of 1,600 statecal calls hit the Claude session limit. They re-run after the 18:50 reset.
- For the 6 open-weight models, grid **and** state-matched predictions were both frozen *before* their populations
  (15:13), so those are fully prospective.
**Adaption models**: 3 seeds × 4 conditions × 6 models, plus a ρ sweep {0.1, 0.25, 0.9} × {norm0, ended+comp, norm100}
for DeepSeek and Gemma-31B. Running, two models at a time.

## Compute
- Claude subscription: about 9.5k calls so far.
- GPT-6-luna: 540 / 1,500.
- Adaption: 1.9 / 50 credits so far.
- Claude subscription hit its session limit at about 15:00 and resets at 18:50. No population data was affected
  (0 invalid). Quota messages are now treated as failures and excluded on resume.

## M6/M7b — Open-weight populations, fully prospective (finished 17:43)
Setup: 6 Adaption models × {norm0, ended, ended+comp, norm100} × 3 seeds (N=8, ρ=.5), plus a ρ sweep for DeepSeek and
Gemma-31B. Grid **and** state-matched predictions were frozen at 15:13, before any of these populations ran.
`scratch/eval_full.log` has the full table; `results/eval_v2.json`. Summed log predictive density over sd_h and q
(higher is better):

| model | best kernel (lpd) | grid flexible | status-blind null | margin over null |
|---|---|---|---|---|
| Gemma-4-31B | state Bonabeau **+24.6** | +14.0 | −547.8 | 572 nats |
| GLM-5.2 | grid Bonabeau +4.9 | +0.9 | −230.1 | 235 |
| Qwen3.8-27B | state Bonabeau +2.9 | +2.2 | −115.3 | 118 |
| Qwen3.5-397B | state NW −3.1 | −21.8 (grid Bonabeau −70.3) | −230.4 | 227 |
| Gemma-4-26B-A4B | state Bonabeau −3.7 | −21.5 (grid Bonabeau −56.1) | −235.5 | 232 |
| DeepSeek-v4-flash | all ≈ +18 | +18.1 | +17.8 | ≈ 0 (expected: η ≈ 0, so kernel = null) |

**Findings:**
- **The micro→macro result replicates prospectively in 5 of 5 status-sensitive open-weight models.** The pairwise
  kernel beats the status-blind null by 118–572 nats.
- **DeepSeek is the built-in negative control.** It is status-blind, so every kernel, the null included, predicts
  the same egalitarian populations, and they are observed.
- **Calibrating where the population lives matters, and here it is shown prospectively.** For the two models where
  grid kernels missed (Qwen-397B, Gemma-26B), state-matched kernels recover 50–67 nats. State-matched Bonabeau is the
  best or near-best kernel in 4 of 6 models.

## X1 / X2 / BNR-rules / hysteresis — evaluation (2026-11-07 22:45; all predictions frozen before runs)
```
== x1
ds1_gemma7: rank1=minority 4/5 (binom p vs 1/8 = 0.001) | type_gap obs=9.84 nw=7.87[6.12,9.39] null=0.05[-4.12,4.32] | pb_corr obs=0.54 nw=0.49[0.38,0.58] null=0.01[-0.28,0.28] | sd_h obs=6.29 nw=5.16[4.85,5.46] null=5.09[4.83,5.35]
ds4_gemma4: rank1=minority 4/5 (binom p vs 1/8 = 0.001) | type_gap obs=9.24 nw=8.43[7.31,9.45] null=0.27[-2.59,3.10] | pb_corr obs=0.79 nw=0.83[0.74,0.87] null=0.03[-0.26,0.31] | sd_h obs=5.36 nw=4.83[4.50,5.16] null=5.08[4.81,5.35]
ds1_glm7_w0: rank1=minority 5/5 (binom p vs 1/8 = 0.000) | type_gap obs=11.69 nw=9.70[8.55,10.86] null=-0.11[-2.04,1.78] | pb_corr obs=0.58 nw=0.58[0.58,0.58] null=-0.00[-0.31,0.31] | sd_h obs=4.28 nw=3.79[3.43,4.16] null=2.23[1.94,2.54]
== x2
norm100  r=0.25  obs L=0.40 kmax=25.0 (n=3) | bnr_q0: L=0.35 90%[0.31,0.40] | kernel: L=0.33 90%[0.27,0.38]
norm100  r=0.5   obs L=0.58 kmax=20.7 (n=3) | bnr_q0: L=0.54 90%[0.48,0.60] | kernel: L=0.54 90%[0.48,0.60]
norm100  r=1.25  obs L=0.98 kmax=4.0 (n=3) | bnr_q0: L=0.95 90%[0.90,1.00] | kernel: L=0.97 90%[0.94,1.00]
ended    r=0.25  obs L=0.00 kmax=11.0 (n=3) | kernel: L=0.01 90%[0.00,0.04]
ended    r=0.5   obs L=0.67 kmax=5.7 (n=3) | kernel: L=0.72 90%[0.58,0.85]
ended    r=1.25  obs L=1.00 kmax=1.3 (n=3) | kernel: L=1.00 90%[1.00,1.00]
norm100 decisive contests: 1492 of 2160
ended decisive contests: 664 of 2160
== bnr_exp
norm100  r=0.5  obs L=0.33 (n=2) v=1.56 | grid: L=0.29 [0.25,0.33] | blind: L=0.00 [0.00,0.00] | BNR theory r/2=0.25, blind r_c=1
norm100  r=1.0  obs L=0.50 (n=2) v=1.00 | grid: L=0.53 [0.42,0.58] | blind: L=0.68 [0.50,0.83] | BNR theory r/2=0.50, blind r_c=1
norm100  r=1.5  obs L=0.75 (n=2) v=0.54 | grid: L=0.76 [0.67,0.83] | blind: L=1.00 [1.00,1.00] | BNR theory r/2=0.75, blind r_c=1
norm100  r=2.0  obs L=0.96 (n=2) v=0.10 | grid: L=0.97 [0.92,1.00] | blind: L=1.00 [1.00,1.00] | BNR theory r/2=1.00, blind r_c=1
norm100  r=2.5  obs L=1.00 (n=2) v=0.03 | grid: L=1.00 [1.00,1.00] | blind: L=1.00 [1.00,1.00] | BNR theory r/2=1.00, blind r_c=1
ended    r=0.5  obs L=0.00 (n=2) v=0.68 | grid: L=0.00 [0.00,0.00] | blind: L=0.00 [0.00,0.00] | BNR theory r/2=0.25, blind r_c=1
ended    r=1.0  obs L=1.00 (n=2) v=0.07 | grid: L=1.00 [1.00,1.00] | blind: L=0.68 [0.50,0.83] | BNR theory r/2=0.50, blind r_c=1
ended    r=1.5  obs L=1.00 (n=2) v=0.04 | grid: L=1.00 [1.00,1.00] | blind: L=1.00 [1.00,1.00] | BNR theory r/2=0.75, blind r_c=1
ended    r=2.0  obs L=1.00 (n=2) v=0.02 | grid: L=1.00 [1.00,1.00] | blind: L=1.00 [1.00,1.00] | BNR theory r/2=1.00, blind r_c=1
ended    r=2.5  obs L=1.00 (n=2) v=0.01 | grid: L=1.00 [1.00,1.00] | blind: L=1.00 [1.00,1.00] | BNR theory r/2=1.00, blind r_c=1
== hysteresis
norm0      equal  obs sd_late=0.92 | nw sd_late=0.98 [0.83,1.16] | bonabeau sd_late=1.06 [0.85,1.31]
norm0      seeded obs sd_late=0.95 leader_rank1=3.0 | nw sd_late=0.99 [0.79,1.21] rank1=6.6 [2,12] | bonabeau sd_late=1.06 [0.80,1.28] rank1=4.0 [1,8]
ended      equal  obs sd_late=2.37 | nw sd_late=2.06 [1.45,2.83] | bonabeau sd_late=2.00 [1.41,2.77]
ended      seeded obs sd_late=1.45 leader_rank1=5.0 | nw sd_late=2.00 [1.34,2.71] rank1=8.7 [1,22] | bonabeau sd_late=2.04 [1.42,2.72] rank1=7.5 [1,17]
norm50     equal  obs sd_late=2.70 | nw sd_late=2.55 [1.83,3.31] | bonabeau sd_late=2.30 [1.61,3.25]
norm50     seeded obs sd_late=3.32 leader_rank1=26.7 | nw sd_late=2.68 [2.13,3.39] rank1=11.7 [1,36] | bonabeau sd_late=2.33 [1.56,3.34] rank1=8.0 [1,23]
```
