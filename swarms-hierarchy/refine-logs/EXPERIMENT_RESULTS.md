# Experiment Results (living document; updated 2026-11-07)

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
