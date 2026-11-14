# Chase study — E3 report (populations against the frozen forecasts)

Written 2026-11-14, after the E3 queue finished at 16:00. All cells of amendment C5 are complete; none is partial.
Evaluation: `code/chase_e3_eval.py` (sha256 `3e16161b…4bcb86`, logged before any outcome was read) →
`results/chase_e3_eval.json`. Forecasts: `results/chase_e2_predictions.json`, stamp `9f2241d0…d86ef734`, unchanged
since the freeze. Population file: `results/chase_e3_pop.jsonl`, 508 groups, sha256 `5967fb05…878e6868`.

Terms. OWN shows each agent only its own record. LOG adds the partner's record against third parties, which is the
information Chase's bystander effect needs. "Linear" is the share of groups of four that end as a strict hierarchy
(chance 0.375). t_tri is triangle transitivity in groups of eight (0 = chance, 1 = fully transitive). DD + DS is
the share of triads whose first two relationships guarantee transitivity (0.50 if relationships form at random).

## 1. Does seeing the partner's record make hierarchies more linear? (H3)

| Cell | Groups of four: OWN | LOG | LOG − OWN [95% CI] | Fisher p (Holm) | N = 8 t_tri: OWN | LOG | LOG − OWN [95% CI] | Mann–Whitney p |
|---|---|---|---|---|---|---|---|---|
| Gemma-31B `comp` | 0.36 (29/80) | 0.61 (49/80) | +0.25 [+0.10, +0.40] | 0.0025 (0.008) | 0.06 | 0.60 | +0.54 [+0.35, +0.73] | 0.0002 |
| Gemma-31B `norm100` | 0.78 (31/40) | 0.95 (38/40) | +0.17 [+0.03, +0.32] | 0.048 (0.095) | 0.32 | 0.79 | +0.46 [+0.04, +0.86] | 0.056 |
| GLM-5.2 `comp` | 0.48 (19/40) | 0.48 (19/40) | 0.00 [−0.22, +0.20] | 1.0 (1.0) | not run | | | |
| DeepSeek `comp` (control) | 0.25 (5/20) | 0.30 (6/20) | +0.05 [−0.25, +0.30] | 1.0 | 0.12 | 0.27 | +0.14 [+0.02, +0.27] | 0.14 |

Seeds at N = 8: 12 per arm in Gemma-31B `comp`, 6 in `norm100`, 4 in the control. Holm is over the three
non-control contrasts in groups of four, as fixed in amendment C5.

- **Gemma-31B `comp`: supported.** Linear groups rise from 36% to 61%, and at N = 8 transitivity rises from chance
  to 0.60. Landau h′ at N = 8 goes from 0.31 to 0.61.
- **Gemma-31B `norm100`: same direction, not significant after correction.** OWN is already at 0.78, so there is
  little room; LOG reaches 0.95. Landau h′ at N = 8 goes from 0.47 to 0.81.
- **GLM-5.2 `comp`: no difference**, as the frozen forecast said (−0.05 [−0.25, +0.15]).
- **Control:** no difference in groups of four. At N = 8 LOG is above OWN and above its forecast (section 2); with
  four seeds per arm this is not distinguishable from noise by the planned test.

In Gemma-31B `comp` the first 40 groups per arm alone give +0.10 (0.43 against 0.53); the result above rests on the
80 groups per arm that amendment C5 fixed before any population call.

## 2. Did the frozen forecasts hold?

Observed values are on the planned number of groups (first 40 or 6 where a cell was extended). Brackets are the
frozen 5th–95th percentile interval. LOSO is the leave-one-seed-out upper bound: the same kernel form fitted on the
population's own decisions.

| Cell | Arm | Observed | Frozen forecast | Inside | LOSO |
|---|---|---|---|---|---|
| Gemma-31B `norm100` | OWN, N = 4 | 0.78 | 0.56 [0.42, 0.70] | no | 0.71 |
| | LOG, N = 4 | 0.95 | 0.95 [0.90, 1.00] | yes | 0.96 |
| | DYAD, N = 4 | 0.55 | 0.37 [0.20, 0.55] | edge | 0.38 |
| | REDACT, N = 4 | 0.75 | 0.60 [0.40, 0.80] | yes | 0.68 |
| | OWN, N = 8 | 0.32 | 0.08 [−0.08, 0.24] | no | 0.29 |
| | LOG, N = 8 | 0.79 | 0.75 [0.62, 0.86] | yes | 0.88 |
| | SCORE+LOG, N = 8 | 0.98 | 0.99 [0.96, 1.00] | yes | 0.99 |
| | SCORE, N = 8 | 1.00 | 0.83 [0.68, 0.94] | no | n/a |
| Gemma-31B `comp` | OWN, N = 4 | 0.43 | 0.36 [0.25, 0.45] | yes | 0.38 |
| | LOG, N = 4 | 0.53 | 0.59 [0.45, 0.75] | yes | 0.62 |
| | DYAD, N = 4 | 0.45 | 0.37 [0.20, 0.55] | yes | 0.37 |
| | REDACT, N = 4 | 0.30 | 0.48 [0.30, 0.70] | edge | 0.41 |
| | OWN, N = 8 | −0.10 | −0.01 [−0.16, 0.13] | yes | 0.02 |
| | LOG, N = 8 | 0.61 | 0.33 [0.13, 0.54] | no | 0.54 |
| | SCORE+LOG, N = 8 | −0.01 | 0.10 [−0.07, 0.29] | yes | 0.13 |
| | SCORE, N = 8 | 0.12 | 0.15 [−0.04, 0.31] | yes | n/a |
| GLM-5.2 `comp` | OWN, N = 4 | 0.48 | 0.56 [0.45, 0.68] | yes | 0.53 |
| | LOG, N = 4 | 0.48 | 0.51 [0.37, 0.62] | yes | 0.52 |
| DeepSeek `comp` | OWN, N = 4 | 0.25 | 0.37 [0.20, 0.55] | yes | 0.38 |
| | LOG, N = 4 | 0.30 | 0.37 [0.20, 0.55] | yes | 0.38 |
| | OWN, N = 8 | 0.12 | −0.01 [−0.18, 0.18] | yes | 0.00 |
| | LOG, N = 8 | 0.27 | −0.01 [−0.16, 0.16] | no | −0.01 |

17 of 22 arm forecasts hold; with 90% intervals about 20 would be expected. All five misses are on the same side:
more hierarchy than forecast.

LOG − OWN contrasts against their frozen intervals: 5 of 7 inside. Outside: Gemma-31B `norm100` in groups of four
(+0.17 against +0.39 [+0.25, +0.53], because OWN was higher than forecast) and Gemma-31B `comp` at N = 8 (+0.71 on
the first six seeds against +0.35 [+0.05, +0.57], because LOG was higher than forecast).

Where the misses come from:
- **Gemma-31B `norm100`, OWN.** The kernel was too flat for agents with deep losing records, which the calibration
  barely sampled. Post hoc diagnosis in `CHASE_LOG.md` (13 Nov). The upper bound recovers most of the gap.
- **Gemma-31B `comp`, LOG at N = 8.** The upper bound (0.54 [0.42, 0.65]) covers the observation, so the kernel
  form is adequate and the frozen coefficients were too weak. Not diagnosed further.
- **Gemma-31B `norm100`, SCORE.** Fully transitive in all six seeds; the forecast came from the earlier project's
  score kernel and was 0.83.
- **DeepSeek `comp`, LOG at N = 8.** Not explained. The upper bound does not cover it either (−0.01), so whatever
  produces it is outside the kernel's features. Four seeds.

## 3. Is the difference forecast by the bystander features? (H4)

The same forecast with the partner-record features removed from the LOG kernel:

| Cell | LOG observed | Forecast with bystander features | Forecast without |
|---|---|---|---|
| Gemma-31B `norm100`, N = 4 | 0.95 | 0.95 | 0.74 |
| Gemma-31B `norm100`, N = 8 | 0.79 | 0.75 | 0.34 |
| Gemma-31B `comp`, N = 4 (80 groups) | 0.61 | 0.59 | 0.42 |
| Gemma-31B `comp`, N = 8 (12 seeds) | 0.60 | 0.33 | 0.03 |

In both Gemma cells the kernel without bystander features leaves LOG near the OWN level, and the kernel with them
reaches the observed level in three of four rows and gets half-way in the fourth. H4 is supported for Gemma-31B.

The outcome-less placebo agrees: showing the partner's lines with outcomes removed (REDACT) gives 0.75 in
`norm100` and 0.30 in `comp`, level with OWN (0.78 and 0.36) and below LOG. No test was pre-registered for this
comparison.

## 4. Chase's sequences

| Cell | DD + DS: OWN | LOG | Frozen LOG forecast | BDID: OWN | LOG |
|---|---|---|---|---|---|
| Gemma-31B `norm100`, N = 4 | 0.53 | 0.70 | 0.59 [0.53, 0.65] | 0.26 | 0.09 |
| Gemma-31B `norm100`, N = 8 | 0.51 | 0.70 | 0.59 [0.54, 0.65] | 0.26 | 0.11 |
| Gemma-31B `comp`, N = 4 | 0.52 | 0.54 | 0.52 [0.46, 0.59] | 0.24 | 0.23 |
| Gemma-31B `comp`, N = 8 | 0.48 | 0.54 | 0.53 [0.49, 0.59] | 0.24 | 0.22 |
| GLM-5.2 `comp`, N = 4 | 0.56 | 0.59 | 0.52 [0.46, 0.58] | 0.24 | 0.18 |

Under `norm100` the pattern Chase described appears: with partner records visible, the two transitivity-guaranteeing
sequences rise to 70% and the sequence in which a bystander later dominates the initial winner (BDID) becomes rare.
The rise is larger than the kernel forecast. Under `comp` linearity rises without any change in sequence shares.
So in that cell the hierarchy does not become linear through which relationships form first; the relationships
must be reorganised afterwards. This report does not establish how.

## 5. Visible scores

In Gemma-31B `norm100` a visible score alone gives full transitivity (1.00), and adding the log changes nothing
(0.98). In Gemma-31B `comp` a visible score gives almost none (0.12), and adding the score to the log removes the
log's effect (−0.01 against 0.60 for the log alone). Both were forecast (0.10 [−0.07, 0.29] for SCORE+LOG in
`comp`). No test was pre-registered for the score arms against LOG.

## 6. What this does and does not show

- The bystander effect on hierarchy structure is present in one of the three models tested (Gemma-31B), under both
  framings in direction and after correction under `comp`.
- It is absent in GLM-5.2 `comp`, as its state-matched kernel forecast, even though E1 found a small partner-record
  slope for that cell (−0.051).
- The forecasts were prospective and mostly held, and the bystander features carry the forecast difference. The
  forecasts erred toward too little hierarchy in five arms.
- Not run: GLM-5.2 `norm100`, GLM-5.2 at N = 8 (amendment C5). Their forecasts are untested.
- N = 8 uses 6 seeds per arm in the planned cells (amendment C2), not the 8 of the pre-registration; the 95%
  interval for Gemma-31B `norm100` at N = 8 is wide ([+0.04, +0.86]).
- Invalid replies became coin flips: at most 0.6% of decisions per arm, except GLM-5.2 LOG at 5.3% (206 of 3,882),
  as the E2 report anticipated.
- The explanations in section 2 other than the first are untested, and the first is post hoc.

## Next

E4 (rewind, mixed-model groups of four) and E1 batch C follow in the order of amendment C2. The Sonnet arm of E1
is at 2,566 of 4,224 calls and is not analysed until complete.
