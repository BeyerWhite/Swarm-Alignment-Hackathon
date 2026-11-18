# E8 — equal turns (`norm0`), larger fresh runs (amendment C8)

Groups of four, 24 rounds, OWN against LOG, fresh seeds from 34000. Evaluation `code/chase_e8_eval.py` (hashed in
the log before any E8 outcome was read) → `results/chase_e8_eval.json`. The H5 reading is the one fixed in
amendment C6 item 6: "reverses" if LOG − OWN is below zero with a 95% interval excluding zero; "contradicted" if
above zero with p < 0.05; "vanishes" if neither and the interval excludes +0.17; otherwise unresolved.

| Run | Model | Groups per arm | Calls | Credits | Time |
|---|---|---|---|---|---|
| E8a | Gemma-4-26B | 140 | 26,915 | 0.97 | 17 Nov 09:14–18:39 |
| E8b calibration | Qwen3.5-397B | — | 1,126 | 0.35 | 18:39–19:03 |
| E8b | Qwen3.5-397B | 80 | 15,211 | 4.47 | 19:03–00:20 |

No failed calls, no invalid answers.

## Summary

Under equal turns, letting agents see other pairs' records does not make hierarchies more linear in either model.
The difference is close to zero, and both intervals exclude +0.17, so by the registered rule H5 now reads
**"vanishes"** for Gemma-26B (E5 left it unresolved) and for Qwen3.5-397B. The mechanism found after the fact in E5
replicates as a registered co-primary: seeing others' records makes Gemma-26B agents lead less often, so fewer
contests are decided by the answers. E5's high own-record share (0.50) was partly a high draw (0.39 on 140 fresh
groups), but the forecasts for that arm are still too low.

## 1. Gemma-4-26B (E8a)

| | OWN | LOG | LOG − OWN [95%] | Fisher p | H5 |
|---|---|---|---|---|---|
| Fresh, 140 per arm | 0.39 (54) | 0.41 (58) | +0.03 [−0.09, +0.14] | 0.71 | vanishes |
| Fresh + E5, 180 per arm (secondary) | 0.41 | 0.43 | +0.02 [−0.08, +0.12] | 0.83 | vanishes |

- **Co-primary: share of contests decided by the answers** (one LEAD, one DEFER), per group: OWN 0.39, LOG 0.29,
  −0.10, Mann–Whitney p = 2 × 10⁻²¹. **Supported.** E5 had −0.10 [−0.13, −0.07].
- LEAD rate 0.33 → 0.27. DD + DS by the E9 rule 0.51 → 0.49: no shift in sequences, unlike Gemma-31B in E9.
- **First 40 fresh groups per arm against the frozen E5 forecast:** OWN 0.48 against 0.24 [0.13, 0.35] (outside);
  LOG 0.43 against 0.38 [0.25, 0.50] (inside); LOG − OWN −0.05 against +0.13 [−0.05, +0.33] (on the edge).
- **Descriptive, not registered:** the same first 40 OWN groups (0.48) lie just outside the E6 part B forecast
  0.30 [0.18, 0.45] and the E6 part A interval 0.34 [0.22, 0.47]. Over all 140 groups the share is 0.39; the
  forecast means (0.24 in E5, 0.30 after calibration in rounds) are still below it. The own-record shortfall in this
  cell is smaller than E5 suggested, but it is not gone.

## 2. Qwen3.5-397B (E8b)

| | OWN | LOG | LOG − OWN [95%] | Fisher p | H5 |
|---|---|---|---|---|---|
| Observed, 80 per arm | 0.31 (25) | 0.30 (24) | −0.01 [−0.15, +0.14] | 1.00 | vanishes |
| Frozen forecast (form X) | 0.34 [0.26, 0.41] | 0.34 [0.25, 0.44] | −0.00 [−0.11, +0.11] | | |

- All forecasts hold: both arms and the contrast are inside their intervals, and so is the LOG forecast without the
  bystander features (0.33 [0.23, 0.41]).
- Decided-contest share (estimate only): 0.27 → 0.21, −0.07, p = 10⁻⁷. LEAD rate 0.20 → 0.16. C8 expected this
  model's LEAD rate to be near one half; in these populations it was 0.20, so its direction behaves like Gemma-26B's.
- DD + DS 0.44 → 0.50.

## 3. What this says about H5

H5 said that under equal turns the bystander effect on linearity reverses or vanishes. With the larger runs it
vanishes in both models tested, by the registered rule. Two limits:

- **A "vanishes" reading needs an effect elsewhere to vanish from.** Neither model has a population run under the
  other framings that shows a LOG − OWN effect. For Qwen3.5-397B the frozen E7 forecast is close to zero under
  `norm100` and `comp` as well (−0.01 and +0.02). So this result does not show that equal-turns norms remove an
  effect the same model would otherwise produce. Only Gemma-31B has a demonstrated effect, and it was not run
  under `norm0`.
- **The mechanism is the same in both models.** Seeing other pairs' records lowers the LEAD rate. Under equal turns
  the rate is already low, so fewer contests are decided and the hierarchy gains nothing. In Gemma-31B, where the
  baseline LEAD rate is high, the same lowering raises the decided share (E5 report, post hoc).
