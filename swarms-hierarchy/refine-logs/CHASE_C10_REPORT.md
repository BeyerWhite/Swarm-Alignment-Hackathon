# E11–E13 — the three no-call follow-ups of amendment C10

Run 20 Nov 2026 on existing groups only, no model calls. The scripts were checked against the hashes in C10 before
the runs: `chase_c10.py` f92cca00…, `chase_e11_eval.py` 14a5a848…, `chase_e12.py` 1689d2ec…, `chase_e13.py`
986c45c4…. `chase_fit.py`, which they import, was never hashed in the log; it is unchanged since the study's first
commit (454d89a). Every rebuilt feature matched the stored values (0 mismatches). Outputs: `results/chase_e11_eval.json`,
`results/chase_e12.json`, `results/chase_e13.json`. E14, the fresh-cell test of the LEAD-rate rule, needed calls;
it ran 21–23 November and is reported in its own section below.

## Summary

- **E11 (how the order is built).** In Qwen3.5-397B the whole gain from partner records arises after relationships
  form, as predicted (P1 supported). In Gemma-4-31B the same is true, contrary to the prediction (P2 not supported).
  In both models partner records raise the share of groups whose first relationships are already linear, yet that
  starting order hardly predicts the final one. In Qwen, reversals under partner records go more often toward the
  agent with the better record (S1, as predicted).
- **E12 (the Qwen `comp` forecast miss).** Reading "neither". Even fitted on the 120 groups' own decisions, the
  forecast's decision model gives 0.67 [0.62, 0.74] linear against the observed 0.81. So the miss is not a matter of
  which records the calibration sampled. The best richer form (partner recency) fits decisions better but does not
  close the gap either. In Gemma-4-31B `comp`, where the forecast was right, the same model covers (0.59 against 0.58).
- **E13 (the Gemma-26B equal-turns gap).** Reading "neither". On all 180 groups the gap remains: 0.33 [0.27, 0.38]
  simulated against 0.41 observed. Agents do differ from each other beyond their records (σ 0.39, LR p = 6 × 10⁻²⁶),
  but simulating those differences does not raise linearity. The round number adds nothing.
- **E14 (the LEAD-rate rule on fresh cells).** The rule fails. Neither of the two cells it predicted to gain
  gained (mean −0.04 [−0.14, +0.07]), while the three predicted nulls stayed null. No fresh cell gained at all.
- **Together.** In two cells, a model of independent single decisions, fitted to the very decisions the groups
  made, produces less hierarchy than the groups did. Something at the level of the group, not in any tested feature
  of a single decision, adds order.

## E11 — formation or later

Linear share, groups of four, `comp`. a: share of groups whose initial relationships form a linear order; s and r:
share finally linear among groups that started linear and among the rest.

| | Arm | a | s | r | Linear |
|---|---|---|---|---|---|
| Qwen3.5-397B | OWN (120) | 0.53 | 0.50 | 0.54 | 0.52 |
| | LOG (120) | 0.65 | 0.78 | 0.86 | 0.81 |
| Gemma-4-31B | OWN (120) | 0.33 | 0.31 | 0.42 | 0.38 |
| | LOG (120) | 0.65 | 0.59 | 0.57 | 0.58 |

| LOG − OWN, 95% group-bootstrap interval | Qwen3.5-397B | Gemma-4-31B |
|---|---|---|
| Total | +0.29 [+0.18, +0.40] | +0.20 [+0.08, +0.33] |
| Formation | −0.01 [−0.03, +0.01] | −0.02 [−0.06, +0.03] |
| Later | +0.30 [+0.19, +0.41] | +0.22 [+0.09, +0.34] |
| of which keep | +0.17 [+0.08, +0.26] | +0.14 [+0.05, +0.23] |
| of which repair | +0.13 [+0.06, +0.21] | +0.08 [−0.02, +0.17] |
| Later − formation | +0.31 [+0.19, +0.42] → **"later"** | +0.23 [+0.08, +0.38] → **"later"** |
| | P1 supported | P2 not supported |

- **The formation term is zero because the start does not carry over, not because starts are alike.** Partner
  records raise the share of groups that start linear (Qwen 0.53 → 0.65, Gemma 0.33 → 0.65). But within an arm,
  groups that started linear end linear no more often than those that did not (s ≈ r). The final order, measured
  over the last half of the rounds, is made in those later rounds.
- **S1, as predicted for Qwen.** Under partner records, 0.59 of reversals put the agent with the better net record
  against third parties on top, against 0.51 with own records only: +0.08 [+0.02, +0.13]. REDACT: 0.51. Gemma-31B,
  not predicted: 0.47 → 0.76 (+0.29 [+0.13, +0.45]), but on few reversals (86 and 49).
- **Descriptive.** Reversals per formed dyad: Qwen 0.79 (OWN), 0.80 (LOG), 0.57 (REDACT); Gemma 0.21 and 0.11.
  Initial transitivity of triads formed entirely by the E9 rule: Qwen 0.85 → 0.87; Gemma 0.74 → 0.95.
- **What it says about Chase's sequences.** In Gemma-31B partner records do produce Chase's start: more DD and DS
  (E9) and initial triads that are nearly all transitive. But that start is not what decides the final hierarchy.
  In Qwen there is no such start, and the gain comes the same way: groups with partner records keep and repair
  their order better over the later rounds, with reversals that follow visible records.

## E12 — why the forecast missed Qwen `comp` with partner records

| Leave-one-seed-out log-loss | B0 (forecast form) | B1 + partner recency | B2 + phinge | B3 + cmp | B4 all |
|---|---|---|---|---|---|
| Qwen3.5-397B `comp` LOG (11,380 decisions) | 0.6080 | 0.5944 | 0.6081 | 0.6081 | 0.5941 |
| Gemma-4-31B `comp` LOG (11,442; no test) | 0.2356 | 0.2315 | 0.2355 | 0.2357 | 0.2317 |

B1 is selected in Qwen (0.014 better than B0). In Gemma-31B it is 0.004 better, below the 0.005 threshold, so B0
is kept there.

| Group-bootstrap adequacy (90%) | Simulated | Observed | Covered |
|---|---|---|---|
| Qwen LOG, B0 | 0.67 [0.62, 0.74] | 0.81 | no |
| Qwen LOG, B1 | 0.66 [0.59, 0.73] | 0.81 | no |
| Qwen OWN, X | 0.56 [0.48, 0.63] | 0.52 | yes |
| Gemma-31B LOG, B0 | 0.59 [0.50, 0.66] | 0.58 | yes |
| Gemma-31B OWN, X | 0.40 [0.32, 0.47] | 0.38 | yes |

Reading: **"neither"**.
- **Coverage.** The calibration covered the states the groups visited: no Qwen decision fell in a cell with fewer
  than five calibration rows (Gemma-31B: 15%).
- **The fitted model is too flat in the partner's record.** Where the partner's third-party net is +3 or better,
  Qwen agents led 0.29 of the time. The model fitted on the calibration rows predicts 0.38 there, and the one
  fitted on the groups' own decisions 0.36. In the middle and lower bins the miss is smaller and of the other sign.
  This was not part of the registered reading.

## E13 — the Gemma-26B equal-turns own-record arm

180 groups (E5 40, E8a 140), 17,272 decisions, observed linear share 0.41.

| | Result |
|---|---|
| Leave-one-seed-out log-loss | C0 0.5760, C1 (+ round) 0.5758 |
| C2 random intercept per agent | σ 0.39, LR 109.5, p = 6 × 10⁻²⁶ (720 agents) |
| Adequacy C0 | 0.33 [0.27, 0.38], not covered |
| Adequacy C1 | 0.34 [0.28, 0.39], not covered |
| Adequacy C2 | 0.33 [0.27, 0.38], not covered |
| Spread of per-agent LEAD rates | 0.109 observed against [0.097, 0.106] under C0 |

Reading: **"neither"**.
- **Agent differences.** They are real here and in both comparison cells: Gemma-31B `norm100` σ 0.73 (LR 40) and
  Qwen `comp` OWN σ 0.33 (LR 35). But at this size they do not change the simulated linear share.
- **The earlier misses.** At 180 groups the forecast gap is smaller than E5's 0.50 suggested, but it is not gone.

## E14 — the LEAD-rate rule on five fresh cells

The rule (frozen 21 Nov 23:13, before any population call, stamp acb78687…): from round-1 calibration rows, a cell
gains from partner records iff q(l_LOG) − q(l_OWN) > 0, with q(l) = 2l(1 − l), the decided share if the two answers
were independent. 80 OWN and 80 LOG groups of four per cell, seeds 54000–54079, run whatever the prediction.
"Gain" means LOG > OWN with one-sided Fisher p < 0.05.

| Fresh cell | l_OWN → l_LOG (calibration) | dq, P(dq > 0) | Predicted | OWN | LOG | LOG − OWN [95%] | Observed |
|---|---|---|---|---|---|---|---|
| Gemma-4-26B `comp` | 0.83 → 0.87 | −0.055, 0.07 | no gain | 0.66 | 0.55 | −0.11 [−0.26, +0.04] | no gain |
| Gemma-4-26B `norm100` | 0.93 → 0.93 | +0.004, 0.53 | gain | 0.44 | 0.33 | −0.11 [−0.26, +0.04] | no gain |
| Gemma-4-31B `norm0` | 0.21 → 0.15 | −0.072, 0.02 | no gain | 0.26 | 0.39 | +0.12 [−0.02, +0.26], p 0.064 | no gain |
| Qwen3.8-27B `comp` | 0.71 → 0.69 | +0.016, 0.71 | gain | 0.50 | 0.54 | +0.04 [−0.11, +0.19] | no gain |
| GLM-5.2 `norm100` | 0.87 → 0.91 | −0.064, 0.05 | no gain | 0.41 | 0.39 | −0.02 [−0.17, +0.12] | no gain |

| | Cells | Mean LOG − OWN [95%] | Criterion | Holds |
|---|---|---|---|---|
| P1, predicted gain | 2 | −0.04 [−0.14, +0.07] | lower bound > 0 | no |
| P2, predicted no gain | 3 | −0.00 [−0.09, +0.08] | upper bound < 0.17 | yes |

Reading: **the rule fails** (3 of 5 cells correct).
- **The test was weak in the direction that mattered.** No fresh cell gained, so a rule that never predicts a gain
  would have scored 5 of 5. Both gain predictions sat on the line: P(dq > 0) 0.53 and 0.71, with 90% intervals
  across zero. So P2 holding says little; what the test shows is that the rule found no new cell that gains.
- **Round-1 LEAD rates did not carry over to groups (descriptive).** In the groups the LEAD rate moved with partner
  records differently from the calibration: Gemma-26B `norm100` 0.92 → 0.95 (calibration flat), so decided contests
  fell from 0.15 to 0.09; Qwen3.8-27B 0.73 → 0.73 (calibration 0.71 → 0.69); Gemma-31B `norm0` 0.20 → 0.20
  (calibration 0.21 → 0.15).
- **The largest difference was in a cell predicted null.** Gemma-4-31B under equal turns rose 0.26 → 0.39 (p 0.064),
  short of the gain criterion. It is the model that gains under `comp` and `norm100`. Not tested further here.
- **Invalid answers.** GLM-5.2 LOG 349 (about 4–5% of its decisions), Gemma-4-31B `norm0` 111 and 86; the others 0–1.
- **Cost.** 12.2 credits (estimated 11.3), shared ledger 99.43 → 111.66, under the cap of 150.

## Post hoc check (after the results above; not registered)

Candidate: the two answers in a contest might be complementary beyond the model, with one agent deferring more often
when the other leads. That would raise decided contests and could build order. It is not complementary. With each
model fitted on all its cell's decisions, the observed share of decided contests was:

| Cell | Observed | Model | 95% range under the model |
|---|---|---|---|
| Qwen `comp` LOG | 0.526 | 0.543 | [0.530, 0.555] |
| Qwen `comp` OWN | 0.491 | 0.489 | [0.477, 0.502] |
| Gemma-31B `comp` LOG | 0.263 | 0.261 | [0.252, 0.270] |
| Gemma-26B `norm0` OWN | 0.388 | 0.417 | [0.407, 0.427] |

Where the models miss, there are slightly fewer decided contests than they imply, not more. A candidate not tested
here is persistence in specific pairs: one agent giving way to one partner beyond what the visible record of that
pair (d, last ten rounds) shows.

## Post hoc predictor screen (24 Nov, after E14; not registered)

Four candidate predictors of which cells respond, scored on the 13 cells with calibration rows and groups, no calls
(`code/chase_predictors_posthoc.py` → `results/chase_predictors_posthoc.json`; full table in the log, 24 Nov).
- **Yielding to strong partners** (LEAD at a partner net of −2..2 minus at +3 or better): every responsive cell
  scores 0.24–0.60, nine of the ten others 0.16 or less. The exception, Gemma-4-26B `comp` (0.29), lost ground (−0.11).
- **The full simulation forecast** from the same rows sorts 12 of 13 as well, but misses Qwen3.5-397B `comp`, the
  miss known since E7. It does separate Gemma-4-26B `comp`. On E14's fresh cells it is 5 of 5 (all null).
- **Room to respond** (how far fitted LEAD probabilities spread) does not predict; the fitted slope on the partner's
  record misses Qwen, whose response is a threshold.
- No single score sorts all 13 cells. Yielding looks necessary in these data, and is not sufficient.

## Limits

- E11–E13 analyse groups already analysed. E11's predictions were formed from the E9 and E10 sequence results.
- The adequacy intervals come from group-bootstrap refits of the same decisions they are compared with. So a miss
  is a strong result, and a cover a weak one.
- "Neither" in E12 and E13 says what the gap is not. It does not say what it is.
