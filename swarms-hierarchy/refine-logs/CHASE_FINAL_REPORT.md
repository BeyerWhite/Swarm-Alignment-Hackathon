# Chase's bystander effect in LLM agent populations — final report

Written 2026-11-16. The study ran from 12 to 15 November 2026 on branch `chase-bystander`. This report summarises
the stage reports (`CHASE_E1_REPORT.md` to `CHASE_E5_REPORT.md`); the pre-registration is
`CHASE_PREREGISTRATION.md` and every change to it is in `CHASE_LOG.md` (amendments C1–C6).

## Summary

Chase proposed that animal dominance hierarchies become linear because individuals react to contests they watch,
not only to the ones they fight (the bystander effect). This study asked whether LLM agents do the same and
whether it makes their groups more hierarchical.

- **Single decisions: yes, in most models.** Seven of nine models are less willing to take the lead against a
  partner whom they can see winning against others. The effect is one-sided and often a threshold: agents defer to
  visible winners and do not press visible losers.
- **Groups: yes in one model, no elsewhere.** In Gemma-4-31B, letting agents see their partner's record raised the
  share of strictly ordered groups of four from 36% to 61%, and from 78% to 95% under a second framing (the second difference does not
  survive correction for multiple tests). No such
  effect was found in GLM-5.2, in a mixed-model group, or in Gemma-4-26B under an equal-turns framing.
- **A real single-decision effect does not imply a group effect.** GLM-5.2 has a statistically clear bystander
  response and no group consequence, and this was forecast before the groups were run.
- **Forecasts made before each population run mostly held.** 22 of 28 arm forecasts fell inside their 90% intervals.
  All six misses were in the same direction: more hierarchy than forecast.
- **No transitive inference.** No model reasoned "A beat B, B beat me, so defer to A" beyond counting wins.
- **LLM hierarchies are far less repeatable than Chase's fish.** Re-running the same mixed group gave the same
  hierarchy 2–3% of the time, against 27% in his data.

## 1. Design

Agents meet in pairs and each answers LEAD or DEFER. A lone LEAD wins; if both give the same answer a coin flip
decides. Agents have no memory; everything they know is in the prompt. The experimental arms differ only in which
record of past contests the prompt shows.

| Arm | The agent sees | Role |
|---|---|---|
| DYAD | its past contests with this partner | baseline |
| OWN | all its own past contests | winner and loser effects only |
| LOG | OWN plus the partner's contests with third parties | bystander information |
| REDACT | as LOG, with the outcomes of the partner's lines removed | placebo for the extra text |
| SCORE, SCORE+LOG | a public score board, alone or with the log | the earlier project's condition |

Three framings were used: `comp` (leading is presented as a matter of competence), `norm100` (organisers are said
to weight records) and `norm0` (organisers are said to weight equal turns).

The study followed one procedure throughout:
1. Measure how single decisions depend on the visible record (E1).
2. Fit a decision model to those measurements, simulate groups with it, and freeze a hash-stamped forecast (E2).
3. Run real groups and score them against the forecast (E3, E5).

Outcome measures:
- **Linear share:** the share of groups of four that end as a strict hierarchy. Chance is 0.375.
- **Transitivity (t_tri):** for groups of eight; 0 is chance, 1 is fully transitive.
- **DD + DS share:** the share of triads whose first two relationships form in one of Chase's two sequences that
  guarantee transitivity. It is 0.50 if relationships form at random.

## 2. What was run

| Stage | Purpose | Size |
|---|---|---|
| E0 | Re-analysis of the earlier project's score-board groups (retrospective) | no calls |
| E1 | Single decisions on constructed records, 9 models, 24 model × framing cells | 50,688 calls |
| E2 | Calibration at records that simulated groups reach; frozen forecasts | 9,400 calls |
| E3 | Populations: Gemma-31B (`comp`, `norm100`), GLM-5.2 (`comp`), DeepSeek control | 508 groups |
| E4 | One mixed-model group of four re-run 40 times per arm | 80 groups |
| E5 | Exploratory: Gemma-26B under `norm0`, own calibration and forecast | 2,625 calls, 132 groups |

In total about 149,000 model decisions and 720 groups. Six open-weight models ran through Adaption (16.6 credits
of the 60-credit cap). Claude Haiku 5.5, Claude Sonnet 5.5 and GPT-6-Luna ran through their command-line tools and
took part in E1 only.

## 3. Results by hypothesis

| | Hypothesis | Verdict |
|---|---|---|
| H1 | Agents lead less against a partner with a better record against others | Supported: 15 of 24 cells, 7 of 9 models |
| H2 | Transitive inference through a shared third party | Not supported in any cell |
| H3 | Groups that see partner records are more linear | Supported in Gemma-31B `comp`; same direction in `norm100`; absent in GLM-5.2 |
| H4 | The difference is forecast by the bystander features of the decision model | Supported for Gemma-31B (no formal test was planned) |
| H5 | Under equal turns the effect reverses or vanishes | Unresolved by the pre-set rule |
| H6 | Re-runs rarely reproduce a hierarchy; not more often with logs; logs raise linearity | First part supported, second consistent, third not supported |

### H1 and H2 — single decisions (E1)

β_p is the change in the probability of LEAD per unit of the partner's net record against third parties. Chase
predicts a negative value. ✓ marks a Holm-adjusted p below 0.05 over all 24 cells.

| Model | `comp` | `norm100` | `norm0` |
|---|---|---|---|
| Gemma-4-31B | −0.089 ✓ | −0.146 ✓ | −0.006 |
| Gemma-4-26B | −0.069 ✓ | −0.028 ✓ | −0.108 ✓ |
| GLM-5.2 | −0.051 ✓ | −0.021 ✓ | −0.016 |
| Qwen3.5-397B | −0.082 ✓ | −0.077 ✓ | −0.046 ✓ |
| Qwen3.8-27B | −0.008 | +0.008 | +0.005 |
| DeepSeek-v4-flash | 0.000 | 0.000 | −0.003 |
| Claude Haiku 5.5 | −0.011 ✓ | −0.052 ✓ | not run |
| Claude Sonnet 5.5 | 0.000 | −0.069 ✓ | not run |
| GPT-6-Luna | −0.079 ✓ | −0.092 ✓ | not run |

- **Every significant effect has the sign Chase predicts.** No cell has a positive slope that survives adjustment.
- **The effect is one-sided.** Most models lead nearly always at baseline, so the partner's record can only lower
  LEAD, and it does so against visible winners. GPT-6-Luna, the one model with a baseline well below 1, also leads
  more against visible losers under `comp` (0.62 → 0.76).
- **It is often a threshold.** Gemma-31B `comp` leads in every call while the partner is at most one win ahead and in
  0.41 of calls when the partner is three ahead. Sonnet under `norm100` does the same (1.00, then 0.54). The grid
  has no point in between.
- **Two models ignore the partner's record:** DeepSeek leads whatever is shown, and Qwen3.8-27B uses its own and
  the direct record only. Sonnet under `comp` answered LEAD in all 2,112 calls.
- **No transitive inference.** The largest contrast is GPT-6-Luna `comp` (+0.21 ± 0.10, adjusted p 0.69). In models
  at ceiling this test is uninformative rather than negative.
- **The placebo is not clean for three models.** For GPT-6-Luna, Gemma-26B and Qwen3.5-397B, partner lines without
  outcomes also lower LEAD. Part of their reaction is to seeing that the partner has a history at all. β_p compares
  records of equal length and is not affected.

### H3 and H4 — groups (E2, E3)

| Cell | Linear share, groups of four: OWN | LOG | LOG − OWN [95% CI] | p (Holm) | t_tri, groups of eight: OWN | LOG | LOG − OWN [95% CI] | p |
|---|---|---|---|---|---|---|---|---|
| Gemma-31B `comp` | 0.36 (29/80) | 0.61 (49/80) | +0.25 [+0.10, +0.40] | 0.0025 (0.008) | 0.06 | 0.60 | +0.54 [+0.35, +0.73] | 0.0002 |
| Gemma-31B `norm100` | 0.78 (31/40) | 0.95 (38/40) | +0.17 [+0.03, +0.32] | 0.048 (0.095) | 0.32 | 0.79 | +0.46 [+0.04, +0.86] | 0.056 |
| GLM-5.2 `comp` | 0.48 (19/40) | 0.48 (19/40) | 0.00 [−0.22, +0.20] | 1.0 (1.0) | not run | | | |
| DeepSeek `comp` (control) | 0.25 (5/20) | 0.30 (6/20) | +0.05 [−0.25, +0.30] | 1.0 | 0.12 | 0.27 | +0.14 [+0.02, +0.27] | 0.14 |

- **Gemma-31B `comp`:** partner records take groups of four from chance to 61% linear and groups of eight from
  chance to clear transitivity. This survives correction.
- **Gemma-31B `norm100`:** the same direction at both sizes, but the own-record groups were already 78% linear, and
  the difference does not survive correction.
- **GLM-5.2 `comp`:** no difference, as its frozen forecast said (−0.05 [−0.25, +0.15]).
- **Why the bystander features get the credit (H4):** the same forecast with the partner-record features removed
  from the decision model leaves LOG near the OWN level (0.74 and 0.42 in groups of four, against 0.95 and 0.61
  observed). The placebo arm agrees: REDACT gave 0.75 and 0.30, level with OWN.
- **Chase's sequences appear in one cell only.** Under `norm100`, DD + DS rises from 0.53 to 0.70 with partner
  records. Under `comp` linearity rises with no change in sequences (0.52 to 0.54), so the hierarchy must be
  sorted out after relationships first form. How was not established.
- **A visible score changes the picture.** Under `norm100` a score board alone gives full transitivity and the log
  adds nothing. Under `comp` the score board gives almost none (0.12), and adding it to the log removes the log's
  effect (−0.01 against 0.60).

### H5 — equal-turns framing (E5, exploratory)

| Gemma-26B `norm0` | OWN | LOG | LOG − OWN [95% CI] | p |
|---|---|---|---|---|
| Linear share, groups of four | 0.50 (20/40) | 0.47 (19/40) | −0.03 [−0.25, +0.20] | 1.0 |
| t_tri, groups of eight | −0.04 | −0.04 | +0.01 [−0.20, +0.21] | 1.0 |

The rule fixed before the run required an interval excluding zero to call a reversal and one excluding +0.17 to
call the effect vanished. Neither holds, so H5 is unresolved. This cell has hardly any hierarchy in either arm:
agents mostly defer, and 60–73% of contests were settled by the coin.

An unplanned comparison gives a plausible mechanism. Partner records lower the LEAD rate in all three Gemma cells.
Where agents lead most of the time, that produces more contests with one LEAD and one DEFER, which are the ones
the records decide (0.40 → 0.68 under `norm100`, 0.21 → 0.27 under `comp`). Where they already defer most of the
time it produces more double-DEFER coin flips (0.39 → 0.29 under `norm0`). So the same response to the partner's
record can build structure or dissolve it depending on the baseline. This was not planned and is unadjusted.

### H6 — re-running the same group (E4)

One group (Gemma-31B, GLM-5.2, Qwen3.8-27B, DeepSeek; `comp`) was re-run 40 times per arm.

| | OWN | LOG | LOG − OWN [95% CI] |
|---|---|---|---|
| Linear groups | 0.50 | 0.60 | +0.10 [−0.12, +0.32] |
| Two re-runs end in the same hierarchy | 0.03 | 0.02 | −0.01 [−0.03, +0.03] |

- Hierarchies are almost never reproduced; random rank orders among linear groups would give 0.04.
- There is a tendency in rank without a fixed order: DeepSeek ends high and Qwen3.8-27B low in both arms, which
  matches their baseline willingness to lead.
- E1 later showed that two of the four members do not use partner records at all, which may be why logs add little
  here.

## 4. How well the forecasts did

| | Inside the frozen 90% interval | Misses |
|---|---|---|
| Arm forecasts, E3 | 17 of 22 | 5, all toward more hierarchy |
| Arm forecasts, E5 | 5 of 6 | 1, toward more hierarchy |
| LOG − OWN contrasts, E3 and E5 | 7 of 9 | 2 |

- **What the forecasts got right:** the size of the bystander effect in Gemma-31B `comp` in groups of four (+0.23
  forecast, +0.25 observed), the level of LOG under `norm100` (0.95 and 0.95), the null in GLM-5.2, and the
  cancelling effect of a score board under `comp`.
- **The recurring miss is the own-record arm.** In Gemma-31B `norm100` own-record groups were 0.78 linear against
  0.56 forecast, and in Gemma-26B `norm0` 0.50 against 0.24.
- **The diagnosed cause (post hoc, first case only):** agents with deep losing records nearly stop leading, and the
  calibration almost never sampled such records. The loser effect feeds itself: deferring produces losses, which
  deepen the record. A second calibration pass in E5, aimed at this, did not close the gap.
- **Consequence:** the forecast bystander gain for `norm100` (+0.39) was more than twice the observed one (+0.17),
  because the baseline was underestimated and not because LOG fell short.

## 5. What this says about Chase's account

- **The ingredient exists.** Most LLMs tested show a bystander effect on single decisions, with the predicted sign.
- **It is sufficient for hierarchy in at least one model.** In Gemma-31B, with scores hidden, adding only the
  partner's third-party record moves groups from chance toward linearity. Chase's 2002 fish went from 56% linear
  when meeting in isolated pairs to 92% when assembled as a group; here the analogous arms gave 36% and 61%
  (`comp`), and 78% and 95% (`norm100`).
- **It is not necessary.** Own-record groups under `norm100` reach 78% linear with no bystander information. The
  post hoc diagnosis attributes this to a loser effect that only appears at long losing records.
- **It does not always matter.** A small bystander slope (GLM-5.2) has no group consequence, and under equal turns
  the response exists in decisions without producing structure.
- **Chase's route through particular sequences is not the general route here.** It shows under one framing only.
- **These hierarchies are not stable properties of the group.** Chase's fish reproduced the same hierarchy in 27%
  of groups and were linear in over 90%; the mixed LLM group did so 2–3% of the time and was linear in 50–60%.

## 6. Limits and deviations

- **One model carries the population result.** Population tests covered Gemma-31B (two framings), GLM-5.2 (one
  framing, groups of four) and Gemma-26B (equal turns only). Four other models with single-decision effects
  (Qwen3.5-397B, Haiku, Sonnet, GPT-6-Luna) were never run as groups.
- **Small counts at N = 8.** Six seeds per arm in most cells, 12 in Gemma-31B `comp`, 4 in the control; the
  pre-registration planned 8.
- **Cells not run:** GLM-5.2 `norm100`, GLM-5.2 at N = 8. Their forecasts are untested.
- **Changes made before the data they affect:** the placebo and baseline arms were cut to 20 groups; Gemma-31B
  `comp` was raised to 80 groups and 12 seeds after the forecast showed 40 would have about 50% power. Its first 40
  groups per arm alone give +0.10.
- **E5 differs in procedure:** a second calibration pass, and the forecast was frozen and stamped automatically
  without the owner seeing it before the groups started.
- **The Claude and GPT arms are wrapped calls.** The Claude tool adds a fixed sentence to the system prompt; the
  GPT tool surrounds each prompt with about 14,600 tokens of agent instructions. These are constant across
  conditions, so contrasts are unaffected, but absolute LEAD levels are those of the wrapped call.
- **Invalid replies became coin flips:** under 1% of decisions everywhere except GLM-5.2 with the full log (5.3%).
- **Chase's 1982 sequence percentages were not verified** and no number from that paper is used. The 2002 figures
  are quoted as verified in the pre-registration.
- **Post hoc material:** the loser-threshold diagnosis, the decided-contests mechanism in section 3, and the low
  placebo arm in E5 (4 of 20 linear) were found after seeing data and are unadjusted.

## 7. What would settle the open questions

- **Population runs for a second responsive model,** ideally one off the ceiling (Qwen3.5-397B or GPT-6-Luna), to
  see whether the Gemma-31B result generalises.
- **Calibration that reaches deep records,** for example by iterating between simulated groups and measurements
  until the visited records stop changing, to fix the own-record forecasts.
- **A larger equal-turns test,** or one in a cell where fewer contests fall to the coin; 40 groups per arm could
  not separate "no effect" from the effect seen elsewhere.
- **The text of Chase (1982, *Behaviour*),** so that the sequence shares can be compared with his.

## 8. Files

| | |
|---|---|
| Plan and changes | `refine-logs/CHASE_PREREGISTRATION.md`, `refine-logs/CHASE_LOG.md` |
| Stage reports | `refine-logs/CHASE_E1_REPORT.md` … `CHASE_E5_REPORT.md` |
| Single decisions | `results/chase_e1_kernel.jsonl`, `chase_e1_cells.csv`, `chase_e1_summary.json` |
| Frozen forecasts | `results/chase_e2_predictions.json`, `results/chase_e5_predictions.json` |
| Populations | `results/chase_e3_pop.jsonl` (E3 and E5), `results/chase_e4_rewind.jsonl` |
| Evaluations | `results/chase_e3_eval.json`, `chase_e4_eval.json`, `chase_e5_eval.json` |
| Code | `code/chase_*.py`; hashes of every frozen script and forecast are in the log |
