# Chase's bystander effect in LLM agent populations — final report

Written 2026-11-16. The study ran from 12 to 15 November 2026 on branch `chase-bystander`. This report summarises
the stage reports (`CHASE_E1_REPORT.md` to `CHASE_E5_REPORT.md`); the pre-registration is
`CHASE_PREREGISTRATION.md` and every change to it is in `CHASE_LOG.md` (amendments C1–C6).

Updated later on 2026-11-16, after the owner supplied Chase's two 1982 papers: section 4 is new, and the
statements about Chase's sequences in the summary and in sections 3, 6, 7 and 8 were revised. Nothing else changed.

Updated 2026-11-19 with the four follow-up experiments registered as amendment C8 (E6 to E9, run 16–19 November;
stage reports `CHASE_E7_REPORT.md` to `CHASE_E9_REPORT.md`, E6 in the log). Section 8 now reports them. The summary
and sections 2, 3, 5, 6, 7 and 9 were revised to match; sections 1 and 4 are unchanged.

## Summary

Chase proposed that animal dominance hierarchies become linear because individuals react to contests they watch,
not only to the ones they fight (the bystander effect). This study asked whether LLM agents do the same and
whether it makes their groups more hierarchical.

- **Single decisions: yes, in most models.** Seven of nine models are less willing to take the lead against a
  partner whom they can see winning against others. The effect is one-sided and often a threshold: agents defer to
  visible winners and do not press visible losers.
- **Groups: yes in Gemma-4-31B, probably in one framing of a second model, no elsewhere.** In Gemma-4-31B, letting
  agents see their partner's record raised the share of strictly ordered groups of four from 36% to 61%, and from
  78% to 95% under a second framing. The second difference did not survive correction at first, and replicated on
  fresh groups (70% to 95%, E9). Qwen3.5-397B showed a similar rise under the competence framing (53% to 80%), but
  not under the second framing. Its run was sized as an estimate, not a test, and its frozen forecast did not
  predict the rise. No effect was found in GLM-5.2, in a mixed-model group, or under an equal-turns framing in either
  model tried there.
- **A real single-decision effect does not imply a group effect.** GLM-5.2 has a statistically clear bystander
  response and no group consequence, and this was forecast before the groups were run.
- **Forecasts made before each population run mostly held.** 22 of 28 arm forecasts in the main study fell inside
  their 90% intervals, and 13 of 14 in the follow-ups. Every miss was in the same direction: more hierarchy than
  forecast.
- **No transitive inference.** No model reasoned "A beat B, B beat me, so defer to A" beyond counting wins.
- **Chase's sequence pattern appears when relationships are counted his way.** His hens formed 87–91% of triads
  through the two sequences that guarantee transitivity. By the study's registered counting rule these agents
  reach at most 70%. By a rule closer to his, in which a relationship forms when one agent yields, Gemma-31B groups
  that see partner records reach 73–91%. This re-count was done after seeing the data. A registered test on fresh
  Gemma-31B groups confirmed it (E9): partner records shift formation toward double dominance in both framings,
  all four predictions supported. In Qwen3.5-397B the sequences stay near chance.
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

Follow-ups (amendment C8, cap raised to 78.06 credits):

| Stage | Purpose | Size |
|---|---|---|
| E6 | Why own-record forecasts were too low: model selection on existing decisions (A), calibration in rounds (B) | 3,000 calls |
| E7 | Second model in groups: Qwen3.5-397B, `norm100` and `comp`, with its own calibration and frozen forecast | 6,075 calibration calls; 272 groups, 33,436 calls |
| E8 | Equal turns, larger: Gemma-26B 140 groups per arm, Qwen3.5-397B 80 per arm | 440 groups, 43,252 calls |
| E9 | Registered sequence test: fresh Gemma-31B groups, `comp` and `norm100` | 200 groups, 19,862 calls |

About 106,000 further calls and 912 groups, 18.2 credits. This study's spend after E7 is 52.9 of
78.06; since 18 November the shared ledger file also records the Rosen study's calls on the same account.

## 3. Results by hypothesis

| | Hypothesis | Verdict |
|---|---|---|
| H1 | Agents lead less against a partner with a better record against others | Supported: 15 of 24 cells, 7 of 9 models |
| H2 | Transitive inference through a shared third party | Not supported in any cell |
| H3 | Groups that see partner records are more linear | Supported in Gemma-31B `comp`; `norm100` replicated in E9; Qwen3.5-397B `comp` +0.28 (estimate), `norm100` none; absent in GLM-5.2 |
| H4 | The difference is forecast by the bystander features of the decision model | Supported for Gemma-31B (no formal test was planned); missed for Qwen3.5-397B `comp` |
| H5 | Under equal turns the effect reverses or vanishes | Unresolved in E5; "vanishes" by the same rule in E8, for Gemma-26B and Qwen3.5-397B |
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
- **By the registered rule, Chase's sequences appear in one cell only.** Under `norm100`, DD + DS rises from 0.53
  to 0.70 with partner records; under `comp` it does not move (0.52 to 0.54). The registered rule takes a pair's
  first contest as the start of their relationship, and in these cells that contest is usually a coin flip.
  Section 4 re-counts with a rule closer to Chase's and finds the rise in both cells.
- **A visible score changes the picture.** Under `norm100` a score board alone gives full transitivity and the log
  adds nothing. Under `comp` the score board gives almost none (0.12), and adding it to the log removes the log's
  effect (−0.01 against 0.60).
- **Follow-ups (section 8):** on 40 fresh groups per arm (E9) Gemma-31B gave 0.70 → 0.95 under `norm100` (p = 0.006)
  and 0.42 → 0.53 under `comp` (p = 0.50). Qwen3.5-397B (E7, 40 per arm, an estimate) gave 0.53 → 0.80 under `comp`
  (p = 0.017 unadjusted) and 0.42 → 0.42 under `norm100`.

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

**E8 follow-up (section 8):** with 140 fresh groups per arm, Gemma-26B gave 0.39 and 0.41 (+0.03 [−0.09, +0.14]);
Qwen3.5-397B with 80 per arm gave 0.31 and 0.30. By the same rule H5 now reads "vanishes" in both. The
decided-contest mechanism above replicated as a registered co-primary (0.39 → 0.29, p = 2 × 10⁻²¹).

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

## 4. Comparison with Chase's 1982 chicken data (post hoc)

The pre-registration could not verify the figures in Chase's 1982 papers and used none of them. Both papers
(*Science* 216: 439–440; *Behaviour* 80: 218–240) were read on 16 November, after all data and after the first
version of this report. Everything in this section is post hoc.

### What Chase reports

| White Leghorn hens | Triads counted | DD | DS | BDID | ISDB | DD + DS |
|---|---|---|---|---|---|---|
| Groups of three (24 groups, 4 hours each) | 23 | 74% | 17% | 4% | 4% | 91% |
| Groups of four, day 1 (14 groups, 6 hours a day) | 55 | 60% | 27% | 6% | 7% | 87% |
| Groups of four, day 2 | 56 | 48% | 38% | 4% | 11% | 86% |

- **Reference value:** if each pair's direction is random, each sequence has probability 0.25, so DD + DS is
  0.50. This is the reference the study used.
- **Linearity:** all 14 groups of four ended linear (13 complete, one guaranteed whatever its missing relationship).
- **Hens that met as a group against hens that met only in pairs:** all 54 complete triads within a group were
  transitive. Of 63 triads put together from hens that had met only as pairs inside different groups, 54 (86%)
  were transitive. Chance is 75%.
- **When a relationship counts as formed:** one hen delivers three aggressive acts and the other does not attack
  back for 30 minutes. Only 2 of 52 relationships formed this way were later reversed.
- **His explanation in 1982:** winners tend to keep winning and losers to keep losing, and with a third hen present
  both tendencies act at once. The papers do not claim that the third hen learns by watching; the 2002 paper names
  that as a candidate.
- The planning note that the pre-registration withdrew as unverified ("about 90%") was correct.

### The study's sequences beside his

The registered rule here takes a pair's first contest as the start of their relationship. That contest is a coin
flip whenever both agents give the same answer, which is most of the time in several cells, so the rule is much
noisier than Chase's. Two alternatives were written into `code/chase_seq_posthoc.py` before it was run, and all
three are shown:
- **Yielded:** the relationship starts at the first contest in which one agent leads and the other defers.
- **Three in a row:** it starts when one agent has won three consecutive contests against the other, coin flips
  included.

DD + DS share in groups of four (chance 0.50; brackets are 95% group-bootstrap intervals):

| Cell | Arm | Registered rule | Yielded | Three in a row | DD / DS under "yielded" |
|---|---|---|---|---|---|
| Gemma-31B `norm100` | DYAD | 0.55 | 0.47 [0.36, 0.58] | 0.58 | 0.29 / 0.18 |
| | OWN | 0.53 | 0.81 [0.73, 0.89] | 0.65 | 0.14 / 0.67 |
| | LOG | 0.70 | 0.91 [0.86, 0.96] | 0.81 | 0.55 / 0.36 |
| Gemma-31B `comp` | OWN | 0.52 | 0.53 [0.47, 0.59] | 0.50 | 0.19 / 0.34 |
| | LOG | 0.54 | 0.73 [0.67, 0.78] | 0.66 | 0.50 / 0.22 |
| GLM-5.2 `comp` | OWN | 0.56 | 0.62 [0.55, 0.68] | 0.56 | 0.21 / 0.41 |
| | LOG | 0.59 | 0.65 [0.59, 0.71] | 0.65 | 0.30 / 0.35 |
| Mixed group (E4) | OWN | 0.49 | 0.72 [0.64, 0.80] | 0.70 | 0.17 / 0.55 |
| | LOG | 0.52 | 0.71 [0.62, 0.79] | 0.63 | 0.18 / 0.53 |
| Gemma-26B `norm0` | OWN | 0.44 | 0.50 [0.41, 0.60] | 0.47 | 0.24 / 0.26 |
| | LOG | 0.54 | 0.54 [0.47, 0.61] | 0.44 | 0.27 / 0.27 |

- **With partner records, Gemma-31B reaches Chase's range.** Under the yielded rule DD + DS is 0.91 in `norm100`
  and 0.73 in `comp`; in groups of eight it is 0.81 in both. His hens gave 0.87 to 0.91.
- **The earlier conclusion for `comp` does not survive the re-count.** The first version of this report said that
  under `comp` linearity rose with no change in sequences. Under the yielded rule DD + DS rises from 0.53 to 0.73
  in groups of four and from 0.55 to 0.81 in groups of eight. The three-in-a-row rule agrees (0.50 to 0.66, and
  0.49 to 0.71).
- **Partner records produce double dominance; own records produce double subordinance.** With partner records the
  commonest sequence is DD, as in the hens: a bystander yields to an agent it has seen win. With own records only,
  under `norm100`, DD + DS is also high (0.81) but two thirds of triads are DS: an agent that has lost yields to
  the next one it meets. The placebo arm looks like OWN (0.83, mostly DS).
- **Winning does not make these agents win more.** DD needs the bystander to give way, which it does only when it
  can see the winner's record. Gemma-31B already leads at ceiling, so a win cannot make it lead more.
- **With direct history only, Gemma-31B's sequences are at chance** (DYAD, 0.47 and 0.53), as they should be.
- **Elsewhere partner records change nothing.** GLM-5.2 shows a modest excess in both arms (0.62 and 0.65) and
  equal turns none. DeepSeek never yields, so the yielded rule forms no relationships at all there.
- **The mixed group shows many DS sequences in both arms.** Fixed differences between members would do this: a
  member that defers more than the others is dominated by each of them in turn.

Transitive share of complete triads at the end of the run (chance 0.75):

| | Met only in pairs | Own record | Partner record added |
|---|---|---|---|
| Chase's hens | 0.86 (constructed triads) | | 1.00 (real triads) |
| Gemma-31B `norm100`, groups of four | 0.80 (DYAD) | 0.94 | 0.99 |
| Gemma-31B `comp`, groups of four | 0.81 (DYAD) | 0.76 | 0.87 |
| GLM-5.2 `comp`, groups of four | not run | 0.81 | 0.84 |
| Gemma-26B `norm0`, groups of four | 0.80 (DYAD) | 0.79 | 0.79 |

The correspondence between his conditions and these arms is loose: his hens in constructed triads may have carried
their own experience from one pairing to the next, which would be closer to OWN than to DYAD.

### Cautions

- **Three rules were tried and the one closest to Chase's criterion gives the clearest pattern.** That is a choice
  made after seeing data. The registered measure remains the first-contest rule.
- **The yielded rule leaves triads out.** Under `comp` it forms two relationships in 220 of 320 triads with own
  records and 250 of 320 with partner records (237 and 586 of 672 in groups of eight), so the arms are compared on
  different subsets.
- **Here these sequences do not guarantee transitivity.** In the hens a formed relationship almost never reversed.
  In this protocol later coin flips can overturn it: Gemma-31B `comp` with a visible score has DD + DS of 0.81
  under the yielded rule and almost no transitivity at the end (0.12).
- **Intervals are unadjusted** and no test was planned.

## 5. How well the forecasts did

| | Inside the frozen 90% interval | Misses |
|---|---|---|
| Arm forecasts, E3 | 17 of 22 | 5, all toward more hierarchy |
| Arm forecasts, E5 | 5 of 6 | 1, toward more hierarchy |
| LOG − OWN contrasts, E3 and E5 | 7 of 9 | 2 |
| Arm forecasts frozen for the follow-ups, E7 and E8b | 13 of 14 | 1 (Qwen `comp` LOG, toward more hierarchy) |
| LOG − OWN contrasts, E7 and E8b | 4 of 5 | 1 (Qwen `comp`, groups of four) |
| Fresh E9 arms against the E2 forecasts | 6 of 6 (one on the edge) | 0 |

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
- **After the follow-ups (E6, E8):** fitting the decision model on the records groups actually reach (calibration
  in rounds) closes the Gemma-31B `norm100` gap: 0.73 [0.60, 0.85] against 0.78 observed. It does not close the
  Gemma-26B `norm0` gap (0.30 [0.17, 0.45] against 0.50). On 140 fresh groups that arm gave 0.39, so E5's 0.50 was
  partly a high draw, but the forecasts are still low there.
- **The new miss is the other way round.** For Qwen3.5-397B `comp` the forecast put both arms near 0.60. Own-record
  groups came in at 0.53 and partner-record groups at 0.80: this time the bystander arm was underestimated.

## 6. What this says about Chase's account

- **The ingredient exists.** Most LLMs tested show a bystander effect on single decisions, with the predicted sign.
- **It is sufficient for hierarchy in at least one model.** In Gemma-31B, with scores hidden, adding only the
  partner's third-party record moves groups from chance toward linearity. Chase's 2002 fish went from 56% linear
  when meeting in isolated pairs to 92% when assembled as a group; here the analogous arms gave 36% and 61%
  (`comp`), and 78% and 95% (`norm100`). Qwen3.5-397B may be a second case, under `comp` only (53% and 80%, an
  estimate not yet replicated, with an ambiguous placebo).
- **It is not necessary.** Own-record groups under `norm100` reach 78% linear with no bystander information. The
  post hoc diagnosis attributes this to a loser effect that only appears at long losing records.
- **It does not always matter.** A small bystander slope (GLM-5.2) has no group consequence, and under equal turns
  the response exists in decisions without producing structure. Qwen3.5-397B has nearly the same slope in `comp`
  and `norm100`, and a group effect in only one of them. The single-decision slope does not predict the group
  outcome on its own.
- **Whether it helps depends on how often agents lead.** Partner records lower the LEAD rate in every cell measured.
  Where agents lead more than half the time this leaves more contests that the records decide, and hierarchy
  rises (Gemma-31B, Qwen `comp`). Where they lead less than half the time it leaves more coin flips, and nothing
  is gained (Gemma-26B and Qwen under equal turns, Qwen `norm100`). This was registered only for Gemma-26B
  under equal turns (E8a, supported); the rest is descriptive.
- **Chase's sequences are there once relationships are counted his way.** By the registered rule they show under
  one framing only. Counted from the first contest in which one agent yields, partner records raise DD + DS in
  both Gemma-31B cells to between 0.73 and 0.91, against 0.87 to 0.91 in his hens (section 4, post hoc). The
  registered test on fresh groups (E9) confirmed the shift toward double dominance. In Qwen3.5-397B the sequences
  stay near chance even where its hierarchy rises, so in that model linearity grows without Chase's pattern.
- **The route to double dominance differs from the one he described.** He attributed the sequences to winners
  continuing to win and losers continuing to lose. Here double subordinance does arise from losers giving way,
  but double dominance comes from bystanders giving way to winners they can see, not from winners pressing on.
- **These hierarchies are not stable properties of the group.** Chase's fish reproduced the same hierarchy in 27%
  of groups and were linear in over 90%; the mixed LLM group did so 2–3% of the time and was linear in 50–60%.

## 7. Limits and deviations

- **One model carries the tested population result.** Population tests covered Gemma-31B (two framings, twice),
  GLM-5.2 (one framing, groups of four) and Gemma-26B (equal turns only). Qwen3.5-397B was run in three framings,
  but under `comp` and `norm100` only as estimates. Haiku, Sonnet and GPT-6-Luna were never run as groups.
- **The follow-up formation rule was chosen after seeing E3.** E9 fixed it before its own data, so E9 is a fair test,
  but the rule was picked from three tried on earlier data.
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
- **Chase's 1982 figures were verified only after the study,** from papers the owner supplied on 16 November. No
  decision before that date used them. The 2002 figures are quoted as verified in the pre-registration.
## 8. The open questions, followed up (E6–E9)

The first version of this report listed four open questions. Amendment C8 registered one experiment for each;
all four have now run.

- **A second responsive model (E7, Qwen3.5-397B).** Under `norm100` partner records change nothing (0.42 and 0.42),
  as the frozen forecast said. Under `comp` groups of four go from 0.53 to 0.80 linear (+0.28 [+0.07, +0.47]) and
  groups of eight from 0.11 to 0.33 t_tri. The forecast had expected no difference, so the run was sized as an
  estimate, and the p values (0.017 and 0.015) are unadjusted. The placebo arm (0.70) lies between the two, so part
  of the rise may come from seeing that the partner has a history at all. `CHASE_E7_REPORT.md`.
- **Calibration that reaches deep records (E6).** Part A compared decision models on the existing own-record
  decisions; the form with every added feature (A5: a second slope for deep losing records, number of own lines
  shown, recency and run length, agent IDs) fitted best. Part B
  re-calibrated in rounds on the records simulated groups visit. That closes the Gemma-31B `norm100` miss (0.73
  [0.60, 0.85] against 0.78) but not the Gemma-26B `norm0` one (0.30 [0.17, 0.45] against 0.50). The registered
  pass rule (both covered) was not met. Log entries of 16–17 November.
- **A larger equal-turns test (E8).** 140 fresh groups per arm for Gemma-26B and 80 for Qwen3.5-397B: LOG − OWN
  +0.03 [−0.09, +0.14] and −0.01 [−0.15, +0.14]. H5 reads "vanishes" in both. With Qwen's `comp` result this now
  has a contrast within one model: an effect under `comp`, none under equal turns. Gemma-26B still has no
  framing in which the effect was shown. `CHASE_E8_REPORT.md`.
- **A registered sequence test (E9).** Fresh Gemma-31B groups, scored by the yielded rule fixed in advance. All four
  primary predictions hold (Holm p 0.003 or below): partner records raise DD + DS under `comp` (0.49 → 0.72) and
  move triads from DS to DD under `norm100` (DD 0.18 → 0.59). `CHASE_E9_REPORT.md`.

Still open:
- A registered replication of Qwen3.5-397B `comp`, with more placebo groups. At the observed shares about 52
  groups per arm give 80% power; a smaller effect on replication would need more.
- Why the Gemma-26B equal-turns own-record arm stays above every forecast.
- Whether the LEAD-rate account (section 6) predicts which cells gain from partner records. It could be registered
  as a rule on calibration data before the next population run.

"no effect" from the effect seen elsewhere.
- **A registered sequence test.** The re-count in section 4 was chosen after seeing the data. New groups scored
  by the yielded rule, fixed in advance, would test it properly.

## 9. Files

| | |
|---|---|
| Plan and changes | `refine-logs/CHASE_PREREGISTRATION.md`, `refine-logs/CHASE_LOG.md` |
| Stage reports | `refine-logs/CHASE_E1_REPORT.md` … `CHASE_E5_REPORT.md`, `CHASE_E7_REPORT.md` … `CHASE_E9_REPORT.md` |
| Single decisions | `results/chase_e1_kernel.jsonl`, `chase_e1_cells.csv`, `chase_e1_summary.json` |
| Frozen forecasts | `results/chase_e2_predictions.json`, `results/chase_e5_predictions.json` |
| Populations | `results/chase_e3_pop.jsonl` (E3 and E5), `results/chase_e4_rewind.jsonl` |
| Evaluations | `results/chase_e3_eval.json`, `chase_e4_eval.json`, `chase_e5_eval.json` |
| Sequence re-count (post hoc) | `code/chase_seq_posthoc.py` → `results/chase_seq_posthoc.json` |
| Follow-ups (C8) | reports `CHASE_E7_REPORT.md` … `CHASE_E9_REPORT.md`; forecasts `results/chase_e6_predictions.json` … `chase_e8_predictions.json`; populations `chase_e7_pop.jsonl`, `chase_e8_pop.jsonl`, `chase_e9_pop.jsonl`; evaluations `code/chase_e7_eval.py` … `chase_e9_eval.py` → `results/chase_e7_eval.json` … `chase_e9_eval.json`; E6 part A `results/chase_e6a.json` |
| Code | `code/chase_*.py`; hashes of every frozen script and forecast are in the log |
