# E7 — a second model in groups: Qwen3.5-397B (amendment C8)

Qwen3.5-397B × `norm100`, `comp`. Groups of four: OWN 40, LOG 40, DYAD 20, REDACT 20 (seeds 24000+); groups of
eight: OWN 8, LOG 8 (seeds 28000+). 272 groups, 33,436 calls, 9.69 credits, run 19 Nov 01:38–13:15, exit 0, no
failed calls, no invalid answers. No other job called the account during the run. Evaluation `code/chase_e7_eval.py`
(sha256 86a294c9…1b04c51, hashed in the log before any E7 group existed) → `results/chase_e7_eval.json`.

**Status of the numbers.** The frozen forecast put LOG − OWN near zero in both framings, which gave the Fisher test
13–14% power even at 120 groups per arm. By the C8 sizing rule both cells were therefore run at 40 per arm as
**estimates, not tests**. All p values below are unadjusted, and none of them confirms a hypothesis.

## Summary

- **`norm100`: no bystander effect on linearity.** Both arms are 0.42 linear. The frozen forecast said this and
  covers every arm.
- **`comp`: a large difference the forecast did not expect.** LOG groups of four are 0.80 linear against 0.53 for
  OWN (+0.28 [+0.07, +0.47], p = 0.017). Groups of eight go the same way: t_tri 0.11 → 0.33 (+0.22 [+0.05, +0.38],
  p = 0.015). The LOG arm and the N4 contrast both fall outside their frozen intervals. This is the first sign of
  the Gemma-31B `comp` result in a second model, but it is an unplanned estimate and needs a registered replication.
- **The placebo is ambiguous.** REDACT (others' lines with their outcomes removed) gave 0.70 under `comp`, between
  OWN and LOG and inside its forecast. With 20 groups it separates from neither (p = 0.27 against OWN, 0.52 against
  LOG). E1 had already found that partner lines without outcomes lower this model's LEAD rate.
- **Sequences move in the predicted direction under `comp` only.** DD share 0.26 → 0.36 (p = 0.11). Under `norm100`
  it is 0.28 → 0.23.

## 1. Arms against the frozen forecast

Linear share in groups of four, t_tri in groups of eight. Frozen forecast: mean and 90% interval, form X (the
headline; DYAD has only K). "Without bystander" is the LOG forecast with the partner-record features removed
(form Xnb).

| Framing | Arm | Groups | Observed | Forecast X | Inside | Without bystander |
|---|---|---|---|---|---|---|
| `norm100` | OWN, N4 | 40 | 0.42 (17) | 0.43 [0.30, 0.55] | yes | |
| | LOG, N4 | 40 | 0.42 (17) | 0.43 [0.30, 0.57] | yes | 0.38 [0.25, 0.50], inside |
| | DYAD, N4 | 20 | 0.25 (5) | 0.38 [0.20, 0.55] (K) | yes | |
| | REDACT, N4 | 20 | 0.35 (7) | 0.33 [0.15, 0.50] | yes | |
| | OWN, N8 | 8 | −0.01 | 0.02 [−0.14, 0.19] | yes | |
| | LOG, N8 | 8 | 0.02 | 0.13 [−0.05, 0.30] | yes | 0.05 [−0.09, 0.22], inside |
| `comp` | OWN, N4 | 40 | 0.53 (21) | 0.60 [0.47, 0.72] | yes | |
| | LOG, N4 | 40 | **0.80** (32) | 0.62 [0.47, 0.78] | **no** | 0.53 [0.40, 0.68], outside |
| | DYAD, N4 | 20 | 0.35 (7) | 0.36 [0.20, 0.50] (K) | yes | |
| | REDACT, N4 | 20 | 0.70 (14) | 0.62 [0.40, 0.80] | yes | |
| | OWN, N8 | 8 | 0.11 | 0.18 [−0.02, 0.36] | yes | |
| | LOG, N8 | 8 | 0.33 | 0.25 [0.09, 0.44] | yes | 0.10 [−0.06, 0.26], outside |

11 of 12 arms are inside their forecasts. The K form gives the same verdicts. The one miss is LOG under `comp`,
and it is a miss toward more hierarchy, the same direction as every earlier miss.

## 2. LOG − OWN

| Framing | Size | OWN | LOG | LOG − OWN [95%] | p (unadjusted) | Frozen forecast | Inside |
|---|---|---|---|---|---|---|---|
| `norm100` | N4 | 0.42 | 0.42 | +0.00 [−0.22, +0.22] | 1.00 (Fisher) | −0.01 [−0.20, +0.20] | yes |
| | N8 | −0.01 | 0.02 | +0.03 [−0.16, +0.23] | 0.92 (Mann–Whitney) | +0.12 [−0.11, +0.36] | yes |
| `comp` | N4 | 0.53 | 0.80 | **+0.28 [+0.07, +0.47]** | 0.017 (Fisher) | +0.02 [−0.17, +0.23] | **no** |
| | N8 | 0.11 | 0.33 | **+0.22 [+0.05, +0.38]** | 0.015 (Mann–Whitney) | +0.07 [−0.16, +0.28] | yes |

For comparison, Gemma-31B `comp` gave +0.25 in E3 (80 groups per arm) and +0.10 in E9 (40 fresh groups). The rough
expectation C8 recorded from E1 rows alone was +0.30 for `norm100` and +0.15 for `comp`; the outcome is the other
way round.

## 3. Sequences (E9 rule, secondary)

Relationships form at the pair's first contest with one LEAD and one DEFER.

| Framing | Arm | DD + DS | DD | DD share LOG − OWN (per group) | p (Mann–Whitney) |
|---|---|---|---|---|---|
| `norm100` | OWN | 0.55 | 0.28 | | |
| | LOG | 0.51 | 0.23 | −0.05 | 0.48 |
| `comp` | OWN | 0.55 | 0.26 | | |
| | LOG | 0.56 | 0.36 | +0.10 | 0.11 |

In Qwen the DD + DS share stays close to the 0.50 of random formation in every arm (DYAD 0.50–0.54, REDACT 0.60).
Gemma-31B reached 0.72–0.94 with partner records in E9. The `comp` shift toward DD has the predicted direction but
is small.

## 4. Descriptive, not registered: how often agents lead

| Framing | Arm | LEAD rate | Contests decided by the answers |
|---|---|---|---|
| `norm100` | OWN | 0.43 | 0.48 |
| | LOG | 0.37 | 0.42 |
| `comp` | OWN | 0.62 | 0.49 |
| | LOG | 0.48 | 0.54 |

Partner records lower the LEAD rate in both framings. Under `comp`, where agents lead more than half the time, that
leaves more contests with one LEAD and one DEFER, which the records then decide. Under `norm100` the baseline is
below one half and the same lowering leaves more double-DEFER coin flips. This is the pattern seen after the fact in
E5 and in E8 (Gemma-31B against Gemma-26B), now within one model across two framings. It was not registered.

## 5. What this says

- **A second model shows the group effect in one framing.** Under `comp`, Qwen3.5-397B groups that see partner
  records are clearly more linear at both sizes. Under `norm100` they are not. That is the reverse of the order
  expected from E1 rows, where `norm100` looked stronger.
- **The forecast missed it.** The decision model fitted to calibration rows predicted almost no difference. The
  E1 single-decision slope for this model is similar in both framings (β_p −0.082 and −0.077), so the slope alone
  does not explain why only `comp` responds. The LEAD-rate pattern in section 4 is one candidate explanation.
- **Whether it is a bystander effect is open.** REDACT lies between OWN and LOG. Part of the gain may come from
  seeing that the partner has a history at all, not from who won. A run with more REDACT groups would settle this.
- **Sequences do not follow Chase's pattern here.** The rise in linearity under `comp` comes with only a small
  shift toward DD, and DD + DS stays near chance.

Limits: estimates under the sizing rule, not tests; one draw of 40 groups per arm in groups of four and 8 in groups
of eight; the `comp` contrast was unexpected and should be replicated with a size fixed in advance (at the observed
shares, 0.53 and 0.80, the Fisher test has 80% power at about 52 groups per arm, by simulation; a replication should
plan for a smaller effect than the one observed).
