# E10 — registered replication of the Qwen3.5-397B `comp` result (amendment C9)

Qwen3.5-397B × `comp`, groups of four, 24 rounds, fresh seeds 44000–44079. Plan A, fixed by the queue from the
credit headroom before the first call: OWN 80, LOG 80, REDACT 80. 240 groups, 22,688 calls, 6.56 credits, run
19 Nov 16:19 – 20 Nov 00:13, exit 0, no failed calls, no invalid answers. No other job called the account during
the run. Evaluation `code/chase_e10_eval.py` (sha256 cb7d65d1…4c25fe1, hashed in C9 before any E10 call) →
`results/chase_e10_eval.json`.

## Summary

The E7 estimate replicates. Groups that see their partners' records are 0.81 linear, against 0.51 for groups that
see only their own record: +0.30, one-sided Fisher p = 5 × 10⁻⁵. The registered placebo settles the question E7
left open. Partner lines with the outcomes removed give exactly the own-record level (0.51), so the gain comes
from who won, not from the extra text. Qwen3.5-397B is the second model, after Gemma-4-31B, in which bystander
information makes groups more hierarchical, and here too only under `comp`.

## 1. Results

| Arm | Groups | Linear | Share |
|---|---|---|---|
| OWN | 80 | 41 | 0.51 |
| REDACT | 80 | 41 | 0.51 |
| LOG | 80 | 65 | 0.81 |

| | Prediction | Result | Reading |
|---|---|---|---|
| P1 (primary) | LOG > OWN | +0.30 [+0.16, +0.44], one-sided Fisher p = 5.0 × 10⁻⁵ | **supported** |
| S1 | LOG − REDACT | +0.30 [+0.16, +0.44], p = 1.0 × 10⁻⁴ | **"outcomes"** |
| | REDACT − OWN | +0.00 [−0.15, +0.15], p = 1.00 | |
| S2 | LEAD rate LOG < OWN | 0.62 → 0.50, p = 4 × 10⁻²³ | **holds** |
| | Decided contests LOG > OWN | 0.49 → 0.52, p = 0.033 | |
| S3 | DD share LOG > OWN (E9 rule) | 0.28 → 0.31, p = 0.30 | direction right, not significant |

S1 had about 25% registered power at the E7 shares. It resolved because REDACT came in lower than in E7: 0.51
here, against 0.70 on 20 groups there.

## 2. Pooled with E7 (descriptive)

120 groups per arm (REDACT 100): OWN 0.52, LOG 0.81, +0.29 [+0.18, +0.40]; REDACT 0.55, so LOG − REDACT +0.26
[+0.14, +0.38] and REDACT − OWN +0.03 [−0.09, +0.17]. DD share 0.27 → 0.33 (p = 0.08). DD + DS 0.56 → 0.58.

Against the frozen E7 forecast at 80 groups: OWN 0.51, inside 0.60 [0.50, 0.71]; LOG 0.81, outside 0.62
[0.51, 0.74]. As in E7, the decision model fitted to calibration rows misses the size of the bystander gain in this
cell.

## 3. Descriptive, not registered: what the placebo shows about the mechanism

| Arm | LEAD rate | Decided contests |
|---|---|---|
| OWN | 0.62 | 0.49 |
| REDACT | 0.59 | 0.53 |
| LOG | 0.50 | 0.52 |

Partner lines without outcomes also lower the LEAD rate slightly and raise the share of decided contests to the
LOG level. Yet REDACT groups end no more linear than OWN groups. So a larger number of decided contests is not
enough on its own. What matters is that agents defer to the partners they can see winning, so that decided contests
line up into one order. The LEAD-rate account of the final report (S2, which holds) describes when the bystander
response has room to act. It is not the whole mechanism.

## 4. What this says

- **The second-model result is now a registered finding.** In Qwen3.5-397B under `comp`, bystander information
  raises the linear share from about one half to about four fifths, by a margin close to Gemma-4-31B's (+0.25 in E3).
- **It is information about outcomes.** The placebo arm equals the own-record arm, the same pattern as Gemma-31B in
  E3 (REDACT level with OWN).
- **It is not built through Chase's sequences in this model.** The DD share barely moves, and DD + DS stays near
  the 0.50 of random formation. Gemma-31B reached 0.72–0.94 with partner records (E9). Qwen hierarchies become
  linear with partner records, but not by the order of formation Chase described. Later contests, guided by the
  visible records, must do the ordering.
- **The framing dependence holds in both models.** Qwen showed no effect under `norm100` (E7) and Gemma-26B none
  under `norm0` (E5, E8). Only Gemma-31B has effects under two framings.

Limits: one model and one framing; groups of four only; the effect size in this cell was found in E7 and is
confirmed here, but the forecasting model still does not predict it.
