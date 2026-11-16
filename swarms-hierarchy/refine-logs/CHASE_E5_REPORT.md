# Chase study — E5 report (equal-turns framing: does the bystander effect reverse or vanish?)

Written 2026-11-15 23:30, after E5 finished (15 Nov 16:48–23:02, exit 0, no failed calls). Exploratory; plan and
reading fixed beforehand in amendment C6 of `CHASE_LOG.md`.

- Cell: Gemma-4-26B-A4B under `norm0` (the group is told that turns should be shared equally). In E1 this cell had
  the largest partner-record effect on a single decision of any `norm0` cell (β_p −0.108 ± 0.009).
- Calibration: 2,625 calls in two passes → `results/chase_e5_statecal.jsonl`.
- Forecast: `results/chase_e5_predictions.json`, stamp `ce785e89…fa51eec`, frozen at 17:43, before the first
  population call (17:43:51); the stamp was written into the log by the queue script.
- Populations: 132 groups, 15,270 decisions, 0 invalid replies, appended to `results/chase_e3_pop.jsonl`
  (now 640 lines, sha256 `6434d9a7…5835d0`; the first 508 lines are E3 and still hash to `5967fb05…`).
- Evaluation: `code/chase_e5_eval.py` (`73d07343…`), which runs the frozen `chase_e3_eval.py` (`3e16161b…`)
  unchanged → `results/chase_e5_eval.json`.
- Cost: 17,895 calls, 0.66 credits (34.67 spent in total, cap 78.06).

## Result on H5

H5 (pre-registration section 2): under `norm0` the LOG − OWN difference reverses or vanishes.

| | OWN | LOG | LOG − OWN [95% CI] | p |
|---|---|---|---|---|
| Groups of four, linear share (chance 0.375) | 0.50 (20/40) | 0.47 (19/40) | −0.03 [−0.25, +0.20] | 1.0 (Fisher) |
| Groups of eight, t_tri (chance 0) | −0.04 | −0.04 | +0.01 [−0.20, +0.21] | 1.0 (Mann–Whitney, 6 seeds) |

By the reading fixed in amendment C6:
- **Not "reverses":** the difference is below zero but its interval includes zero.
- **Not "contradicted":** the difference is not above zero.
- **Not counted as "vanishes" either:** the interval reaches +0.20 and so does not exclude +0.17, the difference
  seen in Gemma-31B `norm100`.

H5 is therefore **unresolved by the pre-set rule**. No bystander effect on hierarchy was detected in this cell; 40
groups per arm cannot rule out one of the size seen elsewhere. The frozen forecast itself gave the test 18% power
for the difference it expected (+0.13).

## Is there a hierarchy at all in this cell?

Barely. No arm of four differs from the chance share of linear groups (0.375) by a binomial test (post hoc;
smallest p 0.10), and at N = 8 transitivity is at chance in both arms.

| Arm | Groups | Linear | Forecast [90%] | Inside | DD + DS |
|---|---|---|---|---|---|
| OWN, N = 4 | 40 | 0.50 | 0.24 [0.12, 0.35] | no | 0.44 |
| LOG, N = 4 | 40 | 0.47 | 0.38 [0.25, 0.50] | yes | 0.54 |
| DYAD, N = 4 | 20 | 0.50 | 0.37 [0.20, 0.55] | yes | 0.57 |
| REDACT, N = 4 | 20 | 0.20 | 0.23 [0.05, 0.40] | yes | 0.55 |
| OWN, N = 8 (t_tri) | 6 | −0.04 | −0.11 [−0.25, +0.02] | yes | 0.51 |
| LOG, N = 8 (t_tri) | 6 | −0.04 | +0.02 [−0.13, +0.20] | yes | 0.49 |

- Forecast coverage: 5 of 6 arms and both contrasts (frozen +0.13 [−0.05, +0.33] and +0.13 [−0.05, +0.34]).
- The OWN forecast for groups of four is too low again, as in both Gemma-31B cells of E3, and so is the
  leave-one-seed-out upper bound (0.32 [0.20, 0.45]). The second calibration pass did not remove this.
- Sequences: at N = 4, DD + DS is 0.44 under OWN and 0.54 under LOG (BDID 0.36 → 0.22); at N = 8 there is no
  difference. No test was planned on this.

## Why so little structure (post hoc)

Under equal turns the agents mostly defer, and a contest in which both give the same answer is settled by a coin
flip.

| Cell, groups of four | LEAD rate: OWN | LOG | Contests decided by the answers: OWN | LOG | LOG − OWN [95% CI] |
|---|---|---|---|---|---|
| Gemma-26B `norm0` | 0.33 | 0.28 | 0.39 | 0.29 | −0.10 [−0.13, −0.07] |
| Gemma-31B `norm100` | 0.79 | 0.66 | 0.40 | 0.68 | +0.27 [+0.22, +0.32] |
| Gemma-31B `comp` | 0.89 | 0.86 | 0.21 | 0.27 | +0.07 [+0.03, +0.10] |

- In all three cells the partner's record lowers the LEAD rate.
- Where agents lead most of the time, leading less often means more contests with one LEAD and one DEFER, so more
  outcomes are decided by the records. Where they already defer most of the time (`norm0`), leading less often
  means more double-DEFER contests, so more outcomes are coin flips (58% both-DEFER under LOG, 48% under OWN).
- The same holds at N = 8: decided contests 0.39 → 0.28 under `norm0`, 0.28 → 0.71 under `norm100`,
  0.13 → 0.37 under `comp`.
- This fits H5's stated mechanism (the log amplifies the prevailing norm) at the level of decisions. It did not
  show up as a measurable drop in linearity. These comparisons were not planned and are not adjusted.

## The placebo arm

REDACT (the partner's record with outcomes removed) gave 4 linear groups of 20, against 20 of 40 under OWN
(Fisher p = 0.029, post hoc, one of several possible comparisons). Its forecast covered it. In E1 the placebo
contrast for this cell was not null (R = −0.229), so the redacted record does change this model's decisions; why
that would lower linearity below OWN and DYAD is not explained, and with 20 groups it may be noise.

## Limits

- One model, one cell, no control cell, unadjusted. Not part of E3's Holm family.
- The forecast procedure differs from E3 (second calibration pass), and the owner did not see the forecast before
  the populations started; the queue froze and stamped it automatically.
- With 60–73% of contests settled by coin flips, the linear share in this cell is a noisy measure of anything the
  agents do.
- In one N = 8 OWN group one pair never met (5 of 6 groups complete for the transitivity count).
