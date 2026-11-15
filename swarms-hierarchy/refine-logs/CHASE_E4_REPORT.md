# Chase study — E4 report (rewind: does the same group form the same hierarchy again?)

Written 2026-11-15, after E4 finished (14 Nov 16:10–19:25, 80 groups, exit 0). Evaluation:
`code/chase_e4_eval.py` (sha256 `d7c05405…ecf254d`, logged before any E4 group existed) →
`results/chase_e4_eval.json`. Data: `results/chase_e4_rewind.jsonl`.

Design, as pre-registered (section 8): one fixed group of four different models — Gemma-31B, GLM-5.2, Qwen3.8-27B,
DeepSeek-v4-flash — re-run 40 times per arm with fresh names, pairings and coin flips; framing `comp`; 24 rounds.
OWN shows each agent only its own record; LOG adds the partner's record against third parties.

## Result

| | OWN | LOG | LOG − OWN [95% CI] | Fisher p |
|---|---|---|---|---|
| Linear groups | 0.50 (20/40) | 0.60 (24/40) | +0.10 [−0.12, +0.32] | 0.50 |
| Re-runs matching the most common hierarchy | 0.10 (4/40) | 0.12 (5/40) | +0.02 [−0.12, +0.17] | 1.0 |
| Two re-runs identical | 0.03 [0.01, 0.05] | 0.02 [0.01, 0.05] | −0.01 [−0.03, +0.03] | |
| Distinct outcomes in 40 re-runs | 26 | 29 | | |
| DD + DS sequences | 0.49 | 0.52 | | |

H6 had three parts:
- **Reproducibility far below 100%: supported.** Two re-runs of the same group end in the same hierarchy 2–3% of
  the time. If rank order among linear groups were random this would be 4%.
- **Reproducibility not higher under LOG: consistent** (difference −0.01 [−0.03, +0.03]).
- **Linearity higher under LOG: not supported.** The direction is as predicted, the difference is not
  distinguishable from zero with 40 re-runs per arm.

The pre-registration quotes Chase's 2002 fish figures as 27.3% identical and over 90% linear. These groups are
less linear and far less reproducible than that.

## Who ends up where

Mean number of other members each model dominates at the end (0–3), and how often it is at the top of a linear
hierarchy:

| Model | Dominates: OWN | LOG | Top: OWN | LOG |
|---|---|---|---|---|
| DeepSeek-v4-flash | 2.00 | 2.10 | 0.20 | 0.33 |
| Gemma-31B | 1.60 | 1.38 | 0.12 | 0.10 |
| GLM-5.2 | 1.52 | 1.62 | 0.15 | 0.15 |
| Qwen3.8-27B | 0.88 | 0.90 | 0.03 | 0.03 |

There is a tendency in rank (DeepSeek high, Qwen3.8-27B low in both arms) without the full hierarchy repeating.
No test was pre-registered for rank by model.

## Reading (post hoc)

- E1 batch C, run after E4, found that Qwen3.8-27B does not use the partner's third-party record, and DeepSeek
  never did. Two of the four members therefore could not produce a bystander effect, which may be why LOG adds
  little here. This was not known when E4 was designed and is not tested by it.
- DeepSeek leads almost unconditionally in E1 and Qwen3.8-27B has the lowest LEAD rate of the four under `comp`
  (0.67 overall), which matches their positions. Rank here looks like a consequence of each model's baseline
  willingness to lead more than of its record.

## Limits

- One group composition, one framing. Chase's design re-assembled many different groups twice; this re-runs one
  group many times, so "two re-runs identical" is the nearest analogue, not the same statistic.
- Invalid replies became coin flips: 1 of 3,872 decisions under OWN, 34 of 3,850 under LOG.
- With 40 re-runs per arm the test has about 75% power for a difference in linear share of 0.30 from a base of
  0.50, and under 40% for 0.20 (simulated after the fact).
