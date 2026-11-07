# Experiment Tracker

| ID | Block | Config | Status | Calls | Result file | Notes |
|---|---|---|---|---|---|---|
| M0 | theory | mean-field + stochastic surrogate | DONE | 0 | — | ρ_c analytic match; stochastic crossover |
| M1a | pilot pairwise | haiku/sonnet/luna × reward/competence | DONE | 864 | results/pairwise_pilot.jsonl | η sign flip (haiku) |
| M1b | pilot population | haiku comp, haiku reward, sonnet comp; N=8 ρ=.5 | DONE | ~950 | results/population_pilot.jsonl | all within ~1.5 SD of surrogate |
| M2 | calibration v2 | haiku 9 cond; sonnet 5; luna 3 | RUNNING | 3,540 | results/calib_v2.jsonl | |
| M3 | inequality probe | haiku ended+comp, norm50 | RUNNING | 192 | results/calib_v2.jsonl (exp=probe) | |
| M4 | predictions | — | TODO | 0 | results/predictions_v2.json | write-once |
| M5 | holdout | — | TODO | 96 | results/holdout.jsonl | |
| M6 | populations | haiku × {ended, ended+comp, norm50} × 4 seeds | TODO | ~3,840 | results/population_v2.jsonl | |
| M7 | evaluation | — | TODO | 0 | results/eval_v2.json | |
| M8 | replication | sonnet / luna | OPTIONAL | ≤1,600 | results/population_v2.jsonl | |
