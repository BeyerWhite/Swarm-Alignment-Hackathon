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

## Real-task game v2 (2026-11-08)
| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| TG2-M0 | M0 | sanity | Gemma-31B none/noplan/w100, 8 rounds | seed 7 | throughput, collisions, wrong | MUST | DONE | 4.75 vs 2.0 tasks/round; 0 wrong |
| TG2-M1 | M1 | C1a, C1b, C2 | 5 models × {none, noplan, w0, w100} | seeds 3101–3108 | incumbency, throughput, top_plan_share | MUST | RUNNING | 160 runs, launched 05:02 |
| TG2-M2 | M2 | anti-claim A2 | 5 models × shuffle | seeds 3101–3108 | throughput, collisions | MUST | TODO | needs `shuffle` arm in code |
| TG2-M3 | M3 | evaluation + B5 | — | — | H1/H2/H3 | MUST | TODO | `taskgame2.py eval` |
| TG1 | pilot | v1 (no coordination value) | 5 models × {w0, w100} | 2101–2110 | — | DONE (48 runs) | stopped 05:01 | appendix |
