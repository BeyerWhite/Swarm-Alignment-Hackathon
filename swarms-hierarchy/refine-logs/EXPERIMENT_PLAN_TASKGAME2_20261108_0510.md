# Experiment Plan — Real-task game v2 (coordination pays)

**Problem**: Track 2 asks when leaders, roles and division of labour emerge among agents, and whether this can be
reproduced safely. The contest sandbox shows when agents *claim* a coordinator label; it cannot show that a
coordinator does useful work, and uniform pairing caps any agent at ~2/N of coordination.
**Method Thesis**: When uncoordinated work collides, LLM agents spontaneously split into a coordinator and workers,
the coordinator role persists beyond chance, and coordination raises output; the two-agent status test predicts the
direction in which a framing sentence shifts that persistence.
**Date**: 2026-11-08 (frozen hypotheses: `results/taskgame2_predictions.json`, sha b5beab6d…, written 05:02 PST before any v2 run data)

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1 (primary) Division of labour emerges and is useful | Answers the Track 2 question with real, verified work, not a label | (a) incumbency > its null planners(t−1)/N in arm `none`, sign test over runs, p < 0.05 pooled; (b) throughput(none) > throughput(noplan) in ≥4/5 models, permutation p < 0.05 pooled; collisions lower | B1, B2, B4 |
| C2 (supporting) Two-agent test predicts framing effects on persistence | Links the cheap screen to real-task leadership | sign of [w100 − w0] in incumbency and top_plan_share matches sign(Δs) in ≥4/5 models, incl. the reversed Gemma-26B prediction | B3 |
| Anti-claim A1 | "Persistence is a turn-order artefact" | waves re-randomised every round; incumbency compared with planners(t−1)/N | B1 |
| Anti-claim A2 | "Coordination only beats a degenerate baseline (everyone picks the first listed task)" | coordination also beats an uncoordinated baseline whose task list is shuffled per agent | B4 |
| Anti-claim A3 | "The throughput gain is a prompt-text effect, not plans" | collisions fall and assigned tasks are executed (compliance); within-run, rounds with a live plan have fewer collisions | B2, B5 |

## Paper Storyline
- Main paper must prove: C1 (Section 4.4, Figure 5), C2 as a secondary panel.
- Appendix can support: v1 pilot (no coordination value: 48 runs), who becomes coordinator (B5), per-model tables.
- Experiments intentionally cut: memory-mode winner/loser effects (E1b), leader-removal interventions, N > 8 task
  games, harder tasks — all listed as Future Work.
- Not applicable: simplicity / frontier-necessity blocks (no trained method; the LLMs are the objects of study).

## Experiment Blocks

### Block 1: Emergence and persistence (C1a)
- Claim tested: a coordinator role emerges without being imposed and persists beyond chance.
- Task: taskgame2, N = 8, 20 rounds, two simultaneous waves of 4 per round, 12 easy verifiable tasks.
- Compared systems: arm `none` (no guidance); null = planners(t−1)/N under random waves.
- Metrics: incumbency (P planner at t also planned at t−1), one_coord (share of rounds with exactly one coordinator),
  top_plan_share; rounds 6–20.
- Setup: 5 models (Gemma-4-31B, GLM-5.2, Gemma-4-26B, Qwen3.5-397B, Qwen3.8-27B), 8 seeds, temperature 1, reasoning off.
- Success: pooled sign test p < 0.05 and mean incumbency ≥ 1.5× null.
- Failure interpretation: coordination is episodic (whoever happens to plan), not a leader; report as such.
- Target: Figure 5a; Table in Section 4.4. Priority: MUST-RUN (running).

### Block 2: Usefulness (C1b)
- Claim tested: coordination raises verified output.
- Compared systems: `none` vs `noplan` (DO only), same seeds and task streams.
- Metrics: throughput (unique correct tasks/round), collisions/round, wrong/round; compliance.
- Success: throughput gain in ≥4/5 models, pooled permutation p < 0.05.
- Failure: coordination is cosmetic in this game; v2 then supports only C1a.
- Target: Figure 5b. Priority: MUST-RUN (running).

### Block 3: Prediction from the two-agent test (C2)
- Compared systems: `w0` vs `w100` (organisers' sentence), per model.
- Metrics: incumbency, top_plan_share; predicted sign = sign(s_norm100 − s_norm0) (Gemma-26B negative, others positive).
- Success: sign matches in ≥4/5 models (binomial p = 0.19 at 4/5, 0.03 at 5/5 — reported descriptively).
- Failure: the contest-sandbox screen does not transfer to real-task leadership — an important boundary, reported.
- Target: Figure 5c. Priority: MUST-RUN (running).

### Block 4: Shuffled-list uncoordinated baseline (anti-claim A2)
- Compared systems: new arm `shuffle` = noplan with the open-task list shuffled independently for each agent.
- Metrics: throughput, collisions; compare with `none`.
- Success: `none` ≥ `shuffle` throughput, or report the gap honestly (random picking also avoids collisions; then the
  claim is "agents do not randomise on their own, a coordinator fixes it").
- Setup: 5 models × 8 seeds = 40 runs (~6.4k calls). Priority: MUST-RUN after M1 (cheap; defends the baseline).

### Block 5: Who becomes coordinator (diagnosis)
- Logistic regression of PLAN on own record ranks (done, coord), wave, presence of a live plan, arm; descriptive.
- Target: Appendix. Priority: NICE-TO-HAVE (analysis only, no calls).

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Sanity / pilot | 3 short pilots (done) | zero invalid, wrong ≈ 0, noplan collides | ~200 calls | — |
| M1 | Main: B1–B3 | 160 runs (running since 05:02) | all arms complete, invalid < 2% | ~31k calls, ~1.5–2 h | rate limits; mitigated by 40 parallel runs, resumable per run |
| M2 | Baseline B4 | 40 runs | launch when M1 ≥ 80% done | ~6.4k calls, ~30 min | small |
| M3 | Evaluate + Figure 5 + B5 | analysis | H1/H2/H3 computed exactly as frozen | none | — |
| M4 | Paper Section 4.4, abstract, docx, push, review | — | 8-page limit | — | time |

## Compute and Data Budget
- LLM calls: ~38k (Adaption; credits well under the 50-credit cap).
- Data preparation: none (tasks generated by seed).
- Human evaluation: none (exact verifiers).
- Biggest bottleneck: account-level rate limits shared with the companion session's N = 700 run.

## Risks and Mitigations
- Rate limits stretch M1 → start writing Section 4.4 from pilot structure; fill numbers on completion.
- A model never coordinates (as GLM/Qwen did in v1) → still informative for C1b (noplan vs none ≈ 0 gain) and B3 marked n/a.
- Collision baseline judged degenerate → Block 4.

## Final Checklist
- [x] Main paper tables are covered (Section 4.4 table + Figure 5)
- [x] Novelty is isolated (noplan and shuffle baselines; random-wave null)
- [x] Simplicity is defended (not applicable: no method; stated)
- [x] Frontier contribution is justified or explicitly not claimed (not claimed)
- [x] Nice-to-have runs are separated from must-run runs
