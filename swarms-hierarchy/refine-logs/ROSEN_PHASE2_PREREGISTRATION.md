# Pre-registration: Rosen study — follow-ups and phase 2

Written 2026-11-19, after phase 1 (`ROSEN_R2_REPORT.md`) and before any API call of the stages below. The owner
approved "all of the follow-ups (including Phase 2)" the same day. This file and the code it names are hash-stamped in
`results/rosen_p2_prereg_freeze.json`; changes go in `ROSEN_LOG.md` as amendments (R-A3, ...). Everything was run
end to end offline first, with a scripted agent that has type persistence and lone-agent reluctance built in
(scratch test; results in the log).

Common to all stages: temperature 1, reasoning off unless stated, 60 output tokens (2,000 with reasoning), calls
paced at 48 per minute through `chase_paced.py`, one shared credit ledger, every population kept, one-sided tests
at α = 0.05 with Holm correction within the family named. Population runs are stamped with this file's freeze
hash (R2x also with its forecast hash) and refuse to start without it.

## Follow-up 1 — a richer kernel, tested prospectively (R1x, freeze, R2x)

Phase 1's forecast missed (25, 18, 57 of 91) because R1 did not measure the lone-agent context, task-type
persistence, or crowding by finished workbenches and turn order. R1x (`rosen_kernel.py extend`, 43 cells × 24
calls, Gemma-4-31B, Qwen3.8-27B, GLM-5.2) measures them: `aut` (lone agent), `hist` (own record all one type vs
mixed), `done` (others hold ready workbenches), `pos` (round-1 turn position after earlier builders or doers).
`rosen_forecast2.py` adds four components fitted on these cells only (see its docstring) and freezes, before any
R2x population, `results/rosen_r2x_predictions.json` with three forecasts: **v2** (primary), **v1** (the phase 1
kernel refitted on R1 rows, same bootstrap, B = 200) and **v2_retro** (v2 on the R2 seeds; retrospective, since
the R1x cells were chosen after seeing R2).

R2x: 8 fresh seeds 41101–41108, arms team m = 2 F ∈ {1, 3, 5, 7, 9, 13}, team m = 1 F = 3, autarky m = 2
F ∈ {1, 3, 7}; the three models.

- **F1a (primary).** Scored metrics as in phase 1 (dol, build_rate, benches, points, coverage; no dol in autarky):
  47 per model. v2's 90% intervals contain more observed means than v1's: exact one-sided McNemar test on the
  discordant metric pairs pooled over the three models. Coverage k of n reported per model.
- **F1b (replication, per model).** Phase 1's H2, H3 (step vs line, now on F ∈ {1, 3, 5, 7, 9, 13}) and H4a on
  the fresh seeds.

## Follow-up 2 — reasoning on, in populations (R2r)

Gemma-4-31B only, reasoning effort "low", the R2 seeds 41001–41008 (paired with R2), arms team m = 2 F ∈ {1, …, 13},
team m = 1 F = 3, autarky m = 2 F ∈ {1, 3, 7}. **Not run:** the same for GLM-5.2 (measured 0.0059 credits per
reasoning call, about 50 credits for this design, beyond the cap).

- **R2r-a.** Build rate at F ∈ {5, 7} (16 populations) higher than in R2 (permutation test).
- **R2r-c.** Team points (m = 2, paired by F and seed, 56 pairs) higher than in R2 (Wilcoxon signed-rank).
- Holm over R2r-a and R2r-c. Descriptive: the DOL step location (phase 1 off: F ≈ 4, rational 8), DOL at m = 1
  (habit check, never-build level ≈ 0.1), lone-agent building at F = 3 (rational: build), invalid rate.

## Phase 2

**R2b — threshold shift (Gemma-4-31B).** Team m = 3, F ∈ {1, …, 13}, R2 seeds. F50 = the F at which an agent's
probability of starting a workbench is one half (agent-level logistic in F). Prediction: F50(m = 3) > F50(m = 2,
from R2), bootstrap 95% CI of the shift above 0 (populations resampled within F). Rational shift +2.7 (8 → 10.7).

**R3 — Becker–Murphy (Gemma-4-31B, GLM-5.2; `rosen_p2.py bm`).** Jobs are chains A → B → C; a step opens when the
previous one is done; a step done by a different agent from the job's previous step pays a handoff fee c,
deducted from that step's points. Workbenches as in phase 1 with F = 3, m = 2. c ∈ {0, 0.5, 1, 2} × J ∈ {2, 4} jobs
per round, 8 seeds. **Deviation from the proposal:** J ∈ {2, 4} instead of {4, 8}, so that market size straddles
the team's capacity (6 vs 12 steps per round for 8 turns); at J = 8 both sizes would be saturated. No kernel or
forecast for phase 2: phase 1 showed that single-decision kernels miss population levels, so phase 2 tests
directions. Benchmark (fee-aware scripted specialist, offline): DOL 0.42 → 0.02 (J = 2) and 0.93 → 0.19 (J = 4) as
c goes 0 → 2.
- **R3-a.** DOL falls with c (stratified by J: sum of Spearman correlations, permutation within J).
- **R3-a′.** Handoff share falls with c (same test).
- **R3-b.** DOL higher at J = 4 than at J = 2 (stratified by c, permutation within c).
- Holm over the three, per model. Descriptive: types per agent, workbenches, points, fees.

**R4 — Yang–Borland (Gemma-4-31B, GLM-5.2; `rosen_p2.py yb`).** No workbench; a correct type-j task earns
1 + λ × (type-j tasks this agent has done before), λ ∈ {0, 0.25, 0.5}; stated horizon 8 or 24 rounds; every game is
stopped after round 8 (the agents are not told), so the stated horizon is the only difference. 8 seeds.
Benchmark: since every task is also practice, a rational team specializes from round 1 for any λ > 0 at either
horizon; a horizon effect therefore shows that the agents value experience as an investment that scales with the
stated horizon, as Yang and Borland's agents do when specializing has a present cost.
- **R4-a.** DOL (rounds 2–8) rises with λ (stratified by horizon).
- **R4-b.** At λ ∈ {0.25, 0.5}, DOL higher with horizon 24 than 8 (stratified by λ).
- Holm over the two, per model. Control (two-sided, descriptive): no horizon difference at λ = 0.

**R5 — N-scaling (Gemma-4-31B).** Team m = 2, F ∈ {1, 3, 5, 7, 9, 11}, N ∈ {4, 16} (arrivals and queue cap scaled
per capita: 2/1/1 and 8/4/4), R2 seeds; N = 8 is R2. Per N, a four-parameter logistic of population DOL in F
gives the location F50 and width s; sharpness w = s / F50, bootstrap over populations. Prediction (Rosen's
corner is in each agent's problem): w does not change with N: the 90% CI of log(w16 / w4) contains 0 and its point
is below log 2 in size. If s is below the grid resolution (s < 1) at every N, the result is reported as "steps
sharper than the grid at every N", which also counts as N-independent. Contrast (descriptive): the same
logistic on the hierarchy order parameter against ρ at N = 8 and 16 (`threshold_eval.json` in the main checkout).

**R6 — real capital (Gemma-4-31B; battery also Qwen3.8-27B and GLM-5.2).** Type-j tasks are x × k_j (x from 12 to
39, k ∈ {17, 23, 29} rotated by seed). Instead of a workbench, an agent works F turns on a notebook for a type;
a ready notebook shows the full x × k_j table in that agent's later prompts. Every correct task earns 1 point:
the gain is measured, not stated. **Deviation from the proposal:** multiplication tables instead of cipher keys,
so that accuracy without the notebook is well above zero and the gain is finite.
- Battery (`rosen_p2.py nbacc`): 60 calls with and 60 without the table per model → p_table, p_none,
  m̂ = p_table / p_none and the rational threshold F\* = 16 (1 − p_none / p_table).
- Populations: team, F ∈ {1, 3, 5, 7, 9}, R2 seeds. Phase 1's H2 (corner, F ≤ 7) and H3 (step vs line) tests;
  the step location is reported against F\*, and accuracy with and without a notebook is measured in the
  populations too.

## Order, budget and stopping

Queue (`code/rosen_p2_queue.sh`, one paced job at a time, after the running Chase E7 queue): R6 battery → R1x →
freeze → R2x → R2r → R2b → R6 populations → R5 → R4 → R3. Estimated 85,000 calls (about 30 hours) and 22 credits
(GLM-5.2 about 14 of them). The ledger stood at 58.8 of the 100-credit cap on 19 November, with Chase E7 still
spending (about 6 more expected). Before each stage the queue checks that ledger + the stage's estimate stays
below 99 and stops otherwise; `llm.py` also refuses calls beyond the cap. A stage cut short by the cap is reported
as such, not dropped.
