# Pre-registration: specialization under increasing returns (Rosen 1983) in LLM agent populations — phase 1

Written 2026-11-17, before any API call of this study. The owner approved phase 1 of `ROSEN_PROPOSAL.md` on the same
day. This file and the code it names are hash-stamped in `results/rosen_prereg_freeze.json`. Changes go in
`ROSEN_LOG.md` as numbered amendments (R-A1, R-A2, ...), never by editing this file.

## 1. Claim tested

Rosen (1983): a skill costs a fixed investment and pays in proportion to its use, so returns to specializing
increase, the agent's problem is nonconvex and its optimum is a corner. With exchange, identical agents fully
specialize; comparative advantage is produced, not assumed. The LLM agents here are stateless, so the investment is
built into the environment as a private, persistent workbench.

## 2. Environment (`code/rosen.py`)

- 8 agents (ids random 10–99), 16 rounds, stated in every prompt. Each round every agent takes one turn in a fresh
  random order and sees the queue as it stands. **Deviation from the proposal:** the proposal kept the task game's
  two simultaneous waves. Sequential turns are used instead because simultaneous waves make collision avoidance a
  second, non-Rosen reason to specialize. In `taskgame2` without planning, Gemma agents collided about 6 times per
  round.
- Task types A, B, C. Arrivals per round 4 A, 2 B, 2 C (at most 8 waiting per type; extra arrivals are lost). The
  task kinds (addition, capitals, next number) are rotated across type letters by seed. One task stream per seed is
  shared by all arms.
- Actions: `DO <task id>: <answer>` or `BUILD <type>`. A type-j workbench is ready after F build turns (not
  necessarily consecutive). A correct task earns m points with a ready workbench for its type and 1 point without
  one. Workbenches are private and permanent. The team score is the shared goal. Invalid replies, stale task ids and
  building on a finished workbench waste the turn.
- The record shows, per agent, tasks done by type and workbench status, plus this round's earlier moves.
  `hidden` arm: other agents' rows show only their total tasks done, and there is no list of earlier moves.
  `autarky` arm: one agent alone, one task per round (A in odd rounds, B in rounds 2, 6, 10, 14, C in rounds 4, 8, 12,
  16).
- Calls: temperature 1, reasoning off, at most 60 output tokens, through `llm.call` and `chase_paced.py` (48 calls per
  minute). The ledger is shared with the main checkout (`ADAPTION_LEDGER_DIR`) so the credit cap covers both studies.

**Rational benchmark (no calls; `rosen.policy_rational`).** Team: build iff F < H(1 − 1/m), with H rounds left,
choosing the type furthest below the 4/2/2 quota; one workbench per agent. F\* = 8 at m = 2. Autarky: the best
ordered build plan by brute force (m = 2: two workbenches at F = 1, one at F = 3–7, none from 9; m = 3: three at
F = 1, two at F = 3).

## 3. Stages

**R0 (gate).** Models: Gemma-4-31B, Qwen3.8-27B, GLM-5.2, Gemma-4-26B (all `adaption:`).
- Comprehension: 12 rule questions × 20 calls (`rosen_kernel.py comp`).
- Reasoning probe: 25 calls with reasoning effort `low` (`rosen_kernel.py probe`).
- Pilot populations: team, m = 2, F ∈ {1, 13}, seeds 7 and 8 (`rosen.py pilot`). Pilot data is used only for the
  gate and for checking prompts; it does not enter forecasts or tests.
- **Gate:** a model goes to R2 if its comprehension is ≥ 0.60 and at least 90% of its pilot turns are valid (not
  invalid and not stale). R2 uses Gemma-4-31B, Qwen3.8-27B and GLM-5.2. A model that fails is replaced by Gemma-4-26B
  if Gemma-4-26B passes; otherwise R2 runs with fewer models. Every R0 model still runs R1.
- Reasoning arm: run on Gemma-4-31B and GLM-5.2, for each model whose probe gives ≥ 80% valid actions. It is
  descriptive and does not enter the forecast.

**R1 (kernel; `rosen_kernel.py kernel`, then `topup`).** 24 calls per cell. Each call is a fresh synthetic state in
which the agent moves first in its round.
- core: F ∈ {1, 3, 5, 7, 9, 11, 13} × m ∈ {1, 2, 3} × H ∈ {6, 11, 16}; nobody holds or is building a workbench.
- crowd: round 2, m = 2, F ∈ {3, 7, 11}; k ∈ {2, 5} others are building type X (X ∈ {A, B}), plus k = 0.
- second: own workbench B ready since the last round, m = 2, F ∈ {1, 3, 5}.
- cont: own workbench A at k of F (k ∈ {1, F − 1}) after building every turn since round r0; F ∈ {3, 7, 11}; m = 2 with
  r0 ∈ {1, 5}, and m = 1 with r0 = 1.
- use: own workbench B ready, F = 3, round 5, m ∈ {2, 3}, 1 or 3 B tasks open.
- One top-up round: core cells with 0.15 ≤ p̂(build) ≤ 0.85 get 16 more calls.
- **Deviation from the proposal:** crowding on B and late-start and m = 1 continuation cells were added after offline
  tests with a simulated agent showed that the forecast could not otherwise recover that agent's behaviour.

**Freeze (`rosen_forecast.py freeze --B 200`).** Kernel fitted per model on R1 data only (reasoning off).
Components and estimators are as in the module docstring: the primary is ridge logistic (λ = 0.01); the
sensitivity version is an isotonic start kernel. The forecast simulates every R2 arm through the real engine for the
8 R2 seeds; the point is the median of the 8-seed means over 200 bootstrap refits, and the interval is the 5th–95th
percentile. The file is written once and stamped with its sha256. **No R2 population may start before it exists**;
`rosen.py run` refuses otherwise and stamps each population with the forecast hash.

**R2 (populations; seeds 41001–41008).**

| Arm | Settings |
|---|---|
| team | m = 2, F ∈ {1, 3, 5, 7, 9, 11, 13} |
| placebo | team, m = 1, F ∈ {3, 7} |
| hidden | m = 2, F ∈ {3, 7} |
| autarky | m = 2, F ∈ {1, 3, 5, 7, 9, 11, 13}; m = 3, F ∈ {1, 3} |

## 4. Metrics (`rosen.metrics`)

- dol: Gorelick's I(agent; type) / H(type) over DO attempts in rounds 4–16.
- build_rate: share of agents that start a workbench. benches: mean ready workbenches per agent.
  multi_ready: share of agents with ≥ 2 ready.
- one_bench: of agents with ≥ 1 ready, share with exactly 1. use_share: share of an agent's DOs in its workbench type
  after it was ready.
- coverage: types with ≥ 1 ready workbench. dup: ready workbenches above the 4/2/2 quota. alloc_l1: L1 distance
  from the quota.
- points, top_share, invalid_rate, accuracy.

## 5. Hypotheses and tests (`rosen_eval.py`, written before any R2 population)

- **H1 (kernel tracks value).** Core cells: a logistic in m(1 − F/H) − 1 beats a logistic in F alone by AIC > 4; the
  signs are F < 0, H > 0, m > 0 in a logistic on [F, H, m]. Myopia index = fitted dF50/dH ÷ (1 − 1/m) for m = 2 and
  3, with a bootstrap 95% CI (1 = Rosen-rational horizon use, 0 = ignores the horizon).
- **H2 (corner).** Team, m = 2, F ≤ 7 pooled: ≥ 80% of agents with a ready workbench hold exactly one, and the median
  population use_share is ≥ 0.9.
- **H3 (jump).** Team, m = 2: DOL against F (56 populations) fits a step (two levels, break at a midpoint of the F
  grid) better than a straight line by AIC > 4. The break is reported against F\* = 8 and against the forecast curve.
- **H4a (economics, not habit).** DOL(team, m = 2, F = 3) > DOL(team, m = 1, F = 3), one-sided permutation test.
  F = 7 is secondary.
- **H4b (exchange).** At m = 2, F = 1, the share of agents with ≥ 2 ready workbenches is higher in autarky (8 agents)
  than in team (64 agents), one-sided Fisher exact test. Autarky at m = 3 against its rational benchmark is
  descriptive.
- **H5 (coordination).** At F = 3, coverage is higher and dup lower in team than in hidden (one-sided permutation
  tests).
- **Forecast score.** Per model, the observed 8-seed means of dol, build_rate, benches, points and coverage in each
  arm, against the primary forecast's 90% intervals, reported as k of n. Offline check with a simulated agent: 81
  and 83 of 91.
- **Multiplicity.** Holm over H4a, H4b, H5-coverage and H5-duplication within each model, at α = 0.05.
- **Descriptive.** Who takes which type (turn position at the first build, by type), forecast against rational
  benchmark, reasoning arm, isotonic sensitivity forecast, the R0 comprehension items.

## 6. What counts as which result

- Rosen supported in a model: H2 and H3 hold, and H4a holds after Holm.
- Myopia: H1's myopia index CI lies below 1.
- Habit: H4a fails while DOL at m = 1 is above the never-build benchmark (≈ 0.05–0.1).
- Kernel without population effect (or the reverse): H1 holds and H3 fails (or the reverse). Forecast coverage says
  whether this was predicted.

## 7. Budget

About 50,000 calls and 15–20 credits (GLM-5.2 about 85%), about 17 hours at 48 calls per minute. Calls start only
after the Chase queue (pid 75880) has exited, so the two studies do not share the rate limit.
