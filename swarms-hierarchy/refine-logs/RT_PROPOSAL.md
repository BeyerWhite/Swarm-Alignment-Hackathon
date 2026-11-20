# Owner decisions, 2026-11-20

- Models for now: Gemma-4-31B and Qwen3.8-27B only (GLM-5.2, Claude Haiku etc. later if results are interesting).
- RT runs before Chase E14 (after the running Rosen queue).
- The effort-cost rule is briefed to the owner before any call uses it.
- Branch `response-threshold` (worktree `.worktrees/rt`, cut from `rosen-specialization`).

# Proposal: measuring the response-threshold kernel in LLM agent populations (Bonabeau–Theraulaz)

Draft 2026-11-20, for the owner's review. No calls made and no code written yet. Study label **RT** (the existing
`threshold.py` in the main checkout is about the hierarchy threshold ρ_c, so new files use the prefix `rt_`).

## 1. The claim and why it is testable now

**Model.** An agent takes up task j with probability T(s_j) = s_j^n / (s_j^n + θ_ij^n), where s_j is the task's
stimulus (backlog) and θ_ij the agent's threshold (Bonabeau, Theraulaz & Deneubourg 1996; n = 2 in their fits).
With reinforcement (Theraulaz, Bonabeau & Deneubourg 1998), θ_ij falls by ξ each time i does j and rises by φ each
time it does not, so identical agents differentiate into specialists. Gautrais et al. (2002) add that
specialization appears only above a critical colony size and demand.

**One identity makes it measurable.** T(s) = σ(n (ln s − ln θ)): the Hill function is a logistic in log stimulus.
With two task types and a REST option, the natural kernel is a multinomial logit

  P(do j | state) ∝ exp(n ln s_j − n ln θ_j(h)),   P(rest) ∝ 1,   ln θ_j(h) = ln θ_0j − ξ·(own recent j turns) + φ·(own recent non-j turns)

so n, θ_0j, ξ and φ are regression coefficients that can be estimated by intervention, exactly as the hackathon
paper estimated η and the Chase study the bystander kernel.

**What is known for LLMs.** Emergent specialization is reported (Dochkina 2026; Ji et al. 2026; MoltBook), but no
study measures uptake as a function of stimulus and own history, so none can test the model. A standard search on
2026-11-20 found nothing newer that does. In insects the model itself is contested: Ulrich et al. (2021, PLoS Biol.)
found that threshold variation alone reproduces only part of the observed patterns in clonal raider ants. LLM agents
allow what ants do not: the stimulus and the agent's own history can be set by hand.

## 2. What the repo already tells us (pilot evidence at zero cost)

- **Rosen R1x `hist` cells** (24 calls each, round 5, own record all one type vs mixed) already contain a crude
  reinforcement measurement:

  | Model | own record all A → does A | all B → does B | all C → does C | mixed record |
  |---|---|---|---|---|
  | GLM-5.2 | 24/24, 20/24 | 22/24, 18/24 | 21/24, 22/24 | mostly A (16–18/24) |
  | Qwen3.8-27B | 15/24, 10/24 | 10/24, 9/24 | 10/24, 12/24 | spread |
  | Gemma-4-31B | 24/24, 24/24 | **0/24**, 2/24 | **1/24**, 1/24 | 24/24 A |

  (pairs are m = 1, m = 2.) GLM keeps doing what it did before: strong reinforcement. Qwen moderate. Gemma ignores its
  history and takes type A, the biggest backlog, which is also listed first: stimulus- or position-driven, ξ ≈ 0.
- **Rosen phase 1, m = 1 (workbenches worthless).** DOL 0.34 ± 0.23 (GLM), 0.34 ± 0.08 (Qwen), 0.10 ± 0.05 (Gemma;
  never-build benchmark ≈ 0.1). The order matches the `hist` cells: the models with history dependence specialize
  without any economic return. The Rosen report called this "role habit"; threshold reinforcement is the formal name
  for it.
- **Hackathon real-task game.** Planner recurrence 0.27 vs 0.21 by chance: weak reinforcement of the planning task.
- **Lessons carried over.** Paper: do not smooth sparse cells toward ½ (it acts as a lapse and biases forecasts
  toward equality); use the unshrunk/isotonic estimator. Rosen: single-decision kernels missed population levels
  because they did not vary own history, turn position and what others had already done, so the kernel must be
  measured in states that populations actually visit. Gemma's position bias means list order must be randomized and
  decoupled from task type.

## 3. Environment (`rt.py`, built on the Rosen workshop engine)

- N = 8 agents, 24 rounds (stated). Every round each agent takes one turn in a fresh random order and sees the queue
  as it is (sequential, as in Rosen, so collisions are not a second reason to specialize).
- **Two task types, A and B** (the classic two-task model). Task kinds (addition, capitals, next number) rotated over
  type letters by seed. Arrivals δ per round split equally between types; backlog capped at 12 per type.
  **List order of the two types is randomized per turn** (Gemma lesson), and logged.
- Actions: `DO <task id>: <answer>` or `REST`.
- **Why REST must cost something to forgo.** Without an effort cost an LLM works whenever any task is open, and
  uptake saturates at 1. Rule stated to agents: every task still waiting at the end of a round costs the team 1
  point; every turn spent on a task costs the team c points of effort (resting costs nothing). c is set once in RT1
  (below) so that uptake at a mid backlog is neither ≈ 0 nor ≈ 1, then frozen.
- **No return to experience.** Doing more A never makes A easier or worth more. That is what separates this study
  from Rosen (returns to investment) and from Rosen R4 Yang–Borland (returns to practice): any specialization here is
  behavioural reinforcement, not economics.
- **The lever unique to LLMs: own-history window K.** The prompt shows "your last K turns: A, A, rest, B, …".
  K = 0 removes the only channel through which reinforcement can act (fixed-threshold model for identical agents);
  K = 8 allows it. Other agents' records (tasks done per type) are shown by default; one arm hides them.
- Metrics: Gorelick DOL index (agent × type, rounds 5–24) **against a within-run permutation null** (the index has
  an upward finite-sample bias); activity inequality (Gini of work turns: Gautrais's "elitism"); specialist
  persistence (P(same type as own previous work turn)); backlog per type; team points.

## 4. Stages

**RT0 — retrospective, no calls.** Fit the kernel form above observationally to Rosen m = 1 populations, the R4 λ = 0
control arms (when the running Rosen queue writes them) and the paper's task game. Descriptive only (observational,
not interventional), but it gives priors for the sizes below.

**RT1 — gate and calibration (per model).** Rule comprehension (10 items × 20 calls; pass ≥ 0.6). Effort-cost
calibration: c ∈ {0.5, 1, 2} × 6 backlog levels × 20 calls; c is the value whose uptake at backlog 4 lies closest to
0.5, frozen before RT2. Two pilot populations per model (used only for prompt checks and for RT2b states).

**RT2 — kernel by intervention (single-agent calls, fresh synthetic states).**
- **2a grid:** backlogs s_A, s_B ∈ {0, 1, 2, 4, 7, 11} (35 combinations without 0/0) × 7 own-history profiles over
  the last 8 turns (none shown; 8A; 5A 1B 2R; 3A 3B 2R; 1A 5B 2R; 8B; 8 rest) × 8 calls. Separates practice (ξ: more
  j at fixed rests) from disuse (φ: more other-type work vs more rest). ≈ 2,000 calls.
- **2b replay:** 150 states sampled from RT1 pilot populations, each queried with its real own history and with A/B
  swapped, × 8 calls. Kernel measured in states populations actually reach (Rosen lesson). ≈ 2,400 calls.
- **2c social:** others' records show A-specialists vs B-specialists vs mixed, at 12 backlog combinations, × 12
  calls (the classical model says this should not matter beyond the backlog). ≈ 430 calls.
- One top-up round for cells with 0.15 ≤ p̂ ≤ 0.85. ≈ 1,000 calls.

**Freeze.** Fit the multinomial Hill-logit (primary) and an unshrunk isotonic version (sensitivity). Simulate every
RT3 arm through the real engine, 200 bootstrap refits, 90% intervals; also two nulls fitted to the same data: a
**history-blind** kernel (ξ = φ = 0, the analogue of the paper's status-blind null) and a **position-only**
"foraging for work" kernel (take the first listed task; Tofts & Franks 1992). Written once and hash-stamped; RT3
refuses to start without it.

**RT3 — populations (per model).**

| Arm | Settings | Populations |
|---|---|---|
| main | K ∈ {0, 8} × demand δ/N ∈ {0.25, 0.5, 0.75, 1.0} × 8 seeds | 64 |
| others hidden | K = 8, δ/N = 0.5, other agents' records hidden | 8 |
| removal | K = 8, δ/N = 0.5, 32 rounds; after round 16 the two most A-specialized agents leave | 8 |

**RT4 — group size (Gautrais 2002).** K = 8, δ/N = 0.5, N ∈ {4, 16}, 8 seeds (N = 8 is RT3). Run on the cheapest
model whose RT2 ξ interval excludes 0 (rule fixed now, applied after the kernel).

**RT5 (optional) — fixed threshold differences.** K = 0, 4 Gemma-31B + 4 Qwen-27B agents, 8 seeds: with no
reinforcement, the classical model predicts specialization only if the two models' fixed type preferences differ.

## 5. Hypotheses (one-sided, α = 0.05, Holm within model)

- **H1 shape.** Uptake of j rises with s_j; the log-stimulus (Hill) kernel beats a linear-stimulus logit by
  ΔAIC > 4. n reported with its CI against the classical n = 2 (descriptive).
- **H2 reinforcement.** ξ > 0 (practice lowers θ). φ > 0 reported. Pilot expectation: GLM strong, Qwen moderate,
  Gemma ≈ 0.
- **H3 reinforcement makes specialists (primary).** DOL above its permutation null is higher at K = 8 than K = 0
  (stratified over demand, 32 vs 32 populations). The kernel predicts *which* models show it: only those with ξ > 0.
  A model with ξ ≈ 0 that specializes anyway falsifies the account for that model.
- **H4 forecast.** Frozen forecast vs the history-blind and position-only nulls by log predictive density (as in
  the paper), plus 90% coverage k of n for DOL, Gini, persistence and points per arm.
- **H5 demand.** Shape of DOL against δ/N: step vs line (ΔAIC > 4, as Rosen H3), compared with the forecast curve.
  The classical model predicts specialization fades as demand reaches capacity (everyone works everything).
- **H6 replacement.** After removal, the A backlog recovers and new A specialists emerge among the remaining agents
  (A-persistence of the remaining agents rises above its pre-removal level); recovery time against the forecast.
- **H7 sufficiency (two-sided).** Hiding others' records leaves DOL within the forecast interval. A significant
  change means agents coordinate through a social channel the classical model lacks (as Rosen's visibility arm
  found for investment).
- **H8 size (RT4).** Specialization rises with N at fixed per-capita demand (Gautrais); width of any onset reported.

**Power.** Seed-to-seed SD of DOL in Rosen populations is 0.05–0.09 for Gemma and Qwen and 0.14–0.23 for GLM. H3
compares 32 against 32 populations, so 80% power at about 0.6 SD: a DOL difference of ≈ 0.04–0.06 (Gemma, Qwen) or
≈ 0.12 (GLM). Single demand cells (8 vs 8) detect only ≈ 1.3 SD, so H5 is about shape, not per-cell tests.

## 6. What the outcomes would mean

| Result | Reading |
|---|---|
| H2 and H3 hold where ξ > 0, fail where ξ ≈ 0, forecast beats nulls | First quantitative mechanism for emergent LLM specialization; the history window is a reinforcement dial a deployer controls |
| DOL rises at K = 0, or in a ξ ≈ 0 model | Specialization comes from something else (fixed type preferences, position, copying): the classical model is insufficient for LLMs, as Ulrich et al. found for ants |
| Kernel right, population levels wrong | Same failure as Rosen phase 1; RT2b/2c say which context is missing |
| Sharp onset in demand or N | Contrast with the smooth hierarchy crossover (paper) and Rosen's per-agent corner |
| No replacement after removal | Roles are brittle: losing specialists leaves work undone (safety-relevant for agent pools) |
| H7 fails | Social coordination channel beyond stimulus; links to the Chase bystander results |

## 7. Relation to the work already in the repo

- **Paper (Bonabeau 1995 hierarchy).** Same mechanism family: reinforcement against forgetting (η and μ there,
  ξ and φ here), same pipeline (kernel by intervention, frozen forecast, null, unshrunk estimator). It supplies the
  mechanism the paper lacked for its planner/worker split.
- **Chase.** The others-hidden arm and RT2c are the task-allocation analogue of the bystander kernel.
- **Rosen.** Same engine and models. Rosen asks whether LLMs specialize for economic reasons; RT asks whether they
  specialize from habit with no return. Phase 1 already hints at the split (Gemma economic, GLM habit). Together
  they separate the two classical explanations of division of labour.

## 8. Cost and time

Measured cost per 1,000 calls (shared ledger, reasoning off): Gemma-4-31B 0.036, Qwen3.8-27B 0.033, GLM-5.2 0.55
credits. Estimates add 30% for the longer prompts (history line, 24–32 rounds).

| Stage | Calls per model | Gemma-31B | Qwen-27B | GLM-5.2 |
|---|---|---|---|---|
| RT1 gate, calibration, pilots | ≈ 950 | 0.04 | 0.04 | 0.7 |
| RT2 kernel (2a–2c + top-up) | ≈ 5,800 | 0.27 | 0.25 | 4.2 |
| RT3 populations (80) | ≈ 15,600 | 0.73 | 0.67 | 11.2 |
| **Per model** | **≈ 22,400** | **≈ 1.0** | **≈ 1.0** | **≈ 16.1** |
| RT4 group size (one model) | ≈ 3,800 | | 0.16 (if Qwen) | 2.7 (if GLM) |
| RT5 mixed (optional) | ≈ 1,500 | ≈ 0.1 | | |

- **Total:** ≈ 72,000 calls, **≈ 18–21 credits with GLM, ≈ 2.3 without.** About 25 hours at 48 calls per minute.
- **Ledger:** 86.7 now. The running Rosen queue still expects ≈ 11 (R2b rest, R6, R5, R4, R3) → ≈ 98. Chase E14,
  if approved, ≈ 2.3 (+9 with GLM `norm100`). RT then ends at ≈ 118–130, under the 150 cap. Every stage gets a
  ledger guard (estimate + ledger < cap − 1) and `llm.py` stops at the cap.
- **Scheduling:** one paced job at a time. The Rosen queue (R2b at 40/56 now) should finish around tomorrow
  morning; RT would queue behind it, before or after E14 (E14 needs ≈ 22 hours).
- **Build time (no calls):** engine, kernel battery, forecast and evaluation, plus an offline parameter-recovery
  check with scripted agents of known n, θ, ξ, φ (as the paper's Table A3 and Rosen did) before anything is frozen.

## 9. Decisions for the owner

1. Models: Gemma-4-31B, Qwen3.8-27B and GLM-5.2 (GLM is ~85% of the cost but is the strongest habit case), or the
   two cheap models only.
2. Order: RT before or after Chase E14.
3. Repo: new branch `response-threshold` in a worktree `.worktrees/rt`, cut from `rosen-specialization`, sharing the
   main ledger via `ADAPTION_LEDGER_DIR`; pre-registration and freezes as in the Rosen study. Nothing pushed unless
   asked.

## References (new to this study)

- Bonabeau, Theraulaz & Deneubourg (1996), Proc. R. Soc. B 263; Theraulaz, Bonabeau & Deneubourg (1998), Proc. R.
  Soc. B 265.
- Gautrais, Theraulaz, Deneubourg & Anderson (2002), Emergent polyethism as a consequence of increased colony size
  in insect societies, J. Theor. Biol. 215, 363–373.
- Ulrich, Kawakatsu, Tokita, Saragosti, Chandra, Tarnita & Kronauer (2021), Response thresholds alone cannot explain
  empirical patterns of division of labor in social insects, PLoS Biol. 19, e3001269.
- Tofts & Franks (1992), Doing the right thing: ants, honeybees and naked mole-rats, TREE 7 (foraging-for-work).
- Gorelick, Bertram, Killeen & Fewell (2004), Normalized mutual entropy in biology, Behav. Ecol. Sociobiol. 56.
