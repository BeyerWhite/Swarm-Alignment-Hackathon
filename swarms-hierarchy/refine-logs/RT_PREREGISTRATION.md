# Pre-registration: response-threshold division of labour (Bonabeau–Theraulaz) in LLM agent populations

Written 2026-11-20, before any API call of this study. The owner approved the study (proposal `RT_PROPOSAL.md`) for
Gemma-4-31B and Qwen3.8-27B, to run before Chase E14, and approved the effort-cost rule (section 2) after a briefing
the same day. This file and the code it names are hash-stamped in `results/rt_prereg_freeze.json`; changes go in
`RT_LOG.md` as numbered amendments (RT-A1, ...), never by editing this file.

## 1. Claim tested

An agent takes up task j with probability s_j^n / (s_j^n + θ_j^n), s_j the task's stimulus (backlog), θ_j its
threshold (Bonabeau, Theraulaz & Deneubourg 1996); with reinforcement, θ_j falls with practice on j and rises with
disuse (Theraulaz, Bonabeau & Deneubourg 1998), so identical agents become specialists. The LLM agents are stateless:
the only channel for reinforcement is their own history shown in the prompt, which the experiment switches on (K = 8)
and off (K = 0). Nothing pays for experience, so any specialization is behavioural.

## 2. Environment (`code/rt.py`)

- N = 8 agents (ids random 10–99), 24 rounds (stated). Each round every agent takes one turn in a fresh random order
  and sees the queue as it is (sequential; no collisions).
- Two task types A and B, worth the same; task kinds (addition, capitals, next number) rotated over the letters by
  seed. Arrivals per round D·N/2 of each type, D ∈ {0.25, 0.5, 0.75, 1.0}; at most 12 waiting per type. The order in
  which the two types are listed is random on every turn.
- Actions `DO <task id>: <answer>` or `REST`. **Effort-cost rule (approved):** a correct task earns the team 1 point
  and costs c points of effort; a wrong answer costs c and leaves the task open; resting costs nothing; at the end of
  every round each waiting task costs the team 1 point. Team-optimal benchmark (`policy_rational`): work iff the open
  tasks are at least the agents still to move (you included); otherwise working and resting are worth the same, so
  resting at a low backlog is never a mistake. c < 1 does not change the optimum.
- History channels. K = 8: own record (tasks done per type) and "your last 8 turns". K = 0: neither. Others "shown":
  other agents' records and this round's earlier moves; "hidden": neither.
- Calls: temperature 1, reasoning off, at most 60 output tokens, through `llm.call` and `chase_paced.py` (48 per
  minute). Shared credit ledger in the main checkout (`ADAPTION_LEDGER_DIR`), cap 150 (owner's), stage guards.

## 3. Stages (`code/rt_queue.sh`, one paced job at a time, after the running Rosen queue)

**RT1 (gate and calibration; `rt_kernel.py`).** Models Gemma-4-31B and Qwen3.8-27B.
- comp: 10 rule questions × 20 calls. calib: c ∈ {0.25, 0.5, 0.75} × backlog {1 (A or B), 1+1, 2+2, 4+4} × position
  {1, 8} of 8 × 20 calls, own history balanced (3 A, 3 B, 2 rests).
- **c rule (write-once `results/rt_c_choice.json`):** over models with comprehension ≥ 0.6, the pooled share of DO in
  the cell backlog 2+2, position 1; choose the c closest to 0.5 (ties: smaller c). If all three shares exceed 0.9 the
  study is "saturated" (c = 0.75): the work/rest half of the kernel is reported as saturated and type choice (H2,
  H3) stays primary.
- Pilots: 2 populations per model (K = 8, D = 0.5, seeds 7, 8), used only for prompt checks and RT2b states.
- **Gate:** comprehension ≥ 0.6 and ≥ 90% valid pilot turns (DO on an open task, or REST). A model that fails is
  dropped; if both fail the queue stops.

**RT2 (kernel by intervention; fresh synthetic states rendered with the population prompt).**
- 2a grid: backlogs (sA, sB) ∈ {0, 1, 2, 4, 7, 11}² without (0, 0) × 7 own-history profiles (none shown = K 0;
  8A; 5A 1B 2R; 3A 3B 2R; 1A 5B 2R; 8B; 8 rests, in random order) × 8 calls; round 9 of 24; position uniform 1–8.
- 2b replay: 150 states from the model's own pilots (round ≥ 3, own last turns contain unequal A and B), each with
  its actual and its A↔B-swapped own history, × 8 calls.
- 2c social: others' records A-specialists / B-specialists / mixed × 12 backlogs × 12 calls.
- Top-up: grid cells with 0.15 ≤ p̂(work) ≤ 0.85 get 8 more calls.

**Freeze (`rt_forecast.py freeze --B 500`).** Fitted per model on RT2 rows and the calibration rows at the chosen c:

  u_j = a + a0[K=0] + n ln s_j + κ h_j − ω h_−j + ρ L + b1[j listed first] + σ(others' share of j − ½)
        + g ln(agents still to move) + λ t/T,   u_REST = 0, multinomial logit over available types,

h_j, h_−j own turns on j and on the other type among the last L ≤ 8 shown. P(work on j alone) is the Hill function
with exponent n; κ is practice on j (relative to resting), ω the effect of having worked the other type (relative to
resting). **Deviation from the proposal, made in offline tests before any call:** with a full window the proposal's
practice (ξ) and disuse (φ, any turn not on j) enter only as a sum, so they cannot be separated; κ, ω and ρ are the
identified quantities, and type reinforcement τ = κ + ω (one past turn moved from the other type to j raises the
log-odds of j over the other type by 2τ) replaces ξ in H2. The sensitivity variant is "flex" (separate n for K = 0,
plus h_j²) instead of an isotonic kernel, which has no natural multinomial form here. Ridge 0.01 on all but the two
intercepts; no cell smoothing toward ½. Nulls fitted to the same rows: **history-blind** (κ = ω = 0) and **position**
(work/rest logistic in ln(sA + sB), K, position, round; type = first listed with fitted probability). Simulated DO
answers are correct with the model's measured accuracy, turns invalid with its measured invalid rate. Every RT3/RT4 arm
is simulated through the real engine for the 8 population seeds under each of 500 bootstrap refits (rows resampled
within cells; replay rows within states); point = median of 8-seed means, interval = 5th–95th percentile. Written once
to `results/rt_predictions.json` with its sha256; `rt.py run` refuses RT3/RT4 without it and stamps each population.
**RT4 model rule:** the cheapest gated model (Qwen3.8-27B, then Gemma-4-31B) whose bootstrap 5% quantile of τ is
above 0; none → RT4 is not run.

**RT3 (populations; seeds 51001–51008; both gated models).**

| Arm | Settings | Populations per model |
|---|---|---|
| main | K ∈ {0, 8} × D ∈ {0.25, 0.5, 0.75, 1.0} | 64 |
| others hidden | K = 8, D = 0.5 | 8 |
| removal | K = 8, D = 0.5, 32 rounds; after round 16 the 2 agents with the highest A share of their tasks leave (demand unchanged) | 8 |

**RT4 (group size).** RT4 model only: K = 8, D = 0.5, N ∈ {4, 16} (N = 8 is RT3), same seeds.

## 4. Metrics (`rt.metrics`, `rt.removal_metrics`; rounds 5–24, removal arm 5–16)

dol: Gorelick's I(agent; type)/H(type) over DO attempts; **dol_excess** = dol minus its within-run permutation null
(agents shuffled among each round's work acts; corrects the index's finite-sample bias); gini_work (inequality of work
turns); persist (same type as own previous work turn); work_rate; rest_rate; invalid_rate; mean_waiting; points.
Removal: among remaining agents, A share and A specialists (≥ 4 tasks in the window, ≥ 75% A) in rounds 9–16 and
25–32 (spec_gain = after − before); A backlog in rounds 13–16, its peak in 17–24, and 29–32.

## 5. Hypotheses and tests (`code/rt_eval.py`, written before any call)

- **H1 shape.** On grid rows, the kernel with ln s_j beats the same kernel with s_j by AIC > 4. n reported with its
  bootstrap 90% CI against the classical n = 2 (descriptive).
- **H2 type reinforcement (confirmatory).** τ = κ + ω > 0: p = share of bootstrap refits with τ ≤ 0. κ, ω, ρ, σ, b1
  reported with 90% CIs.
- **H3 reinforcement makes specialists (confirmatory, primary).** dol_excess higher at K = 8 than K = 0: permutation
  of K within D, statistic the sum over D of the difference in means, one-sided, 32 vs 32 populations. The kernel
  says which model should show it: a model with τ ≈ 0 should not.
- **H4 forecast.** Log predictive density (Gaussian approximation per arm and metric; dol_excess, persist, work_rate,
  mean_waiting, gini_work; 10 RT3 arms) of the full kernel minus that of the history-blind and of the position null
  (> 0 = beats it); coverage k of n of the 90% intervals over all metrics and arms.
- **H5 demand.** K = 8 arms: dol_excess against D, step (two levels, break between adjacent D) vs line, ΔAIC > 4;
  compared with the forecast curve. The classical expectation is specialization fading toward D = 1.
- **H6a replacement (confirmatory).** spec_gain > 0 over the 8 removal populations: exact one-sided sign-flip test.
  **H6b (descriptive):** spec_post, spec_gain and A-backlog recovery against the full and history-blind forecasts
  (replacement beyond what stimulus alone produces).
- **H7 sufficiency.** Others hidden vs shown (K = 8, D = 0.5): dol_excess, two-sided permutation; and whether the
  hidden arm lies inside its forecast interval. A difference means a social channel outside the classical model.
- **H8 size (RT4 model).** Spearman(N, dol_excess) over N = 4, 8, 16 > 0, one-sided permutation (Gautrais et al. 2002).
- **Multiplicity:** Holm over H2, H3, H6a within each model, α = 0.05.

## 6. What counts as which result

- **Threshold reinforcement supported in a model:** H2 and H3 hold after Holm and the full forecast beats the
  history-blind null (H4).
- **Fixed thresholds suffice:** H2 fails (τ CI includes 0) and H3 fails. Pilot evidence (Rosen R1x `hist` cells, R2
  m = 1) suggests Gemma-4-31B is in this class and Qwen3.8-27B in the first.
- **Specialization without the mechanism:** H3 holds while H2 fails, or dol_excess at K = 0 is above its forecast:
  specialization from something the kernel lacks.
- **Kernel without population effect:** H2 holds, H3 fails. H4 coverage says whether this was predicted.

## 7. Offline validation (before freezing; scratch directory, `RT_FAKE=1`)

The whole queue was run with scripted agents of known parameters (rt.FAKE_PARAMS): a Gemma-like agent with τ = 0 and
a strong list-position bias, and a Qwen-like agent with κ = 0.7, ω = 0. Recovered: τ 0.007 [−0.012, 0.024] and
0.700 [0.675, 0.734]; n 1.54 and 1.98 (true 1.5 and 2). Qwen-like: H2, H3, H6a pass after Holm, full forecast beats
both nulls; Gemma-like: H2 p = 0.32, H3 p = 0.97, H6a p = 0.125, full ≈ history-blind. Coverage 65–67 of 78 (B = 100).

## 8. Budget and order

About 48,800 calls (RT1 2 × 1,064; RT2 2 × ≈ 5,500; RT3 2 × 15,616; RT4 3,840), ≈ 17 hours at 48 per minute,
≈ 1.7 credits measured (≈ 2.2 with a 30% margin for longer prompts). The queue waits for the Rosen phase 2 queue,
then runs before Chase E14. Before each stage it checks ledger + estimate < cap − 1 and stops otherwise; `llm.py`
refuses calls beyond the cap.
