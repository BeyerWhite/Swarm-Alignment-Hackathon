# Proposal: specialization under increasing returns (Rosen 1983) in LLM agent populations

Draft for the owner's review, 2026-11-17. Nothing has been run and nothing has been committed. Once approved,
this becomes `ROSEN_PREREGISTRATION.md` on a new branch, with a lab log `ROSEN_LOG.md`, as in the Chase study.

Source: *Open Directions*, section 2, and recommended sequencing item 2.

## 1. The claim being tested

Rosen (1983): output of task j = time spent on j × skill in j, and skill in j costs a **fixed** investment. Because
the investment pays off in proportion to use, returns to a skill grow with use, the agent's problem is nonconvex,
and the optimum is a **corner**. With exchange, *identical* agents fully specialize: comparative advantage is
produced by the model, not assumed. Without a fixed cost (or with too large a one) there is no specialization.

Three refinements from the doc:
- **Becker–Murphy (1992):** coordination (handoff) costs limit specialization; a bigger market supports more.
- **Yang–Borland (1991):** with learning by doing, specialization grows over time, faster with a longer horizon.
- **Forward-lookingness:** the investment only pays if the agent expects to use it; myopic agents under-invest.

LLM agents in this repo are stateless, so the harness has no human capital and Rosen predicts **no** pressure to
specialize in it today. The investment has to be built in. That is the main new piece of engineering.

## 2. The environment: "workshop game" (new `code/rosen.py`, built on `taskgame2.py`)

Kept from `taskgame2.py`: 8 agents, a shared queue of easy verifiable tasks, two simultaneous waves of four per
round (same-wave agents cannot see each other and can collide), a shared team goal, strict one-line parser,
shuffled record order, `llm.call` with the credit ledger, `chase_paced.py` pacing.

Changed:
- **Three task types** with neutral labels (A = add two numbers, B = write a word in capitals, C = next number;
  which kind gets which letter is rotated across seeds). New tasks arrive at **4 A, 2 B, 2 C per round**, so the
  team optimum is 4 A-specialists, 2 B, 2 C. Unfinished tasks stay open (cap 12 per type).
- **Two actions:** `DO <task id>: <answer>` or `BUILD <type>`. A type-j **workbench** is ready after F `BUILD j`
  turns (progress is shown). A correct task scores **m points with the workbench, 1 without**. The word
  "specialize" never appears; nothing suggests building.
- **Public record:** per agent, tasks done by type and workbench status (ready / k of F built).
- **No PLAN action** in phase 1, so the planner channel from the paper does not confound specialization.
- T = 16 rounds, stated in every prompt ("Round t of 16"). One population = 128 calls.

**Rational benchmark (no API calls).** An agent that builds at round 1 and then works one type earns m(T − F)
instead of T, so building pays iff **F < F\* = T(1 − 1/m)**: F\* = 8 at m = 2 and 10.7 at m = 3. A second
workbench never pays while one's own type has enough work. Rosen's prediction for the team is therefore a **step**:
full specialization (4/2/2 split) for F < F\*, none for F > F\*. Mid-game the threshold shrinks to H(1 − 1/m) with H
rounds left. `code/rosen_theory.py` computes this benchmark and simulates optimal-policy teams under the exact
protocol (collisions included).

## 3. Experiments

Same three-step design as the paper and the Chase study: measure each model's decision kernel by intervention,
freeze a population forecast (write-once, sha256-stamped), then run populations and score the forecast.

### Phase 1 (recommended now)

**R0 — Build and pilot.** Write `rosen.py`, `rosen_kernel.py`, `rosen_theory.py`, `rosen_eval.py`. Then:
- *Comprehension battery*, 10 items × 20 calls × 4 models: e.g. "with F = 6, m = 2 and 10 rounds left, how many
  points does building A and then doing only A earn?" This separates "does not understand the rules" from "does
  not respond to them", which is what a null result would otherwise leave open.
- *Pilot populations*, F ∈ {1, 13} × 2 seeds × 4 models: parse rate, invalid rate, and whether anything builds.
- *Reasoning probe*: does Adaption honour `reasoning: low` on these models, and at what cost per call?
- **Gate (registered):** a model goes on to populations only if parse rate ≥ 90% and comprehension ≥ 60%.

**R1 — Investment kernel (doc item 4: are agents forward-looking?).** Single-agent calls on synthetic game states.
- Core grid: F ∈ {1, 3, 5, 7, 9, 11, 13} × m ∈ {1, 2, 3} × rounds left H ∈ {6, 11, 16}, 24 calls per cell.
- Crowding: how many of the 7 others already hold a type-j workbench (0, 2, 5) × own workbench (none / one in
  another type) × F ∈ {3, 7, 11}. This is the symmetry-breaking kernel: do agents avoid crowded types?
- Use side: with a workbench in j, P(do a j task | j tasks open).
- Reasoning sub-arm: 12 core cells with reasoning effort *low*, on 2 models.
- Models: Gemma-4-31B, Qwen3.8-27B, GLM-5.2, Gemma-4-26B. Adaptive top-up of noisy cells near the threshold, as
  in the Chase calibration rounds.

Fit P(build) as a logistic in Rosen's net value ΔV = m(H − F) − H against a logistic in F alone. The fitted
coefficient on H against its rational value is the **myopia index**. Feed the fitted kernels (the paper's
unshrunk isotonic estimator, plus a logistic for comparison) into the game simulator and **freeze** forecasts of
every R2 metric with 90% intervals.

**R2 — Populations (doc items 1 and 5: corner, not gradient; symmetry breaking).** On 3 models (default:
Gemma-4-31B, Qwen3.8-27B, GLM-5.2; final list by the R0 gate), 8 seeds per arm:

| Arm | Settings | Populations / model | Purpose |
|---|---|---|---|
| F-sweep | m = 2, F ∈ {1, 3, 5, 7, 9, 11, 13} | 56 | the step at F\* = 8 |
| Placebo | m = 1 (workbench useless), F ∈ {3, 7} | 16 | specialization from habit or collision avoidance, not economics |
| Hidden record | m = 2, F ∈ {3, 7}, others' workbenches not shown | 16 | coordination of who takes which type |
| Autarky | one agent, private queue with the same per-capita mix, F-sweep | 56 short games | Rosen: no exchange, no specialization |

**Metrics.** Gorelick's division-of-labour index (mutual information between agent and task type ÷ task-type
entropy) over rounds 4–16; workbenches per agent; share of an agent's work in its top type; coverage of the three
types and distance from the 4/2/2 optimum; team points against the rational benchmark and against no investment.

**Registered hypotheses.**
- **H1 (kernel tracks value):** P(build) falls with F and rises with m and H; the ΔV logistic beats the F-only
  logistic by ΔAIC > 4 in a model. Reported per model, with the myopia index.
- **H2 (corner):** in F < F\* arms, at least 80% of agents who build hold exactly one workbench, and the
  distribution of top-type share across agents is bimodal (near 1 vs near the demand share), not centred between.
- **H3 (jump):** DOL against F fits a step better than a straight line (ΔAIC > 4), and the observed DOL falls
  inside the frozen forecast's 90% interval in most arms (reported as k of n, like the Chase 22 of 28).
- **H4 (economics, not habit):** DOL at m = 2, F = 3 exceeds DOL at m = 1, F = 3 (permutation test, same seeds);
  in autarky the top-type share stays near the demand share (0.5) at every F.
- **H5 (symmetry breaking):** visible records give higher coverage and fewer duplicate workbenches than hidden
  ones; which agent takes which type is not predicted by agent id or wave order.

### Phase 2 (only if phase 1 shows a response to F; decided after reading R2)

- **R2b — Threshold shift.** F-sweep at m = 3 on the best model. Rosen moves the step from 8 to 10.7; seeing the
  jump move where the model says is the sharpest single test.
- **R3 — Becker–Murphy (doc item 2).** Jobs become three-step chains (A then B then C); a step done by a
  different agent from the previous step costs a handoff fee c ∈ {0, 0.5, 1, 2}. Market size: 4 or 8 jobs per
  round. Prediction: types per agent rise with c and fall with market size. Kernel plus populations on 2 models.
- **R4 — Yang–Borland (doc item 3).** No BUILD action; points per task rise with own practice in that type
  (1 + λ × tasks done, λ ∈ {0, 0.25, 0.5}). Stated horizon 8 or 24 rounds; compare the *first 8 rounds* of both,
  so the horizon is the only difference. Prediction: earlier specialization with the longer horizon.
- **R5 — N-scaling.** N ∈ {4, 16} at four F values. The paper's hierarchy transition is a finite-N smooth
  crossover produced by collective dynamics. Rosen's corner sits in each individual's problem, so its width
  should *not* depend on N. This is the direct "jump against smooth crossover" contrast, measured with the same
  dimensionless sharpness (transition width ÷ location) on both order parameters, using the `fsweep_*` data.
- **R6 — Real capital.** The workbench becomes a genuine notebook: tasks are cipher decodings, BUILD turns
  derive the key from examples, and the key is shown in later prompts. The gain m is then measured, not set.
  This answers "is it just a points game?".
- Optional: put PLAN back in. Do specialists replace the coordinator the paper found? This is a Becker–Murphy
  question in the paper's own setting.

## 4. Costs

Per-call prices come from the spend ledger for the task-game calls of 8 November (~500-token prompts, reasoning
off), scaled ×1.4 for the longer record: Gemma-4-31B ≈ 0.00006, Qwen3.8-27B ≈ 0.00005, Gemma-4-26B ≈ 0.00009,
GLM-5.2 ≈ 0.00084 credits per call.

| Stage | Calls | Credits | Hours at 48 calls/min |
|---|---|---|---|
| R0 pilot + comprehension + reasoning probe | ~3,000 | ~0.8 | ~1 |
| R1 kernel, 4 models (with top-ups) | ~10,000 | ~2.6 (+~2.5 if the reasoning arm is on GLM) | ~3.5 |
| R2 populations, 3 models | ~36,500 | ~11.6 | ~12.7 |
| **Phase 1 total** | **~50,000** | **~15–20** | **~17** |
| Phase 2 (R2b, R3, R4, R5, R6) | ~42,000 | ~10–20 | ~15 |

- **GLM-5.2 is about 85% of the credits** (~13 of ~15). Without it, phase 1 costs ~3 credits.
- **The binding cost is time, not credits.** The account allows ~50 requests per minute, and the Chase queue
  (E8a at 217 of 280 populations, then E8 calibration and E8b) is using that limit now. Two paced jobs at once
  would both stall in 429 back-off.
- **Credit cap:** the ledger stands at 38.2 against `llm.py`'s cap of 100 (set for the hackathon), and the Chase
  queue is still spending. Phase 1 fits. Phase 1 + 2 probably needs the cap raised to ~150, which is the owner's
  call.
- Building the harness, the theory benchmark and the forecast simulator costs no credits.

## 5. What the results could show

| Outcome | Reading |
|---|---|
| Step at F\*, one workbench per agent, no specialization at m = 1 or in autarky | Identical LLM agents produce comparative advantage from tool economics alone. Specialization and role lock-in in agent swarms can be forecast from, and steered by, the cost of building reusable tools. First LLM test of Rosen. |
| Step, but at a lower F than F\*, and the H coefficient below its rational value | Myopia: agents under-invest in reusable capital. A measurable gap with a direct bearing on agentic tool-building. |
| Smooth decline in F, matched by the kernel forecast | The corner is blurred by sampling noise in each decision, not by collective dynamics. The forecast still works; there is no discontinuity to find. |
| Similar specialization at m = 1 | Specialization is habit or collision avoidance, not economics. Emergent-specialization reports (Dochkina 2026; Ji et al. 2026) may be measuring this. A clean negative. |
| Clear kernel, no population effect (or the reverse) | The Chase GLM pattern again: a single-decision effect that does not reach the group. The forecast says in advance which one to expect. |
| Comprehension fails | The test is not interpretable on that model; it leaves the study through the R0 gate rather than producing a false null. |

## 6. How this relates to the work already in the repo

- **The paper's hierarchy result** is a smooth crossover at finite N (M0). Rosen predicts a jump whose source is
  each agent's nonconvex problem, so phase 1 tests for the discontinuity the hierarchy work did not find, and R5
  makes the contrast quantitative.
- **The task game (`taskgame2.py`)** found a persistent but weak coordinator role (incumbency 0.23 against a null
  of 0.18, 24 of 30 runs above null) and large collision losses without planning (for example Gemma-4-31B
  throughput 4.5 with PLAN against 2.0 without). Specializing by type is a decentralized way to avoid collisions,
  which is why the m = 1 placebo is needed and why putting PLAN back in phase 2 is interesting.
- **The Chase study** showed that single-decision effects do not always reach the group (GLM-5.2) and that all
  its forecast misses went the same way (more structure than forecast). Here R1 → R2 is the same test of kernel
  against population, and the placebo arm checks whether populations again "over-organize" beyond the kernel.
- **Reused infrastructure:** wave protocol, task generator and checker, strict parser, `llm.py` credit ledger and
  cap, `chase_paced.py`, write-once hash-stamped forecasts, the isotonic kernel estimator, adaptive calibration
  rounds, and pre-registration with numbered amendments. Same model pool, so kernels can be compared with the
  LEAD/DEFER kernels already measured (for example, is DeepSeek-v4-flash, which always LEADs, also always BUILD?).

## 7. Practical points

- **Branch:** `rosen-specialization`, cut from the tip of `chase-bystander`, which holds `chase_paced.py`. No
  branch switching in the main checkout while the Chase queue runs. A separate worktree would get its own
  `results/` folder and so its own spend ledger, which would bypass the shared credit cap. The fix is for the new
  code to write to the main checkout's ledger, or to wait for the queue to finish.
- **Key:** the key in this conversation is the same one already stored in the gitignored
  `.secrets/adaption.env`. Nothing new needs storing, and it will not be written anywhere else.
- **Order of work:** build and test the harness and the zero-cost theory simulator now; start API calls when the
  Chase queue finishes (or pause it). R0 → gate → R1 → freeze → R2 → evaluation report → decide phase 2.
