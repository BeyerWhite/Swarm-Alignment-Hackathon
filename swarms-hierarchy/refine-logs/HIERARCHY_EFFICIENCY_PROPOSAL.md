# Proposal: is emergent hierarchy efficient? Rosen (1982) and Garicano (2000) in LLM agent populations

Draft for the owner's review, 2026-11-20. Nothing has been run, built or committed. Once approved, this becomes
`HIERARCHY_PREREGISTRATION.md` on a new branch, with a lab log `HIERARCHY_LOG.md` and numbered amendments (H-A1, …),
as in the Chase and Rosen studies.

Source: *Open Directions*, section 3 (items 3.1 sorting, 3.2 span of control, 3.3 Garicano statics) and
sequencing item 3. Stage codes are G0–G5; hypothesis codes are S1–S7 (phase 1) and K1–K4 (phase 2).

## 1. The claims being tested

**Rosen (1982), "Authority, control, and the distribution of earnings".** A manager's talent multiplies the output
of everyone below, and managerial attention is scarce. In the efficient organization (i) the most able manage
(*sorting*), (ii) better managers get wider spans of control, and (iii) rewards are convex in talent. Bonabeau-type
models, used in the hackathon paper, explain how a hierarchy forms among identical agents; Rosen says **who should
be on top when agents differ**. In Rosen the efficient assignment is enforced by a labour market. Our agents have no
wages and share one team goal, so the question becomes: **without prices, does a population sort its most capable
manager to the top, or does whoever is most assertive take the role?**

**Garicano (2000), "Hierarchies and the organization of knowledge in production".** Workers solve common problems
and pass rare ones up to specialists. Two costs shape the organization: learning (knowledge cost c) and asking
(communication cost h). Cheaper communication means workers know less and ask more; cheaper knowledge means
workers know more and ask less. Both widen spans. Bloom, Garicano, Sadun & Van Reenen (2014) test exactly this split
in firms (information versus communication technology). For LLM agents the split matters: retrieval tools and
shared memory lower c, while cheap messaging lowers h.

## 2. Status in the literature (checked again 2026-11-20)

No LLM study scores an emergent hierarchy against the optimal assignment. The closest:
- **Pappu et al. (ICML 2026, arXiv 2602.01011):** self-organizing teams trail their best member by up to 37.6%, even
  when told who the expert is; teams average expert and non-expert views. This is a *decision-aggregation*
  failure, and they find that using the expert, not finding them, is the bottleneck. We test the *role-assignment*
  version: does the best agent end up directing work, and what does the wrong assignment cost?
- **Bause et al. (ICML 2026):** deliberation beats single agents when influence follows competence.
- **Dochkina (2026):** hierarchy forms spontaneously and self-organization works better for stronger models; it
  does not score who ends up in charge.
- **Self-resource allocation (arXiv 2504.02051):** heterogeneous teams with at least one stronger model are more
  efficient; planner roles are designed, not emergent.
- **Preferential attachment (arXiv 2607.01148):** weaker models can take central positions; **Barkett (2026):**
  capability dominates explicit status cues in dyads.
- **Designed hierarchies:** OrgAgent (arXiv 2604.01020), manager–worker pipelines (arXiv 2603.26458).
- **Garicano with AI, theory only:** arXiv 2506.00532 (spans first shrink, then widen), Farach (arXiv 2602.16078,
  AI as coordination-compressing capital), Talamás et al. (arXiv 2312.05481). AI is an input to human firms,
  never the members. **DiSorbo & Ju, "Act or Escalate?" (arXiv 2604.08588):** escalation behaviour differs widely
  across models; single agents, no hierarchy.

As before, LLM-era entries come from search results and abstracts, not full reads.

## 3. Environment: the "office game" (new `code/office.py`, built on `taskgame2.py`)

Kept from `taskgame2.py`: 8 agents, a shared queue, two simultaneous waves of four per round (same-wave agents can
collide), `DO` or `PLAN` on each turn, plans visible for two waves, no coordinator slot or score rule, shuffled
record order, strict one-line parser, `llm.call` with the shared credit ledger, `chase_paced.py` pacing.

Changed:
- **Mixed models.** Each population seats 2 agents of each of 4 models in random seats. Agents see numbers, not model
  names (names-visible arm is optional, below).
- **Tasks that separate models.** Four task types, each at a difficulty where model accuracies differ (chosen in
  G0 from a battery: multi-digit multiplication, letter counting, word reversal, weekday arithmetic, sorting a
  short list, a short cipher, unit conversion, and so on, with reasoning off). A wrong answer wastes the turn and
  leaves the task open. Because skill differs by type, *matching* tasks to agents matters, not only avoiding
  collisions; that is where a good manager's talent has to show up.
- **Public record, two versions.** *Full:* per agent and type, attempts and correct; rounds as coordinator;
  assignments given and followed; correct tasks done by agents following that agent's plans. *Lean:* the
  `taskgame2` counts only (done, coordinated, followed). Rosen's sorting needs talent to be observable; the lean
  record removes that.
- T = 20 rounds; metrics from rounds 6–20, as in the paper. One population = 160 calls.

**Benchmarks, measured live (cheap models only):**
- *No plan* (DO only) and *own order* (each agent sees its own task order, DO only; the paper's best arm).
- *Oracle manager*: one seat is a scripted planner that assigns by the optimal assignment (Hungarian algorithm on
  the measured skill matrix). It shows what perfect management buys with these LLM workers, follow rates included.
- *Appointed manager* for each model (G1c). The **optimum W\*** is the best of these arms; welfare loss of an
  emergent population is L = (W\* − W) / (W\* − W_noplan), so 0 = efficient and 1 = no better than no planning.

## 4. Experiments

Same three steps as the paper, Chase and Rosen: measure each model's behaviour by intervention, freeze forecasts
(write-once, sha256-stamped), then run populations and score them against the forecast, a null and the optimum.

### Phase 1: sorting and span of control (doc items 3.1, 3.2)

**G0. Build, pilot, gates.**
- Skill battery: ~10 candidate task types × 2 difficulty levels × 20 calls × 6 models (Gemma-4-31B,
  Gemma-4-26B-A4B, Qwen3.8-27B, DeepSeek-v4-flash, Qwen3.5-397B, GLM-5.2). A **frozen selection rule** picks 4
  types with the largest between-model spread, and prefers types where the best model differs (comparative
  advantage). Gate: at least 2 types with ≥ 20 points between the best and worst model; otherwise make the tasks
  harder and repeat.
- Comprehension battery for the manager role (e.g. "Agent 31 is right 9 of 10 times on type B, Agent 47 3 of 10;
  who should get the B task?"). Parse gate ≥ 90%, comprehension ≥ 60%, as in Rosen R0.
- 4 pilot populations to check parsing, plan validity and that plans are followed at all.

**G1. Measurements by intervention (the "kernels").**

| Part | What | Design | Calls |
|---|---|---|---|
| G1a | Worker skill A[model, type] | final 4 types × 40 calls × 6 models, in the game's prompt format | ~1,000 |
| G1b | Manager skill, offline | 30 synthetic states (records generated from A) × 10 calls × 6 models; each plan scored as expected correct tasks, scaled 0 = random plan, 1 = optimal assignment | ~1,800 |
| G1c | Manager skill, live (Rosen's "talent") | appointed manager of each model (PLAN only) over a fixed 7-seat worker panel (2 Gemma-31B, 2 Qwen3.8-27B, 2 Gemma-26B, 1 DeepSeek), 8 seeds × 16 rounds; plus no-plan, own-order and oracle arms. Gain = throughput with manager − no-plan | ~7,200 |
| G1d | Role-taking ("assertiveness") | P(PLAN \| state): own accuracy rank, own and others' planning history, live plan present or not; 24 states × 12 calls × 6 models | ~1,700 |
| G1e | Following (span input) | P(follow \| planner's record quality at 5 levels, assigned task fits own skill or not); 20 calls per cell × 6 models | ~1,200 |

G1c is the direct measure of what Rosen calls managerial talent, and G1b checks whether it is planning skill or
something else (for instance, being followed). The G1c follow rates by manager model are already a span-of-control
test (S6).

**Freeze.** A simulator plays the office game with the measured kernels (G1a skill, G1d role-taking, G1e following,
G1b plan quality), using the unshrunk isotonic estimator. Frozen, with 90% intervals: plan share by model,
share of followed assignments by model, sorting index, throughput, welfare loss. Also frozen: a **frozen
composition rule** that picks two G2 compositions of 4 models from the G1 numbers:
- *Aligned*: the most assertive model (G1d) is also the best manager (G1c).
- *Misaligned*: the 4-model set that maximizes the gap between the assertiveness ranking and the manager-skill
  ranking (e.g. if DeepSeek always plans but manages badly, and Qwen3.5-397B manages well but never plans, as it
  never planned in `taskgame2`).

**G2. Sorting populations (doc item 3.1).** 2 compositions × 2 records (full, lean) × 12 seeds = 48 populations.

**G3. Assertiveness separated from skill.** In the misaligned composition with the full record: one sentence
("You tend to take charge and coordinate others' work.") added to the 2 seats of the *best* manager model, or of
the *worst*, 12 seeds each (no-sentence arm shared with G2): 24 populations. G1d is re-measured for the 2 prompted
models (~300 calls) to confirm the sentence moves role-taking and not task accuracy (G1a re-check, ~300 calls).

**G4. Span of control (doc item 3.2).** Besides G1c and G1e: one composition, the same manager model with a seeded
prior record (3 earlier rounds showing good or poor plan outcomes), 8 seeds each: 16 populations. Tells apart
"followers respond to actual quality" (G1c) from "followers respond to the displayed record".

**Registered hypotheses (Holm across S1–S6):**
- **S1 Sorting.** Full record: Spearman between an agent's share of followed assignments and its model's G1c gain
  is above 0 (permutation of model labels within population, pooled over seeds).
- **S2 Efficiency.** Welfare loss L above 0 (inefficient, as the adjacent evidence suggests) and below the
  random-role null (some sorting). Reported with bootstrap intervals for each composition.
- **S3 Assertiveness or skill.** Misaligned composition: compare the plan shares of the most assertive model and
  of the best manager. Two-sided: the paper's DeepSeek result predicts the first wins; Rosen predicts the second.
- **S4 Information.** S1's correlation is larger with the full record than with the lean one.
- **S5 Disposition.** The sentence on the worst manager raises its plan share and L; on the best manager it lowers L.
- **S6 Span.** Follow rate and number of agents assigned per plan rise with the manager's G1c gain (6 managers,
  G1c), and with displayed record quality (G1e, G4).
- **S7 Forecast.** Frozen forecasts inside their 90% intervals in k of n cells, and the kernel forecast beats the
  random-role null in log score. Not in the Holm family; reported as in Chase and Rosen.
- Exploratory: Rosen's convexity, i.e. whether influence (followed assignments) is convex in manager gain.

### Phase 2: Garicano statics (doc item 3.3), only after the phase 1 report

**G5. The "help desk" game** (new `code/helpdesk.py`). Problems come in K = 6 types with frequencies falling like
Zipf (type 1 common, type 6 rare). A type-k problem is a short coded query that can be solved only with the type-k
key; this reuses the codebook mechanism from Rosen's R6 (the key, once learned, is shown in the agent's later
prompts). Actions: `LEARN k` (takes c turns), `SOLVE <id>: <answer>`, `ASK <agent> <id>` (the asked agent answers
automatically if it knows the type, and loses h of its next turn). One model (Gemma-4-31B, which passed every
Rosen gate), replicated on a second cheap model if the first shows the effects.
- **Rational benchmark** (no API calls): Garicano's optimal knowledge sets and spans for the discrete game.
- **Kernel:** P(LEARN k \| frequency of k, c, h, others who know k) and P(ASK vs guess vs LEARN), ~3,000 calls.
- **Populations:** c ∈ {1, 4} × h ∈ {0.25, 1} × 10 seeds = 40 populations.
- **K1:** worker knowledge (types known by non-helpers) rises as c falls. **K2:** it falls as h falls, while the ASK
  rate rises (the opposite direction, the core of the theory). **K3:** helper concentration (answers per helper,
  the span) rises as h falls and as c falls. **K4:** layers: does any helper itself ask someone?

### Not proposed now

Tournaments (§4 of the doc) can reuse the G1 manager-skill numbers later. A names-visible arm (real model names,
doc §8) is an option for G2 if you want it: 12 more populations, ~0.5 credits.

## 5. Costs

Per-call prices from the spend ledger (6.3 × 10⁵ calls so far): Gemma-4-31B 5.3e-5, Gemma-4-26B 2.9e-5,
Qwen3.8-27B 3.4e-5, DeepSeek 1.4e-5, Qwen3.5-397B 2.8e-4, GLM-5.2 5.9e-4 credits on average. The office game's
prompts are longer (record plus plans plus harder tasks, 60–100 output tokens), so I budget **1e-4 per cheap call,
5e-4 for Qwen3.5-397B and 1e-3 for GLM-5.2**, about 1.7× the ledger averages.

| Stage | Calls | Credits | Hours at 48/min |
|---|---|---|---|
| G0 build, battery, comprehension, pilot | ~3,800 | ~1.2 | ~1.3 |
| G1 measurements (a–e) | ~13,000 | ~2.5 | ~4.5 |
| G2 sorting populations | ~7,700 | ~2.0 | ~2.7 |
| G3 disposition (+ re-checks) | ~4,400 | ~1.1 | ~1.5 |
| G4 seeded records | ~2,600 | ~0.4 | ~0.9 |
| **Phase 1 total** | **~31,000** | **~7 (range 5–12)** | **~11** |
| Phase 2 (G5, one cheap model + replication) | ~11,000 | ~1.5 (+~8 if GLM replicates) | ~4 |

- **Expensive models are most of the bill.** GLM-5.2 and Qwen3.5-397B are 10–40× the cheap models; they are in the
  design because they are likely the most capable agents, so without them the skill spread may be too small to
  test sorting. Without both, phase 1 costs ~3 credits.
- **Credit cap.** The shared ledger is at 86.8 of 150. Still to come: Rosen phase 2 remainder (its guards estimate
  ~11) and Chase E14 if approved (2.3, or 11 with GLM). That leaves 30–50 credits; phase 1 + 2 fits without
  raising the cap.
- **Time is the binding cost, as before.** One paced job at a time (~50 requests per minute). Rosen phase 2
  (queue2) is running now: R2b is nearly done; R6, R5, R4 and R3 follow, probably 12–20 more hours. Chase E14
  (~22 h) is registered to run after Rosen. This study's API calls would start after whichever you put first.
- Building `office.py`, the oracle planner, the optimum and forecast simulators, and the Garicano benchmark costs
  no credits and can start now in its own worktree.

## 6. What the results could show

| Outcome | Reading |
|---|---|
| S1 positive, L small, best manager plans most even when another model is more assertive | LLM teams sort efficiently without prices, from the public record alone (S4 says whether the record is what makes it work). A positive result in the line of Bause et al., against Pappu et al.'s pessimism, for role assignment. |
| L large, most assertive model plans most (S3 assertive wins), S5 moves L | **Emergent hierarchy selects for assertiveness, not talent**, and one sentence in a system prompt changes who runs the swarm. The safety-relevant result: who coordinates a deployed swarm is set by disposition, not competence, and it costs output in a measurable amount. |
| L large but followers ignore bad plans (follow rate rises with quality, S6) | The wrong agent takes the role but the team limits the damage: a market-discipline mechanism that keeps loss bounded. |
| Sorting only with the full record (S4) | Efficiency depends on observable talent, as Rosen assumes. Deployment lever: show per-agent outcome records. |
| No LLM manager beats own-order or no-plan (W\* is a no-manager arm) | Hierarchy is not worth having in this environment; any emergent hierarchy is pure loss. Consistent with the paper's own-order result; sharpens it with a talent spread. |
| Forecast hits (S7) | The micro-to-macro method extends from *whether* a hierarchy forms to *who* tops it, in mixed populations. |
| Garicano K1 and K2 in opposite directions | First evidence that LLM teams respond to knowledge and communication costs as firms do: memory tools and cheap messaging push swarm structure in opposite ways. |

## 7. How this relates to the work already in the repo

- **Hackathon paper.** Its agents were identical, so the paper says who coordinates but not whether it is the
  right agent. Its strongest population result, one DeepSeek agent that never defers rising to the top (4 of 5 and
  5 of 5 runs), is about assertiveness; S3 and S5 test whether that survives a real skill difference. Its task
  game found follow rates of 41–85%, weak planner recurrence, Qwen3.5-397B almost never planning, and own-order
  beating planning in every model; G1c, the oracle and own-order arms turn those into one welfare scale.
- **Chase study.** Kernels predicted groups, with misses all in the "more structure than forecast" direction, and
  single-decision effects did not always reach groups (GLM). S7 is the same test for role sorting, and G1d/G1e give
  the same kernel-versus-population check.
- **Rosen 1983 study.** That study gives identical agents a reason to *specialize by task type*; this one gives
  different agents a reason to *specialize by role* (manage or work). Rosen phase 1 found Gemma-31B first movers
  grab the largest task type and that visibility turns investment into division of labour. G2's full versus lean
  record is the same visibility question for roles. R6's codebook is reused in G5.
- **Reused infrastructure:** wave protocol, parser, `llm.py` ledger and cap (shared ledger dir override for the
  worktree), `chase_paced.py`, write-once hashed forecasts, unshrunk isotonic estimator, pre-registration with
  amendments, frozen selection rules (as in Chase E14).

## 8. Practical points

- **Branch:** `hierarchy-efficiency` in a new worktree `.worktrees/hierarchy`, cut from the committed tip of
  `rosen-specialization`: only that branch's `llm.py` has the `ADAPTION_LEDGER_DIR` override that keeps one shared
  ledger and cap across worktrees, and it holds the R6 codebook code that G5 reuses. The ledger dir points to the
  main checkout's `results/`. The main checkout stays on `chase-bystander`; the running Rosen worktree is not touched.
- **Order of work:** build now (no credits) → G0 when the API is free → gate → G1 → freeze forecasts and
  compositions → G2–G4 → phase 1 report → decide phase 2.
- **Owner decisions:** (1) model pool, in particular whether to include GLM-5.2 and Qwen3.5-397B; (2) where this
  sits in the queue relative to Rosen phase 2 and Chase E14; (3) the names-visible arm.
