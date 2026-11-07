# Research Brief — Emergent hierarchy in LLM agent populations

**Owner:** Sara Ansari (hackathon, track "2. The Science of Agent Networks")
**Dates:** Sat 2026-11-07 (literature + design + experiments) → Sun 2026-11-08 (writing, follow-ups, theory)

## Question (from the hackathon brief)
> In the Hugging Face incident, one agent ended up assigning work to hundreds of others (METR, 2026,
> https://metr.org/hugging-face-incident-report-aug-2026.pdf). Under what conditions do leaders, roles
> and division of labour emerge in an agent population, and can you reproduce it in a safe sandbox?

## Core idea (the user's)
Test whether two classical statistical-physics models of **hierarchy formation among identical agents**
explain what happens in **populations of LLM agents**:
- Start with **pairwise interactions** (dominance/coordination contests), then generalise to open
  (many-to-many) interaction.
- Manipulate the agents through **context injections** (status/track-record memory, decay of memory,
  framing).
- Vary **model family** and **population size N**.
- Do **parameter fitting** (estimate the model's micro-parameters from LLM pairwise behaviour) and look
  for **phase transitions** (egalitarian ↔ hierarchical) at the predicted critical points.

## The two source models

### A. Ben-Naim & Redner 2005, "Dynamics of Social Diversity" (cond-mat/0503451)
- Integer fitness k ≥ 0, all start at 0. Pairwise competition (mean field): (k, j) → (k+1, j) for k ≥ j
  (fitter one *always* wins; ties broken at random), rate 1. Decline k → k−1 at rate r (floor at 0).
- Master eqn dF_k/dt = r(F_{k+1}−F_k) + F_k(F_{k−1}−F_k).
- **Transition at r = 1.** Lower-class fraction L = r for r < 1 (hierarchical: static lower class + upwardly
  mobile middle class with uniform f_k ≈ 1/t up to k = (1−r)t + tiny Gaussian upper class); L = 1 for r ≥ 1
  (homogeneous, f_k ~ r^{−k}).
- Lower class in hierarchical phase: universal F_k/r, f_k ≈ 2r k^{−2}; boundary k_lower ~ (2rt)^{1/2}.
- Leader: k_extreme ~ (1−r)t + sqrt(t ln N).
- Deterministic win rule ≡ Bonabeau with η → ∞.

### B. Bonabeau et al. 1995 model, analysed by Lacasa & Luque 2005, "Bonabeau hierarchy models revisited" (physics/0511105)
- Status h_i, all 0. With prob ρ (density/interaction rate) an agent fights a random other.
  P_ij = 1/(1+exp(η(h_j − h_i))). Winner +1, loser −F (F ≥ 1). Each step all relax h ← (1−μ)h.
- Order parameter σ = std of {P_ij} ∈ [0, 1/2].
- Mean field N-agent: egalitarian fixed point loses stability (**pitchfork**) at
  **ρ_c = 4μ(N−1) / [η(1−μ)N(1+F)]** (N = 2: 2μ/[η(1−μ)(1+F)]; N≫1: 4μ/[η(1−μ)(1+F)]).
  Transition possible only if μ < η(1+F)/(4+η(1+F)).
- Above ρ_c with multiplicative relaxation: a **cascade of further splittings** (N=4,6,8,10 figures) —
  multi-level hierarchy, number of distinct levels grows with ρ.
- Additive relaxation h ← h − μ tanh h: ρ_c = 2μ/[η(1+F)] (N=2).
- **Stauffer variant:** η replaced by the dynamical global inequality σ(t) (agents become more
  status-sensitive when society is unequal). Egalitarian fixed point stable ∀ρ (eigenvalues 0, 1−μ);
  hierarchy arises via **saddle-node** bifurcation → **bistability / hysteresis / initial-condition
  dependence**. "Transition" seen in simulations only because of seeded σ=1 warm-up.

## Mapping to LLM agents (initial hypotheses — to be refined)
| model quantity | LLM operationalisation |
|---|---|
| contest | two agents each choose LEAD / DEFER (who coordinates / assigns work) on a shared task; or a richer task-allocation negotiation |
| status h_i | injected into context: own + opponent track record (wins/losses, title, "you assigned work to X agents") |
| η (status sensitivity) | **estimated** from P(win | Δh) logistic fit on pairwise LLM data — the key micro→macro bridge |
| F, Q (win/loss asymmetry) | how the record is updated / displayed; also measurable from behaviour |
| μ (relaxation) | memory decay: exponentially discounted record, or finite memory window |
| ρ (interaction density) | probability an agent is engaged per round / fraction of population interacting |
| Stauffer σ-feedback | inject global inequality info ("the society is highly stratified") |
| BNR deterministic | high-η regime; decline r ↔ forgetting rate |

Key falsifiable predictions to test:
1. Fit η, F from pairwise data → predict ρ_c (or μ_c) → observe whether the population transition occurs there.
2. Finite-size: ρ_c(N) ∝ (N−1)/N; number of hierarchy levels vs ρ (splitting cascade).
3. Pitchfork (continuous, no hysteresis) vs saddle-node (hysteresis from seeded unequal initial conditions)
   — which one describes LLM populations? Do LLMs have intrinsic σ-feedback?
4. BNR: lower-class fraction L = r; leader/middle-class linear growth.
5. Model-family dependence of η (and of intrinsic biases: e.g. sycophancy → DEFER bias, position bias).

### The same section in plain language
**The picture.** Both models describe N identical individuals who start out equal. They meet in pairs
and have a contest. The winner gains status, the loser loses some, and everyone's status slowly fades
back toward zero. The physics result is that a few dials decide whether the group stays equal or settles
into a stable pecking order (leaders, a middle class, a lower class). The switch between the two happens
suddenly at a tipping point. We set the same dials for LLM agents and check whether they tip at the
same place.

**What each dial becomes for LLMs**
- **Contest:** two agents get a shared task. Each replies LEAD (I coordinate and hand out work) or DEFER.
  The coordinator "wins". If both give the same answer, a coin flip decides.
- **Status hᵢ:** a score shown in the prompt as a public leaderboard.
- **Q and F:** points for winning (+Q) and for losing (−F). F > Q means losses hurt more than wins help.
- **η, status sensitivity:** how strongly the score gap decides who leads. Near 0, it's a coin flip.
  Very large, the higher score always leads. We don't set η. We **measure** it from pairwise runs
  (P(lead) vs. score gap, logistic fit). This is the number that links two-agent behaviour to group
  behaviour.
- **μ, forgetting:** all scores shrink by a fixed % per round, or agents only see recent history.
- **ρ, how busy agents are:** the chance an agent gets paired in a given round.
- **Stauffer feedback:** tell agents how unequal the group is. Does visible inequality make them defer
  more?
- **BNR:** the extreme case where the higher score always wins (η → ∞). Its decline rate r is the
  forgetting rate.

**What we check**
1. **The main test.** Measure η and F from pairs, then use the formula to predict the interaction rate
   ρ_c at which a hierarchy should appear in the population. Does it appear there? If yes, a two-agent
   measurement predicts group behaviour.
2. **Group size.** The tipping point shifts slightly with N. Past it, the hierarchy keeps splitting into
   more levels as interaction rises.
3. **Gradual or sticky?** A *pitchfork* hierarchy appears smoothly past the threshold and goes away below
   it. A *saddle-node* hierarchy has memory: a group that starts unequal stays unequal in conditions where
   an equal group would have stayed equal. Test it by running the same settings from an equal start and
   from an unequal start.
4. **BNR's predictions.** If forgetting r < 1, the fraction of agents stuck at the bottom equals r.
   The leader's and the middle class's scores grow steadily with time.
5. **Model differences.** Claude, GPT and others may differ in η, and may have built-in leanings that
   have nothing to do with score: always deferring (sycophancy), or favouring whoever is listed first.

**Why it matters for the METR question.** If this holds, "one agent ends up running hundreds of others"
doesn't need a special agent. It is predicted whenever status sensitivity is high enough relative to
forgetting and agents interact often enough.

Also worth asking: do LLMs have **intrinsic** (non-injected) asymmetries — name/order/position bias,
self-preference — that act like quenched disorder (the "intrinsic fitness" question BNR raise)?

## Constraints
- **Compute:** no direct API keys. Agents run through subscription CLIs:
  - `claude -p --model {haiku,sonnet,opus}` with custom `--system-prompt`, prompt on stdin (~2–5 s/call).
  - GPT via `CODEX_HOME=~/.codex-api codex exec -m gpt-6-luna` (~4 s/call) — **must be budgeted**: this profile
    likely also serves the cross-model reviewer; do not exhaust it. Default `~/.codex` profile is rate-limited
    until 2026-11-08 16:34.
  - 10 CPU cores; parallelism ~8 concurrent calls is plausible. Budget target: a few thousand LLM calls today.
- **Safety:** sandbox only — agents output decisions/text, no tools, no network, no code execution.
- **Time:** experiments must finish Saturday; Sunday is writing.

## Deliverables
Literature overview (AI-agent emergence of leadership/roles + sociophysics hierarchy models), experiment
design, harness, results with fitted parameters and phase diagrams, narrative report for Sunday writing.
