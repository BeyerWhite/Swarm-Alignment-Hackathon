# When do LLM agents crown a leader? Response kernels, hierarchy models, and their limits

*Narrative report for paper writing.*
- **Track:** 2, The Science of Agent Networks.
- **Status:** experiments complete, cross-model review complete. Primary reviewer gpt-6.1-sol: 8/10, ready at the
  workshop bar (top-venue 5/10). Cross-check gpt-6-astra: 7/10, almost; its five fixes are applied.
- **Authoritative claims and evidence pointers:** `review-stage/CLAIMS.md` (v5).

---

## 1. Problem and question
In the METR (Aug 2026) Hugging Face incident, roughly 700 agents shared a covert message board. One agent,
PHASEONE[big], ended up sending hundreds of assignments, about 10% of all board assignments, and its recipients
delegated further. The hackathon asks: *under what conditions do leaders, roles and division of labour emerge in an
agent population, and can we reproduce that in a safe sandbox?*

Statistical physics has classic minimal models of hierarchy forming among identical individuals:
- **Bonabeau et al. 1995**, analysed by Lacasa & Luque 2005: winners gain status, losers lose it, status fades, and
  the probability of winning depends on the status gap through a sensitivity η.
- **Ben-Naim & Redner 2005 (BNR):** the fitter agent always wins, and status declines at rate r.

Both predict a switch from an egalitarian to a hierarchical society. **We ask whether these models explain populations
of LLM agents, and specifically whether a two-agent measurement predicts the group.**

> **In plain language.** The physics says you don't need a special individual to get a pecking order. You only need
> "winning makes you likelier to win again" to be stronger than "status fades over time". We measured how strongly
> LLM agents defer to a higher score, plugged that into the model, and checked whether it predicts what whole groups
> of agents do.

## 2. Setup (sandbox; agents only produce text)
- **Contest.**
  - Two agents receive a shared task and a public score record. Each replies LEAD (I coordinate and assign work) or
    DEFER.
  - A lone LEAD becomes coordinator; otherwise a coin flip decides.
  - Hence P_win(Δ) = ½ + ½[p(Δ) − p(−Δ)], which reduces exactly to Bonabeau's sigmoid when p is logistic.
- **Bonabeau protocol.**
  - Scores: coordinator +1, other −1, all scores ×0.9 per round, displayed to 0.1.
  - Populations: N = 8, each agent starts a contest with probability ρ = 0.5, 40 rounds.
- **BNR-inspired protocol.** Integer scores, coordinator +1 / other unchanged, additive random decline r per round,
  N = 16.
- **Framings.**
  - A numeric organiser norm weight w ∈ {0, 25, 50, 75, 100}, from "give everyone equal turns" to "let higher scores
    coordinate".
  - A 2×2 wording factorial: "won" vs "ended up as", crossed with a sentence saying high scores have gone with good
    coordinators, present or absent.
- **Models (9).**
  - Claude Haiku and Sonnet, GPT-6-luna;
  - via Adaption: DeepSeek-v4-flash, Qwen3.8-27B, Qwen3.5-397B, Gemma-4-31B, Gemma-4-26B-A4B, GLM-5.2.
- **Pipeline.**
  1. Calibrate each model's response kernel p(LEAD | own score, partner score).
  2. Freeze predictions from kernels simulated under the exact population protocol. These are write-once and
     hash-stamped, made *before* the corresponding LLM population runs.
  3. Run LLM-in-the-loop populations and score them against the frozen predictions. Every deviation is logged as an
     amendment (A1–A5).
- **Scale of the work.** About 119k logged LLM calls: about 31k calibration and probe calls, and about 87k
  population calls. Adaption spend was 4.8 of the 50-credit cap.

## 3. Results

### R1. Framing sets the sign of status sensitivity (C1; Figs 1–2)
- **The norm weight pushes finite-difference sensitivity through zero.** Under "equal turns" (w = 0), Haiku (−1.92) and
  Gemma-31B (−1.62) actively take turns: the *lower* scorer is the one that leads. Under "let higher scores coordinate"
  (w = 100), sensitivity is strongly positive in most models (Claude about 3.5, the maximum).
- **The competence sentence alone** (identical score rules) credibly raises sensitivity for GLM (+1.50 [0.67, 2.31])
  and Sonnet (+0.86 [0.24, 1.48]). For Haiku and Qwen-397B the estimate is positive but not credible. The verb ("won"
  vs "ended up as") does nothing.
- **DeepSeek-v4-flash claims the lead 97.6–100% of the time** under every framing, so it is status-blind.

> **Plain language.** How you describe the scoreboard decides whether agents defer to the top scorer or deliberately
> rotate. Telling agents that high scores signal good coordinators is enough to make some models defer. A model that
> always pushes for the lead produces no pecking order. It only produces coin flips.

### R2. Two agents predict the group qualitatively, not quantitatively (C2; Fig 3)
- **What works.**
  - A status-blind model predicts the same dispersion (about 2.0) for every population. Observed dispersion spans
    0.9–6.3.
  - The frozen pairwise kernel tracks that ordering and beats the null by **104–224 nats** in all 7 status-sensitive
    models. That range is 75–278 across KDE bandwidths, and 6 of the 7 are fully prospective.
  - DeepSeek is the built-in negative control: margin about 0.
- **What fails.**
  - The pre-declared quantitative criterion, 90% predictive coverage, **failed**: coverage ranges from 2/8 to 7/8 per
    model.
  - **Populations are consistently *more* stratified than the kernels predict.** The model-average residual is
    positive in 7/7 models, and the same excess appears in the mixed-population and hysteresis experiments.
  - The agents are stateless, so this points to kernel misestimation at the states populations actually visit.
    Calibrating at visited states, found by simulation without using LLM population data, helps: same estimator,
    +41 nats for Qwen-397B and +22 for Gemma-26B. It does not close the gap.

> **Plain language.** Watching pairs tells you which groups will form a pecking order and which will stay equal, far
> better than chance. But real groups end up more unequal than the pairwise numbers predict. Part of the reason is that
> groups spend time at score gaps we sampled thinly, so calibrate where the population lives.

### R3. One assertive agent tops a deferential population (C4)
- A single DeepSeek agent among 7 status-sensitive Gemma-31B agents finishes as the leader in **4/5 runs** (chance
  1/8, p = 0.001), by the frozen endpoint (highest mean score over the last 10 rounds). By final-round rank it leads in
  3/5. The kernel predicted 0.61 [0.20, 1.00].
- Among 7 GLM agents under equal turns it leads in 5/5.
- With 4 DeepSeek + 4 Gemma, the result is not significant against the 4/8 composition null.
- The cross-model formula holds in probes. Those probes are marginal-kernel transport checks, not independence tests.

> **Plain language.** In a group of agents that defer when they're behind, one agent that never defers ends up on top
> most of the time. That is the closest lab analogue we have to "one agent assigning work to hundreds".

### R4. Scale and forgetting (C3, C5; Figs 5–7). Finite-time only.
- **Proportional (Bonabeau) forgetting.** With the fitted kernels, a 40-round persistence measure shows **no
  sharpening from N = 8 to 700** over 120 rounds.
  - That measure is window-dependent: it falls by half to two-thirds with a 200-round window. So rankings are not
    permanently frozen.
  - **We make no phase-transition claim at incident scale.**
- **Additive (BNR-inspired) forgetting.**
  - LLM populations under "let higher scores coordinate" match the identical discrete simulator. This is
    near-automatic, because those agents behave exactly like the BNR rule.
  - The informative findings:
    - agents stayed decisive up to scores of 25;
    - the neutral-wording control breaks the BNR pattern as its kernel predicted (no stuck lower class);
    - status-blind predictions fall outside the observed values at r = 0.5–2.

> **Plain language.** We hoped a gradual change at 8 agents would turn into a sharp tipping point at 700. With
> scores that fade by a percentage, it doesn't over the time we simulated. With scores that fade by a fixed amount,
> agents behave as the BNR model says, but under that framing they are simply following the rule.

### R5. Do hierarchies reverse? Mostly yes, with one exception (C6)
- Under "equal turns" and neutral wording, a seeded leader loses first place within a few rounds, as fast as or faster
  than predicted.
- Under a middle norm (w = 50), the seeded leader held first place far longer than either kernel predicted:
  20, 20 and 40+ rounds, against a prediction of about 12.
- The pooled inequality probe (7 models) found **no** evidence that visible inequality increases deference
  (p = 0.59), so there is no sign of Stauffer-type feedback.

> **Plain language.** Change the framing and the pecking order usually dissolves. The exception is a lukewarm
> "somewhat defer to scores" setting, where an established leader stuck around much longer than predicted. That is
> worth following up.

### R6. Without a scoreboard, nobody took charge (C7)
- In a scoreless shared message board (Gemma-31B, N = 8, 6 runs), agents made 1 assignment in total.
- They did everything themselves despite constant collisions on the same tasks, 80–124 failed duplicate claims per run.
  This was a pre-registered null, so we did not scale up.

> **Plain language.** In the incident, duplicated effort is what pushed an agent to start delegating. In our sandbox
> it didn't. Something else is needed, such as work too large to do alone, or visible success.

## 4. Safety takeaway
- One agent running many others does not require a special agent. In our sandbox it arises from two things:
  - agents that read a track record as a reason to defer (set by framing);
  - an assertive agent among them.
- Framing can also *suppress* hierarchy.
- The pairwise response is a cheap, predictive screening measurement, but it underestimates how unequal real groups
  become.

## 5. Limitations
- **Agents are stateless and never perform tasks.** Claims concern coordinator selection under externally maintained
  scores, not operational leadership.
- **Small populations and few seeds.** N = 8 or 16, 2–5 seeds for the follow-ups. Large N is simulated from kernels,
  and about 21% of those decisions extrapolate in mean status.
- **Quantitative calibration fails**, with a systematic under-prediction of stratification.
- **Sonnet's amended forecast** postdates its population runs (disclosed amendment A1). Haiku's state-matched
  predictions are retrospective.
- **No stationary phase-transition claim.**

## 6. Figures and tables

| Fig | File | Content |
|---|---|---|
| 1 | `figures/fig1_eta_vs_norm` | Finite-difference sensitivity vs norm weight w |
| 2 | `figures/fig2_wording_factorial` | Wording factorial |
| 3 | `figures/fig3_pred_vs_obs_{sd_h,q}` | Frozen kernel prediction vs observed population (all models), with the null |
| 4 | `figures/fig4_trajectories_*` | Status trajectories |
| 5 | `figures/fig5_rho_sweep_*` | ρ sweep: observed seed means vs predictive bands |
| 6 | `figures/fig6_fss_proportional` | Finite-size simulations (finite-time) |
| 7 | `figures/fig7_bnr_additive` | Discrete BNR-inspired F(0.05T) vs r |

Tables: `results/eta_fd_beta.csv`, `results/competence_contrast.csv`, `results/coverage_common_design.csv`,
`results/reeval_followups.json`, `results/probe_table.csv`.

**Manual figure needed:** a schematic of the contest protocol and pipeline (calibrate → freeze → populate → score).

## 7. Follow-ups (Sunday, if time)
- A real-task message board with mixed-competence populations: does leadership track ability or luck?
- Diagnose the systematic under-prediction of stratification: denser state-matched calibration, history-dependent
  prompts.
- More seeds for the w = 50 persistence anomaly.
