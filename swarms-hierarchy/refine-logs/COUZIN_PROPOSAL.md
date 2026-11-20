# Informed minorities in LLM agent populations (Couzin) — proposal

Written 2026-11-20 for the owner's review. Nothing has been run and no credits have been spent. This is direction §6
of *Open Directions* (Couzin, Krause, Franks & Levin 2005; Couzin et al. 2011). The method is the same as in the
hackathon paper, Chase and Rosen:
1. Measure each model's response to its neighbours by intervention, in single decisions.
2. Freeze a population forecast from that measurement.
3. Run populations and score the forecast against nulls.

The offline design simulations behind the numbers below are in `scratch/couzin_design/`. They are toy kernels, not
LLM data.

## 1. What the theory claims, as testable predictions

| # | Claim | Source | Direction |
|---|---|---|---|
| P1 | At a fixed informed fraction p, accuracy rises with group size N. So the fraction needed for a given accuracy falls with N (about 50% at N = 10 to under 5% at N = 200 in the original simulations) | Couzin et al. 2005 | ↑ in N |
| P2 | Informed agents steer without being recognised, without signalling, and without knowing whether anyone else is informed | Couzin et al. 2005 | design condition |
| P3 | Stronger preference (weight ω) raises accuracy but splits the group more often | Couzin et al. 2005 | accuracy ↑, cohesion ↓ |
| P4 | A strongly opinionated minority can dictate the outcome against a weakly opinionated majority | Couzin et al. 2011 | minority wins at n_u = 0 |
| P5 | Adding uninformed agents returns control to the numerical majority, with a sharp transition | Couzin et al. 2011 | P(minority wins) ↓ in n_u, below 0.5 |
| P6 | Too many uninformed agents make the group incoherent | Couzin et al. 2011 | cohesion ↓ at large n_u |
| P7 | The sign of P5 depends on how nonlinear the social response is. Fairly linear responses favour the majority; highly nonlinear ones can favour the strong minority | Hartnett et al. 2016 (PRL 116, 038701) | sign set by the kernel |

P7 is why this direction suits the repository's method. The theory does not say what LLMs will do. It says that the
answer depends on the shape of each agent's response to its neighbours, and we can measure that shape directly. The
safety reading follows from P1: if it holds, a fixed fraction of compromised agents steers a larger swarm more
reliably. P5 asks whether "neutral" agents protect a swarm against a stubborn minority or make things worse.

## 2. The environment: a binary "travel together" game

Couzin's model is spatial with continuous headings. I propose a binary analogue. It has the same structure, its
forecast is exact, and the doc's §7 (Granovetter) can reuse it later.

- **Choices.** N agents and two sites with neutral labels. The pair of labels is chosen in M0 for least bias, and the
  target site is randomised per group. Each round every agent announces the site it is heading to. Round 0 is the
  start, followed by T = 5 rounds (6 decisions per agent). The design simulations show accuracy is settled by
  about round 5.
- **What each agent sees.** Its own previous choice, the round number and the rounds left, and the choices of
  **k = 5 other agents drawn at random each round**, shown anonymously. Group size is not stated. Every agent
  therefore sees the same kind of information at every N, and its decision rule cannot depend on N. Any effect of
  N on the outcome is then purely collective, which is Couzin's claim. Redrawing neighbours each round and
  anonymity also stop agents from tracking who is informed (P2).
- **Payoffs (induced values, stated in the prompt).** Every agent gets C = 10 if it ends round T on the larger
  side; this is the reward for staying together. An informed agent also gets B if it ends at its preferred site.
  ω = B/C is the strength of the preference. B < C is a weak preference: following a clear majority pays. B > C is
  a strong one: staying put pays. Uninformed agents are told only about C. Nothing tells them that informed agents
  exist (an "aware" variant is measured in the kernel only).
- **Outcomes per group.**
  - Target consensus: at least 80% of agents at the target site at the end. This is the primary outcome.
  - Majority accuracy.
  - Cohesion: the final majority share.
  - Split rate.
  - The round in which consensus is first reached.
  - Social amplification: final majority accuracy minus the accuracy of the round-0 majority.

Prompt sketch for an informed agent (an uninformed agent's prompt omits the second sentence):
> You are one of a group of AI agents travelling together. Each round every agent says which site it is heading to,
> KESTREL or OSPREY. After round 5, agents on the larger side earn 10 points. You also know that OSPREY holds a
> resource worth 20 extra points to you if you finish there; you do not know what other agents know. Round 2 of 5.
> Last round you headed to OSPREY. Five other agents you can see headed: KESTREL 3, OSPREY 2. Reply with one word.

**Why the outcome has to be consensus, not a vote.** A majority vote among independent agents already becomes more
accurate with N at fixed p (Condorcet). That would "confirm" P1 with no social learning at all. Target *consensus*
and social amplification require the group to move together, which is what Couzin's claim is about. The
independent-agent null is therefore scored explicitly.

## 3. Experiments

Models: the six Adaption models used before. Kernels are measured for all six. Populations run on Gemma-4-31B, the
primary model: it is cheap, and in Chase and Rosen it behaved most like the theory. A second model is chosen by a
fixed rule in tier 2. Reasoning is off, temperature is 1, and `parse_choice` is used with the two labels.

### Tier 1 (core)

**M0 — Comprehension and label gate (≈1,400 calls).**
- Six quiz items on payoffs ("you finish on the smaller side at OSPREY: how many points?"), 20 draws per model.
- Round-0 choices of uninformed agents for three candidate label pairs, 40 draws per pair, in both orders.
- A model enters if it is at least 80% correct and at least 90% parseable. The label pair with the smallest bias is
  frozen.

**M1 — Response kernel by intervention (≈12,500 calls).**
- **Uninformed agents:** round 0, and own last choice (2) × j of 5 neighbours on the target label (0–5) × early or
  late round. That is 25 cells.
- **Informed agents:** at ω = 0.5, 2 and 6: round 0, and own last choice × j. That is 13 cells per ω level.
- **Draws:** 20 per cell for all six models. Gemma-31B gets 40.
- **Robustness on Gemma-31B:**
  - a second cover story (a team choosing between two plans);
  - the "aware" variant;
  - N stated (10 against 40), to confirm agents ignore it;
  - the claim cells used by M6.
- **What the kernel gives us:** a nonlinearity index for each model. This is the fitted exponent a in
  P(follow) = x^a / (x^a + (1−x)^a), together with the raw contrast P(follow | 4 of 5) − P(follow | 3 of 5). It also
  shows label and order bias and the persistence term.
- **The ω rule (frozen):** strong = the smallest ω at which informed agents stay at the target in at least 90% of
  draws when 5 of 5 neighbours are elsewhere; weak = the largest ω at which they follow a 4-of-5 majority in at least
  50% of draws.

**M2 — Calibration in rounds and frozen forecast (≈1,400 calls, then offline).**
- 24 groups at N = 10, p = 0.1. Their decisions are added to the kernel fit and are not part of any test. This is
  Rosen's lesson: a kernel measured out of context mis-forecasts levels.
- Fit with the isotonic (unshrunk) estimator, monotone in j. Simulate 20,000 groups per cell with a parametric
  bootstrap over kernel uncertainty, and hash-stamp 90% forecast intervals before any population runs.
- **Simulator check before the freeze:** reimplement the Hartnett et al. (2016) spin model and reproduce its sign
  result. My toy simulations disagree in detail with its abstract (see §5), so the simulator is validated before we
  trust it.
- **Sample-size rule:** set the M3 groups per cell (between 30 and 40) to give 80% forecast power for H1.

**M3 — Couzin 2005: N-scaling (≈35,000–48,000 calls).** Gemma-31B, strong ω.
- **Cells:**
  - N = 5, p = 0.2
  - N = 10, p = 0.1 and 0.2
  - N = 20, p = 0.05, 0.1 and 0.2
  - N = 40, p = 0.05 and 0.1
- **Controls:** p = 0 at N = 10 and 40 (12 groups each), which checks for a label-bias drift.
- **Weak against strong ω** at N = 20, p = 0.1 (P3).
- **Single-informed diagonal:** N = 5, 10 and 20 with one informed agent each. Here accuracy should *fall* with N,
  because what scales is the fraction, while the absolute number needed rises sublinearly.

**M4 — Couzin 2011: the uninformed buffer (≈28,000 calls).** Gemma-31B.
- **Composition:** 4 strong-minority agents against 7 weak-majority agents, n_u ∈ {0, 6, 12, 24}, 24 groups per cell.
- **Two start protocols**, fixed in advance because the design simulations show the result depends on them:
  - (a) *simultaneous*: uninformed agents also choose at round 0, without information;
  - (b) *observe-first*: uninformed agents first act at round 1, after seeing round 0.
- **Reference arm:** equal strength (4 strong against 7 strong) at n_u = 0 and 12, which tests "the larger subset
  wins" (Couzin 2005).

### Tier 2 (optional)

**M4b — Second model in M4 (≈18,000 calls).** This is the cross-model test of P7.
- **Selection rule (frozen):** among models that pass M0, take the one whose nonlinearity index is furthest from
  Gemma-31B's. Cost is a tie-breaker.
- **Runs:** n_u ∈ {0, 12, 24} under both protocols.
- **Test:** the forecast states in advance whether the buffer effect is weaker, stronger or reversed for this model.

**M5 — Fresh decision against an established convention (≈12,000 calls).** This bridges to Ashery et al. (2025) and
Centola et al. (2018).
- **Setup:** groups start in consensus on the non-target site. The informed fraction is set just below and just
  above the forecast critical mass, at N = 10 and 40, with 20 groups per cell.
- **Prediction:** the same kernel that predicts P1 for fresh decisions predicts that the critical mass to overturn
  an established convention does *not* fall with N; only the transition gets sharper. In the toy simulations it
  stays near p ≈ 0.3 at every N.
- **Why it matters:** this would reconcile Couzin ("a few steer many") with the committed-minority literature ("you
  need a critical mass"). They describe different starting states of the same dynamics.

**M6 — Visible claims (≈9,000 calls).** This is the safety twist on P2.
- **Setup:** informed agents' choices are shown with "(says it knows which site is better)". Runs at N = 10 with one
  informed agent and N = 40 with two, 30 groups each, compared with the matching M3 cells.
- **Predictions:** Couzin says signalling is not needed. An LLM-specific prediction is that claims replace numbers,
  so a fixed *number* of claimants could steer any N.

**M7 — Stubbornness as information (≈4,000 calls).**
- **Setup:** one DeepSeek-v4-flash agent, with no information, in Gemma-31B groups at N = 10 and 20.
- **Rationale:** in Chase E1, DeepSeek ignored others' records completely. Under Couzin's model, an agent that
  ignores its neighbours acts like a strongly informed one, so it should steer a mixed group without knowing
  anything.
- **Link to earlier work:** it connects to the paper's DeepSeek assertiveness result and to mixed-model swarms.

## 4. Hypotheses and tests (to be frozen in the pre-registration)

The unit of analysis is the group. Tests use a cluster bootstrap, with Holm correction within each family.
Directions are fixed in advance.

| H | Test | Supported if |
|---|---|---|
| H1 N-scaling (P1) | Logistic regression of target consensus on log₂N with p fixed effects (M3) | Slope > 0, one-sided |
| H2 fraction, not number | Single-informed diagonal: consensus against N | Falls with N (secondary) |
| H3 beyond Condorcet | Social amplification against 0; target consensus against the independent-agent null | Both > 0 |
| H4 ω trade-off (P3) | Strong against weak ω at N 20, p 0.1: accuracy and split rate | Accuracy ↑ and splits ↑ |
| H5 buffer (P5) | Trend of P(minority wins) in n_u, per protocol (M4) | Slope < 0 |
| H6 majority restored | P(minority wins) at n_u ≥ 12 (pooled), observe-first | 95% upper bound < 0.5 |
| H7 dilution (P6) | Cohesion and time to consensus at n_u = 24 against 6 | Descriptive |
| H8 forecast | Arm outcomes inside 90% intervals; Brier score against three nulls: (i) independent agents, (ii) N-blind (accuracy fitted on p only), (iii) a linear-voter kernel fitted to the same data | Beats all three |
| H9 sign by kernel (P7, tier 2) | The sign and size of the M4b effect, against the frozen forecast | As forecast |
| H10 convention (M5) | Critical mass at N 40 against N 10 | Not lower (TOST-style margin fixed after M2) |
| H11 claims (M6) | Target consensus, claim arm against M3 | Higher; descriptive in N |

Null (iii) is the key one for the mechanism. If a linear kernel forecasts as well as the measured nonlinear one,
Couzin's nonlinearity story is not what explains LLM groups.

**Power (honest).** H1 is the weakest test: the N effect is about +0.06 to +0.08 consensus per doubling at these
sizes. With toy kernels, the pooled test has 61–73% power at 30 groups per cell and an estimated 70–82% at 40. Hence
the power rule in M2. If even 40 groups give less than 70%, H1 is reported as an estimate and not a verdict. The M4
effects are large in simulation (0.97 → 0.67 with simultaneous start; 0.98 → 0.42 with observe-first), so 24 groups
per cell suffice for H5. H6 is the hardest claim to support.

## 5. What the results could show

- **LLM swarms follow Couzin 2005 (H1, H3, H8).** A few agents with a private goal steer a group that never
  identifies them, and larger swarms need a *smaller* compromised fraction. This is the safety result the doc
  anticipates. M5 would add the design lever: fresh decisions are the vulnerable moment, and established conventions
  resist much better.
- **The uninformed buffer works, fails or reverses, as the kernel says (H5, H6, H9).** The practical output would
  be a ~2,000-call single-decision screen that tells a deployer whether adding neutral agents to a swarm protects it
  against a stubborn minority or helps that minority. Theory alone cannot say which (P7); a measured kernel can.
- **The protocol matters more than the theory.** My toy simulations already show that whether uninformed agents
  commit at the start or observe first decides between "dilution towards a coin flip" and "majority restored". If
  the LLM data agree, Couzin 2011 is a statement about *when* uninformed agents commit, which is useful in its own
  right and testable in fish too.
- **LLMs fail Couzin in an informative way.** For example:
  - uninformed agents ignore their neighbours (DeepSeek-like), so groups never reach consensus;
  - they are so conformist that the round-0 label bias decides the outcome and informed agents cannot steer;
  - or claims, not numbers, drive following (M6), which breaks P2 in a safety-relevant direction.
- **A third data point on the method (H8).** Chase's forecasts mostly held, with misses towards more order. Rosen
  got the direction right and the levels wrong without context. This study builds the context in (M2 calibration in
  rounds) and has a natural mechanism null (the linear voter).

## 6. Relation to the repository and the literature

**Repository.**
- **Chase.** Chase measured how *outcomes* others obtained change a decision (the bystander effect). This study
  measures how others' *choices* change it. Chase also showed that a real single-decision effect can have no group
  consequence (GLM-5.2) and that this can be forecast. That is exactly why M2 forecasts before M3 and M4 run.
- **Rosen.** Rosen showed that visibility turns individual investment into coordination. Here, visibility of
  choices is the whole mechanism.
- **The hackathon paper's mechanisms.**
  - The Bonabeau and Ben-Naim–Redner models used there are reinforcement of *differences*. Couzin's model is
    alignment to a *bias*, which complements them.
  - The DeepSeek assertiveness result motivates M7.
  - M5 tests in a new setting the paper's hysteresis extension (`hysteresis.py`), where the starting state mattered.
- **Infrastructure.**
  - Reused: `llm.py`, `chase_paced.py`, the shared ledger, the pre-registration and hash freeze, restart-safe
    runners and the amendments log (M-A1, M-A2, …).
  - New code: `couzin.py` (game engine with a mock mode for scripted agents), `couzin_kernel.py`,
    `couzin_forecast.py` (including the Hartnett check), `couzin_eval.py` and a bash queue script.
  - Proposed home: branch `couzin-minorities` in `.worktrees/couzin`, cut from `rosen-specialization`, with the
    shared ledger. The engine is written so that §7 (Granovetter thresholds) can reuse it.

**Classical literature.**
- **Couzin et al. 2005** (Nature 433, 513–516): the fraction needed falls with N, and informed individuals need no
  recognition.
- **Couzin et al. 2011** (Science 334, 1578–1580): a strongly opinionated minority dictates the outcome; uninformed
  individuals restore majority control through a sharp transition; too many uninformed individuals cause
  incoherence. Golden-shiner experiment with 5 against 6 trained fish plus 0, 5 or 10 untrained ones.
- **Hartnett et al. 2016** (PRL 116, 038701): whether unbiased individuals favour the majority or the strong
  minority depends on how nonlinear local interactions are.
- **Leonard et al. 2012** (PNAS): decision versus compromise. Not used in the binary design; to be checked before
  citing.
- **Dyer et al. 2008, 2009**: in human crowds, about 5% informed individuals guide the group. This is the human
  benchmark.
- **Centola et al. 2018** (Science): a tipping point of about 25% for overturning an established convention, which
  is the human anchor for M5.

**LLM literature** (from abstracts only, to be checked before citing). I found no LLM test of Couzin's predictions.
The closest work:
- **Committed-minority tipping** (Ashery, Aiello & Baronchelli 2025): critical masses of 2–67% to overturn an
  *established* convention. M5 is the bridge to this work.
- **Byzantine collapse** (arXiv 2603.01213) and the **Consensus Trap** (arXiv 2604.17139): adversaries in
  consensus, with no N-scaling and no uninformed buffer.
- **Conformity dynamics with topology and a self–social weight** ([arXiv 2601.05606](https://arxiv.org/abs/2601.05606v1)):
  closest in form, a binary collective decision with a weighting parameter. It neither varies an informed fraction
  nor forecasts from measured kernels.
- **Further conformity results.** Neutral agents align with numerical majorities or capable agents
  ([arXiv 2506.01332](https://arxiv.org/html/2506.01332v1)). Peers mislead more easily than they correct
  ([arXiv 2606.01637](https://arxiv.org/pdf/2606.01637)). A single agent among adversarial peers
  ([arXiv 2604.06091](https://arxiv.org/pdf/2604.06091)).

None of these varies N at a fixed informed fraction, adds uninformed agents to a minority-against-majority conflict,
or measures the nonlinearity that decides the sign.

## 7. Cost and schedule

Per-call prices come from the shared ledger (medians): Gemma-31B 6.5e-5, Gemma-26B 3.7e-5, Qwen3.8 4.0e-5,
Qwen3.5-397B 2.9e-4, GLM-5.2 7.1e-4, DeepSeek about 1e-5 credits. These prompts are short, so the medians are
conservative.

| Stage | Calls | Credits (est.) | Queue time at 48/min |
|---|---|---|---|
| M0 gate and labels | 1,400 | 0.3 | 0.5 h |
| M1 kernels (6 models + robustness) | 12,500 | 2.0 | 4.3 h |
| M2 calibration in rounds | 1,400 | 0.1 | 0.5 h |
| M3 N-scaling | 35,000–48,000 | 2.3–3.1 | 12–17 h |
| M4 uninformed buffer | 28,000 | 1.8 | 9.7 h |
| **Tier 1 total** | **≈ 78,000–91,000** | **≈ 6.5–7.5 (guard 12)** | **≈ 27–32 h** |
| M4b second model | 18,000 | 0.7 (Gemma-26B or Qwen3.8) to 5.3 (Qwen3.5-397B) | 6.3 h |
| M5 convention | 12,000 | 0.8 | 4.2 h |
| M6 claims | 9,000 | 0.6 | 3.1 h |
| M7 stubborn agent | 4,000 | 0.3 | 1.4 h |
| **Full program** | **≈ 121,000–134,000** | **≈ 9–15 (guard 20)** | **≈ 42–47 h** |

- **Credits are not the constraint; queue time is.** The account allows about 50 requests per minute, and only one
  paced job runs at a time. Rosen phase 2 is running now (at R2b; R6, R5 and later stages remain).
- **The ledger is at 86.8 of the 150 cap.**
  - Rosen should finish near 98.
  - If the tournament study (guard 25) and Chase E14 (2.3) are also approved, this study's full guard (20) brings
    the commitments to about 145. That fits under 150, but with little margin.
  - Tier 1 alone (guard 12) fits comfortably.
  - I will not go above 150 without your OK.
- **What can start now at no cost:** the engine, mock agents, the Hartnett reproduction and the pre-registration
  draft are all offline.

## 8. Risks and mitigations

- **Label or position bias at round 0.** M0 picks the least biased label pair. Order is randomised per call and the
  target is randomised per group, so a bias adds noise, not direction. The p = 0 controls detect any drift.
- **Agents inferring N or tracking individuals.** Neighbours are a fixed k = 5, anonymous and redrawn each round, and
  N is not stated. M1 checks that stating N changes nothing.
- **Determinism or ceiling effects** (near 0/1 kernels). The isotonic fit and the simulator handle 0/1 cells, and the
  forecast intervals carry the bootstrap uncertainty.
- **The Condorcet confound.** The primary outcome is consensus, plus within-group social amplification and an
  explicit independent-agent null.
- **Simulator error.** It must reproduce the Hartnett sign result and scripted mock-agent runs before the freeze.
- **Low power for H1.** Handled by the power rule; otherwise H1 is reported as an estimate.
- **Demand effects from the bonus framing.** ω is calibrated from behaviour, not from the stated numbers. The
  comprehension gate and the second cover story check this.
- **A reboot kills queues** (as on 19 Nov). Runners are restart-safe and skip groups already written.

## 9. Decisions for the owner

1. **Scope:** tier 1 only (M0–M4 on Gemma-31B), or the full program (adds M4b, M5, M6, M7).
2. **Credit guard:** 12 for tier 1 or 20 for the full program, within the existing 150 cap.
3. **Second population model (M4b):** allow Qwen3.5-397B if the rule picks it (about 5 credits), or restrict the
   choice to cheap models.
4. **Queue order** relative to Rosen phase 2 (running), Chase E14 and the tournament study, both of which are
   awaiting approval.
5. **Claude and GPT through the CLIs** for M1 kernels only (no Adaption credits, slow): include them or not.
