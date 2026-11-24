# Informed minorities in LLM agent populations (Couzin): interim report

As of 2026-11-24 00:20. Branch `couzin-minorities`. This report covers what has run so far:
- **Main study:** only queue 1 (the gate and a 3-group pilot). The population experiments M1–M7 are queued and have
  not started.
- **Claude follow-up (amendment M-A1):** complete for Claude Haiku. Claude Sonnet has finished its gate, kernel,
  calibration, frozen forecast and the first of five settings.

Nothing below is a final result of the main study.

## 1. What was tested

Couzin et al. (2011) claim that a strongly opinionated minority dictates a group's choice, and that adding agents
with no preference ("uninformed") hands control back to the numerical majority. Hartnett et al. (2016) showed that
the direction of the uninformed effect depends on how nonlinear agents' response to their neighbours is. That
response can be measured, so the outcome can be forecast before the groups run.

- **The game.** A binary "travel together" game.
  - Every agent earns 10 points for ending on the larger side.
  - Informed agents also get a bonus B = ω × 10 for ending at their preferred site.
  - Each round an agent sees 5 random other agents' previous choices, anonymously. Group size is never stated.
- **The test design (M4b).** 4 strong-minority agents and 7 weak-majority agents, plus n_u = 0, 12 or 24 uninformed
  agents, all the same model, 24 groups per setting.
  - *Simultaneous:* uninformed agents also choose at round 0.
  - *Observe-first:* uninformed agents first choose at round 1, after seeing round 0.
- **Method.** For each model:
  - single decisions measured by intervention (the kernel: 70 cells × 20 draws);
  - 8 calibration groups;
  - a forecast frozen before its groups ran;
  - the groups, scored against the forecast.
- All prompts, rules and tests were hash-stamped beforehand (`COUZIN_PREREGISTRATION.md`,
  `COUZIN_CLAUDE_FOLLOWUP.md`).

## 2. Main study so far (queue 1, Adaption models)

- **Gate.** All six models passed the payoff questions and the label check, with 0.82–1.00 correct and at least
  0.97 parseable. The frozen label pair is ORION / LYRA.
- **Position bias at round 0.** Most models choose whichever option is named first: P = 1.00 for Gemma-31B,
  Gemma-26B and Qwen3.5-397B, and 0.70–0.97 for the others. The naming order is randomised per call, so round-0
  choices behave as a fair coin, not as a label bias. Claude Sonnet shows the same pattern (1.00); Haiku shows it
  at 0.85–0.97.
- **Pilot** (3 groups, Gemma-31B, not tested). No invalid decisions. Both groups with 2 of 10 informed agents
  reached consensus on the informed agents' site, at rounds 2 and 3. In the 4-against-7 group with 6
  observe-first uninformed agents, the weak majority won.
- **Cost.** Queue 1 spent 0.057 credits. The study's budget meter had reported 1.66; that bug is fixed as amendment
  M-A2 and described in `COUZIN_LOG.md`.

## 3. Claude Haiku (complete)

**Kernel: close to majority rule.**
- Nonlinearity index a = 4.7 (a linear voter is a = 1).
- An uninformed agent that last chose A stays at A with probability 0.06 when 2 of 5 neighbours are there, and
  0.97 when 3 of 5 are.
- The bonus is weak against conformity. Even at B = 60 against C = 10, an informed agent stays at its site only
  53% of the time when all 5 visible agents are elsewhere. So no level met the registered "strong" criterion, and
  the frozen fallback (ω = 6) was used. At ω = 0.5, informed agents follow a 4-of-5 majority every time.

**Groups** (24 per setting; no invalid decisions):

| Uninformed agents | Simultaneous: minority wins | Forecast [90%] | Observe-first: minority wins | Forecast [90%] |
|---|---|---|---|---|
| 0 | 0.38 | 0.68 [0.46, 0.88] ✗ | (same arm) | |
| 12 | 0.38 | 0.52 [0.33, 0.71] ✓ | 0.12 | 0.27 [0.08, 0.46] ✓ |
| 24 | 0.33 | 0.50 [0.33, 0.71] ✓ | 0.00 | 0.15 [0.00, 0.33] ✓ |

**Registered tests:**
- **F1obs (forecast sign):** supported. The forecast change from 0 to 24 uninformed was −0.53. The observed slope
  is −0.15 per agent on the logit scale, 90% CI [−0.24, −0.07], p = 0.0016 (Holm over Haiku's two F1 tests:
  0.003).
- **F1sim:** not supported. The forecast was −0.18; the observed slope is flat (−0.008, CI [−0.05, 0.03]),
  p = 0.38.
- **F2 (uninformed agents reduce minority wins, the H5 analogue):** supported for observe-first, not for
  simultaneous.
- **F3 (majority restored, the H6 analogue):** supported. With observe-first and 12 or more uninformed agents, the
  minority wins 0.06 of groups (one-sided 95% upper bound 0.15 < 0.5).
- **Coverage:** 4 of 5 settings inside their forecast intervals.
- **Cohesion** rises with uninformed agents: from 0.74 to 0.91 (simultaneous) and from 0.81 to 0.88
  (observe-first). There was no sign of the incoherence Couzin et al. predicted at large n_u.

**Reading.**
1. **The uninformed buffer works for Haiku, but only when uninformed agents observe before committing.** If they
   commit at round 0 with no information, they change almost nothing. The design simulations predicted that the
   start protocol decides the outcome; this is the first LLM evidence for it.
2. **The strong minority does not dictate, even with no uninformed agents** (0.38 < 0.5). The forecast expected
   0.68, and this is the one miss. Within real rounds, Haiku's informed agents gave up their site more readily than
   the single-decision measurements implied, which is the same in-context gap the Rosen study found. The reason is
   that Haiku's conformity outweighs even a six-fold bonus, so the premise of Couzin 2011 (a minority that will not
   move) does not hold.
3. **Safety reading.** For a strongly conformist model, a minority with a private goal is not enough to steer the
   group, and neutral agents that watch before committing make it harder still.

## 4. Claude Sonnet (in progress)

**Kernel: essentially deterministic majority rule.**
- Index a = 55. Uninformed agents follow any 3-of-5 majority every time.
- Unlike Haiku, a moderate bonus already makes Sonnet stubborn. At ω = 2 it stays at its site 100% of the time
  against 5 of 5. So the frozen rule set strong = 2 and weak = 0.5.

**Frozen forecast: the opposite sign to Haiku.**
- The forecast says the strong minority almost never wins with no uninformed agents (0.03).
- Under simultaneous start, adding uninformed agents *helps the minority*: 0.22 at 12 and 0.34 at 24. This is
  Hartnett's highly nonlinear regime.
- Under observe-first, the forecast is flat at 0, so there is no sign test, only coverage.

**Observed so far:** with no uninformed agents, the minority wins 0.04 (24 groups; forecast 0.03 [0.00, 0.08]),
inside the interval. The four remaining settings (about 14,700 calls) decide the most interesting test in the
follow-up: whether the sign of the buffer effect flips between two models as their measured kernels predict.

**Usage limits.** The Sonnet run hit the Claude usage limit after about 40 minutes and stopped cleanly, as
registered. It was resumed at 00:18 on 24 Nov. It will probably need several usage windows to finish.

## 5. What comes next

- **Sonnet:** the remaining four settings, then the final follow-up evaluation (`couzin_followup.py eval`).
- **Evaluator fix before that evaluation.** The current script computes a test even when a model has data for only
  one setting, and that meaningless test enters the Holm correction. It needs a small amendment to require all
  three n_u levels. The Haiku numbers above use Holm over Haiku's own tests.
- **Main study queue 2:**
  - M1 kernels on six models → M2 calibration → frozen forecast → M3 N-scaling, M4 buffer, M4b, M5 convention,
    M6 claims, M7 stubborn agent.
  - Size: about 120k calls, about 40 h, about 12 credits.
  - It waits behind the response-threshold rest-framing run and Rosen phase 3. The ledger is at about 123 of the
    200 cap.
- **Gemma comparison.** Once M4 and M4b run, the Haiku and Sonnet results can be set against Gemma's. The question
  is whether the sign of the uninformed effect tracks the measured nonlinearity across all models.
