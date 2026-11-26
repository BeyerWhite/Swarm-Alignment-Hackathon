# Informed minorities in LLM agent populations (Couzin): final report

Written 2026-11-26. Branch `couzin-minorities`. This replaces `COUZIN_INTERIM_REPORT.md`.

- **Pre-registration:** `COUZIN_PREREGISTRATION.md`, with amendments M-A1 (the Claude follow-up,
  `COUZIN_CLAUDE_FOLLOWUP.md`), M-A2 (budget meter) and M-A3 (evaluators test only complete arms). Lab log:
  `COUZIN_LOG.md`.
- **Runs:**
  - Adaption queue 2: 2026-11-24 08:51 to 2026-11-26 08:27.
  - Claude follow-up: 2026-11-21 to 2026-11-24. It needed 26 restarts through usage limits; no decision affected by
    a limit was kept.
- **Spend:** this study used 1.98 Adaption credits in all (estimated 12; guard 20). The shared ledger is at 130.5 of
  the 200 cap. The Claude follow-up used no Adaption credits.

## 1. Design in brief

A binary "travel together" game:
- Every agent earns 10 points for ending on the larger side.
- Informed agents also get a bonus ω × 10 for ending at their preferred site.
- Each round an agent sees the previous choices of 5 random other agents, anonymously. Group size is never stated.
- Round 0, then 5 rounds.

For each model:
- single decisions are measured by intervention (the kernel);
- calibration groups are played;
- a forecast is frozen before the test groups run.

The test groups were run on Gemma-4-31B (all experiments); on DeepSeek-v4-flash, which the frozen rule picked as the
model least like Gemma (M4b); and on Claude Haiku and Claude Sonnet (the M4b design). In every population there were
no invalid decisions or nearly none (at most 0.3%).

## 2. Headline results

1. **Who wins a fresh decision depends on how many agents are informed, not on what fraction.**
   - Target consensus rises with group size at a fixed informed fraction (H1). This is Couzin 2005's claim.
   - But 1 informed agent steers about 55% of groups at N = 5, 10 and 20 alike. 2 agents steer about 70% at N = 10,
     20 and 40. 4 agents steer 72–82%.
   - In an exploratory fit, consensus rises with the number of informed agents: +0.53 log-odds per doubling,
     SE 0.20. Given that number, group size adds nothing: −0.07, SE 0.16.
   - **Safety reading:** a fixed handful of agents with a private goal steers a swarm of any size from 5 to 40.
     Couzin's "the larger the group, the smaller the fraction needed" holds in its strongest form.
2. **Whether uninformed agents protect the majority or help a stubborn minority differs between models. The
   measured kernels forecast the direction for every model:**
   - **Claude Haiku:** uninformed agents restore the majority when they observe before committing. The minority's
     win rate falls from 0.38 to 0.12 to 0.00.
   - **Claude Sonnet:** uninformed agents *help* the strong minority when they commit at the start (0.04 → 0.17 →
     0.33), as forecast, with all 5 settings inside their intervals. This is Hartnett's highly nonlinear regime.
   - **DeepSeek:** uninformed agents help the minority strongly under both protocols. The minority wins 0.62 at
     24 uninformed agents, a reversal of Couzin 2011. The forecast had the right sign but far too small a size.
   - **Gemma-31B:** the 4 "strong" agents never win against 7 weak ones (0.00). Only at 24 uninformed agents with
     simultaneous start does the minority win a few groups (0.17; forecast 0.09).
   - The single nonlinearity index does *not* order these signs. DeepSeek (a = 0.3) and Sonnet (a = 55) both show
     "uninformed help the minority". The full measured kernel, including how stubborn informed agents are, is what
     forecasts the sign.
3. **A strongly opinionated minority does not dictate in most LLM populations.**
   - Couzin 2011's premise is a minority that will not move. It failed for Gemma, Haiku and DeepSeek: even a six-fold
     bonus does not make their informed agents hold out against a unanimous neighbourhood.
   - Sonnet is the exception (stubborn at ω = 2), and it is the model where uninformed agents empowered the minority.
4. **The forecasts beat every null but were overconfident.** By Brier score, the measured-kernel forecast beat the
   independent-agent, linear-voter and N-blind nulls (H8). But only 24 of 33 arms fell inside their 90% intervals.
   The misses are mostly at small N in M3 (the forecast was too optimistic about consensus) and in M5 at N = 40.

## 3. Main study, Gemma-4-31B (and DeepSeek for M4b)

**Kernels (M1).**

| Model | Nonlinearity index a | ω rule (weak, strong) | Flags |
|---|---|---|---|
| Gemma-4-31B | 3.5 | 0.5, 6 (fallback) | weak and strong not behaviourally distinct |
| Gemma-4-26B | 4.6 | 0.5, 6 (fallback) | no level met either criterion |
| Qwen3.8-27B | 0.55 | 0.5, 6 (fallback) | strong criterion not met |
| Qwen3.5-397B | 0.40 | 0.5, 6 (fallback) | neither criterion met |
| GLM-5.2 | 0.89 | 0.5, 6 (fallback) | neither criterion met |
| DeepSeek-v4-flash | 0.34 | 0.5, 6 (fallback) | neither criterion met |

No Adaption model reached the registered "strong" criterion: staying at its site at least 90% of the time against a
unanimous neighbourhood. The ω fallbacks were therefore used throughout, which weakens the strong-against-weak
contrasts (H4, M4).

**Tests.** Holm correction within families; p values are one-sided.

| H | Result | Verdict |
|---|---|---|
| H1: N-scaling of consensus at fixed p | slope +0.48 per doubling of N, p = 0.002 (Holm 0.007) | Supported in direction. Registered as an *estimate*: the power rule gave 0.50 at 40 groups, below 0.7 |
| H2 (secondary): one informed agent, N 5 → 20 | 0.60, 0.53, 0.60; flat | Not supported. A fixed number steers equally (see §2.1) |
| H3a: social amplification > 0 | +0.01, CI lower bound −0.03 | Not supported |
| H3b: consensus above the independent-agent null | +0.59, lower bound 0.55, Holm 0.001 | Supported |
| H4a, H4b: strong against weak ω | differences −0.03 and −0.13 | Not supported (ω levels not behaviourally distinct) |
| H5sim, H5obs: uninformed agents reduce minority wins | the minority wins 0.00 in 6 of 7 arms; logistic slope not estimable (separation) | Not supported: nothing to reduce |
| H6: majority restored (observe-first, n_u ≥ 12) | P(minority) 0.02, upper bound 0.09 | Supported, but trivially: the majority also won without uninformed agents |
| H7: cohesion at n_u 24 against 6 | 0.68 → 0.82 (sim), 0.74 → 0.82 (obs); splits fall | No incoherence: uninformed agents *raise* cohesion |
| H8: forecast against nulls | Brier 0.159 against 0.398 (independent), 0.224 (linear), 0.162 (N-blind); all upper bounds < 0 | Supported. Coverage 24/33 |
| H9: M4b (DeepSeek) sign as forecast | sim: +0.058 against forecast +0.007 (p 0.012); obs: +0.045 against +0.007 (p 0.040); Holm (family C) 0.16 | Same sign in both protocols; not significant after Holm; size far beyond the forecast (3 of 5 arms outside) |
| H10: convention critical mass not lower at N 40 | consensus N40 minus N10 = +0.30 (upper bound 0.40 > margin 0.10) | Not supported: larger groups overturned the convention *more* easily (N 40, p 0.2: 0.60 against a forecast of 0.09) |
| H11a, H11b: visible claims raise consensus | +0.04 and −0.04 | Not supported: claims change nothing |
| M7 (secondary): one DeepSeek agent steers | follow 0.58 / 0.57 against 0.58 / 0.43 for a random Gemma agent | Not supported |

**M3 cells: target consensus, observed against forecast.**

| Informed fraction | N = 5 | N = 10 | N = 20 | N = 40 |
|---|---|---|---|---|
| 0.05 | | | 0.60 (0.62) | 0.68 (0.65) |
| 0.1 | | 0.53 (0.72) ✗ | 0.72 (0.78) | 0.72 (0.83) |
| 0.2 | 0.60 (0.90) ✗ | 0.70 (0.92) ✗ | 0.82 (0.97) ✗ | |

**Convention start (M5).** Every agent starts at the other site.
- At N = 10: 2 informed agents overturn the convention in 5% of groups and 4 in 95%.
- At N = 40: 8 informed agents overturn it in 60% and 16 in 100%.
- Overturning a convention takes far more informed agents than a fresh decision does. Unlike fresh decisions, the
  fraction matters as well as the number, since 4 of 10 succeed more often than 8 of 40. I had predicted that the
  critical fraction would not fall with N. It did fall, so H10 failed.

**Label effect at the group level.** In single round-0 decisions the models pick the first-named option. The naming
order is randomised, so that is a fair coin. Yet ORION won 64% of M3 consensus outcomes (230 against 131).
- So a small label asymmetry in later decisions is amplified by the group.
- Because the target site was randomised per group, this adds noise, not direction.
- It explains the one off-forecast p = 0 control (N = 10: ORION was the target in 8 of 12 groups).
- The simulator folds label symmetry after round 0, which may be part of why it was overconfident.

## 4. Claude follow-up (M4b design; amendment M-A1)

| Uninformed agents | Haiku, sim | Haiku, obs | Sonnet, sim | Sonnet, obs |
|---|---|---|---|---|
| 0 | 0.38 (0.68 ✗) | (same arm) | 0.04 (0.03) | (same arm) |
| 12 | 0.38 (0.52) | 0.12 (0.27) | 0.17 (0.22) | 0.00 (0.01) |
| 24 | 0.33 (0.50) | 0.00 (0.15) | 0.33 (0.34) | 0.00 (0.00) |

Values are P(minority wins), with the frozen forecast in brackets; 24 groups each.

- **Haiku** (a = 4.7; informed agents give in readily):
  - F1obs: forecast sign confirmed, p = 0.002 (Holm 0.005).
  - F1sim: same sign, but not significant (p = 0.38).
  - F3 (majority restored): supported (0.06; upper bound 0.15).
  - Coverage 4/5.
- **Sonnet** (a = 55; stubborn at ω = 2):
  - F1sim: forecast sign confirmed, p = 0.007 (Holm 0.014). Uninformed agents help the minority.
  - F1obs: flat forecast, so coverage is the test.
  - F3: supported (0.00; upper bound 0.05).
  - Coverage 5/5.

## 5. What this means

- **Couzin 2005 holds for Gemma, and more strongly than expected.** A few hidden agents steer the group, and the
  absolute number of them is what matters, not the fraction. For a deployer, "keep the compromised fraction small"
  does not protect a large swarm. It is the count that has to stay low.
- **Couzin 2011 depends on the model and on the start protocol.** Neutral agents can protect the majority (Haiku,
  observe-first), do nothing (Gemma, where the minority is powerless anyway), or help a stubborn minority (Sonnet
  with simultaneous start; DeepSeek with both).
  - A ~2,000-call kernel measurement forecast the direction in each case.
  - Hartnett et al.'s point, that the sign is set by the response function, carries over to LLMs. A single
    nonlinearity number is not enough to predict it; the informed agents' stubbornness matters too.
- **Method.** This is the fourth study in the repository to forecast group outcomes from measured single decisions,
  freezing each forecast before the groups ran. The forecasts beat every null and got the direction right. They
  were systematically overconfident on levels, which is the same lesson as in Rosen.
  - The next version should widen its intervals by a calibration factor measured on held-out calibration groups.
  - It should also stop folding label symmetry after round 0.

## 6. Limits

- One primary model (Gemma-31B) for M3, M5, M6 and M7.
- The ω fallbacks were used for every Adaption model.
- H1 is reported as an estimate under the registered power rule.
- The "number, not fraction" finding (§2.1) is exploratory: it was not a registered test.
- The Claude follow-up was run as an addition to the pre-registration (M-A1), not as part of it.
- GPT and Qwen were not run. They were to follow if M4b looked interesting, which it did.
