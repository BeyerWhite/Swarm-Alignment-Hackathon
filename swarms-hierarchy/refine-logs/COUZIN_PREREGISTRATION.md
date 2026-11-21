# Informed minorities in LLM agent populations (Couzin) — pre-registration

Frozen before any Adaption call of this study (stamp: `results/couzin_prereg_freeze.json`, check with
`code/couzin_freeze.py check`; the queues refuse to start if a stamped file changes). Design rationale, literature
and cost reasoning are in `COUZIN_PROPOSAL.md`. Changes after the stamp are amendments M-A1, M-A2, … in
`COUZIN_LOG.md`, with a new stamp.

## 0. Approval and owner decisions

- 2026-11-20: the owner chose the **full program** (M0–M4 plus M4b, M5, M6, M7).
- The other four decisions in proposal §9 were not answered. These defaults apply until the owner says otherwise:
  1. **Credit guard:** 20 credits for this study's own spend, measured by `code/couzin_budget.py` from this
     worktree's call log. Every stage also checks that the shared ledger plus the stage estimate stays at least 1
     credit below the owner's cap (150). The cap is never raised.
  2. **M4b model:** cheap models only (Gemma-4-26B, Qwen3.8-27B, DeepSeek-v4-flash), not Qwen3.5-397B.
  3. **Queue order:** after every queue already waiting on 2026-11-21 (RT, Chase E14, tournament queue 1, the
     hierarchy G0 gate). Queue 2 is launched by hand after queue 1 has been inspected.
  4. **Claude and GPT kernels:** not included.

## 1. Game (code: `code/couzin.py`)

- N agents, two sites. Round 0, then rounds 1..T with **T = 5**.
- Each round every active agent names a site. It sees:
  - its own previous choice;
  - the round number;
  - the previous-round choices of **k = min(5, active others)** agents drawn without replacement each round,
    shown as anonymous counts.
- N is never stated, except in the M1 "N stated" robustness cells.
- **Payoffs:** C = 10 to every agent on the larger side after round T. An exact tie pays nobody. An informed agent
  also gets B = ω·C if it ends at its preferred site. ω ∈ {0.5, 2, 6}, so B ∈ {5, 20, 60}.
- **Uninformed agents** are told only about C. The *aware* variant (M1 only) adds that some agents may know of a
  bonus.
- **Labels:** chosen in M0 from KESTREL/OSPREY, MAPLE/CEDAR and ORION/LYRA. The order in which the two labels are
  named is randomised per call. Which label is the target (A) is randomised per group and per kernel draw.
- **Replies:** parsed with `llm.parse_choice`; up to 3 calls per decision. If none parses, the agent keeps its
  previous choice (a random one at round 0) and the decision counts as invalid.
- **Protocols:**
  - *sim*: every agent chooses at round 0.
  - *obs* (observe-first): uninformed agents first choose at round 1, seeing only agents that chose at round 0.
- **Convention start (M5):** every agent is at B before round 1, and rounds 1..T are played.
- **Claims (M6):** a visible informed agent is reported as having "said it knows which site is better", and
  informed agents are told the others can see that.
- **Group outcomes:**
  - `consensusA`: final share at A ≥ 0.8. This is the primary outcome for M3, M5 and M6.
  - `majA`: final majority at A, with ties scored 0.5. This is the primary outcome for M4 and M4b, where A is the
    strong minority's site.
  - `cohesion`: the final majority share.
  - `split`: cohesion below 0.8.
  - `amp`: majority at A at the end minus majority at A among the round-0 choices.
  - `t_consensus`: the first round at which consensus is reached.
  - M7 only: `follow_extra` (the final majority equals the DeepSeek agent's round-0 site) and `follow_base` (the
    same for one Gemma agent drawn by seed).

## 2. Models and stages

The six models are Gemma-4-31B (primary), Gemma-4-26B, Qwen3.8-27B, Qwen3.5-397B, GLM-5.2 and DeepSeek-v4-flash.
All run with reasoning off, temperature 1 and 16 output tokens.

| Stage | What | Size |
|---|---|---|
| M0 | 6 payoff items × 20 draws × 6 models; label screen: 3 pairs × 2 orders × 20 draws × 6 models | 1,440 |
| Pilot | 2 groups M3_N10_p0.2 and 1 group M4_obs_nu6 (primary model; pilot files, never tested) | ≈ 220 |
| M1 | 70 cells (see below) × 20 draws for every gated model, 40 for the primary; robustness sets on the primary × 20 | ≈ 12,700 |
| M2 | Calibration: M2_N10_p0.1 × 24 groups, M2c_obs_nu6 × 8 groups (primary), M2c_obs_nu6 × 8 groups (M4b model) | ≈ 3,000 |
| M3 | 8 cells (N, p) = (5, .2), (10, .1), (10, .2), (20, .05), (20, .1), (20, .2), (40, .05), (40, .1), strong ω, G3 groups each; p = 0 controls at N 10 and 40 (12 groups each); weak ω at N 20, p .1 (G3 groups) | 30k–48k |
| M4 | 4 strong (A) against 7 weak (B), n_u ∈ {0, 6, 12, 24}, sim and obs (n_u = 0 once), 24 groups each; reference 4 strong against 7 strong at n_u = 0 (sim) and 12 (obs) | ≈ 28,000 |
| M4b | as M4 on the rule-selected model: sim n_u 0, 12, 24; obs n_u 12, 24; 24 groups each | ≈ 18,000 |
| M5 | convention start, N 10 and 40 at the two frozen levels of p, 20 groups each | ≈ 10,000 |
| M6 | claims: N 10 with 1 informed agent, N 40 with 2, 30 groups each | ≈ 9,000 |
| M7 | one uninformed DeepSeek agent in uninformed Gemma groups, N 10 and 20, 30 groups each | ≈ 5,400 |

**M1 cells.** Coding is relative to the counted option A; for informed agents A is the preferred site.
- Uninformed:
  - round 0 (1 cell);
  - own last A or B × j = 0..5 visible at A × t ∈ {1, 4} (24 cells);
  - first choice after observing, j = 0..5 at t = 1 (6 cells).
- Informed, at each ω: round 0, plus own last A or B × j = 0..5 at t = 2 (13 cells per ω).
- Robustness sets (primary model):
  - the "team" cover story for all 70 cells;
  - the aware variant for the 31 uninformed cells;
  - N stated as 10 or 40 (own × j at t = 2);
  - one visible claimant on A (j ≥ 1) or on B (j ≤ 4).

## 3. Frozen rules (applied by code, write-once outputs)

1. **M0 gate** (`couzin_kernel.py gate` → `couzin_m0_gate.json`):
   - A model passes with at least 80% correct payoff answers, at least 90% parseable answers, and at least 90%
     parseable game replies.
   - The label pair is the one that minimises, over passed models, the largest |P(first label) − 0.5|, with order
     effects averaged out.
   - If the primary model fails, the queue stops and the owner decides.
2. **ω rule** (per model, `couzin_kernel.py rules` → `couzin_m1_rules.json`):
   - *strong* = the smallest ω with P(stay at the preferred site | own preferred, 0 of 5 there) ≥ 0.9.
   - *weak* = the largest ω with P(follow a 4-of-5 majority away | own preferred, 1 of 5 there) ≥ 0.5.
   - Fallbacks: strong 6 and weak 0.5. If weak ≥ strong, weak becomes the next lower level. The flags are recorded.
3. **Nonlinearity index:** the MLE exponent a in logit P(A) = a·logit((j+½)/6) + β·own, over uninformed decisions
   with an own previous choice. The contrast P(A | 4 of 5) − P(A | 3 of 5) is also reported.
4. **M4b model:** among the cheap candidates that passed M0, the one with the largest |a − a_primary|. Ties go to
   the lower cost.
5. **Kernel estimator:**
   - Uninformed tables are folded by symmetry (own B, j ↔ own A, 5 − j).
   - Each table is monotone in j (weighted pool-adjacent-violators on the **unshrunk** proportions s/n; empty cells take their neighbours' value). No shrinkage toward 0.5: in the offline check a (s + ½)/(n + 1) floor, i.e. about 2% defection from a 5-of-5 majority, broke simulated consensus.
   - The M2 calibration decisions are pooled with the M1 counts. Decisions with k < 5 are not used.
   - Phases: early (t ≤ 2) and late (t ≥ 3) for uninformed agents. Informed tables pool all t ≥ 1.
6. **Simulator** (`couzin_forecast.py`):
   - Same game. j is hypergeometric given the counts. For k < 5, j is mapped to j·5/k with linear interpolation.
   - Claims use the claimant table when visible claimants are on one side only.
7. **Simulator check before any forecast** (Hartnett et al. 2016):
   - Hartnett's local rule is P(+) = ½[1 + tanh(b·h)/tanh(b)], with h = (ω n₊ − n₋)/(ω n₊ + n₋).
   - Setup: k = 4 random neighbours, N = 200, 150 rounds, 300 groups. Biased agents are 60% weak majority
     (ω₊ = 1.5) and 40% strong minority (1/ω₋ = 2). The uninformed share ρ₀ ∈ {0, .3, .6, .9}.
   - The check passes iff the slope of P(majority wins) over ρ₀ is > 0 at b = 0.3 and < 0 at b = 10. If it fails,
     no forecast is written.
   - Offline result on 2026-11-20, with no LLM data: 0.04 → 0.38 at b = 0.3 (slope +0.38) and 0.92 → 0.59 at
     b = 10 (slope −0.38). **Passed.**
   - **Interpretation, fixed now:** in this finite-time setting, uninformed agents mainly pull the outcome toward a
     coin flip. Nonlinearity decides who wins without them. So "uninformed agents restore the majority" in Couzin's
     strong sense needs P(minority wins) to fall **below 0.5**. That is H6, and H5 alone does not establish it.
8. **Forecast** (`couzin_forecast.py freeze` → `couzin_forecast.json`, written once, after M1 and M2 and before M3):
   - 200 bootstrap kernel draws (each cell resampled binomially at its observed proportion, then the monotone fit), with 100
     simulated groups per draw for every planned arm.
   - Each arm gets a point forecast and a 90% interval for its mean over its planned number of groups (a kernel draw,
     then groups from that draw).
   - The independent-agent null (round-0 probabilities in every round) and the linear-voter null (a = 1, fitted by
     MLE to the same decisions) are computed for every arm.
   - The table counts are hashed into the file.
9. **M3 group count (power rule):** G3 is the smallest of 30, 35, 40 that gives at least 80% simulated power for H1
   at the forecast cell rates. Otherwise G3 = 40. If power at 40 is below 0.7, H1 is reported as an estimate, not a
   verdict.
10. **M5 levels:** p_c is the smallest p in 0.1…0.9 with forecast consensus ≥ 0.5 at N = 10 under the convention
    start. The levels are (max(p_c − .1, .1), that level + .2), capped at .9; if no p reaches 0.5, they are
    (0.8, 0.9). The same fractions are used at N = 40.
11. **N-blind null:**
    - For M3 strong cells: the mean forecast over the cells with the same p.
    - For M4 arms: the mean forecast over the arms of the same protocol, with n_u = 0 included in both.

## 4. Hypotheses and tests (`code/couzin_eval.py`)

The unit is the group. Bootstrap tests resample groups within arms (5,000 resamples). All tests are one-sided in the
stated direction at α = .05, with Holm correction within each family. Calibration and pilot groups are never tested.

| Family | H | Test | Supported if |
|---|---|---|---|
| A | H1 | logistic `consensusA ~ log2 N + p fixed effects`, 8 M3 cells, Wald z | slope > 0 (verdict or estimate per rule 9) |
| — | H2 (secondary) | single-informed diagonal N5 p.2, N10 p.1, N20 p.05: slope on log2 N | slope < 0 |
| A | H3a | mean `amp` over M3 cells | > 0 |
| A | H3b | `consensusA` minus the independent-agent null, group-weighted over M3 cells | > 0 |
| A | H4a, H4b | N20 p.1, strong minus weak ω: `majA` (H4a) and `split` (H4b) | both > 0 |
| B | H5sim, H5obs | logistic `majA ~ n_u` per protocol (M4, n_u = 0 shared) | slope < 0 |
| B | H6 | observe-first, n_u 12 and 24 pooled: P(minority wins), Wilson one-sided 95% upper bound | < 0.5 |
| — | H7 | cohesion and split at n_u 24 against 6 | descriptive |
| — | H8 | coverage of the 90% intervals over all arms (primary outcome per arm); Brier scores over M3 + M4 groups against the independent, linear-voter and N-blind nulls | the model's Brier is lower than each null's (95% upper bound of the difference < 0) |
| C | H9 | M4b, per protocol: sign of the observed slope of `majA` in n_u equals the sign of the forecast slope; arms inside intervals reported | same sign (one-sided z in the forecast direction; p = the larger of the two protocols) |
| C | H10 | M5: mean over the two levels of consensus(N40) − consensus(N10) | 95% upper bound < 0.10 |
| C | H11a, H11b | M6 minus M3 matched arm (N10: M3_N10_p0.1; N40: M3_N40_p0.05) on `consensusA` | > 0 |
| — | M7 (secondary) | `follow_extra` − `follow_base` | > 0 |

**Reading H8 null (iii).** If the linear voter forecasts as well as the measured kernel, the nonlinearity mechanism
does not explain LLM groups, whatever H1 and H5 show.

## 5. Queues, guards and restarts

- Both queues are started through `code/couzin_launch.sh <1|2> <cap> [pid …]`. The launcher waits for the listed
  pids, then until no other study's stage script (`bash code|scratch/…queue[N].sh` or `…_q[N].sh`) and no
  `chase_paced.py` job runs, checked twice, 2 minutes apart. Then it execs the queue.
  - The launcher's name deliberately does not match the "queue" pattern that other studies' wrappers wait on.
  - Its busy check ignores waiting `bash -c` wrappers, so two studies cannot deadlock waiting for each other.
- `code/couzin_queue1.sh`: M0, gate, pilot.
- `code/couzin_queue2.sh`: M1 → rules → M2 → Hartnett check and forecast → M3 → M4 → M4b → M5 → M6 → M7 →
  evaluation. It is launched by hand after queue 1 has been inspected.
- **Guards** before every stage:
  - shared ledger + estimate < cap − 1, with cap 150;
  - this study's spend + estimate ≤ 20 (`code/couzin_budget.py`).
- **Stage estimates (credits):** M0 0.3, pilot 0.05, M1 2.5, M2 0.3, M3 3.5, M4 2.0, M4b 1.2, M5 0.8, M6 0.7,
  M7 0.4. Total about 11.8, within the 20 guard.
- **Restarts:** runners append rows and skip what is already written. Write-once steps are skipped when their file
  exists.
- **Offline checks** were run before the stamp, all with scripted agents and no API calls (`COUZIN_LOG.md`, M-0):
  - `DRY=1` runs of both queues;
  - `code/couzin_offline_check.py` (engine against simulator with exact tables);
  - `couzin.py mock`.
