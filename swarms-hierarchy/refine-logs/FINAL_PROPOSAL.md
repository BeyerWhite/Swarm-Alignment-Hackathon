# Final Proposal — Framing sets the sign of status reinforcement in LLM agent populations

## Problem anchor (frozen)
Under what conditions do leaders, persistent roles and division of labour emerge in a population of identical
LLM agents, and can we reproduce and *predict* this in a safe sandbox? We test whether the classical
hierarchy-formation models explain it: Bonabeau/Stauffer (via Lacasa & Luque 2005) and Ben-Naim–Redner 2005.
This is motivated by the METR (Aug 2026) incident, where one agent issued hundreds of assignments.

## Claim (defensible form)
We test whether **contextual framing reverses status-dependent coordinator choice** in identical LLM agents. We
also test whether **independently calibrated pairwise response kernels predict population hierarchy persistence
or suppression without population-level refitting.**

In plain language: we measure how a single agent reacts to a score gap in a two-agent contest, and fit the
physics model's "status sensitivity" η from that. We then predict whether a whole population develops a stable
pecking order, and check the prediction against real LLM populations. The way the score is described decides
whether agents defer to high scorers (η > 0: hierarchy can form) or take turns (η < 0: hierarchy is actively
suppressed).

## Method
1. **Contest protocol.**
   - Both agents answer LEAD or DEFER. A lone LEAD wins; otherwise a coin flip decides. The coordinator gets +1,
     the other −1, and all scores shrink 10% per round.
   - The win probability is P_win(Δ) = ½ + [p(Δ) − p(−Δ)]/2, so a symmetric logistic p reduces exactly to
     Bonabeau's σ(ηΔ).
   - Our protocol updates both contestants, so its mean-field threshold is ½ × Lacasa–Luque ρ_c.
2. **Calibration (micro).**
   - Measure p(LEAD | displayed own score, displayed partner score) on a grid: Δ ∈ ±{0.4, 1, 2, 4, 8, 16} and 0,
     crossed with mean status ∈ {−4, 0, 4}.
   - Framing conditions:
     - a numeric norm weight w ∈ {0, 25, 50, 75, 100};
     - a 2×2 wording factorial: verb ("won" vs "ended up as") × competence sentence (present vs absent).
   - Models: Claude Haiku (primary), Claude Sonnet and GPT-6-luna (replication).
3. **Kernels.** Four kernels are fitted:
   - flexible: Jeffreys-smoothed bilinear interpolation over the grid;
   - Bonabeau: σ(a + bΔ), reporting the implied local η = 4P′_win(0);
   - logistic3: adds a mean-status term;
   - null: status-blind, with the matched base LEAD rate.
   Uncertainty comes from Beta-posterior cell draws (flexible, null) or a row bootstrap (logistic kernels).
4. **Pre-registered macro predictions.**
   - Each kernel's predictive distribution is simulated under the *identical* population protocol: N=8, ρ=0.5,
     μ=0.1, 40 rounds, synchronous rounds, final-20-round window.
   - The predictions file is write-once and SHA256-stamped with the source and calibration hashes. It is
     written before any v2 population run.
5. **LLM-in-the-loop populations (macro).** Haiku, conditions {ended, ended+comp, norm50}, 4 seeds each.
6. **Evaluation.**
   - Primary metrics: status dispersion sd_h and frozen-order fraction q.
   - Kernel comparison: summed log predictive density (Gaussian KDE over predictive samples) across runs and
     both primary metrics. A kernel is "clearly better" if it wins by more than 3 nats.
   - Also reported: 90% interval coverage, standardised errors, and conditional winner log-scores on the
     observed population states.
   - Descriptive only: leader share, rank persistence, de Vries h′.
7. **Discriminating holdout.** 48 role-swapped contests (96 calls) at the displayed off-grid score pair where
   the flexible and Bonabeau kernels' implied win probabilities differ most. This is scored prospectively.
8. **Inequality-feedback probe (exploratory).**
   - The focal pair is placed at ranks 4–5 of an 8-agent leaderboard. Only the background spread changes: low
     (±1.5…2.5) vs high (±5…11).
   - Fixed-η Bonabeau predicts no change; Stauffer-type feedback predicts a steeper response under inequality.

## Non-claims
- No sharp phase transition: at N=8–16 the stochastic model gives a crossover.
- No claim about competence or productive division of labour: agents choose roles under externally maintained
  scores and never perform tasks.
- No claim of universality across models beyond what each model's calibration shows.
- Explicit-norm conditions (w) are instruction-following tests. The wording factorial carries the
  framing-causality claim.

## Sunday extensions (if time)
- Open delegation ledger (assignment hubs, as with PHASEONE[big]).
- Mixed-family populations as quenched disorder.
- BNR L = r with integer status and additive decline.
- Hysteresis from seeded unequal starts.
- Scalar-status sufficiency probe.
