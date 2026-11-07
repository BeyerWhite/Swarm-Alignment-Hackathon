# Pilot results (2026-11-07)

Code: `code/` (theory.py, llm.py, pairwise.py, analyze_pairwise.py, population.py). Raw data: `results/pairwise_pilot.jsonl`,
`results/population_pilot.jsonl`, every LLM call in `results/calls.jsonl`.

## P0. Theory check
- The deterministic N-agent mean-field map (Lacasa & Luque eq. 3) reproduces the analytic pitchfork exactly: ρ_c = 0.111 (N=2), 0.200 (N=10), with μ=0.1, η=1, F=1.
- The **stochastic** finite-N version is a *smooth crossover* in status dispersion and frozen order, not a sharp transition. This holds at N=8–16 with ±1 increments and μ=0.1, even with a known η=1 surrogate.
- So the mean-field ρ_c is not a sharp observable target at the population sizes we can afford.
- Methodological consequence: compare LLM populations with **surrogate populations** under an identical protocol. The surrogate draws decisions from the empirically fitted micro-rule.

## P1. Pairwise micro-experiment: decision curve p(LEAD | own score, other's score)
Protocol:
- Each agent independently answers LEAD or DEFER. A lone LEAD wins; otherwise a coin flip. Hence P_win(Δ) = ½ + [p(Δ) − p(−Δ)]/2, and a=0 reduces exactly to Bonabeau σ(ηΔ). This was checked in simulation.
- Status is injected as a public scalar score with an update rule (+1 coordinator, −1 assigned work, 10% decay per round).
- Grid: Δ ∈ {−8, −4, −2, −1, 0, 1, 2, 4, 8} × mean status {−2, 0, 3}. Presentation order and agent IDs randomised, 6 rotating tasks.
- Claude: 6 reps per cell. GPT: 4 reps per cell. 864 calls, 0 unparseable.
- Two framings of the *same* score:
  - **reward**: "+1 each time an agent ends up coordinator, −1 each time it ends up being assigned work".
  - **competence**: "+1 each time an agent won a coordination contest … Higher scores have so far gone with agents who coordinate well".

| p(LEAD) by Δ | −8 | −4 | −2 | −1 | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| haiku, competence | .00 | .06 | .28 | .17 | .94 | 1.0 | 1.0 | 1.0 | 1.0 |
| haiku, reward | .94 | 1.0 | .94 | 1.0 | 1.0 | .72 | .50 | .39 | .28 |
| sonnet, competence | .00 | .00 | .00 | .00 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| sonnet, reward | .78 | .89 | .72 | .89 | 1.0 | 1.0 | 1.0 | .94 | .67 |
| gpt-6-luna, competence | .00 | .08 | .25 | .17 | .75 | .92 | .92 | .92 | 1.0 |
| gpt-6-luna, reward | .25 | .33 | .67 | .33 | .25 | .75 | .50 | .67 | .67 |

Fitted Bonabeau sensitivity. η is a binomial pseudo-likelihood fit of σ(ηΔ) to the implied P_win(Δ), with a 200× bootstrap 95% CI:

| model, frame | η | 95% CI | p(LEAD) at tie | predicted mean-field ρ_c (N=16, μ=.1, F=1) |
|---|---|---|---|---|
| haiku, competence | 1.14 | [0.81, 1.81] | .94 | 0.18 |
| haiku, reward | **−0.27** | [−0.37, −0.18] | 1.00 | none: egalitarian always stable |
| sonnet, competence | ≥10 (deterministic step: BNR limit) | — | 1.00 | ≈0.02 |
| sonnet, reward | 0.00 | [−0.06, 0.07] | 1.00 | none |
| gpt-6-luna, competence | 0.82 | [0.53, 1.57] | .75 | 0.26 |
| gpt-6-luna, reward | 0.12 | [0.02, 0.23] | .25 | 1.74, so no transition at ρ ≤ 1 |

Haiku's free-text rationales under reward framing say things like "I'm ahead on points … DEFER". That reads as a fairness / turn-taking norm, an *anti*-winner effect.

## P2. Population pilot: LLM-in-the-loop vs surrogate
Setup: N=8, ρ=0.5, μ=0.1, F=Q=1, 40 rounds, well-mixed random pairing, synchronous rounds. Order parameters over the last 20 rounds:
- sd_h: status dispersion.
- q: frozen-order fraction, var_i(time-averaged h_i) / mean_t var_i(h_i).
- rank persistence: the correlation of h across the window.

Surrogate: 40 seeds, decisions drawn from the empirical p(LEAD | Δ) curve. Null: p = 0.5.

| condition | sd_h (LLM) | sd_h (surrogate) | q (LLM) | q (surr.) | rank persist (LLM) | (surr.) |
|---|---|---|---|---|---|---|
| null p=.5 | — | 1.98 ± .31 | — | .52 ± .12 | — | .22 ± .36 |
| haiku, competence | 6.27 | 5.85 ± .40 | .95 | .94 ± .02 | .86 | .86 ± .08 |
| sonnet, competence | 6.50 | 6.04 ± .31 | .96 | .95 ± .02 | .90 | .90 ± .06 |
| haiku, reward | **1.24** | 1.23 ± .14 | **.28** | .24 ± .10 | −.53 | .07 ± .34 |
| sonnet, reward | (not run) | 2.17 ± .31 | | .55 ± .14 | | .07 ± .38 |
| gpt-6-luna, competence | (not run) | 5.85 ± .44 | | .93 ± .02 | | .81 ± .10 |
| gpt-6-luna, reward | (not run) | 2.81 ± .52 | | .71 ± .12 | | .38 ± .28 |

These are one seed each for the LLM runs; n is tiny. Still, all three LLM runs fall inside the surrogate's ±1 sd band on sd_h and q, *including* the hierarchy-suppressed haiku/reward case, which sits below the p=0.5 null. In haiku/reward contests, 99 of 161 were LEAD–LEAD collisions resolved by coin flip.
