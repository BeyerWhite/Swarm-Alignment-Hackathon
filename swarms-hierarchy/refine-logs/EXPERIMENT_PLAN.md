# Experiment Plan (2026-11-07)

Compute: subscription CLIs only. Claude runs at about 2 calls/s combined at current load. GPT-6-luna is capped at
1,500 calls in total; about 540 are used so far. No GPUs. "Budget" below means LLM calls.

| Block | Purpose / claim | Runs | Calls (est.) | Order | Status |
|---|---|---|---|---|---|
| M0 theory check | Mean-field pitchfork reproduces analytic ρ_c; the stochastic finite-N model is a crossover | sims | 0 | 1 | DONE |
| M1 pilot | Framing flips the sign of η; LLM populations track surrogates | 864 + ~950 | ~1.8k | 2 | DONE |
| M2 calibration v2 | Kernel per model × condition, neutral wording, wider grid, displayed scores | Haiku 9 cond; Sonnet 5; luna 3 | 3,540 | 3 | RUNNING |
| M3 inequality probe | Stauffer-type feedback? (exploratory) | Haiku ended+comp, norm50 | 192 | 3 | RUNNING (with M2) |
| M4 freeze predictions | Write-once, hash-stamped predictive distributions; 4 kernels × configs | sims | 0 | 4 | after M2 |
| M5 holdout | Off-grid pair maximising the flexible vs Bonabeau P_win difference; 48 role-swapped contests | Haiku | 96 | 5 | after M4 |
| M6 populations | Micro→macro test: Haiku {ended, ended+comp, norm50} × 4 seeds, N=8, ρ=0.5, μ=0.1, T=40 | 12 | ~3,840 | 6 | after M4 |
| M7 evaluation | Log predictive score per kernel, coverage, z-scores, conditional winner log-score | — | 0 | 7 | after M6 |
| M8 replication (optional) | Sonnet ended+comp and norm50 populations, 2 seeds; luna ended+comp 1 seed | ≤5 | ≤1,600 | 8 | if time |

**Primary success criteria (pre-declared):**
1. **Micro, Haiku.** The local η changes sign between `ended` and `ended+comp`, with bootstrap 95% CIs that
   exclude 0 on opposite sides. This is the wording-only reversal.
2. **Macro.** For each Haiku condition, the observed mean sd_h and q across the 4 seeds lie inside the
   flexible kernel's 90% predictive interval for the 4-seed mean. The flexible kernel's summed log predictive
   density also beats the null by more than 3 nats.
3. **Bonabeau sufficiency.** The Bonabeau kernel is within 3 nats of the flexible kernel. Otherwise we report
   that the one-η reduction fails.

**Kill / pivot conditions:**
- If populations miss the flexible predictions systematically, the micro→macro claim fails. We report that,
  together with the conditional winner log-score diagnostic.
- If `ended` vs `ended+comp` show no sign change, the framing-reversal claim is restricted to the explicit-norm
  conditions.

## Amendments (logged before the affected data were looked at)
- **A1 (strict parser):** the code review found 4/2,592 calibration replies mislabelled. Calibration is now re-parsed
  from raw responses. The original Sonnet freeze (sha256 4f6cb046…) is archived in `results/frozen_original/`
  and superseded by `predictions_v2_claude_a1.json`. Sonnet population runs had already finished but were **not
  inspected** before the amended freeze.
- **A2 (seed-mean evaluation):** one calibration draw is shared across the seeds of a replicate, and displayed scores
  are replayed exactly from the event logs.
- **A3 (Haiku conditions):** {ended, ended+comp, norm50} → {norm0, ended, ended+comp, norm100} × 3 seeds, the same
  ~3.8k calls. Reason: the v2 calibration showed η ∈ [0.03, 0.35] for the original three, too narrow to
  discriminate; the new set spans η −1.45 … +2.22. This was chosen from calibration only, before any Haiku
  population data existed.
- **A4 (new resource, 2026-11-07 ~15:45):** Adaption hosted open-weight models (deepseek-v4-flash, qwen3.8-27b,
  qwen3.5-397b, gemma-4-31b-it, gemma-4-26b-a4b-it, glm-5.2) with a hard 50-credit cap. These give a cross-family
  replication of M2–M7: the same 9 calibration conditions, then frozen predictions, then populations with more seeds.
- **A5 (finite-size scaling, no LLM calls, 2026-11-07 ~16:00; `code/fss.py`, `results/fss_w.jsonl`):** fitted kernels
  (Haiku, Gemma-31B, GLM-5.2, Qwen3.8-27B; flexible and Bonabeau) were simulated at N ∈ {8…700} across the norm weight w
  (ρ = 0.5).
  - **The order-parameter curves are N-independent up to N = 700, with no growing susceptibility peak.** Under
    proportional (Bonabeau) forgetting, scores are capped at about Q/μ, so ±1 noise never vanishes. The mean-field
    pitchfork does not survive, and no tipping point emerges with scale.
  - Under **additive (BNR) forgetting**, toy kernels give a true transition:
    - L = r/r_c, with r_c = 2 = the maximum per-agent win rate, independent of N;
    - the leader's excess grows with N, consistent with √(t ln N);
    - a status-blind control collapses at r_c = 1.
  - Proposed follow-up: a BNR-protocol LLM test.
