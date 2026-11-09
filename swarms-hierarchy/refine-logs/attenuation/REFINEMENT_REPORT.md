# Refinement Report — attenuation critique (2026-11-08)
**Rounds:** 2 (shortened from 5 for the deadline). **Final score:** 7.70 / 10 (gpt-6.1-sol, reasoning effort ultra). **Verdict:** REVISE.
**Cross-check:** gpt-6-astra, reasoning effort high, fresh thread 01a1be8b-90af-7221-b8f8-85f1fc1e8047. It agrees with REVISE and verified every number; 7 issues were fixed.
**Problem anchor:** see `../round-0-initial-proposal.md` (unchanged across rounds).

**Final thesis:** shrinking sparse, saturated two-agent calibration cells toward indifference is a major, not the only, cause of the one-sided micro→macro miscalibration.
- **Post-hoc evidence:**
  - matched decomposition: −58% / −86% mean signed residual;
  - known-kernel simulation with dose–response;
  - lapse sweep;
  - other regimes: N = 700 approximate agreement, threshold 13/16.
- **Prospective pilot D:** the unshrunk monotone procedure out-scored Jeffreys in both conditions. It passed the frozen rule for Qwen3.5-397B and failed it for Gemma-4-26B.

| Round | Overall | Verdict | Main concerns | Changes |
|---|---|---|---|---|
| 1 | 6.45 | REVISE | mechanism over-claimed; uncertainty; dose test; pilot spec | matched decomposition, known-kernel simulation, lapse sweep, pilot design |
| 2 | 7.70 | REVISE | pilot freeze integrity; claim strength | snapshot hashing, exclusive creation, frozen manifest, seed checks; claims narrowed |

**Files:**
- `round-0-initial-proposal.md`, `round-1-review.md`, `round-1-refinement.md`, `round-2-review.md`, `score-history.md`
- `../RESPONSE_TO_ATTENUATION_CRITIQUE.md`

**Remaining weaknesses:**
- interval coverage below nominal;
- the Gemma-4-26B prospective failure;
- neutral-wording seeded leader;
- X1 rank-1 prediction;
- anchor creation not atomic.
