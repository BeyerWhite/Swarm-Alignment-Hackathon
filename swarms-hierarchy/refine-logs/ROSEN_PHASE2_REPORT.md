# Rosen study — follow-ups and phase 2: report

Written 2026-11-21 from `results/rosen_p2_eval.json` (`code/rosen_eval2.py --B 2000`), with spot checks of the
population files. Design: `ROSEN_PHASE2_PREREGISTRATION.md` (freeze 289cd99c…) plus amendment R-A3 (codebook
notebook tasks, freeze a7aaf12d…). Every stage ran in full, 19–21 November; about 22.8 credits for the phase
(ledger 68.2 → 97.6, less Chase E10's 6.6, which ran in between). The cap was 78.06 at the start, not the 100 the
pre-registration assumed; the owner raised it to 150 on 2026-11-19 (log). All tests one-sided, α = 0.05, Holm
within the family named.

## Summary

| Stage | Question | Result |
|---|---|---|
| R2x, F1a | Does the richer kernel (v2) forecast fresh populations better than v1? | **Not shown.** v2 right where v1 wrong 27 times, the reverse 16 (McNemar p = 0.063). Better for Gemma (16 vs 9 of 47) and GLM (15 vs 9), not Qwen (26 vs 28). Both still miss most means. |
| R2x, F1b | Do phase 1's results replicate on fresh seeds? | **Yes, model by model.** Gemma: one workbench each (H2), step at F = 4 (ΔAIC 16.7), H4a p < 0.001. GLM: H2 and H4a (p = 0.001), step weak (ΔAIC 2.4). Qwen: none, as before. |
| R2r | Reasoning on (Gemma): more investment, more points? | **More building, not more points.** Build rate at F = 5, 7 +0.43 (Holm p = 0.001); points +2.8 per population (p = 0.28). The step moves from F ≈ 4 to **F ≈ 8, the rational threshold**. No building at m = 1; the lone agent builds at F = 1 and 3. 9% invalid (mostly empty replies). |
| R2b | Higher payoff (m = 3) moves the threshold up? | **Yes, by more than rational.** F50 4.2 → 9.9, shift +5.7 (95% CI 4.8 to 6.6; rational +2.7). The m = 3 step (F ≈ 10) sits near its rational 10.7, so the m = 2 under-investment largely goes away. |
| R6 (R-A3) | Same corner and threshold when the gain is real (codebook)? | **Corner yes, no under-investment.** Battery: Gemma 0.98 with vs 0.50 without the notebook (m̂ = 1.97, F\* = 7.9). In populations every agent makes one notebook at every F ∈ {1, …, 9} (H2: 95% hold exactly one); they then answer 3,241/3,241 correctly and almost never guess (39 untooled answers in 40 populations, 44% right). Building is rational for F ≤ 7 and costs points only at F = 9 (52 vs about 64 without notebooks). H3's AIC "step" is a rise from F = 1 (some double notebooks) to F ≥ 3, not a threshold. |
| R5 | Is the step equally sharp at N = 4, 8, 16? | **Not resolved; the location moves with N.** log(w16/w4) = −1.30, 90% CI −2.48 to 1.17: contains 0 but the point is beyond log 2, so the registered criterion fails. The step location rises with team size: F50 ≈ 2.2 (N = 4), 4.3 (N = 8), 7.5 (N = 16), CIs disjoint between 4 and 16, although the rational threshold does not depend on N (arrivals are per capita). The Rosen steps are much sharper than the hierarchy transition (w ≈ 0.07–0.27 vs 2.8–3.9). |
| R4 | Yang–Borland: learning by doing → specialization; horizon effect? | **Specialization yes, horizon no.** R4-a holds for both (Holm p = 0.001): at λ > 0 Gemma's DOL is 0.96–1.00 from round 2 (rational), GLM's 0.62–0.82. R4-b fails: Gemma at ceiling (no difference), GLM *less* specialized with the 24-round horizon (−0.36, p = 0.99). λ = 0 control: no horizon difference. |
| R3 | Becker–Murphy: handoff fees cut specialization; bigger market raises it? | **Mostly no.** DOL stays near 0.1 in every cell (benchmark specialist 0.42 / 0.93 at c = 0). R3-a: GLM yes (Holm p = 0.019, DOL 0.15 → 0.11), Gemma no. R3-a′ (handoffs fall with c): Gemma yes (Holm p = 0.0015, all from J = 4: 0.87 → 0.40), GLM no. R3-b (more DOL at J = 4): no for both. At J = 2 both models hand off almost every step whatever the fee; Gemma at c = 2 ends with negative points (−11.8). |

## Reading

1. **The phase 1 pattern is robust.** On fresh seeds Gemma shows Rosen's corner solution and a sharp threshold, GLM
   the corner with role habit, Qwen no economic response. The prospective forecast improvement was in the
   predicted direction but not significant; single-decision kernels still miss population levels.
2. **The investment threshold responds to economics, but is biased low with stated gains.** With a stated m = 2
   Gemma stops at half the rational threshold (F ≈ 4 vs 8). Three things remove the shortfall: reasoning on (step
   at 8), a higher stated gain (m = 3: step near rational 10.7), and a measured gain (R6: builds through F = 7,
   over-invests at 9). One reading is that agents discount a stated multiplier but not a felt one; the R6
   instructions also say that without a notebook one "can only guess", which may make the notebook look
   necessary rather than profitable. R6 cannot separate the two, and its grid ends at F = 9, one step above F\*.
3. **Team size moves the threshold, which Rosen's per-agent problem does not predict.** The N = 16 prompt states
   8/4/4 arrivals instead of 4/2/2 (same per agent). Larger absolute numbers appear to read as a larger market —
   Smith's "extent of the market" intuition applied where it does not hold. This is a post-hoc interpretation of
   an unregistered pattern and should be tested directly (same N, scaled numbers in the prompt).
4. **Learning by doing produces specialization at once; horizon framing does not.** The registered horizon effect
   (R4-b) is absent for Gemma (already at the ceiling) and reversed for GLM.
5. **Coordination costs are largely ignored.** In the Becker–Murphy chains neither model specializes, and at
   J = 2 both pay the handoff fee on nearly every step, even at a loss. Only Gemma in the busier market (J = 4)
   cuts handoffs when there is a fee.

## Caveats

- Gemma only for R2r, R2b, R5 and R6 (GLM reasoning populations left out for cost, as registered).
- R6 changed from multiplication to codes before any R6 population (R-A3); the multiplication battery showed no
  gain (`battery_mul`, m̂ = 1.0 for all three models).
- R5's width test is underpowered (CIs on w span an order of magnitude at N = 4 and 8).
- Effects named as post-hoc above (R6 framing, N and market size) are not tests.
- The R2x run was interrupted by a reboot and resumed; unfinished populations were not written, so every kept
  population is complete.
