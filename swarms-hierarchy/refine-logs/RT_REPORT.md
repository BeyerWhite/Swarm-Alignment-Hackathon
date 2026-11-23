# Response-threshold study — results (RT1–RT4)

Written 2026-11-22 after the queue finished (21 Nov 22:05). Pre-registration `RT_PREREGISTRATION.md` (stamp 724e0787…),
forecast `results/rt_predictions.json` (frozen 21 Nov 09:34, before any RT3 population), evaluation
`results/rt_eval.json` (`code/rt_eval.py`, unchanged since the freeze). Models Gemma-4-31B and Qwen3.8-27B. Cost
≈ 1.8 credits (ledger 97.6 → 99.4). No change to code, cells, seeds or tests after the freeze.

## Verdict by the pre-registered criteria

| | Gemma-4-31B | Qwen3.8-27B |
|---|---|---|
| Comprehension / valid pilot turns | 1.00 / 0.97 | 0.995 / 0.98 |
| Work uptake (calibration, backlog 2+2, first mover) | 1.00, 1.00, 0.97 at c = 0.25, 0.5, 0.75 | pooled with Gemma: **saturated** |
| H1 log stimulus beats linear (ΔAIC > 4) | no (n ≈ 0) | no (n = −0.27) |
| H2 type reinforcement τ = κ + ω > 0 | **no**: τ = −0.14 [−0.16, −0.12] (balancing) | **yes**: τ = 0.055 [0.039, 0.071], Holm p = 0.006 |
| H3 specialization higher with own history (K = 8 vs 0) | no (p = 0.63) | no (p = 0.65) |
| H4 full forecast vs history-blind / position null (ΔLPD) | +5.5 / +8.0; coverage 67 of 78 | +121,233 / −8,505; coverage 41 of 78 |
| H5 step in specialization against demand | no | no |
| H6a new A specialists after removal | no (+0.25, p = 0.31) | no (−0.13, p = 0.81) |
| H7 hiding others' records changes specialization | no (p = 0.69) | no (p = 0.93) |
| H8 specialization rises with group size (RT4) | not run | no (ρ = −0.09, p = 0.68) |
| **Reading (section 6)** | No thresholds and no reinforcement: work by list position | Kernel without population effect, as the kernel predicted |

## What happened

**The work-or-rest half of the model could not be tested: both models always work.** At every calibration cost the
agents took an open task 97–100% of the time, though resting was free and a task netted only 0.25 after effort at
c = 0.75. The pre-registered saturation rule set c = 0.75 and made type choice (A or B) the primary half of the
kernel. In the grid, P(work) was 0.96–1.00 in every history profile. So these LLM agents have no response threshold
for *whether* to work in this framing: whenever work is visible they do it, which is the "always work" benchmark
policy. This is why n is near 0: with uptake at 1, n only measures how the backlog ratio moves the A/B choice, and it
barely does.

**Gemma-4-31B picks by list position and balances its own record.** It takes the type listed first 86% of the time
(b1 = 2.2) whatever the backlogs (P(A) 0.50 when A has fewer tasks, 0.61 when more). Its history effect is negative:
after 8 turns on B it chooses A 68% of the time. Since the listing order is random each turn, this is Tofts & Franks'
"foraging for work" with no memory; specialization stays at its permutation null in every arm (dol_excess −0.02 to
+0.02).

**Qwen3.8-27B has a weak, real type habit, too weak to make specialists.** After 8 turns on A it chooses A 73% of
the time, after 8 on B 53% (τ = 0.055 per turn, H2 holds), on top of a preference for the letter A (about 0.62).
That is a tenth of the reinforcement in the offline test (τ = 0.7), and the frozen forecast, which carries it,
predicted no specialization: observed dol_excess was −0.02 to +0.04, inside the forecast intervals for every
history arm. Raw DOL was 0.078 at both K = 0 and K = 8.

**Where the Qwen forecast failed: answers, not choices (post hoc).** Qwen got "what number comes after N" wrong in
1,250 of 4,232 population attempts (30%; e.g. 520 → 610, 680 → 680), and every one of its 1,276 wrong answers in
RT3 was on that task kind; its other kinds were correct. The forecast used one pooled accuracy (0.94 from the
kernel), so it under-predicted waiting tasks and over-predicted points in every arm; those metrics account for most
of the 37 misses and the huge LPD. The specialization metrics (dol_excess, persist) were mostly inside their
intervals. Gemma's misses are the invalid rate (4% in the kernel, 1% in populations) and the backlog at D = 1.

## Relation to earlier results

- **Rosen phase 1** showed Qwen with raw DOL 0.34 at m = 1 (no return to workbenches) and strong own-type choices in
  the R1x `hist` cells. In this cleaner design (two equal types, listing order random every turn, no workbenches)
  Qwen's DOL is 0.078 and its persistence is weak. The Rosen "habit" for Qwen was therefore probably carried by the
  unequal 4/2/2 demand and fixed listing, not by reinforcement. GLM-5.2, the strongest habit case there (88–100%
  own-type choices), was not tested here.
- **Hackathon paper:** agents there also chose largely by listing order when uncoordinated (most took the first
  task). The same holds for Gemma here.
- **Ulrich et al. (2021)** found response thresholds alone insufficient for ants. For these two LLMs the threshold
  model fails earlier: there is no uptake threshold to measure, and type choice is position-driven (Gemma) or weakly
  habitual (Qwen).

## Limits

- One framing of the effort cost; a framing in which resting is attractive (private effort, capacity limits) might
  reveal an uptake threshold. The saturation outcome was pre-registered, not a failure of the run.
- Two cheap models, reasoning off, 8 seeds per arm.
- The accuracy-by-task-kind diagnosis is post hoc.

## Possible next steps (need the owner's OK)

1. **GLM-5.2 on the frozen design** (owner's earlier plan if results are interesting): it is the one model with
   strong own-type persistence in the Rosen pilot, so it is the real test of reinforcement-driven specialization.
   About 22,000 calls, ≈ 16 credits, ≈ 8 hours.
2. **A framing that makes resting attractive**, to test the uptake threshold itself (new pre-registration).
3. Claude Haiku, through the CLI on the Pro plan (no Adaption credits, slower).

## Amendment RT-A1 — GLM-5.2 on the frozen design

Run 23 Nov 07:35–15:03 (ledger 112.63 → 122.61, ≈ 10.0 credits; estimate was 12–16). Forecast
`results/rt_predictions_glm.json` (sha256 05b3040b…, frozen 09:34, before any GLM population), evaluation
`results/rt_eval_glm.json`. Comprehension 1.00, valid pilot turns 1.00, invalid/stale 0 in the kernel.

| Test | GLM-5.2 |
|---|---|
| Work uptake (grid) | P(work) 1.00 in every history profile: saturated, as Gemma and Qwen |
| H1 log stimulus beats linear | no (n = 0.07 [0.01, 0.12]; ΔAIC −47) |
| H2 τ = κ + ω > 0 | **no**: τ = −0.11 [−0.126, −0.096] (balancing) |
| H3 specialization higher with own history | nominal p = 0.049, Holm p = 0.098: **no** |
| H4 full forecast vs history-blind / position null (ΔLPD) | −25.2 / −25.2; coverage 67 / 75 / 71 of 78 |
| H5 step against demand | no |
| H6a new specialists after removal | gain +0.75, nominal p = 0.031, Holm p = 0.094: **no** |
| H7 hiding others' records | no (p = 0.83) |

**What GLM does: strong alternation, not habit.** After 8 turns on A it picks A 17% of the time; after 8 on B, 64%
(grid, both types open). That is the opposite of the 88–100% own-type persistence in the Rosen pilot, so the Rosen
"habit" was a property of that environment (unequal demand, fixed listing, workbenches), not of the model. GLM
ignores list position (b1 ≈ 0, unlike Gemma's 86% first-listed) and leans against the type others are doing
(σ = −1.5 [−1.75, −1.36]).

**Populations.** dol_excess is +0.008 to +0.029 with own history (K = 8) and −0.004 to +0.003 without (K = 0), so
showing the record raises specialization slightly, against the balancing kernel's direction, but the effect does not
survive Holm and persistence stays at 0.50. The frozen full kernel forecast worse than both nulls (ΔLPD −25): the
history terms fitted in the kernel did not transfer to populations. Points match the always-work benchmark except at
D = 1 (waiting 0.24 at K = 0, 0.69 at K = 8).

**Reading.** None of the three models has a response threshold in the effort frame, and none reinforces its own type:
Gemma follows list position, Qwen has a weak habit, GLM balances. The reinforcement half of Theraulaz et al. (1998)
is not what these agents do.
