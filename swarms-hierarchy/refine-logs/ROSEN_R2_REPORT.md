# Rosen phase 1 — results (R1 kernel, R2 populations)

Written 2026-11-18 after R2 finished. Pre-registration: `ROSEN_PREREGISTRATION.md`. Forecast stamp 58c603b1…
(all 480 R2 populations carry it). Evaluation: `results/rosen_r2_eval.json` (`code/rosen_eval.py` after amendment
R-A2, below). Cost of R1 + R2: 14.3 credits (ledger 44.2 → 58.5 of 100).

## Verdict by the pre-registered criteria (§6)

| | Gemma-4-31B | GLM-5.2 | Qwen3.8-27B |
|---|---|---|---|
| H1 kernel tracks value (ΔAIC > 4, signs) | yes (740) | yes (261) | yes (65) |
| H2 corner (≥ 80% hold one workbench, median use ≥ 0.9) | yes (100%, 1.00) | yes (99%, 1.00) | **no** (76%, 0.82) |
| H3 step in DOL vs F (ΔAIC > 4) | yes (15.5; break at F = 4) | **no** (0.6) | **no** (−0.8) |
| H4a DOL m = 2 > m = 1 at F = 3 (Holm) | yes (+0.55, p = 0.004) | yes (+0.30, p = 0.036) | **no** (+0.03, p = 0.60) |
| H4b lone agent builds more workbenches at F = 1 (Holm) | **no** (0 of 8 vs 0 of 64) | **no** (0/8 vs 2/64) | no (8/8 vs 39/64, raw p = 0.026, Holm 0.105) |
| H5 team vs hidden at F = 3: coverage / duplication (Holm) | no / **yes** (dup 0.1 vs 2.9, p = 0.004) | no / no | no / no |
| Myopia (index CI below 1) | no: 2.36 [2.05, 2.72] | no: 3.64 [2.54, 5.75] | no: 3.19 [1.84, 6.65] |
| Forecast inside 90% interval | 25 of 91 | 18 of 91 | 57 of 91 |
| **Reading** | **Rosen supported**, threshold too low | Rosen effect plus role habit | No economic response |

## What happened

**Gemma-4-31B behaves like Rosen's agents, with a lower threshold.** At F = 1 every agent builds exactly one
workbench, the population covers all three types and DOL is 0.73 (220 of a possible 240 points). Every one of the
129 agents that built anything held exactly one workbench and used it for all its later work. DOL falls in a step
(0.73, 0.65 at F = 1, 3; 0.30, 0.15 at F = 5, 7; about 0.1 above), but the break is at F ≈ 4, not at the rational
F\* = 8: at F = 5 and 7 only a third and an eighth of agents invest although investing pays. At m = 1 nobody builds
and DOL is at the never-build level (0.09), so the specialization is caused by the increasing return, not habit.
Building happens almost only in round 1, mostly by the agents who move first; first movers take type A (the
largest demand; mean turn position 2.7, against 4.4–4.8 for B and C).

**GLM-5.2: the Rosen effect sits on top of role-taking.** The workbench effect is there (H2, H4a), but DOL stays at
0.45–0.52 at F = 7–13, where hardly anyone builds, and is 0.33 at m = 1 (never-build benchmark 0.10). GLM agents
keep to a task type without any return for doing so, as already seen in the pilot. This flattens the DOL curve, so
H3 fails. By §6 this is "kernel without (a clean) population step": part of GLM's division of labour is economic,
part is habit.

**Qwen3.8-27B does not respond to the economics.** It starts workbenches at every F (84–100% of agents) and at
m = 1, where a workbench pays nothing (94% at F = 3); the m = 1 prompt states that plainly. Agents hold 1.9
workbenches at F = 1 and score far less than not building at all at high F (83 against 128 at F = 13). DOL is flat
(0.23–0.43) and does not depend on m.

**Horizon: not myopic, the opposite.** In single decisions Gemma-31B's threshold at the start of the game is close
to rational (m = 2, 16 rounds left: builds up to F = 7, stops by F = 11; F\* = 8), but with 11 rounds left it builds
only at F = 1 (F\* = 5.5) and with 6 left almost never (F\* = 3). GLM shows the same shape more softly. The
pre-registered index is therefore above 1 for every model: willingness to invest falls off with the horizon two to
three times faster than value does. The populations inherit this: investment is a round-1 decision, and an agent
that did not build then almost never builds later.

**Exchange vs autarky (H4b) went the wrong way.** A lone Gemma-31B agent never builds at m = 2 for F ≥ 3 (rational:
one workbench up to F = 7) and builds in only 3 of 8 runs at F = 1; at m = 3 it builds at F = 1 but never at
F = 3. GLM alone is similar. The autarky prompt was checked and states the rules as in the team arm. In these
models investment is something a member of a team does, not something a lone worker does, which is the reverse of
the "lone agent invests broadly" prediction. Only Qwen builds more alone, and it overbuilds everywhere.

**Visibility makes investment into division of labour.** When agents cannot see others' workbenches (hidden arm,
F = 3), 88% of Gemma agents build, nearly all type A: duplication 2.9 against 0.1 in the team arm, DOL 0.20 against
0.65, and fewer points (155 against 164). With visibility, later movers see A taken and either take B or C or do
not build. Coverage was not significantly different.

**Reasoning (descriptive, 12 cells).** With reasoning effort "low", choices move toward the rational benchmark:
mean distance from it falls from 0.19 to 0.10 for Gemma-31B and from 0.32 to 0.07 for GLM, at the cost of 10%
invalid replies (44 of the 60 are empty replies; the rest are GLM explanations written before the action line). The
reasoning calls had a 2,000-token limit; an earlier version of this report wrongly said the replies were cut off
at 60 tokens. The most striking case: at F = 11, m = 3, 16 rounds left (just above
F\* = 10.7), Gemma builds 88% of the time without reasoning and 0% with it.

## The forecast failed for two of three models

The pre-registered forecast, which simulates populations from the single-decision kernel, put 25 (Gemma), 18 (GLM)
and 57 (Qwen) of 91 observed means inside its 90% intervals; the offline check with a simulated agent had 81–83.
Three reasons, each visible in the per-arm table (`forecast_coverage` in the eval file):
1. **Autarky was extrapolated.** R1 measured only team states; the forecast assumed the same kernel for a lone
   agent and predicted building that never happened (Gemma: forecast build rate 1.0 at F = 3, observed 0).
2. **Type choice is not memoryless.** The forecast picks task types afresh each turn, so it cannot produce GLM's
   persistent roles (forecast DOL 0.10 at F = 13, observed 0.52).
3. **Population crowding is stronger than the kernel cells showed.** Gemma's forecast builds at 0.64 and 0.50 at
   F = 5 and 7; observed 0.33 and 0.12. The crowd cells varied only how many others were building, not how many
   had already finished, or the turn position within round 1.

So the kernel predicts the direction of the population result (who specializes and when), but it does not predict
the levels: population behaviour depends on context (team vs alone, history of one's own choices, order of moves)
that the single-decision states did not vary.

## Changes after data

- **R-A2 (evaluation bug, found after R2).** The H5 duplication test passed the two arms in the wrong order and
  tested "team duplicates more than hidden". The pre-registration states the direction (lower in team), so the
  call was corrected to match it and the evaluation re-run. The first output is kept as
  `results/rosen_r2_eval_v1_signbug.json`. This changes one result: Gemma-31B's H5-dup goes from p = 1.0 to
  p = 0.004 (Holm). Every other test and number is unchanged.
- No other change to data, code or tests after R2 started.

## Limits

- 8 populations per cell; the H3 step uses the 56 team populations per model. H4b compares 8 lone agents with
  64 team agents.
- One task stream design (4/2/2), one prompt wording, three models, temperature 1, reasoning off for all
  populations.
- The reasoning arm is 12 single-decision cells, not populations.

## Possible next steps (need the owner's OK; phase 2 not started)

- Measure the kernel in the contexts the forecast missed (autarky states, own task history, finished workbenches of
  others, turn position), then re-forecast. That would test whether a richer single-decision kernel can predict
  populations.
- Populations with reasoning on for Gemma-31B and GLM, to see whether the threshold moves to F\* = 8.
- Phase 2 of the proposal (Becker–Murphy coordination costs, Yang–Borland learning by doing).
