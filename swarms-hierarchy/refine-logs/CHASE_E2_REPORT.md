# Chase study — E2 gate report (frozen population forecasts)

Written 2026-11-13, after E2 calibration and the forecast freeze, before any population call.
Forecast file: `results/chase_e2_predictions.json`, created 06:24:55, sha256 `9f2241d0cd4c…d86ef734`, write-once.
No population has been run. E3 does not start without the owner's go-ahead.

## Data

State-matched calibration: 9,400 calls (4 cells × 2,250, DeepSeek `comp` control 400), all cells complete.
137 replies unparseable (1.5%). They are concentrated in GLM-5.2 with the full log shown (4–6% of those calls):
the model starts an "## Analysis" essay instead of answering, mostly at long records where it has been losing.
Unparsed rows are left out of the kernel fits. In a population an unparsed reply becomes a coin flip, so GLM-5.2
LOG groups will carry about 5% noise decisions.

## What the forecasts say

"Linear" is the share of groups of four that end as a strict hierarchy (chance 0.375). t_tri is the transitivity
index at N = 8 (0 = chance, 1 = fully transitive). Brackets are 5th–95th percentiles of the predictive
distribution at the planned number of groups. OWN shows each agent only its own record; LOG adds the partner's
record against third parties.

| Cell | Groups of four: OWN | LOG | LOG − OWN | Power | N = 8 t_tri: OWN | LOG | LOG − OWN | Power |
|---|---|---|---|---|---|---|---|---|
| Gemma-31B `norm100` | 0.56 | 0.95 | +0.39 [0.25, 0.53] | 0.99 | 0.08 | 0.75 | +0.68 [0.44, 0.89] | 0.97 |
| Gemma-31B `comp` | 0.36 | 0.59 | +0.23 [0.07, 0.40] | 0.48 | −0.01 | 0.33 | +0.35 [0.05, 0.57] | 0.47 |
| GLM-5.2 `comp` | 0.56 | 0.51 | −0.05 [−0.25, 0.15] | 0.06 | 0.22 | 0.23 | +0.02 [−0.26, 0.29] | 0.07 |
| GLM-5.2 `norm100` | 0.40 | 0.42 | +0.02 [−0.18, 0.20] | 0.06 | 0.02 | 0.03 | 0.00 [−0.20, 0.23] | 0.03 |
| DeepSeek `comp` (control) | 0.37 | 0.37 | 0.00 [−0.25, 0.25] | 0.02 | −0.01 | −0.01 | 0.00 [−0.29, 0.28] | 0.02 |

No cell triggers the ceiling rule of amendment C2 (OWN above 0.85 linear).

**Gemma-31B.** Showing the partner's third-party record is forecast to turn near-random groups into hierarchies:
almost all groups of four linear under `norm100`, and a clear but smaller gain under `comp`. Withholding the
bystander features from the LOG kernel removes most of the gain (`norm100`: 0.74 instead of 0.95; `comp`: 0.42
instead of 0.59), so the forecast attributes the structure to the bystander response and not to the longer prompt.

**GLM-5.2.** The bystander slope that was significant in E1 is too weak to matter at the group level. Its
coefficient on the partner's record is about a fifth of Gemma-31B's (−0.3 against −1.4 on the same scale), and the
forecast is no difference between LOG and OWN in either framing. This is a prediction in its own right: a real
pairwise effect with no structural consequence.

**DeepSeek.** Chance in both arms, as expected of a model that always leads.

## Other forecasts in the file

- **Chase's sequence signature is forecast to be weak.** The share of triads formed by the two transitivity-
  guaranteeing sequences (DD + DS) stays between 0.50 and 0.59 in every arm, including Gemma-31B `norm100` LOG
  where nearly every group ends linear. The forecast route to hierarchy is late sorting, not Chase's early
  double-dominance and double-subordinance sequences. On the old score board the same models showed 0.67–0.71.
- **DYAD arm** (direct history only): chance in every cell (0.36–0.39). **REDACT arm**: between chance and the
  OWN arm, as it should be for a placebo that uses the same features as OWN.
- **Score board with the log added (N = 8).** Gemma-31B `norm100`: t_tri 0.99, against 0.83 for the score board
  alone. Gemma-31B `comp`: 0.10 against 0.15. GLM-5.2: 0.52 against 0.77 (`comp`), 0.77 against 0.80 (`norm100`).
- **Chain term.** Adding the transitive-chain feature (K4 over K3) changes no forecast by more than 0.05, in line
  with the nil transitive-inference contrast of E1.

## How much the forecast depends on the kernel form

The headline uses the pre-specified ladder (logistic on signed-log features). The file also holds a linear-logit
kernel (LIN) and a smoother with no functional form (FLEX).

| Cell | Arm, statistic | Ladder | LIN | FLEX |
|---|---|---|---|---|
| Gemma-31B `norm100` | LOG, linear (four) | 0.95 | 0.98 | 0.77 |
| | OWN, linear (four) | 0.56 | 0.63 | 0.50 |
| | LOG, t_tri (N = 8) | 0.75 | 0.88 | 0.57 |
| Gemma-31B `comp` | LOG, linear (four) | 0.59 | 0.63 | 0.46 |
| | OWN, linear (four) | 0.36 | 0.37 | 0.37 |
| | LOG, t_tri (N = 8) | 0.33 | 0.49 | 0.22 |

The direction is the same under all three. The size is not: FLEX, which shrinks toward the base rate away from
measured states, gives roughly half the gain. For Gemma-31B `comp` that is the difference between a detectable
effect and an undetectable one. E3 will show which form was right; that comparison is part of the test.

## Sizing (supplementary calculation, `results/chase_e2_power_supp.json`)

As announced in amendment C3, power was recomputed at other group counts from the frozen kernels alone
(`code/chase_e2_power.py`, no LLM calls, no population data).

| Cell, kernels | Groups of four per arm: 40 | 80 | 100 | N = 8 seeds per arm: 6 | 12 | 16 |
|---|---|---|---|---|---|---|
| Gemma-31B `norm100`, ladder | 0.98 | 0.99 | 1.00 | 0.94 | 1.00 | 1.00 |
| Gemma-31B `norm100`, FLEX | 0.59 | 0.90 | 0.94 | 0.85 | 0.99 | 1.00 |
| Gemma-31B `comp`, ladder | 0.49 | 0.78 | 0.85 | 0.40 | 0.81 | 0.90 |
| Gemma-31B `comp`, FLEX | 0.10 | 0.14 | 0.20 | 0.16 | 0.39 | 0.48 |

- Gemma-31B `norm100` is well powered as planned.
- Gemma-31B `comp` is not. Doubling OWN and LOG to 80 groups of four and 12 seeds at N = 8 brings power to about
  0.8 if the ladder kernel is right. If FLEX is right, no affordable size detects the effect.

## Proposal for E3 (owner's decision)

Measured calls per group: 98 for a group of four (24 rounds), 329 at N = 8 (40 rounds).

**Option A, focused (recommended).** About 65,500 calls, 23 hours at the rate limit, about 7 credits.
- Gemma-31B `norm100`: as planned (OWN 40, LOG 40, DYAD 20, REDACT 20 groups of four; 6 seeds each of OWN, LOG,
  SCORE+LOG, SCORE at N = 8).
- Gemma-31B `comp`: as planned, plus OWN and LOG raised to 80 groups of four and 12 seeds at N = 8.
- GLM-5.2 `comp`: OWN 40 and LOG 40 groups of four only, as a test of the no-difference forecast.
- GLM-5.2 `norm100`: not run. Its kernel slope is the smallest and its forecast is the same null.
- DeepSeek `comp` control: as planned.
- This departs from amendment C2, which gave every responsive cell the full set of arms.

**Option B, as written in amendment C2.** About 85,200 calls, 30 hours, about 26 credits (GLM-5.2 is the
expensive model). All four cells in full; Gemma-31B `comp` stays at power 0.4–0.5.

Either way spend stays inside the 60-credit cap (6.3 used so far). The order would be Gemma-31B `norm100` first,
since it is the sharpest test.

## What E3 can and cannot show

- If Gemma-31B LOG groups are far more linear than OWN groups, as forecast, the bystander channel is sufficient
  to produce hierarchy in an LLM population with no score board, which is Chase's claim transposed.
- If they are not, the pairwise kernel measured on single decisions fails to carry to groups, which would be a
  failure of the paradigm rather than of Chase.
- If GLM-5.2 shows a LOG − OWN gap well outside [−0.25, +0.15], the null forecast is wrong.
- The DD + DS forecast is the most Chase-specific test: he argues hierarchies are built through particular
  sequences. The forecast says these agents get to linearity without them.
