# Chase study — E1 gate report (bystander kernel)

Written 2026-11-13 02:42, after E1 batches A and B and before any E2 call. Analysis: the frozen
`code/chase_e1_analyze.py` (hash verified against `results/chase_e1_freeze.json`), run on
`results/chase_e1_kernel_AB.jsonl`, which is `chase_e1_kernel.jsonl` restricted to the three batch A and B models
(65 stray rows of batch C models from the first unpaced launch are excluded and will be analysed with batch C).

**Data.** 3 models × 3 framings × 2,112 calls = 19,008 rows, all cells complete; 12 replies unparseable (0.06%).
Holm adjustment is over these 9 cells. The final E1 table will adjust over every cell run (batch C, Claude, GPT).

## Result in one paragraph

Agents that can see their partner's record against third parties LEAD less against a partner who has been winning.
This bystander effect is present in all four planned cells (Gemma-31B and GLM-5.2 under `comp` and `norm100`) and
absent under the equal-turns norm and in DeepSeek, which leads regardless of anything shown. There is no sign of
transitive inference through a shared third party, and lines with the outcomes removed do nothing. The effect is
one-sided: with a baseline LEAD rate near 1, agents can only become more deferential, and they do so toward visible
winners.

## Primary estimands

β_p is the change in P(LEAD) per unit of the partner's net third-party record (linear probability, HC1 standard
error), controlling for the agent's own record. Chase's prediction is β_p < 0. "p = +3 vs −3" is the raw
difference in LEAD rate between a partner seen winning three more than losing and the reverse.

| Model | Framing | β_p ± SE | Holm p | p = +3 vs −3 | T ± SE | Holm p | Responsive |
|---|---|---|---|---|---|---|---|
| Gemma-31B | comp | −0.089 ± 0.008 | < 0.001 | −0.59 | 0.000 ± 0.000 | 1.00 | yes |
| Gemma-31B | norm100 | −0.146 ± 0.007 | < 0.001 | −0.85 | −0.042 ± 0.025 | 0.85 | yes |
| Gemma-31B | norm0 | −0.006 ± 0.006 | 0.99 | −0.03 | +0.010 ± 0.010 | 1.00 | no |
| GLM-5.2 | comp | −0.051 ± 0.009 | < 0.001 | −0.31 | −0.031 ± 0.062 | 1.00 | yes |
| GLM-5.2 | norm100 | −0.021 ± 0.006 | 0.007 | −0.15 | +0.021 ± 0.038 | 1.00 | yes |
| GLM-5.2 | norm0 | −0.016 ± 0.009 | 0.33 | −0.12 | 0.000 ± 0.025 | 1.00 | no |
| DeepSeek | comp | 0.000 | 1.00 | 0.00 | −0.010 ± 0.010 | 1.00 | no |
| DeepSeek | norm100 | 0.000 | 1.00 | 0.00 | 0.000 | 1.00 | no |
| DeepSeek | norm0 | −0.003 ± 0.002 | 0.99 | −0.02 | 0.000 | 1.00 | no |

## Shape of the bystander effect

LEAD rate in the LOG grid by the partner's net third-party record p, pooled over the agent's own record:

| Model | Framing | p = −3 | −1 | 0 | +1 | +3 |
|---|---|---|---|---|---|---|
| Gemma-31B | comp | 1.00 | 1.00 | 1.00 | 1.00 | 0.41 |
| Gemma-31B | norm100 | 1.00 | 1.00 | 1.00 | 0.65 | 0.15 |
| Gemma-31B | norm0 | 0.67 | 0.67 | 0.67 | 0.65 | 0.64 |
| GLM-5.2 | comp | 0.85 | 0.85 | 0.86 | 0.77 | 0.54 |
| GLM-5.2 | norm100 | 0.98 | 0.92 | 0.99 | 0.94 | 0.83 |
| GLM-5.2 | norm0 | 0.32 | 0.06 | 0.16 | 0.09 | 0.20 |
| DeepSeek | all | 1.00 | 0.99–1.00 | 1.00 | 1.00 | 0.98–1.00 |

- In Gemma-31B the effect is a threshold, not a slope: nothing happens until the partner's record is clearly
  positive, then LEAD collapses. The linear β_p understates this.
- In GLM-5.2 it is weaker and more gradual, and under `norm100` it is small (a 15-point drop at p = +3).
- There is no "bystander loser" effect (more LEAD against visible losers) because the baseline is already at or
  near 1. Whether one exists cannot be read from these cells.

## Secondary estimands

- **Transitive inference (T).** Zero within error in every cell. The shared-versus-distinct third-party contrasts
  S_cwin and S_closs are also zero. Caveat: the relational cells show one or two events, and in Gemma-31B a single
  event does not move LEAD off 1.00 in the grid either, so in that model T is tested at ceiling and is
  uninformative rather than negative. In GLM-5.2 `comp` the cells are off ceiling (0.84–0.98) and T is still nil.
- **Redaction placebo (R).** Lines with the outcome removed change nothing (R between −0.01 and +0.03, all within
  error). The bystander effect comes from who won, not from the amount of text.
- **Own record (winner and loser effects), OWN arm.** Gemma-31B ignores its own record under `comp` and
  `norm100` (LEAD 0.88–1.00 at every level). GLM-5.2 under `comp` shows a loser effect (LEAD 0.50 at own net −2,
  1.00 at 0, β_o = +0.075 ± 0.025). Under `norm0` Gemma-31B takes turns: LEAD 1.00 with a losing or even record,
  0.00 with a winning one.
- **Direct history, DYAD arm.** GLM-5.2 `comp` follows it (LEAD 0.31 after two losses to this partner, 1.00 after
  two wins). Gemma-31B `norm0` takes turns on it (1.00 → 0.00). Elsewhere effects are small.
- **Conflict cells.** When direct history and third-party information disagree (beat this partner, but the partner
  beat someone who beat you; and the reverse), Gemma-31B and GLM-5.2 stay near their baseline under `comp` and
  `norm100` (0.78–1.00).

## What this means against the hypotheses

- **H1 (bystander effect exists in the pairwise kernel): supported** in four of four planned cells.
- **Transitive inference: not supported**, with the ceiling caveat above for Gemma-31B.
- **Framing dependence: as predicted.** The effect needs a framing in which a record is a reason to defer; under
  the equal-turns norm agents use their own record for turn-taking and ignore the partner's.
- **Control: as predicted.** DeepSeek is status-blind here as it was on the score board.
- Nothing here says anything yet about group structure. That is E2 (forecast) and E3 (populations).

## Decisions taken from this report (pre-specified rules)

- Cells admitted to E2 and E3: Gemma-31B × `comp`, `norm100`; GLM-5.2 × `comp`, `norm100` (amendment C2, rule 2).
  DeepSeek × `comp` is the control. No replacement is needed, so batch C stays last in the queue.
- E2 state-matched calibration started 02:41 for these cells (9,400 calls, about 3.3 hours).

## Addendum, 2026-11-13 13:15 — Claude Haiku cells

Added after the gate, as planned in amendments C2 and C4: `claude:haiku` (`claude-haiku-5-5`) × `comp`, `norm100`,
same cells and seed, 4,224 calls through the Claude Code CLI with no tools; every reply parsed. Frozen analysis run
on `results/chase_e1_kernel_ABH.jsonl` (batches A and B plus Haiku); Holm is now over 11 cells. The nine earlier
rows keep their verdicts (for example GLM-5.2 `norm100`, β_p Holm p 0.007 → 0.008). The gate outputs are
kept as `results/chase_e1_cells_AB.csv` and `results/chase_e1_summary_AB.json`. Haiku does not enter the E3
selection rule.

| Model | Framing | β_p ± SE | Holm p (11 cells) | p = +3 vs −3 | T ± SE | Responsive |
|---|---|---|---|---|---|---|
| Haiku | comp | −0.011 ± 0.004 | 0.028 | −0.07 | −0.021 ± 0.021 | yes |
| Haiku | norm100 | −0.052 ± 0.007 | < 0.001 | −0.31 | 0.000 ± 0.038 | yes |

LEAD rate in the LOG grid by the partner's net third-party record:

| Framing | p = −3 | −1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|
| comp | 1.00 | 1.00 | 0.99 | 0.99 | 0.94 | 0.93 |
| norm100 | 1.00 | 0.98 | 0.96 | 0.95 | 0.78 | 0.69 |

- The bystander effect is there in both framings, with the same sign and the same one-sided shape as in Gemma-31B
  and GLM-5.2, but smaller: under `norm100` about a third of Gemma-31B's drop, under `comp` a few points.
- No transitive inference (T nil) and no effect of redacted lines (R within ±0.01), as in the other models.
- Under `norm100` Haiku also responds to its own record and to direct history: LEAD 0.75 at own net −2 against
  0.99 at +2 (β_o = +0.050 ± 0.016), and 0.62 after two losses to this partner against 1.00 after two wins
  (β_d = +0.094 ± 0.019). Under `comp` these are small.
- Caveat from amendment C4: with a subscription sign-in the CLI adds one fixed sentence to the system prompt and
  a fixed reminder block. Both are constant across cells, so they cannot produce the contrasts above, but the
  absolute LEAD levels are those of this wrapped call, not of a bare API call.

## Addendum, 2026-11-13 15:30 — GPT cells

`gpt:gpt-6-luna` × `comp`, `norm100`, same cells and seed, 4,224 calls through `codex exec`; every reply parsed on
the first try. Frozen analysis run on `results/chase_e1_kernel_ABHG.jsonl`; Holm is now over 13 cells and no
earlier verdict changes. GPT does not enter the E3 selection rule. Caveat from amendment C4: each call carries
about 14,600 tokens of Codex agent instructions around the study prompt, so this is a coding agent answering the
question, not a bare model.

| Model | Framing | β_p ± SE | Holm p (13 cells) | p = +3 vs −3 | T ± SE | Holm p | R ± SE | Responsive |
|---|---|---|---|---|---|---|---|---|
| GPT-6-Luna | comp | −0.079 ± 0.010 | < 0.001 | −0.47 | +0.208 ± 0.095 | 0.37 | −0.146 ± 0.068 | yes |
| GPT-6-Luna | norm100 | −0.092 ± 0.010 | < 0.001 | −0.56 | −0.052 ± 0.069 | 1.00 | −0.312 ± 0.061 | yes |

LEAD rate in the LOG grid by the partner's net third-party record:

| Framing | p = −3 | −1 | 0 | +1 | +3 |
|---|---|---|---|---|---|
| comp | 0.76 | 0.73 | 0.62 | 0.55 | 0.29 |
| norm100 | 0.80 | 0.89 | 0.86 | 0.73 | 0.24 |

- **The bystander effect is strong, and GPT is the only model off the ceiling.** Its baseline LEAD rate is 0.41
  under `comp`, so both directions can be seen there: LEAD rises against a partner seen losing (0.62 → 0.76) and
  falls against one seen winning (0.62 → 0.29). Under `norm100` only the fall appears.
- **Transitive inference: a hint under `comp`, not established.** With a chain through a shared third party LEAD
  is 0.79 when the chain favours the agent and 0.57 when it does not; with distinct third parties 0.65 and 0.64.
  T = +0.21 ± 0.10, which does not survive the Holm adjustment (0.37). Nil under `norm100`.
- **The redaction placebo is not null for GPT.** Three partner lines with the outcomes removed lower LEAD by 15
  points (`comp`) and 31 points (`norm100`) relative to the full log at p = 0. Part of GPT's reaction to the log
  is therefore a reaction to seeing that the partner has a history at all. β_p is not touched by this, because it
  compares records of equal length that differ only in who won.
- **Direct history dominates.** LEAD is 0.09 (`comp`) and 0.25 (`norm100`) after two losses to this partner, and
  0.84 and 0.97 after two wins (β_d = +0.20 and +0.17).

## Things to keep in mind when reading E2

- Gemma-31B's OWN-arm kernel is close to "always LEAD", so its own-record populations should be driven mostly by
  coin flips. The ceiling concern raised in amendment C3 (own-record arm already near 85% linear) is therefore less
  likely for Gemma-31B and more likely for GLM-5.2 `comp`, which does respond to its own and direct record.
- E1 only reaches |p| = 3. Populations reach much larger records. The calibration step measures those states
  directly; until then any forecast beyond the grid is extrapolation.
