# E11–E13 — the three no-call follow-ups of amendment C10

Run 20 Nov 2026 on existing groups only, no model calls. The scripts were checked against the hashes in C10 before
the runs: `chase_c10.py` f92cca00…, `chase_e11_eval.py` 14a5a848…, `chase_e12.py` 1689d2ec…, `chase_e13.py`
986c45c4…. `chase_fit.py`, which they import, was never hashed in the log; it is unchanged since the study's first
commit (454d89a). Every rebuilt feature matched the stored values (0 mismatches). Outputs: `results/chase_e11_eval.json`,
`results/chase_e12.json`, `results/chase_e13.json`. E14, the fresh-cell test of the LEAD-rate rule, needs calls
and has not been run.

## Summary

- **E11 (how the order is built).** In Qwen3.5-397B the whole gain from partner records arises after relationships
  form, as predicted (P1 supported). In Gemma-4-31B the same is true, contrary to the prediction (P2 not supported).
  In both models partner records raise the share of groups whose first relationships are already linear, yet that
  starting order hardly predicts the final one. In Qwen, reversals under partner records go more often toward the
  agent with the better record (S1, as predicted).
- **E12 (the Qwen `comp` forecast miss).** Reading "neither". Even fitted on the 120 groups' own decisions, the
  forecast's decision model gives 0.67 [0.62, 0.74] linear against the observed 0.81. So the miss is not a matter of
  which records the calibration sampled. The best richer form (partner recency) fits decisions better but does not
  close the gap either. In Gemma-4-31B `comp`, where the forecast was right, the same model covers (0.59 against 0.58).
- **E13 (the Gemma-26B equal-turns gap).** Reading "neither". On all 180 groups the gap remains: 0.33 [0.27, 0.38]
  simulated against 0.41 observed. Agents do differ from each other beyond their records (σ 0.39, LR p = 6 × 10⁻²⁶),
  but simulating those differences does not raise linearity. The round number adds nothing.
- **Together.** In two cells, a model of independent single decisions, fitted to the very decisions the groups
  made, produces less hierarchy than the groups did. Something at the level of the group, not in any tested feature
  of a single decision, adds order.

## E11 — formation or later

Linear share, groups of four, `comp`. a: share of groups whose initial relationships form a linear order; s and r:
share finally linear among groups that started linear and among the rest.

| | Arm | a | s | r | Linear |
|---|---|---|---|---|---|
| Qwen3.5-397B | OWN (120) | 0.53 | 0.50 | 0.54 | 0.52 |
| | LOG (120) | 0.65 | 0.78 | 0.86 | 0.81 |
| Gemma-4-31B | OWN (120) | 0.33 | 0.31 | 0.42 | 0.38 |
| | LOG (120) | 0.65 | 0.59 | 0.57 | 0.58 |

| LOG − OWN, 95% group-bootstrap interval | Qwen3.5-397B | Gemma-4-31B |
|---|---|---|
| Total | +0.29 [+0.18, +0.40] | +0.20 [+0.08, +0.33] |
| Formation | −0.01 [−0.03, +0.01] | −0.02 [−0.06, +0.03] |
| Later | +0.30 [+0.19, +0.41] | +0.22 [+0.09, +0.34] |
| of which keep | +0.17 [+0.08, +0.26] | +0.14 [+0.05, +0.23] |
| of which repair | +0.13 [+0.06, +0.21] | +0.08 [−0.02, +0.17] |
| Later − formation | +0.31 [+0.19, +0.42] → **"later"** | +0.23 [+0.08, +0.38] → **"later"** |
| | P1 supported | P2 not supported |

- **The formation term is zero because the start does not carry over, not because starts are alike.** Partner
  records raise the share of groups that start linear (Qwen 0.53 → 0.65, Gemma 0.33 → 0.65). But within an arm,
  groups that started linear end linear no more often than those that did not (s ≈ r). The final order, measured
  over the last half of the rounds, is made in those later rounds.
- **S1, as predicted for Qwen.** Under partner records, 0.59 of reversals put the agent with the better net record
  against third parties on top, against 0.51 with own records only: +0.08 [+0.02, +0.13]. REDACT: 0.51. Gemma-31B,
  not predicted: 0.47 → 0.76 (+0.29 [+0.13, +0.45]), but on few reversals (86 and 49).
- **Descriptive.** Reversals per formed dyad: Qwen 0.79 (OWN), 0.80 (LOG), 0.57 (REDACT); Gemma 0.21 and 0.11.
  Initial transitivity of triads formed entirely by the E9 rule: Qwen 0.85 → 0.87; Gemma 0.74 → 0.95.
- **What it says about Chase's sequences.** In Gemma-31B partner records do produce Chase's start: more DD and DS
  (E9) and initial triads that are nearly all transitive. But that start is not what decides the final hierarchy.
  In Qwen there is no such start, and the gain comes the same way: groups with partner records keep and repair
  their order better over the later rounds, with reversals that follow visible records.

## E12 — why the forecast missed Qwen `comp` with partner records

| Leave-one-seed-out log-loss | B0 (forecast form) | B1 + partner recency | B2 + phinge | B3 + cmp | B4 all |
|---|---|---|---|---|---|
| Qwen3.5-397B `comp` LOG (11,380 decisions) | 0.6080 | 0.5944 | 0.6081 | 0.6081 | 0.5941 |
| Gemma-4-31B `comp` LOG (11,442; no test) | 0.2356 | 0.2315 | 0.2355 | 0.2357 | 0.2317 |

B1 is selected in Qwen (0.014 better than B0). In Gemma-31B it is 0.004 better, below the 0.005 threshold, so B0
is kept there.

| Group-bootstrap adequacy (90%) | Simulated | Observed | Covered |
|---|---|---|---|
| Qwen LOG, B0 | 0.67 [0.62, 0.74] | 0.81 | no |
| Qwen LOG, B1 | 0.66 [0.59, 0.73] | 0.81 | no |
| Qwen OWN, X | 0.56 [0.48, 0.63] | 0.52 | yes |
| Gemma-31B LOG, B0 | 0.59 [0.50, 0.66] | 0.58 | yes |
| Gemma-31B OWN, X | 0.40 [0.32, 0.47] | 0.38 | yes |

Reading: **"neither"**.
- **Coverage.** The calibration covered the states the groups visited: no Qwen decision fell in a cell with fewer
  than five calibration rows (Gemma-31B: 15%).
- **The fitted model is too flat in the partner's record.** Where the partner's third-party net is +3 or better,
  Qwen agents led 0.29 of the time. The model fitted on the calibration rows predicts 0.38 there, and the one
  fitted on the groups' own decisions 0.36. In the middle and lower bins the miss is smaller and of the other sign.
  This was not part of the registered reading.

## E13 — the Gemma-26B equal-turns own-record arm

180 groups (E5 40, E8a 140), 17,272 decisions, observed linear share 0.41.

| | Result |
|---|---|
| Leave-one-seed-out log-loss | C0 0.5760, C1 (+ round) 0.5758 |
| C2 random intercept per agent | σ 0.39, LR 109.5, p = 6 × 10⁻²⁶ (720 agents) |
| Adequacy C0 | 0.33 [0.27, 0.38], not covered |
| Adequacy C1 | 0.34 [0.28, 0.39], not covered |
| Adequacy C2 | 0.33 [0.27, 0.38], not covered |
| Spread of per-agent LEAD rates | 0.109 observed against [0.097, 0.106] under C0 |

Reading: **"neither"**.
- **Agent differences.** They are real here and in both comparison cells: Gemma-31B `norm100` σ 0.73 (LR 40) and
  Qwen `comp` OWN σ 0.33 (LR 35). But at this size they do not change the simulated linear share.
- **The earlier misses.** At 180 groups the forecast gap is smaller than E5's 0.50 suggested, but it is not gone.

## Post hoc check (after the results above; not registered)

Candidate: the two answers in a contest might be complementary beyond the model, with one agent deferring more often
when the other leads. That would raise decided contests and could build order. It is not complementary. With each
model fitted on all its cell's decisions, the observed share of decided contests was:

| Cell | Observed | Model | 95% range under the model |
|---|---|---|---|
| Qwen `comp` LOG | 0.526 | 0.543 | [0.530, 0.555] |
| Qwen `comp` OWN | 0.491 | 0.489 | [0.477, 0.502] |
| Gemma-31B `comp` LOG | 0.263 | 0.261 | [0.252, 0.270] |
| Gemma-26B `norm0` OWN | 0.388 | 0.417 | [0.407, 0.427] |

Where the models miss, there are slightly fewer decided contests than they imply, not more. A candidate not tested
here is persistence in specific pairs: one agent giving way to one partner beyond what the visible record of that
pair (d, last ten rounds) shows.

## Limits

- All three analyse groups already analysed. E11's predictions were formed from the E9 and E10 sequence results.
- The adequacy intervals come from group-bootstrap refits of the same decisions they are compared with. So a miss
  is a strong result, and a cover a weak one.
- "Neither" in E12 and E13 says what the gap is not. It does not say what it is.
