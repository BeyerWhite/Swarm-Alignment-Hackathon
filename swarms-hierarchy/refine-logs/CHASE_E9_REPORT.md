# E9 — the registered sequence test (amendment C8)

Fresh Gemma-4-31B groups of four, seeds 14000–14039 (DYAD 14000–14019), two framings (`comp`, `norm100`), arms
OWN 40, LOG 40, DYAD 20: 200 groups, 19,862 calls, 0.77 credits, run 16 Nov 23:05 – 17 Nov 05:59, exit 0, no failed
calls, 10 invalid answers in 19,200 decisions. Evaluation `code/chase_e9_eval.py` (sha256 848276f0…4799db79, as
registered) → `results/chase_e9_eval.json`. Formation rule fixed in C8: a relationship forms at the pair's first
contest in which one agent leads and the other defers. Shares are over triads with two formed relationships.

## Summary

All four primary predictions hold. Seeing other pairs' records changes the order in which relationships form, as
Chase's account says it should: under `comp` it raises the share of the two orders that guarantee a transitive triad
(DD + DS) from 0.49 to 0.72; under `norm100` it moves triads from DS to DD (DD 0.18 → 0.59, DS 0.68 → 0.36). The
post hoc pattern from E3 replicates on fresh groups. P5 (no-information baseline near one half) holds under `comp`
and misses narrowly under `norm100`.

## 1. Sequences by arm

| Framing | Arm | Groups | Triads classified | DD | DS | BDID | ISDB | DD + DS [95%] | Linear |
|---|---|---|---|---|---|---|---|---|---|
| `comp` | OWN | 40 | 130 / 160 | 0.24 | 0.25 | 0.25 | 0.26 | 0.49 [0.41, 0.57] | 0.42 |
| | LOG | 40 | 120 / 160 | 0.50 | 0.22 | 0.12 | 0.17 | 0.72 [0.63, 0.80] | 0.53 |
| | DYAD | 20 | 66 / 80 | 0.24 | 0.30 | 0.17 | 0.29 | 0.55 [0.45, 0.65] | 0.30 |
| `norm100` | OWN | 40 | 133 / 160 | 0.18 | 0.68 | 0.02 | 0.12 | 0.86 [0.79, 0.92] | 0.70 |
| | LOG | 40 | 157 / 160 | 0.59 | 0.36 | 0.01 | 0.04 | 0.94 [0.90, 0.98] | 0.95 |
| | DYAD | 20 | 77 / 80 | 0.17 | 0.22 | 0.35 | 0.26 | 0.39 [0.29, 0.48] | 0.35 |

Chase's hens: DD + DS 0.91 in triads, 0.86–0.87 in tetrads; a random order gives 0.50.

## 2. Primary predictions (per-group shares, two-sided Mann–Whitney, Holm over four)

| | Prediction | LOG − OWN | p | Holm p | Result |
|---|---|---|---|---|---|
| P1 | `comp` DD + DS, LOG > OWN | +0.22 | 0.0026 | 0.0032 | supported |
| P2 | `comp` DD, LOG > OWN | +0.24 | 0.0016 | 0.0032 | supported |
| P3 | `norm100` DD, LOG > OWN | +0.41 | 5 × 10⁻⁸ | 2 × 10⁻⁷ | supported |
| P4 | `norm100` DS, OWN > LOG | −0.32 | 7 × 10⁻⁶ | 2 × 10⁻⁵ | supported |

Pooled differences with group-bootstrap intervals: `comp` DD + DS +0.22 [+0.11, +0.34], DD +0.26 [+0.13, +0.39];
`norm100` DD +0.41 [+0.29, +0.51], DS −0.32 [−0.42, −0.21].

P5 (criterion): DYAD DD + DS in [0.40, 0.60] — `comp` 0.55, met; `norm100` 0.39 [0.29, 0.48], missed by 0.01. With
no records shown, the order of formation is close to random, as intended; under `norm100` BDID is somewhat
common (0.35).

Estimate only: `norm100` DD + DS, LOG − OWN = +0.09 [+0.01, +0.16] (registered power 0.33; OWN is already at 0.86).

## 3. Secondary

- **S1, does the order matter for the end state?** Triads formed by DD or DS end transitive more often than those
  formed by BDID or ISDB: `comp` 0.84 vs 0.77, +0.07 [−0.00, +0.16]; `norm100` 0.95 vs 0.81, +0.14 [+0.05, +0.21].
  The order still matters, but less than in Chase's hens, where orders that do not guarantee transitivity are
  rare and the final relations were almost all transitive. Here later contests repair many BDID and ISDB triads.
- **S2, DD + DS as a share of all triads** (unformed triads counted against): `comp` 0.40 → 0.54; `norm100`
  0.71 → 0.93. LOG also forms more relationships within the 24 rounds under `norm100` (157 of 160 triads against
  133).
- **S3, linear share (fresh replication of H3)**: `norm100` 0.70 → 0.95, +0.25 [+0.10, +0.40], Fisher p = 0.006;
  `comp` 0.42 → 0.53, +0.10 [−0.15, +0.35], p = 0.50. E3 had +0.17 (`norm100`) and +0.25 (`comp`, Holm 0.008), so
  the effect replicates in `norm100` and is smaller and not significant in `comp` this time. Pooled over E3 and E9
  (descriptive, 80 groups per arm): `norm100` 0.74 → 0.95, `comp` 0.43 → 0.53.
- **S4, against the frozen E2 intervals** (section 2 of `CHASE_E3_REPORT.md`):

| Framing | Arm | E9 observed | Frozen forecast | Inside | E3 observed |
|---|---|---|---|---|---|
| `norm100` | OWN | 0.70 | 0.56 [0.42, 0.70] | edge | 0.78 |
| | LOG | 0.95 | 0.95 [0.90, 1.00] | yes | 0.95 |
| | DYAD | 0.35 | 0.37 [0.20, 0.55] | yes | 0.55 |
| `comp` | OWN | 0.42 | 0.36 [0.25, 0.45] | yes | 0.43 |
| | LOG | 0.53 | 0.59 [0.45, 0.75] | yes | 0.53 |
| | DYAD | 0.30 | 0.37 [0.20, 0.55] | yes | 0.45 |

  All six fresh arms fall inside or on the edge of the forecasts made before E3. The `norm100` OWN miss in E3
  (0.78) looks partly like a high draw: the fresh 0.70 sits at the upper edge. The E6 part B forecast for the same arm,
  frozen at 07:05 on 17 Nov without the E9 groups, is 0.73 [0.60, 0.85]; this comparison was not registered.

## 4. What this says

Chase's claim is that hierarchies become linear because of the order in which relationships form, and that
bystanders who watch fights produce the orders that guarantee transitivity. The pre-registered test of this (H2,
relationship formed at the first contest, coin flips included) was not supported; E9 is the registered test with a
formation rule closer to Chase's, and it holds: with the record of other pairs visible, the next relationship is more often
one that shares the winner with the first (DD), as when an onlooker defers to an animal it has seen win. In `norm100`
this comes at the expense of DS, which already guaranteed transitivity, so the gain in DD + DS there is small; the
linear share still rises, because LOG also settles relationships sooner (S2) and leaves fewer triads unrepaired.
In `comp` the shift is from BDID and ISDB to DD, which is the change Chase's account needs. The second link, that
these orders are what make the final hierarchy linear, holds more weakly than in hens (S1).

Limits: one model; the formation rule ("decided") was chosen after the E3 data were seen and was fixed before E9;
groups of four only; P5 is a criterion, not a test.
