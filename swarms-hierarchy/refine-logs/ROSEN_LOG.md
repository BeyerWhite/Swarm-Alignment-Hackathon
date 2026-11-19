# Rosen study — lab log and amendments

Chronological. The pre-registration (`ROSEN_PREREGISTRATION.md`) and the study code are hash-stamped in
`results/rosen_prereg_freeze.json` and are not edited; changes are recorded here.

## 2026-11-17

**Approval.** The owner approved phase 1 of `ROSEN_PROPOSAL.md` (R0, R1, R2, with GLM-5.2).

**Setup.** Branch `rosen-specialization`, cut from `chase-bystander` at 83ea57a, in the worktree
`.worktrees/rosen` (excluded locally via `.git/info/exclude`). The main checkout stays on `chase-bystander`, where the
Chase E8 queue is running. `llm.py` gained two backward-compatible options: `ADAPTION_LEDGER_DIR`, which points this
worktree at the main checkout's spend ledger so the 100-credit cap counts both studies, and a `reasoning` argument
whose default `none` leaves earlier calls unchanged.

**Offline validation (no API calls).** A simulated, noisy-rational agent was run through every stage
(comprehension, probe, kernel, top-up, freeze, pilot, R2, eval) in a scratch results folder.
- Rational benchmark: step between F = 7 and F = 9 (DOL 1.0 to 0.05–0.08), as F\* = 8 predicts.
- First version of the kernel model: 64 and 61 of 91 forecast metrics inside the 90% intervals for agents whose
  own kernel was measured.
- Three causes found and fixed, all before any real data:
  1. Crowding was measured only on type A, and its strength was capped at 2.
  2. Continuation was modelled on k/F, with rounds left confounded with k.
  3. A single continuation model was shared by m = 1 and m = 2.
- This led to new crowd-on-B cells and late-start and m = 1 continuation cells; continuation is now modelled on
  turns built and rounds left, per m; the ridge is 0.01.
- Final: 81 and 83 of 91 inside. The myopia index for the near-rational agent is 1.06 [0.94, 1.17] (m = 2) and
  0.96 [0.87, 1.04] (m = 3). H3 finds the break at 8.0.

**Design changes from the proposal**, recorded in the pre-registration:
- Sequential turns instead of simultaneous waves, so collision avoidance is not a second reason to specialize.
- Autarky tests the breadth of investment (several workbenches when they are cheap, as its brute-force optimum
  does), not "no specialization". With interchangeable points a lone agent also concentrates its work after
  building.
- The extra kernel cells above. R1 is about 2,500 calls per model, up from about 2,000.

**Queue.** `code/rosen_r0_queue.sh` waits for the Chase queue (pid 75880) to exit, then runs R0 and the gate summary.
R1 and R2 are launched after R0 has been inspected.

## 2026-11-18

**R0 done (00:21–01:29).** The Chase queue exited at 00:20; R0 cost 1.0 credit (ledger 43.22 → 44.22; Chase had
spent 5.0 since the proposal). Gate (`results/rosen_r0_gate.json`), all four pass:

| Model | Comprehension | Pilot valid | Probe valid (reasoning low) |
|---|---|---|---|
| Gemma-4-31B | 0.96 | 0.998 | 0.92 |
| Qwen3.8-27B | 0.73 | 0.965 | 1.00 |
| GLM-5.2 | 0.87 | 0.998 | 0.92 |
| Gemma-4-26B | 0.67 | 1.000 | 0.60 |

R2 therefore uses Gemma-4-31B, Qwen3.8-27B and GLM-5.2; the reasoning arm runs on Gemma-4-31B and GLM-5.2.

- Comprehension misses are all on the four point-projection items (1, 3, 4, 12); the rule items are 0.9–1.0 for
  every model. Gemma-4-26B answers 11 to item 1 (16 − 5, without the multiplier) every time and says building for
  12 of 16 turns pays (item 12) every time. Qwen spreads widely on the arithmetic. This is a valuation weakness,
  not a misunderstanding of the rules, and is what R1/H1 measures.
- Pilot (team, m = 2, F ∈ {1, 13}, 2 seeds; not used for forecasts or tests): at F = 1 Gemma-31B and GLM build one
  workbench each and cover the 4/2/2 quota (3 of 4 populations exactly); Qwen builds about 2 workbenches per agent
  (dup 8–10) and at F = 13 every Qwen agent starts building, scoring about 72 vs 128 for never building; Gemma-26B
  almost never builds (1 build turn in 512).
- GLM at F = 13 shows DOL 0.38–0.58 with no workbench: several agents do one type 12–13 times in 13 rounds. This
  is role-taking without increasing returns; the m = 1 placebo (H4a) is the test that separates it from Rosen.

**Amendment R-A1 (before any R1 call).** The reply parser rejected `DO T8 (C): 80`, a well-formed move with a type
tag. That accounted for 7 of Qwen's 18 invalid pilot turns and the single invalid turn of Gemma-31B and of GLM.
`rosen.parse` now accepts an optional `(X)` or `(type X)` tag after the task id and ignores it (the id fixes the
type); a tag outside A–C is still invalid. The other invalid replies (reasoning preambles cut off at 60 tokens) stay
invalid. R1, the forecast and R2 all use the new parser, so they stay consistent; the R0 gate numbers above were
computed with the old one. New `code/rosen.py` sha256 7d3c582e1b2f31fc6490c3a16bccbc71e8d55dc3cb8aaf9558e2c55cd174d723.

**2026-11-18 05:55 — R2 forecast frozen (written by the queue script).** `results/rosen_r2_predictions.json`, stamp 58c603b1b9769d521fae889ee3fda41f35bf431e3a8d7bb9385c9e801803557b, file sha256 fdcc53cba4c3b7b6e202c4e70e361d546aca3db796b922c8bb93c5baf1506b20. Models: adaption:gemma-4-31b-it adaption:qwen3.8-27b adaption:glm-5.2.

**R1 and R2 done (01:30 → 19:19).** R1: 9,888 kernel calls, 1,568 top-up calls, 576 reasoning-arm calls. Forecast
frozen at 05:55 before any R2 population; all 480 R2 populations (3 models × 20 settings × 8 seeds) carry its stamp.
R1 + R2 cost 14.3 credits (ledger 44.22 → 58.55).

**Amendment R-A2 (evaluation bug, found after R2).** `rosen_eval.py` passed the arms of the H5 duplication test in
the wrong order, so it tested dup(team) > dup(hidden), the opposite of the pre-registered direction. Fixed to match
the pre-registration and re-run; the first output is kept as `results/rosen_r2_eval_v1_signbug.json`. Only
Gemma-31B's H5-dup changes (p 1.0 → 0.004 after Holm). New `code/rosen_eval.py` sha256
06a869c70d2ab5aec534f291521e1379c9ce88fe43680abf016b7720b95a5081.

**Results** (full write-up: `ROSEN_R2_REPORT.md`). Gemma-31B meets the pre-registered "Rosen supported" criterion
(H2, H3, H4a), with the DOL step at F ≈ 4 rather than F* = 8. GLM shows the workbench effect (H2, H4a) plus role-taking
without returns, so H3 fails. Qwen does not respond to F or m. H4b fails for all (lone agents invest less, not more).
Myopia index above 1 for every model. Forecast coverage 25, 18 and 57 of 91.
