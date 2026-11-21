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

## 2026-11-19

**Approval.** The owner approved all follow-ups in `ROSEN_R2_REPORT.md` and phase 2 ("Go ahead with all of the
follow-ups (including Phase 2)").

**Correction to the report.** The report said the reasoning arm's invalid replies were cut off at 60 tokens. The
reasoning calls had a 2,000-token limit; 44 of the 60 invalid replies are empty and the rest are GLM explanations
written before the action line. The report is corrected.

**Design** (`ROSEN_PHASE2_PREREGISTRATION.md`, frozen in `results/rosen_p2_prereg_freeze.json` at 02:00:59, stamp
289cd99c…). New code: `rosen_forecast2.py` (v2 kernel), `rosen_p2.py` (R3 and R4 engines, run commands, R6 battery),
`rosen_eval2.py`, `rosen_p2_queue.sh`; `rosen.py` gained group size and the notebook variant (defaults checked to
reproduce phase 1's engine output exactly); `rosen_kernel.py` gained the R1x cells. Not run: GLM-5.2 reasoning
populations (about 50 credits) and the optional PLAN arm.

**Offline validation** (scripted agent with type persistence 0.45 and a lone-agent penalty; no API calls).
- v2 recovers the built-in quirks (persistence estimated at 0.52 and 0.42).
- On fresh seeds v2 covers 36 and 42 of 47 metrics against v1's 32 and 35 (pooled McNemar p = 0.006); v2 retro
  covers 87 and 85 of 91.
- R2b finds a +2.0 shift (CI 1.7–2.4) for an agent whose rational shift is 2.7.
- R3 and R4 detect the built-in fee, market-size, λ and horizon effects; R6 measures the built-in gain
  (m̂ = 2.95, populations 0.97 vs 0.51 accuracy); R5 finds the same width at N = 4, 8 and 16.
- Fixes found by the test: benchmark policies for R3 that took loss-making steps, a fit that failed when DOL
  exceeded 1 by rounding, a step test that needed at least three F values, and R5's F grid widened to 11.

**Queue.** `code/rosen_p2_queue.sh` waits for the Chase E7 queue (pid 25354), then runs the registered order with a
ledger guard. Estimated 85,000 calls, about 30 hours and 22 credits; ledger 58.8 at freeze, Chase E7 still spending.

**2026-11-19 14:36 — R2x forecast frozen (written by the queue script).** `results/rosen_r2x_predictions.json`, stamp badcbeefa1b8b83b0fcf729d6d4b5c0dde31721d64e0ce1bf5df5cf91efc2063, file sha256 f408a816684349c318cff2082258e7a8616aadb0820d79dbdb22511029d6a267.

**2026-11-19 15:55 — R6 battery result (observation, no change yet).** Accuracy on the x × k_j tasks is the same with and
without the table: Gemma-4-31B 56/60 and 56/60, Qwen3.8-27B and GLM-5.2 60/60 in both. All four Gemma misses in each
condition are empty or degenerate replies, not arithmetic errors. So m̂ = 1 and F\* = 0: the notebook as registered is
worthless capital, and the R6 populations would test investment in a useless asset, not the threshold. The offline
test had assumed 50% accuracy without the table. Decision on R6 (amend before its populations run, or run it as a
placebo) is with the owner; the queue reaches R6 after R2x, R2r and R2b.

### Amendment R-A3 — R6 notebook tasks become codes (2026-11-19 16:30, before any R6 population or codebook call)

Owner's choice ("Fix it: codebook") after the multiplication battery showed no gain (entry above). Changes:
1. Notebook tasks are codes. Each type has a fixed, secret code word, north or south, for every item x = 12 … 39
   (half of each, shuffled per seed and type: `rosen.code_map`). A task asks "What is the code of item x: north or
   south?". A ready notebook lists the type's 28 codes in the agent's prompt; the system prompt says that without
   a ready notebook an agent can only guess. Everything else in the notebook variant (F turns, 1 point per correct
   task, private and permanent, arrivals, seeds, plan team m = 1, F ∈ {1, 3, 5, 7, 9}) is unchanged.
2. Battery: as registered (60 with, 60 without the table, per model, three models), to
   `results/rosen_r6_acc_code.jsonl`, run before the R6 populations. Expected p_none ≈ 0.5, so m̂ ≈ 2 and
   F\* ≈ 8, the same rational threshold as the stated m = 2 in R2.
3. Tests unchanged (H2 corner at F ≤ 7, H3 step vs line, step location against F\*). Added, descriptive: the R6
   step location against R2's (stated m = 2, step at F ≈ 4), i.e. measured against stated gain at equal F\*.
4. The multiplication battery is kept and reported as `battery_mul`. `NB_TASKS = "mul"` reproduces the
   registered variant exactly (checked: identical populations and prompts).
5. Checks: populations of every other stage are identical to the frozen code (162 scripted populations over arms,
   F, m, N and seeds, plus prompts and metrics). Offline codebook run with the scripted agent: battery 1.0 vs 0.47;
   in populations tooled accuracy 1.00, untooled 0.50; the evaluation reads both batteries.
6. R6 runs (battery and populations) refuse to start without `results/rosen_p2_RA3_freeze.json`.

**2026-11-19 16:30 — correction: the effective credit cap is 78.06, not 100.** Checked at the owner's request.
`llm.py` defaults to 100 (Sara's project cap) but reads `ADAPTION_CREDIT_CAP`, and `.secrets/adaption.env` sets
it to 78.06 (Chase amendment C8). The phase 2 queue sources that file, so every stage runs under 78.06; its own
guard and the pre-registration assumed 99/100, which was wrong. The ledger is 70.06 with R2x 35/240 done
(R2x costs about 5.8 credits, mostly GLM-5.2), so R2x should finish near 75 and R2r will hit the cap part way.
`llm.py` then raises (fails closed: unfinished populations are not written) and the queue stops. The Chase E10
queue (owner-approved, cap 99) is waiting on this queue's PID and starts then. Finishing phase 2 afterwards needs a
cap set by the owner; the decision is with the owner.

**2026-11-19 16:13 — cap raised to 150 by the owner; queue handed over.** The owner asked for all caps to be 150.
Changed: `.secrets/adaption.env` (ADAPTION_CREDIT_CAP 78.06 → 150) and the default in `code/llm.py` (100 → 150,
both the main checkout and this worktree; uncommitted). The `llm.py` change is to one constant and its comment, so
its sha256 now differs from the one stamped in the earlier freezes. Running processes keep the cap they started with.
Not changed: the Chase E10 queue (frozen script, cap 99 by its own rule; it needs about 7.1).
Handover: the phase 2 queue (bash 27487) was paused with SIGSTOP before it could start R2r under the old 78.06 cap;
R2x (python 44653) continues, and when it exits the paused queue is ended, which lets the waiting E10 queue start.
`code/rosen_p2_queue2.sh` (sha256 ee396a15a65c7c333e6cad1a823242604ce4e2d87be7d938eadb5966d2cb468f) waits for E10 (pid 47129), exports cap 150 for
its own run, guards each stage at ledger + estimate < cap − 1, and runs the rest in the registered order with the
R-A3 codebook battery before R6: R2x remainder (if any), R2r, R2b, R6 battery, R6, R5, R4, R3.

**2026-11-19 16:21 — reboot; queues restored.** The Mac rebooted at 16:17 and every queue died: the paused phase 2
queue, R2x (39/240 populations written; unfinished populations are not written), the waiting
`rosen_p2_queue2.sh` and the waiting Chase E10 queue. Ledger 70.23. The Chase session relaunched E10 at 16:19
(plan A, cap 99, now running). `code/rosen_p2_queue2.sh` (R2x estimate changed to 5.0 for the larger remainder;
sha256 822845b0…) relaunched detached at 16:20 with cap 150, waiting for the E10 queue (pid 1554). It resumes R2x
(restart-safe) and runs the rest as registered with R-A3.

**2026-11-21 — phase 2 done and evaluated.** `rosen_p2_queue2.sh` finished 05:09 (ledger 97.59), all stages complete:
R2x 240, R2r 88, R2b 56, R6 battery 360 and populations 40 (all stamped R-A3 a7aaf12d…), R5 96, R4 96, R3 128.
`rosen_eval2.py --B 2000` → `results/rosen_p2_eval.json`. Spot checks: R6 records all `nb_tasks = code`; N = 16
prompts state 8/4/4 arrivals; R2r invalids are 595 empty replies of 614. Report: `ROSEN_PHASE2_REPORT.md`.
