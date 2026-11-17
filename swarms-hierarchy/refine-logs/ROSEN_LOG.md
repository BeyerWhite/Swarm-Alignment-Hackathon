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
