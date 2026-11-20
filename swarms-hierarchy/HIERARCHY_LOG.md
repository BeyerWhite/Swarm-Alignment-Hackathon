# Lab log — hierarchy-efficiency study (Rosen 1982; Garicano 2000)

Pre-registration: `refine-logs/HIERARCHY_PREREGISTRATION.md`; proposal: `refine-logs/HIERARCHY_EFFICIENCY_PROPOSAL.md`.
Amendments are numbered H-A1, H-A2, …

## 2026-11-20

- Owner approved phase 1 as proposed, GLM-5.2 and Qwen3.5-397B included.
- Built `hier_tasks.py`, `office.py`, `hier_measure.py`, `hier_sim.py`, `hier_eval.py`, queues. Offline checks, no
  API calls: every generator's answer passes its checker (11 kinds × 2 levels × 200); `office.py mock` runs all
  four arms; `hier_sim.py selftest`; `hier_dryrun.py` (fake backend) runs the whole pipeline G0 → evaluation and
  recovers the effects planted in the fake agents (assertive model wins S3, the sentence raises plan share).
- The dry run showed that 8 G1c seeds rank managers unreliably when gains are close (bootstrap P(best stays best)
  0.47 with the fake agents); the registered top-up rule was added before the stamp.
- Pre-registration and code stamped in `results/hier_prereg_freeze.json` before the first API call.
- 14:23 `hier_q0.sh 150 1644` launched; stopped at 14:30 before any call. Two queues from other studies wait for the
  same Rosen pid (tournament queue 1, pid 28072; the response-threshold wrapper, pid 28523), and the stamped
  other-queue check did not recognise their script names, so G0 would have started together with tournament queue 1.
- **Amendment H-A1** (before any API call): `hier_common.sh` `other_queue` now refuses to start while any
  `*queue[N].sh` script runs. Stamp rewritten (`hier_prereg_freeze.json`; the first one kept as
  `hier_prereg_freeze_v0_superseded.json`). G0 relaunched under a wrapper that waits until no other study's queue
  script is running, so this study runs after the Rosen, tournament and response-threshold queues already launched.
