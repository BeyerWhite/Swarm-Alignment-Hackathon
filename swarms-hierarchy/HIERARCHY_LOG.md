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
