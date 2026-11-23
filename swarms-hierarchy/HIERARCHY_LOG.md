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
- 14:23 `hier_q0.sh 150 1644` launched; stopped at 14:24 before any call. Two queues from other studies wait for the
  same Rosen pid (tournament queue 1, pid 28072; the response-threshold wrapper, pid 28523), and the stamped
  other-queue check did not recognise their script names, so G0 would have started together with tournament queue 1.
- **Amendment H-A1** (before any API call): `hier_common.sh` `other_queue` now refuses to start while any
  `*queue[N].sh` script runs. Stamp rewritten (`hier_prereg_freeze.json`; the first one kept as
  `hier_prereg_freeze_v0_superseded.json`). G0 relaunched under a wrapper that waits until no other study's queue
  script is running, so this study runs after the Rosen, tournament and response-threshold queues already launched.
- 14:25 G0 wrapper launched (pid 29071, log `scratch/hier_q0.log`).

## 2026-11-23

- **G0 ran** 05:29–06:54 (the wrapper waited until no other study's queue ran). Cost 0.57 credits (ledger 112.01 →
  112.58). All six models passed the gates (battery parse 0.96–1.00, comprehension 0.98–1.00, pilot valid replies
  0.975–1.00). The frozen rule chose remainder L1, digitsum L2, reverse L2 and shift L2 (spreads 0.55–0.70, mean
  accuracies 0.26–0.61), with different best models (Gemma-26B; Qwen3.5-397B; GLM-5.2; Qwen3.5-397B), so matching
  tasks to agents matters. Pilot (not used for hypotheses): throughput 0.7–2.5 of 8 per round; follow rates
  0.94–0.98; Gemma-26B planned most (0.47–0.57 of its turns), DeepSeek almost never.
- **Amendment H-A2** (after G0, before any G1 call). Inspecting the pilot showed that the numeric checker scored the
  first integer of a reply, so answers that show their working ("638 % 7 = 1", "1+7+…+3 = 66") were marked wrong
  even when right. Only DeepSeek writes such replies: its pilot accuracy is 0.28 under the old checker and 0.49 under
  the new one; the other models are unchanged. Fix in `hier_tasks.extract`: a leading number counts only if no
  operator follows it, otherwise the number after the first "=", otherwise the first number. `code/hier_rescore.py`
  rebuilt every battery prompt, matched all 2,640 rows to their logged full responses (0 missing) and re-scored
  them: 19 rows change (DeepSeek multiply L2, digitsum L1/L2, add3 L2; one Gemma-31B subtract L2). The frozen
  selection rule and the gates were re-applied unchanged: the **same four kinds** are chosen, and all models pass.
  Old files kept: `hier_g0_battery_v0_firstint.jsonl`, `hier_g0_selection_v0_superseded.json`,
  `hier_g0_gate_v0_superseded.json`. Offline dry run passes. Stamp rewritten (previous kept as
  `hier_prereg_freeze_v1_superseded.json`).
- 18:50 G1–G4 queued, last in line as the owner chose: a wrapper (log `scratch/hier_q1.log`, cap 200, the owner's cap
  since 21 Nov) waits for the RT phase-2 launcher (72577), Rosen phase 3 (60296) and Couzin queue 2 (75855), then
  for no `*queue[N].sh` script and no paced job (two checks three minutes apart), then runs `hier_q1.sh 200`.
  The wrapper's command line does not contain the script name, because `rosen_p3_wait.sh` counts any
  `hier_q[N].sh` process as busy (a first wrapper that did was stopped after 30 s, before it could block Rosen).
