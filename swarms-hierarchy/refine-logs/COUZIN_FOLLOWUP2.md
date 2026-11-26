# Couzin study, amendment M-A4 — follow-ups after the final report

Frozen before any call of these follow-ups (stamp `results/couzin_followup_freeze.json`; the previous stamps are kept
as `couzin_followup_freeze_v1.json` and `_v2.json`). The registered results in `COUZIN_FINAL_REPORT.md` are
unchanged.

## 0. Owner decision

On 2026-11-26 the owner asked to "do the follow-ups" listed in the final report:
1. the M4b design on GPT and on Qwen;
2. a test for H5 that survives separation.

Choices I made and recorded here:
- **GPT model:** gpt-6-luna, the model used earlier in this repository. It runs through `codex exec` with the
  owner's ChatGPT login, not an API key, so it uses plan limits rather than per-token billing.
- **Qwen model:** Qwen3.5-397B, the model discussed for M4b on 2026-11-20 (decision 3). The cheap Qwen3.8-27B is not
  run.

## 1. GPT (gpt-6-luna), CLI

- **Procedure:** the same as M-A1 (`COUZIN_CLAUDE_FOLLOWUP.md`): C0 gate, C1 kernel (70 cells × 20 draws), ω rule,
  M2c calibration (8 groups), frozen forecast, the 5 M4b arms × 24 groups, evaluation F1–F3 with the M-A3
  complete-arms rule.
- **Size:** about 19,600 calls.
- **Runner:** `code/couzin_cli_run.sh` with `COUZIN_CLI_MODELS="gpt:gpt-6-luna"`; results in `results/couzin_gpt/`.
- **Limits:** `GPT_CALL_BUDGET = 25000`, raised from llm.py's default of 1,500, which this follow-up needs.
  `LLM_MAX_CONC = 6`.
- **Stop rule, extended.** `codex` may write no output when a usage limit hits, and llm.py would then return an
  empty reply, not a quota message. So for CLI models an empty reply after llm.py's retries also stops the run
  (exit 75), and nothing affected is written.
  - This was tested offline: an empty GPT reply stopped the run and refused later calls; empty Adaption replies are
    unaffected.
  - Auto-resume as for Claude: 30-minute waits, at most 60 attempts.

## 2. Qwen3.5-397B, Adaption

- **Reuse.** The main study's frozen M0 gate (label pair ORION / LYRA) and its M1 kernel rows for this model
  (1,400 rows, 20 draws per cell) are reused through `couzin_followup.py seed`. This is the same measurement the
  main study used.
- **Then:** ω rule, M2c calibration (8 groups), frozen forecast, the 5 M4b arms × 24 groups, evaluation F1–F3.
- **Size and cost:** about 18,200 calls; about 2.3 credits at the measured median of 1.3e-4 per call.
- **Runner:** `code/couzin_queue3.sh`, launched through `couzin_launch.sh 3 200 <pid>`, so it waits for its turn on
  the account.
- **Guards:** the same as queues 1–2: ledger + estimate < cap − 1 (cap 200), and study spend + estimate ≤ 20 (the
  study has spent 1.98).
- **Results:** `results/couzin_qwen/`.

## 3. Separation-proof trend test (post-hoc)

- **The problem.** The registered H5 / F2 use a Wald z from logistic regression, which is useless under separation:
  Gemma's minority won 0 of 24 groups in 6 of 7 M4 arms.
- **The script.** `code/couzin_exact.py` reports, for every M4-type dataset:
  - a permutation test of the trend in minority wins over n_u (20,000 permutations within a protocol), one-sided
    each way;
  - a Firth-penalised logistic slope with its 90% interval.
- **Status.** It was added after the data were seen. It is a sensitivity analysis and does not replace the
  registered verdicts.
- **First run (2026-11-26), on the existing data:**

| Dataset | Simultaneous: minority wins as n_u rises | Observe-first |
|---|---|---|
| Gemma-31B (M4) | 0, 0, 0, 0.17: rises, p = 0.003; Firth +0.21 [+0.05, +0.37] | 0, 0, 0, 0.04: n.s. (p rises 0.25) |
| DeepSeek (M4b) | 0.29, 0.54, 0.62: rises, p = 0.014 | 0.29, 0.33, 0.54: rises, p = 0.054 |
| Claude Haiku | 0.38, 0.38, 0.33: n.s. | 0.38, 0.12, 0.00: falls, p = 0.0005 |
| Claude Sonnet | 0.04, 0.17, 0.33: rises, p = 0.007 | 0.04, 0.00, 0.00: n.s. |

- **Reading.** Uninformed agents never restore the majority for Gemma. When they commit at the start, they help the
  strong minority, the same direction as Sonnet and DeepSeek. Only Haiku under observe-first shows Couzin 2011's
  restoring effect.
- **Later data.** The script is rerun when the GPT and Qwen data are in.
