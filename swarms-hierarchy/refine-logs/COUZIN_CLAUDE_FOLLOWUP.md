# Couzin study, amendment M-A1 — the M4b design on Claude models

Frozen before any Claude call of this follow-up (stamp: `results/couzin_followup_freeze.json`, check with
`code/couzin_followup.py check`). This is an addition to `COUZIN_PREREGISTRATION.md`; it changes nothing there, and
the main study's stamped files are untouched.

## 0. Owner decision

On 2026-11-21 the owner asked to add the stop-on-quota rule and run the Claude experiments now. Earlier the same day
the plan had been to run them only if M4b looked interesting, after M4b. The owner chose not to wait.

- **Models:** Claude Haiku, then Claude Sonnet, through `claude -p` (`llm.py` backend `claude:`, custom system
  prompt, all tools disallowed). This was my default, since "the Claude experiments" named no specific model.
- **No Adaption credits.** The run neither waits for nor blocks the Adaption queues.
- **GPT and Qwen** stay conditional on M4b as before. GPT also needs `GPT_CALL_BUDGET` raised (1,500 calls).

## 1. Design (identical to M4b unless stated)

- **Same game and engine** (`code/couzin.py`, frozen). Results go to `results/couzin_claude/`.
- **Gate (C0):** the M0 payoff items and label screen for the two Claude models, 20 draws (≈ 480 calls).
  - Same pass rule as M0.
  - The label pair is chosen by the M0 rule **over the Claude models**. The main study's pair is not yet known; the
    target label is randomised per group, so a different pair adds no direction.
- **Kernel (C1):** the 70 main M1 cells × 20 draws per model (1,400 calls). No robustness sets.
- **Rules:** the frozen ω rule and nonlinearity index, per model (`couzin_m1_rules.json`, write-once per model).
- **Calibration (C2):** M2c_obs_nu6, 8 groups per model (≈ 770 calls), pooled into the kernel. Never tested.
- **Forecast (write-once per model, `couzin_forecast_<model>.json`):**
  - the Hartnett check must pass;
  - 200 bootstrap kernel draws × 100 simulated groups;
  - 90% intervals for the arm means at 24 groups;
  - the forecast slope of P(minority wins) over n_u ∈ {0, 12, 24}, per protocol.
- **Populations:** M4b_sim_nu0, M4b_sim_nu12, M4b_sim_nu24, M4b_obs_nu12 and M4b_obs_nu24, 24 groups each (≈ 17,400
  calls per model).
  - Composition: 4 strong-minority agents against 7 weak-majority agents, plus n_u uninformed agents, all the same
    Claude model.
- **Total:** about 19,600 calls per model. That is 4.5–8 h for Haiku at the 2,000–4,200 calls per hour seen in
  earlier CLI runs; Sonnet is probably similar, but up to about 32 h at its slowest recorded rate.
- **Concurrency:** `LLM_MAX_CONC = 8`.

## 2. Stop-on-quota rule

- `llm.py` turns a CLI usage-limit or login message into an empty reply. The engine would then keep the agent's
  previous choice, so a quota hit would silently bias groups toward staying put.
- `code/couzin_followup.py` wraps `llm.call`. Once `llm.py` detects any quota message, every later call raises before
  or after it. The process exits with code 75 and the runner stops.
- A group is written only after all of its decisions succeed. A kernel row is written only after its decision
  succeeds. So no written row contains a decision affected by the limit.
- After the limit resets, the same command resumes; runners skip rows already written.
- **Offline test, 2026-11-21:** a forced stop during the Haiku groups left 18 complete groups with no invalid
  decisions; the rerun resumed with no duplicates.

## 3. Tests (per model; `couzin_followup.py eval` → `couzin_followup_eval.json`)

| Test | What | Supported if |
|---|---|---|
| F1sim, F1obs | Logistic slope of `majA` (minority wins) on n_u, per protocol, against the frozen forecast slope | The observed slope has the forecast sign (one-sided z in the forecast direction; Holm over models × protocols). Only applies if the forecast change from n_u 0 to 24 is at least 0.05; otherwise the forecast is "flat" and coverage of the arm intervals is the test |
| F2sim, F2obs | The H5 analogue: the same slope | < 0 (one-sided) |
| F3 | The H6 analogue: observe-first, n_u 12 and 24 pooled, P(minority wins), Wilson one-sided 95% upper bound | < 0.5 |
| Coverage | Arm means inside their 90% intervals | Reported (5 arms per model) |

Comparison with Gemma (M4) and the rule-selected M4b model is descriptive and is done once those data exist. The
question is whether the sign of the uninformed-buffer effect follows the measured nonlinearity index across all the
models.
