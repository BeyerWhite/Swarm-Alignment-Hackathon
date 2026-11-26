# Couzin study — lab log

Pre-registration: `COUZIN_PREREGISTRATION.md` (stamp `results/couzin_prereg_freeze.json`). Proposal:
`COUZIN_PROPOSAL.md`. Amendments are numbered M-A1, M-A2, …

## M-0 (2026-11-20/21): approval, build and offline checks (before the stamp; no API calls)

- **2026-11-20.** The owner chose the full program. Decisions 2–5 were left open; the defaults are in
  pre-registration §0:
  - a 20-credit study guard;
  - cheap M4b candidates only;
  - queue after everything already waiting;
  - no Claude or GPT kernels.
- **Branch and worktree.** Branch `couzin-minorities` in `.worktrees/couzin`, cut from `rosen-specialization`
  (2be2929). It uses the shared ledger via `ADAPTION_LEDGER_DIR`.
- **Hartnett et al. (2016) model, as restated by Sarkanych et al., arXiv 2409.11475:**
  - G = ½[1 + tanh(b h)/tanh(b)], with h = (ω n₊ − n₋)/(ω n₊ + n₋);
  - b → 0 gives the voter model; b → ∞ gives majority rule.
  - The simulator check (pre-registration rule 7) passed. P(majority) over ρ₀ = 0, .3, .6, .9 was
    0.04, 0.11, 0.23, 0.38 at b = 0.3 and 0.92, 0.84, 0.71, 0.59 at b = 10.
  - **Reading:** uninformed agents mainly dilute toward a coin flip, and nonlinearity decides who wins without them.
    So H6 (falling below 0.5), not H5, is the strong Couzin 2011 claim.
- **Offline check, engine against simulator** with the exact tables of the scripted agents (8 arms, 150 groups):
  every outcome agreed within |z| < 1.7. Passed.
- **Estimator fix found by the dry run.** The first estimator, (s + ½)/(n + 1) with Beta bootstrap draws, put a
  floor of about 2% on "leave a 5-of-5 majority". Compounded over N agents and 5 rounds, that broke simulated
  consensus: M4 n_u = 0 was forecast at 0.58 against 0.89 with exact tables. The proposal had specified an unshrunk
  estimator, so the code now uses s/n with a monotone fit and binomial bootstrap draws.
- **Interval calibration.** 6 independent offline replications at the real draw counts (20 per cell, 40 for the
  primary model, plus M2 calibration), 9 arms. The 90% kernel interval contained the exact-table truth in 49 of 54
  cases (M3 cells 24/24, M4 arms 21/24, M6 6/6). The `DRY=1` run uses half the draws and a quarter of the groups,
  so its lower coverage (21/33) is expected.
- **Sensitivity to note.** The ω rule compares estimated cells with fixed thresholds. In one replication the
  scripted weak agents sat at the threshold (true P(follow 4 of 5) = 0.51 at ω = 2) and the rule chose weak ω = 2
  instead of 0.5. That changes the M4 composition, but the forecast uses the same choice, so the test stays valid.
  The chosen ω values are reported with their cell estimates.
- **Deadlock found before launch.** The hierarchy G0 wrapper (pid 29071) waits while any process matching
  `queue[0-9]*\.sh` runs. A Couzin queue waiting under such a name for that wrapper would deadlock. The queues are
  therefore started through `code/couzin_launch.sh`, and the busy check only counts running stage scripts.
- **Power at the dry-run scale.** The power rule gave 0.43 at 40 groups for the scripted kernel, so H1 would be
  reported as an estimate. The real value comes from the frozen forecast.

## Owner decisions, 2026-11-21 (no amendment needed: the stamped files are unchanged)

- **Cap raised to 200.** The shared Adaption cap goes from 150 to 200. `.secrets/adaption.env` in the main checkout
  now says 200. Couzin queue 1 was still waiting, so it was relaunched with cap 200
  (`couzin_launch.sh 1 200 29071 29645 29491 28523`, log `scratch/couzin_queue1.log`). Queue 2 is launched with 200.
  The study's own 20-credit guard is unchanged.
- **Conditional follow-up (not part of this pre-registration).** If M4b shows interesting results, the M4b design is
  to be run with Claude, GPT and Qwen as well.
  - The trigger and the design are fixed in a new amendment once the M4b results are in, before anything runs.
  - Each new model first needs its M1 kernel (about 1,400 calls), so the forecast can predict its sign before its
    populations run.
  - **Cost and limits:**
    - Qwen3.5-397B: about 5–6 credits for kernel plus populations.
    - Claude: runs through the `claude` CLI, uses no Adaption credits, and is slow.
    - GPT: runs through `codex`. `llm.py` limits it to 1,500 calls (`GPT_CALL_BUDGET`). A population arm set of
      about 18,000 calls needs the owner to raise that budget or accept smaller arms.

## M-A1 (2026-11-21): Claude follow-up with a stop-on-quota rule

- **Owner request:** "add the stop-on-quota rule, and then run the Claude experiments." The design is
  `COUZIN_CLAUDE_FOLLOWUP.md`: the M4b arms on Claude Haiku, then Sonnet, through `claude -p`, with their own gate,
  kernel, calibration and frozen forecast per model. It is stamped in `results/couzin_followup_freeze.json`.
- **The main study is unchanged.** Its stamped files and its queue 1 (still waiting for the Adaption account) are
  untouched.
- **Offline checks:**
  - a `DRY=1` run of `code/couzin_claude_run.sh` passed;
  - a forced quota stop left only complete groups, and the rerun resumed without duplicates.
- **Added before the stamp:** if the forecast change from n_u 0 to 24 is below 0.05, F1 has no sign test. Coverage
  is the test instead, because the sign of a near-zero forecast slope means nothing.
- **Smoke test:** one real decision each from Haiku and Sonnet, each a one-word valid reply in about 4 s
  (`scratch/couzin_followup_smoke`; not data).
- **Launch:** `nohup caffeinate -i bash code/couzin_claude_run.sh`, log `scratch/couzin_claude_run.log`.

## Queue 1 done (2026-11-23 07:13–07:33)

- **Gate:** all six models passed. The frozen label pair is ORION / LYRA (`results/couzin_m0_gate.json`).
- **Pilot** (3 groups; never tested): both M3_N10_p0.2 groups reached target consensus, at rounds 3 and 2. In the
  M4_obs_nu6 group the weak majority won (share at the minority's site 0.24).

## M-A2 (2026-11-23): budget meter fixed; new stamp

- **The bug.** `couzin_budget.py` read costs from this worktree's `results/calls.jsonl`. This worktree's `llm.py`
  (the `rosen-specialization` tip) does not log usage there, so every call was charged the 0.001 fail-closed amount.
  - The meter reported 1.656 credits for queue 1.
  - The real spend, matched against the shared ledger, was 0.057.
  - At about 0.001 per call, the 20-credit guard would have stopped queue 2 after about 20k of its ~120k calls.
- **The fix.** Each Couzin call is now matched to its shared-ledger entry: same model, nearest timestamp within
  2 s. Every queue 1 call matched, with a largest gap of 2 ms. An unmatched call with a reply still counts 0.001.
- **New stamp.** `couzin_budget.py` is a stamped file, so the old stamp is kept as
  `results/couzin_prereg_freeze_v1.json` and a new `results/couzin_prereg_freeze.json` was written. Nothing else
  changed. The Claude follow-up's own stamp does not cover this file.

## 2026-11-23 18:46–18:49: Claude run resumed; queue 2 launched

- **Claude run.** It had stopped on the stop-on-quota rule on 2026-11-21 at 19:59, during the Haiku groups. 121 of
  128 Haiku groups were done, with no invalid decisions. It was resumed with the same command; the done steps were
  skipped.
- **Interim Haiku result** (not the registered evaluation, which runs when all groups are in):
  - Nonlinearity index a = 4.72, a strongly conformist response.
  - The ω rule fell back to strong = 6, because no level reached 90% staying.
  - P(minority wins): simultaneous start 0.38 / 0.38 / 0.33 at n_u 0 / 12 / 24; observe-first 0.12 / 0.00 at
    n_u 12 / 24 (17 of 24 groups).
  - The n_u = 0 arm is outside its forecast interval: 0.38 observed against 0.68 forecast, [0.46, 0.88]. The other
    four arms are inside.
- **Label screen.** Most models choose the first-named option at round 0: P = 1.00 for Gemma-31B, Gemma-26B and
  Qwen3.5-397B. The naming order is randomised per call, so round-0 choices are a fair coin, not a label bias.
- **Queue 2** was launched through `couzin_launch.sh 2 200 72577 60296`. It waits for the RT phase 2 launcher
  (72577) and the Rosen phase 3 waiter (60296). Neither name matches the busy pattern, so their pids are waited for
  explicitly; after that it waits for an idle account.

## 2026-11-24 00:30: auto-resume of the Claude follow-up (owner OK)

- **What it does.** `code/couzin_claude_autoresume.sh` (not a stamped file) waited for the running runner
  (pid 48268). After every stop-on-quota exit (code 75) it waits 30 minutes and reruns `couzin_claude_run.sh`,
  for at most 60 attempts.
- **When it stops.** On completion, or on any other failure, so a real error is never retried blindly.
- **Effect on the data.** None. The stop-on-quota rule still keeps every affected decision out of the data.
- **Logs.** Earlier runner logs are appended to `scratch/couzin_claude_run_all.log`; the wrapper's own log is
  `scratch/couzin_claude_autoresume.log`.
- **Progress at hand-over:** Sonnet M4b_sim_nu0 24/24 and M4b_sim_nu12 11/24.

## 2026-11-24 00:32: owner reordered the account; M-A3 evaluator fix

- **Reorder (owner request: Couzin queue 2 next in line).** Both jobs restarted below were only waiting, so nothing
  in flight was interrupted.
  - Couzin queue 2 was relaunched as `couzin_launch.sh 2 200 72577` (pid 71693). It waits only for the running RT
    phase 2 job, then for an idle account.
  - The tournament T-A1 retest (`tour_queue3.sh`) was relaunched inside a `bash -c` wrapper that waits for pid 71693,
    then execs the same command. The Couzin busy check ignores `bash -c` wrappers, so the two cannot deadlock. A note
    is in `tournament/.../scratch/tour_queue3.log`.
  - Rosen phase 3 (`rosen_p3_wait.sh`, unchanged) waits while any `queue[N].sh` text is visible, so it stays behind
    both.
- **M-A3: evaluators run a test only on complete arms.**
  - *The bug.* `couzin_followup.py eval` computed F1 and F2 from whatever n_u levels had data. A model with only the
    n_u = 0 arm produced a meaningless slope that entered the Holm correction. `couzin_eval.py` had the same weakness
    for H1–H6, H9–H11 and M7.
  - *The fix.* A test now runs only when every arm it uses has its planned number of groups: 24 per arm in the
    follow-up, and the frozen forecast's `groups` in the main study. Otherwise it is reported as "incomplete" and
    stays out of Holm. In the follow-up, a complete set of arms with no variation (every group scoring the same) is
    reported as "slope not estimable", and F2 counts as not supported.
  - *Checks.* On the real Claude data, Haiku's numbers are unchanged; Holm is now over Haiku's own two F1 tests, so
    F1obs p_holm = 0.0031, as in the interim report. Sonnet is all "incomplete". On the dry-run data, all tests ran
    with complete arms; with M4_obs_nu24 removed, only H5obs and H6 were skipped.
  - *Stamps.* Main study: the old stamp is kept as `results/couzin_prereg_freeze_v2.json`. Follow-up: the old stamp
    is kept as `results/couzin_followup_freeze_v1.json`. Both were then restamped. No data were evaluated as final
    before the fix.

## 2026-11-24 00:39: owner puts the tournament retest second in line

- **New order:** RT-R (running) → tournament T-A1 retest → Couzin queue 2 → Rosen phase 3.
- **Tournament retest.** Relaunched unchanged and directly as `tour_queue3.sh 150 58327 72577` (pid 83181), no
  longer inside the waiting wrapper. Couzin's busy check matches it, so Couzin queue 2 (pid 71693, unchanged) starts
  only after it ends. This was verified live: the check reported "busy".
- **Rosen phase 3.** The waiter was relaunched as `rosen_p3_wait.sh 200 71693`. Without that, Rosen and Couzin
  could both have started in the gap after the retest. Couzin's launcher name does not match Rosen's pattern, and
  their idle checks have the same timing. Waiting on pid 71693 covers the whole of Couzin queue 2, because the
  launcher execs the queue under the same pid.
- **Interruptions.** None: every restarted process was only waiting. Notes were left in both studies' logs.

## 2026-11-26: both runs complete; final report

- **Claude follow-up.** Finished 2026-11-24 16:03 after 26 auto-resume attempts. Every arm has 24 groups. Results
  are in `results/couzin_claude/couzin_followup_eval.json`.
- **Queue 2.** Ran 2026-11-24 08:51 to 2026-11-26 08:27 with no stage failure. The study's total Adaption spend is
  1.98 credits. Results are in `results/couzin_eval.json` and `couzin_eval.txt`.
- **Report.** `COUZIN_FINAL_REPORT.md` replaces the interim report.
- **Two analyses added after the fact (not registered):**
  - Consensus against the number of informed agents and group size (§2.1 of the report).
  - The label tally of M3 consensus outcomes (ORION 230, LYRA 131).
- **H5 is degenerate.** Gemma's minority won 0.00 in 6 of 7 arms, so the logistic slope is not estimable
  (separation; SE ≈ 40). It is reported as "not supported: nothing to reduce". The M-A3 rule does not cover
  separation; a later analysis would need an exact or Firth test.
