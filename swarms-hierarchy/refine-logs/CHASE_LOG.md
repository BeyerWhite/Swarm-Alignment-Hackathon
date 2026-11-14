# Chase study — lab log and amendments

Chronological. The pre-registration (`CHASE_PREREGISTRATION.md`) is hash-stamped in
`results/chase_e1_freeze.json` and is not edited; changes are recorded here.

## 2026-11-12

**Source check.** Chase et al. (2002) read in full on PMC; Chase (1982, Science) abstract read on PubMed; Chase et
al. (2022, Social Networks) read in full. The sequence-type percentages in Chase (1982, Behaviour) could not be
accessed and are not used. Details in section 1 of the pre-registration.

**E0 (retrospective, no calls).** `results/chase_e0_retro.json`, 31 model × condition cells of the existing N = 8
score-board populations (2–3 runs each).
- Status-sensitive conditions already show Chase's pattern: DD + DS share up to 0.71 and t_tri up to +1.00
  (`norm100`: Gemma-31B 0.70 / +1.00, GLM-5.2 0.71 / +1.00, Haiku 0.67 / +1.00, Sonnet 0.68 / +1.00).
- Status-blind DeepSeek sits at the null: DD + DS 0.46–0.48, t_tri −0.05 to +0.21.
- Equal-turns norm gives *more* cycles than chance in three models: t_tri −0.36 (Gemma-31B, Haiku), −0.32 (Sonnet).
- Against surrogates driven by the pairwise score kernels, observed t_tri is above the 95th percentile in 9 of 31
  cells and below the 5th in none. The surrogate intervals use point-estimate kernels, so they are somewhat too
  narrow (DeepSeek `norm100` is one of the nine). Read this as suggestive only.

**Smoke test.** 444 calls, six models, `norm100`, 1–4 reps per cell, written to `scratch/chase_e1_smoke.jsonl` and
kept out of the E1 data. 0 unparseable replies; 0.05 credits. Inspected only for parse rate and which cells
produced any DEFER.

**Freeze.** `results/chase_e1_freeze.json` written 19:49:24, before any row of `results/chase_e1_kernel.jsonl`.

### Amendment C1 (operational; no change to prompts, cells, seeds or analysis)
- The account is limited to about 50 inference requests per minute ("Inference rate limit reached for this user;
  retry after 60 seconds"). The first launch (24 workers) produced 141 rows in about two minutes and then spent
  its time in 429 back-off. It was stopped; the 141 rows are kept, since job ids and prompts are deterministic.
- New launcher `code/chase_paced.py` spaces requests at 48 per minute. Frozen files are unchanged.
- E1 now runs in priority order instead of fully shuffled, so that the planned population models finish first:
  - tier A: Gemma-31B, GLM-5.2, DeepSeek × `comp`, `norm100` (12,672 calls, about 4.4 h);
  - tier B: the same models × `norm0` (6,336 calls, about 2.2 h);
  - tier C: Gemma-26B, Qwen3.8-27B, Qwen3.5-397B × all framings (19,008 calls, about 6.6 h).
- Consequence for inference: a gate report after tier A applies Holm over the cells available at that time and says
  so; the final E1 table applies it over all 18.

### Amendment C2 (rate-driven; written 2026-11-12 20:35, before any E1 result was analysed)
The owner chose to plan around the 50-per-minute limit (about 2,900 calls per hour). At that rate the pre-registered
plan needs about 60 hours of API time. The changes below cut it to about 40 hours and put the least important
calls last. They were fixed while E1 batch A was 15% complete; the only E1-format rows inspected so far are the
smoke rows (1–4 per cell, parse check only).

1. **Queue order.** E1 batch A → E1 batch B → E1 gate report → E2 → E2 gate report → E3 → E4 → E1 batch C.
   Batch C (Gemma-26B, Qwen3.8-27B, Qwen3.5-397B) is no longer needed before E2; it serves cross-family breadth and
   runs last. Claude and GPT arms, once their CLIs are set up, are added to E1 with the same cells and seed.
2. **Which cells get populations.** E3 runs only for the model × framing cells that are responsive in E1 (Holm over
   the cells of batches A and B), among Gemma-31B and GLM-5.2 × `comp`, `norm100`. This is stricter than section 5
   of the pre-registration, which admitted both framings of a model if either was responsive. The replacement rule
   for a model that is unresponsive in both framings then draws on batch C, so batch C moves ahead of E2 only in
   that case.
3. **E3 group counts.**
   - Groups of four: OWN 40, LOG 40, DYAD 20, REDACT 20 (was 40 each). The primary LOG-versus-OWN test keeps its
     planned size.
   - N = 8: OWN, LOG and SCORE+LOG 6 seeds each (was 8). SCORE: 3 new seeds, reported alongside the 3 hackathon
     runs of the matching legacy condition, which stay labelled retrospective.
   - DeepSeek control: OWN and LOG only, 20 groups of four and 4 seeds at N = 8.
4. **Ceiling rule.** If the frozen E2 forecast puts the OWN arm above 0.85 linear in groups of four for a cell,
   that cell's groups-of-four run is cut to OWN 20 and LOG 20 and its N = 8 run becomes the primary test. The E2
   file states this per cell before any population call.
5. **Call budget after this amendment.** Per responsive cell 18,240 calls (11,520 at N = 4, 6,720 at N = 8); at
   most four cells, 72,960. DeepSeek control 6,400. E2 about 9,000. E4 7,680. E1 batches A and B 19,008; batch C
   19,008. Total at most about 134,000 calls, about 46 hours, of which batch C is the last 6.6.

### Amendment C3 (E2 specification; written 2026-11-12 evening, before any E1 result was analysed)
Section 6 of the pre-registration left the E2 details open. They are fixed here, in code tested on synthetic rows
only (`scratch/e2test/`, toy kernel, no LLM calls). Code: `code/chase_fit.py`, `code/chase_e2.py`.

- **Functional form.** Surrogate groups visit records with |o| and |p| of 10 or more, far outside the E1 grid, and
  the main project's holdout (M5) found the linear logit overconfident at large gaps. The ladder K0–K4 therefore
  uses the signed-log transform s(x) = sign(x)·log2(1 + |x|) of each feature. Two comparison kernels are forecast
  alongside it: LIN (top ladder level on raw features, the strict Bonabeau form) and FLEX (Nadaraya–Watson on the
  transformed features, bandwidth 0.6, no functional form).
- **Features by arm.** DYAD: d. OWN and REDACT: d, o. LOG: d, o, p, tc. SCORE+LOG: the displayed score gap plus
  d, o, p, tc. SCORE: the paper's own state-matched kernel for the matching legacy condition.
- **State-matched calls per responsive cell.** Groups of four: DYAD 150, OWN 300, LOG 450, REDACT 150. N = 8:
  OWN 300, LOG 450, SCORE+LOG 450. DeepSeek control: 100 each for OWN and LOG at both sizes. Records are sampled
  uniformly from decisions in surrogate groups driven by kernels fitted on E1 rows only (top two ladder levels;
  for SCORE+LOG the legacy score kernel alone and combined with the LOG kernel on the logit scale; for REDACT the
  OWN kernel, since E1 has no direct-history cells in the REDACT format).
- **Group-size offset.** E1 rows were measured with an 8-agent system prompt. Fits for groups of four include an
  intercept dummy for rows measured at another size; forecasts set it to zero.
- **Forecast file.** 200 row-bootstrap refits per kernel; each simulates the planned number of groups; the file
  stores the predictive distribution of the pooled statistics, the LOG-minus-OWN contrast, its power (Fisher exact
  on linear counts for groups of four; Mann–Whitney on per-group t_tri at N = 8), the forecast for LOG when the
  bystander features are withheld from the kernel (K2 fitted on LOG rows), and the ceiling flag of amendment C2.
- **Design note from the synthetic test.** With a toy kernel of moderate strength the planned sizes gave power of
  only 0.3 (groups of four, own-record arm already 85% linear) and 0.5 (N = 8, six seeds). If the real forecast
  shows the same, the E2 report will propose a different group size or count before E3, chosen from the forecast
  alone.

## 2026-11-13

**E1 batch A complete (00:22).** 2,112 rows in each of the six cells (Gemma-31B, GLM-5.2, DeepSeek × `comp`,
`norm100`); 10 of 12,672 replies unparseable. No outcome inspected. Batch B was queued behind a wait loop that
matched its own command line and so never started; it was stopped and batch B launched directly at 00:24.

### Amendment C4 (CLI backends; written before any Claude or GPT row of E1 exists)
The owner asked for the `claude` and `codex` CLIs to be set up so the Claude and GPT arms can be added to E1 with
the same cells and seed. Frozen files are unchanged; the additions are in a new launcher, `code/chase_cli.py`.

- **Installed.** Claude Code 2.1.323 (official installer, `~/.local/bin/claude`), signed in with the owner's
  subscription. Codex CLI 0.165.0-alpha.1.2 (the copy bundled with the ChatGPT app, linked to
  `~/.local/bin/codex`); the harness runs it under `CODEX_HOME=~/.codex-api`, not yet signed in.
- **What a `claude -p` call carries under the flags in `llm.py`.** The disallow list predates this CLI version: 14
  tool definitions remain (among them SendMessage and ListAgents), about 4,600 tokens. Launched from inside the
  desktop app the call also inherits the app's environment and carries about 21,000 tokens.
- **Launcher.** `chase_cli.py` clears the desktop app's variables for child processes, adds `--tools ""` and
  `--no-session-persistence` to every `claude` call, and stops a run after 12 consecutive failed calls. With these
  flags a call carries the study's system prompt and user prompt (about 1,000 tokens) plus two things that cannot
  be removed under a subscription sign-in: one sentence prefixed to the system prompt ("You are a Claude agent,
  built on Anthropic's Claude Agent SDK.") and a reminder block holding the account's e-mail address. Both are
  constant across cells and arms. The hackathon's Claude runs used the unmodified flags, so the two sets of
  Claude data are not strictly like for like.
- **Model ids.** `claude:haiku` resolves to claude-haiku-5-5 and `claude:sonnet` to claude-sonnet-5-5.
- **Smoke test.** 50 calls, `claude:haiku`, `norm100`, one per E1 cell, in `scratch/chase_e1_cli_smoke.jsonl`,
  kept out of the E1 data: 50 of 50 parsed on the first try, 23 s at four workers. Inspected for parse rate only.
  API-equivalent cost as reported by the CLI: about $0.0002 per call for Haiku and $0.004 for Sonnet.
- **Inference.** Claude and GPT cells join the family for the final E1 Holm adjustment. They do not enter the
  rule that selects cells for E3, which stays restricted to Gemma-31B and GLM-5.2.

**Amendment C4, addendum (Codex backend; written before any GPT row of E1 exists).**
- Signed in under `CODEX_HOME=~/.codex-api` with the owner's ChatGPT account. `gpt:gpt-6-luna` (the hackathon's GPT
  model) is available.
- A `codex exec` call under the flags in `llm.py` carries about 14,600 input tokens: Codex's own agent instructions
  and tool definitions around a study prompt of roughly 400 tokens, with the study's system prompt placed in the
  user message. Switching off every tool-related feature flag only brings this to about 11,400, so the call is
  left as the hackathon ran it. The GPT arm is therefore a coding agent answering the prompt, not a bare model,
  and is reported with that caveat.
- `chase_cli.py` adds `--ephemeral` (no session file; same token count) and closes standard input for `codex exec`.
  Without the latter, a run started from a non-terminal waits on standard input until the 180 s timeout; the first
  smoke attempt hung this way and wrote no rows.
- Smoke test: 50 calls, `norm100`, one per E1 cell, in `scratch/chase_e1_cli_smoke.jsonl`: 50 of 50 parsed on the
  first try, 47 s at four workers. Inspected for parse rate only.
- `llm.py` caps GPT calls at 1,500 per process unless `GPT_CALL_BUDGET` is set; one E1 cell is 2,112 calls.

**E1 batch B complete (02:40); E1 gate analysis.** 19,008 rows over 9 cells, 12 unparseable. Frozen analysis run on
`results/chase_e1_kernel_AB.jsonl` (batch A and B models only); hashes of all frozen files verified unchanged.
Report: `refine-logs/CHASE_E1_REPORT.md`. Bystander slope β_p < 0 with Holm p < 0.01 in Gemma-31B and GLM-5.2 under
`comp` and `norm100`; nil under `norm0` and in DeepSeek; transitive-inference contrast T nil everywhere.
All four planned cells are admitted; no replacement from batch C.

**E2 calibration launched 02:41** (owner approved automatic start on 2026-11-12): 4 cells × 2,250 + DeepSeek `comp`
control 400 = 9,400 calls, `results/chase_e2_statecal.jsonl`. Stop before any population run stays in place.

**E2 calibration complete (06:20); forecasts frozen (06:24:55).** 9,400 rows, 137 unparseable (GLM-5.2 with the
full log: 4–6%). `results/chase_e2_predictions.json`, sha256 9f2241d0cd4cfd455b8cbeb1bf02d45551c49c1e57b0bcaf4db4ed94d86ef734,
200 bootstrap refits per kernel. Report: `refine-logs/CHASE_E2_REPORT.md`.
- Forecast LOG minus OWN, groups of four linear: Gemma-31B `norm100` +0.39, `comp` +0.23; GLM-5.2 `comp` −0.05,
  `norm100` +0.02; DeepSeek 0.00. No cell triggers the ceiling rule.
- Supplementary sizing from the frozen kernels only (`code/chase_e2_power.py`, `results/chase_e2_power_supp.json`):
  Gemma-31B `comp` has power 0.4–0.5 at the planned sizes, about 0.8 at 80 groups of four and 12 seeds per arm.
- Correction to amendment C2, item 5: measured calls per group are 98 (groups of four) and 329 (N = 8), not 96 and
  280. A full cell is about 19,700 calls.
- A sizing proposal (raise Gemma-31B `comp`, reduce GLM-5.2) is with the owner. **No population call has been
  made; E3 waits for the owner's decision.**

### Amendment C5 (E3 sizes; written 2026-11-13 12:19, after the forecast freeze and before any population call)
The owner chose the focused plan ("Option A") of the E2 report. Every change below is derived from the frozen
forecast file and the supplementary sizing file; no population data exist.

1. **Cells and group counts.**
   - Gemma-31B `norm100`: as in amendment C2 (groups of four: OWN 40, LOG 40, DYAD 20, REDACT 20; N = 8: OWN, LOG,
     SCORE+LOG, SCORE 6 seeds each).
   - Gemma-31B `comp`: the same, with OWN and LOG raised to 80 groups of four and 12 seeds at N = 8. The added
     groups continue the seed sequence (4040–4079, 8006–8011), the same seeds in both arms.
   - GLM-5.2 `comp`: OWN 40 and LOG 40 groups of four only.
   - GLM-5.2 `norm100`: not run.
   - DeepSeek `comp` control: as in amendment C2.
   - This replaces the rule of amendment C2, item 2, that every responsive cell gets the full set of arms.
2. **SCORE arm.** Six new seeds per Gemma-31B cell, the group count the forecast file simulated. Amendment C2,
   item 3, had 3 new seeds read alongside 3 hackathon runs; the hackathon runs are still reported, labelled
   retrospective, but the prospective comparison now rests on new seeds only.
3. **Tests.** Unchanged in kind: Fisher exact on linear counts (groups of four), Mann–Whitney on per-group t_tri
   (N = 8), LOG against OWN.
   - Gemma-31B `comp`: the primary tests use all 80 groups and all 12 seeds. Coverage of the frozen predictive
     interval is scored on the first 40 groups and first 6 seeds, the size at which that interval was computed;
     the full-sample estimate is reported with its own interval against the frozen point forecast.
   - GLM-5.2 `comp`: the forecast under test is no difference; the observed LOG − OWN gap is read against the
     frozen interval [−0.25, +0.15]. Its DYAD, REDACT and N = 8 forecasts, and every GLM-5.2 `norm100` forecast,
     stay in the file as untested and are reported as such, not as confirmed nulls.
   - Multiplicity: Holm over the three groups-of-four LOG-versus-OWN contrasts that are run (Gemma-31B `norm100`,
     Gemma-31B `comp`, GLM-5.2 `comp`). N = 8 contrasts are secondary.
4. **No stopping rule.** All counts are fixed here. While the queue runs only call counts, invalid-reply counts and
   errors are looked at; no outcome is analysed for a cell until that cell's groups are complete.
5. **Budget and order.** 65,492 calls, about 23 hours at 48 per minute: Gemma-31B `norm100` (19,656) → Gemma-31B
   `comp` as planned (19,656) → its added groups (7,840 + 3,948) → GLM-5.2 `comp` (7,840) → DeepSeek control
   (6,552). One paced process at a time. Then E4 and E1 batch C as in amendment C2.

**E3 launched 12:20** (amendment C5 queue, `scratch/chase_e3_queue.sh`, sha256 bc03c3e5…; one paced process at
48 requests per minute). First stage: Gemma-31B `norm100`, 144 groups. A dry count of the six stages gave
144 / 144 / 80 / 12 / 80 / 48 groups, 65,492 calls, matching the amendment.

**E3 evaluation script fixed** while the first stage was running and before any group outcome was read:
`code/chase_e3_eval.py`, sha256 3e16161bb1d3433f76ad4cecd0cf8941f20fe9e557e01c07c8b4bd21b44bcb86.
- It analyses a cell only when all of its amendment C5 groups are present.
- Per arm: observed statistic against the frozen interval (scored on the planned number of groups), the LIN, FLEX
  and K0 forecasts, and the leave-one-seed-out upper bound of section 7 (top-level kernel refitted on the other
  seeds' population decisions; not available for SCORE, whose kernel is the paper's).
- LOG against OWN: Fisher exact and Mann–Whitney as pre-registered, group-bootstrap 95% interval, Holm over the
  three groups-of-four contrasts.
- Tested on synthetic groups driven by the E1 and E2 kernels (`scratch/e3test/`): forecast-driven arms fall inside
  the frozen intervals, a deliberately different SCORE kernel falls outside, incomplete cells are skipped.
- Note for reading the control: arms share seeds, so a model that always leads produces identical groups in OWN
  and LOG. Shared seeds make the unpaired tests conservative.

**E1 Claude arm launched 12:27** (owner chose "Haiku only"): `claude:haiku` (`claude-haiku-5-5`) × `comp`,
`norm100`, same cells and seed as batches A and B, 4,224 calls at four workers through `code/chase_cli.py`
(amendment C4), rows appended to `results/chase_e1_kernel.jsonl`. Runs alongside E3; it does not use the Adaption
rate limit. Sonnet and GPT are not started.

**E3 operational check (12:51, six groups written; no outcome read).** 1,455 calls in 31 minutes (47 per minute),
records well-formed (24 and 40 rounds, two feature rows per contest). 28 replies unparsed: 12 empty responses and
16 where Gemma-31B emits a degenerate string ("our our our …"). Most are recovered by the two retries; six prompts
failed all three tries and became coin flips, about 0.4% of decisions. The evaluation reports invalid counts per arm.

**E1 Claude arm complete (13:12).** `claude:haiku` × `comp`, `norm100`: 4,224 rows, all parsed, no failed calls.
Frozen analysis rerun on `results/chase_e1_kernel_ABH.jsonl` (batches A and B plus Haiku, Holm over 11 cells);
gate outputs kept as `chase_e1_cells_AB.csv` and `chase_e1_summary_AB.json`. Haiku β_p: `comp` −0.011 ± 0.004
(Holm 0.028), `norm100` −0.052 ± 0.007 (Holm < 0.001); T nil. Earlier verdicts unchanged. Addendum in
`refine-logs/CHASE_E1_REPORT.md`. After the run the owner's Pro plan showed 47% of the 5-hour window and 6% of the
week used (this session's own use included; no reading was taken before the run, so the arm's share is unknown).

**E1 GPT arm launched 14:00** (owner approved): `gpt:gpt-6-luna` × `comp`, `norm100`, same cells and seed, 4,224
calls at four workers through `code/chase_cli.py`, `GPT_CALL_BUDGET=7000`, rows appended to
`results/chase_e1_kernel.jsonl`. Uses the owner's ChatGPT plan under `~/.codex-api`; runs alongside E3.

**E1 GPT arm complete (15:28).** `gpt:gpt-6-luna` × `comp`, `norm100`: 4,224 rows, all parsed on the first try.
Frozen analysis rerun on `results/chase_e1_kernel_ABHG.jsonl` (Holm over 13 cells; earlier verdicts unchanged).
β_p: `comp` −0.079 ± 0.010, `norm100` −0.092 ± 0.010 (both Holm < 0.001). T `comp` +0.208 ± 0.095 (Holm 0.37).
Redaction contrast R non-zero in GPT only (−0.146 ± 0.068, −0.312 ± 0.061). Addendum in the E1 report.
Remaining for the final E1 table: batch C (after E3 and E4); Sonnet not started.

**Branch pushed (17:53).** Commit 454d89a on `chase-bystander`, pushed to `origin`. Left out because the running
E3 queue is still writing them: `results/chase_e3_pop.jsonl` and the Adaption spend ledger.

**E1 Sonnet arm started 17:53 and paused 17:56** (owner allowed it if it could run in parallel). `claude:sonnet`
× `comp`, `norm100`: 340 of 4,224 rows written, all parsed; no outcome looked at. Paused by hand because of plan
usage: the owner's 5-hour window went from 9% to 24% over these calls (this session's own use included), so the
full arm needs roughly one and a half to two windows and would lock the session out mid-run. The rows stay in
`results/chase_e1_kernel.jsonl`; a restart continues from them. How to run the remainder is with the owner.

**Batch C not started.** It uses the same Adaption account as E3, and the 50-per-minute limit is per user, so
running it now would only take calls from E3. It stays after E3 and E4 (amendment C2).

**E1 Sonnet arm in chunks (owner's choice, 20:41).** `scratch/chase_e1_sonnet_chunks.sh` runs the normal E1 runner
and stops it after a fixed number of new rows, one chunk per 5-hour plan window (starts 5 h 5 min apart). First
chunk 20:41–20:45: 413 rows, total 753 of 4,224, all parsed. The window then stood at 55%, so 753 Sonnet calls cost
about 40% of a window, more than the third estimated. Remaining chunks cut from 750 to 600 rows; next start 01:46,
six chunks, last one about 03:10 on 2026-11-15. No Sonnet outcome is looked at until the arm is complete.

**E3, first cell complete (19:19): Gemma-31B `norm100`.** 144 groups; stage 2 (Gemma-31B `comp`) running.
Evaluation by the hash-logged `code/chase_e3_eval.py` (sha256 3e16161b…, unchanged) on a snapshot of complete
records; output `results/chase_e3_eval.json` (will be rewritten as cells complete). Invalid replies at most 0.6% of
decisions per arm.

| Arm | Statistic | Observed | Frozen forecast [5%, 95%] | In interval | Leave-one-seed-out bound |
|---|---|---|---|---|---|
| OWN, groups of four | linear | 0.78 (31/40) | 0.56 [0.42, 0.70] | no | 0.71 |
| LOG, groups of four | linear | 0.95 (38/40) | 0.95 [0.90, 1.00] | yes | 0.96 |
| DYAD, groups of four | linear | 0.55 (11/20) | 0.37 [0.20, 0.55] | yes (edge) | 0.38 |
| REDACT, groups of four | linear | 0.75 (15/20) | 0.60 [0.40, 0.80] | yes | 0.68 |
| OWN, N = 8 | t_tri | 0.32 | 0.08 [−0.08, 0.24] | no | 0.29 |
| LOG, N = 8 | t_tri | 0.79 | 0.75 [0.62, 0.86] | yes | 0.88 |
| SCORE+LOG, N = 8 | t_tri | 0.98 | 0.99 [0.96, 1.00] | yes | 0.99 |
| SCORE, N = 8 | t_tri | 1.00 | 0.83 [0.68, 0.94] | no | n/a |

- **LOG minus OWN.** Groups of four: +0.17, bootstrap 95% interval [+0.03, +0.32], Fisher p = 0.048 (Holm pending
  the other two cells); frozen forecast +0.39 [+0.25, +0.53], observed outside. N = 8: +0.46 [+0.04, +0.86],
  Mann–Whitney p = 0.056; frozen +0.68 [+0.44, +0.89], observed inside.
- **Reading.** The LOG forecast was right. The OWN forecast was too low: groups whose agents see only their own
  record are much more linear than the E2 kernel said (0.78 against 0.56; kernel-null 0.38), and a kernel refitted
  on those populations' own decisions reaches 0.71. So the direction of H3 holds, but the bystander gain is less
  than half the forecast because the baseline was underestimated, not because LOG fell short.
- **Chase's sequences.** DD + DS share under LOG is 0.70 at both sizes (OWN 0.53 and 0.51), above the frozen
  forecast (0.59 [0.53, 0.65]) and above the leave-one-seed-out bound (0.57, 0.63). The bystander-dominates-
  initial-dominant sequence is rare under LOG (0.09 and 0.11; a quarter under OWN). The frozen forecast that the
  sequence signature would be weak is wrong for this cell.
- **Kernel form.** For LOG the ladder forecast is on target (0.95; LIN 0.98; FLEX 0.77, observed outside). For OWN
  every form is too low.

**Post hoc (not pre-registered): why the OWN forecast was low in Gemma-31B `norm100`.** Asked by the owner after
the first cell. Population decisions of the OWN arm were tabulated against the frozen K2 kernel and its
calibration rows.
- Populations reach deep losing records that the calibration barely sampled. In groups of four, 10% of decisions
  (382 of 3,824) were made at an own net record of −6 or worse; the state-matched calibration has 2 of 299 rows
  there, and the E1 grid stops at ±2.
- In those states agents nearly stop leading: LEAD 0.16 (groups of four) and 0.30 (N = 8). The frozen kernel,
  evaluated on the same states, gives 0.33 and 0.61. At −6 to −3 it is 0.50 against 0.60, and 0.69 against 0.81.
  From −3 upward kernel and populations agree within 0.05.
- Refitting the same kernel form on population decisions raises the own-record coefficient from 0.69 to 1.05
  (groups of four) and 1.31 (N = 8), and the direct-record coefficient from 1.68 to 2.08 and 2.32.
- Reading: the loser effect is a threshold beyond the measured range, and it feeds itself (deferring produces
  losses, which deepen the record). The calibration states came from surrogate groups driven by the E1 kernel,
  which had almost no own-record response, so those surrogates rarely produced deep losers. The LOG kernel did not
  have this problem because the partner-record response was already strong inside the E1 grid.
- The leave-one-seed-out bound (0.71) is still below the observed 0.78 but the observation is inside its interval;
  whether features outside the kernel (order or recency of the lines) add anything is not established.
