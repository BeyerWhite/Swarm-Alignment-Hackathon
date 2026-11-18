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

**2026-11-14 16:00 — E3 complete.** The queue of amendment C5 finished with every stage at exit 0: Gemma-31B
`norm100` (13 Nov 12:20–19:19), Gemma-31B `comp` with its extra groups (19:19–06:50), GLM-5.2 `comp` (06:50–13:26),
DeepSeek `comp` control (13:26–16:00). 508 groups in `results/chase_e3_pop.jsonl` (sha256 5967fb05…878e6868), none
partial. Evaluated once with `code/chase_e3_eval.py` (hash unchanged, 3e16161b…); forecast file unchanged (stamp
9f2241d0…). Report: `refine-logs/CHASE_E3_REPORT.md`.
- LOG − OWN, groups of four: Gemma-31B `comp` +0.25 [+0.10, +0.40], Fisher p = 0.0025, Holm 0.008 (80 groups per
  arm); Gemma-31B `norm100` +0.17 [+0.03, +0.32], p = 0.048, Holm 0.095; GLM-5.2 `comp` 0.00, p = 1.
- LOG − OWN, N = 8 t_tri: Gemma-31B `comp` +0.54 [+0.35, +0.73], Mann–Whitney p = 0.0002 (12 seeds);
  `norm100` +0.46 [+0.04, +0.86], p = 0.056 (6 seeds); control +0.14 [+0.02, +0.27], p = 0.14 (4 seeds).
- Frozen forecasts: 17 of 22 arm forecasts inside their 90% interval, all five misses toward more hierarchy than
  forecast; 5 of 7 LOG − OWN contrasts inside.
- Adaption spend at completion: 31.26 total, 13.2 credits since the baseline of 18.06 (cap 60).

**2026-11-14 16:10 — E4 and E1 batch C queued (before any E4 call).** One paced process at a time, as in amendment
C2: first E4, then batch C. Script `scratch/chase_e4c_queue.sh`.
- E4 as pre-registered (section 8): fixed membership Gemma-31B, GLM-5.2, Qwen3.8-27B, DeepSeek-v4-flash, one agent
  each; framing `comp`; arms OWN and LOG; 40 re-runs per arm, seeds 9000–9039, the same seeds in both arms; 24
  rounds. About 7,680 calls. Qwen3.8-27B enters E4 before its E1 cells exist (batch C runs afterwards); the
  pre-registration fixed the membership, so this is not a deviation, but nothing is known yet about how that
  model responds to records.
- Batch C: Gemma-4-26B-A4B, Qwen3.8-27B, Qwen3.5-397B × `comp`, `norm100`, `norm0`, 2,112 calls each (65 rows
  already exist from the stray start on 12 Nov and are skipped). It appends to the same kernel file as the Sonnet
  chunks; rows are written one line at a time.
- No E4 outcome is read until both arms are complete and the E4 evaluation script's hash is in this log.

**2026-11-14 16:11 — E4 launched (16:10) and its evaluation fixed.** `code/chase_e4_eval.py`, sha256
d7c05405da783e38993e2b2584fa96c347982f0f87e9d19aef1dd65d1ecf254d, written and tested only on E3 groups relabelled as a fake rewind file
(`scratch/e4test/`); 0 E4 groups were on disk at that moment and none was read. Fixed in the script: a re-run's
hierarchy is its set of final relationships over the six dyads; modal share and the probability that two re-runs
are identical, per arm; LOG versus OWN by Fisher exact on linear and on modal counts; bootstrap intervals over
re-runs with the modal pattern held fixed and copies of the same re-run excluded from pairs. It refuses to run
unless both arms hold all 40 re-runs. Chase's 2002 figures are quoted only as the pre-registration quotes them.

**2026-11-15 02:00 — E4 and E1 batch C complete.** Queue exit 0: E4 14 Nov 16:10–19:25 (80 groups), batch C
19:25–02:00 (19,008 rows in total for the three models, none invalid). Adaption spend 34.01 total, 15.95 credits
since the baseline (cap 60).
- E4, evaluated once with `code/chase_e4_eval.py` (hash unchanged, d7c05405…): linear OWN 0.50, LOG 0.60,
  difference +0.10 [−0.12, +0.32], Fisher p = 0.50; two re-runs identical 0.03 (OWN) and 0.02 (LOG); modal
  hierarchy 4/40 and 5/40. H6: low reproducibility supported, "not higher under LOG" consistent, "linearity higher
  under LOG" not supported. Report: `refine-logs/CHASE_E4_REPORT.md`.
- E1, frozen analysis (hash unchanged, 8057f824…) on `results/chase_e1_kernel_ABCHG.jsonl` (46,464 rows, all arms
  except Sonnet), Holm over 22 cells: β_p Gemma-4-26B −0.069 (`comp`), −0.028 (`norm100`), −0.108 (`norm0`);
  Qwen3.5-397B −0.082, −0.077, −0.046; Qwen3.8-27B −0.008, +0.008, +0.005 (unresponsive). 14 of 22 cells
  responsive, all through β_p; no T survives adjustment in any cell. Earlier verdicts unchanged. Addendum in
  `refine-logs/CHASE_E1_REPORT.md`.
- Two models respond to the partner's record under `norm0`. The pre-registration makes a `norm0` population run
  (H5, exploratory) conditional on this. None is planned in amendment C5 and none has been started.
- Sonnet: 3,771 of 4,224 rows; last chunk due 03:11. Not analysed.

**2026-11-15 03:15 — Sonnet arm complete; final E1 table.** Last chunk 03:11–03:14, runner exit 0; 4,224 rows, 4,224
distinct jobs, none invalid. Frozen analysis (hash unchanged, 8057f824…) on the full
`results/chase_e1_kernel.jsonl`, 50,688 rows, 24 cells, Holm over 24.
- Sonnet `comp`: LEAD in 2,112 of 2,112 calls; unresponsive. Sonnet `norm100`: β_p −0.069 ± 0.008 (Holm < 0.001),
  LEAD 1.00 up to p = +1 and 0.54 at p = +3; β_d +0.191 ± 0.019; T +0.021 ± 0.021; R −0.010 ± 0.010.
- Final: 15 of 24 cells responsive, all through β_p < 0; no T survives adjustment (largest GPT `comp`, Holm 0.69).
  Earlier verdicts unchanged. Addendum and the 24-cell table are in `refine-logs/CHASE_E1_REPORT.md`.
- Every stage of amendments C2 and C5 is now complete. Nothing is running. The conditional `norm0` population run
  (H5) has not been started and awaits the owner's decision.

### Amendment C6 (E5: equal-turns population run; written 2026-11-15 16:48, before any E5 call)

The owner approved this run on 15 Nov ("Yes, run it") after the batch C result. It is the conditional, exploratory
population test of H5 (pre-registration section 2): under `norm0` the LOG − OWN difference reverses or vanishes.

1. **Cell.** Gemma-4-26B-A4B under `norm0`, chosen because it has the largest partner-record slope under `norm0`
   in E1 (β_p −0.108 ± 0.009; Qwen3.5-397B −0.046). One cell only. No control cell.
2. **Arms and counts.** Groups of four, 24 rounds: OWN 40, LOG 40, DYAD 20, REDACT 20, seeds 4000+. N = 8, 40
   rounds: OWN 6, LOG 6, seeds 8000+. No score arms: the legacy score kernel does not cover this model.
3. **Calibration.** Pass 1 exactly as E2 (1,800 state-matched calls). Pass 2 is new: 825 further calls at records
   visited by surrogate groups driven by the kernels refitted on E1 + pass-1 rows at the population's own group
   size. Reason: the post hoc diagnosis of 13 Nov found that surrogates driven by E1 kernels under-sample the
   records real populations reach. On the old Gemma-31B `norm100` data (dry run, no calls) pass 2 raises the share
   of own-record states at −6 or worse in groups of four from 2 of 299 to about 5%. Forecasts in this cell are
   therefore not made by the same procedure as the E3 forecasts.
4. **Forecast.** `code/chase_e5.py predict` → `results/chase_e5_predictions.json`, write-once and stamped, same
   kernels, bootstrap and intervals as E2. The queue starts the populations as soon as the forecast is frozen and
   writes the stamp into this log first. Deviation from pre-registration section 6: the owner does not see this
   forecast before the populations start; the approval covered the run as a whole.
5. **Tests, fixed now.** As E3: LOG versus OWN by Fisher exact on linear groups of four and Mann–Whitney on
   per-group t_tri at N = 8, group-bootstrap 95% intervals, every arm against its frozen interval, the
   leave-one-seed-out bound. Evaluation by the unchanged `code/chase_e3_eval.py` (3e16161b…) through
   `code/chase_e5_eval.py` (73d07343…), which only sets the cell, the counts and the file names. The cell is
   exploratory, reported unadjusted, and is not added to the Holm family of E3.
6. **Reading of H5, fixed now.** "Reverses": LOG − OWN linear share below zero with a 95% interval excluding zero.
   "Contradicted": LOG − OWN above zero with p < 0.05. Anything else is reported as the interval it is; a
   non-significant difference is not counted as "vanishes" unless the interval excludes +0.17, the smaller of the
   two differences found for Gemma-31B in E3.
7. **No outcome is read until the cell is complete.** Operational checks (group counts, error lines) only.
8. **Budget.** About 18,000 calls, 7 hours, 1.6 credits (34.01 total spent, cap 78.06). The estimate given to the
   owner was 14,000 calls and 5 hours; the difference is the DYAD and REDACT arms and the second pass.
9. **Files.** Calibration rows go to `results/chase_e5_statecal.jsonl`; the frozen E2 files are not touched.
   Populations are appended to `results/chase_e3_pop.jsonl`; the E3 hash 5967fb05… refers to its first 508 lines.
   Wrapper `code/chase_e5.py` 7f2cf7b0…; queue `scratch/chase_e5_queue.sh` db475bde….

**2026-11-15 17:43 — E5 forecast frozen (written by the queue script, before any E5 population call).**
`results/chase_e5_predictions.json`, stamp ce785e89e7672db4d19382af8c9b19c14b7af3f05a672b942f181f4d4fa51eec, file sha256 adbaef6fc3943810846ef02557a5b09919460d981e3f321e3ba6c24c71623484.

**2026-11-15 23:30 — E5 complete.**
Queue ran 16:48–23:02, exit 0, no failed calls: calibration 1,800 + 825 rows, forecast frozen 17:43 (entry above),
132 population groups (N4: OWN 40, LOG 40, DYAD 20, REDACT 20; N8: OWN 6, LOG 6), 15,270 decisions, 0 invalid.
`results/chase_e3_pop.jsonl` now 640 lines, sha256 6434d9a72a84f0d35a4357ed50f883c86b49adea4a0f3250fb4e6b3b795835d0
(first 508 lines still 5967fb05…). Hashes of `chase_e3_eval.py` (3e16161b…), `chase_e5_eval.py` (73d07343…) and
`chase_e5.py` (7f2cf7b0…) checked before evaluation. Spend 34.67 total (E5: 0.66 credits, 17,895 calls).
Evaluation → `results/chase_e5_eval.json`; report `refine-logs/CHASE_E5_REPORT.md`.
- N4 linear: OWN 0.50 (20/40), LOG 0.47 (19/40), LOG − OWN −0.03 [−0.25, +0.20], Fisher p = 1.0.
  N8 t_tri: −0.04 and −0.04, +0.01 [−0.20, +0.21], Mann–Whitney p = 1.0.
- H5 by the rule of amendment C6 item 6: not "reverses", not "contradicted", and not counted as "vanishes" (interval
  does not exclude +0.17). Unresolved.
- Forecast coverage 5 of 6 arms, 2 of 2 contrasts; the miss is OWN N4 (0.50 against 0.24 [0.12, 0.35], LOSO 0.32
  [0.20, 0.45]) — the second calibration pass did not remove the OWN shortfall.
- Post hoc, unadjusted: LOG lowers the LEAD rate in this cell (0.33 → 0.28) as in both Gemma-31B cells, but here it
  lowers the share of contests decided by the answers (0.39 → 0.29, [−0.13, −0.07]) where there it raised it;
  REDACT linear 4/20 against OWN 20/40 (Fisher p = 0.029).

**2026-11-16 01:05 — Final report written.** `refine-logs/CHASE_FINAL_REPORT.md` summarises E0–E5; its numbers were checked against the evaluation files. No new analysis and no model calls.
Correction to `CHASE_E5_REPORT.md`: the own-record forecast for groups of four was too low in Gemma-31B `norm100` in E3, not in both Gemma-31B cells (under `comp` the observation was inside the interval).

## 2026-11-16

### Amendment C7 (source verification and a post hoc re-count; written after all data and the first final report)
The owner supplied both 1982 papers as PDFs. Read in full on 16 Nov: Chase (1982), Science 216: 439–440, and Chase
(1982), Behaviour 80: 218–240. The pre-registration is not edited; its section 1 listed the Behaviour figures as
not verified.

1. **Figures now verified.**
   - Groups of three: 24 groups of white Leghorn hens, 4 hours each; 23 formed at least two relationships: DD 17
     (74%), DS 4 (17%), BDID 1, ISDB 1; DD + DS 91% (χ² = 30.4, 3 d.f.).
   - Groups of four: 14 groups, 6 hours on each of two days; day 1, 55 of 56 component triads: DD 33 (60%), DS 15
     (27%), BDID 3, ISDB 4; DD + DS 87% (χ² = 42.4). Day 2 (56 triads): 48.2%, 37.5%, 3.6%, 10.7%; 86%.
   - 13 of 14 groups of four linear and one guaranteed linear.
   - Real triads 54 of 54 transitive; constructed triads (hens that met only as pairs in different groups) 54 of
     63 (86%); chance 75%.
   - Criterion for a relationship: three aggressive acts and no counter-attack for 30 minutes; 2 of 52 reversed.
   - Random reference: 0.25 per sequence, as assumed here.
   - The withdrawn planning figure of "about 90%" was correct.
2. **Post hoc re-count.** `code/chase_seq_posthoc.py` (sha256 76002a2d…0d23c2f9) → `results/chase_seq_posthoc.json`.
   No model calls. Sequence shares under the registered first-contest rule and two alternatives fixed in the script
   before its first run: "decided" (first contest with one LEAD and one DEFER) and "three" (three consecutive wins).
   The script was run once; all three rules are reported for every cell and arm.
   - Gemma-31B `norm100`, groups of four, DD + DS (first / decided / three): OWN 0.53 / 0.81 / 0.65; LOG 0.70 /
     0.91 / 0.81; DYAD 0.55 / 0.47 / 0.58. N = 8: OWN 0.51 / 0.81 / 0.68; LOG 0.70 / 0.81 / 0.77.
   - Gemma-31B `comp`: OWN 0.52 / 0.53 / 0.50; LOG 0.54 / 0.73 / 0.66. N = 8: OWN 0.48 / 0.55 / 0.49; LOG 0.54 /
     0.81 / 0.71.
   - Under "decided", LOG is mostly DD (0.50–0.55) and OWN mostly DS (0.34–0.67).
   - GLM-5.2 `comp` 0.62 / 0.65; mixed group 0.72 / 0.71; Gemma-26B `norm0` 0.50 / 0.54 (OWN / LOG, "decided").
3. **Consequence.** The statement in the E3 report and in the first final report that under `comp` linearity rises
   without a change in sequences holds only for the registered rule. Addendum added to `CHASE_E3_REPORT.md`;
   `CHASE_FINAL_REPORT.md` revised (new section 4; summary and sections 3, 6, 7, 8 edited). The registered rule
   remains the registered measure; the re-count is labelled post hoc everywhere.


### Amendment C8 (follow-up experiments E6–E9; written 2026-11-16 23:04, before any call of any of them)
The owner asked for experiments on the four open questions of section 8 of the final report and approved all four
("All four"). Sizes below were derived from existing data only: cost per call from the spend ledger, rough
forecasts from kernels fitted on E1 rows, and power by resampling existing groups. No model call was made for the
design. The pre-registration is not edited. Spend at this point: 34.67 of the cap of 78.06.

**Common to all four.** Engine unchanged (`chase_pop.run`, `LLMDecider`); new runner `code/chase_f_run.py`
(sha256 6449002015f958e9ddbceb6a76eaee5cad3e3d6aa3456fb458988b5c1708208a) only chooses cells, arms, seeds and the output file. One paced process at a time, 48 requests per
minute. Groups of four run 24 rounds, groups of eight 40. Seeds are the same in every arm of a cell. No outcome of a
cell is analysed until all of its groups are present; only call counts, invalid counts and errors are looked at
while a stage runs. Scripts not yet written are hashed in this log before their first call or, for evaluations,
before any outcome is read. Estimated total: about 126,000 calls, 44 hours, 27 credits.

**Order.** E9 → E6 part B → E7 calibration and forecast freeze → E8a → E8b → E7 populations. E6 part A needs no
calls and is done while E9 runs. The E7 populations wait for the owner to see the frozen forecast.

**E9 — registered sequence test (question 4).**
1. Cells: Gemma-4-31B × `comp`, `norm100`. Groups of four: OWN 40, LOG 40, DYAD 20. Fresh seeds 14000–14039 (DYAD
   14000–14019). File `results/chase_e9_pop.jsonl`. 200 groups, about 19,600 calls.
2. Formation rule: a relationship forms at the pair's first contest in which one agent leads and the other defers
   (rule "decided" of `code/chase_seq_posthoc.py`, sha256 76002a2d6fdfcbdbda65d1db6acbbad7e0380d79d38798e9f6ad1b9e0d23c2f9, imported unchanged).
3. Evaluation `code/chase_e9_eval.py` (sha256 848276f016248ddff65882c35b5d36ff1871f93ceb308f2a80b1af464799db79), tested only on E3 groups relabelled to the new seed range.
   Unit: the group; a group's share is over its triads with two formed relationships.
4. Primary predictions, two-sided Mann–Whitney on per-group shares, Holm over the four, supported if adjusted
   p < 0.05 with the predicted direction: P1 `comp` DD + DS, LOG > OWN; P2 `comp` DD, LOG > OWN; P3 `norm100` DD,
   LOG > OWN; P4 `norm100` DS, OWN > LOG. Power at 40 groups per arm, from resampling the E3 groups: 0.90 (P1),
   about 1 (P2–P4).
5. P5, criterion without a test: pooled DD + DS in the DYAD arm lies in [0.40, 0.60] in each framing.
6. Estimate only: `norm100` DD + DS, LOG − OWN (power 0.33).
7. Secondary: S1 triads formed by DD or DS end transitive more often than triads formed by BDID or ISDB; S2 DD + DS
   as a share of all triads; S3 linear share, LOG against OWN, Fisher exact, unadjusted (a fresh replication of H3;
   pooled with E3 only descriptively); S4 the OWN, LOG and DYAD linear shares against the frozen E2 intervals
   printed in section 2 of `CHASE_E3_REPORT.md`.
8. Queue `scratch/chase_e9_queue.sh` (sha256 e812b9d42013184d0654cf07fbf5cdd96437db75e707ecf78fa8a10eb70b547b).

**E6 — why the own-record forecasts are too low (question 2).**
Part A (no calls). Decisions of the OWN-arm groups already in `results/chase_e3_pop.jsonl` (first 640 lines), with
extra features rebuilt from the stored histories. Candidate decision models, all logistic on signed-log features
with the ridge of `chase_fit.py`:
- A0: direct net d, own net o (the frozen form K2);
- A1: A0 + a second slope for own net below −3;
- A2: A0 + number of own lines shown;
- A3: A0 + the most recent own outcome and the length of the current winning or losing run;
- A4: A0 + whether the agent's ID is lower than the partner's, and the agent's ID;
- A5: all of these; and FLEX, the form-free smoother of `chase_fit.py` on d and o.
Criterion: leave-one-seed-out log-loss per decision, groups of four and eight together, in the two cells whose
forecasts missed (Gemma-31B `norm100`, Gemma-26B `norm0`). Rule: take the forms whose log-loss averaged over the
two cells is within 0.002 of the lowest; among them the one with the fewest added features is selected, provided it
improves on A0 by at least 0.005 in both cells; otherwise A0 is kept. Reported as well: the same table for
Gemma-31B `comp` and GLM-5.2 `comp`, and whether groups simulated from the selected form, fitted leaving one seed
out, reach the observed linear share and t_tri.
Part B (about 4,800 calls). Calibration in rounds for the OWN arm of the same two cells, at both sizes, with the
form selected in part A. Each round: fit on all usable rows so far, simulate groups, and query 300 records per size,
half drawn uniformly from the decisions the simulated groups visit and half in equal numbers from bins of own net
(≤ −9, −8 to −6, −5 to −3, −2 to −1, 0, 1 to 2, 3 to 5, ≥ 6). Stop when the point forecasts (linear share in groups
of four, t_tri in groups of eight) both move by less than 0.03 between rounds, or after four rounds. Then one
write-once forecast with 200 bootstrap refits at the planned group counts (40 and 6). Rows that lack a feature the
selected form needs are not used in its fits. Pass rule: the 90% interval covers the observed linear share in groups
of four in both cells (0.78 and 0.50). This is a retrospective check, since the outcomes are known. The exact
implementation is hashed here before the first part B call.

**E7 — a second model (question 1).**
1. Model: Qwen3.5-397B × `norm100`, `comp`. Rough expectation from E1 rows alone: LOG − OWN linear share +0.30
   and +0.15 (the same method gave +0.36 and +0.14 for Gemma-31B `norm100` and `comp`, against +0.17 and +0.25
   observed).
2. Calibration: round 1 as E2 (groups of four: DYAD 150, OWN 300, LOG 450, REDACT 150; groups of eight: OWN 300,
   LOG 450). Rounds 2 to 4 as in E6 part B for OWN, LOG and REDACT (150, 225, 75 at four; 150, 225 at eight), LOG
   stratified on own net and partner net in three levels each (≤ −3, −2 to 2, ≥ 3). Same stopping rule, applied to
   OWN and LOG. The forecasting form is the E6 selection; its added features enter every arm that shows the own
   record. Forecast written once to `results/chase_e7_predictions.json`.
3. Arms: groups of four OWN, LOG, DYAD 20, REDACT 20; groups of eight OWN 8, LOG 8.
4. Sizing rule: OWN and LOG groups of four per arm are the smallest of 40, 80, 120 at which the frozen forecast
   gives the Fisher test 80% power. A cell with less than 50% power at 120 is run at 40 per arm as an estimate and
   is not a test.
5. Tests: Fisher exact on linear counts, LOG against OWN, Holm over the cells run as tests; Mann–Whitney on t_tri at
   N = 8, secondary; every arm against its frozen interval; the LOG forecast without bystander features, as in E3.
   Secondary, by the E9 rule: DD share, LOG > OWN.
6. Seeds 24000+ (four), 28000+ (eight). Files `results/chase_e7_statecal.jsonl`, `results/chase_e7_pop.jsonl`.
7. Gate: no E7 population call before the owner has seen the frozen forecast and the sizes it implies.

**E8 — equal turns (question 3).**
- E8a: Gemma-4-26B × `norm0`, groups of four, OWN 140 and LOG 140, fresh seeds 34000–34139, file
  `results/chase_e8_pop.jsonl`. Primary: the reading of H5 fixed in amendment C6 item 6, on the fresh groups only
  (at 140 per arm a true difference of zero excludes +0.17 with probability about 0.8). Co-primary, registered here:
  the per-group share of contests decided by the answers is lower under LOG than OWN (Mann–Whitney; first run
  −0.10 [−0.13, −0.07]). Secondary: fresh and E5 groups pooled (180 per arm); the first 40 fresh groups per arm
  against the frozen E5 forecast.
- E8b: Qwen3.5-397B × `norm0`, groups of four, OWN 80 and LOG 80, seeds 34000–34079. Calibration as E7 for OWN and
  LOG at four only; forecast `results/chase_e8_predictions.json` frozen and stamped into this log by the queue
  script before the first population call, without a gate (as in amendment C6 item 4). Same reading of H5. The share
  of decided contests is reported as an estimate only, since this model's baseline LEAD rate is near one half and
  the direction is not predicted.
- Checked on existing groups and not registered as outcomes: directional consistency, Landau h′ and the Gini
  coefficient of wins show no LOG − OWN difference in the E5 groups.

**2026-11-16 23:05 — E9 launched (queue `scratch/chase_e9_queue.sh`). E6 part A script fixed before its first run:** `code/chase_e6a.py`, sha256 da0375ad4384463824ec6e88fdbb7a2f59e3d8f449d91f314fe54b90b34ca44f. No calls.

**2026-11-16 23:12 — E6 part A result, and the implementation of calibration in rounds (before any E6B, E7 or E8 call).**
Part A, run once (`results/chase_e6a.json`; rebuilt d and o match the stored values in every decision):

| Leave-one-seed-out log-loss | A0 | A1 hinge | A2 lines | A3 recency | A4 IDs | A5 all | FLEX |
|---|---|---|---|---|---|---|---|
| Gemma-31B `norm100` (5,750 decisions) | 0.1638 | 0.1604 | 0.1631 | 0.1622 | 0.1643 | 0.1581 | 0.1644 |
| Gemma-26B `norm0` (5,692) | 0.6102 | 0.6098 | 0.5788 | 0.5993 | 0.6107 | 0.5639 | 0.5909 |
| Gemma-31B `comp` (11,424; not used for selection) | 0.2604 | 0.2602 | 0.2601 | 0.2508 | 0.2580 | 0.2468 | 0.2409 |
| GLM-5.2 `comp` (3,842; not used for selection) | 0.3920 | 0.3920 | 0.3919 | 0.3869 | 0.3921 | 0.3861 | 0.3897 |

- By the rule, **A5 is selected**: it alone is within 0.002 of the lowest average, and it improves on A0 by 0.0057 and
  0.0463 in the two selection cells. Most of the gain under `norm0` comes from the number of own lines shown.
- Adequacy, groups simulated from fits that leave one seed out (90% intervals): Gemma-31B `norm100` linear share
  0.70 [0.60, 0.82] with A0 and 0.71 [0.57, 0.82] with A5, observed 0.78; t_tri at N = 8 0.27 and 0.30, observed
  0.32. Gemma-26B `norm0` linear share 0.31 [0.20, 0.41] with A0 and 0.34 [0.22, 0.47] with A5, observed 0.50,
  outside both; t_tri −0.07 and −0.06, observed −0.04.
- Reading: in Gemma-31B `norm100` the frozen form was adequate once fitted on the records groups actually reach,
  so the earlier miss was a calibration-coverage problem, which part B addresses. In Gemma-26B `norm0` no form
  reaches the observed 0.50, including the richest one fitted on those groups' own decisions. Either something
  outside all these features matters, or 20 linear groups of 40 was a high draw. E8a gives 140 fresh OWN groups in
  this cell; the leave-one-seed-out value to compare with is 0.34 [0.22, 0.47] at 40 groups.
- Implementation for part B, E7 and E8b: `code/chase_cal.py`, sha256 ca828a3bbed0488afdf1e5e6e16052d41409fcd5a0443dd0f8f7add6948743ff, tested end to end only with a stand-in
  rule in place of the model (`CHASE_CAL_FAKE=1`, outputs in a scratch directory, deleted). Fixed in it:
  - "X" = frozen form + the six A5 features for OWN, REDACT and LOG (headline); DYAD keeps the frozen form;
    "K" = frozen form fitted on the same rows (comparison); LOG without bystander features = X without p and tc.
  - Fits use only this stage's state-matched rows at the population's own group size, since earlier rows lack the
    new features. E6 round 1 draws its records from groups driven by the frozen form fitted on all existing rows;
    E7 and E8b round 1 from kernels fitted on E1 rows, uniform sampling, as in E2.
  - Later rounds: groups driven by X and K fitted on the rows so far; 60 simulated groups of four and 12 of eight
    per kernel; half the records uniform, half stratified as registered.
  - Stopping per cell: point forecasts from 2,000 simulated groups of four and 150 of eight with the same
    simulation seeds every round; stop below a largest move of 0.03, or after round 4.
  - Forecast files hold 200 row-bootstrap refits; for E7 they hold OWN and LOG at 40, 80 and 120 groups and apply
    the sizing rule.
- Queue `scratch/chase_f_queue.sh` (sha256 81abbda36e9a8beb60fd4df7e8add5a963ff144c171f67ea34d3558ad5c51b61) starts when E9 exits cleanly: E6 part B → E6 forecast → E7
  calibration → E7 forecast → E8a populations → E8b calibration, forecast and populations. It stamps each forecast
  into this log and does not start the E7 populations. Evaluation scripts for E7 and E8 are still to be written and
  will be hashed here before any of their outcomes is read.

**2026-11-16 23:12 — second queue started** (waits on the E9 queue process, then runs as described above). E9 at that moment: 0 groups written (first groups still in progress), no errors in its log.

**2026-11-17 07:05 — E6 forecast frozen (written by the queue script).**
`results/chase_e6_predictions.json`, stamp 48f8e4abcff4ed617a6194eb742d16bbc9462c2bbfd412f7649ab4d6a59eba90, file sha256 c328a82d09f270cd735a29d74fa91d76d354f2bc750a069196a95a8a17efc54b.

**2026-11-17 07:45 — E9 complete and evaluated.** Queue 16 Nov 23:05 – 17 Nov 05:59, exit 0; 200 of 200 groups,
19,862 calls, 0.77 credits, 10 invalid answers. `results/chase_e9_pop.jsonl` sha256 f451b652237f981ee74e636b1fe31dd52dc86d3c84f6ecda77479d6bcc4a309d.
Script hashes checked against C8 before the evaluation was run (chase_e9_eval.py 848276f0…, chase_seq_posthoc.py
76002a2d…, both match). `results/chase_e9_eval.json` sha256 9e3f3dad01fa74e67303d0995a995b2d6aa3830f0b06757c680fbf911a1e5a6f.
- Primary: P1 `comp` DD + DS +0.22 (Holm 0.003), P2 `comp` DD +0.24 (Holm 0.003), P3 `norm100` DD +0.41
  (Holm 2e-7), P4 `norm100` DS −0.32 (Holm 2e-5): all four supported.
- P5: DYAD DD + DS `comp` 0.55 (met), `norm100` 0.39 [0.29, 0.48] (missed by 0.01).
- S1 transitive after DD/DS vs BDID/ISDB: `comp` +0.07 [−0.00, +0.16], `norm100` +0.14 [+0.05, +0.21].
- S3 linear LOG − OWN: `norm100` +0.25, Fisher p = 0.006; `comp` +0.10, p = 0.50.
- S4: all six arms inside or on the edge of the frozen E2 intervals (`norm100` OWN 0.70 on the upper edge 0.70).
Report: `refine-logs/CHASE_E9_REPORT.md`.

**2026-11-17 07:45 — E6 part B result (retrospective check, outcomes known).** Calibration rounds: Gemma-31B
`norm100` three rounds (stopped at a move of 0.007), Gemma-26B `norm0` two rounds (0.016); 3,000 rows. Frozen
forecast (stamp 48f8e4ab…, entry above), headline form X:
- Gemma-31B `norm100` OWN N4: 0.73 [0.60, 0.85] against observed 0.78 → covered (K: 0.75 [0.62, 0.85]).
- Gemma-26B `norm0` OWN N4: 0.30 [0.17, 0.45] against observed 0.50 → not covered (K: 0.27 [0.15, 0.40]).
- N8, reported: Gemma-31B `norm100` 0.34 [0.14, 0.54] against 0.32, covered; Gemma-26B `norm0` −0.08 [−0.20, 0.05]
  against −0.04, covered.
Pass rule (both N4 cells covered): **not met**. Calibration in rounds closes the Gemma-31B `norm100` gap (the
frozen E2 forecast was 0.56 [0.42, 0.70]); the Gemma-26B `norm0` gap remains. E8a's 140 fresh OWN groups will show
whether 0.50 was a high draw. Not registered, descriptive: the fresh E9 `norm100` OWN share, 0.70, lies inside
the part B forecast, which was computed from calibration rows only.

**2026-11-17 09:14 — E7 forecast frozen (written by the queue script).**
`results/chase_e7_predictions.json`, stamp d5614939f26938b768b274e20295f2f810eb005165e3087e55e25c2a58d9c923, file sha256 019836c2188c4f63e574a8e389be74d869964f572cf64663d721159f39aa7474.

**2026-11-17 19:03 — E8 forecast frozen (written by the queue script).**
`results/chase_e8_predictions.json`, stamp b7ca86f4da62c1752ec93afe282b04217c1e3c64e780a41ada647bd6c127438b, file sha256 38dd67fa464cba11fa617d76a4c2fc7c69e50b5edef6919b4c2ef255e5835c89.

**2026-11-18 00:40 — second queue complete; E8 evaluation fixed before any E8 outcome is read.** Queue exit 0 at
00:20: E6B, E7 calibration (three rounds, forecast frozen 09:13 on 17 Nov), E8a populations (280 groups,
09:14–18:39), E8 calibration (two rounds, forecast frozen 19:03), E8b populations (160 groups, 19:03–00:20).
`results/chase_e8_pop.jsonl`: 440 lines, sha256 2b9821f6e5d294aee23749b6388e73e2f9ef7d980d119aeed974e56836416a19. Only the queue log's
group counts, invalid counts and the calibration lines were looked at. Spend 43.22 of 78.06.
Evaluation `code/chase_e8_eval.py`, sha256 31b61cf27904832b4fb95ccc86eb9afe3fe179d076ed910364c7615730096625, implements the E8 entries of C8 and the H5 reading of C6 item 6
through the unchanged functions of `chase_e3_eval.py`; tested only on E5 groups relabelled to the E8 models and seeds.
Unregistered descriptive additions: LEAD rate and DD + DS (E9 rule) per arm; the first 40 fresh OWN groups against
the E6 part B forecast and the E6 part A interval.

**2026-11-18 00:55 — E8 evaluated** (`results/chase_e8_eval.json`, sha256 91327a2728ca8ae882c467d667dddcd2af61191f11f7c83f91b40bb44e4d8703; report `refine-logs/CHASE_E8_REPORT.md`).
- E8a Gemma-26B `norm0`, 140 fresh groups per arm: linear OWN 0.39, LOG 0.41, +0.03 [−0.09, +0.14], Fisher
  p = 0.71 → H5 **vanishes** (C6 item 6). Co-primary decided share 0.39 → 0.29, p = 2e-21, **supported**. Pooled
  with E5 (180 per arm): +0.02 [−0.08, +0.12], vanishes. First 40 fresh groups against the E5 forecast: OWN 0.48
  outside [0.13, 0.35]; LOG 0.43 inside; contrast −0.05 on the edge of [−0.05, +0.33].
- E8b Qwen3.5-397B `norm0`, 80 per arm: OWN 0.31, LOG 0.30, −0.01 [−0.15, +0.14], p = 1.0 → H5 **vanishes**. All
  frozen E8 intervals cover the observations. Decided share 0.27 → 0.21 (estimate). LEAD rate 0.20, not near one
  half as C8 expected.
- Limit, recorded with the result: neither model has a population run showing a LOG − OWN effect under another
  framing; the frozen E7 forecast for Qwen3.5-397B is near zero under `norm100` and `comp` too.

**2026-11-18 00:55 — E7 gate.** Frozen forecast (stamp d5614939…, 17 Nov 09:13): LOG − OWN linear share −0.01
[−0.20, +0.20] (`norm100`) and +0.02 [−0.18, +0.23] (`comp`) at 40 per arm; Fisher power at 120 per arm 0.14 and
0.13. By the C8 sizing rule neither cell is a test; both would run at 40 per arm as estimates. Shown to the owner
now; no E7 population call before their answer.

**2026-11-18 12:50 — E7 approved and queued.** The owner saw the frozen forecast and the sizes (entry of 00:55) and
answered "Yes, run E7". Run as registered: Qwen3.5-397B × `norm100`, `comp`; groups of four OWN 40, LOG 40, DYAD 20,
REDACT 20 (seeds 24000+); groups of eight OWN 8, LOG 8 (seeds 28000+); both cells are estimates, not tests, by the
sizing rule. 272 groups, about 33,000 calls, 10 credits, 11.5 hours. Queue `scratch/chase_e7_queue.sh` (sha256
52192e241e7b70ffb8e09efdb3ecedd6cb3aadaa20a52c90125c244a82a9c17f) waits for the Rosen study's queue on the same
account (PID 94307) so that only one paced process runs. The E7 evaluation script will be written and hashed here
before any E7 outcome is read. Note: since 18 Nov 00:21 the Rosen study's calls are also written to
`results/adaption_spend.jsonl` in this checkout, so the ledger total includes them (44.22 after its R0 stage).
