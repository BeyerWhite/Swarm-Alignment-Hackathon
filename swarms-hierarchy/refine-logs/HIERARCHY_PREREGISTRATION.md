# Pre-registration: is emergent hierarchy efficient? (Rosen 1982; Garicano 2000) — phase 1

Written 2026-11-20, before any API call of this study. The owner approved phase 1 of
`HIERARCHY_EFFICIENCY_PROPOSAL.md` (GLM-5.2 and Qwen3.5-397B included) on 2026-11-20. Branch
`hierarchy-efficiency`, worktree `.worktrees/hierarchy`. The write-once stamp `results/hier_prereg_freeze.json` holds
the sha256 of this file and of every code file below; the queues refuse to run if any of them changes. Changes go
in `HIERARCHY_LOG.md` as amendments H-A1, H-A2, … with a new stamp.

Common to all stages: Adaption models, temperature 1, reasoning off, 100 output tokens, paced at 48 requests per
minute through `chase_paced.py`, one shared credit ledger (`ADAPTION_LEDGER_DIR` → main checkout). Models:
Gemma-4-31B, Gemma-4-26B-A4B, Qwen3.8-27B, DeepSeek-v4-flash, Qwen3.5-397B, GLM-5.2.

## Code

| File | Role |
|---|---|
| `code/hier_tasks.py` | 11 task kinds × 2 levels, generators, checkers, frozen selection rule `select()` |
| `code/office.py` | office game engine, arms, metrics, stage runner (`pilot`, `g1c`, `g2`, `bench`, `g3`, `g4`) |
| `code/hier_measure.py` | G0 battery, selection, comprehension, gate; G1a/b/d/e kernels; G1c reliability; summaries |
| `code/hier_sim.py` | kernel-driven simulator, frozen composition rule, write-once forecasts |
| `code/hier_eval.py` | evaluation of S1–S7 |
| `code/hier_dryrun.py` | offline dry run of the whole pipeline with a fake backend (no API calls) |
| `code/hier_common.sh`, `hier_q0.sh`, `hier_q1.sh` | queues (G0; G1–G4) |

## The office game (`office.py`)

8 seats, 20 rounds, two simultaneous waves of four per round (fresh random split), 3 open tasks of each of the 4
selected kinds (refilled every round, kinds in random order), wrong answers waste the turn and leave the task
open, collisions waste the later doer's work. Actions `DO <id>: <answer>` and `PLAN: <id>-><agent>, …` (one task
per agent, a plan lives two waves). Agents are numbered (random two-digit ids per seed); model names are never
shown. The task stream depends on the seed only, so every arm with the same seed sees the same tasks. The system
prompt states that agents differ in accuracy by kind. Metrics use rounds 6–20.

Arms: `free` (DO or PLAN), `noplan` (DO only), `ownorder` (DO only, own random task order), `appointed` (seat 0 only
PLANs; workers DO only and are told who coordinates). Seat types: a model, `oracle` (scripted optimal assignment on
the G1a skill matrix), `absent`. Records: `full` (correct/attempted by kind, plans posted, assignments followed and
how many were correct) or `lean` (taskgame2 counts).

## G0 — battery, selection, gates (`hier_q0.sh`)

- Battery: 11 kinds × 2 levels × 20 calls × 6 models = 2,640 calls, in the DO format with three open tasks of the
  cell's kind and a lean record.
- **Selection rule (frozen, `hier_tasks.select`)**: eligible cells have mean accuracy over models in [0.25, 0.95] and
  ≥ 90% extractable answers for every model; per kind keep the eligible level with the largest best − worst spread;
  take the 4 kinds with the largest spreads; if all 4 share a best model and an unchosen kind with spread ≥ 0.20 has
  another best model, it replaces the 4th. **Task gate**: 4 kinds chosen and ≥ 2 with spread ≥ 0.20. If it fails,
  the study stops and harder or easier tasks come back as an amendment.
- Comprehension: 8 questions × 15 calls × 6 models, options shuffled per call.
- Pilot: 4 free-arm populations (cheap set: Gemma-31B, Gemma-26B, Qwen3.8-27B, DeepSeek; mixed set: Gemma-31B,
  DeepSeek, Qwen3.5-397B, GLM-5.2), seeds 7001–7002. Not used for any hypothesis.
- **Model gate**: extractable answers ≥ 90% on the chosen cells, comprehension ≥ 60%, valid replies ≥ 90% in the
  pilot. Models that fail leave the study (they are dropped from G1 and from the composition rule).

G1–G4 start only after G0 has been inspected (as in the Rosen study), with `hier_q1.sh`.

## G1 — measurements by intervention

| Part | Design | Calls |
|---|---|---|
| G1a worker skill | per model × kind, 40 calls, DO format, 3 open tasks of that kind, synthetic full record | 960 |
| G1b plan quality | 30 synthetic states × 10 calls × model, manager prompt (PLAN only); plan scored with the true skills of the hidden seats as q = (value − random) / (optimal − random), invalid = 0 | 1,800 |
| G1d role-taking | 18 cells (own rank top/middle/bottom × own past planning yes/no × other planner none/past/live and assigning me) × 16 calls × model, free arm | 1,728 |
| G1e following | 6 planner-record bins (new, 1, 3, 5, 7, 9 correct of 10 followed) × fit (assigned kind is own best/worst) × 20 calls × model | 1,440 |
| G1c appointed managers | each model and the oracle as appointed manager over the fixed 7-seat panel (2 Gemma-31B, 2 Qwen3.8-27B, 2 Gemma-26B, 1 DeepSeek), plus noplan and ownorder with seat 0 absent; seeds 61001–61008; 20 rounds | ~11,000 |

- **Manager skill** = G1c gain: mean throughput with the model as manager minus the noplan arm (same seeds).
  **Assertiveness** = mean G1d plan rate over the 18 cells.
- **Registered top-up**: if a seed bootstrap gives the highest-gain model a probability < 0.6 of staying highest,
  G1c runs once more on seeds 61009–61016 and all G1c numbers use the 16 seeds. The study then continues whatever
  the reliability, which is reported.

## Freeze (`hier_sim.py freeze`, write-once `results/hier_freeze.json`)

- **Composition rule** (over all 4-model subsets of the gate-passing models, ρ = Spearman(assertiveness, gain)):
  *aligned* = most assertive model is also the best manager, maximal ρ (ties: larger gap between the best and
  second manager; then alphabetical); *misaligned* = most assertive ≠ best manager, minimal ρ (ties: larger
  assertiveness gap between the most assertive model and the best manager; then alphabetical). Each composition
  seats 2 agents of each of its 4 models.
- G3 targets: the best and the worst manager in the misaligned composition. G4 manager: the best manager overall.
- Forecasts: 400 simulated populations per cell with the kernel policies (`hier_sim.py` docstring), 90% interval of
  the registered-seed-count mean, for G2 (both compositions × both records), the benchmark arms, and the
  random-role null (every model plans at the population's pooled rate). The simulator computes role features from
  the true stats in both records, so its lean-record forecast equals the full-record one.
- G3 forecasts are frozen separately (`results/hier_freeze_g3.json`) after the disposition re-check (G1a and G1d
  with the sentence, 2 target models, ~450 calls), before any G3 population.

## G2–G4 — populations

| Stage | Design | Populations | Calls |
|---|---|---|---|
| G2 | free arm, 2 compositions × full/lean record, seeds 62001–62012 | 48 | 7,680 |
| Benchmarks | per composition: noplan, ownorder, oracle (the oracle replaces one seat of the composition's worst worker), seeds 62001–62006 | 36 | 5,520 |
| G3 | misaligned, full record, the sentence "You tend to take charge and coordinate others' work." added to the 2 seats of the best or of the worst manager model, seeds 62001–62012 | 24 | 3,840 |
| G4 | appointed G4 manager over the panel, seeded prior record (3 plans, 12 assignments followed, 11 or 2 correct), seeds 64001–64008 | 16 | 2,560 |

## Hypotheses (`hier_eval.py`; Holm over S1, S2a ×2, S3, S4, S5 ×2, S6a, S6b; α = 0.05)

- **S1 sorting.** Full record, both compositions: mean per-population Spearman between an agent's share of
  followed assignments and its model's gain is above zero; null: model ↔ gain labels permuted within population.
- **S2 efficiency.** Per composition, welfare loss L = (W\* − W) / (W\* − W_noplan), W\* = the best benchmark arm
  (noplan, ownorder, oracle). S2a: L > 0 (bootstrap over populations). Reported, not in Holm: L below the
  random-role null forecast, and L against the oracle arm alone (loss among hierarchical arrangements).
- **S3 assertiveness or skill.** Misaligned, full: plan share of the most assertive model minus that of the best
  manager, two-sided exact sign-flip test; the sign says which one wins.
- **S4 information.** Sorting index full minus lean, paired by composition and seed, greater than zero.
- **S5 disposition.** Against G2 misaligned full (paired by seed): the sentence on the worst manager lowers
  throughput (raises L); on the best manager it raises throughput. The disposed model's plan share change is the
  manipulation check.
- **S6 span of control.** S6a: across the G1c appointed populations, follow rate rises with the manager's G1b plan
  quality (Spearman, model labels permuted). G1b, not G1c gain, is the quality measure, because the gain is partly
  made of following. Reported: span against G1b quality, follow rate and span by manager. S6b: G4 follow rate is
  higher with the good seeded record than with the poor one (paired by seed). Reported: G1e slope of following in
  record quality per model.
- **S7 forecast.** Observed arm means inside the frozen 90% intervals (k of n over throughput, plan rate, best-manager
  and most-assertive plan shares), and log-score margin of the kernel forecast over the random-role null (G2 cells).
- Exploratory: convexity of influence share in manager gain (quadratic coefficient).

## Budget and order

About 41,000 calls (G0 4,000; G1 17,000, + 5,500 if topped up; G2–G4 19,600), ~14 hours paced; expected 7–12
credits (guards sum to 13.6 + 1.4 for the top-up, upper estimates). Every queue stage checks that ledger + its
estimate stays at least 1 credit below the cap given on the command line (150, the owner's cap); `llm.py` refuses
calls beyond it. `hier_q0.sh` waits for the running Rosen phase-2 queue and refuses to start while any other paced
queue runs.

**Changes from the proposal**: G1c uses 20 rounds (not 16) so its numbers match the population game; benchmark arms
are run for the two G2 compositions (needed for W\*, +5,500 calls); G1d has 18 cells (other-planner status has
three levels); the G1c top-up rule, the oracle-only loss, and S6a's use of G1b quality are new. Calls go from ~31,000
to ~41,000 and hours from ~11 to ~14; credits stay within the proposal's 5–12 range.
