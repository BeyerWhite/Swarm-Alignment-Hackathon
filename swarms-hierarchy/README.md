# Who Leads the Swarm? Pairwise response kernels and hierarchy in LLM agent populations

Swarm Alignment Hackathon (Apart Research, Nov 2026), Track 2: "The Science of Agent Networks".

We test whether classical models of hierarchy formation among identical agents (Bonabeau / Lacasa & Luque 2005;
Ben-Naim & Redner 2005) explain who ends up coordinating in populations of LLM agents. The method:
1. measure each model's pairwise response to a public score record;
2. freeze population predictions;
3. run LLM-in-the-loop populations and score them against those predictions.

**Report:** `paper/` (`Swarm_Hierarchy_Apart_Report.docx` from the official template; `paper.md` is the source).

## Prior work / pre-existing code
**All project code in this folder was written during the sprint (7–8 November 2026). No pre-existing project code was
used.** We used pre-existing open-source research-orchestration tooling (ARIS Claude Code skills) for literature search,
the cross-model review loop and run tracking. That tooling is not included here.

## Layout
| Path | Contents |
|---|---|
| `paper/` | Report source (`paper.md`), docx built from the Apart template, figures (`paper/figures/`, produced by `paper/figs.py`) |
| `code/` | Harness and analysis: `prompts_v2.py`, `llm.py`, `calibrate.py`, `statecal.py`, `population2.py`, `predict.py`, `evaluate.py`, `reeval_followups.py`, extensions (`x1.py`, `hysteresis.py`, `x2.py`, `bnr_exp.py`, `board.py`), simulations (`fss.py`, `bnr_x2.py`) |
| `results/` | Raw data and analysis outputs (see below) |
| `refine-logs/` | Pre-registration, experiment plan with amendments A1–A5, lab log |
| `review-stage/` | Final claims (`CLAIMS.md`) and full cross-model review history |
| `idea-stage/` | Literature survey and idea selection |

## Data (`results/`)
| File | Contents |
|---|---|
| `calls.jsonl.gz` | Every LLM call: model, system prompt, prompt, response |
| `calib_v2*.jsonl`, `statecal_*.jsonl` | Pairwise calibration |
| `predictions_v2*.json`, `x1_predictions.json`, `hyst_predictions.json`, `x2_predictions.json` | Predictions frozen *before* the matching runs |
| `population_v2*.jsonl` | Main LLM populations |
| `x1_pop.jsonl`, `hyst_pop.jsonl`, `x2_pop.jsonl`, `bnr_pop.jsonl`, `board_pilot.jsonl` | Extension runs |
| `eval_v2.json`, `coverage_common_design.csv`, `reeval_followups.json` | Authoritative evaluations |
| `eta_fd_beta.csv`, `competence_contrast.csv`, `probe_table.csv` | Kernel summaries |

Agents are text-only: no tools, no network, no code execution. API keys are not included.
