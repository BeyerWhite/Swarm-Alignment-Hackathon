# Emergent hierarchy in LLM agent populations

Track 2, "The Science of Agent Networks". Can classical models of hierarchy formation among identical agents explain who ends up coordinating in populations of LLM agents? The two models are Bonabeau / Lacasa–Luque 2005 and Ben-Naim–Redner 2005. The work is motivated by the METR (Aug 2026) Hugging Face incident.

**Snapshot taken 2026-11-07 17:44 PST.**

## Layout
| Path | Contents |
|---|---|
| `RESEARCH_BRIEF.md` | Question, source models, mapping to LLM agents |
| `refine-logs/` | Proposal, experiment plan (with amendments), **living results** (`EXPERIMENT_RESULTS.md`), tracker |
| `idea-stage/` | Idea report and pilot results |
| `code/` | Harness: prompts, LLM backends, calibration, population engine, frozen predictions, evaluation, figures |
| `results/` | Raw and summary data (see below) |
| `figures/` | Figures (PNG and PDF) |

## Data (`results/`)
| File | Contents |
|---|---|
| `calls.jsonl.gz` | Every LLM call: model, system prompt, prompt, response, latency |
| `calib_v2*.jsonl`, `statecal_*.jsonl` | Pairwise calibration: P(LEAD \| own score, partner score) by framing |
| `population_v2*.jsonl` | LLM-in-the-loop populations (N=8, 40 rounds) |
| `predictions_v2*.json` | Predictions frozen and hash-stamped *before* the matching population runs |
| `eta_table.csv`, `probe_table.csv`, `eval_v2.json` | Fitted status sensitivity η, inequality probe, prediction-vs-observation scores |

Agents are text-only (no tools, no network, no code execution). Models: Claude Haiku and Sonnet, GPT-6-luna, and open-weight models via the Adaption API. API keys are not included.
