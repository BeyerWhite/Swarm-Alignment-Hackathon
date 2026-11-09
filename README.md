# Artificial Sociology: Predicting Who Coordinates an LLM Agent Swarm from Two-Agent Tests

**Sara Ansari** (UT Austin) · **Beyer Chase White** (Independent)
Swarm Alignment Hackathon, Apart Research, November 2026. Track 2: The Science of Agent Networks.

When many AI agents share a workspace, one of them can end up directing the rest. We measure how pairs of LLM agents respond to a public status record, freeze forecasts for whole populations, and then run the populations. The two-agent tests predict which populations become hierarchical.

## Where everything is

All project material is in [`swarms-hierarchy/`](swarms-hierarchy/). See its [README](swarms-hierarchy/README.md) for the full file guide.

| What | Path |
|---|---|
| Report (Apart template) and figures | [`swarms-hierarchy/paper/`](swarms-hierarchy/paper/) |
| Harness, calibration, prediction, evaluation and figure code | [`swarms-hierarchy/code/`](swarms-hierarchy/code/) |
| Real-task game (planner/worker roles) | [`swarms-hierarchy/code/taskgame2.py`](swarms-hierarchy/code/taskgame2.py) |
| Pre-registration and amendment log | [`swarms-hierarchy/refine-logs/EXPERIMENT_PLAN.md`](swarms-hierarchy/refine-logs/EXPERIMENT_PLAN.md) |
| Every logged LLM call | [`swarms-hierarchy/results/calls.jsonl.gz`](swarms-hierarchy/results/) |
| Frozen (hash-stamped) predictions, population logs, evaluations | [`swarms-hierarchy/results/`](swarms-hierarchy/results/) |
| Claims and cross-model review history | [`swarms-hierarchy/review-stage/`](swarms-hierarchy/review-stage/) |

All data was generated during the sprint (7–8 November 2026). The agents are text-only: no tools, no network access, no code execution. API keys are not included.
