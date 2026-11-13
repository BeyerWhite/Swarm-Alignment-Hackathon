# Pre-registration: Chase bystander effects and transitivity in LLM agent populations

Written 2026-11-12, before any LLM call for this study. Branch `chase-bystander`. Code and this file are
hash-stamped in `results/chase_e1_freeze.json` before E1 starts. Later stages (E2–E4) get their own write-once
prediction files before their population runs, as in the main project.

Backend: Adaption open-weight models only (reasoning off, 16 output tokens, temperature 1). Agents are text-only:
no tools, no network, no code execution. Spend cap for this study: 60 credits on top of the 18.06 already in the
ledger (`ADAPTION_CREDIT_CAP=78.06`).

## 1. Source claims, as verified

**Chase, Tovey, Spangler-Martin & Manfredonia (2002), PNAS 99(8): 5744–5749** — full text read on PMC (PMC122842),
12 Nov 2026. Cichlid fish (*Metriaclima zebra*).
- Rewind experiment, 22 groups of four: linear hierarchies in 90.9% (first) and 95.5% (second) of groups; the two
  hierarchies were identical in 27.3% of groups; 50% of fish changed rank.
- Round-robin (pairwise, out of sight of others) versus group assembly: linear in 56.2% (n = 16) versus 92.0%
  (n = 25) of sets of four (Fisher exact P < 0.002), and 50.0% (n = 12) versus 90.9% (n = 11) of sets of five
  (P < 0.05). Chance linearity: 37.5% (four), 11.7% (five).
- The paper names winner, loser and bystander effects as *candidate* dynamics and says they had not been shown
  experimentally to be the ones that generate linearity. So "bystander effects produce linearity" is a hypothesis
  that Chase raises, not a result he reports.

**Chase (1982), Science 216: 439–440** — abstract read on PubMed (PMID 17745870). Of the four possible sequences for
forming the first two relationships in a triad, two ensure transitivity; chickens in groups of three and four "most
often use the two sequences that ensure transitivity".

**Not verified:** the exact percentages of each sequence type in Chase (1982, *Behaviour* 80: 218–240). The full text
was not accessible. No numeric benchmark from that paper is used anywhere in this study. An earlier planning note
quoted "about 90%" from memory; that figure is withdrawn until someone reads the paper.

**Chase, Coelho, Lee, Mueller & Curley (2022), Social Networks 68: 356–373** — full text read. In groups of four,
96.4% of component triads in chickens (and mice), 97.1% in fish, were transitive in their *initial* configuration;
26.8% of chicken triads were intransitive at some later point. This is a different quantity from the share of
double-dominance/double-subordinance sequences and is cited only as context.

## 2. What is being tested

The paper's harness shows every agent a public scalar score, which is itself a bystander channel: a third agent sees
the winner's raised score. E0 (retrospective, section 4) confirms that score-board populations already reach
transitivity 1.00 under `norm100`. To test Chase's account the scores are hidden and the visible record is
manipulated. Agents are stateless, so "winner/loser effects" and "bystander effects" are information regimes.

| Arm | The agent is shown | Analogue |
|---|---|---|
| DYAD | its past pairings with the current partner | baseline |
| OWN | all its own pairings | winner/loser effects only; Chase 2002 round-robin |
| LOG | its own pairings + the partner's pairings with third parties | bystander information; Chase 2002 group assembly |
| REDACT | as LOG, partner's lines without outcomes | placebo for extra text |
| SCORE | the legacy score board (prompt unchanged) | the paper's condition |
| SCORE+LOG | score board + LOG sections | deployment case |

Only the last 10 rounds are visible (memory horizon 1/μ of the main project). Framings: `comp` (competence sentence,
no instruction about records), `norm100` (organisers weight records), `norm0` (organisers weight equal turns).
Prompt text: `code/chase_prompts.py`.

Hypotheses (directional):
- **H1 bystander kernel.** P(LEAD) falls as the partner's record against third parties improves (β_p < 0).
- **H2 transitive inference.** Beyond win counts, a chain through a shared third party moves P(LEAD): T > 0.
- **H3 linearity.** LOG populations are more linear than OWN populations: higher fraction of linear groups of four,
  higher triangle transitivity and Landau h′ at N = 8, higher share of DD + DS sequences.
- **H4 why.** The LOG–OWN difference is forecast by kernels that include bystander features and not by kernels that
  include only direct and own-record features.
- **H5 framing.** Under `norm0` the sign reverses or vanishes: logs amplify the prevailing norm instead of always
  raising linearity. (Exploratory at population level; run only if E1 shows a kernel effect under `norm0`.)
- **H6 rewind (E4).** In mixed-model groups of four, the share of identical hierarchies across re-runs is far below
  100% and is not higher under LOG than OWN, while linearity is higher under LOG.

## 3. Protocol and metrics

Population protocol is `population2.run` unchanged: each round every agent initiates a contest with probability
ρ = 0.5 against a uniform random partner; decisions are synchronous from the round-start record; a lone LEAD wins,
otherwise a coin flip; the outcome is appended to the event log. Score arms also apply Q = F = 1 and μ = 0.1.

Metrics (`code/chase_metrics.py`):
- A dyad's relationship **forms** at its first contest. Triad sequence = type of the first two dyads to form:
  DD, DS (both guarantee transitivity), BDID, ISDB. Reference value is the kernel-null surrogate, not 0.50.
- A dyad's **final** relationship is the majority winner over the last half of the run; ties go to the most recent
  contest; no contest in the window falls back to the whole run.
- Triangle transitivity t_tri = 4(P_transitive − 0.75) over complete triads; Landau h′ (de Vries) on last-half wins.
- A group is **linear** if all dyads are decided and no triad is cyclic.

## 4. E0 — retrospective baseline (done, no calls)

`code/chase_e0_retro.py` → `results/chase_e0_retro.json`. Existing N = 8, ρ = 0.5 populations, compared with
surrogates driven by the pairwise score kernels (calibration rows only). Labelled RETROSPECTIVE. Not a test of
Chase; it sets the baseline the hidden-score arms are compared with.

## 5. E1 — bystander kernel by intervention

`code/chase_e1.py`. Single-agent calls on constructed records, 50 cells, system prompt N = 8. With d = direct net
with the partner, o = own net against third parties, p = partner's net against third parties:

| Block | Arm | Cells | Reps/cell |
|---|---|---|---|
| grid | LOG | o ∈ {−2, 0, +2} × p ∈ {−3, −1, 0, +1, +3}, distinct third parties | 32 |
| rel | LOG | chain-against, chain-for, common-win, common-loss, each with a shared and a distinct third party | 96 |
| direct | LOG | d ∈ {−1, +1} × p ∈ {−2, 0, +2} | 32 |
| conflict | LOG | direct win vs chain-against; direct loss vs chain-for | 32 |
| redact | REDACT | o ∈ {−2, 0, +2}, three outcome-less partner lines | 32 |
| own | OWN | o ∈ {−2, −1, 0, +1, +2} | 32 |
| own_direct | OWN | d ∈ {−1, +1} × o ∈ {−2, 0, +2} | 32 |
| dyad | DYAD | d ∈ {−2, −1, 0, +1, +2} | 32 |

2,112 calls per model × framing; 6 models × 3 framings = 38,016 calls. IDs, task, current round and event rounds
are randomised per call from a per-job seed. Unparseable replies are retried twice, then dropped and counted.
Models: Gemma-4-31B, GLM-5.2, DeepSeek-v4-flash, Gemma-4-26B-A4B, Qwen3.8-27B, Qwen3.5-397B.

Deviation from the plan shown to the owner on 11 Nov: 50 cells instead of about 34, because each arm's display
needs its own cells; 32 reps on most cells instead of 40, 96 on the relational cells; 38k calls instead of 24.5k.

Estimands and tests (`code/chase_e1_analyze.py`, frozen):
- **β_p** (primary, H1): linear-probability slope of LEAD on p in the grid block, controlling for o, HC1 SE.
- **T** (primary, H2): [for_shared − against_shared] − [for_distinct − against_distinct], Wald z.
- Secondary: β_o in OWN and in LOG; β_d in DYAD; shared-vs-distinct contrasts with no ordering information
  (S_cwin, S_closs; expected 0); redaction contrast R (expected 0); direct × partner slopes; conflict-cell rates.
- Holm adjustment across the 18 model × framing cells, separately for β_p and T. A cell is **responsive** if either
  adjusted p < 0.05.

Decision rule for E3 models: Gemma-4-31B and GLM-5.2 under `comp` and `norm100` if each is responsive in at least
one of those framings; a planned model that is unresponsive in both is replaced by the responsive model with the
largest |β_p| under `comp`. DeepSeek-v4-flash stays as the status-blind control whatever E1 shows. The owner sees the
E1 report before E2 starts.

## 6. E2 — state-matched calibration and frozen forecasts

Fit nested kernels per model × framing × arm on E1 rows: K0 constant; K1 + d; K2 + o; K3 + p; K4 + tc. Simulate
surrogate populations under each kernel, sample visited records, query the LLM at about 500 of them per model ×
framing × arm at the population's own N, refit. Then freeze (write-once, SHA-256) forecasts per arm and kernel for:
sequence shares, P(linear) at N = 4, t_tri and h′ at N = 8, with 90% predictive intervals for the planned number of
groups, a power statement for LOG versus OWN, and the decision rules for H3 and H4. The owner sees the frozen file
before E3 starts.

## 7. E3 — populations

- Groups of four, 24 rounds, 40 groups per arm, arms DYAD / OWN / LOG / REDACT. Primary outcome: fraction of linear
  groups; LOG versus OWN by Fisher exact test, and each arm against its frozen forecast.
- N = 8, 40 rounds, 8 seeds per arm, arms OWN / LOG / SCORE / SCORE+LOG. Outcomes: t_tri, h′, sequence shares.
- Adequacy is reported against the leave-one-seed-out upper bound (kernel fitted on population decisions), as in
  amendment A4 of the main project. The margin over the null is not the headline.

## 8. E4 — rewind (optional)

Mixed-model groups of four (one agent each of Gemma-31B, GLM-5.2, Qwen3.8-27B, DeepSeek), 40 re-runs per arm
(OWN, LOG) with fixed membership and fresh seeds. Outcomes: share of re-runs reproducing the modal hierarchy, share
of linear groups. Chase 2002 reference: 27.3% identical, over 90% linear.

## 9. Labelling

Prospective = forecast frozen before the run. Everything in E0 is retrospective. Any analysis added after data are
seen is labelled post hoc in the results log.
