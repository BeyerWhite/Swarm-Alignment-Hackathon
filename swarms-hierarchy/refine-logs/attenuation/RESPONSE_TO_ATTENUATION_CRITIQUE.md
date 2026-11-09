# Response to "Post-hoc analysis: the systematic under-prediction is mostly kernel attenuation"

**Project:** `swarms-hierarchy/` (canonical copy `/Users/saraansari/Claude/Outputs/Swarms`)
**Critique:** `refine-logs/attenuation/critique.md` (copy of the Downloads file)
**Date:** 2026-11-08, about 19:05 PST
**Status:** sections 1–4 are **post hoc**: not pre-registered and done after all population runs. No frozen prediction file was edited. Section 5 (pilot D) is **prospective** and finished at 20:21: it passed in one of two conditions.

---

## In plain language

The critique said our biggest failure, real groups ending up more unequal than predicted, was mostly caused by how we turned two-agent data into predictions, not by anything groups do. We measured each model's behaviour with only about 4 calls per situation. Our method then nudged every estimate toward 50/50, which made simulated agents look more indecisive than real ones. Over 40 rounds that small bias added up to a large underestimate.

We ran the critique's tests (all except its per-agent ID check), and they support it:
- removing the nudge removes most of the one-sided error on our real data;
- a simulation where we know the truth shows the nudge alone causes the error, and that it shrinks as calls per cell increase;
- unnudged predictions also match the 700-agent run and most of the threshold sweep.

A new frozen test on fresh populations gave a split verdict. The corrected method was better than the old one in both conditions. It hit the target for Qwen, but still under-predicted inequality for Gemma-26B. So the nudge explains much of the miss, but not all of it. Some error remains, mainly where groups wander outside the range we measured. The paper now says the pre-registered test failed, gives this diagnosis as post hoc, and recommends a fix: don't nudge sparse estimates toward 50/50, and spend calls on the situations groups actually reach.

---

## 1. The claim

**Accepted, with qualifications from our reviewer (GPT-6.1-sol, reasoning effort "ultra").**

- **The calibration is sparse and saturated.** The grid has a median of **4 calls per cell** (mean 5.4; 2,769 cells), and **63% of cells** are observed at exactly 0 or 1.
- **The smoothing is a lapse, but only relative to the empirical kernel.** For $\hat p = k/n$, Jeffreys smoothing gives exactly $\hat p_J = \lambda_n + (1-2\lambda_n)\hat p$ with $\lambda_n = 1/(2(n+1))$. Relative to the empirical kernel this is a lapse. Observed saturation does not prove true saturation.
- **Beta(k+½, n−k+½) draws add no further mean attenuation.** Their expectation equals the Jeffreys point estimate.
- **The direction of the macro bias depends on the kernel.** Weakening deference lowers dispersion; weakening anti-deference raises it (see B2 and the lapse sweep). This matches the two-sided seeded-leader errors.

## 2. Evidence: checks rerun properly

### Task A: re-evaluation through the paper's own pipeline
Code: `code/posthoc_attenuation.py` → `results/posthoc_attenuation.{json,csv}`.
- Same B × S as the frozen predictions (25 × 8).
- Same seed-mean predictive distributions and KDE bandwidth rule.

| Estimator (27 status-sensitive cells) | obs > pred (sd) | mean residual (sd) | mean z (sd) | coverage (sd and q) | LPD margin vs null |
|---|---|---|---|---|---|
| Frozen flexible (Jeffreys) | 0.82 | +0.56 | +1.48 | 32/54 | 104–224 |
| `flex_mle` (Beta(k+0.05, n−k+0.05)) | 0.63 | +0.27 | +0.80 | 43/54 | 105–250 |
| **`flex_iso`** (monotone, unshrunk) | **0.48** | **+0.07** | **+0.34** | 42/54 | 105–248 |
| `nw_noshrink` (α = 0) | 0.44 | +0.04 | +0.11 | 46/54 | 107–253 |

**The critique's acceptance criterion passes:** the share of cells with obs > pred is about 0.5, mean z is near 0, and the margin over the null is kept. Our reviewer flagged two caveats:
- Aggregates hide per-model errors. Isotonic mean z is still about +1.8 for Haiku and Gemma-26B, and −1.0 for GLM.
- Coverage (78–85%) is still below the nominal 90%.

**Matched decomposition** on identical data, point estimates, 24 seeds per cell (`code/posthoc_mechanism.py`):

| Estimator | Positive residuals | Mean signed residual |
|---|---|---|
| Jeffreys (grid) | 22/27 | +0.62 |
| raw (grid) | 16/27 | +0.26 |
| isotonic (grid) | 15/27 | +0.13 |
| NW α = 1 (grid + state) | 21/27 | +0.48 |
| NW α = 0 (grid + state) | 13/27 | +0.07 |

- Removing shrinkage cuts the mean signed residual by **58%** (grid) and **86%** (NW).
- Mean absolute error falls by about **40%**.

**Seeded leaders** (Gemma-4-31B; rounds the seeded leader holds rank 1, capped at 40):

| Framing | Observed | Frozen NW | flex_mle | flex_iso | nw_noshrink |
|---|---|---|---|---|---|
| equal turns (norm0) | 3.0 | 6.6 | 4.0 | 4.4 | 4.2 |
| neutral (ended) | 5.0 | 8.7 | 9.7 | 8.8 | 8.4 |
| w = 50 | 26.7 | 11.7 | 18.5 | 21.1 | 17.5 |

- Two of three framings move strongly toward the observation.
- **Neutral wording is not improved.**

**Other regimes** (`code/posthoc_extra.py` → `results/posthoc_extra.json`; reduced B × S, diagnostic only):
- **N = 700** (one seed; only 3 × 2 simulations, ranges are min–max): observed sd(h) 5.49 and q 0.923, approximate point agreement.
  - Unshrunk kernels: `flex_iso` 5.38 / 0.921, `nw_noshrink` 5.48 / 0.924.
  - Frozen kernels: flexible 4.61 [4.17, 5.01], NW 4.81.
- **Threshold sweep** (16 points): coverage of q is **13/16** with `nw_noshrink`, against 7/16 for the frozen NW. Mean residual falls from +0.107 to +0.063. `flex_iso` on grid rows only reached 8/16.
- **Mixed-model populations (X1):** predicted type gap and dispersion move toward the observations (e.g. 1 DeepSeek + 7 Gemma: type gap 7.6 → 8.3, observed 9.8). But P(the minority agent ranks first) moves *away* in that arm (0.58 → 0.40, observed 0.80). **Mixed.**

### 2c: per-agent ID disorder
Not rerun. The paper does not rely on it.

## 3. Task B: mechanism checks

### B1: dose–response in calls per cell
We replaced subsampling with a known-kernel simulation, which isolates estimator bias:
- the true kernel is a fitted isotonic kernel;
- synthetic calibration was generated at 0.5×, 1× and 4× the real per-cell counts;
- 6 replicates each.

| Truth | Median calls per cell | Jeffreys bias in sd(h) | Raw bias |
|---|---|---|---|
| Gemma-4-31B norm100 | 2 / 4 / 16 | −1.49 / −1.00 / −0.29 | −0.07 / −0.06 / +0.04 |
| Claude Haiku norm100 | 2 / 4 / 16 | −1.60 / −1.02 / −0.22 | +0.01 / +0.01 / +0.01 |

Across all six known kernels at the real counts, Jeffreys bias is −0.78 to −1.02 for deferential kernels and +0.07 for turn-taking ones. Raw and isotonic bias is at most 0.23.

*Our reviewer's caveat:* the truths are fitted monotone kernels. So this shows the mechanism **can** reproduce the observed attenuation, not that real kernels equal these truths.

### B2: lapse decomposition
- We fitted $P_{\rm win} = \lambda + (1-2\lambda)\sigma(\eta\Delta)$ to Jeffreys-smoothed vs raw cell win probabilities.
- The median $\hat\lambda_J - \hat\lambda_{\rm raw}$ is **0.054**, against $\tfrac12/(n+1) \approx 0.10$ at the median n = 4. That's the same order of magnitude, about half.
- **Artificial-lapse sweep:**

| Kernel | Type | sd(h) as λ goes 0 → 0.15 |
|---|---|---|
| Gemma-31B norm100 | deferential | 5.83 → 4.50 |
| Haiku norm0 | turn-taking | 0.91 → 1.07 |

### B3: coverage residual in the four leftover cells
| Cell | Decisions | Share with \|mean\| > 4 (outside the grid hull) |
|---|---|---|
| Gemma-26B norm0 | 904 | **23.9%** |
| Gemma-26B norm100 | 904 | 13.1% |
| Qwen3.5-397B norm0 | 920 | 4.3% |
| Haiku ended | 928 | 4.5% |

- At matched displayed states, population lead rates mostly agree with grid rates within the Wilson intervals.
- The exception is that agents behind by 1 defer more in populations than in the grid for Gemma-26B norm100: 4/13 (0.31) vs 6/8 (0.75).
- So the remaining residual is consistent with coverage (states outside the calibrated hull), plus a possible real population effect for some cells.

## 4. Task C: paper edits (done)
1. **§4.2 and abstract:**
   - the pre-registered criterion is still reported as failed;
   - a new paragraph, "Why the kernels under-predict (post hoc)", reports the matched decomposition (58% / 86%), the known-kernel dose–response, the lapse sweep, N = 700 and threshold coverage, and the residual left outside the hull;
   - the abstract says "post hoc, most of this bias traces to smoothing sparse calibration cells toward indifference."
2. **Appendix C (seeded leaders):** unshrunk kernels move predictions toward observation under equal turns and at w = 50, but not under neutral wording.
3. **§5:** the "not upper limits" framing is replaced by a methods recommendation: avoid shrinking saturated cells toward ½, use monotone smoothing, allocate calls to visited states.
4. **Appendix D:** amendment **A10** (post-hoc attenuation re-evaluation). It's numbered A10 because A9 was already used for the real-task game. Pilot D is listed separately.
5. No post-hoc estimator is described as prospective or as passing the pre-registered criterion.

The paper still fits 8 pages. The Word file is rebuilt (`paper/Swarm_Hierarchy_Apart_Report.docx`).

## 5. Task D: prospective confirmation (done: 1 of 2 conditions pass)
`code/pilotD.py`; a targeted pilot on the two conditions the critique suggested.
- **Conditions:** Gemma-4-26B norm100 and Qwen3.5-397B norm0.
- **Anchors:** the 10 most-visited positive displayed Δ bins plus their mirrors, 20 per condition. **30 calls each**, 1,200 in total. Frozen in `results/pilotD_anchors.json`.
- **Kernels:** a 2 × 2 frozen before any new population:
  - {Jeffreys, isotonic} × {old grid, old grid + new anchors};
  - the isotonic predictive draws use clipped resampling, so they are regularised (the point estimate is unshrunk PAVA).
- **Primary endpoint:** sd(h) of exactly the 5 declared seeds.
- **Success rule:** LPD(isotonic, new) > LPD(Jeffreys, new) **and** |obs − pred| ≤ 0.5. Its conditional operating characteristics are frozen with the predictions.
- **Safeguards added after review:**
  - completion check;
  - a single hashed calibration snapshot;
  - old-grid row hashes;
  - exclusive file creation;
  - runs follow the frozen manifest;
  - evaluation refuses partial seed sets.
- **Frozen** at 19:16:07 (sha `233e80f2…`), before any pilot population existed. Conditional operating characteristics: success probability 0.87 if the candidate is true, 0.00–0.08 if the reference is true.
- **Outcome** (20:21; exactly the 5 declared seeds per condition; hash and protocol checks passed):

| Condition | Observed sd(h) | Jeffreys, new data | Isotonic, new data (candidate) | LPD candidate vs reference | Success rule |
|---|---|---|---|---|---|
| Qwen3.5-397B norm0 | 2.91 | 2.09 [1.81, 2.49] (miss) | **3.37 [2.85, 3.90] (hit; error 0.46)** | −0.74 vs −6.12 | **pass** |
| Gemma-4-26B norm100 | 5.25 | 3.23 [2.72, 3.68] (miss) | 4.02 [3.55, 4.52] (miss; error 1.23) | −13.2 vs −43.8 | **fail** (error > 0.5) |

- **Secondary measure, rank stability q:** observed 0.77 and 0.91. The candidate covers Qwen (0.80) but not Gemma (0.86), and is closer than Jeffreys in both.
- **Old vs new calibration data.** Isotonic on the old grid gave 2.94 for Qwen (closer than the new-data fit) and 3.51 for Gemma (further). So the visited-state calls did not consistently help; the estimator change did.
- **Reading:**
  - removing shrinkage improves prospective prediction in both conditions;
  - the candidate procedure (unshrunk *and* monotone, with different uncertainty draws) met its target in one condition (Qwen), which does not isolate shrinkage alone;
  - Gemma-4-26B is still substantially under-predicted, consistent with the B3 finding that agents behind defer more inside its populations and that many decisions fall outside the calibrated range.
- **Bottom line:** shrinkage is a major cause of the miscalibration, **not the only one**.

## 6. Refinement review record
| Round | Reviewer | Overall | Verdict |
|---|---|---|---|
| 1 | gpt-6.1-sol (ultra), thread `01a1be26-dee4-7cc2-9c33-7586386c3a9f` | 6.45 | REVISE |
| 2 | same thread | 7.70 | REVISE |

The final GPT-6-astra (high) cross-check follows below (section 8). Full records are in `refine-logs/round-*-review.md` and `refine-logs/score-history.md`.

## 7. What remains open
- Interval coverage is still below nominal (42–46/54).
- Seeded-leader persistence under neutral wording is not explained.
- In X1, unshrinking worsens the predicted P(minority ranks first).
- The residual is concentrated outside the calibrated hull (24% of Gemma-26B norm0 decisions).
- Pilot D failed for Gemma-4-26B (under-prediction persists after unshrinking), so a second mechanism remains for some models.

## 8. Independent cross-check (GPT-6-astra, reasoning effort high; fresh thread `01a1be8b-90af-7221-b8f8-85f1fc1e8047`)
- **Verdict:** agrees with REVISE. The primary verdict (gpt-6.1-sol ultra, round 2: 7.70, REVISE) stands.
- **Verified:**
  - every headline number (sparsity; 22/27 → 13/27; 58.3% / 85.9% signed and 41.7% / 40.6% absolute; dose–response; lapse sweep; threshold; N = 700 means; outside-hull shares);
  - pilot D's evaluation, reproduced exactly; it follows the frozen rule (Qwen pass, error 0.459; Gemma fail, error 1.226);
  - prospectivity, supported by file timestamps and hashes.
- **Disagreement with the primary review:** the primary said dose in calibration count was untested. The cross-check notes it has since been simulated (two truths, synthetic only).
- **Issues raised and fixed:**
  - Figure 3 dropped the null points (pandas read `null` as missing);
  - the §4.2 heading said "unbiased" (now "better once calibration shrinkage is removed");
  - Figure 3(a) was labelled "prospective" although Sonnet's forecast was amended;
  - interval construction is now disclosed in the caption;
  - residual localisation softened to "possibly";
  - the pilot is described as comparing procedures, not isolating shrinkage;
  - NW shrinks toward the base rate, not ½;
  - the freeze time is corrected to 19:16:07.
- **Noted, not changed:**
  - anchor creation is not atomic;
  - runtime checks validate the prediction hash, not every source hash;
  - `code/llm.py` now differs from its frozen hash only in the credit-cap line (50 → 100, edited at 19:43, not by this session).
