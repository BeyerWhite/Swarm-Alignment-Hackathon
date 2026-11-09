# Paper Improvement Log
Reviewer routing (user CLAUDE.md, overrides skill default): primary gpt-6.1-sol, effort ultra, fresh thread each round; cross-check gpt-6-astra, effort high, after the final round. Paper source: paper.md → .docx (build_docx.py); "compiled PDF" = Chrome render (preview_main.pdf / preview_full.pdf).

## Score Progression
| Round | Score | Verdict | Key changes |
|---|---|---|---|
| Round 0 (original) | 6/10 | Almost | baseline (reviewed in round 1) |
| Round 1 | (see round 2 review) | | throughput claims corrected; both frozen framing endpoints reported with paired tests (Table A2); Appendix B → v2 prompt; role-emergence wording; abstract/intro open on problem→gap→idea→result; post-hoc paragraph condensed with Table A3; Code and Data moved to appendix; 4.4 retitled; review history moved to Appendix D; Figures 2, 3, 5 typography/titles |

## Round 1 review (gpt-6.1-sol ultra, thread 01a1be97-3244-7b42-a3e9-3ae2c0625e15) — 6/10, Almost
<details><summary>Raw review</summary>

Overall 6/10 weak accept. Strengths: prospective micro→population prediction (104–224 nats); calibration diagnosis with same-data comparisons, synthetic recovery and prospective pilot; framing reversals and mixed-model result; transparent unfavorable results. No critical flaw. MAJOR: (1) "doubling output in four of five models" overstated (ratios 2.25, 1.95, 1.46, 1.24, 1.02×); (2) transfer account omits frozen top-planner-share endpoint (2/5 match, GLM opposite) and over-interprets raw recurrence (Gemma-31B recurrence above null +0.043, paired p = .242); replace "predicts the framing effect only where it is large" with "Gemma-4-31B matches the frozen framing prediction"; (3) Appendix B describes version 1 protocol and code path; (4) role emergence overstated — coordination is an offered action; Gemma-26B plans on ~48% of turns; "reproduces the incident" → analogy; (5) abstract/intro open on background; post-hoc paragraph too dense — move secondary numbers to appendix table. MINOR: permutation pooling across shared seeds (paired: shuffle p ≤ .039); H1 eligibility consistency; "71 cells" terminology; review-history narration in Methods; "same optimism" defeat framing; retitle 4.4; conclusion self-negation. Visual: Figures 1–3 small text; Fig 5a title overstates; Fig 5b caption vs prose definition mismatch; Appendix D heading stranded; angle brackets lost in prompts; move Code and Data to appendix. Verdict: Almost.
</details>

### Fixes implemented (round 1)
1. Throughput: "planning raises output in four of five models, about doubling it for the two Gemma models" (abstract, intro, 4.4, discussion, Fig 5a title).
2. Framing transfer: both frozen endpoints with exact paired sign-flip tests and null-adjusted recurrence (new Table A2); 4.4 paragraph "Gemma-4-31B matches the frozen framing prediction" incl. the null-adjusted caveat and Qwen-397B's opposite move.
3. Appendix B: version-2 prompt quoted, templates as code, `code/taskgame2.py`; Code and Data path corrected.
4. Role wording: "voluntarily take planning and execution roles"; "planning roles recur, weakly", with the planner-recurrence definition used in prose and Fig 5b caption; incident "analogue".
5. Abstract and introduction open with the deployment problem and gap; incident shortened.
6. Post-hoc paragraph condensed; secondary numbers in new Table A3 (all unfavorable values kept).
7. Minor: paired tests (throughput p = 0.008; shuffled-list p ≤ 0.039); "71 model–framing configurations of 39 score-pair cells"; review history → Appendix D; "same under-prediction"; 4.4 retitled; conclusion ends on the highest-throughput finding; Code and Data → Appendix E; Figures 2/3 larger text, Fig 3 annotation moved.
Not changed: H1 eligibility (sign test unaffected per reviewer); Appendix D heading orphan (Chrome preview only; docx layout differs).

## Round 2 review (gpt-6.1-sol ultra, fresh thread 01a1bea2-e50d-7500-9c26-660b5c2b232d) — 7/10 Accept, verdict Almost
<details><summary>Raw review (condensed verbatim points)</summary>

No CRITICAL issue. MAJOR 1: size-scaling claims (abstract, contribution 3) omit that sharpness beyond N = 16 rests on fitted-kernel simulations. MAJOR 2: "144 runs" conflates 90 common-design runs with 54 contact-rate extension runs; abstract conflates Bonabeau dynamics with the separate BNR extension. MAJOR 3: estimators and pilot success rule insufficiently specified (interpolation/clamping, isotonic construction, uncertainty procedures, NW α, pilot criterion, 600 calls per condition). MINOR: 2/N wording (expected share, not cap; incident scale); Table A2 Qwen 7 pairs; "24 of 30 non-tied" (32 eligible); label point-estimate vs predictive residuals; "But emergent coordination is not the best fix" → "Diverse task views raise output most"; move "we claim no stationary phase transition" to Limitations; abstract should open with the forecasting problem; harmonize "mostly followed" vs "about half carried out". Missing reference: an isotonic-regression reference. Visual: Figures 1–4 small internal text; References/Appendix headings stranded in the Chrome preview; Table A2 wrapping; Table A3 abbreviations. Verdict: Almost.
</details>

### Fixes implemented (round 2)
1. Abstract rewritten to open with the forecasting problem and gap; "measured response kernels embedded in Bonabeau-style dynamics"; "90 common-design populations"; size claim scoped to fitted-kernel simulations; contribution 3 and conclusion scoped likewise.
2. Methods: 90 common-design runs (framings listed; Sonnet 3 × 2) + 54 contact-rate extension runs.
3. New Appendix E specifying all estimators, uncertainty procedures and the pilot rule (600 calls per condition); Code and Data → Appendix F; Barlow et al. (1972) added for isotonic regression.
4. Minor: 2/N as expected share at incident scale, finite-window statement moved to Limitations; Qwen 7 pairs; "non-tied eligible runs"; point-estimate vs predictive residuals labelled; "Diverse task views raise output most"; following rates 41–85% stated; Table A3 abbreviations defined.
Not changed: Chrome-preview heading orphans (the submitted .docx paginates independently); Figure 1/4 internal font (page budget).

## Final cross-check (gpt-6-astra high, fresh thread 01a1bea8-b252-7060-bb7f-46dfb1ec2a2e) — 7/10, Almost
Agrees with the primary score and verdict (7/10, Almost). Per-finding status on the round-2 version: MAJOR 3 resolved; MAJOR 1 and 2 partly (residual size-scope wording in §4.3/§5; contribution 2 still said 144); most minor items resolved; figure text size and appendix pagination not fully resolved.
**Disagreements / additions flagged against the primary review:**
- Table A3 threshold coverage was wrong (7 → 6 of 16) because post-hoc bounds were rounded before the inclusion test.
- Recurrence statistics mixed samples (37 runs for the means vs 32 eligible for the sign test).
- 0.58 in Table A3 was a post-hoc rerun, not the frozen forecast (0.61).
- 41–85% is an attempt rate at live-assignment turns, not the fraction of assignments completed.
- Missing reproducibility details: round-start scores, seed-mean construction, KDE bandwidth, the pilot's resampling probability.
- Deployment recommendations were broader than the validation.
### Fixes implemented after cross-check
1. `posthoc_extra.py` keeps full precision for coverage and was re-run; Table A3 frozen threshold coverage is now 6/16 (unshrunk 13/16 unchanged).
2. Recurrence restated on the 32 eligible runs: 0.27 vs 0.21; 24 of 30 non-tied; p = 0.0007.
3. Table A3 labels 0.58 as a post-hoc rerun (frozen 0.61).
4. The following rate is defined as attempts at live-assignment turns.
5. Methods add round-start scores, the seed-mean construction and the KDE bandwidth; Appendix E names the fitted isotonic probability.
6. Recommendations scoped to "in this sandbox… one of two prospective conditions"; abstract last line scoped; §4.3/§5 size wording now says fitted-kernel simulations; contribution 2 says 90 common-design populations; 2/N is "would not typically".
7. To stay within 8 pages, the order-bias limitation moved to Appendix A.
Not fixed: small internal text in Figures 1 and 4; Chrome-preview pagination of the Appendix D heading (the .docx paginates independently).

## Score Progression (final)
| Round | Reviewer | Score | Verdict |
|---|---|---|---|
| 1 (original paper) | gpt-6.1-sol ultra | 6/10 | Almost |
| 2 (after round-1 fixes) | gpt-6.1-sol ultra | 7/10 | Almost |
| Cross-check (after round-2 fixes) | gpt-6-astra high | 7/10 | Almost |

## Files
- Paper source: `paper_round0_original.md` → `paper.md`
- PDF previews: `main_round0_original.pdf`, `main_round1.pdf`, `main_round2.pdf`
- Word documents: `Swarm_Hierarchy_Apart_Report_round{0_original,1,2}.docx`
