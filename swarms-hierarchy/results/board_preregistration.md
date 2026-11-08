# Message-board pilot — pre-registration (written 2026-11-07, before any board run)
Model: adaption:gemma-4-31b-it. N=8, 30 rounds, rho=0.8, memory M=20, seeds 1-3; arms: neutral (all arrive at round 0)
vs staggered (2 of 8 agents present from round 0, the rest arrive at rounds 3-10). No scores anywhere.

Primary order parameter: share of all ASSIGN messages sent by the single most active assigner (assign_top_share),
and of ACCEPTED assignments (accepted_top_share).
Null: activity-matched permutation — re-attribute each ASSIGN to a random agent active in that round (1,000 perms).
Symmetry breaking = observed assign_top_share above the 95th percentile of the null in >= 2/3 seeds of an arm.
Secondary: Gini of ASSIGN counts; number of coordinators (agents with >= max(2, 10%) of ASSIGNs); delegation depth >= 2.
Pre-declared prediction (incident-inspired): early arrivals become the top assigners in the staggered arm
(top assigner is an early agent in >= 2/3 seeds). If no arm breaks symmetry, report "symmetry breaking requires X"
and do not scale up.
- Note (pre-run, after a 4-round smoke test that is excluded from analysis): task dependencies loosened to 0/0/0/1/2 parents so several tasks are open in parallel.
