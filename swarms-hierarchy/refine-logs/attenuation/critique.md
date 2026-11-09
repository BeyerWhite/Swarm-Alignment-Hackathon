# Post-hoc analysis: the systematic under-prediction is mostly kernel attenuation (shrinkage toward ½)

**For:** an agent working in `swarms-hierarchy/` of `BeyerWhite/Swarm-Alignment-Hackathon`. Paths are relative to that folder.
**Status:** exploratory and post-hoc. Nothing below was pre-registered. Label every number from this file as post-hoc
in the paper. Do **not** edit frozen prediction files or earlier results. Write new files only.
**Deadline context:** the paper is due 2026-11-08 (tonight). Tasks are ordered by value per minute.

---

## 1. Claim

The paper has two failures:
- **(C2)** populations are more unequal than the frozen kernels predict (dispersion residual > 0 in 22/27 cells and
  in 7/7 models);
- **(C6)** seeded-leader persistence is mis-predicted.

These are one phenomenon. Real agents are **more decisive** than the estimated kernel:
- In deference conditions they are more stratified and their ranks persist longer.
- In the equal-turns condition (norm0, anti-deference), the seeded leader is overthrown faster than predicted
  (3.0 vs 6.6 rounds).

The main cause is the estimator, not the agents.
- **Few calls per cell.** The grid has 13 Δ values × 3 mean levels; 216 calls gives about 5 calls per cell.
- **Every estimator in the pipeline shrinks toward ½:**
  - `kernel_flexible` uses Jeffreys rates (k+½)/(n+1);
  - predictive draws come from Beta(k+½, n−k+½);
  - `kernel_nw` adds pseudo-count α = 1 toward the base rate.
- **Saturated cells can only err one way.** Many cells are near 0 or 1, where any error moves toward indecision. With
  n ≈ 5, a cell whose true rate is 1 is estimated at ≈ 0.92.
- **The result acts as an artificial lapse rate** λ_eff ≈ ½/(n+1) ≈ 0.08, and larger once posterior draws are added. It
  flattens the antisymmetric part of P_win = ½ + ½[ℓ(Δ) − ℓ(−Δ)], so more contests are decided by a coin flip.
- **Effect on the population:**
  - Deference conditions: less stratification and faster rank mixing than reality.
  - Turn-taking conditions: slower rotation than reality.
  - Population metrics amplify a slope bias by roughly 1/(1 − β), so a modest kernel bias becomes a large bias in sd(h).

## 2. Evidence (quick checks, point estimates, 24 simulated seeds; must be redone properly — Task A)

### 2a. Common design (N = 8, ρ = 0.5, μ = 0.1, 40 rounds, unseeded)
Mean sd(h) from simulating the grid-interpolated kernel (same bilinear interpolation as `kernel_flexible`):

| estimator | mean residual (obs − pred) | fraction obs > pred (27 cells) |
|---|---|---|
| posterior draws (as in frozen predictions) | +0.61 | 0.85 |
| Jeffreys point estimate | +0.61 | — |
| **raw MLE (eps → 0)** | **+0.24** | **0.56** (≈ chance) |

Per cell (obs / Jeffreys / MLE):

| cell | obs | Jeffreys | MLE |
|---|---|---|---|
| Haiku ended+comp | 6.26 | 4.69 | 5.64 |
| Haiku norm100 | 6.25 | 5.14 | 5.90 |
| GLM norm100 | 5.74 | 4.69 | 5.75 |
| Gemma-31B norm100 | 5.60 | 4.81 | 5.93 |

Full table: re-run Script 3 (Appendix).

### 2b. Seeded leaders (Gemma-4-31B, `results/hyst_pop.jsonl`, seeded runs only)
Rounds until the seeded leader first loses rank 1 (observed matches the paper):

| cond | obs | Jeffreys | MLE | paper's frozen NW prediction |
|---|---|---|---|---|
| norm0 | 3.0 | 4.8 | 4.2 | 6.6 |
| ended | 5.0 | 10.1 | 8.6 | 8.7 |
| norm50 | 26.7 | 12.3 | 17.5 | 11.7 |

All three move toward the observed values. The norm50 persistence is therefore mostly the same artifact; this
supports the paper's "not Stauffer feedback" reading.

### 2c. Alternative ruled out: quenched per-agent disorder via agent ID
- Calibration samples a fresh agent ID per call. Populations keep IDs fixed. ID-specific LEAD bias would therefore
  act as hidden fitness, as in the mixed-model (DeepSeek) result.
- Test: split-half correlation of each agent's LEAD residual, conditioned on displayed state (binned Δ × mean,
  estimated from population data), between odd and even rounds, against a within-run permutation null.
- Result: null overall (Fisher combined p = 0.54 across 27 non-DeepSeek cells). The only weak signal is Gemma-26B
  norm100 (r = 0.40, p = 0.03).

### 2d. Remaining residual (+0.24 after MLE) is concentrated in a few cells

| cell | obs sd(h) | MLE prediction |
|---|---|---|
| Gemma-26B norm0 | 5.63 | 4.49 |
| Gemma-26B norm100 | 4.54 | 3.35 |
| Qwen3.5-397B norm0 | 3.15 | 1.98 |
| Haiku ended | 3.67 | 2.32 |

- These are the models where state-matched calibration helped most (+22, +41 nats).
- Likely cause: coverage. Populations sit near mean score 0 at small negative Δ, where the grid is thin (~5 calls
  per cell), and the kernel clamps at mean ±4.
- At matched displayed states, behind agents in populations defer more than the grid cells imply for some models
  (Haiku ended+comp, Qwen-397B ended). With n = 10–30 per cell this is only suggestive.

---

## 3. Tasks

### Task A — Proper post-hoc re-evaluation (highest priority, no LLM calls)
1. Add new kernel estimators in a **new** module (e.g. `code/posthoc_kernels.py`). Do not change `population2.py`.
   - `flex_mle`: same as `kernel_flexible`, but rate = k/n. For uncertainty, use a **near-flat** posterior
     Beta(k+ε, n−k+ε), ε = 0.05.
   - `flex_iso`: per mean level, isotonic regression of ℓ in Δ. Increasing for conditions with s > 0, decreasing for
     s < 0, with the sign taken from the calibration-only s. Weight by n. Then interpolate as before. This borrows
     strength across neighbouring Δ without pulling toward ½. Uncertainty: cell bootstrap, then re-fit.
   - `nw_noshrink`: `kernel_nw` with α = 0.
   - Also include the original `flexible` (Jeffreys) as the reference.
2. Re-run the paper's evaluation through the existing authoritative pipeline: `code/reeval_followups.py`,
   `evaluate.seed_mean_dist`, and the LPD/coverage code for C2. Use the same B × S simulation counts, seed-mean
   predictive distributions and KDE bandwidth rule as the paper. Do it for:
   - the common design (all 8 models), giving coverage, LPD margin vs null, and mean residual z per model;
   - hysteresis (C6), giving rounds held, with the same censoring;
   - X1 mixed populations (C4), giving sd(h) and type gap;
   - the N = 700 run and the threshold sweep (C3), giving q and sd(h).
3. Output `results/posthoc_attenuation.json` and `results/posthoc_attenuation.csv`. Each row: estimator, model, cond,
   metric, obs, pred mean, 90% interval, z, hit. Add one summary row per estimator: fraction of cells with obs > pred,
   and mean z.
4. **Acceptance:** report the numbers as they come out. The finding holds if `flex_mle` and/or `flex_iso` bring the
   fraction of cells with obs > pred to about 0.5, and the mean residual z near 0, while keeping the margin over the
   null. If they don't, say so; that falsifies Section 1.

### Task B — Mechanism checks (cheap, no LLM calls)
1. **Dose–response in n.** Subsample calibration calls per cell to 50% and 25%. Re-fit the Jeffreys kernel and plot the
   predicted mean sd(h) against calls per cell. Attenuation predicts the bias grows as n falls; the MLE curve should
   stay flat in expectation.
2. **Lapse decomposition.** Per model × cond, fit P_win(Δ) = λ + (1−2λ)σ(ηΔ) to (a) the Jeffreys-smoothed cells and
   (b) raw calls. Report λ̂ for each. The difference should be ≈ ½/(n+1) plus a draw-induced term.
3. **Coverage residual (2d).** For the four residual cells, compare population LEAD rates with grid rates at matched
   displayed states (|Δ_pop − Δ_grid| ≤ 0.15, |mean| ≤ 5). Report counts and Wilson intervals. Also report the share of
   population decisions outside the grid hull (|mean| > 4).

### Task C — Paper edits (after A; keep it honest)
1. **§4.2 and abstract.** Replace "populations were on average more unequal than predicted" plus the "optimistic
   screen" framing with something like the following, filled in with Task A's real numbers:
   "Populations were more unequal than predicted; a post-hoc analysis attributes most of this to shrinkage toward ½
   in sparsely sampled, saturated calibration cells (≈5 calls/cell). Unshrunk estimates remove most of the bias
   (obs > pred in X% vs 85% of cells)."
   Keep the statement that the pre-registered criterion failed.
2. **§4.3 / Appendix C, seeded leaders.** Note that the norm50 under-prediction shrinks under unshrunk kernels,
   consistent with the attenuation explanation and the null inequality probe.
3. **§5.** Replace "two-agent tests ... should not be read as upper limits" with a methods recommendation:
   - avoid shrink-to-½ priors at saturated cells;
   - allocate calls to the cells populations visit (small negative Δ near mean 0);
   - use monotone or hierarchical smoothing.
4. **Appendix D.** Add an amendment entry, "A9 (post-hoc, after all runs): attenuation re-evaluation", pointing to
   Task A outputs and this file.
5. Do not describe any post-hoc estimator as prospective or as having passed the pre-registered criterion.

### Task D — Prospective confirmation (only if time and quota remain; otherwise list as future work)
- Pick 2 cells with large residual and cheap models: Gemma-26B norm100 and Qwen-397B norm0.
- New calibration: ≥30 calls per cell at visited states.
- Freeze `flex_iso` predictions (write-once, hash-stamped) **before** running 5 new population seeds per cell.
- Budget: roughly 600 calibration + 1,600 population calls per cell. Check the Adaption cap first.

---

## 4. Notes for the agent
- Decisions in `population_v2*.jsonl`: round t uses the scores *after* round t−1 (`history[t-1]['h']`; zeros or `h0`
  at t = 0), rounded to 0.1. Events are `[i, j, choice_i, choice_j, winner]`.
- Calibration rows: `exp == "kernel"`, with fields `delta`, `mean`, `choice`. Fresh IDs per call (see
  `calibrate.py:kernel_jobs`).
- Seeded hysteresis runs are the rows of `results/hyst_pop.jsonl` with `h0` not null.
- Run analysis scripts with `python3 -I` from `swarms-hierarchy/`.

---

## Appendix — quick-check scripts (reproduce Section 2; run from `swarms-hierarchy/`, outputs to `analysis_posthoc/`)

### Script 1: extract population decisions and compare with the kernel by gap bin (`analysis_posthoc/resid.py`)
```python
import json, glob, sys, os, numpy as np, pandas as pd
sys.path.insert(0, 'code'); os.makedirs('analysis_posthoc', exist_ok=True)
from population2 import kernel_flexible
rows = []
for f in ['results/calib_v2.jsonl'] + glob.glob('results/calib_v2_adaption_*.jsonl'):
    for l in open(f):
        r = json.loads(l)
        if r.get('exp', 'kernel') != 'kernel' or r.get('choice') not in ('LEAD', 'DEFER'): continue
        rows.append(dict(model=r['model'], cond=r['cond'], delta=r['delta'], mean=r['mean'], lead=int(r['choice'] == 'LEAD')))
cal = pd.DataFrame(rows)
dec = []
for f in ['results/population_v2.jsonl'] + glob.glob('results/population_v2_adaption_*.jsonl'):
    for l in open(f):
        r = json.loads(l)
        if r.get('rho') != 0.5 or r.get('N') != 8 or r.get('h0') is not None: continue
        H = [np.zeros(r['N'])] + [np.array(x['h']) for x in r['history']]
        for t, x in enumerate(r['history']):
            h = H[t]
            for i, j, ci, cj, w in x['events']:
                for a, b, c in ((i, j, ci), (j, i, cj)):
                    ha, hb = round(h[a], 1), round(h[b], 1)
                    dec.append(dict(model=r['model'], cond=r['cond'], seed=r['seed'], t=t, agent=a, name=r['names'][a],
                                    d=round(ha - hb, 1), m=(ha + hb) / 2, lead=int(c == 'LEAD')))
D = pd.DataFrame(dec)
D.to_pickle('analysis_posthoc/dec.pkl')
lab = ['0', '0.1-0.3', '0.4-0.9', '1-1.9', '2-3.9', '4-7.9', '8+']
D['ab'] = pd.cut(D.d.abs(), bins=[0, .05, .35, .95, 1.95, 3.95, 7.95, 99], labels=lab, include_lowest=True)
out = []
for (mo, co), g in D.groupby(['model', 'cond']):
    c = cal[(cal.model == mo) & (cal.cond == co)]
    if len(c) < 100: continue
    k = kernel_flexible(c); g = g.assign(pk=[k(d, m) for d, m in zip(g.d, g.m)], sgn=np.sign(g.d))
    for b, gb in g[g.d != 0].groupby('ab', observed=True):
        out.append(dict(model=mo, cond=co, bin=b, n=len(gb),
                        obs=gb[gb.sgn > 0].lead.mean() - gb[gb.sgn < 0].lead.mean(),
                        pred=gb[gb.sgn > 0].pk.mean() - gb[gb.sgn < 0].pk.mean()))
pd.DataFrame(out).to_csv('analysis_posthoc/resid_by_gap.csv', index=False)
```

### Script 2: split-half test for persistent per-agent propensity (`analysis_posthoc/split.py`; needs Script 1)
```python
import pandas as pd, numpy as np
from scipy.stats import combine_pvalues
D = pd.read_pickle('analysis_posthoc/dec.pkl')
D['run'] = D.model + '|' + D.cond + '|' + D.seed.astype(str)
D['db'] = np.sign(D.d) * pd.cut(D.d.abs(), [-.01, .05, .35, .95, 1.95, 3.95, 7.95, 99], labels=False)
D['mb'] = np.floor(D.m / 2)
rng = np.random.default_rng(0); res = []
for (mo, co), g in D.groupby(['model', 'cond']):
    g = g.assign(r=g.lead - g.groupby(['db', 'mb']).lead.transform('mean'), half=g.t % 2)
    a = g.groupby(['run', 'agent', 'half']).r.mean().unstack('half').dropna()
    if len(a) < 10: continue
    c = np.corrcoef(a[0], a[1])[0, 1]
    null = [np.corrcoef(a[0], a[1].groupby(level='run').transform(lambda s: rng.permutation(s.values)))[0, 1] for _ in range(500)]
    res.append(dict(model=mo, cond=co, r_split=c, p=np.mean(np.array(null) >= c)))
R = pd.DataFrame(res); print(R)
print(combine_pvalues(np.clip(R[~R.model.str.contains('deepseek')].p, 1e-3, 1), method='fisher'))
```

### Script 3: attenuation check, common design + seeded leaders (`analysis_posthoc/atten.py`)
```python
import sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, 'code')
import population2 as P
rows = []
for f in ['results/calib_v2.jsonl'] + glob.glob('results/calib_v2_adaption_*.jsonl'):
    for l in open(f):
        r = json.loads(l)
        if r.get('exp', 'kernel') != 'kernel' or r.get('choice') not in ('LEAD', 'DEFER'): continue
        rows.append(dict(model=r['model'], cond=r['cond'], delta=r['delta'], mean=r['mean'], lead=int(r['choice'] == 'LEAD')))
cal = pd.DataFrame(rows)

def kern_grid(df, eps):  # eps=0.5 -> Jeffreys point estimate; eps->0 -> MLE
    c = df.groupby(["delta", "mean"]).lead.agg(["sum", "count"])
    g = ((c["sum"] + eps) / (c["count"] + 2 * eps)).unstack("mean").sort_index()
    g = g.interpolate(method="index", axis=0, limit_direction="both")
    g = g.T.sort_index().interpolate(method="index", axis=0, limit_direction="both").T
    D, M, Z = np.array(g.index, float), np.array(g.columns, float), g.values
    def p(delta, mean):
        d = min(max(delta, D[0]), D[-1]); m = min(max(mean, M[0]), M[-1])
        return float(np.interp(m, M, np.array([np.interp(d, D, Z[:, k]) for k in range(len(M))])))
    return p

obs = {}
for f in ['results/population_v2.jsonl'] + glob.glob('results/population_v2_adaption_*.jsonl'):
    for l in open(f):
        r = json.loads(l)
        if r.get('rho') != 0.5 or r.get('N') != 8 or r.get('h0') is not None: continue
        obs.setdefault((r['model'], r['cond']), []).append(P.order_params(r)['sd_h'])
sim = lambda p, S=24: np.mean([P.order_params(P.run(8, .5, .1, 40, P.KernelDecider(p), seed=s))['sd_h'] for s in range(S)])
def sim_draws(df, B=12):
    rng = np.random.default_rng(1)
    return np.mean([P.order_params(P.run(8, .5, .1, 40, P.KernelDecider(P.kernel_flexible(df, rng=rng)), seed=1000 * b + s))['sd_h']
                    for b in range(B) for s in range(3)])
res = []
for (mo, co), v in sorted(obs.items()):
    df = cal[(cal.model == mo) & (cal.cond == co)]
    if len(df) < 100 or 'deepseek' in mo: continue
    res.append(dict(model=mo, cond=co, obs=np.mean(v), draws=sim_draws(df), jeffreys=sim(kern_grid(df, .5)), mle=sim(kern_grid(df, 1e-9))))
R = pd.DataFrame(res); print(R.round(2).to_string(index=False))
print('mean resid  draws', (R.obs - R.draws).mean(), ' mle', (R.obs - R.mle).mean())
print('frac obs>pred  draws', (R.obs > R.draws).mean(), ' mle', (R.obs > R.mle).mean())

def held(rec):  # rounds until seeded leader first loses rank 1
    L = int(np.argmax(rec['h0']))
    for x in rec['history']:
        if int(np.argmax(x['h'])) != L: return x['t']
    return len(rec['history'])
H = [r for r in (json.loads(l) for l in open('results/hyst_pop.jsonl')) if r['h0']]
for co in sorted({r['cond'] for r in H}):
    rs = [r for r in H if r['cond'] == co]; df = cal[(cal.model == rs[0]['model']) & (cal.cond == co)]
    if len(df) < 100: continue
    ps = lambda k: np.mean([held(P.run(8, rs[0]['rho'], .1, rs[0]['rounds'], P.KernelDecider(k), seed=s, h0=rs[s % len(rs)]['h0'])) for s in range(60)])
    print(co, 'obs', np.mean([held(r) for r in rs]), 'jeffreys', ps(kern_grid(df, .5)), 'mle', ps(kern_grid(df, 1e-9)))
```
