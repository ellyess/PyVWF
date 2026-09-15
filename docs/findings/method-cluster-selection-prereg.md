# Selecting each configuration's cluster count: registered design

**Date:** 2026-09-15. Registered before any sweep runs.
**Scope:** how many clusters each of the fourteen configurations contributes
control points at, when the control-point pool is rebuilt. Terms follow
`CONTEXT.md`.

## The question

> Does choosing each configuration's cluster count from its own data beat one
> fixed rule applied to all of them, and what protocol chooses it without
> leaking the test year?

## Why the existing evidence cannot answer it

**Every cluster sweep this project has run selected on the test year.**
`method-cluster-count.md` reports test-year RMSE by `k` for nine turbine-level
regions and reads an optimum off it; `method-cluster-count-dk.md` does the same
for Denmark onshore over 19 cluster counts and four time slices. Choosing `k`
from those curves and then reporting test-year skill at the chosen `k` is
selection on the outcome.

**The curves are also flat where it matters.** Denmark onshore moves from
0.0857 RMSE at `k=200` to 0.0851 at `k=3300`, so a sixteenfold increase in
cluster count buys 0.0006. `method-cluster-count.md` carries a dated notice
saying the sweep's corrected frames were not kept, so the differences between
cluster counts were never resampled and the UK optimum is untested. **Picking
the minimum of a curve that flat is picking noise**, which is why the rule
below selects the smallest defensible cluster count rather than the best one.

## What exists, and what has to run

| Configuration | Pool's current count | Sweep on current code |
|---|---|---|
| BE, ES, FR, IE, IT, NL, NO, PT, SE | 3, 4, 10, 3, 3, 5, 5, 3, 4 | none: `cluster_list = [1, N]`, two points, one trivial |
| DK-onshore | 884 | yes, `output/validation/dk_onshore_sweep_2026-07-24`, test-year selected, annual-mean roughness |
| DE-onshore | 500 | region level only, to `k=200`, not split by domain, test-year selected |
| UK-onshore | 293 | the same |
| DK-offshore, UK-offshore | 2, 10 | none |

Thirteen of the fourteen need a sweep. The one that exists predates the
per-timestep roughness treatment and is re-run with the others.

## The protocol

**Selection is nested inside the training years and never touches the test
year.**

1. For each configuration, hold out one training year. Fit on the remaining
   training years at every cluster count in the grid. Score on the held-out
   training year.
2. Rotate over every training year, giving one fold per training year.
3. For each cluster count, take the mean fold score and its standard error
   across folds, `sd / sqrt(folds)` with `sd` the sample standard deviation.
4. **The one-standard-error rule:** let `k*` minimise the mean. The selected
   cluster count is the **smallest** cluster count whose mean is at or below
   `mean(k*) + SE(k*)`.
5. Refit at the selected count on **all** training years, and report on the
   single untouched test year.

Folds per configuration, from each one's own training years: seven for BE, ES,
FR, IT, NL, NO, PT and SE (2015 to 2021); five for IE (2017 to 2021); five for
DK-onshore and DK-offshore (2015 to 2019); four for DE-onshore, UK-onshore and
UK-offshore (2015 to 2018).

**The standard error is estimated from four to seven numbers and is itself
noisy.** That is a limitation and not a defect of the rule: a noisy standard
error widens the interval, the interval admits a smaller cluster count, and the
rule errs toward the simpler model. It cannot err toward the more complex one.

## The metric

**Both, jointly, with the tiebreak fixed now.** The rule in step 4 is applied
separately to RMSE and to MAE.

- If the two select the same cluster count, that is the count.
- **If they disagree, the smaller count is taken.**

The project currently disagrees with itself here: both existing sweep documents
select on RMSE, while the chapter's grid tables, the domain-split study and its
gates are all in MAE. Running both closes that by making it visible rather than
by picking a side.

**How often they disagree is reported**, with the size of the disagreement in
cluster counts and in test-year score. That is a finding about the sweeps and
not a by-product.

## The cluster-count grids, fixed now

Country-level, capped at the number of grid points in that configuration's
fleet: **1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100**.

Turbine-level onshore, capped at fleet size: **1, 10, 25, 50, 100, 200, 500,
1000**.

Turbine-level offshore, capped at fleet size: **1, 2, 3, 5, 10, 25, 50, 100**.

The country-level grid deliberately reaches well past what a national series
can plausibly identify. The standing caveat is that N offsets fitted against
one national series are under-determined; the protocol should demonstrate that
rather than assume it, and a grid that stops at 10 could not.

## Held constant

The time slice at `fixed`, which is what the pool's control points use; the
wind archive at `input/era5/EU_2026-09` with the per-timestep roughness
treatment, per the 2026-09-15 amendment to `method-domain-split-prereg.md`;
each configuration's shipped bounding box, with Denmark's study-scoped 15.4
east; the curve library each row already uses; and the single test year, which
no selection step reads.

## The declared baseline

**Per-country selection is compared against one fixed rule, and the comparison
is registered here rather than chosen afterwards.** Two baselines, both fixed
before any new sweep:

- **B1, the incumbent:** the chapter's own cluster counts, listed in the table
  above. This is what the rebuild replaces.
- **B2, one rule for everyone:** `k = 100` for every turbine-level
  configuration and `N = 5` for every country-level one. 100 is where
  `method-cluster-count-dk.md` says essentially all the skill is captured, and
  5 is the median of the chapter's country-level counts. **Both values come
  from test-year-selected curves**, which makes B2 a strong baseline rather
  than a fair protocol, and that is the right direction for a baseline: if
  per-country selection cannot beat a contaminated fixed rule, it has not
  earned its cost.

All three are evaluated the same way, by refitting on all training years and
scoring the single test year.

## Gates

| Gate | Requirement | Outcome |
|---|---|---|
| **C-G1** | The protocol completes for all fourteen configurations under both metrics, with no configuration failing to produce a selected count. A configuration that cannot be fitted at some cluster count has that count dropped from its grid, recorded, and the rule applied to what remains. | |
| **C-G2** | Per-country selection beats **B2** on test-year MAE, by more than 0.002, in at least eight of the fourteen configurations. 0.002 is the screen this project already uses for a difference that is negligible in a capacity factor, and eight of fourteen is a simple majority. | |
| **C-G3** | Per-country selection beats **B1** on test-year MAE in at least eight of the fourteen. This is the weaker question, since B1 was not chosen by any protocol. | |

**If C-G2 fails, per-country selection is not adopted**, and the finding is
that one fixed rule is as good, which is worth the same as the opposite result
and is reported with the same prominence.

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| C-P1 | The one-standard-error rule selects a cluster count well below the minimising one in most configurations, because the curves are flat. Stated as: the selected count is below `k*` in at least nine of the fourteen. | |
| C-P2 | **C-G2 fails.** I expect per-country selection not to beat B2 by 0.002 in eight configurations, on the Denmark evidence that the curve is flat over a factor of sixteen. This is the prediction that the rebuild is unnecessary, and it is stated so that it can be refuted. | |
| C-P3 | RMSE and MAE select the same cluster count in a majority of configurations, and where they disagree the test-year MAE difference between the two choices is below 0.002. | |
| C-P4 | Every country-level configuration selects below 10, because a single national series cannot identify more. | |
| C-P5 | The two offshore configurations select the smallest counts in their grids, having the fewest units and the least spatial spread. | |

## Cost

Measured anchor: Denmark onshore, 4,866 turbines, 19 cluster counts by four
time slices, `PYVWF_OFFSET_WORKERS=4`, **1.5 hours**
(`method-cluster-count-dk.md`). The grids registered above are smaller and only
one time slice is fitted, so the earlier estimate of 20 to 28 hours was for a
larger design than this one.

83 fold-fits plus fourteen refits and fourteen test-year evaluations. **The
estimate is not reported here**, because one country row is run first for the
express purpose of measuring it, and an estimate written beside a measurement
that is about to arrive is an invitation to remember the wrong one.

## Committed in advance

- The k grids, the fold structure, the rule, the tiebreak and both baselines
  are fixed by this document and are not adjusted after a sweep is read.
- All fourteen configurations are reported, including any whose selection is
  the count it already had.
- A configuration is not dropped after its result is seen. A cluster count
  dropped for failing to fit is recorded with the reason.
- The test year is read once per configuration per candidate, after selection.
- If C-G2 fails, that is the finding.
