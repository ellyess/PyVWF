# Selecting each configuration's cluster count: registered design

**Date:** 2026-09-15. Registered before any sweep runs.
**Scope:** how many clusters each configuration contributes control points at,
when the control-point pool is rebuilt. **Since the 2026-09-15 amendment below,
the five turbine-level configurations only**; the nine country-level ones ask a
different question and are registered in
`method-national-single-cluster-prereg.md`. Terms follow `CONTEXT.md`.

## Amendment, 2026-09-15: forward chaining replaces leave-one-year-out

**Dated before the run it governs.** The protocol below holds out one training
year and rotates over all of them. That cannot be expressed against this
harness: `train_years` is an inclusive `[start, end]` pair, validated in
`vwf.harness.regions` and consumed as a range by the observation sources, so a
year held out from the middle of the window has no representation. Adding an
exclusion list to the config contract to make one study's protocol run is the
tail wagging the dog.

**Fold *i* therefore trains on a contiguous prefix of the training years and
validates on the next one.** For a window of 2015 to 2021 that gives six folds,
validating 2016 through 2021, each trained on everything before it. The test
year is untouched, as before, and the folds never train on years after the year
they validate, which leave-one-year-out does.

Folds per configuration become: six for BE, ES, FR, IT, NL, NO, PT and SE; four
for IE; four for DK-onshore and DK-offshore; three for DE-onshore, UK-onshore
and UK-offshore.

**Two conservatisms now compound, and this is stated here so it is not
discovered later.** Forward chaining gives the early folds less training data
than the late ones, which penalises a large cluster count where it has fewest
years to fit on. The one-standard-error rule already prefers the smallest
defensible count. Both push the same way, toward fewer clusters. **Where the
selections are reported, that has to be said beside them**, and a selection at
the bottom of its grid is not evidence that the bottom is best.

## Amendment, 2026-09-15: the country-level grid registered above cannot run

**Recorded, not resolved.** Belgium was run first to measure the cost and
exercise the protocol, and it raised instead:

> country-level run asked for 2 clusters but the grid points define 3. The
> country path does not cluster, so the two must agree.

`vwf.data.assign_country_clusters` accepts 1, or the number of clusters the
grid points already carry, and refuses everything else. **No clustering step
runs on the country path**: the grid points arrive with their `cluster` column
set, and for the zonal countries it holds the bidding zones. So the registered
country-level grid of 1 to 100 is not a grid this pipeline can fit, and each of
the nine country-level configurations has a two-member candidate set:

| BE | ES | FR | IE | IT | NL | NO | PT | SE |
|---|---|---|---|---|---|---|---|---|
| 1 or 3 | 1 or 4 | 1 or 10 | 1 or 3 | 1 or 3 | 1 or 5 | 1 or 5 | 1 or 3 | 1 or 4 |

Genuine cluster-count selection therefore exists for the **five turbine-level
configurations only**, where k-means runs: DE-onshore, DK-onshore, DK-offshore,
UK-onshore and UK-offshore.

What this does to the design:

- The turbine-level grids, the fold structure, the one-standard-error rule, the
  metric rule and both baselines are unaffected and stand.
- **C-P4 is retired**, not made untestable and left standing. Every
  country-level configuration selects below 10 by construction, so the
  prediction could no longer be wrong, and one that cannot be wrong is struck
  rather than scored.
- The question the study asks of the nine country rows shrinks to whether one
  national cluster beats the grid's own count, which is a narrower question
  than the one registered and is not the same as choosing a cluster count. It
  moves out of this document and is registered under its own name in
  `method-national-single-cluster-prereg.md`.
- Whether a country's count can be varied at all is a question about how the
  grid-point files are built, not about `cluster_list`.
  `scripts/region_tools/weight_country_grid_points.py` reweights an existing
  grid and re-derives zones from their labels; it does not create clusters.
  Nothing here investigates that further.

**This amendment records the constraint and changes no gate.** What replaces
the country-level half of the design is not decided in it.

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

Four of the five turbine-level rows have no usable sweep, and the fifth,
Denmark onshore, predates the per-timestep roughness treatment and is re-run
with them. The nine country rows have no sweep to run: see the second
amendment.

## The protocol

**Selection is nested inside the training years and never touches the test
year.**

1. For each configuration, fit on a contiguous prefix of the training years at
   every cluster count in the grid, and score on the next training year. (This
   replaces leave-one-year-out; see the amendment above.)
2. Rotate the prefix forward, giving one fold per training year after the
   first.
3. For each cluster count, take the mean fold score and its standard error
   across folds, `sd / sqrt(folds)` with `sd` the sample standard deviation.
4. **The one-standard-error rule:** let `k*` minimise the mean. The selected
   cluster count is the **smallest** cluster count whose mean is at or below
   `mean(k*) + SE(k*)`.
5. Refit at the selected count on **all** training years, and report on the
   single untouched test year.

Fold counts are in the forward-chaining amendment above.

**The standard error is estimated from three to six numbers and is itself
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

~~Country-level, capped at the number of grid points in that configuration's
fleet: 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100.~~ **Struck, 2026-09-15**;
see the second amendment.

Turbine-level onshore, capped at fleet size: **1, 10, 25, 50, 100, 200, 500,
1000**.

Turbine-level offshore, capped at fleet size: **1, 2, 3, 5, 10, 25, 50, 100**.

The country-level grid above is struck by the second amendment: the country
path admits two counts and neither is chosen from a grid. It is left in place,
marked, so that what was registered can be read against what replaced it.

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
  configuration. 100 is where `method-cluster-count-dk.md` says essentially all
  the skill is captured. **That value comes from test-year-selected curves**, which makes B2 a strong baseline rather
  than a fair protocol, and that is the right direction for a baseline: if
  per-country selection cannot beat a contaminated fixed rule, it has not
  earned its cost.

All three are evaluated the same way, by refitting on all training years and
scoring the single test year.

## Gates

| Gate | Requirement | Outcome |
|---|---|---|
| **C-G1** | The protocol completes for all five turbine-level configurations under both metrics, with no configuration failing to produce a selected count. A configuration that cannot be fitted at some cluster count has that count dropped from its grid, recorded, and the rule applied to what remains. | |
| **C-G2** | Per-configuration selection beats **B2** on test-year MAE, by more than 0.002, in at least three of the five turbine-level configurations. 0.002 is the screen this project already uses for a difference that is negligible in a capacity factor, and three of five is a simple majority. | |
| **C-G3** | Per-configuration selection beats **B1** on test-year MAE in at least three of the five. This is the weaker question, since B1 was not chosen by any protocol. | |

**If C-G2 fails, per-configuration selection is not adopted**, and the finding is
that one fixed rule is as good, which is worth the same as the opposite result
and is reported with the same prominence.

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| C-P1 | The one-standard-error rule selects a cluster count well below the minimising one in most configurations, because the curves are flat. Stated as: the selected count is below `k*` in at least three of the five. | |
| C-P2 | **C-G2 fails.** I expect per-configuration selection not to beat B2 by 0.002 in three of the five, on the Denmark evidence that the curve is flat over a factor of sixteen. This is the prediction that the rebuild is unnecessary, and it is stated so that it can be refuted. | |
| C-P3 | RMSE and MAE select the same cluster count in a majority of configurations, and where they disagree the test-year MAE difference between the two choices is below 0.002. | |
| ~~C-P4~~ | **Retired, 2026-09-15, not answered.** It predicted that every country-level configuration selects below 10. The country path admits only two counts, so the prediction became true by construction rather than by evidence, and a prediction that cannot be wrong is not one. The question it was reaching for is registered separately in `method-national-single-cluster-prereg.md`. | |
| C-P5 | The two offshore configurations select the smallest counts in their grids, having the fewest units and the least spatial spread. | |

## Cost

**Measured, Belgium, 2026-09-15**, at commit `4af1ba4` from a clean tree, both
legal cluster counts at the `fixed` slice on `era5/EU_2026-09` with the
per-timestep roughness: **42.0 s to train** (21.0 s per cluster count) and
**37.3 s to evaluate**, 79.3 s in total.
Run: `output/cluster_sweep_cost_2026-09-15/BE/`.

Earlier anchor: Denmark onshore, 4,866 turbines, 19 cluster counts by four
time slices, `PYVWF_OFFSET_WORKERS=4`, **1.5 hours**
(`method-cluster-count-dk.md`). The grids registered above are smaller and only
one time slice is fitted, so the earlier estimate of 20 to 28 hours was for a
larger design than this one.

The five turbine-level rows this document now covers are not costed by the
Belgian measurement: Belgium fits 44 grid points and Denmark onshore fits 4,866
turbines. From the Denmark anchor they are roughly **5 hours**, and that figure
is an extrapolation and not a measurement. The nine country rows cost roughly
**1.6 hours** on the Belgian measurement, and they belong to
`method-national-single-cluster-prereg.md`.

## Committed in advance

- The k grids, the fold structure, the rule, the tiebreak and both baselines
  are fixed by this document and are not adjusted after a sweep is read.
- All five turbine-level configurations are reported, including any whose
  selection is the count it already had.
- A configuration is not dropped after its result is seen. A cluster count
  dropped for failing to fit is recorded with the reason.
- The test year is read once per configuration per candidate, after selection.
- If C-G2 fails, that is the finding.
