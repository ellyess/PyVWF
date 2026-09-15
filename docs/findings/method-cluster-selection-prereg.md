# Selecting each configuration's cluster count: registered design

**Date:** 2026-09-15. Registered before any sweep runs.
**Scope:** how many clusters each configuration contributes control points at,
when the control-point pool is rebuilt. **Since the 2026-09-15 amendment below,
the five turbine-level configurations only**; the nine country-level ones ask a
different question and are registered in
`method-national-single-cluster-prereg.md`. Terms follow `CONTEXT.md`.

## Amendment, 2026-09-15: the curve shape is a reported result, not a means to a selection

**Written before the four remaining rows landed.** Only Denmark offshore had
run. Held outside the repository until the runs finished, because four
manifest-writing runs were in flight and a file created in the tree would have
stamped `git_dirty: true` into every manifest of all four.

**The flat-curve concern that motivated this protocol does not generalise, and
Denmark contradicts itself.** Denmark onshore moves 0.0857 to 0.0851 across a
sixteenfold increase in cluster count, which is the flat curve that made
minimum-picking look like noise-fitting and led to the one-standard-error rule.
Denmark offshore, the same country, has a mean spread of **20.8% in RMSE and
39.9% in MAE** across its grid, with `k=1` and `k=2` beating everything above
them in every fold and a four-count plateau from `k=10`. Those are opposite
shapes.

So **"select per row" is not noise-fitting where the curve has real
structure**, and whether it is noise-fitting is a property of the row, not of
the method. The one-standard-error rule still applies everywhere, because it
costs almost nothing on a structured curve and protects a flat one.

**Each row therefore reports its curve shape as a result in its own right**,
beside the selection it produced, on three statistics computed from the mean
fold score per cluster count:

- **spread**, the worst mean over the best mean, minus one;
- **counts within 1% of the best**, which is how many choices are effectively
  tied for first;
- **the largest plateau**, the biggest group of counts lying within 1% of that
  group's own floor.

The label follows from them rather than from a reading: **flat** below 5%
spread, **plateaued** where the largest plateau covers half the grid or more,
and **structured** otherwise. A row may be both plateaued and structured, and
Denmark offshore is: a sharp preference at the bottom and a plateau above
`k=10`. The statistics are reported whatever the label, because the label is a
convenience and the spread is the evidence.

**This changes no gate.** C-G1, C-G2 and C-G3 stand as registered, and the
curve shape is reported alongside them rather than feeding them.

### Added 2026-09-15T16:0x, after Denmark offshore and before the other four

Both additions are marked as later than the section above, which was fixed at
sha256 `75e7044e533fdabf0614c940e3c0cf557d82c16eeae31aa13f6ec3accd3f7ee3` when
only Denmark offshore had run.

**Each row reports the one-standard-error rule's own cost.** On a curve with a
clear top group and an unstable ordering inside it, the rule reliably takes the
smaller of two near-equals: Denmark offshore's minimising count was `k=2`, the
rule took `k=1`, and on the untouched test year `k=2` is better by 0.0019,
inside the 0.002 screen. Nothing is wrong there, and it is only visible because
the minimising count happened also to be the chapter's and so was already being
evaluated.

So the gap between the selection and the minimising count on the test year is
**reported for every row**, not only where it coincides with a baseline. **If
any row's gap exceeds 0.002 that is a finding about the rule, not about the
row**, and it is reported as such.

The final run of each row evaluates the selection and both baselines. Where the
minimising count is not among those three it is not evaluated, so this quantity
is obtained by one additional evaluation per affected row after the study
completes. That adds a reported number and touches no gate.

**The plateau above `k=3` has a candidate explanation, and it is testable.**
Denmark has two offshore wind farms. `k=1` and `k=2` therefore ask for clusters
that correspond to something; `k=3` and above do not. The step from `k=2` to
`k=3` is a 20.8% jump in mean RMSE which then flattens across `k=10` to `k=100`
into a four-count plateau, which is the shape of a partition that has stopped
carrying information and is dividing the same units into ever smaller groups,
each with too few to fit a stable scalar.

**Counted rather than assumed, before the United Kingdom's curve was known.**
Grouping each training fleet into spatial components linked at 5 km:

| | units | groups | group sizes |
|---|---|---|---|
| DK offshore | 318 | **9** | 162, 111, then 10, 10, 8, 7, 5, 3, 2 |
| UK offshore | 981 | **22** | 118, 116, 100, 80, 75, 60, 50, 43, 40, 36, 35, 32, 30, 30, and 8 more |

So "Denmark has two offshore farms" is too simple: it has nine groups, of which
**two hold 86% of the units** and seven hold 45 between them. The break at
`k=3` sits exactly where the partition runs out of large groups to separate,
which is a sharper version of the same claim.

**The prediction, fixed before the result.** The United Kingdom's 22 groups are
far more evenly sized, the largest holding 12% of units against Denmark's 51%.
If the mechanism is that a cluster count stops paying once clusters no longer
correspond to anything, **UK offshore should stay useful well past `k=3`, into
the region of 10 to 22, and its plateau should begin near its own group count
rather than near Denmark's.** If instead UK offshore also breaks at `k=3`, the
mechanism is wrong and the break is a property of the method, not of the fleet.
Either answer is worth more than the selection, because it would say what a
cluster count means rather than which one scores best.

## Registered before it runs, 2026-09-15: the offshore rows are reproduced, not re-run

**Denmark offshore's and United Kingdom offshore's run directories were
deleted, and the deletion was a mistake.** Clearing the two contaminated
regions, the whole of `output/cluster_selection_2026-09-15/DK/` and `UK/` was
removed, and those directories also held the two **clean** offshore rows'
manifests, factors and `metrics.csv`. Deleted by region rather than by run
directory name, which is the same class of error as resolving a file by listing
rather than by name (`AGENTS.md`). The rows' numbers survive in
`final_DK_offshore.csv`, `final_UK_offshore.csv` and their logs; the provenance
a manifest carries does not.

**Both rows are re-run as a reproduction, and the expected outcome is fixed
here before it runs.** Nothing in their inputs changed but the run name, so:

- **Bit-identical numbers restore the provenance and the study is unaffected.**
- **Any difference is a finding about determinism**, reported as such and
  investigated before the selections are trusted, because a pipeline that does
  not reproduce itself makes every figure in this document provisional.
- **Nothing about either selection can change.** Both are already recorded, the
  inputs did not move, and this is not an opportunity to re-select. If a
  selection moved, that is the determinism finding, not a new selection.

This is a reproduction and is **not part of the study**. It adds no row, reads
no gate, and its only outcome is whether the two rows come back identical.

## Defect, 2026-09-15: two fleet modes wrote to one run directory

**Found after the first run of all five rows, before any gate was read.**
`vwf.harness.driver._run_dir` keys a run directory on the region code and the
run name and on nothing else. This study varies the **fleet mode**, which the
path does not carry, and the runner's run name held only the fold year. So
`DK onshore` and `DK offshore` both wrote to `.../DK/train-fold-2016`, and the
same for the United Kingdom.

Two things turned that into a contaminated result rather than an error.
**`run_evaluate` scores every `factors_*.csv` it finds in the training
directory**, so the onshore folds scored their own eight counts plus the
offshore run's `k=2`, `3` and `5`, which are offshore factors applied to an
onshore fleet. And **running one row per process, which the memory rule asks
for, prevents none of this**: the rule is about processes and the collision is
in the path.

Nine runs hold factors they did not declare, all from this study: the four
Denmark onshore folds, the three United Kingdom onshore folds, and **both
`train-final` runs**, so the contaminated rows' test-year numbers are affected
as well as their fold scores. An audit of all 224 train runs in `output/`
against the `cluster_list` each manifest declares found **no earlier study
affected**: every one of them varied something it also put in the path, by run
name or by a condition directory above the region.

**UK onshore and DK onshore are re-run in full. DK offshore, UK offshore and DE
onshore are not**, being clean and registered to run once; re-running them for
uniformity would be re-running because results have been seen.

Fixed in the runner: the fleet mode goes in the run name, each row writes its
own score and selection files rather than one that a per-process loop
overwrites, and the evaluation refuses a metrics table holding counts the run
did not fit.

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
per-timestep roughness, **on the maintained fleet-weighted grid of 44 points**
(not the uniform 105-point grid the control-point pool was built from): **42.0 s
to train** (21.0 s per cluster count) and **37.3 s to evaluate**, 79.3 s in
total. The run's skill figures are on that grid too and are recorded in
`../design/manuscript-chapters-45.md` under T9.
Run: `output/cluster_sweep_cost_2026-09-15/BE/`.

Earlier anchor: Denmark onshore, 4,866 turbines, 19 cluster counts by four
time slices, `PYVWF_OFFSET_WORKERS=4`, **1.5 hours**
(`method-cluster-count-dk.md`). The grids registered above are smaller and only
one time slice is fitted, so the earlier estimate of 20 to 28 hours was for a
larger design than this one.

**Measured, all five rows, 2026-09-15: 2 hours 6 minutes**, against the 5-hour
extrapolation, which was 2.4 times too high.

| Row | units | grid | folds | minutes |
|---|---|---|---|---|
| DK offshore | 318 | to k=100 | 4 | 15.1 |
| UK offshore | 981 | to k=100 | 3 | 16.8 |
| DK onshore | ~4,900 | to k=1000 | 4 | 25.3 |
| UK onshore | ~4,800 | to k=1000 | 3 | 33.2 |
| DE onshore | ~4,800 | to k=1000 | 3 | 35.8 |

**Fixed cost dominates, not fleet size**, and the next study should estimate on
that basis. United Kingdom offshore at 981 units took 16.8 minutes against
Denmark offshore's 15.1 at 318, a threefold fleet for 11% more time, because
each fold pays for an ERA5 preparation and a wind interpolation whatever the
fleet. The onshore rows cost more for their grids reaching `k=1000`, not for
their fleets. The nine country rows cost roughly
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
