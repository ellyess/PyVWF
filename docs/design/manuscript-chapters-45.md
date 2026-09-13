# Merging thesis chapters 4 and 5: what has to be decided

**Date:** 2026-09-13
**Status:** open. Nothing here is decided, and nothing has been ported or run.
**Scope:** the manuscript merging the gridded-interpolation chapter and the
machine-learning chapter, co-authored, drafted against results produced from
this repository as it stands. The chapters themselves are accepted, on Spiral,
read-only and outside this repository. Terms follow `CONTEXT.md`.

This document records the decisions the merge needs and the evidence each one
rests on, so that they are made once and on purpose. Phase 0 inventoried what
the chapters did; phase 1a examined the control points they rest on; phase 1b
surveyed what porting their code would take.

## The shape of the problem

**The chapters' code is not in this repository.** It was removed from `main` on
2026-07-06 (commit `e8208e1`, "Scope reduction (scripts): move grid/ml driver
scripts to development") and survives on the local `development` branch, whose
head `f2ed688` of 2026-06-17 is 315 commits behind the research branch. What
this branch implements of chapter 4 is nearest-centroid assignment, which is
step 2 of the chapter's eight; `src/vwf/harness/export.py` says so in its own
docstring, naming IDW as "a future refinement, not a claim this file makes".

So this is a port, not a re-run, and the manuscript's first decision follows
from that.

## The two purposes, which must not be merged

The chapters answer one question and deliver one artefact, and they are
different things with different standards.

**The question is whether correction factors generalise across country
borders.** Interpolation was the first attempt and terrain-informed machine
learning the second, so the merged manuscript's spine is cross-border
generalisation tested the same way for both. A cross-validation score is the
right evidence for it.

**The artefact is a gridded correction field for `atlite` and PyPSA-Eur.** A
file feeding an energy system model has to be right everywhere it is read, not
on average: its coverage, its masking and its neutral-value behaviour are
properties of the product, and no cross-validation score tests them. The IDW
product neutralises about 35% of the European domain beyond 5 degrees from any
control point, which is a headline property of that file rather than a
footnote to a method comparison.

Keeping these apart decides what evidence each part of the manuscript needs.

## G1. Chapter 4 never ran a country holdout, and that is the gap

Chapter 5 ran leave-one-country-out and reported collapse: Random Forest
falls to an R-squared of 0.019 on the Germany holdout and -0.376 on the United
Kingdom, where Ridge alone stays positive at 0.091. **Chapter 4 has no country
holdout at all**, in its text or in its code: `spatial_cv_split` in
`compare_unified_corrections_to_grid.py` sorts by longitude and cuts five
contiguous bands, and nothing else is offered.

So the two chapters answered the same question at different rigour, and the
harder test is the one that destroyed the machine learning. **A merged
manuscript whose spine is cross-border generalisation has to put IDW and
kriging through the same country holdouts**, or it compares a method tested
gently against one tested harshly and reports the difference as a finding.

What that takes is in "Running leave-one-country-out for the interpolators"
below. It is cheap.

## D1. Does the manuscript reproduce the chapters or supersede them?

Reproducing means porting about 3,000 lines of library code and 5,200 lines of
driver scripts to run against today's pipeline, and accepting that the numbers
will move because the pipeline has changed underneath them. Superseding means
taking the chapters' questions and answering them on today's rows, with the
chapters cited for what they claimed.

Evidence that bears on it is in D5 and D6: the turbine-level half of the
control-point pool is stable to almost everything that has changed, and the
country-level half is not comparable without more work.

## D2. The Netherlands is the existence proof, not an example

Chapter 4's Dutch result is the one cross-border generalisation result that has
survived scrutiny here. With five national clusters of its own, the country-only
correction reaches an MAE of 0.116; the kriging grid, borrowing from German and
Belgian control points, reaches 0.056. Every Dutch centroid's five nearest
control points are German onshore or Belgian, 15 to 153 km away, **and none of
them is degenerate**, so the result does not rest on the five artefacts in the
pool.

If the manuscript's spine is cross-border generalisation, this is the central
result and should be treated as such rather than listed among examples.

**The coverage defect travels with it, everywhere it appears.** This project
excludes the Netherlands: `CLAUDE.md` records an ENTSO-E coverage defect that
caps the Dutch capacity factor at 0.57, which no rescaling fixes. A result built
on observations that cannot exceed 0.57 is not disqualified by that, but it
cannot be quoted without it, and the manuscript has to state it at each
appearance rather than once in a limitations section.

## D3. Cluster counts

The chapter's control points are DK onshore 884, DE onshore 500, UK onshore
293, and its validation tables quote best counts of DK 700, DE 500, UK 300.
Today's scorecard rows are DK k=100, DE k=100, UK k=50. **Reproducing the
chapter and using today's rows are different papers**, and the difference is
not cosmetic: the number of clusters sets the spatial density of the control
points, which is the independent variable the whole interpolation argument
rests on.

## D4. Per-farm against per-turbine: settled from the data

Chapter 4's text says Germany and the United Kingdom are per-farm and Denmark
per-turbine; its Table 1 labels all three "Per-turbine capacity factors". **The
text is right and the table is wrong**, and the register settles it:

| Row | Register rows | Distinct observation units | Turbines per unit | Median unit capacity |
|---|---|---|---|---|
| DK | 5,618 | 5,618 | 1 | 660 kW |
| DE | 10,889 | **1,162 plants** | 5 (median) | 5,925 kW |
| UK | 6,618 | **360 accreditations** | 11 (median) | 16,200 kW |

German IDs are `<plant> <unit>` and British ones `<accreditation>-<index>`, so
the metadata is per turbine in all three while the observations are per plant
for Germany and per accreditation for the United Kingdom. Independent
confirmation: the British evaluation scores 348 units against a fleet of 5,998,
which is the accreditation count less those with no observation in the test
year, and not a turbine count.

The manuscript should state this once, whatever else is decided.

## D5. The country-level tier is real-curve-shaped, not fallback-shaped

The concern was that the 40 country-level control points are artefacts of the
fallback curve: the curve library study established that a country row whose
grid names a curve the library lacks has every unit simulated on a 167 W/m2
fallback, and that the fitted wind scalar absorbs the mismatch while staying
inside the plausible band, so no degeneracy rule catches it.

**It is not what happened.** Comparing the chapter's mean scalar per row with
today's two fits, at the same cluster count and the same fixed time slice:

| Row | Chapter | C0, fallback | C1, real curves | chapter / C0 | chapter / C1 |
|---|---|---|---|---|---|
| BE | 0.934 | 0.450 | 0.826 | 2.08 | 1.13 |
| FR | 1.357 | 0.807 | 1.520 | 1.68 | 0.89 |
| IE | 1.220 | 0.677 | 0.929 | 1.80 | 1.31 |
| SE | 1.234 | 0.659 | 1.342 | 1.87 | 0.92 |
| ES | 0.957 | 1.037 | 1.586 | 0.92 | 0.60 |
| IT | 1.665 | 1.523 | 3.083 | 1.09 | 0.54 |

Across the four rows with no other known defect, **the chapter's scalars match
today's real-curve fit to a median ratio of 1.02**, range 0.89 to 1.31, mean
absolute deviation 0.158; against the fallback fit the median ratio is 1.84 and
the deviation 0.858. Spain and Italy are the outliers in both directions, and
they are two of the three rows whose published winds were extrapolated up to
five degrees past the ERA5 data (`method-eu-rerun.md`), which is an independent
reason for them to sit apart.

Matching the cluster count and the time slice moved the figures by at most 0.10,
so neither was a confound. Training years are identical, 2015 to 2021. What
remains is the ERA5 extent and the roughness treatment, and for the four clean
rows those leave a 2% median discrepancy, which is small enough that the tier
should be treated as real-curve-shaped.

**How this was got wrong first, and the correction.** Italy was chosen as the
single test row because its statistic was the most fallback-like of the eight.
That is selection on the outcome: the row most likely to confirm the hypothesis
was tested, it did confirm it, and the conclusion would have been the opposite
of the truth. Extending the same test to every comparable row, and then setting
aside the two rows with an independent known defect, reverses it. Italy is also
one of those two.

**Nothing from the chapter era is attributable to a code state.** The
chapter-era outputs carry no run manifests: no version, no commit, no
`git_dirty`, no curve library sha256. So the library those runs used cannot be
read from provenance, and the conclusion above is an inference from the
numbers. Anything the manuscript says about how the chapter's figures were
produced rests on that inference and should say so.

## D6. The turbine tier reruns cleanly, and that is a finding

| Row | Chapter mean scalar | Today's mean scalar | Chapter clusters | Today's k |
|---|---|---|---|---|
| DK onshore | 0.789 | 0.780 | 884 | 100 |
| DE onshore | 0.916 | 0.900 | 500 | 100 |
| UK onshore | 0.974 | 0.882 | 293 | 50 |

**Across an eight-fold change in cluster count, three years, a rebuilt
pipeline, a changed roughness treatment and a re-run ERA5 archive, the mean
fitted scalar of each turbine-level row moves by 0.009, 0.016 and 0.092.** The
turbine-level tier is 1,689 of the 1,729 control points and carries one
degenerate fit between them.

This is the strongest evidence the manuscript has that the turbine half of
these chapters can be reproduced on today's code and give the same answer. It
also bounds what a re-run can be expected to change: not the corrections
themselves, but what is built on top of them.

## T1. A methods-section defect in chapter 4: log stated, linear computed

**Chapter 4's methods section says the scalar error is evaluated in log space,
"for symmetry around unity", with the equation written out; its code computes
it in linear space, and the published numbers are the linear ones.**

```python
scalar_mae = np.abs(scalar_pred - scalar_true).mean()
```

Reproducing the chapter's arithmetic on its own control-point table, with its
own fold construction, matches every published IDW figure:

| Quantity | Reimplementation, linear | Published | Reimplementation, log |
|---|---|---|---|
| scalar MAE | 0.1607 | 0.1610 | 0.1826 |
| scalar MAE, fold sd | 0.0610 | 0.0610 | 0.0547 |
| scalar RMSE | 0.2470 | 0.2470 | 0.2490 |
| offset MAE | 0.6408 | 0.6410 | 0.6408 |

This is a defect in an accepted chapter, and **nothing is fixed in the thesis**.
It is recorded because the stated justification was never applied and anyone
reproducing from the prose lands 13% high.

**Chapter 5 does not share it.** It makes no log-space claim anywhere, and the
driver that produced its headline tables, `run_turbine_model_comparisons.py`,
does not pass the `log_target` flag, which defaults to false. So chapter 5 is
internally consistent and linear. **Both chapters' published scalar figures are
therefore in the same space**, which is what makes their numbers comparable at
all, and it is comparability the merged manuscript needs.

## T2. The metric is the manuscript's decision, not an inheritance

Linear absolute error on a scalar running from 0.216 to 4.644 weights the upper
tail, and the upper tail is where the country-level tier and all five
degenerate points sit. Log error weights proportional departures from unity
equally in both directions, which is what the chapter said it wanted.

The paper has to state which it uses and why. **If it uses log, chapter 4's
scalar figures move**: IDW from 0.161 to 0.183, and every method's figure with
it, so the comparison table is restated rather than quoted. If it uses linear,
it says so and notes that the chapter's stated justification does not describe
the chapter's numbers.

## T3. The PyPSA-Eur deliverable's own defect: the shipped grids are not what the tables measured

**This is separate from the manuscript and it is the more serious of the two.**
The chapter ships two NetCDF files for `atlite` and PyPSA-Eur, and they are not
built the way the grids its tables evaluate are built.

`generate_best_correction_grids.py`, which produces the shipped files,
interpolates **all 1,729 control points as one field with no onshore and
offshore split**. Both files record `n_control_points: 1729` and neither
carries a domain attribute. `compare_unified_corrections_to_grid.py`, which
produces the grids behind Tables 6 and 7, splits the pool by `cluster_mode`
first and interpolates each domain separately.

So the artefact and the evaluation are different objects. **The shipped file was
never the thing the tables measured**, and anyone regenerating from the shipped
pipeline gets a third object again. The labelling problem below and the offshore
pool question are both downstream of this one.

**The labelling problem.** Chapter 4 evaluates a hybrid kriging, each target on
its own cross-validation-optimal configuration, and rejects it at 4 improvements
against 9 degradations. It adopts "OK, exponential, geographic", and describes
the shipped file as "standard kriging with variance-based masking". The shipped
file is the hybrid: `europe_corrections_kriging_best.nc` carries `method:
kriging_hybrid`, `scalar_variogram: spherical`, `scalar_coordinates: euclidean`,
`offset_variogram: linear`, `offset_coordinates: geographic`.

**It is a prose defect and not a results defect**, which was worth establishing.
Tables 6 and 7 match `grid_comparison/europe_corrections_kriging.nc`, the
standard configuration, at Denmark offshore 0.1113 against a published 0.111.
The hybrid tables match the shipped file at 0.1229 against a published 0.123. So
the chapter's numbers are sound and its description of the file it ships is not.

**Why this product needs its own standard.** A cross-validation score says
nothing about whether a file is right everywhere it is read. Coverage, masking
and neutral-value behaviour are the properties that matter for a file feeding an
energy system model, the IDW product neutralises about 35% of the European
domain beyond 5 degrees from any control point, and none of that is tested by
the method comparison the chapter uses to choose between IDW and kriging.

### Where it has gone, and the hold

| Question | Answer |
|---|---|
| Committed or published | **No.** Untracked in every branch here and on `development`; `bias-extra/` is untracked in the consuming repository too. Not in the Zenodo deposit, which archives the git tree while `output/` is ignored. |
| Copied | **Yes**, into `pypsa-eur-wind/bias-extra/` and `pypsa-eur-wind-archive/bias-extra/`, sha256 `f7165b26...`, byte-identical to the source. |
| Wired in | **Yes.** A patched atlite maps the keyword `kriging` to it, and `config/scenarios-validation.yaml` selects `bias_corr: kriging` in four scenarios. |
| Any result from it | **No.** Every bias-corrected result is `biasidw`. No `biaskriging` output exists. |
| The IDW file | **Correctly described**: `method: idw`, `idw_power: 2.0`, `max_distance_deg: 5.0`, which is what the chapter adopts, and it is the file behind the thesis PyPSA-Eur results. |

**No correction notice is issued.** Nothing is published, nothing has run from
it, and the file that produced the thesis results is correctly described. The
notice becomes due the moment either changes.

**The four kriging scenarios must not be run before this is settled.** They are
`base-s100000-biaskriging` and its siblings in
`pypsa-eur-wind/config/scenarios-validation.yaml`, and running one would
produce a result from a file that is neither the chapter's standard kriging nor
the split pipeline its tables evaluate.

**That config is shared.** It is tracked and its commit is contained in
`origin/master` of `github.com/ellyess/pypsa-eur-wind`, which is **public**, a
fork of PyPSA/pypsa-eur with no forks of its own. The mitigation is that
`bias-extra/*.nc` is untracked, so a clone gets the scenario definition and not
the grid, and the run fails on a missing file rather than silently using a
mislabelled one. Nothing has been edited in that repository from here.

## T4. The spatial-join defect was latent, and where it could have bitten

The defect fixed in the `src/vwf/geospatial.py` port needs polygons that
overlap **within one file**, and the project's two files are not alike:

| File | Polygons | Genuinely overlapping pairs |
|---|---|---|
| `country_shapes.geojson` | 25 | **0**, shared borders only |
| `offshore_shapes.geojson` | 19 | **44** |

So it could only ever have reached the offshore side. That is **12 of the 1,729
control points**, the turbine-level offshore clusters of Denmark and the United
Kingdom, and not the 40-point country-level tier, whose points carry
`cluster_mode = "all"` and which the chapter's own code assigns to onshore.

**It reached nothing.** Run on the real 1,729 points with the project's shape
files, the ported and original implementations agree on every one: 1,687
onshore, 32 offshore, 10 unknown, zero disagreements. And the published grid
files record `n_control_points: 1729` with no domain split at all, so
`export_pyvwf_grid`, which is the only caller of the classification, is not
behind any published figure. The defect is real, reproducible on constructed
input, and latent on this one.

**A separate disagreement is not latent.** The pool's declared `cluster_mode`
and the shape classification differ on **30 of 1,729 points**: 19 clusters
declared onshore fall inside offshore shapes (11 Danish, 8 British), one
declared offshore falls onshore, and 10 fall outside both files and are
unknown. Any code path that splits by shapes rather than by the declared mode
therefore kriges a different offshore pool from the one the chapter describes:
32 points rather than 12, and Denmark's offshore pool would not be the two
points its documented failure case rests on. Which pool a ported
`export_pyvwf_grid` should use is a decision, not a detail.

## Running leave-one-country-out for the interpolators

Chapter 5's holdouts are defined by its own reported test-set sizes: Germany
500 samples, which is the centroid count for DE onshore, and the United Kingdom
303, which is UK onshore 293 plus UK offshore 10. So the folds are **by country
with onshore and offshore combined, on the 1,729 centroid-level points**, which
is directly comparable to chapter 4's control-point pool because it is the same
pool.

What it takes:

- **Fold definition**: group the pool's `country_code` on its prefix, giving 12
  folds (DE, DK, UK and the nine country-level rows). Two decisions inside it:
  whether the Netherlands fold is scored at all, since NL is the existence
  proof and holding it out is exactly the cross-border test; and what to do
  with a fold whose own country supplies the only nearby control points, where
  IDW falls back on whatever remains within its weighting range.
- **Implementation**: none new. IDW and nearest neighbour are ten lines each,
  kriging is `pykrige` 1.7.3, already installed. The chapter's own
  `interpolate_idw_point`, `interpolate_kriging_points` and
  `interpolate_rbf_points` arrive with the grid port and should be used instead
  of a second implementation.
- **Metric**: chapter 5 reports R-squared and chapter 4 reports MAE. Both have
  to be computed on the same folds for the comparison to mean anything, and
  R-squared on a 12-point fold is unstable, which is worth registering before
  the numbers exist rather than after.
- **Cost**: seconds to minutes for all methods and folds. It is cheap enough
  that there is no reason to test a subset of countries, which is the mistake
  recorded under D5.

Not run.
