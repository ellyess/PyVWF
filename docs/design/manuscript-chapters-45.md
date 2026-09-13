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

## Resolved: the chapter's prose and its code disagree about the scalar error

**Chapter 4's methods section says the scalar error is evaluated in log space,
"for symmetry around unity", and its code computes it in linear space.** The
published figures are the linear ones:

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

So the chapter is reproducible and its stated method is not what it did. This
matters for the manuscript in three ways. Anyone reproducing from the prose
lands 13% high. The stated justification, symmetry around unity, was never
applied, and the scalar distribution is asymmetric about 1, running from 0.216
to 4.644, so linear absolute error weights the upper tail more heavily than the
lower. And the merged manuscript has to say which space it uses and use it.

It no longer blocks anything. It is a known, quantified difference.

## Live defect: the offshore shapes overlap each other

The spatial-join classification ported on 2026-09-13 had a defect where a point
falls inside two overlapping polygons **of the same file** (see the commit for
`src/vwf/geospatial.py`). That needs internally overlapping shapes to bite, and
the project's shape files are not alike in this:

| File | Polygons | Genuinely overlapping pairs |
|---|---|---|
| `country_shapes.geojson` | 25 | **0** (shared borders only) |
| `offshore_shapes.geojson` | 19 | **44** |

So the onshore side was never exposed and the offshore side was. Where that
reaches chapter 4's published numbers depends on what the grid export does with
the classification, which is answered when `extensions/grid` lands.

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
