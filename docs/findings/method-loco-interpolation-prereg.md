# Leave-one-country-out for the interpolators: registered design

**Date:** 2026-09-13, committed before any fold is scored.
**Scope:** whether IDW, ordinary kriging, RBF and nearest neighbour generalise
across country borders, tested on the same holdouts chapter 5 used for its
machine learning. Terms follow `CONTEXT.md`.

**Everything below is fixed before any number exists.** The outcome column is
filled afterwards.

## Why this exists

The merged manuscript's question is whether correction factors generalise
across country borders. Chapter 5 tested that with leave-one-country-out and
found collapse: Random Forest reaches an R-squared of 0.019 on the Germany
holdout and -0.376 on the United Kingdom. **Chapter 4 never ran a country
holdout at all**, in its text or its code; its `spatial_cv_split` sorts control
points by longitude and cuts five contiguous bands.

So the two methods were judged at different rigour, and the harder test is the
one the machine learning failed. Until the interpolators face the same test,
any statement that interpolation generalises better than machine learning is a
comparison between a gentle test and a harsh one.

## The folds

**Twelve, one per country, onshore and offshore combined**, on the 1,729
centroid-level control points of `all_corrections_centroids.csv`. That is the
same pool chapter 4 interpolates and the same pool chapter 5 trains on at
centroid level, so the two are directly comparable without rescoring either.

The fold definition is recovered from chapter 5's own reported test-set sizes:
Germany 500, which is its DE onshore centroid count, and the United Kingdom
303, which is UK onshore 293 plus UK offshore 10. Combining the modes is
therefore what chapter 5 did, and this follows it rather than inventing a
partition.

Folds: DE (500), DK (886), UK (303), FR (10), NL (5), NO (5), ES (4), SE (4),
BE (3), IE (3), IT (3), PT (3).

## The two decisions, fixed here

**The Netherlands fold is scored, and it is the primary result.** Holding out
the Netherlands removes the five national clusters and leaves the correction to
be predicted from German and Belgian control points, which is precisely the
cross-border transfer the manuscript is about. It is also the case chapter 4
already reports as its best cross-border outcome, at a gridded MAE of 0.056
against a cluster-based 0.116, so scoring it here places that result on the
same footing as everything else. **The ENTSO-E coverage defect capping Dutch
capacity factor at 0.57 is stated wherever the Dutch number appears.**

**A fold whose own country supplies the only nearby control points is scored,
not excused.** IDW falls back on whatever remains within its weighting range,
which for a peripheral country may be several hundred kilometres away, and the
error that produces is the answer to the question rather than an obstacle to
it. Two quantities are reported beside every fold so a reader can tell a hard
fold from a broken one: the distance from each held-out point to the nearest
remaining control point, and the share of the fold's points beyond 5 degrees,
which is the threshold the IDW product uses to mask a cell as unconstrained.
**No fold is dropped for being hard**, since dropping the hard folds is how a
transfer result gets flattered.

## The metric

**Mean absolute error is primary, with R-squared reported beside it.**

Chapter 4 reports MAE and chapter 5 reports R-squared, and the merged paper
needs one answer. MAE is primary because R-squared on a three-point fold, which
six of the twelve are, is not a usable number: it is a ratio to the variance of
the held-out points themselves, and three points from one country have almost
none. R-squared is still reported, because chapter 5's headline is stated in it
and dropping it would make the comparison with the machine learning impossible.

**A negative R-squared is a result and not a failure to be hidden**, and
chapter 5's own leave-one-country-out figures are negative.

Scalar error is computed in **both** linear and log space and both are
reported. The chapter-4 defect recorded in the manuscript decisions document is
the reason: its prose specifies log and its code computes linear, and this
study will not inherit that ambiguity silently.

## What runs

All twelve folds, for nearest neighbour, IDW, ordinary kriging and RBF.
**Every fold is scored: no subset.** The cost is seconds to minutes, and
testing a subset chosen after seeing which folds look interesting is the error
recorded under D5 of the manuscript decisions document, where a single row was
picked because it was the most confirming and gave the opposite of the right
answer.

The interpolation functions are the chapter's own, ported from
`development:scripts/pyvwf_to_grid/compare_unified_corrections_to_grid.py`
(`interpolate_idw_point`, `interpolate_kriging_points`,
`interpolate_rbf_points`, `interpolate_nearest`). A second implementation would
be a second definition of the same number.

## Gates

| Gate | Requirement | Outcome |
|---|---|---|
| **L1** | The interpolators survive the test the machine learning failed: at least one interpolation method has a lower MAE than the best machine-learning model on at least 8 of the 12 folds, scored on the same held-out points. | |
| **L2** | Cross-border transfer works at all where a country has neighbours: for the folds whose held-out points have a remaining control point within 2 degrees, the best interpolator's MAE is below the MAE of predicting every held-out point at the pool's mean. Below that, distance-weighted interpolation is adding nothing over a constant. | |
| **L3** | The Netherlands result survives its own holdout: NL's fold MAE under the best interpolator is lower than the MAE of the country-only cluster correction chapter 4 reports for it. **Indeterminate** if the two are not on the same observations, which is checked before the gate is read. | |

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| N1 | L1 passes. Interpolation degrades under country holdout, as the machine learning did, but less, because it has no region-specific decision boundaries to memorise. | |
| N2 | Every method's MAE is worse under leave-one-country-out than under the longitude folds chapter 4 used, in at least 10 of the 12 folds. The longitude bands cut through countries, so they leave same-country neighbours in the training set. | |
| N3 | The spread across methods narrows under this test relative to the longitude folds. With no same-country neighbours, the choice of weighting matters less than the absence of nearby data. | |
| N4 | L3 holds: the Netherlands keeps its advantage, because its neighbours are the densest part of the pool. | |

## Committed in advance

- All twelve folds are reported, including any that are embarrassing.
- The fold sizes, the nearest-remaining-control-point distances and the shares
  beyond 5 degrees are reported whatever the scores say.
- A method is not dropped after its result is seen.
- This design is not revised after any fold is scored; a change is recorded as
  a dated deviation with whether it came before or after a result was seen.
