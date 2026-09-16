# Leave-one-country-out for the interpolators: registered design

**Date:** 2026-09-13, committed before any fold is scored.
**Scope:** whether IDW, ordinary kriging, RBF and nearest neighbour generalise
across country borders, tested on the same holdouts chapter 5 used for its
machine learning. Terms follow `CONTEXT.md`.

**Everything below is fixed before any number exists.** The outcome column is
filled afterwards.

## VOID, 2026-09-16: the study ran and its gates cannot be read

**The run happened and its scores stand. The gates do not, and they were
unreadable from the day they were registered.** Scores:
`output/loco_2026-09-13/loco_scores.csv` and `loco_fold_geometry.csv`, twelve
country folds by four interpolators by two distance metrics.

Per gate:

- **L1 is unreadable: the comparator does not exist.** It requires the best
  machine-learning model's mean absolute error **on the same held-out points**.
  The only machine-learning result on disk,
  `output/ml_retest/expanded_loro_scalar.csv`, holds out eight **regions**,
  four of them non-European, against this study's twelve European
  **countries**. Only DE, DK and UK appear in both, and even there the held-out
  sets differ: the machine-learning work used 100 centroids per region while
  this holds out the pool's own 500, 886 and 303. No score on these folds can
  be produced until the machine-learning module is ported, which is phase 3 and
  has not started.
- **N1 is contingent on L1** and goes with it.
- **N2 and N3 are unreadable: the comparator has no matching unit.** Both
  compare against the longitude folds chapter 4 used. Those are longitude
  bands, not countries, so "in at least 10 of the 12 folds" does not map onto
  them, no per-country figure exists to compare with, and the chapter's
  published longitude-band numbers are uniform-grid baselines, which
  `../design/manuscript-chapters-45.md` D0 has since made historical. No
  longitude-band run exists under `output/`.
- **L2 is readable and its restriction is ambiguous.** The pool-mean baseline
  needs no choice. But the gate restricts to the folds "whose held-out points
  have a remaining control point within 2 degrees" without saying whether that
  means every held-out point, the median one or any one, and
  `loco_fold_geometry.csv` records `km_to_nearest_min`, `median` and `max` plus
  `deg_to_nearest_median`, so all three readings are available and select
  different fold sets. **Choosing now is choosing with the scores in view**, so
  it is not chosen.
- **L3 and N4 stay retired** with their dated reason, below.

**The scores stand as a measurement.** The same twelve folds were rerun on the
reference-wind target and every number is reported in
`method-why-corrections-do-not-transfer.md`, which is where this result now
lives. Nothing is lost except the gated reading.

### What a re-registration would need, not done here

- **L1** waits on port phase 3, so that a machine-learning model can be scored
  on these folds and these held-out points.
- **N2 and N3** need longitude-band runs on the maintained grids, so the
  comparison is within one grid and one fold definition.
- **L2** needs its statistic fixed in advance: which distance, over which
  points.

Whether these questions are still worth asking is for a later session, after
the machine-learning module exists. **They are not re-registered now**, because
re-registering a gate whose comparator still does not exist would repeat the
error that voided this one.

## Amendment, 2026-09-15: the chapter comparison is dropped, and this re-runs on the rebuilt pool

The workstream's founding assumption
(`../design/manuscript-chapters-45.md`, D0) is that country-level results are
computed on the maintained fleet-weighted grids, and that the chapter's
figures, computed on the uniform grids, are a historical baseline rather than a
target. This study reads the control-point pool, which is built from the
uniform grids, so:

- **The result already produced stands as a finding about the chapter's pool**,
  and is labelled that way wherever it appears. It is not a statement about the
  pool the manuscript will ship.
- **L3's comparison against a chapter figure is retired**, not scored. It set
  the Netherlands fold against the country-only cluster correction chapter 4
  reports, which is a uniform-grid number, and the fold under a rebuilt pool
  would be a maintained-grid one. Comparing them would be comparing two fleets.
  The question L3 asked stays open and needs a within-grid form.
- **The study re-runs on the rebuilt pool once it exists.** L1 and L2 carry
  over unchanged: both are internal comparisons between methods, or between a
  method and the pool's own mean, and neither reads a chapter figure.

  *[Parked 2026-09-16: no rebuilt pool is being built, so this re-run has no
  pool to run on. The selection study's counts are not adopted for a pool, the
  country tier cannot be selected while the national study is blocked on
  training windows, and the transfer synthesis removed the pool as the leading
  suspect. See the dated note under D0 in
  `../design/manuscript-chapters-45.md`. The study is also void on its own
  gates, above.]*

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

*[Amended 2026-09-13, before any fold was scored. **Distance is great-circle
for this study, with Euclidean degrees reported beside it.** The chapter
measures IDW, nearest neighbour and RBF in Euclidean degrees and kriging in
great-circle, which stretches the first three east to west by about a factor of
two at 60 degrees north, on the axis this study's country borders run across.
Measured on the chapter's own folds before this was decided, great-circle
improves IDW by 1.3% on both targets and changes no ranking (T7 of the
manuscript decisions document). The split follows D1: reproduction keeps the
chapter's metric, new work uses the defensible one, and this study is new work.
RBF is the exception and is reported as one: `scipy`'s interpolator fits on
coordinates directly and has no metric option, so it runs in degrees under both
columns.]*

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
| ~~**L1**~~ | **VOID 2026-09-16, unreadable.** No machine-learning score exists on these folds and none can be produced until port phase 3. See the void notice. | |
| ~~**L2**~~ | **VOID 2026-09-16, not chosen.** Readable, but the 2-degree restriction never said which statistic it applies to and three were recorded, so choosing now would be choosing with the scores in view. See the void notice. | |
| ~~**L3**~~ | **Retired 2026-09-15, not scored.** It set NL's fold MAE against the country-only cluster correction chapter 4 reports, which is a uniform-grid figure, while a rebuilt pool is maintained-grid. The two are different fleets and the comparison cannot be read. The question stays open and needs a within-grid form; see the amendment above. | |

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| ~~N1~~ | **VOID 2026-09-16**, contingent on L1. | |
| ~~N2~~ | **VOID 2026-09-16, unreadable.** Longitude bands have no per-country figure to compare against. | |
| ~~N3~~ | **VOID 2026-09-16, unreadable.** Same comparator as N2. | |
| ~~N4~~ | **Retired 2026-09-15 with L3**, which it predicted. | |

## Committed in advance

- All twelve folds are reported, including any that are embarrassing.
- The fold sizes, the nearest-remaining-control-point distances and the shares
  beyond 5 degrees are reported whatever the scores say.
- A method is not dropped after its result is seen.
- This design is not revised after any fold is scored; a change is recorded as
  a dated deviation with whether it came before or after a result was seen.
