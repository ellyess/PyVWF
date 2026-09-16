# Why correction factors do not transfer: four eliminations and one survivor

**Date:** 2026-09-16
**Scope:** why a bias correction fitted in some countries fails to predict the
correction in an unseen one, whether by spatial interpolation (thesis chapter
4) or by a learned model (chapter 5). Terms follow `CONTEXT.md`.

**Four candidate explanations have been tested and none survives.** It is not
sample count, not the target's conditioning, not the identifiability defect in
the fitted pair, and not regime coverage. What remains is the reading
`method-ml-transfer.md` reached from a different direction and stated as a
verdict: **the correction factor is substantially a reanalysis-resolution
artefact rather than a transferable physical property.** That is now supported
from two independent directions and it is falsifiable, which the four
eliminated explanations no longer are.

**The strongest single result is that regime coverage fails in the direction
nobody would have predicted.** Folds further outside the covered feature space
do marginally better, not worse.

## The eliminations, in the order they were tested

### 1. Sample count. Ruled out.

**Predicted:** a denser pool trains a better model, so the 1,729 control points
should beat a thinner one.

**Measured:** the development-branch experiment trained on 1,474 Europe-only
centroids and scored leave-one-country-out R-squared of -0.10 to -1.91; the
retest on 100 centroids per region scored 1 of 5 positive
(`method-ml-transfer.md`). Denser was worse. Separately, Denmark onshore's 884
points occupy 340 cells of a coarse terrain-feature grid against 200 points
occupying 114, so 4.4 times the points buys 3.0 times the coverage and carries
2.6 points per occupied cell against 1.75.

**Rules out:** more of the same points. It does **not** rule out that thinning
is free, which is a different claim: 200 points hold a third of 884's occupied
cells.

### 2. Target conditioning. Ruled out, analytically.

**Predicted:** the fitted scalar and offset are collinear at r = -0.99 within
every dense row, so a better-conditioned target should interpolate better.

**Measured:** it cannot. Any interpolator whose prediction is a weighted sum of
training values with weights depending only on the coordinates commutes with an
invertible linear reparameterisation of the target
(`method-correction-identifiability.md`). Measured across twelve holdouts,
inverse distance weighting and nearest neighbour disagree between the two bases
by **exactly zero**, radial basis functions by 2.3e-8, and only ordinary
kriging by 0.209 m/s, its weights coming from a variogram fitted to the values.
The reparameterised target is also no flatter than the coefficients, 0.73
against 0.75 within-row share of variance, so no improvement could have been
attributed to that either.

**Rules out:** relabelling the target. Interpolating two coefficients whose
split is a tie-break remains wrong; it simply costs nothing in accuracy.

### 3. The identifiability defect. Real, and explains neither collapse.

**Predicted:** the affine fit solves one equation in two unknowns, so both
chapters operate on a coordinate set by a tie-break, and correcting that should
help.

**Measured:** the defect is real and is recorded in full in
`method-correction-identifiability.md`. Correcting it changes no prediction any
geometry-weighted interpolator makes, by the result above, and the machine
learning's own variance decomposition already put 80% of the variance
within-region, which the collinearity says lies mostly along the unidentified
direction.

**Rules out:** the defect as the cause. It remains a defect worth fixing on its
own terms.

### 4. Regime coverage. Ruled out, in the wrong direction.

**Predicted**, by `method-ml-transfer.md` before any of this ran: transfer
fails where the training set has no analogue of the held-out regime, so folds
sitting outside the covered feature space should do worst.

**Measured**, per country holdout, as distance in standardised terrain-feature
space to the nearest training point, against the leave-one-country-out error
already measured. Data: `output/regime_coverage_2026-09-16/`, from
`scripts/analysis/regime_coverage.py`.

| Fold | points | median feature distance | share outside training cells | LOCO MAE, m/s | R-squared |
|---|---|---|---|---|---|
| BE | 3 | 0.377 | 0.000 | 0.201 | +0.378 |
| DK | 886 | 0.088 | 0.017 | 0.503 | -0.208 |
| DE | 500 | 0.107 | 0.190 | 0.717 | -0.143 |
| UK | 303 | 0.415 | 0.446 | 1.010 | -0.142 |
| IE | 3 | 0.608 | 0.333 | 1.598 | -4.367 |
| NO | 5 | **2.602** | **0.800** | 2.843 | -0.484 |
| ES | 4 | 0.841 | 0.500 | 2.906 | -1.667 |
| SE | 4 | 0.208 | 0.000 | 3.231 | -0.198 |
| NL | 5 | **0.033** | **0.000** | 4.124 | -41.040 |
| FR | 10 | 0.324 | 0.100 | 4.483 | -0.141 |
| IT | 3 | 0.630 | 0.333 | 8.755 | +0.010 |
| PT | 3 | 0.321 | 0.000 | 13.811 | -1.488 |

| Coverage measure against LOCO MAE | Pearson | Spearman |
|---|---|---|
| median feature distance | **-0.016** | **+0.070** |
| 90th percentile distance | +0.408 | +0.308 |
| share outside training cells | **-0.158** | **-0.114** |

**Three folds carry the argument.** The Netherlands is the best-covered fold in
the pool, median feature distance 0.033 and nothing outside the training cells,
and it has the worst error of any small fold. Portugal also has nothing outside
the training cells and the worst error of all, 13.8 m/s. Norway is far outside
the covered space by every measure, 80% of its points in unoccupied cells, and
sits mid-table.

**The sign is the finding.** Two of the three coverage measures correlate
negatively with error: folds further outside the covered space do marginally
better. This does not fail to support the coverage hypothesis, it rules it out
in the direction anyone would have predicted.

## Which figures carry the weight, and which do not

**The conclusion rests on the mean absolute errors, not on the R-squared
values.** Several folds hold three to five points, where R-squared is a ratio
to almost no variance and cannot bear the weight its magnitude suggests:

- **The Netherlands' -41.0 is five points.** The figure the argument uses is
  its MAE of **4.12 m/s**, against a mean corrected speed of 7.49, which is bad
  on its own terms without any reference to variance.
- **Ireland's -4.37 is three points.** Its MAE of 1.60 is what is quoted.
- **Italy's +0.010 is three points** and is not evidence of anything. It
  appears in the table for completeness and supports no claim.
- **The 90th-percentile correlation of +0.408 is the one coverage measure with
  any signal, and it is driven by Italy's single outlier at 13.5 out of three
  points.** It is reported and not relied on. The median distance, which that
  outlier cannot move, is -0.016.

The three dense folds, Denmark at 886 points, Germany at 500 and the United
Kingdom at 303, have MAEs of 0.50 to 1.01 and R-squared of -0.14 to -0.21.
Those R-squared values are on enough points to mean something, and they say a
fold's own mean beats the interpolation.

## The contradiction with the earlier recommendation

`method-ml-transfer.md` reached the right verdict and recommended the wrong
next step. Its verdict, that in the failing regimes the correction is a
reanalysis-resolution artefact rather than a learnable bias, survives
everything here. Its recommendation, "data acquisition first, ML re-evaluation
second", does not: the test above is what separates them, and it says adding
regions would not be expected to help, because coverage does not predict which
folds transfer.

**The recommendation was reasonable, and the coverage figures are why.** Of six
non-European scorecard regions measured in the same feature space:

| Region | points | occupied cells | new to Europe | share of its own cells that are new |
|---|---|---|---|---|
| **US** | 1,091 | 195 | **113** | 58% |
| BR | 125 | 58 | 28 | 48% |
| AU-NEM | 67 | 51 | 28 | 55% |
| CL | 47 | 29 | 15 | 52% |
| **NZ** | 8 | 7 | **7, all of them** | 100% |
| AR | 59 | 17 | 3 | 18% |

Europe's 1,729 points occupy 264 cells of 4,096, and the fleet is
overwhelmingly flat: median elevation 37 m and median slope 0.42 degrees. The
United States alone would raise the occupied count by 43%. There was every
reason to expect that to matter. It does not follow that it would.

## The surviving explanation, and how to falsify it

**The claim.** Most of what a fitted correction contains is where a
0.25-degree reanalysis fails locally, not a physical bias that terrain and
climate encode. Such a quantity is not predictable in an unseen place however
many examples you have, however well conditioned the target, and however
diverse the regimes.

**Its support, from two independent directions.** The variance decomposition in
`method-ml-transfer.md` puts over 80% of the variance within-region for both
parameters, so there is little between-region signal to learn even in
principle. And four candidate explanations of the failure have now been tested
and eliminated, which is weak evidence individually and stronger jointly
because each was the leading alternative when it was tested.

**A surviving hypothesis with no falsification test is not much better than the
ones that failed**, so: **refit the correction on a finer wind product and see
whether the extreme scalars disappear at source.** If they do, and if the
residual corrections then transfer across borders, the claim is confirmed. If
the corrections remain as unpredictable on a 3 km wind as on a 31 km one, the
claim is false and the cause lies somewhere none of this has looked.

**What that costs.** Two candidate products cover Europe: CERRA at 5.5 km,
1984 to 2021, and the New European Wind Atlas at 3 km, 1989 to 2018. Each
would need acquisition, a loader beside `vwf.datasets.era5`, and the roughness
treatment settling on a product that may carry its own. The correction
pipeline itself is product-agnostic, so the expensive parts are the download
and a new observation-to-wind alignment, not the fitting. It is the largest
single piece of work this repository has left and it is the only one that
tests the surviving claim. Named, not costed in hours, and not started.

## Caveats

- **Twelve folds, one held-out year each**, and five of the twelve hold five
  points or fewer. Screening-level throughout.
- **The feature space is terrain only**: elevation, slope, roughness and
  curvature from ETOPO at 30 arcseconds, the same derivation the
  machine-learning work used. A regime that differs climatically but not
  topographically is invisible to it, and the Netherlands, which is the
  best-covered fold and among the worst-predicted, is exactly where that would
  bite. This is the weakest link in the coverage elimination and it is not
  closed.
- **The pool is the chapter-era one**, built on the uniform grids, and five of
  its fourteen contributing configurations were fitted on extrapolated winds.
- **Eliminating four explanations does not prove the fifth.** It makes it the
  standing hypothesis, which is why the falsification test above matters more
  than any of the eliminations.
