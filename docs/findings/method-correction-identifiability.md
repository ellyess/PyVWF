# The scalar and offset are not separately identified, and both chapters act as if they were

**Date:** 2026-09-16
**Scope:** whether the affine correction's two parameters are determined by the
data they are fitted to, and what follows for thesis chapters 4 and 5. Terms
follow `CONTEXT.md`.

**The fit solves one equation in two unknowns.** For each cluster, time slice
and year it matches a single number, the capacity-weighted mean observed
capacity factor, by choosing two parameters. The solutions form a curve, every
point of which fits equally well, and the algorithm selects one by a rule
outside that objective. **The chapter's reported r = -0.867 between scalar and
offset is that curve**, and the chapter reads it as "compensatory behaviour
between multiplicative and additive adjustments", which describes the symptom
and not the cause. Chapter 4 then interpolates the two coordinates
independently across space, and chapter 5 trains a model to predict one of
them. **Neither is operating on a quantity the data determined.**

## The objective, from the code

`vwf.correction.calculate_scalar` aggregates to one observed and one simulated
mean per cluster, slice and year, and sets

    scalar = obs / sim

on the **uncorrected** simulation. `vwf.correction._find_offset_iterative`
then searches the offset until the corrected mean matches:

    error = obs - mean_simulated_cf(scalar, offset)

That second step is one equation in one unknown **given** the scalar. The
scalar itself is not fitted to the corrected objective at all: it is pinned by
a ratio of uncorrected means, which is a different quantity. So the pair is
chosen by a tie-break, not identified by a fit, and a larger scalar with a more
negative offset satisfies the same objective.

This is a statement about the estimator, readable from the code, and it does
not need a probe of the likelihood surface.

## How collinear, pooled and per row

Data: `output/pyvwf_to_grid/all_corrections_centroids.csv`, the 1,729 control
points of the chapter's pool, fitted at each row's own training years and
`fixed` slice. Produced by `scripts/analysis/correction_identifiability.py`;
tables in `output/identifiability_2026-09-16/`.

**Pooled across all 1,729 points, r = -0.867**, reproducing the chapter's
figure exactly.

**Pooling understates it.** Per row, for the rows with enough points to
support the statistic:

| Row | points | r | share of variance along one line | pivot speed, m/s |
|---|---|---|---|---|
| DE onshore | 500 | **-0.998** | 0.999 | 3.97 |
| UK onshore | 293 | **-0.997** | 0.998 | 4.04 |
| DK onshore | 884 | **-0.989** | 0.995 | 4.06 |
| UK offshore | 10 | -0.979 | 0.989 | 3.81 |
| FR | 10 | +0.355 | 0.678 | -0.12 |

In the four dense rows, **99% or more of the joint variation lies along a
single line**. The pooled figure mixes rows whose lines differ, which is why it
is weaker than any of them.

## Where the lines cross, and what is well determined

Corrected speed is `scalar * v + offset`, so a row's corrections are a pencil
of straight lines. Their spread across a row is minimised at
`v* = -Cov(scalar, offset) / Var(scalar)`.

**The four dense rows pivot at 3.81 to 4.06 m/s, independently.** Three
countries, onshore and offshore, agreeing to within a quarter of a metre per
second on where their corrections coincide.

Relative spread, as a coefficient of variation within each row:

| Row | scalar | offset | corrected speed at 5 m/s | at 8 m/s | **at the pivot** |
|---|---|---|---|---|---|
| DE onshore | 0.284 | 2.917 | 0.056 | 0.137 | **0.018** |
| DK onshore | 0.187 | 0.687 | 0.034 | 0.082 | **0.022** |
| UK onshore | 0.337 | 13.695 | 0.067 | 0.165 | **0.026** |
| UK offshore | 0.300 | 0.627 | 0.063 | 0.130 | **0.042** |

**The corrected speed at the pivot is 10 to 500 times better conditioned than
the coefficients it is built from.** United Kingdom onshore's offset has a
coefficient of variation of 13.7; the corrected speed at its pivot has 0.026.

## What follows

**A reparameterisation costs nothing.** The corrected speed at two reference
winds is a linear bijection of the scalar and offset, so nothing is lost and
only the conditioning changes: one coordinate becomes nearly constant within a
row and the other carries the variation.

**But the near-constant coordinate is not the useful target.** A quantity with
a coefficient of variation of 0.02 across a row is easy to predict and says
almost nothing. The informative coordinate is the slope, which is the one the
tie-break sets. So reparameterisation makes the problem legible; it does not by
itself make the correction predictable, and claiming otherwise would repeat the
error this document is about.

**The pivot near 4 m/s needs an explanation before it is used.** It sits close
to a typical cut-in speed, where capacity factor is near zero and the
observations constrain the fit least. Two readings are open and this document
does not choose between them: the corrections genuinely agree at low wind
because that is where the reanalysis is unbiased, or they agree there because
the objective cannot see that region and the tie-break defaults to a common
value. The second would make the pivot an artefact of the estimator. What would
distinguish them is a fit whose objective uses more than one statistic.

## The pivot probed, 2026-09-16: well conditioned because unconstrained

**The pivot is not an artefact of where the offset search starts, and it is
not evidence about physics either.** Probed by re-solving every cluster's
offset from six initial steps and two iteration caps, on the selection study's
own factors, which are a current fleet on `era5/EU_2026-09` rather than the
chapter-era pool. Data: `output/pivot_probe_2026-09-16/`, produced by
`scripts/analysis/pivot_probe.py`.

| Row | clusters | shipped | from 10 | from 4 | from 3 | from 1 | from 0.5 | from 0.25 |
|---|---|---|---|---|---|---|---|---|
| DE onshore | 500 | 3.912 | 3.765 | 3.765 | 3.765 | 3.765 | 2.197 | 0.540 |
| DK onshore | 884 | 3.912 | **3.912** | **3.912** | **3.912** | **3.912** | 3.131 | 0.324 |
| UK onshore | 300 | 4.011 | **4.011** | **4.011** | **4.011** | **4.011** | 2.156 | 0.568 |

Identical to three decimals for every initial step at or above 1 m/s, and
unchanged by capping iterations at 30 instead of 100. **So the search is
finding a root rather than stopping where it started**, and the pivot belongs
to the scalar rule and the data. It also reproduces across fleet and archive:
3.91 to 4.01 here against 3.97 to 4.06 on the chapter-era pool.

**Germany's re-solve is not a reproduction and Denmark's and the United
Kingdom's are.** The probe reconstructs each cluster's target from its shipped
pair rather than from the original observation, which for Germany shifts the
pivot by 0.15. The other two return their shipped pivot exactly, and that is
what makes them worth quoting.

### The objective cannot see where the lines cross

The fit matches one capacity-weighted mean capacity factor, so what it sees is
energy:

| Row | days below 4 m/s | **share of capacity-factor mass below 4 m/s** |
|---|---|---|
| DE onshore | 16.6% | **2.47%** |
| DK onshore | 9.7% | **0.82%** |
| UK onshore | 13.6% | **1.74%** |

**Under 2.5% of the quantity being matched comes from below the pivot.** The
lines cross where the data have almost no leverage, so the crossing is an
extrapolation of the fitted relationship rather than a measurement.

### The misreading, twice in one finding

A pencil of lines has a crossing point by construction, and the spread of its
members is smallest there by definition. **So the corrected speed at the pivot
is well conditioned because it is unconstrained, not because the clusters
agree.** Reading its coefficient of variation of 0.02 as agreement between
clusters is the same error as reading r = -0.867 as compensatory behaviour:
both take a property of the fitting geometry for a property of the wind.

The first reading is the chapter's and the second was this document's own
proposal, made in the section above before the probe ran. That is why this is
worth writing rather than recording: **the same misreading appeared twice in
one finding, once from the chapter and once from the correction to it.**

### What this fixes about the reference wind

The pivot is rejected as a reference wind, for the reason it first looked
attractive. A target should be informative, and one whose variation is
suppressed by construction is not. **A reference wind is taken inside the range
the objective can see, 8 to 12 m/s**, where the corrected speed's within-row
spread is 0.08 to 0.21 rather than 0.02. That spread is the evidence the target
is carrying something, and the pivot's absence of it is the evidence it is not.

## What this says about the two chapters

**Chapter 4.** Interpolating the scalar and the offset independently across
space interpolates two coordinates whose split is set by a tie-break. Two
neighbouring clusters may sit at different places along their own lines while
describing nearly the same correction, and the interpolation of each
coefficient then carries that arbitrariness into the surface. This is a better
explanation of the pool's behaviour than any of the interpolation choices
examined so far, and it is consistent with the seven implausible control points
being extreme in their coefficients while their corrected speeds are ordinary.

**Chapter 5.** A model trained to predict the scalar is predicting where a
tie-break landed. Its own variance decomposition already reports that over 80%
of the variance is within-region (`method-ml-transfer.md`), and the within-row
collinearity says most of that within-region variation lies along the
unidentified direction.

**Neither chapter is wrong in its own terms**, and both report their results
honestly. The defect is upstream of both, in what the fitted pair means.

## Caveats

- **This is a property of this estimator**, not of affine corrections in
  general. A fit whose objective used the distribution rather than the mean
  would identify both parameters.
- **The pivot is measured on fitted values, not on uncertainties.** No
  per-fit confidence region exists, so the width of each individual ridge is
  unmeasured; what is measured is that the ensemble of fits lies on one.
- **The offset search's step schedule throttles below about 1 m/s**, which is
  a latent defect in the estimator recorded separately in `../../STATUS.md`.
  It does not affect any fit in this repository: the shipped initial step is
  10.0 and the only other value it has ever had is 3.0, and the two give
  identical results.
- **Rows with three to five points** are reported for completeness and support
  no statistic. Denmark offshore has two and is omitted.
- **Every result rests on the chapter-era pool**, built on the uniform grids,
  and on a single held-out test year per row. Screening-level.
