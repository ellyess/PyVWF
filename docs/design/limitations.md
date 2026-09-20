# Limitations

What PyVWF does not do, and what its numbers do not support. This page is the
home of the full list. The project README carries the three limitations that
decide whether someone should use the package at all, and links here for the
rest.

Per-region results and their caveats are separate, and live in the scorecard,
[`docs/findings/scorecard.md`](https://github.com/ellyess/PyVWF/blob/main/docs/findings/scorecard.md)
in the repository.

## The correction is statistical, not physical

The affine correction learns a scalar and an offset on the wind speed from
observed generation. It does not model the physics that produced the error it
removes. Wake effects are not modelled explicitly, so a wake loss that a fleet
carries is absorbed into the fitted factors rather than represented. Choice of
power curve strongly influences the result, and a unit matched to a curve from
another manufacturer carries that mismatch into every number computed from it.

The consequence is that a fitted factor is not a physical measurement of a
site. It is the adjustment that made one fleet's simulated generation match its
observed generation over one period, on the curve library that run used.

## The result is screening-level

Nothing here is MEASNET or DNV accredited, and none of it is investment advice.
The validation ranks and diagnoses. It does not stand in for an accredited
yield assessment, and a number from it should not be used as if it did.

## Every region rests on one test year

The correction is fitted on the training years and scored on a single test
year that it never saw. The reportable result is the drop from uncorrected to
corrected on that year.

Two things follow, and both are easy to get wrong:

- **An ordering between two close configurations is not meaningful.** Neither
  is the exact best cluster count. A sweep picks its best on the same test year
  it reports, so the choice is not independent of the score.
- **A gain is not always distinguishable from zero.** Where the test year's
  units are resampled and the interval on the gain includes zero, the scorecard
  marks the row. A small fleet can have a single unit decide its result.

## The correction does not always help

There are regions where it makes the error worse, and regions where it removes
the mean bias and adds little skill. There is one region excluded outright,
because a defect in the observed series cannot be repaired by rescaling. The
scorecard names each, with the numbers and the reasoning.

This is recorded rather than tuned away. A negative result stands unless a
gate fixed before the run is met.

## A good aggregate can hide a degenerate fit

A fleet average can improve while individual clusters do not. A run reports
`fit_quality` on its factors table for exactly this reason: the scalar range,
the count of implausible scalars, and the count of failed offsets. A fit is
degenerate when any scalar falls outside 0.2 to 3.0, or any offset did not
converge.

The factors of a degenerate fit should not be reused, even where the
aggregate metric of that run looks sound. Judge a correction by `fit_quality`,
not by the skill metric. The cluster count does not predict fit quality, so no
rule of the form "use this many clusters" is safe.

Where a cluster's accepted years are not a strict majority of the training
years, its factor is refused: it carries no scalar and no offset, and its units
get no corrected values. A refused factor counts as a failed offset. This
removes the worst fits from the applied factors, and it also removes those
units from the scored rows of every variant of the run.

## Country-level offsets are under-determined

A country-level run fits its offsets against one national series per period.
The offsets are therefore under-determined, and they largely repair the
scalar's cube-law overshoot rather than capturing an additive spatial bias.
The reasoning is in `method-country-level.md`, in the findings tree.

## Reanalysis resolution bounds the accuracy

The correction operates on ERA5 at 0.25 degrees. That resolution limits
turbine-level accuracy, and no amount of fitting recovers structure the wind
product does not carry. Where a residual error is a reanalysis-resolution
artefact, the fix is a finer wind product rather than more observations or
more clusters.

Accuracy also depends on the quality and the representativeness of the
observations a region is fitted against. An observed series with an unscreened
confound, such as curtailment, carries that confound into the fitted factors.

## What is reproducible, and how

- Dependency ranges are declared in `pyproject.toml`.
- Every harness run writes a manifest recording the package version, the git
  state, the region config, and the identity and sha256 of the curve library
  behind the numbers.
- Methods are deterministic where possible.

For published work, state the ERA5 version, the training years and the power
curve source alongside the citation. A result whose curve library is not named
cannot be checked.

Published results are legacy rather than a target. The repository aims at the
most accurate results it can produce now, and a recorded output is superseded
by a better one rather than preserved. The code state behind each published
result is recorded in [publications.md](../publications.md).
