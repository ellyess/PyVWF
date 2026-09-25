# Is the joint national fit's target reachable?

**Date:** 2026-09-25. Registered before any run.
**Scope:** every period the joint national offset fit sees in the six
scoreable country-level rows (FR, BE, IE, SE, NO, ES) and the two suspended ones
(IT, PT), trained with the scorecard configs on `input/combined`. Training years
2015-21 (IE 2017-21). No test year is involved: the pass reads training fits
only. Issue #68. Terms follow `CONTEXT.md`.

**The pass classifies each period's target as reachable or not, and crosses
that with what the fit did.** It decides nothing about the method. What to
build afterwards, and whether the Italy and Portugal suspensions are worded
differently, is the maintainer's decision after reading the table.

## Why

`find_offsets_country_level` refuses a period when the optimiser reports
failure or an offset sits within `OFFSET_XTOL` of the ±10 m/s bound. A power
curve with cut-out, or a rated plateau, peaks at an interior offset, so a target
above that peak can be fitted "successfully" at the peak with no offset on a
bound, and nothing refuses it.

That case has been shown on synthetic data, before this registration:
`tests/test_reachability_pass.py::test_an_observation_above_the_range_is_accepted_with_the_whole_gap_as_error`.
On a two-cluster fleet whose curve is derated to 0.5, the fit accepts an
observation of 0.55, puts its offsets at 9.45 and 7.79 m/s, and stops at the
reachable maximum, leaving 0.05 as error.

The licensed-curve record (`output/country_curves_2026-09-25/diag/`, made at
`bf97f4a`) is the motivation for running this on real data: 90 of 540 national
fits are refused (IT 54, PT 33, NO 2, ES 1; 71 with an offset on the bound, 19
ending in an abnormal line search), and 4 accepted fits leave an error above
1e-6, the largest 4.2e-4 (IT k=1, 2018 spring, one offset at -9.96 m/s). Those
counts were seen before this registration; the reachable ranges were not.

## Definitions

For one period, with each cluster's fitted scalar as the fit receives it:

- **Reachable range**: [Σ w_c lo_c, Σ w_c hi_c], where w_c is the cluster's
  share of fleet capacity, and lo_c and hi_c are the lowest and highest capacity
  factor the cluster's simulation reaches over offsets from -10 + `OFFSET_XTOL`
  to 10 - `OFFSET_XTOL` m/s. Each extreme is taken on a 0.25 m/s grid and then
  refined by a bounded one-dimensional search (`xatol` 1e-5 m/s) over the grid
  step either side of the best grid point. The simulation is the objective's
  own, `train_simulate_wind_from_ws` on the cluster's winds for the period.
  Offsets at which the simulation returns no value (every speed off the curve)
  are left out of the extremes.
- **Unreachable**: the observation lies above the range's top or below its
  bottom by more than **1e-6** in capacity factor, the repository's
  `BRACKETED_MAX_RESIDUAL`.
- **Near-miss**: reachable, but within **0.05** in capacity factor of either
  end of the range.
- **Fit outcome**: accepted; refused on a bound (the optimiser succeeded, an
  offset is within `OFFSET_XTOL` of ±10); refused abnormal (the optimiser
  reported failure).

## Control, checked before any margin is read

The pass trains each row through the harness with the fit unchanged, so every
period's fit outcome, offsets and objective value must reproduce the record
above. A row whose outcomes do not match is reported and not classified; a
mismatch means the pass does not describe the fits it claims to.

## The table the pass reports, first

For all eight rows together and per row: counts of periods by
(reachable, near-miss, unreachable above, unreachable below) × (accepted,
refused on a bound, refused abnormal), and the four high-error accepted fits
named with their class and margin. Every margin is given for every refused
period in a data file.

## Readings, fixed now

- **The refusals are mostly unreachable** if more than half of the 90 refused
  periods are unreachable. Then Italy and Portugal are a data or curve
  question: the representative turbine, at the fitted scalar, cannot produce
  the observed national output, and no change to the optimiser helps.
- **The refusals are mostly reachable** otherwise. Then the optimiser is
  failing on attainable targets, and the next change is in the fit.
- **Accepted but unreachable** periods are the case the synthetic test shows.
  Any count above zero is reported as a defect of the fit's refusal, whatever
  the other readings.
- **The check is worth building** into the fit if any accepted period is
  unreachable, or is a near-miss. If every observation sits at least 0.05 inside
  its range, the check is not built and the plateau case is documented as a
  known limitation, as the red team proposed.

No prediction of the split is registered; the record does not support one.

## Protocol

From the repository root, on a clean tree, one row per process, through the run
lock:

```bash
PYVWF_INPUT=input/combined PYTHONPATH=src python scripts/dev/run_locked.py -- \
    python scripts/studies/method-joint-fit-reachability/reachability_pass.py \
    output/reachability_2026-09-25 <stem>
```

for `<stem>` in fr, be, ie, se, no, es, it, pt. Each run writes
`reachability.json` beside its training run.

## Caveats

- Each grid carries one representative turbine per country, so "unreachable"
  means unreachable for that turbine at the fitted scalar, not for the fleet.
- The scalar is fitted before the offsets and held fixed here. A target
  unreachable at the fitted scalar might be reachable at another.
- The refinement is local to the best grid point. A second peak narrower than
  the 0.25 m/s grid step could be missed, which would understate the maximum
  and overstate unreachability; the test on the synthetic fleet checks the
  refined maximum against the grid and against 40 random offset pairs.
