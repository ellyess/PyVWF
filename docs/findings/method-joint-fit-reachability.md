# Is the joint national fit's target reachable?

**Date:** 2026-09-25
**Scope:** every period the joint national offset fit sees in the eight
country-level rows (FR, BE, IE, SE, NO, ES, and the suspended IT and PT),
trained with the scorecard configs on the licensed library (`input/combined`,
`power_curves.csv` sha256 `689cfee7…`, 0% of capacity substituted in every
run). Training years 2015-21 (IE 2017-21). No test year is read: the study
classifies training fits. Registered in
`method-joint-fit-reachability-prereg.md` before the pass ran (`03e5a94`), with
a follow-up added after its table was seen (`b23b2b9`). Issue #68. Terms follow
`CONTEXT.md`.

**The 90 refused national fits split into 73 whose observation lies below
anything the fit can produce and 17 whose target is within reach, but 61 of
those 73 become reachable once an off-curve value counts as zero output
instead of dropping out of the mean.** So, by the readings registered before
each run, the pass's "mostly unreachable" reading holds for the fit as
written and is withdrawn by the follow-up: most of the refusals behind the
Italy and Portugal suspensions come from the objective's treatment of
off-curve values. 12 stay unreachable on either range: 6 in Italy, all summer,
and 6 in Portugal. No accepted fit is unreachable on either range.

## The pass: the range as the fit computes it

Each period's reachable range is the capacity-weighted sum of each cluster's
lowest and highest capacity factor over the offsets the fit accepts, at the
period's fitted scalars, with the fit's own simulation, which leaves an
off-curve value out of the mean. Unreachable: the observation beyond the range
by more than 1e-6. Near-miss: within 0.05 of either end. Definitions in the
registration.

**Control:** every one of the 540 periods reproduces the licensed-curve record
(`output/country_curves_2026-09-25/diag/`) exactly: fit outcome, offsets and
objective value.

All eight rows, by class and fit outcome:

| Class | Accepted | Refused on a bound | Refused abnormal |
|---|---|---|---|
| reachable | 418 | 1 | 3 |
| near-miss | 32 | 5 | 8 |
| unreachable above | 0 | 0 | 0 |
| unreachable below | 0 | 65 | 8 |

Per row:

| Row | Class | Accepted | Refused on a bound | Refused abnormal |
|---|---|---|---|---|
| FR | reachable | 70 | 0 | 0 |
| BE | reachable | 70 | 0 | 0 |
| IE | reachable | 50 | 0 | 0 |
| SE | reachable | 70 | 0 | 0 |
| NO | reachable | 68 | 1 | 1 |
| ES | reachable | 69 | 0 | 1 |
| IT | reachable | 7 | 0 | 1 |
| IT | near-miss | 9 | 3 | 1 |
| IT | unreachable below | 0 | 43 | 6 |
| PT | reachable | 14 | 0 | 0 |
| PT | near-miss | 23 | 2 | 7 |
| PT | unreachable below | 0 | 22 | 2 |

The four accepted fits that leave an error above 1e-6:

| Row | Clusters | Period | Error | Class | Margin to top | Margin to bottom |
|---|---|---|---|---|---|---|
| IT | 1 | 2018 spring | 4.2e-4 | near-miss | 0.584 | 0.0014 |
| ES | 4 | 2021 spring | 5.6e-5 | reachable | 0.680 | 0.182 |
| PT | 1 | 2018 spring | 1.5e-5 | reachable | 0.404 | 0.095 |
| PT | 2 | 2016 | 1.4e-5 | near-miss | 0.470 | 0.037 |

**Against the registered readings:**

- **Mostly unreachable.** 73 of the 90 refused periods, 81%, are unreachable,
  all of them below. The registration reads this as a data or curve question
  for Italy and Portugal.
- **Accepted but unreachable: none.** The case the synthetic test shows, a
  fit accepted at an interior peak below its target, does not occur here.
- **The check is worth building.** 32 accepted periods are near-misses, which
  meets the registered condition. Every one is near the bottom of its range.
- **17 refusals are of reachable or near-miss targets:** the optimiser failed
  on a target it could have met, 1 on a bound and 3 abnormally among the
  reachable, 5 and 8 among the near-misses.

Data: `output/reachability_2026-09-25/<CODE>/train-reach/reachability.json`,
tabulated in `reachability_periods.csv` by
`scripts/studies/method-joint-fit-reachability/reachability_tables.py`.

## The follow-up: off-curve values counted as zero

**Added after the pass's table was seen, and registered as an addendum before
it ran.** Every unreachable period was below its range, and the floors were
implausible: Italy's lowest national capacity factor had a median of 0.238 over
its periods, and ran from 0.21 to 0.33 in the unreachable ones, at offsets
where most corrected speeds are below zero. An off-curve value has no capacity
factor on the curve and drops out of the fit's mean (`CONTEXT.md`, off-curve
value), so a large negative offset leaves only the windiest steps in the
average. The follow-up recomputes each period's range from the same runs,
scalars, grid and refinement, with each cluster's corrected speeds clipped to
the curve's 0 and 40 m/s ends, where the curve reads zero, so an off-curve
value counts as zero output. Missing input speeds stay missing. The same
thresholds.

**Control:** every fit in the follow-up's runs is identical to the pass's.

All eight rows, by class on the zero-counted range and fit outcome:

| Class | Accepted | Refused on a bound | Refused abnormal |
|---|---|---|---|
| reachable | 450 | 44 | 16 |
| near-miss | 0 | 16 | 2 |
| unreachable above | 0 | 0 | 0 |
| unreachable below | 0 | 11 | 1 |

Per row, where anything differs from all-reachable:

| Row | Class | Accepted | Refused on a bound | Refused abnormal |
|---|---|---|---|---|
| NO | reachable | 68 | 1 | 1 |
| ES | reachable | 69 | 0 | 1 |
| IT | reachable | 16 | 35 | 5 |
| IT | near-miss | 0 | 6 | 2 |
| IT | unreachable below | 0 | 5 | 1 |
| PT | reachable | 37 | 8 | 9 |
| PT | near-miss | 0 | 10 | 0 |
| PT | unreachable below | 0 | 6 | 0 |

FR, BE, IE and SE are reachable and accepted in every period on both ranges.

**The 73 unreachable refusals on the zero-counted range:** 43 reachable, 18
near-miss, 12 still unreachable. By the addendum's registered reading, more
than half becoming reachable or near-miss (61 of 73) withdraws the pass's
"mostly unreachable" reading: those refusals are produced by the objective
dropping off-curve values, not by the curve or the data.

Median national floor and ceiling by row, off-curve values dropped against
counted as zero:

| Row | Floor, dropped | Floor, zero | Ceiling, dropped | Ceiling, zero |
|---|---|---|---|---|
| BE | 0.0000 | 0.0000 | 0.8996 | 0.8996 |
| ES | 0.0692 | 0.0154 | 0.9182 | 0.9182 |
| FR | 0.0107 | 0.0017 | 0.9351 | 0.9351 |
| IE | 0.0103 | 0.0010 | 0.9551 | 0.9551 |
| IT | 0.2379 | 0.1102 | 0.8038 | 0.8006 |
| NO | 0.1280 | 0.0524 | 0.8272 | 0.8257 |
| PT | 0.2901 | 0.2067 | 0.7715 | 0.7637 |
| SE | 0.0352 | 0.0094 | 0.9018 | 0.9018 |

**The 12 that stay unreachable:**

| Row | Clusters | Period | Outcome | Observed | Floor, zero | Off-curve share at the floor |
|---|---|---|---|---|---|---|
| IT | 1 | 2015 summer | on a bound | 0.1314 | 0.1716 | 0.35 |
| IT | 1 | 2019 summer | abnormal | 0.1381 | 0.1783 | 0.36 |
| IT | 1 | 2021 summer | on a bound | 0.1666 | 0.2537 | 0.20 |
| IT | 3 | 2015 summer | on a bound | 0.1314 | 0.1785 | 0.49 |
| IT | 3 | 2019 summer | on a bound | 0.1381 | 0.1837 | 0.47 |
| IT | 3 | 2021 summer | on a bound | 0.1666 | 0.2651 | 0.31 |
| PT | 2 | 2016 autumn | on a bound | 0.2572 | 0.2858 | 0.69 |
| PT | 2 | 2016 summer | on a bound | 0.2414 | 0.2753 | 0.90 |
| PT | 2 | 2020 summer | on a bound | 0.2099 | 0.2822 | 0.98 |
| PT | 2 | 2021 autumn | on a bound | 0.2615 | 0.2616 | 0.70 |
| PT | 2 | 2021 spring | on a bound | 0.2421 | 0.2579 | 0.75 |
| PT | 2 | 2021 summer | on a bound | 0.2111 | 0.2691 | 0.99 |

The off-curve share is the largest over a period's clusters of the
capacity-weighted share of present steps off the curve at that cluster's
lowest-output offset. Data: `reachability_offcurve.json` beside each run, and
`reachability_offcurve_periods.csv`, from
`scripts/studies/method-joint-fit-reachability/offcurve_tables.py`.

Under the zero-counted range no accepted period is a near-miss, where the pass
counted 32; the registered "worth building" reading was made on the pass's
range, and is not re-read here.

## What changed after a result was seen

The follow-up, as above: its question, method and readings were fixed in the
addendum (`b23b2b9`) after the pass's table and before its own run. Nothing
else changed. The pass's table stands as it was reported, beside the
follow-up's.

## Caveats

- Each grid carries one representative turbine per country, so "reachable"
  means reachable for that turbine at the fitted scalar, not for the fleet.
- The scalar is fitted before the offsets and held fixed; a target unreachable
  at the fitted scalar might be reachable at another.
- Counting off-curve values as zero is one treatment, not the fit's: the fit
  still drops them. Which one the method should use is not decided here.
- The range refinement is local to the best grid point on a 0.25 m/s grid; a
  narrower second peak would be missed. The synthetic tests check the range
  against 40 random offset pairs and against the grid.
- Italy's and Portugal's scalars are high (up to 3.49 and 3.96 in the
  scorecard's licensed-curve runs), which is part of why their winds leave the
  curve; that is a property of the fits this study classifies, not something it
  tests.
- Screening-level: this classifies training fits and measures no skill.
