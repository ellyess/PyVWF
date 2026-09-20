# Where shear-derived roughness has no value

PyVWF does not use the reanalysis surface-roughness field. It derives the
roughness length z0 from the wind speeds themselves, by inverting the log wind
profile between the 10 m and 100 m levels:

    z0 = exp( (w100 ln 10 - w10 ln 100) / (w100 - w10) )

The reason is that the reanalysis roughness is known to be underestimated over
heterogeneous land use and near coastlines. The cost is that the estimator is
undefined for some real wind conditions, and in some terrain it is undefined
for long stretches.

## When it has no value

The expression needs the wind to increase with height. Two cases break it:

- **No shear.** When the two levels report the same speed the denominator is
  zero. The code masks a denominator smaller than 1e-4.
- **Inverted shear.** When the 10 m speed exceeds the 100 m speed the
  expression returns a roughness above 1 m, which the code also rejects,
  because a log profile fitted through those two points does not describe a
  surface layer.

Both are physical states, not data faults. Inverted or vanishing shear happens
in stable nocturnal layers, in drainage flows, and in terrain where the two
levels sit in different flow regimes.

## It is not rare in mountains

In the Brazilian bounding box, 99 of its 25,921 ERA5 cells lose their roughness
for at least one day of 2024. They lie along the Andes, from 34 S to 2.75 N
between 70.75 W and 68 W. Nine of those cell-months have no value on any day.
The worst cell, at 30.5 S and 70 W, is undefined on 85 days of the year, 73 of
them consecutive; the median undefined cell loses 5 days.

That is the general lesson: in complex terrain the estimator can be undefined
for a whole month, not just an hour, and a gap-filling rule that looks
sufficient on flat ground will not close it.

**Method, 2026-09-20.** Counted from the daily roughness field the region's
own input carries, `input/era5/BR_daily/era5_br_daily_2024.nc`
(sha256 `7aab731b…`), against the fleet metadata
`input/observations/turbine/BR/br_md.csv` (sha256 `11f850a8…`), both under the
default input root. A cell is undefined on a day when that field is missing
there. Distances are great-circle, and the degree figure is the plain
Euclidean separation in degrees, on the same cell-unit pair. The count imports
no repository code: it opens the two files and reduces them, so it is pinned by
those two hashes rather than by a commit. Read-only; it wrote nothing.

## What follows for a new region

- A bounding box that includes mountains will contain cells with no derived
  roughness, whether or not any unit stands there. Brazil's are far from its
  fleet: the closest undefined cell to any of the 193 units is 1,290 km away
  (13.2 degrees), at 27.5 S and 68.5 W, and the median is 1,363 km. The grid
  step is 0.25 degrees, so no undefined cell is among the four a unit's winds
  are interpolated from, and none reaches a simulated unit. A fleet sited in
  such terrain would be affected directly.
- A missing roughness propagates: the hub-height speed is missing, so the
  capacity factor is missing, and a monthly mean is then taken over the days
  that remain (`../guides/output-structure.md`).
- The off-curve and missing-value counts every run records are the place to
  check it. A region whose fleet sits in complex terrain should be read with
  those counts beside its results.

This is a limitation of the method, not a defect in the code. The alternative,
using the reanalysis roughness field, carries the bias the derivation exists to
avoid. Nothing here recommends a change; it records where the estimator stops
working, so that a future region in mountainous terrain is read with that in
mind.
