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

In the Brazilian bounding box, 99 of 25,921 ERA5 cells lose their roughness for
at least one day of 2024. They lie on the Andes edge, near 27 S and 68.5 W. In
those cells the 10 m speed is at or above the 100 m speed for **every hour of a
month**, so no hour of that month yields a roughness. The longest run is 85
days.

That is the general lesson: in complex terrain the estimator can be undefined
for a whole month, not just an hour, and a gap-filling rule that looks
sufficient on flat ground will not close it.

## What follows for a new region

- A bounding box that includes mountains will contain cells with no derived
  roughness, whether or not any unit stands there. Brazil's are 13.2 degrees
  from the nearest unit, so nothing reaches a simulated unit; a fleet sited in
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
