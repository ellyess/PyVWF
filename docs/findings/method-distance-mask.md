# The 5-degree distance mask neutralises the safest cells on the grid

**Date:** 2026-09-15
**Scope:** whether the distance mask thesis chapter 4 applies to its gridded
correction surface does what its rationale says it does. Terms follow
`CONTEXT.md`.

**The rationale is inverted by measurement.** The mask neutralises every grid
cell more than 5 degrees from any control point, on the reasoning that an
interpolated correction that far from data cannot be trusted. Beyond 5 degrees
the kriged surface reverts to the pool mean and holds **the tamest values on
the grid**. Every one of the 72 cells whose correction cannot produce a
capacity factor at all lies **within** 5 degrees. So the mask neutralises 8,269
safe cells and leaves 691 unsafe ones untouched, and the kriging variance,
which is the principled form of the same idea, fails in the same direction.

## What was measured

One ordinary kriging fit, exponential variogram, geographic coordinates, from
the undivided pool of 1,729 control points in
`output/pyvwf_to_grid/all_corrections_centroids.csv` onto the chapter's grid:
longitude -10 to 30 and latitude 35 to 72 at 0.25 degrees, 161 by 149 cells,
**23,989 in total**. No mask of any kind is applied.

Data: `output/unmasked_bands_2026-09-15/unmasked_surface_cells.csv`, one row
per cell, and `run.log` beside it. Produced by
`scripts/analysis/unmasked_surface_bands.py` at commit `bdb5f66` from a clean
tree. Region shapes: `input/reference/shapes/country_shapes.geojson` and
`offshore_shapes.geojson`.

The control points are the chapter-era fits, whose training years and single
test year differ by configuration: the nine country-level configurations train
on 2015 to 2021 and test on 2023, except Ireland which trains on 2017 to 2021;
Germany and the United Kingdom train on 2015 to 2018 and test on 2019; Denmark
trains on 2015 to 2019 and tests on 2020.

The curve table is the combined library at
`input/combined/reference/power_curves.csv` (sha256 `689cfee7...`). Only one
property of it is used here: its speed column runs from **0 to 40 m/s** in
4,001 steps, so a corrected speed outside that interval has no capacity factor.
Corrections are read at a reference speed of **8 m/s**, which is near the daily
mean these fleets see, because a scalar and an offset are hard to read together
and a corrected speed is not.

## How the cells divide

Bands are in Euclidean degrees, the units the chapter's threshold is stated in.

| Band | Cells | Share of grid | Inside a region shape | Share of band |
|---|---|---|---|---|
| 0 to 1 | 4,375 | 18.24% | 4,265 | 97.49% |
| 1 to 2 | 4,143 | 17.27% | 3,970 | 95.82% |
| 2 to 5 | 7,202 | 30.02% | 5,255 | 72.97% |
| beyond 5 | 8,269 | 34.47% | 2,357 | 28.50% |

**The 34.5% beyond the mask is 8,269 cells, of which 2,357 lie inside any
onshore or offshore shape.** The remaining 5,912 are open Atlantic, North
Africa and land east of every fleet in the pool. Removing the mask therefore
newly corrects **9.8% of the grid** in cells anyone could plausibly sample, not
34.5%.

## What each band holds

Every value below is over all cells of the band.

`scalar`:

| Band | min | 5% | 50% | 95% | max |
|---|---|---|---|---|---|
| 0 to 1 | 0.234 | 0.558 | 0.935 | 2.582 | 4.323 |
| 1 to 2 | 0.408 | 0.598 | 1.159 | 2.460 | 3.252 |
| 2 to 5 | 0.532 | 0.722 | 1.158 | 1.741 | 2.787 |
| beyond 5 | 0.669 | 0.855 | 1.144 | 1.324 | 1.534 |

`offset`, in m/s:

| Band | min | 5% | 50% | 95% | max |
|---|---|---|---|---|---|
| 0 to 1 | -8.355 | -4.746 | -0.023 | 1.461 | 3.032 |
| 1 to 2 | -7.535 | -6.597 | -0.694 | 0.882 | 1.718 |
| 2 to 5 | -7.088 | -6.260 | -1.399 | 0.280 | 1.258 |
| beyond 5 | -6.068 | -5.285 | -1.893 | -0.945 | **-0.328** |

Corrected speed at 8 m/s:

| Band | min | 5% | 50% | 95% | max |
|---|---|---|---|---|---|
| 0 to 1 | -0.273 | 4.218 | 7.579 | 15.286 | 34.252 |
| 1 to 2 | -0.249 | 2.747 | 8.080 | 16.461 | 23.568 |
| 2 to 5 | -0.149 | 1.014 | 7.499 | 11.290 | 19.688 |
| beyond 5 | 0.765 | 2.775 | 6.359 | 9.253 | 10.521 |

Kriging variance of the scalar:

| Band | min | 5% | 50% | 95% | max |
|---|---|---|---|---|---|
| 0 to 1 | 0.000 | 0.007 | 0.054 | 0.130 | 0.200 |
| 1 to 2 | 0.071 | 0.103 | 0.159 | 0.265 | 0.374 |
| 2 to 5 | 0.135 | 0.182 | 0.337 | 0.555 | 0.681 |
| beyond 5 | 0.319 | 0.452 | 0.708 | 1.049 | 1.207 |

Restricting to cells inside a region shape moves nothing material: beyond 5
degrees the scalar's 5th to 95th percentile range is 0.957 to 1.378 rather than
0.855 to 1.324, and the corrected-speed median is 5.173 rather than 6.359.

## Where the unusable values are

A cell is counted here as **off curve** when 8 m/s corrects to a speed outside
0 to 40 m/s, so the cell has no capacity factor at all, and as **extreme** when
it corrects to below 1 or above 20 m/s, which is a value on the curve but not a
credible one.

| Band | Cells | Extreme | Off curve |
|---|---|---|---|
| 0 to 1 | 4,375 | 148 | 19 |
| 1 to 2 | 4,143 | 185 | 31 |
| 2 to 5 | 7,202 | 358 | 22 |
| **beyond 5** | **8,269** | **16** | **0** |

707 cells are extreme and 72 are off curve. **691 of the 707 and all 72 of the
72 are inside the region the mask keeps.**

The same 707, cut by kriging variance instead of distance:

| Scalar-variance quintile | Cells | Extreme | Median distance, degrees |
|---|---|---|---|
| lowest | 4,798 | 129 | 0.506 |
| 2nd | 4,798 | 149 | 1.673 |
| 3rd | 4,797 | 184 | 3.087 |
| 4th | 4,798 | 232 | 5.735 |
| **highest** | 4,798 | **13** | 10.186 |

## Reading

**Far from data the kriged surface reverts to the pool mean rather than
diverging from it.** The scalar's full range contracts from 0.234 to 4.323 in
the nearest band to 0.669 to 1.534 in the farthest, and beyond 5 degrees the
offset is negative in every single cell, its maximum being -0.328. That is
ordinary kriging returning to the global mean where the variogram has no
information left. It is the expected behaviour of the estimator, and it is the
opposite of what a mask against untrustworthy extrapolation implies.

**The unusable values are near the control points, not far from them.** All 72
off-curve cells sit within 5 degrees, and 691 of the 707 extreme ones. They sit
where the pool disagrees with itself. Taking the five nearest control points to
each of the 15,720 cells within 5 degrees:

| Cells within 5 degrees | Count | Median spread of the 5 nearest scalars | Share outside that range | Median distance, degrees |
|---|---|---|---|---|
| not extreme | 15,029 | 0.752 | 7.8% | 1.810 |
| extreme | 691 | **2.368** | 12.4% | 2.098 |

An extreme cell's nearest control points disagree about three times as widely
as an ordinary cell's. **But only 12.4% of extreme cells hold a value outside
their neighbours' own range**, against 7.8% of the rest, so this is mostly not
the interpolation overshooting. It is the surface faithfully reporting control
points that are themselves extreme and mutually inconsistent. The defect is in
the pool's fits, not in the interpolation that reads them, which is why no
change to the interpolation or its masking addresses it.

**The kriging variance does not rescue the idea.** It is nearly a monotone
function of distance, with median 0.054, 0.159, 0.337 and 0.708 across the four
bands, so a variance threshold selects almost the same cells a distance
threshold does. Its highest quintile contains 13 of the 707 extreme cells and
its lowest contains 129. Extremes concentrate in the fourth quintile, at 232,
and then collapse: the highest-variance cells are the far-field cells that have
already reverted to the mean.

**So geometry does not select the risky cells, and no threshold on it will.**
A guard on the correction's own behaviour does, which is what a per-cell
plausibility flag is for. That is a design consequence and is recorded in
`../design/manuscript-chapters-45.md`.

**What the mask does support is a claim about provenance, not safety.** Beyond
5 degrees the value is the pool mean, so it carries no information about the
place it is applied to, which is the same conclusion the leave-one-country-out
result reaches from the other direction
(`method-loco-interpolation-prereg.md`). A correction that is the pool mean is
not dangerous; it is uninformative, and those are different warnings to give a
user.

## The units defect, 2026-09-15

The first run of the measurement wrote a column named
`distance_great_circle_degrees` that held kilometres, and compared it against
the Euclidean distance on shared band edges. That table reported 4,158 cells
within one degree of a control point as being beyond five, because one degree
of latitude is about 111 km. `vwf.extensions.grid.interpolation.degree_distances`
documents in terms that the two metrics are not in the same units and that a
threshold such as `MAX_DISTANCE_DEG` belongs to the Euclidean metric alone; the
script ignored its own dependency.

Fixed at commit `1bc281f`: the column is `distance_great_circle_km` and the
comparison is each Euclidean band's kilometre range.

| Band, degrees | min km | median km | max km |
|---|---|---|---|
| 0 to 1 | 0.2 | 39.4 | 110.3 |
| 1 to 2 | 40.5 | 129.1 | 221.2 |
| 2 to 5 | 80.7 | 268.1 | 551.6 |
| beyond 5 | 201.4 | 636.9 | 1,447.4 |

The bands overlap in kilometres because a degree of longitude shortens toward
the pole, which is why the two cannot share a threshold at all. No value in the
tables above was affected: the distance bands, the band shares, the region
shares and every distribution come from the Euclidean metric, which was correct
in both runs.

## Caveats

- **One interpolation method.** This is ordinary kriging with an exponential
  variogram. Reverting to the pool mean far from data is a property of kriging;
  inverse distance weighting reverts to a distance-weighted mean of the whole
  pool, which is a different function and is not measured here. The chapter
  ships both.
- **One pool.** The control points are the chapter's, at the chapter's cluster
  counts. The pool is being rebuilt at per-country cluster counts, and the
  distribution of extreme values between nearby control points is exactly the
  quantity that rebuild could move.
- **Five of the fourteen contributing configurations were fitted on
  extrapolated winds.** Spain, Italy, Norway, Portugal and Sweden reach beyond
  the chapter's ERA5 archive, by 42.80%, 89.35%, 35.68%, 84.92% and 5.46% of
  capacity (`method-domain-split-prereg.md`, amendment of 2026-09-15). Their
  scalars and offsets are among the values interpolated here.
- **The reference speed is a single point.** A correction that is credible at
  8 m/s may not be at 3 or at 20, and the counts above would differ at another
  speed. The per-cell frame is written out so the threshold can be recut.
- **Screening-level.** Nothing here is an accredited yield assessment.
