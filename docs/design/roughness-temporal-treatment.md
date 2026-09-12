# Two temporal treatments of the same roughness

Every PyVWF region derives the surface roughness length z0 the same way, by
inverting the log wind profile between the reanalysis 10 m and 100 m winds:

    z0 = exp( (w100 ln 10 - w10 ln 100) / (w100 - w10) )

The regions do not agree on what happens next. The difference is not the
formula. It is whether the result of the formula varies in time.

| Treatment | How it is produced | Regions |
|---|---|---|
| One annual mean | The hourly z0 is computed, then averaged over the year into a single static field, in `src/vwf/datasets/combine_era5_files.py` | DE, DK, UK and the eight country-level regions, which all read the European files |
| Hourly, then daily | The hourly z0 is computed and averaged to daily along with the winds, in `vwf.datasets.era5.prep_era5` or `scripts/era5/combine.py` | US, BR, AU-NEM, NZ, CL, AR |

So eleven of the seventeen scorecard rows apply a climatological roughness, and
six apply one that varies through the year. The annual mean is the majority
treatment, not the exception: it covers every European row, including the three
turbine-level rows that carry the most units.

## Why they differ

The European files were pre-combined into one file per year, and that step
stored a single representative roughness field rather than an hourly one. The
other regions kept the hourly derivation. There is no record of the annual mean
being chosen over the alternative, and nothing in the method documents states
which treatment a result used.

## The consequence for reading results

Results from the two groups are not directly comparable. The scorecard is an
index of per-region results and invites reading rows against each other; half
the rows differ from the other half in an input, not only in fleet,
observations and climate. Any statement that ranks or contrasts regions
inherits this, including work that pools regions, such as transfer between
regions and the physics-informed study.

A paper drawn from work that uses these results has to state which temporal
treatment produced them. The equation alone does not say.

## Where the treatment can matter at all

The roughness reaches a simulated capacity factor by one route, the hub-height
profile:

    w(h) = w100 ln(h / z0) / ln(100 / z0)

At h = 100 m the factor is exactly 1 whatever z0 is, so the roughness cancels.
Away from 100 m it enters as a ratio of logarithms, so what either treatment
can do to a speed is bounded by the distance from that reference. Taking the
span from smooth water to broken forest, z0 = 0.01 m against z0 = 0.25 m:

| Hub height | Factor at z0 = 0.01 m | at z0 = 0.25 m | Difference |
|---|---|---|---|
| 12 m | 0.770 | 0.646 | -0.124 |
| 30 m | 0.869 | 0.799 | -0.070 |
| 45 m | 0.913 | 0.867 | -0.047 |
| 60 m | 0.945 | 0.915 | -0.030 |
| 80 m | 0.976 | 0.963 | -0.013 |
| 90 m | 0.989 | 0.982 | -0.006 |
| 100 m | 1.000 | 1.000 | +0.000 |
| 120 m | 1.020 | 1.030 | +0.011 |
| 140 m | 1.037 | 1.056 | +0.020 |

Read a row as the ceiling on what the treatment can do to a unit at that
height, since moving from an annual mean to an hourly value is a change in z0
and nothing else. The sensitivity grows with the distance from 100 m on both
sides, and reverses sign above it: below the reference a rougher surface lowers
the speed, above it a rougher surface raises it.

The scorecard's fleets sit very differently against that table. Hub heights of
the training fleets behind the rows, from their `train_turb_info` files:

| Row | Hub heights | Median | Capacity-weighted mean |
|---|---|---|---|
| DK | 12 to 106.5 m | 45 m | 64 m |
| NZ | 30 to 80 m | 68 m | 64 m |
| UK | 25 to 125 m | 70 m | 75 m |
| US | 20 to 130 m | 80 m | 83 m |
| DE | 40.5 to 149 m | 85 m | 93 m |
| AR | 45 to 130 m | 93 m | 100 m |
| CL | 80 to 145 m | 100 m | 103 m |
| AU-NEM, BR | 100 m, uniform | 100 m | 100 m |
| Country-level rows | uniform per region: BE 100, SE 100, FR 90, ES 90, IE 85, IT 80, NO 80, PT 80 | | |

Two consequences, and neither follows the split between the treatments:

- **Some rows cannot show the treatment at all.** AU-NEM and BR give every unit
  a hub height of exactly 100 m, as do the BE and SE grids, so the factor is
  identically 1 and the roughness cancels. These rows are insensitive to the
  question by construction, and two of them are in the hourly group.
- **No country-level row can test it.** Every one gives its grid points a
  single uniform height between 80 and 100 m, where the whole span above is
  worth at most 1.3% of the speed. A null from such a row says the treatment
  could not be seen there, not that it does not matter.

The exposure is in the turbine-level rows that carry real hub heights, and most
of all in DK, whose median unit stands at 45 m. That is why DK is the row the
pre-registered comparison rests on
(`../findings/method-roughness-treatment-prereg.md`, deviation D1).

## Which is better is not known

Neither treatment is recommended here.

- The hourly derivation follows the wind conditions and is the reading a
  reader would take from the equation.
- The annual mean may be the more stable estimator. The hourly z0 is noisy, and
  it is undefined outright for some hours and, in complex terrain, for whole
  months (`undefined-roughness-in-complex-terrain.md`). An average over a year
  is not exposed to a single bad hour.

The two are being compared on Denmark, whose results also appear in a published
paper, under a pre-registration that fixes the question, the conditions and the
gate before the comparison runs. France was registered as a second row and is
still run and reported, but for the reason in the section above it cannot show
the treatment, so Denmark decides alone. Until that reports, no result changes
and no treatment is called correct.
