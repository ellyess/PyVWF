# Three routes to the same roughness, and two treatments of it

Every PyVWF region derives the surface roughness length z0 the same way, by
inverting the log wind profile between the reanalysis 10 m and 100 m winds:

    z0 = exp( (w100 ln 10 - w10 ln 100) / (w100 - w10) )

The regions do not agree on what happens next. The difference is not the
formula. It is whether the result of the formula varies in time.

There are three routes, not two, and they produce two distinct treatments:

| Route | How it is produced | What the file carries | Scorecard rows |
|---|---|---|---|
| **A, annual mean** | The hourly z0 is computed, then averaged over the year into a single static field, in `src/vwf/datasets/combine_era5_files.py` | hourly winds and a stored `z0`, one field per year | none |
| **B, per timestep at load** | The hourly z0 is computed in `vwf.datasets.era5.prep_era5` when the file carries no roughness, then averaged to daily with the winds | hourly winds only | DE, DK, UK, the eight country-level regions, AU-NEM, NZ, CL and AR: 15 |
| **C, per timestep, stored daily** | The hourly z0 is computed in `scripts/era5/combine.py` and averaged to daily there, so the file arrives with it | daily `wnd100m` and `roughness`, and **no 10 m winds** | US, BR: 2 |

**All seventeen scorecard rows now apply the per-timestep treatment**, fifteen
by route B and two by route C. Route A produced the eleven European rows
published before 2026-09-13, which the scorecard keeps in its superseded
block; the European re-run moved them to `era5/EU_2026-09` with
`roughness = "derived"`, and `tests/test_roughness_treatment.py` pins all
eleven scorecard configs to that pair. Until that re-run the annual mean was
the majority treatment and covered every European row.

**The maintained configs have not followed.** Every
`configs/regions/<stem>.toml` for a European region still names `era5/EU` and
sets no `roughness` key, so it defaults to `stored` and applies the annual
mean. A reader who runs one of those files, as the guides tell them to, gets
the superseded treatment rather than the method. Only the scorecard configs
under `configs/regions/scorecard/` carry the method's pair. This is logged as
work on the configs, not on the documents.

**B and C are the same treatment, computed at different stages.** Both derive
z0 hour by hour and average it to daily, with the same clipping. They differ
only in where the backfill of undefined hours reaches: `prep_era5` backfills
across the whole loaded record, `combine.py` within one month file, so an
undefined hour at a month's end can be filled in B and not in C. That is an
edge-level difference and has not been measured.

**The record cannot tell A from C.** This matters for the archive rather
than for a new run, since no current row is on route A. A run's manifest reports
`era5_roughness.applied` as `stored` whenever the file carries a roughness
field, so the US and Brazilian rows, which are on the per-timestep treatment,
are labelled exactly as the European rows, which are not. Reading the manifest
alone, C is indistinguishable from A.

**Route C cannot answer the question at all.** `combine.py` drops the 10 m
winds, so `roughness = "derived"` raises on those files: the only roughness
they can supply is the one they carry. The comparison that decided the method
could not have been run on the US or Brazil, whatever their hub heights.

Two pieces of candidate work follow, and neither is started:

- whether the manifest should record which kind of stored roughness a run
  applied, rather than only that one was stored;
- whether the US and Brazilian rows should move to the raw-monthly route, as
  the other fifteen rows use, so that they can answer the question and carry an
  unambiguous label. The cost is load time and memory on two continent-sized
  boxes, which is why the daily pre-combine exists.

## Why they differ

The European files were pre-combined into one file per year, and that step
stored a single representative roughness field rather than an hourly one. The
other regions kept the hourly derivation, two of them pre-computed to daily
because their boxes are too large to load hourly. There is no record of the
annual mean being chosen over the alternative, and nothing in the method documents states
which treatment a result used.

## The consequence for reading results

Results from the two groups are not directly comparable. The scorecard is an
index of per-region results and invites reading rows against each other, and
until the European re-run of 2026-09-13 half the rows differed from the other
half in an input, not only in fleet, observations and climate. Every current
row is on one treatment, so the split no longer confounds the table. It still
confounds any comparison that reaches back to a superseded row, and any work
that pooled regions before the re-run, such as transfer between regions and
the physics-informed study.

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
  question by construction.
- **No country-level row can test it.** Every one gives its grid points a
  single uniform height between 80 and 100 m, where the whole span above is
  worth at most 1.3% of the speed. A null from such a row says the treatment
  could not be seen there, not that it does not matter.

The exposure is in the turbine-level rows that carry real hub heights, and most
of all in DK, whose median unit stands at 45 m. That is why DK is the row the
pre-registered comparison rests on
(`../findings/method-roughness-treatment-prereg.md`, deviation D1).

## Which treatment the method uses

**The per-timestep derivation is the method, adopted on 2026-09-12**
(`../findings/method-roughness-treatment.md`). It was adopted on method
fidelity and comparability: it is what the published method describes, most
rows cannot show any difference between the treatments at all, and a single
treatment removes a split that confounds every comparison between regions. The
measured accuracy effect, on Denmark, is 0.0002 in corrected RMSE, which is
resolved and far too small to carry the change on its own.

The two arguments that stood before the comparison are recorded, since neither
was settled by it:

- The hourly derivation follows the wind conditions and is the reading a
  reader would take from the equation.
- The annual mean may be the more stable estimator. The hourly z0 is noisy, and
  it is undefined outright for some hours and, in complex terrain, for whole
  months (`undefined-roughness-in-complex-terrain.md`). An average over a year
  is not exposed to a single bad hour.

Nothing in the comparison spoke to the second point: no row tested sits in
complex terrain, and at Denmark the hourly derivation cost two unit-months out
of a fleet of 5,410.

## Which input carries which treatment

The treatment follows the ERA5 directory a configuration reads, so a row's
committed configuration names it:

| Directory | Contents | Route |
|---|---|---|
| `era5/EU` | hourly winds plus a stored annual-mean `z0`, box 42 to 72 N, 12 W to 22 E | A |
| `era5/EU_2026-09` | hourly winds only, no stored roughness, box 36 to 72 N, 12 W to 31.5 E | B |
| `era5/AU`, `era5/NZ`, `era5/CL`, `era5/AR` | raw monthly hourly winds | B |
| `era5/US_daily`, `era5/BR_daily` | daily `wnd100m` and `roughness`, no 10 m winds | C |

`era5/EU` is kept so the rows published before the change stay reproducible
against the input that produced them. The scorecard states the treatment per
row, and every run's manifest records the treatment it applied, with the limit
noted above that it cannot yet distinguish A from C.
