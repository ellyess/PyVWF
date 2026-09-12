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
six apply one that varies through the year.

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
gate before the comparison runs. Until that reports, no result changes and no
treatment is called correct.
