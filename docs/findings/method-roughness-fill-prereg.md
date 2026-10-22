# Year-local roughness fill: registered change, rows and gates

**Date:** 2026-10-22 (registered on commit, before any run of the change)
**Scope:** how the per-timestep roughness derived at load (the second roughness
route, `prep_era5`) fills the hours its inversion leaves undefined. Terms follow
`docs/CONTEXT.md`; the routes are in `docs/design/roughness-temporal-treatment.md`.

**Route B's roughness fill is to be confined to each calendar year, so that a
run's roughness depends only on ERA5 from its own years. Today the last hours of
a year can take their roughness from the next year, which for seven
turbine-level rows is the test year.** What crosses is test-year reanalysis
input, ERA5 shear from the first hours of 1 January, not test-year observed
generation. At the training years' end it reaches 0 to 742 cell-hours at the
cells units interpolate from (below). The change moves results by an amount
predicted here to be about 1e-5 in capacity factor. Its second purpose is
engineering: a load of whole years becomes exact by construction, so the
January margin and the guard added by PR #139 can go.

**Everything below the pre-check is fixed before any run of the change.** The
outcome is filled in afterwards. A condition added later is labelled post hoc
and cannot pass a gate. Gates G1 and G2 were fixed in a reviewed proposal before
the pre-check ran, but that proposal was not committed, so their order against
the pre-check cannot be shown from git; the pre-check result is recorded here,
at registration, with that stated.

## The question

`log_roughness_from_shear` (`src/pyvwf/era5.py`) inverts the log wind profile
between the ERA5 10 m and 100 m winds for every hour. Where the shear is near
zero or the result would put the roughness length at 1 m or more, which
includes every inverted profile, the hour is undefined and is back-filled from
the next defined hour, along the whole loaded record, with no limit.

So when a cell's last hours of a year are undefined, they take their value from
the next year. For a training run whose test year follows its training years,
that value is test-year ERA5. **Does confining the fill to each calendar year,
back then forward, remove that dependence while leaving every other value
unchanged and moving results only negligibly?**

The fill was never chosen on evidence: it came from a notebook (`6cdac5b`,
2026-01-05) and moved into the shared function unchanged (`c891ef1`).

## The change

- In `_apply_roughness`, apply `log_roughness_from_shear` per calendar year, in
  a loop over years writing into a preallocated array, then forward-fill
  within the year. A year's leading undefined run still takes its year's first
  defined value; its trailing run takes its year's last defined value.
- The shared function, the third roughness route (`scripts/era5/combine.py`,
  already month-local), the annual-mean route and `src/pyvwf/pinn/era5_stats.py`
  are unchanged. The divergence between routes is recorded in
  `docs/design/roughness-temporal-treatment.md` and in the
  `log_roughness_from_shear` docstring.
- A cell with no defined hour in a year stays NaN, and `prep_era5` warns,
  naming the cell-years. What a NaN roughness does downstream (wind speed,
  capacity factor, scoring) is checked and stated with the change.
- `prep_era5`'s January margin, guard, whole-record reload and trim are removed;
  `years=(start, end)` loads exactly `[start-01-01, end-12-31]`. A test,
  `test_a_window_matches_the_whole_record_cut_with_no_margin`, replaces the
  three margin tests.
- `pyvwf_roughness_fill` ("within each calendar year") is set only when the
  derived treatment is applied, and recorded in the manifest's `era5_roughness`
  block. A manifest without it means the record-wide back-fill.
- The four `tests/test_pin_offset_fits.py` fixtures are re-recorded in the same
  commit, with the size of each movement.

## Rows

| Row | Scorecard config | Level | Input root | Training years | Test year | Fit quality |
|---|---|---|---|---|---|---|
| DK k100 | `dk_k100.toml` | turbine | `input/combined` | 2015-2019 | 2020 | not degenerate |
| CL k10 | `cl_k10.toml` | turbine | `input` | 2021-2023 | 2024 | degenerate † |
| DE k100 | `de_k100.toml` | turbine | `input/combined` | 2015-2018 | 2019 | degenerate † |
| FR country N=10 | `fr_country.toml` | country | `input/combined` | 2015-2021 | 2023 | not degenerate |
| ES country N=4 | `es_country.toml` | country | `input/combined` | 2015-2021 | 2023 | not degenerate |

DK, CL, FR and ES are the four pinned fits, so their movement is measured by
re-recording them. DE is added because its training years' end carries the
most leaked cell-hours at unit cells. Each is run at its scorecard
configuration. The other ten route-B scorecard rows (AR, AU-NEM, NZ, UK, BE,
IE, IT, NO, PT, SE) move by construction and are re-run at the next scorecard
refresh; the scorecard gets a dated notice saying so.

## Pre-check, run 2026-10-22, before any code change

Data: `output/roughness_fill_2026-10-22/precheck.jsonl`, written by
`output/roughness_fill_2026-10-22/precheck.py`.

For every route-B scorecard config, each of its training years and its test
year was read on the config's bbox, a month at a time, and pyvwf's inversion
mask computed per hour. Per cell: the trailing undefined run at the year's end
(under the year-local fill, the forward-fill distance), and whether the year
has any defined hour. "Unit cells" are the four grid cells around each unit in
the region's adapter metadata, every unit it lists, which is a superset of any
one run's fleet.

The mask is a copy of the function's, checked against it on DK, December 2019:
back-filling the copy reproduced `log_roughness_from_shear`'s NaN pattern and
values exactly. The DK counts at its 2019 and 2020 year ends (95 cells and 642
cell-hours; 40 and 103) match an independent count made the same day for
PR #139.

| Row | Training years | Test year | Bbox cells | Unit cells | Undefined cell-years, bbox | Undefined cell-years, unit cells | Longest trailing run at a unit cell (h) | Longest in bbox (h) | Trailing cell-hours, unit cells | Trailing cell-hours, bbox |
|---|---|---|---|---|---|---|---|---|---|---|
| DK k100 | 2015-2019 | 2020 | 425 | 217 | 0 | 0 | 10 | 10 | 374 | 926 |
| DE k100 | 2015-2018 | 2019 | 1485 | 717 | 0 | 0 | 20 | 57 | 7,334 | 14,374 |
| UK k50 | 2015-2018 | 2019 | 2793 | 435 | 0 | 0 | 24 | 27 | 1,471 | 4,064 |
| AR k10 | 2021-2023 | 2024 | 5429 | 112 | 0 | 0 | 2 | 10 | 39 | 3,228 |
| AU-NEM k45 | 2020-2022 | 2023 | 13837 | 221 | 0 | 0 | 5 | 14 | 45 | 1,263 |
| CL k10 | 2021-2023 | 2024 | 2697 | 84 | 0 | 0 | 12 | 12 | 22 | 363 |
| NZ k7 | 2019-2023 | 2024 | 2597 | 32 | 0 | 0 | 0 | 10 | 0 | 339 |
| BE country | 2015-2021 | 2023 | 180 | 86 | 0 | 0 | 10 | 10 | 699 | 1,722 |
| ES country | 2015-2021 | 2023 | 1696 | 212 | 0 | 0 | 12 | 12 | 2,478 | 14,644 |
| FR country | 2015-2021 | 2023 | 2881 | 702 | 0 | 0 | 32 | 102 | 9,752 | 26,146 |
| IE country | 2017-2021 | 2023 | 336 | 148 | 0 | 0 | 19 | 20 | 823 | 1,125 |
| IT country | 2015-2021 | 2023 | 2016 | 116 | 0 | 0 | 14 | 102 | 1,013 | 13,593 |
| NO country | 2015-2021 | 2023 | 6327 | 110 | 0 | 0 | 40 | 302 | 1,109 | 70,412 |
| PT country | 2015-2021 | 2023 | 294 | 102 | 0 | 0 | 10 | 10 | 1,005 | 3,559 |
| SE country | 2015-2021 | 2023 | 2915 | 179 | 0 | 0 | 50 | 206 | 2,324 | 40,140 |

Trailing cell-hours are summed over every year end of the row's training years
and test year. The last training year's end alone, at unit cells, is the leak
the change removes; its cells, cell-hours and longest run, from the same data:

| Row | Training years | Test year | Cells | Cell-hours | Longest run (h) | Next year is |
|---|---|---|---|---|---|---|
| DE k100 | 2015-2018 | 2019 | 100 | 742 | 20 | the test year |
| DK k100 | 2015-2019 | 2020 | 38 | 214 | 8 | the test year |
| UK k50 | 2015-2018 | 2019 | 14 | 43 | 8 | the test year |
| AU-NEM k45 | 2020-2022 | 2023 | 15 | 31 | 5 | the test year |
| AR k10 | 2021-2023 | 2024 | 11 | 12 | 2 | the test year |
| CL k10 | 2021-2023 | 2024 | 1 | 2 | 2 | the test year |
| NZ k7 | 2019-2023 | 2024 | 0 | 0 | 0 | the test year |

The eight country rows train 2015-2021 and are tested on 2023, so their
training years' end is filled from 2022, which is neither: they move with no
test-year gain. 2022 was not read by the pre-check.

The undefined-year count never fired on real data, so its positive case is
untested; it uses the same mask as the trailing runs, which was checked.

## Gates

- **G1, kill.** If any unit cell has a whole scorecard year undefined in any
  route-B row, the change is dropped and the margin stays. Bbox-only cases are
  reported and do not gate. **Outcome: 0 at unit cells and 0 over every bbox,
  all fifteen rows. Not killed.**
- **G2, fill distance.** If the longest trailing run at a unit cell exceeds
  168 h in any route-B row, stop and bring it to the maintainer. 168 h is a
  judgement, one week, not a derived figure. **Outcome: largest 50 h (SE,
  2017). Passes.** The largest bbox runs (NO 302 h, SE 206 h, FR and IT 102 h)
  are at cells no unit uses.
- **G3, correctness.** Both must hold:
  - oracle unit tests: a trailing run takes its year's last defined value; a
    run on 1 January is not filled from the previous year; a whole undefined
    year stays NaN and warns; the previous fill and the new one differ at a
    year end;
  - diff mask: old against new `prep_era5` output on DK (training read and test
    read) and CL (the same). Every (cell, day) that differs must be a day
    carrying a trailing run in the pre-check data, including the test year's
    own 31 December, and every other value is bit-identical.
- **G4, registered prediction.** For DK, FR and ES, |ΔRMSE| and |ΔMBE| on the
  test year are at most 1e-4 in capacity factor; about 1e-5 is expected,
  because the change touches one or two days per year end in a few percent of
  unit cells. CL and DE are degenerate fits and are predicted below 1e-3,
  reported with their fit quality and not gating. A breach stops adoption; any
  explanation of a breach is recorded as a dated deviation and labelled post
  hoc. The direction of the movement does not decide adoption: the change is
  made for what the fit may read, not for the score.
- **G5, stamp.** `python scripts/dev/stamp.py realdata` passes after the pins
  are re-recorded. `pinn/` is untouched, so no `pinn` stamp is needed.

## Metrics

For each row in Rows, before and after: every factor's scalar and offset, and
the test year's RMSE and MBE for the uncorrected and corrected rows of
`metrics.csv`, each with n scored. DE also records peak RSS of its training run,
in its own process, before any heavier region is re-run.

## Registered predictions

- P1: G3's diff mask holds exactly on DK and CL.
- P2: the four re-recorded pins move, and every factor change is below 1e-3 in
  scalar and 1e-3 m/s in offset for DK, FR and ES.
- P3: DK, FR and ES meet G4 at about 1e-5.
- P4: CL and DE move by less than 1e-3 in RMSE and MBE.

## Consequences, stated in advance

- G3 and G4 pass: the change lands with the margin removed, the pins
  re-recorded, a dated scorecard notice for the ten rows not re-run, and a
  CHANGELOG line.
- G3 fails: the implementation is wrong; it is fixed and G3 rerun, nothing else
  changes.
- G4 fails on DK, FR or ES: adoption stops until the movement is explained in a
  dated deviation.

## Deviations

None.

## Caveats

- The leak removed is of reanalysis input, small, and confined to the training
  years' last day; it is not a leak of observations. Its effect on any
  published number has not been measured; G4 measures it on five rows.
- Unit cells come from adapter metadata, a superset of each run's fleet, so the
  unit-cell counts are upper bounds for any one run.
- Every result rests on a single test year, and the scorecard is
  screening-level.
- Ten route-B rows will not match the code until the next scorecard refresh.
