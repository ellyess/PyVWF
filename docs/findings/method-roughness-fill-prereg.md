# Year-local roughness fill: registered change, rows and gates

**Date:** 2026-10-22 (registered on commit, before any run of the change)
**Scope:** how the per-timestep roughness derived at load (the second roughness
route, `prep_era5`) fills the hours its inversion leaves undefined. Terms follow
`docs/CONTEXT.md`; the routes are in `docs/design/roughness-temporal-treatment.md`.

**Correction notice, 2026-10-25: four rows' observation files were rebuilt
after the Outcome, so R1 cannot be met for France as written, and R2 compares
Belgium, Italy and Portugal with runs on their old files.** Every run here read
the files then in `input/combined/`. France's 2023 test file left out offshore
output and its 2018-19 capacity fell 1.0 to 2.9 GW below the fleet; Belgium's
held onshore output alone in 2015-19 and 2023 and onshore plus offshore in
2020-21; Italy's and Portugal's 2023 test files left out offshore output
(`scorecard.md`, notices of 2026-10-22 and 2026-10-23 and the note under its
suspended rows). Each row has since been rerun on its rebuilt files, on the
record-wide fill. R1 asks France's refreshed `metrics.csv` to equal the
Outcome's new-fill one, made on the old files, and stops the refresh on a
mismatch, so on the rebuilt files it would stop for a reason other than the
fill. The amendment of 2026-10-25 below, made after the outcome, says what the
four rows are compared with instead. No gate or threshold changes.

G3 stands, as do the Outcome's tables as measured on the old files, G4's breach
included. That breach, FR country N=10 season (training years 2015-2021, test
year 2023), is a difference between two simulations of the same twelve months,
so its ΔMBE does not depend on the observations: re-scored against the rebuilt
2023 file it is again -1.61e-4, while ΔRMSE goes from -2.3e-5 to -2.9e-6. But
those factors were fitted to the old training file. Fitted to RTE's, France's
N=10 scalars are 4.71 fixed and 9.62 season, not 4.61 and 9.94, and the fill's
effect on those fits is not measured: the breach stands in kind, and its size
on the rebuilt files is not known. Data: `output/roughness_fill_2026-10-25/`
(re-score script and output; it reproduces both Outcome figures on the old
file) and `output/fr_rte_2026-10-23/promote/train-promoted-2026-10-24/`.

**Correction notice, 2026-10-22: the Rows table called FR country N=10 "not
degenerate", and it is degenerate.** `docs/findings/scorecard.md` already marks
France's N=10 fixed fit with a dagger for a scalar of 4.61, outside the 0.2 to
3.0 a degenerate fit is defined by (`docs/CONTEXT.md`), and in this study's own
runs the N=10 fits reach scalars of 4.61 (fixed) and 9.94 (season).
*[2026-10-25: those runs read France's old files; on RTE's the scorecard daggers
4.71, and the season fit reaches 9.62. See the notice of that date.]* The error
was in the registration, made before any run; it was found after G4's results
were seen, while reading the row that breached G4. The Rows table keeps the
wrong entry, marked. G4 is judged as registered, with FR among the gating
rows; the Deviations section records the breach and what this misclassification
would have changed, labelled post hoc.

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
| FR country N=10 | `fr_country.toml` | country | `input/combined` | 2015-2021 | 2023 | not degenerate [wrong, corrected 2026-10-22: degenerate †, scalar 4.61; see the correction notice] [2026-10-25: 4.71 on RTE's files] |
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

## Outcome, 2026-10-22

Run on the uncommitted change, on top of the registration commit. "Old" is the
record-wide fill (the code at the registration commit), "new" the year-local
fill. Each condition ran in its own process, its condition in the run path.

### G3, correctness: passes

The oracle tests pass on the new fill. On the old fill, the trailing-gap, the
record-wide comparison, the undefined-year and the fill-attribute tests fail;
the no-margin window test fails once the old fill's margin is removed; the
leading-gap test passes on both, since the old fill also filled a year's first
gap from inside the year.

Data: `output/roughness_fill_2026-10-22/diffmask.jsonl`. Every other variable,
coordinate and dtype is identical in all four reads.

| Row | Read | Years | Cell-days | Differing | Predicted | Differing outside prediction | Predicted but equal |
|---|---|---|---|---|---|---|---|
| DK k100 | training | 2015-2019 | 776,050 | 172 | 172 | 0 | 0 |
| DK k100 | test | 2020 | 155,550 | 34 | 40 | 0 | 6 |
| CL k10 | training | 2021-2023 | 2,953,215 | 129 | 155 | 0 | 26 |
| CL k10 | test | 2024 | 987,102 | 27 | 27 | 0 | 0 |

Why 32 predicted cell-days came out equal under both fills was not checked.

### G4, movement: breached on FR

Data: each region's `metrics.csv` under
`output/bench/roughness-fill-g4/{old,new}/<CODE>/evaluate-<year>-g4-{old,new}/`,
and `factors_*.csv` under `.../train-g4-{old,new}/`, tabulated in
`output/roughness_fill_2026-10-22/g4_tables.txt`. Curve library, from each
run's `curve_resolution.csv`: DK and DE on `input/combined`, with curves of
both `open` and `external` origin; FR and ES on `input/combined`, one
`external` curve each; CL on `input`, `open` curves only. No model key was
substituted in any run. Every row of each `metrics.csv`, uncorrected first, on
the test year (Δ is new minus old):

| Row (training years / test year) | Variant | k | Time slice | RMSE old | RMSE new | ΔRMSE | MBE old | MBE new | ΔMBE | n | Max scalar | Failed offsets |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DK k100 (2015-2019 / 2020) | uncorrected | 1 | none | 0.148239 | 0.148239 | -1.0e-08 | 0.111780 | 0.111780 | -1.5e-08 | 64,090 | | |
| DK k100 (2015-2019 / 2020) | affine-wind | 100 | season | 0.085303 | 0.085303 | 5.6e-09 | 0.022330 | 0.022330 | -2.7e-08 | 64,090 | 1.135 | 0 |
| CL k10 (2021-2023 / 2024) | uncorrected | 1 | none | 0.109976 | 0.109976 | 0 | -0.015355 | -0.015355 | 0 | 628 | | |
| CL k10 (2021-2023 / 2024) | affine-wind | 10 | fixed | 0.103648 | 0.103648 | 0 | 0.001201 | 0.001201 | 0 | 628 | 3.246 | 2 |
| CL k10 (2021-2023 / 2024) | affine-wind | 10 | season | 0.099659 | 0.099659 | 0 | 0.003138 | 0.003138 | 0 | 628 | 4.045 | 8 |
| DE k100 (2015-2018 / 2019) | uncorrected | 1 | none | 0.086113 | 0.086113 | -1.1e-07 | 0.042733 | 0.042732 | -6.8e-08 | 54,114 | | |
| DE k100 (2015-2018 / 2019) | affine-wind | 100 | fixed | 0.057044 | 0.057044 | -1.1e-07 | 0.000606 | 0.000606 | 6.7e-09 | 54,114 | 1.346 | 1 |
| FR country (2015-2021 / 2023) | uncorrected | 1 | none | 0.020404 | 0.020404 | 1.9e-09 | -0.003451 | -0.003451 | -3.1e-08 | 12 | | |
| FR country (2015-2021 / 2023) | affine-wind | 1 | fixed | 0.029097 | 0.029097 | -1.8e-07 | 0.013697 | 0.013697 | -2.4e-07 | 12 | 1.066 | 0 |
| FR country (2015-2021 / 2023) | affine-wind | 10 | fixed | 0.025954 | 0.025947 | -6.7e-06 | 0.010624 | 0.010616 | -7.8e-06 | 12 | 4.606 | 0 |
| FR country (2015-2021 / 2023) | affine-wind | 1 | season | 0.026969 | 0.026969 | -3.4e-08 | 0.014764 | 0.014763 | -2.2e-07 | 12 | 1.235 | 0 |
| FR country (2015-2021 / 2023) | affine-wind | 10 | season | 0.023953 | 0.023930 | -2.3e-05 | 0.012109 | 0.011948 | **-1.61e-04** | 12 | 9.936 | 0 |
| ES country (2015-2021 / 2023) | uncorrected | 1 | none | 0.070853 | 0.070854 | 8.2e-07 | -0.069641 | -0.069642 | -7.3e-07 | 12 | | |
| ES country (2015-2021 / 2023) | affine-wind | 1 | fixed | 0.033926 | 0.033925 | -3.9e-07 | 0.009751 | 0.009750 | -1.2e-06 | 12 | 1.497 | 0 |
| ES country (2015-2021 / 2023) | affine-wind | 4 | fixed | 0.030240 | 0.030268 | 2.8e-05 | 0.009773 | 0.009833 | 6.0e-05 | 12 | 1.901 | 0 |
| ES country (2015-2021 / 2023) | affine-wind | 1 | season | 0.031298 | 0.031298 | 2.8e-07 | 0.009907 | 0.009906 | -8.6e-07 | 12 | 1.851 | 0 |
| ES country (2015-2021 / 2023) | affine-wind | 4 | season | 0.027111 | 0.027116 | 4.3e-06 | 0.011081 | 0.011100 | 1.8e-05 | 12 | 2.446 | 0 |

n is scored unit-months for turbine-level rows and months for country-level
ones. Factors, largest change between the two fills:

| Row | Factors file | Factors | Changed | Max \|Δscalar\| | Max \|Δoffset\| (m/s) |
|---|---|---|---|---|---|
| DK k100 | `factors_season_100.csv` | 400 | 63 | 1.85e-05 | 9.04e-05 |
| CL k10 | `factors_fixed_10.csv` | 10 | 0 | 0 | 0 |
| CL k10 | `factors_season_10.csv` | 40 | 0 | 0 | 0 |
| DE k100 | `factors_fixed_100.csv` | 100 | 95 | 1.35e-05 | 5.77e-05 |
| FR country | `factors_fixed_1.csv` | 1 | 1 | 8.73e-07 | 3.55e-06 |
| FR country | `factors_fixed_10.csv` | 10 | 10 | 1.46e-06 | 0.00541 |
| FR country | `factors_season_1.csv` | 4 | 1 | 2.38e-06 | 9.71e-06 |
| FR country | `factors_season_10.csv` | 40 | 10 | 4.08e-06 | 0.151 |
| ES country | `factors_fixed_1.csv` | 1 | 1 | 3.95e-07 | 8.41e-08 |
| ES country | `factors_fixed_4.csv` | 4 | 4 | 6.96e-07 | 0.00849 |
| ES country | `factors_season_1.csv` | 4 | 1 | 9.95e-07 | 3.65e-07 |
| ES country | `factors_season_4.csv` | 16 | 4 | 1.47e-06 | 0.0337 |

Against the registered gate: on DK every row is within 1e-4; on ES the largest
is |ΔMBE| 6.0e-5; on FR, N=10 season has |ΔMBE| 1.61e-4, above 1e-4. **G4 is
breached and adoption stops.** CL and DE, not gating, move by at most 1.1e-7,
inside their predicted 1e-3. Peak memory after training, from
`output/roughness_fill_2026-10-22/g4_runs.jsonl`: DE 6.17 GB old, 6.83 GB new,
one run each.

G5 was not run: the pins were not re-recorded, since adoption stopped.

## Deviations

**2026-10-22, after G4's results were seen; post hoc.** G4 was breached by FR
country N=10, season (training years 2015-2021, test year 2023): |ΔMBE| 1.61e-4
against the registered 1e-4. Adoption stopped, as registered, pending the
maintainer's decision.

Read after the breach, and not part of any gate:

- The breaching fit is degenerate. Its largest scalar is 9.94, and the N=10
  fixed fit's is 4.61, the value the scorecard daggers. *[2026-10-25: on the
  old files; the scorecard now daggers 4.71, from RTE's.]* The Rows table had
  classified FR N=10 as not degenerate, which was wrong (correction notice at
  the top). Had it been classified correctly, the registration's own rule
  would have reported this row rather than gated on it. That reclassification
  is made after the result and cannot pass G4.
- In the same fit, ten of forty factors changed, with offsets moving by up to
  0.151 m/s, while scalars moved by at most 4.1e-6. The N=1 FR fits and every
  ES fit moved by less than 1e-4 in MBE.
- FR's test year is scored on 12 national months.

**2026-10-22, maintainer decision, made after G4's results were seen.** The
year-local fill is adopted despite the breach. This is a decision against the
registered consequence, not a pass of G4: G4 stays recorded as breached. The
pins are re-recorded with the year-local fill and the realdata stamp is run,
as G5 asks.

## Amendment, 2026-10-22: the scorecard refresh, fixed before it runs

The fifteen route-B scorecard rows were computed with the record-wide fill.
They are re-run with the year-local fill; US and BR, which read stored daily
roughness, are not, and stay on `output/validation/refresh_2026-09-20/`. This
amendment is committed before any refresh run.

**How it runs.** Training and evaluation, each step in its own process, at one
commit on `main`, from a clean tree, through `scripts/dev/run_locked.py`, into
`output/validation/refresh_2026-10-22/<CODE>/` (the date of the run if later),
with the scorecard configs as committed in `configs/regions/scorecard/`, their
full time slices, and `PYVWF_OFFSET_WORKERS=0`. One region at a time, smallest
peak memory first, nothing else running, and peak memory logged per process.

| Row | Scorecard config | Input root | Training years | Test year | Published run | Order |
|---|---|---|---|---|---|---|
| BE country | `be_country.toml` | `input/combined` | 2015-2021 | 2023 | `offshore_regen_2026-10-16/after/BE`, `6b3e6d9` | 1 |
| PT country (suspended) | `pt_country.toml` | `input/combined` | 2015-2021 | 2023 | `offshore_regen_2026-10-16/after/PT`, `6b3e6d9` | 2 |
| IE country | `ie_country.toml` | `input/combined` | 2017-2021 | 2023 | `offshore_regen_2026-10-16/after/IE`, `6b3e6d9` | 3 |
| NZ k7 | `nz_k7.toml` | `input/combined` | 2019-2023 | 2024 | training `refresh_2026-09-20/NZ`, `967b485`; evaluation `c1_turbine_order_2026-09-24/after`, `a94670c` | 4 |
| DK k100 | `dk_k100.toml` | `input/combined` | 2015-2019 | 2020 | `refresh_2026-09-20/DK`, `967b485` | 5 |
| CL k10 | `cl_k10.toml` | `input` | 2021-2023 | 2024 | `refresh_2026-09-20/CL`, `967b485` | 6 |
| AR k10 | `ar_k10.toml` | `input` | 2021-2023 | 2024 | `refresh_2026-09-20/AR`, `967b485` | 7 |
| ES country | `es_country.toml` | `input/combined` | 2015-2021 | 2023 | `offshore_regen_2026-10-16/after/ES`, `6b3e6d9` | 8 |
| DE k100 | `de_k100.toml` | `input/combined` | 2015-2018 | 2019 | `refresh_2026-09-20/DE`, `967b485` | 9 |
| IT country (suspended) | `it_country.toml` | `input/combined` | 2015-2021 | 2023 | `offshore_regen_2026-10-16/after/IT`, `6b3e6d9` | 10 |
| FR country | `fr_country.toml` | `input/combined` | 2015-2021 | 2023 | `offshore_regen_2026-10-16/after/FR`, `6b3e6d9` | 11 |
| UK k50 | `uk_k50.toml` | `input/combined` | 2015-2018 | 2019 | `refresh_2026-09-20/UK`, `967b485` | 12 |
| SE country | `se_country.toml` | `input/combined` | 2015-2021 | 2023 | `offshore_regen_2026-10-16/after/SE`, `6b3e6d9` [2026-10-25: superseded; see the bracket under the amendment's table] | 13 |
| AU-NEM k45 | `au_nem_k45.toml` | `input/combined` | 2020-2022 | 2023 | training `refresh_2026-09-20/AU-NEM`, `967b485`; evaluation `c1_turbine_order_2026-09-24/after`, `a94670c` | 14 |
| NO country | `no_country.toml` | `input/combined` | 2015-2021 | 2023 | `offshore_regen_2026-10-16/after/NO`, `6b3e6d9` | 15 |

*[2026-10-25: the BE, PT, IT and FR published runs read observation files since
rebuilt; the amendment of 2026-10-25 names the runs that replace them.]*

Published runs are under `output/` (`output/validation/` for the
`refresh_2026-09-20` ones). Input roots are read from each published run's
manifest (`curve_library.power_curves_path`), not assumed. Peak memory
measured on 2026-10-22 for DK (4.2 GB), CL (5.6 to 6.0), DE (6.2 to 6.8), ES
(5.7 to 8.4) and FR (8.3 to 10.0); the others are estimates, with NO, the last,
the largest. If a process comes within 2 GB of the machine's 16 GB, the refresh
stops there.

**Why the five measured rows can be held to an exact answer.** On 2026-10-22
the old-fill runs of the Outcome section reproduced the published runs'
`metrics.csv` exactly, every row of every file, for DK, DE and CL (published at
`967b485`) and FR and ES (published at `6b3e6d9`): no code change between those
commits and the registration commit moves them except the fill. The commit the
refresh runs at adds only `#142`, which keeps `add_models` on an object column,
the column it already had under the local pandas 2.3.3; the realdata stamp
passed 43 of 43 on the merged code. *[2026-10-25: for FR this holds on the old
files only; see the amendment of 2026-10-25.]*

**Expectations, fixed now:**

- **R1, the five measured rows (DK, DE, CL, FR, ES).** Every row of each
  refreshed `metrics.csv` equals the new-fill `metrics.csv` of the Outcome
  section exactly. A mismatch means something other than the fill moved, and
  stops the refresh before anything is published. *[2026-10-25: for FR, read
  on the old files; see the amendment of 2026-10-25.]*
- **R2, the other ten.** Against each row's published `metrics.csv`, on every
  row of the file: the non-degenerate rows (NZ, BE, IE, SE) move by at most
  1e-4 in RMSE and in MBE; the degenerate rows (AR, AU-NEM, UK, NO) are
  predicted below 1e-3 and reported with their fit quality; IT and PT are
  reported and stay suspended.
- **R3, stop rule.** Any row that moves by more than 1e-3 in RMSE or MBE, or
  whose scored count, fleet count, accepted years, refused factors, curve
  resolution or § share changes, is not published until the movement is
  explained in a dated deviation, labelled post hoc.
- **R4, curve-match audit.** For the turbine rows, the other-brand,
  reference-curve and unverifiable shares from
  `scripts/analysis/curve_match_audit.py` are unchanged, since neither fleets
  nor curves change. A change stops as R3 does.
- **The ‡ marker** is recomputed by `scripts/analysis/baseline_bootstrap.py` on
  the refreshed turbine runs and set by its existing rule; any marker that
  changes is reported, with its interval.

**Consequences, stated in advance.** If R1 to R4 hold, the scorecard tables,
their markers and the fit-quality table take the refreshed figures, every
number citing its run directory; the 2026-10-22 notice gains a dated note
saying the rows were refreshed and by how much they moved; the paragraph that
says every row comes from one refresh says instead that fifteen come from this
refresh and US and BR from `refresh_2026-09-20`; and the README's copy of the
table follows. The diff is shown to the maintainer before it is committed.

## Amendment, 2026-10-25: rows whose observations changed, after the outcome

**Post hoc: made after G4's results and after the reruns below were seen, and
before any refresh run.** It changes what four rows are compared with, and no
threshold; R1 to R4 otherwise stand as fixed on 2026-10-22.

The four rows were rerun on their rebuilt files (correction notice of
2026-10-25) at commits that do not contain the year-local fill (`5350fac`), and
no manifest of these runs records a `fill` key, which means the record-wide
back-fill. Training years 2015-2021, test year 2023, `input/combined`:

| Row | Rebuilt files | Run on them, record-wide fill | Commit |
|---|---|---|---|
| BE country | Elia `ods031`, every split | `be_elia_2026-10-22/promote/` | `dc2ce83` (the code of `6c5272d`) |
| PT country (suspended) | 2023 test file, with offshore | `it_pt_2023_2026-10-24/PT/{train-offshore,evaluate-2023-offshore}-2026-10-24/` | `7fc0e16` |
| IT country (suspended) | 2023 test file, with offshore | `it_pt_2023_2026-10-24/IT/{train-offshore,evaluate-2023-offshore}-2026-10-24/` | `7fc0e16` |
| FR country | RTE, every split | `fr_rte_2026-10-23/promote/` | `65efa06` |

*[2026-10-25: Sweden's capacity-factor denominator was rebuilt from
Energimyndigheten's fleet after this amendment (`scorecard.md`, notice of that
date). Its runs are at `768a278`, which contains the year-local fill, so they
measure R2 directly: a control on the old files differs from Sweden's
published run by at most 6e-6 in RMSE and MBE with identical scalars, within
R2's 1e-4 for a non-degenerate row, and the run on the rebuilt files
(`se_register_2026-10-25/promote/`) is Sweden's baseline for R3. Data:
`output/se_register_2026-10-25/`.]*
*[2026-10-25: France's files were then rebuilt from ODRE's copy of the same
RTE series (`scorecard.md`, bracket in the notice of 2026-10-23), which equals
them hour by hour except one March hour a year and one 2021 value. Its runs
are at `9a5682c`, which contains the year-local fill. A control on the RTE
files differs from `fr_rte_2026-10-23/promote/` by at most 6.5e-5 in RMSE,
5.4e-5 in MBE and 9e-7 in the largest scalar, within the 1e-3 predicted above
for FR's degenerate row, so R2 is measured for FR here. The run on the ODRE
files (`fr_odre_2026-10-25/promote/`) is FR's baseline for R3. Data:
`output/fr_odre_2026-10-25/`.]*

All under `output/`. For these rows they replace the published runs of the
2026-10-22 table as the baseline of R2 and R3: against the old runs, each row
would move by its observations' change as well as the fill's.

- **R1, FR, is kept as written, as a check of the code.** The refresh also
  trains and evaluates FR at the refresh commit on an input root that differs
  from `input/combined/` only in holding France's old files
  (`fr_*.entsoe.bak.csv`), into a `control/` subdirectory that is not
  published. Every row of its `metrics.csv` must equal the Outcome's new-fill
  `metrics.csv`; a mismatch stops the refresh, as registered.
- **FR's published row, on RTE's files, is held to R2's rule for degenerate
  rows** (its N=10 fits are; correction notice of 2026-10-22): against
  `fr_rte_2026-10-23/promote/`, predicted to move by less than 1e-3, and
  reported with its fit quality. This is weaker than R1, which no row on
  rebuilt files can meet: no new-fill run on them exists to equal.
- **BE** stays among R2's non-degenerate rows (at most 1e-4); **IT and PT** are
  reported and stay suspended. Each is measured against the run above.
  *[2026-10-27: Portugal's capacity was then rebuilt from REN's month-end
  fleet (`scorecard.md`, note of that date under the suspended rows). A control
  at `4ccab24`, which contains the year-local fill, on the old files differs
  from `it_pt_2023_2026-10-24/PT/` by at most 1.7e-3 in RMSE and 1.8e-3 in MBE
  on its 3 scored months, more than R3's 1e-3; the row is suspended, so this
  publishes nothing. The run on the rebuilt files (`pt_ren_2026-10-27/promote/`)
  is PT's baseline. Data: `output/pt_ren_2026-10-27/`.]*
- **R3** applies to all four, against the same runs.

## Caveats

- The leak removed is of reanalysis input, small, and confined to the training
  years' last day; it is not a leak of observations. Its effect on any
  published number has not been measured; G4 measures it on five rows.
- Unit cells come from adapter metadata, a superset of each run's fleet, so the
  unit-cell counts are upper bounds for any one run.
- Every result rests on a single test year, and the scorecard is
  screening-level.
- Ten route-B rows will not match the code until the next scorecard refresh.
