# Does weighting each training year by that year's grid improve the country rows?

**Date:** 2026-10-16
**Scope:** the six country-level rows the study registered (FR, BE, IE, SE, NO,
ES), training years 2015-21 (IE 2017-21), test year 2023, on the grids rebuilt
on 2026-10-16 with offshore farms at their own points (#100). Registered in
`method-country-weighting-prereg.md` (`eaddb85`, editorial fix `c52f516`)
before any of the figures below existed. Issue #67. Terms follow `CONTEXT.md`.

**Correction notice, 2026-10-25: France's and Belgium's rows were computed on
observation files later found defective, and their figures are withdrawn; on
the rebuilt files both rows are still excluded, so the stop stands.** The table
read the training files as they stood on 2026-10-16 (`scorecard.md`, notices
of 2026-10-22 and 2026-10-23). Belgium's held onshore generation and capacity
alone in 2015-19 and onshore plus offshore in 2020-21, so its 1 January figure
stepped where the two fetches were joined. France's capacity was ENTSO-E's
register, which falls 1.0 to 2.9 GW below RTE's fleet in 2018-19. Both files
have since been rebuilt, France's from RTE and Belgium's from Elia (`ods031`).
The study's driver, unchanged since `b6cc1b2`, re-run on the rebuilt files
(training years 2015-21, test year not read), with the grid totals of the
table below unchanged. Observed denominators in MW:

| Year | FR denominator | FR disagreement | BE denominator | BE disagreement |
|---|---|---|---|---|
| 2015 | 9,808 | 24.2% | 1,898 | 27.8% |
| 2016 | 11,054 | 22.4% | 1,961 | 22.0% |
| 2017 | 12,651 | 20.8% | 2,291 | 24.5% |
| 2018 | 14,288 | 18.4% | 2,890 | 28.6% |
| 2019 | 15,763 | 16.3% | 3,477 | 28.3% |
| 2020 | 17,003 | 14.7% | 4,234 | 26.7% |

France is excluded at 24.2% (was 29.7%) and Belgium at 28.6% (was 33.6%), under
each of the three January readings (France 24.2%, 22.9% and 23.5%; Belgium
28.6% under all three). Belgium's 1 January figures are equal in 2016 and 2017
only (1,961 MW), so it is excluded by disagreement, not by flatness. IE, SE, NO
and ES are unchanged to the megawatt; Spain is still the one scoreable row
against the three the registration needs, and the stop stands. Withdrawn: the
FR and BE rows of both tables, "France by up to 30%", "Belgium 34%", Belgium's
flat 1 January figures, and the reading of its denominator as stepping. Its
grid is now smaller than its denominator in every compared year, the opposite
sign from France and Norway. The grids are built from the Global Wind Power
Tracker, which lists no farm below 10 MW, and Belgium's 2023 grid holds 1,608
MW onshore against Elia's 2,991 MW monitored (`scorecard.md`, notice of
2026-10-22); that the cutoff explains the training years too is not measured
here. Data:
`output/country_weighting_2026-10-16/denominators_2026-10-25/` (the driver's
`denominators.csv`, `row_rule.json` and log on the rebuilt files, and
`recompute_denominators.py` with its output, which first reproduces the stored
FR and BE rows from the old files to within 2e-12 and recomputes the other two
readings).

**The study stops at its row rule: one of the six rows is scoreable, and the
registration needs at least three.** For five rows the mid-year grid total and
the observed capacity-factor denominator disagree by more than 10% in at least
one compared year (France by up to 30%, Belgium 34%, Ireland 17%, Sweden 54%,
Norway 31%), and Belgium's and Sweden's 1 January figures are also flat over
three or more consecutive years. *[2026-10-25: France's and Belgium's figures
are from the old observation files; on the rebuilt files they are 24% and 29%,
and Belgium's are not flat. See the correction notice.]* Only Spain stays
within 10% (at most 8.3%). So the question is not tested here: Stage 0 and
Stage 1 did not run, nothing was fitted, and the 2023 test year was not read.
What the table does show is that the grid capacity these rows are simulated
with does not track the capacity their observations are divided by, in five of
six rows.

## The table

For each row and compared training year Y: the sum of `capacity` in
`<c>_grid_points_<Y>_midyear.csv`, against the mean of the 1 January
`capacity_mw` figures for Y and Y+1 in the row's training observation file.
2021 is not compared, because no file carries a 2022 figure. Rule as
registered: excluded if the two disagree by more than 10% in any compared year,
or if the 1 January figures are equal to within 1 MW in three or more
consecutive training years. All values in MW.

| Row | Year | Mid-year grid | 1 Jan Y | 1 Jan Y+1 | Observed denominator | Disagreement |
|---|---|---|---|---|---|---|
| FR | 2015 | 12,181 | 10,322 | 10,322 | 10,322 | 18.0% |
| FR | 2016 | 13,529 | 10,322 | 11,761 | 11,042 | 22.5% |
| FR | 2017 | 15,278 | 11,761 | 13,569 | 12,665 | 20.6% |
| FR | 2018 | 16,915 | 13,569 | 12,518 | 13,044 | 29.7% |
| FR | 2019 | 18,334 | 12,518 | 16,592 | 14,555 | 26.0% |
| FR | 2020 | 19,508 | 16,592 | 16,592 | 16,592 | 17.6% |
| BE | 2015 | 1,370 | 1,249 | 1,249 | 1,249 | 9.7% |
| BE | 2016 | 1,530 | 1,249 | 1,249 | 1,249 | 22.5% |
| BE | 2017 | 1,730 | 1,249 | 1,745 | 1,497 | 15.6% |
| BE | 2018 | 2,065 | 1,745 | 1,979 | 1,862 | 10.9% |
| BE | 2019 | 2,494 | 1,979 | 4,671 | 3,325 | 25.0% |
| BE | 2020 | 3,102 | 4,671 | 4,671 | 4,671 | 33.6% |
| IE | 2017 | 2,436 | 2,752 | 3,096 | 2,924 | 16.7% |
| IE | 2018 | 2,924 | 3,096 | 3,363 | 3,229 | 9.4% |
| IE | 2019 | 3,229 | 3,363 | 3,645 | 3,504 | 7.8% |
| IE | 2020 | 3,504 | 3,645 | 3,722 | 3,683 | 4.9% |
| SE | 2015 | 3,838 | 8,354 | 8,354 | 8,354 | 54.1% |
| SE | 2016 | 4,528 | 8,354 | 8,354 | 8,354 | 45.8% |
| SE | 2017 | 4,910 | 8,354 | 8,354 | 8,354 | 41.2% |
| SE | 2018 | 5,307 | 8,354 | 8,354 | 8,354 | 36.5% |
| SE | 2019 | 5,931 | 8,354 | 9,648 | 9,001 | 34.1% |
| SE | 2020 | 7,276 | 9,648 | 9,648 | 9,648 | 24.6% |
| NO | 2015 | 821 | 627 | 627 | 627 | 30.8% |
| NO | 2016 | 828 | 627 | 692 | 660 | 25.4% |
| NO | 2017 | 1,000 | 692 | 904 | 798 | 25.4% |
| NO | 2018 | 1,297 | 904 | 1,232 | 1,068 | 21.4% |
| NO | 2019 | 1,932 | 1,232 | 2,262 | 1,747 | 10.6% |
| NO | 2020 | 3,105 | 2,262 | 3,866 | 3,064 | 1.3% |
| ES | 2015 | 24,058 | 22,740 | 22,740 | 22,740 | 5.8% |
| ES | 2016 | 24,073 | 22,740 | 22,772 | 22,756 | 5.8% |
| ES | 2017 | 24,092 | 22,772 | 22,813 | 22,792 | 5.7% |
| ES | 2018 | 24,244 | 22,813 | 22,834 | 22,823 | 6.2% |
| ES | 2019 | 25,241 | 22,834 | 24,447 | 23,640 | 6.8% |
| ES | 2020 | 26,485 | 24,447 | 24,447 | 24,447 | 8.3% |

| Row | Largest disagreement | Flat 1 January figures | Verdict |
|---|---|---|---|
| FR | 29.7% | no | excluded |
| BE | 33.6% | yes (2015-17, 1,249 MW) | excluded |
| IE | 16.7% | no | excluded |
| SE | 54.1% | yes (2015-19, 8,354 MW) | excluded |
| NO | 30.8% | no | excluded |
| ES | 8.3% | no | scoreable |

*[2026-10-25: the FR and BE rows of both tables read the old observation files
and are withdrawn; the correction notice gives them on the rebuilt files.]*

Data: `output/country_weighting_2026-10-16/denominators/` (`denominators.csv`,
`row_rule.json`, `reading_check.txt`), written at `b6cc1b2` from a clean tree
through the run lock; the mid-year grids at `0d2ff5e`
(`output/country_weighting_2026-10-16/midyear_grids.log`), with 0 MW left out
in every row and year, and the 80 end-of-year grids byte-identical to the
checksums the registration names. No power curve was read: the row rule uses
grid files and observation files only.

Ireland's denominator is built from the same register as its grid
(`repair_country_capacity.py`), as the registration notes, so its agreement
would not have been independent evidence; it is excluded anyway, by its first
compared year.

## What the stop means, and what it does not

The registration fixed this outcome before the table existed: with fewer than
three scoreable rows, "a majority of one or two rows is not a test", and the
study reports the table and stops. So there is no finding on whether per-year
weighting helps. The red team's expectation that G1 would be underpowered was
not reached either; the study ended earlier, on its inputs.

The disagreements have two different shapes:

- **The grid larger than the denominator, persistently.** France (18% to
  30%) and Norway (31% falling to 1%) carry more capacity in the Global Wind
  Power Tracker than in ENTSO-E's installed-capacity figures, in every compared
  year. A fixed ratio would not matter to a capacity factor, but these ratios
  move from year to year, so per-year weighting would track the tracker's
  growth, not the denominator's. *[2026-10-25: on RTE's denominator France's
  disagreement is 15% to 24%, falling every year, and the grid is still the
  larger.]*
- **The denominator stepping, or not moving at all.** Belgium's figure holds
  at 1,249 MW for three years and then more than doubles between 1 January 2019
  and 1 January 2020, while the grid grows steadily; Sweden's holds at 8,354 MW
  for five years, the defect the 2026-09-15 correction notice in
  `method-country-level.md` names. *[2026-10-25: Belgium's step was the old
  file's join of onshore figures (2015-19) to onshore plus offshore (2020-21),
  and the hold was in those onshore figures, not the fleet. On Elia's figures
  the denominator is flat only from 1 January 2016 to 1 January 2017, and is
  22.0% to 28.6% above the grid in every compared year.]*

## Checks made after the table was seen

- **The reading of "1 January figure".** The driver reads it at the first
  timestamp of the year's January rows. Several January series step within the
  month (France's 2016 January rows span 1,439 MW *[2026-10-25: in the old
  file; 125 MW in RTE's, whose capacity is interpolated through the year]*),
  so the table was re-read with the last January timestamp and with the
  January median. The verdict is the same under all three: five rows excluded,
  Spain scoreable at 8.3%, 6.5% and 6.5% (`reading_check.txt`).
- Nothing else was changed or re-run after the table was seen.

## Caveats

- The rule compares a mid-year grid with the mean of two 1 January figures,
  both conventions the registration fixed. A different convention would give a
  different table; the registration says how a change would be handled, and
  none was made.
- The registration's two decisions taken on 2026-10-16, the year convention and
  offshore sites at their own points, were taken by Claude Code overnight and
  had not been reviewed by the maintainer when this was written (see the
  registration).
- One row (Spain) passing says the method could be tested where the register
  and the denominator agree; a future registration could start from a row set
  chosen on that criterion before any skill figure is read.
- Screening-level throughout; no correction was fitted and no skill measured.
