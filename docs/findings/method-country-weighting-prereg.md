# Does weighting each training year by that year's grid improve the country rows?

**Date:** 2026-10-16. Registered before any Stage 0 or Stage 1 run (drafted
2026-09-25, revised 2026-09-29 and 2026-10-16).
**Scope:** the six scoreable country-level rows (FR, BE, IE, SE, NO, ES), with
the scorecard configs (`configs/regions/scorecard/<stem>_country.toml`) on the
licensed library (`input/combined`, `power_curves.csv` sha256 `689cfee7…`), and
the country grids as rebuilt on 2026-10-16: on the corrected fleet register
(`acfbbfa`, #76), with offshore projects at their own grid points and Hywind
Tampen excluded from Norway (`fc8c7f5`, `6b3e6d9`, branch
`offshore-grid-regen`, #100; files and checksums in
`output/offshore_regen_2026-10-16/`). Training years 2015-21 (IE 2017-21), test
year 2023. Italy and Portugal are suspended (#57) and are not in the study.
Issue #67. Terms follow `CONTEXT.md`.

**Who decided what.** The thresholds, row rule, bootstrap and gates are the
maintainer's decisions of 2026-09-25 to 29. Two decisions were still open on
2026-09-29: the year convention below, and how Belgium's offshore fleet is
placed. On 2026-10-16 the maintainer asked for open work to proceed overnight
without them, and Claude Code took both that night, each as the option already
recommended on issue #67 and in the 2026-09-29 draft (mid-year; offshore sites
at their own points). The maintainer had not reviewed either when this was
committed. If the maintainer changes either, the change is a dated amendment
to this file made before any Stage 0 output exists. Once a Stage 0 file exists,
neither can change: an override after that starts a new registration, and this
one's result is still reported.

**Condition A's grids are pinned by checksum.** The end-of-year grids written
on 2026-10-16 are listed with their sha256 in
`output/offshore_regen_2026-10-16/after.sha256` (80 files, 2015-24 for the
eight rows). The pull request that adds the mid-year grids must leave every
one of them byte-identical, and the Stage 0 driver refuses to run unless A's
grid is `<c>_grid_points_<last training year>.csv` with the listed checksum.

**The study asks whether simulating each training year with the fleet as it
stood during that year, instead of the last training year's fleet for every
year, lowers the corrected national monthly RMSE on the 2023 test year.** It
runs in two stages. Stage 0 fits nothing and can stop the study. Stage 1 fits
three conditions and is read against two gates, G1 on skill and G2 on fit
quality. The gates, row rules, bootstrap and readings below are fixed before any
number they judge exists; what had been seen before registration is listed in
its own section.

## Why

`EntsoeFileSource.fleet_year` gives the training split the grid of the last
training year (2021), and every training year is simulated with it. The
observed series comes from each year's real fleet. Where the fleet grew or
moved over the window, the early years are simulated with capacity that was not
there yet, and the scalar and offsets are fitted against that mismatch.

## Which fleet a year means

A grid's year is a convention, and the two sides of the comparison use
different ones. The per-year grids
(`input/observations/country/grid_points/<c>/<c>_grid_points_<year>.csv`)
count a project from its start year, so each is the fleet at the end of its
year. ENTSO-E's installed capacity, the observed denominator of every row but
Ireland, is dated 1 January.

Condition B uses **the fleet at mid-year**: a project counts at full capacity in
a year it ran throughout, at half capacity in its start year and in its
retirement year (a quarter if both fall in one year), and at full capacity in
every year if it is still undated after the register backfill. These grids are
`<c>_grid_points_<year>_midyear.csv`, with the same points, clusters, models and
heights as the end-of-year files, so only capacity differs. The point set is the
end-of-year build's, so a project that counts in a mid-year fleet but sits on a
site no end-of-year grid in 2015-24 carries (an offshore farm retired in 2015,
say) has no point; its capacity is left out, and the builder reports the
megawatts left out per row and year. Condition A keeps
the scorecard's grid, the end-of-year 2021 file, for every training year, so
the difference between A and B includes the change of convention in 2021 as
well as the weighting of 2015-20; both are part of what B tests.

## Row set and the denominator rule

Each row's observed capacity factor is generation over an installed-capacity
denominator, and per-year weighting only helps if the grid tracks that
denominator. So the row set is decided by a rule on the denominator, fixed here,
before the table it is applied to is built.

For each row and training year Y:

- **Grid total**: the sum of `capacity` (MW) in `<c>_grid_points_<Y>_midyear.csv`.
- **Observed denominator**: the mean of the 1 January figures for Y and Y+1,
  each read as `capacity_mw` in the January rows of that year in the training
  observation file (`<c>_train_2015_2021.csv`, or `_aggregated.csv` for NO and
  SE; `ie_train_2017_2021.csv` for IE). No file carries a 2022 figure, so the
  comparison covers 2015-20 (IE 2017-20), and 2021 is not compared.

A row is **excluded** if, in any compared year, the two disagree by more than
**10%** (|grid − observed| / observed), or if its 1 January figures are
**flat**: equal to within 1 MW in three or more consecutive training years. An
excluded row is still run and reported, separately, and does not count toward
any gate.

IE's denominator is built from the same register as its grid
(`repair_country_capacity.py`, on the end-of-year fleet), so its agreement is
not independent evidence that the grid tracks the fleet; it is reported with
that note.

**Scoreable rows** are the rows the rule keeps. FR (maximum scalar 4.61) and NO
(3.05) are degenerate fits in the scorecard before the offshore change, and
whatever their fit on the offshore grids they stay in if the rule keeps them,
marked with a dagger. **If fewer than three rows are scoreable, the study stops
here** and reports the table: a majority of one or two rows is not a test.

## Stage 0: no fitting

For each row and training year, simulate the uncorrected national monthly
capacity factor twice, with A's grid (end-of-year 2021) and with that year's
mid-year grid, on the same ERA5, curves and months. Over the training window
report:

- **RMS difference** between the two monthly series;
- **scalar shift**: |mean(mid-year) / mean(A's grid) − 1|, the change the
  per-year grids imply for a training-window scalar.

A row **clears** Stage 0 if its scalar shift is at least **1%** and its RMS
difference is at least **0.005**. **The study stops after Stage 0 unless a
strict majority of the scoreable rows clears it**: floor(n/2) + 1 of n, so 3
of 4 or 5, 4 of 6. The Stage 0 count and G1's are the same number.

Stage 0 also reports, per row, before any fit:

- **The extremes.** The grid points with the smallest non-zero and the largest
  2021 capacity, with their capacity in each training year and their capacity
  share under each weighting; and every grid point or cluster with zero
  capacity in A's grid or in any training year's mid-year grid.
- **The signed ratio** mean(mid-year) / mean(A's grid), beside the absolute
  shift the kill rule reads, for the mechanism reading below.
- **The capacity left out** of each mid-year grid because it had no point
  (see "Which fleet a year means"), per year.
- **Which rows can move at all.** A row with one cluster (N=1) can respond only
  through its scalar; a row with N>1 also through the capacity mix across
  clusters and the joint offset fit.

The Stage 0 driver refuses to run unless the input root is `input/combined`
and every grid point's model is in the curve table, so no point is simulated
on a substitute curve.

## Stage 1: the conditions

Run only if Stage 0 passes.

| Condition | Weighting | Training years |
|---|---|---|
| A | last-year: the end-of-year 2021 grid for every year, re-run at the study commit | 2015-21 (IE 2017-21) |
| B | per-year: each year's mid-year grid | 2015-21 (IE 2017-21) |
| C | last-year, as A | 2019-21 |

Everything else is fixed per row: the scorecard config's cluster counts and
time slices, ERA5, curves, and the 2023 test grid (`fleet_year` of the test
split is the test year in every condition, end-of-year as the scorecard). C
separates weighting from recency: a gain from B that C also gets comes from
fitting to recent years, not from the weighting.

**Implementation, reviewed as pull requests before the stage that needs it:**

- Before Stage 0: `weight_country_grid_points.py` writes the mid-year grids
  beside the end-of-year ones, from the same fleet records.
- Before Stage 1: `[observations] fleet_weighting = "per-year"`, with
  `"last-year"` the default. Turbine-level configs refuse the key, and so does
  any correction model without the capability, declared as a ClassVar flag.
- The key is recorded in the manifest and checked by the evaluate-time training
  check. The manifest records the sha256 of every per-year grid file read.
- Per-year weighting applies within and across clusters: in each cluster's
  capacity-weighted mean and in the joint national offset fit. Per-(ID, year)
  capacity is carried on `gen_cf` in `train_set`, and each period's `clus_info`
  is built from that year's capacity. `CorrectionModel.fit` is unchanged.
- A cluster with zero capacity in a year is skipped in that year, on the
  per-year path only. The existing joint-fit pins stay bit-identical.
- The per-year grid loader is an optional `ObservationSource` method. It raises
  on a missing year, or on any ID, cluster, model, lon, lat or height that
  differs across years; only capacity may change.
- C's observations are the 2019-21 months of each row's 2015-21 training file,
  checked equal row by row.

**Run paths.** The condition is in the path, so no two conditions share a run
directory: `output/country_weighting_<date>/<condition>/<CODE>/`, with
`stage0/` beside them.

## Controls, checked before any gate is read

- **A reproduces the scorecard.** Condition A's `metrics.csv` for every row
  must match the run on the offshore grids
  (`output/offshore_regen_2026-10-16/after/<CODE>/evaluate-2023-after/metrics.csv`),
  which becomes the scorecard's when #100 merges.
  A row that does not match is reported and not gated; a mismatch means the
  study commit changed something other than the weighting.
- **B moved what it must move.** In every row that cleared Stage 0, B's fitted
  scalars must differ from A's. A row where they are bit-identical did not
  receive the condition, and is reported as such and not gated.
- **No curve substitution.** Every run reads `input/combined` and records 0% of
  capacity substituted; a country-level run refuses otherwise.

## Scoring

The metric is the corrected national monthly RMSE on the 2023 test year, on
each row's reported variant: FR fixed N=10, BE season N=3, IE season N=1, SE
fixed N=4, NO fixed N=4, ES fixed N=4. Every variant of every condition is
reported in full; only the reported variant is gated.

A refused factor's grid points get no corrected values. The national capacity
factor is then the capacity-weighted mean over the points that have one (the
harness's `weighted_mean`, which leaves a missing value out of both the sum and
the weights), and a month in which no cluster has a value is missing. Paired
comparisons use the months both conditions score; any month dropped is counted
and reported. The capacity share behind refused factors is reported per
condition and row.

**Bootstrap:** `pyvwf.harness.bootstrap`, 1,000 paired draws over the 12 test
months, seed 20260911, 95% percentile intervals on the difference in RMSE, with
the same draws for every condition of a row.

## Gates, fixed now

- **G1 (skill).** B lowers the corrected national RMSE against A, with the
  interval on B − A below zero, in a strict majority of the scoreable rows
  (floor(n/2) + 1 of n). In no scoreable row does B raise it with an interval
  above zero.
- **G2 (fit quality).** B introduces no degenerate fit that A does not have, and
  refuses no factor A kept. Refused factors and each cluster's `n_years` are
  reported for every condition.

G1 and G2 both have to pass for the study to support per-year weighting.

## Readings, fixed now

- **Weighting helps** if G1 and G2 pass and B beats C (the interval on B − C
  below zero) in the rows that carry G1.
- **Weighting and recency are not separated** wherever B does not beat C. That
  sentence travels with any gain reported for those rows.
- **Weighting does not help** if G1 fails. That is the finding, reported with
  the full table; the red team expects G1 to be underpowered with 12 test
  months, so a failure is read as "no effect shown", not as "no effect".
- **Weighting harms fit quality** if G2 fails, whatever G1 says.
- **Secondary, not gating:** G1 is also read over the scoreable rows without a
  degenerate fit, and reported beside the main reading.
- **Mechanism.** The effect is attributed from the fitted scalars and offsets,
  which are deterministic, not from whether two bootstrap intervals overlap.
  For an N=1 row the fixed scalar under B should move by about the inverse of
  the Stage 0 scalar shift; a row whose scalar moved in the other direction is
  reported as unexplained.

No prediction of which rows pass is registered.

## What was seen before registration

This registration was drafted on 2026-09-25 and revised before it was
committed. In between, data preparation for the study showed figures the rules
above apply to. They are listed so a reader can judge what the rules could have
been fitted to.

- **Condition A's results.** Every figure of A on the grids before the
  offshore change is the scorecard's, public before the study was drafted. A on
  the offshore grids had been trained and evaluated in all eight rows when this
  was committed (`output/offshore_regen_2026-10-16/after/`), but none of its
  metrics had been read.
- **Grid totals.** The GWPT fleet totals of every country for 2015, 2019 and
  2023, printed while pins were re-recorded (`acfbbfa`); each row's end-of-year
  grid total for 2015 and 2024, printed while the grids were rebuilt
  (`output/grid_regen_2026-09-29/regen_grids.log`); and on 2026-10-16 each
  row's end-of-year totals for 2015, 2021 and 2023 with the capacity on the new
  offshore points (`output/offshore_regen_2026-10-16/grid_diff.txt`). No
  mid-year total has been computed.
- **Ireland's denominator.** Its 2017-21 capacities and mean observed capacity
  factor, printed when it was rebuilt from the register
  (`output/grid_regen_2026-09-29/repair_ie.log`). It is built from the same
  register as the grid, as noted above.
- **Sweden's denominator.** Flat at 8,354 MW over 2015-19, known since the
  #47 audit and stated in issue #67 and in
  `scripts/region_tools/repair_country_capacity.py`.
- **Not seen:** the ENTSO-E denominators of FR, BE, NO and ES in any year, and
  any Stage 0 quantity.

The thresholds (10%, flat in three or more years, 1% and 0.005, the strict
majority, the bootstrap) were chosen on 2026-09-25, before any of the grid and
denominator figures above were printed. The year convention and the choice of
1 January figures for Y and Y+1 were drafted on 2026-09-29, after them, and
taken on 2026-10-16 as described at the top.

## Protocol

From the repository root, on a clean tree, one row per process, through the run
lock, with `PYVWF_INPUT=input/combined`:

1. The mid-year grids: the pull request that writes them, then the files.
2. Build and report the denominator table; apply the row rule.
3. Stage 0: `scripts/studies/method-country-weighting/stage0.py`, per row, into
   `output/country_weighting_<date>/stage0/`. Report the table; apply the kill
   rule.
4. If Stage 0 passes: the implementation pull request, reviewed and merged.
5. Stage 1: train and evaluate A, B and C per row. Controls first, then the
   full tables, then the gates.

The findings document, `method-country-weighting.md`, gives the full metrics
tables for every condition, row and variant before any reading.

## Caveats

- One test year, 12 months. G1 is likely underpowered; a pass is screening
  evidence, not an accredited result.
- Each grid carries one representative Vestas turbine per country, at one hub
  height, offshore points included, so per-year weighting changes where
  capacity sits and how much there is, not which machines.
- The mid-year fleet is a convention. A project that came online late in its
  start year counts at half capacity for a year it barely ran, and the observed
  side divides a whole year's generation by a 1 January capacity.
- About 2.6 GW of France's onshore fleet and 4.1 GW of Spain's have no start
  year after the register backfill and count in every year, which flattens the
  growth B can show there.
- Stage 0 uses the training simulation as it stands: an hour whose corrected
  or uncorrected wind leaves the power curve drops out of that point's monthly
  mean rather than counting as zero output (the treatment
  `method-joint-fit-reachability.md` found behind most of Italy's and
  Portugal's refusals). Both conditions share it.
- The offsets of an N>1 national fit are under-determined against one national
  series per period; B changes their weights, not that.
- FR and NO are degenerate fits in the scorecard. A gain in either is reported
  with its fit quality.
- C trains on three years (2019-21) against A's and B's seven (IE five), so it
  may refuse more factors; its refusals are reported, and a row where C refuses
  the reported variant's factors cannot separate recency there.
