# Validation scorecard

How much PyVWF's affine wind-speed correction reduces the error between
ERA5-simulated and observed capacity factors, per region, on a held-out year.
Every row states its training years and its single test year, and every
current row's figures are read from the `metrics.csv` of the run named under
[Reproducing a row](#reproducing-a-row); the resampled intervals, the
suspended rows and the superseded rows say where theirs come from. This is screening-level validation,
not an accredited yield assessment.

This page is the short form, from 2026-10-27. The full scorecard, with every
dated correction notice and its before-and-after tables, is kept at the
research archive tag
([`docs/findings/scorecard.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md);
Zenodo [10.5281/zenodo.23696129](https://doi.org/10.5281/zenodo.23696129)).
Each notice has one line in the [corrections log](#corrections-log). The rules
for the ‡, § and † markers are in [`README.md`](README.md).

## Turbine / plant-level (observed capacity factor per farm)

Matched turbine curves and hub heights, the affine fit at the cluster count
each row reports, fleet scope.

| Region | Fleet (test) | Train → test | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Corr r | Best cfg | Roughness | Other brand | Reference curve | Unverifiable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Germany (DE) | 4814 turbines (4807 scored) | 2015-18 → 2019 | 0.086 | **0.057** | +0.043 | +0.001 | 0.86 | k100 fixed † | per timestep | 40.0% | 8.9% | 0.0% |
| Denmark (DK) § 0.6% | 5410 turbines | 2015-19 → 2020 | 0.148 | **0.085** | +0.112 | +0.022 | 0.83 | k100 season | per timestep | 15.0% | 1.7% | 3.1% |
| Brazil (BR) | 151 complexes (140 scored) | 2021-23 → 2024 | 0.130 | **0.097** | -0.037 | -0.014 | 0.79 | k60 fixed † | per timestep, stored daily | n/a | n/a | 100.0% |
| United States (US) | 520 plants (512 scored) | 2019-21 → 2022 | 0.108 | **0.096** | +0.024 | +0.023 | 0.79 | k250 fixed † | per timestep, stored daily | 48.3% | 22.0% | 1.1% |
| Australia (AU-NEM) | 77 farms (75 scored) | 2020-22 → 2023 | 0.116 | **0.079** | +0.009 | +0.001 | 0.69 | k45 season † | per timestep | 2.8% | 84.5% | 4.5% |
| United Kingdom (UK) | 348 farms (344 scored) | 2015-18 → 2019 | 0.139 | **0.116** ‡ | +0.028 | -0.037 | 0.70 | k50 fixed † | per timestep | 21.8% | 7.5% | 0.0% |
| New Zealand (NZ) | 12 farms | 2019-23 → 2024 | 0.157 | **0.105** ‡ | -0.062 | +0.025 | 0.74 | k7 fixed | per timestep | 41.9% | 47.3% | 0.0% |
| Chile (CL) | 59 plants (53 scored) | 2021-23 → 2024 | 0.110 | **0.104** ‡ | -0.015 | +0.001 | 0.43 | k10 fixed † | per timestep | 3.5% | 91.6% | 0.0% |
| Argentina (AR) | 59 plants (57 scored) | 2021-23 → 2024 | 0.140 | **0.122** | +0.025 | +0.004 | 0.44 | k10 fixed † | per timestep | 0.2% | 96.7% | 0.0% |

Corrected RMSE is below uncorrected in all nine rows, and corrected MBE is
nearer zero in eight; the UK's moves from +0.028 to -0.037. Where a fleet is
scored on fewer units than it has, the units of refused clusters get no
corrected values and every variant, uncorrected included, is scored on the
units all variants can score.

**Best cfg** is the configuration each row has reported since it was chosen,
and it was chosen on the same test year it is scored on. In three rows another
variant of the same run now scores lower on that year: BR season k60 (0.0896,
against 0.0967), CL season k10 (0.0997, against 0.1036) and AR season k10
(0.1210, against 0.1215). Each of those three carries more failed offsets than
the reported variant. The US's two variants tie at 0.0960. The reported
configurations are kept.

**‡ The gain cannot be distinguished from zero when the test year's units are
resampled** (1,000 paired draws with replacement,
`scripts/analysis/baseline_bootstrap.py`). The RMSE gain and its 95% interval
are 0.052 [-0.032, 0.115] for NZ, on the run in the table. For the UK, 0.031
[-0.001, 0.069], and for CL, 0.006 [-0.007, 0.021], they come from the runs
before the 2026-09-20 refresh, so they differ from the gains the table implies
(the UK's is 0.023). In the UK and NZ a few units the correction makes worse
decide the result. With 12 farms, 9.3 of them capacity-effective, any one NZ
farm can decide it, in any test year. The intervals understate the
uncertainty: they are conditional on one test year, treat neighbouring units
as independent, and leave out the choice of cluster count, made on that same
year. On those earlier runs the gains of DE, DK, US and BR were resolved. AR's
was resolved only narrowly (interval 0.002 to 0.030), and its MAE gain
interval includes zero; since the intervals understate the uncertainty, a
margin of 0.002 is not a clean result. AU-NEM's, recomputed after the 2026-09-24 fix, is
0.037 [0.019, 0.058].

**§ Part of the fleet lies outside the loaded ERA5 extent, and its winds were
extrapolated; the share of capacity follows the marker.** DK's bounding box
stops at 13.5°E and Bornholm lies near 14.9°E, so 47 of the 5,446 units in its
test fleet, 0.6% of capacity, sit up to 1.64° beyond the data. The European
files cover Bornholm; the region's box stops first. Widening it would make a
different DK row, with its own configuration. No other row carries §.

**† The fit behind this row is degenerate.** `fit_quality` on the factors file
each row reports, with the calibrated bounds: scalar in 0.2 to 3.0, offsets
required to converge
([`method-scalar-bounds.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-scalar-bounds.md)).
Failed offsets counts factors, one per cluster and period, so one refused
cluster in a season fit counts four times.

| Region | Config | Max scalar | Implausible scalars | Failed offsets |
|---|---|---|---|---|
| Chile (CL) | k10 fixed | **3.25** | 1 | **2** |
| Brazil (BR) | k60 fixed | 2.87 | 0 | **2** |
| United States (US) | k250 fixed | 2.72 | 0 | **5** |
| Australia (AU-NEM) | k45 season | 2.64 | 0 | **4** |
| United Kingdom (UK) | k50 fixed | 1.86 | 0 | **2** |
| Germany (DE) | k100 fixed | 1.35 | 0 | **1** |
| Argentina (AR) | k10 fixed | 1.29 | 0 | **1** |

NZ (maximum scalar 1.81) and DK (1.14) are clean. A factor whose accepted
years are not a majority of its training years is refused: it carries no
scalar, its units get no corrected values, and it counts as a failed offset.
So every maximum scalar above covers applied factors only, and all are inside
the bounds except Chile's. Chile is the worst: two of its ten clusters are
refused (their per-year scalars reached 80 and 29), and no `min_cluster_size`
setting satisfies all three of its pre-registered gates. The aggregate figures
of the daggered rows stand, but their per-cluster factors should not be used,
and the gridded exports built from them carry a `degenerate` layer.

## Country-level (national aggregate, test year 2023)

Capacity-weighted national monthly capacity factor. Training years 2015-21 (IE
2017-21), test year 2023. Each grid simulates its own Vestas curve from the
licensed curve library (`power_curves.csv` sha256 `689cfee7…`), 0% substituted.

| Region | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Best cfg | Roughness | Substituted |
|---|---|---|---|---|---|---|---|
| France (FR) | 0.019 | 0.022 | -0.012 | +0.009 | N=10 fixed † | per timestep | 0% |
| Belgium (BE) | 0.081 | **0.018** | +0.079 | +0.001 | N=3 season | per timestep | 0% |
| Ireland (IE) | 0.045 | **0.021** | +0.036 | +0.006 | N=1 season | per timestep | 0% |
| Sweden (SE) | 0.092 | **0.027** | -0.089 | +0.015 | N=4 fixed | per timestep | 0% |
| Norway (NO) | 0.131 | **0.026** | -0.127 | -0.012 | N=4 fixed † | per timestep | 0% |
| Spain (ES) | 0.071 | **0.030** | -0.070 | +0.010 | N=4 fixed | per timestep | 0% |

**France's corrected RMSE, 0.022, is above its uncorrected 0.019: the
correction adds error,** on a degenerate fit. Norway's falls from 0.131 to
0.026, on a degenerate fit too.

**Best cfg** is again the configuration each row has reported, chosen on the
2023 test year, and in every row another variant of the same run now scores
lower on that year: FR season N=10 (0.0197, with four implausible scalars and
a maximum of 9.62, and still above France's uncorrected 0.019, as every French
variant is), BE fixed N=3 (0.0111), IE season N=3 (0.0211), SE season
N=1 (0.0199), NO season N=4 (0.0227, four implausible scalars) and ES season
N=4 (0.0271). The reported configurations are kept.

**† The fit behind this row is degenerate** (bounds as in the turbine-level
table):

| Region | Config | Max scalar | Implausible scalars | Failed offsets |
|---|---|---|---|---|
| France (FR) | N=10 fixed | **4.71** | 1 | 0 |
| Norway (NO) | N=4 fixed | **3.18** | 1 | 0 |

**A country-level fit is under-determined.** It fits N offsets against one
national series per month. An earlier reading, that the offsets repair the
scalar's cube-law overshoot, came from fits that a routing defect sent to the
per-cluster solver, and it is not re-established for the joint fit.

**Suspended rows, from 2026-09-25.** On their grids' own curves most of the
joint fits of Italy and Portugal are refused. Italy scores no month and
Portugal 3 of 12, so neither is a result. Both were rerun on corrected
observations (2026-10-24, and for Portugal again on 2026-10-27), and neither
changed that. The figures they carried until 2026-09-25, on the 100 kW
fallback curve:

| Region | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Best cfg | Was |
|---|---|---|---|---|---|---|
| Italy (IT) | 0.070 | 0.015 | -0.069 | -0.004 | N=3 season | in the table, 100% substituted |
| Portugal (PT) | 0.089 | 0.027 | -0.085 | +0.018 | N=1 season | in the table, 100% substituted |

The Netherlands is excluded: an ENTSO-E coverage defect makes its
capacity-factor series unusable.

**Superseded rows, 2026-09-13.** The published figures of every European row
before the re-run of 2026-09-13, kept for the record. All eleven ran on
`era5/EU`, the annual-mean roughness and the narrower ERA5 box. A superseded
row is not withdrawn: it states what was published, on what input, and when it
was replaced. The three that were suspended were never results, and are kept
so the correction can be checked. Belgium's figures read observations that
left out offshore generation. Each superseded row has the training and test
years of its current row. Their configurations are at the archive tag in
`configs/regions/scorecard/superseded/2026-09-13/`. On 2026-10-27, after their
runs, they gained `roughness = "stored"` (`8105277`), the treatment they had
run under, because the default changed to derived on 2026-09-20; `uk_k50.toml`
also had a comment renamed (`4d06163`).

| Region | Published uncorr RMSE | Published corr RMSE | Published uncorr MBE | Published corr MBE | Best cfg | Was |
|---|---|---|---|---|---|---|
| Germany (DE) | 0.086 | 0.057 | +0.042 | +0.001 | k100 fixed | in the table |
| Denmark (DK) | 0.147 | 0.085 | +0.110 | +0.023 | k100 season | in the table, § 0.6% |
| United Kingdom (UK) | 0.145 | 0.115 ‡ | +0.037 | -0.038 | k50 fixed | in the table |
| France (FR) | 0.171 | 0.012 | +0.165 | +0.006 | N=10 fixed | in the table |
| Belgium (BE) | 0.340 | 0.020 | +0.337 | -0.002 | N=3 season | in the table |
| Ireland (IE) | 0.172 | 0.021 | +0.168 | +0.009 | N=1 season | in the table |
| Sweden (SE) | 0.088 | 0.030 | +0.084 | -0.027 | N=4 fixed | in the table, § 0.8% |
| Norway (NO) | 0.034 | 0.039 | +0.024 | -0.030 | N=4 fixed | in the table, § 4.4% |
| Italy (IT) | 0.066 | 0.034 | +0.062 | -0.020 | N=3 season | suspended |
| Portugal (PT) | 0.110 | 0.074 | -0.097 | +0.029 | N=1 season | suspended |
| Spain (ES) | 0.135 | 0.026 | +0.130 | +0.016 | N=4 fixed | suspended |

## Curve matching (the last three turbine-level columns)

- **Other brand** is the share of a row's fitted training fleet, by capacity,
  simulated on another manufacturer's curve.
- **Reference curve** is the share on a research reference design or a generic
  composite curve, never the unit's own machine. On the open library it is
  often the only kind available, so it describes the library more than the
  matching.
- **Unverifiable** is the share where either side cannot be identified.

The shares come from `scripts/analysis/curve_match_audit.py`. Every row except
BR is matched on specific power, so the first two columns measure how far a row
rests on the assumption that specific power fixes a curve's shape:

- DK and UK use `add_models`: a fuzzy manufacturer match, then nearest specific
  power. It does not read the model designation both registers record.
- DE uses the same route, but its register has no model designation, so no
  assignment better than nearest specific power is available from it. Its
  largest case, 13.4% of capacity on Vestas turbines simulated on Gamesa
  curves, is a limit of the register rather than of the matching.
- US, NZ, CL and AR use nearest specific power within a rating band, with no
  manufacturer step. AU-NEM uses a specific-power class.
- BR assigns one uniform curve to every complex and records no manufacturer,
  so its shares are n/a.

The rule names brands, not lineages: a Bonus turbine on a Siemens curve counts
as other brand.

## What must not be overclaimed

- **Seven of the nine turbine-level rows rest on degenerate fits** (CL, BR, US,
  AU-NEM, UK, DE, AR). Only DK and NZ are clean, and NZ's gain is not resolved.
  None of this is visible in the skill metric, so check `fit_quality`, not the
  metric.
- **An aggregate that barely moves does not mean the fit did not move.** A
  change of fitting code once moved the United States' RMSE by 0.0005 while
  more than doubling its worst fitted scalar, from 20.54 to 46.39.
- **One test year per region,** and each reported configuration was chosen on
  it. Orderings of close configurations are not meaningful; the result is the
  drop from uncorrected to corrected.
- **CL and AR are caveated, not wins.** ERA5 exaggerates the north-south wind
  gradient in both. The correction removes the mean bias but adds limited
  skill. Chile is unstable at low cluster counts, and cluster count does not
  predict fit health there.
- **At the country level, France is worse corrected, Norway improves only on a
  degenerate fit, Italy and Portugal are suspended and the Netherlands is
  excluded.** Reporting any of them as a corrected region would be false.
- **Neither the country-level rows nor DE, DK, UK and US are third-party
  reproducible.** They need the licensed curve library. Each country grid
  carries one representative turbine, not the fleet's machines.
- **The US carries an unscreened curtailment confound** (ERCOT and SPP), and
  its near-zero fleet MBE is partly an aggregation artefact.
- **AU-NEM's farm-level behaviour in curtailed South Australia is not
  established.** The figures behind that caveat were withdrawn on 2026-09-24
  and not re-measured.
- Screening-level throughout: not MEASNET- or DNV-accredited, and not
  investment advice.

## Reproducing a row

Each row reruns from its configuration in `configs/regions/scorecard/`
(`<code>_k<N>.toml` or `<code>_country.toml`), not from the maintained
`configs/regions/<code>.toml`: for seven of the nine turbine-level regions the
maintained sweep does not contain the reported cluster count. Run it at the
row's commit, from the repository root, through `scripts/dev/run_locked.py`,
with `PYVWF_INPUT=input/combined` for every row on the licensed library:

```bash
python scripts/dev/run_locked.py -- pyvwf-validate train --region configs/regions/scorecard/dk_k100.toml
python scripts/dev/run_locked.py -- pyvwf-validate evaluate --region configs/regions/scorecard/dk_k100.toml \
    --train-run output/validation/DK/train-<timestamp>
```

Every run below has a clean-tree manifest. Paths are under `output/`, which is
not in the repository.

| Row | Commit | Run | Roughness fill | Curve library |
|---|---|---|---|---|
| DE, DK, UK | `967b485` | `validation/refresh_2026-09-20/<CODE>/` | record-wide | licensed |
| US, BR | `967b485` | `validation/refresh_2026-09-20/<CODE>/` | none (stored daily) | licensed |
| AU-NEM, NZ | trained `967b485`, evaluated `a94670c` | `c1_turbine_order_2026-09-24/after/<CODE>/` | record-wide | licensed |
| CL, AR | `967b485` | `validation/refresh_2026-09-20/<CODE>/` | record-wide | open |
| FR | `9a5682c` | `fr_odre_2026-10-25/promote/` | year-local | licensed |
| BE | `6c5272d` | `be_elia_2026-10-22/` | record-wide | licensed |
| IE, ES | `6b3e6d9` | `offshore_regen_2026-10-16/after/<CODE>/` | record-wide | licensed |
| SE | `768a278` | `se_register_2026-10-25/promote/` | year-local | licensed |
| NO | `b9e812b` | `no_nve_2026-10-25/promote/` | year-local | licensed |
| IT (suspended) | `7fc0e16` | `it_pt_2023_2026-10-24/IT/` | record-wide | licensed |
| PT (suspended) | `4ccab24` | `pt_ren_2026-10-27/promote/` | year-local | licensed |

- **Curve library.** Licensed is `power_curves.csv` sha256 `689cfee7…` under
  `input/combined/`, which is not redistributable; open is the bundled library,
  sha256 `56314f39…`. DE, DK, UK and US simulate 74% to 90% of their capacity
  on curves only the licensed library holds. BR, AU-NEM and NZ simulate only on
  curves the open library also holds, and reproduced their `metrics.csv` byte
  for byte on it on 2026-09-11; that comparison has not been repeated on the
  runs above.
- **Roughness fill.** Since 2026-10-22 the roughness derived at load is filled
  within each calendar year, where it was filled along the whole record. Rows
  marked record-wide predate the change and stand until the next refresh. Five
  rows run under both fills moved by at most 2.8e-5 in RMSE. France's
  registered gate for the change was breached, and the change was adopted by
  the maintainer's decision.
- **Configurations.** Each file is the one its run was made from, with three
  later edits that keep each run's meaning: `br_k60.toml` and `us_k250.toml`
  gained `roughness = "stored"` (`a023b4d`), the treatment they already ran
  under, when the default changed to derived; and a comment in `uk_k50.toml`
  and `us_k250.toml` names the renamed package (`4d06163`).
- **Observations.** Turbine and plant level: DK Danish Energy Agency; DE the
  public turbine register; UK REPD and Ofgem; US EIA-923; BR ONS; AU-NEM AEMO;
  NZ EMI; CL Coordinador (CEN); AR CAMMESA, with capacities rebuilt from turbine
  specifications. Country level: ENTSO-E Transparency for IE and ES, for
  Sweden's generation and for Norway's 2023 generation; Elia Open Data (`ods031`) for BE;
  RTE's series through ODRE (Licence Ouverte v2.0) for FR; Energimyndigheten's
  wind fleet as Sweden's denominator; NVE's metered farm output and turbine
  register for Norway's training years. Confidential inputs (WindStats DE and
  ES, the Ofgem certificate warehouse, the licensed curve library) are not
  redistributed. All correction operates on ERA5 at 0.25°.

## Corrections log

One line per dated notice. Each links the notice, with its before-and-after
tables, in the full scorecard at the archive tag.

| Date | What changed | Notice |
|---|---|---|
| 2026-09-11 | The UK and NZ gains cannot be distinguished from zero when units are resampled; both rows gained ‡ | [L877](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L877) |
| 2026-09-11 | Units with failed offsets or off-curve speeds had dropped out of the corrected score; every variant is now scored on common rows, CL's gain fell to 0.006 and it gained ‡ | [L944](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L944) |
| 2026-09-11 | IT, PT and ES suspended: the ERA5 download did not cover them. Resolved 2026-09-13 by a wider download | [L1018](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L1018) |
| 2026-09-12 | DK also has extrapolated winds, 0.6% of capacity, and gained § | [L1090](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L1090) |
| 2026-09-12 | The rows did not share one roughness treatment; the per-timestep derivation was adopted and the eleven European rows rerun on 2026-09-13, superseding their figures | [L1109](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L1109) |
| 2026-09-13 | Denmark's curve-match shares were computed on truncated manufacturer names; unverifiable fell from 23.0% to 3.1% | [L1274](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L1274) |
| 2026-09-24 | AU-NEM's and NZ's corrected figures had been simulated on other units' curves and capacities; both re-evaluated | [L809](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L809) |
| 2026-09-25 | Country-level fits with more than one cluster had used the per-cluster solver, not the joint national fit | [L723](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L723) |
| 2026-09-25 | The joint fit had stopped at its starting point in 47 of 540 training periods; tolerance fixed in `26e0b67` | [L646](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L646) |
| 2026-09-25 | Country rows moved from the 100 kW fallback curve to each grid's own turbine; IT and PT suspended | [L549](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L549) |
| 2026-09-29 | Ireland's row moved with the corrected wind-fleet register | [L483](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L483) |
| 2026-10-16 | Offshore farms got their own grid points; Belgium's row moved | [L381](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L381) |
| 2026-10-22 | Belgium's observations had measured onshore output only, in five of seven training years and the test year; rerun on Elia's complete series | [L267](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L267) |
| 2026-10-22 | The derived roughness fill changed; rows not rerun since lag the code | [L363](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L363) |
| 2026-10-23 | France's observations had left out offshore output; rerun on RTE's series, read from ODRE on 2026-10-25 with no change at the precision shown | [L177](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L177) |
| 2026-10-24 | Italy's and Portugal's test files had left out offshore output; both rerun, both still suspended | [L1488](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L1488) |
| 2026-10-25 | Sweden's capacity-factor denominator had been built from its own generation; rerun on Energimyndigheten's fleet | [L105](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L105) |
| 2026-10-25 | Norway's observations measured a different share of its fleet each year; rerun on NVE's metered output | [L25](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L25) |
| 2026-10-27 | Portugal's denominator replaced by REN's month-end fleet; rerun, still suspended | [L1511](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/scorecard.md?plain=1#L1511) |
| 2026-10-27 | This short form replaced the full scorecard. It adds that the reported configuration is not the lowest-RMSE variant in BR, CL and AR, (the full scorecard said so for all six country rows), and that three scorecard configurations were edited after their runs | this page |
