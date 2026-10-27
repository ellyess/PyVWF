# Multi-region validation scorecard

**Reproduction record, added 2026-09-18.** Drivers:
`scripts/studies/scorecard/missing_value_audit.py`,
`scripts/studies/scorecard/off_curve_sensitivity.py`,
`scripts/studies/scorecard/training_objective_check.py`,
`scripts/studies/scorecard/unit_concentration.py`. Until 2026-09-18 they were
in `scripts/analysis/`, the path any command below uses;
`scripts/studies/README.md` maps each old path to its new one. Numbers:
`unit_concentration.py`'s output records no commit; the driver's last commit
before it was written is `4b6d143`, and rerun at `51807f8` it reproduces its
recorded files byte for byte (`tests/test_pin_bootstrap_reproduction.py`). The
outputs of `missing_value_audit.py`, `off_curve_sensitivity.py` and
`training_objective_check.py` record no commit either, and each was written on
2026-09-11 minutes before the driver's first commit (`bbaf5b3`, `b7826d3` and
`b7826d3`), so the exact code that produced them is not recorded. The runs they
read are pinned by their own manifests.

Per region, how much PyVWF's affine wind-speed correction reduces the error
between ERA5-simulated and observed capacity factors on a held-out year. Every
number is read from a `metrics.csv` under `output/validation/`, with the source
path given so each is auditable. Screening-level validation, one test year per
region, not an accredited yield assessment.

**Correction notice, 2026-10-25 (second): Norway's observations measured a
different share of its fleet in each training year, and the row is rerun on
NVE's metered farm output.** Training years 2015-21, test year 2023, national
scope, the scorecard config `configs/regions/scorecard/no_country.toml`,
licensed curve library (`power_curves.csv` sha256 `689cfee7…`, one curve,
Vestas.V90.3000, origin `external`, 0% substituted, 0% extrapolated), roughness
filled within each calendar year (#140).

*What was wrong.* Norway's training file summed the bidding zones ENTSO-E
reports. Matched hour by hour, it held 0.873 to 0.990 of the output NVE's
farms metered each year, and Statistics Norway's official monthly wind
production (table 12824) equals that metered output in every month of
2015-21. In October 2015-18 it also doubled the clock-change hour. Its
capacity was ENTSO-E's register, 26% below NVE's turbine register in 2015 and
32% above it in 2020. In the 2023 test year ENTSO-E's generation matches
Statistics Norway (0.974 to 1.006 by month), and only the capacity was off.

| Year | ENTSO-E file, GWh | NVE metered, counted, GWh | Statistics Norway, GWh | ENTSO-E / NVE metered | Stored capacity, mean MW | NVE register, mean MW (counted) | Mean CF stored | Mean CF rebuilt |
|---|---|---|---|---|---|---|---|---|
| 2015 | 2,293 | 2,488 | 2,515 | 0.916 | 627 | 852 (841) | 0.421 | 0.338 |
| 2016 | 2,103 | 2,121 | 2,116 | 0.990 | 692 | 869 (866) | 0.346 | 0.279 |
| 2017 | 2,660 | 2,708 | 2,850 | 0.934 | 903 | 959 (932) | 0.336 | 0.326 |
| 2018 | 3,380 | 3,705 | 3,876 | 0.873 | 1,231 | 1,435 (1,322) | 0.314 | 0.316 |
| 2019 | 5,231 | 5,353 | 5,536 | 0.946 | 2,259 | 1,988 (1,899) | 0.265 | 0.324 |
| 2020 | 9,293 | 8,991 | 9,911 | 0.938 | 3,861 | 3,015 (2,758) | 0.274 | 0.370 |
| 2021 | 10,656 | 11,480 | 11,767 | 0.906 | 5,102 | 4,118 (3,999) | 0.238 | 0.326 |
| 2023 (test) | 13,901 | (ENTSO-E) | 13,965 | | 5,130 | 5,080 | 0.309 | 0.312 |

*What changed.* `scripts/region_tools/build_no_obs_from_nve.py` (`b9e812b`)
builds the training files and the 2019 test file from NVE's hourly output per
farm, metered by Statnett at the grid connection
(`vindprod2002-2021_kraftverk_utcplus1.xlsx`; NVE's 2021 report, MEV WS
2022-001, section 2.5), over NVE's turbine register. A farm counts in a month,
output and capacity alike, only when it is metered every hour, its registered
capacity is above zero and fixed all month, and its output never exceeds 1.05
times that capacity: the register dates turbines when they are formally
commissioned, often months after they start producing. That keeps 0.907 to
0.998 of each year's metered energy, which is why the counted energy above
falls short of Statistics Norway's in the build-out years. NVE's other file,
`timeseries_utc_kwh_2022_all.zip`, is modelled back to 2003 for every farm and
is not used. The 2023 test file keeps ENTSO-E's generation over the register's
whole fleet, since the metered file ends in 2021. The files were built first in
a separate directory and promoted into `input/combined/` with the builder's
`--replace`, which keeps each old file as `.entsoe.bak.csv`; each promoted file
is byte-identical to the separate build, and every split passes the physical
gates. Data: `output/no_sources_2026-10-25/` (sources, alignment and rule
checks) and `output/no_nve_2026-10-25/` (old files, build, runs).

Every variant: the published run of 2026-10-16 (`6b3e6d9`, record-wide fill),
a control at `b9e812b` on the old files, and the run at `b9e812b` on the
rebuilt files, both from a clean tree. The control differs from the published
run by at most 2.9e-4 in RMSE and MBE, the year-local fill's effect, with the
same scalars to 4e-5, so the change from control to rebuilt is the
observations'.

| Variant | RMSE published | RMSE control | RMSE rebuilt | MBE control | MBE rebuilt | Max scalar control / rebuilt | Implausible / failed rebuilt | Months scored |
|---|---|---|---|---|---|---|---|---|
| uncorrected | 0.1277 | 0.1277 | 0.1308 | -0.1236 | -0.1267 |  |  | 12 of 12 |
| fixed N=1 | 0.0376 | 0.0376 | 0.0359 | -0.0313 | -0.0293 | 1.47 / 1.52 | 0 / 0 | 12 of 12 |
| **fixed N=4** (reported) | 0.0260 | 0.0260 | 0.0257 | -0.0138 | -0.0119 | 3.05 / 3.18 | 1 / 0 | 12 of 12 |
| season N=1 | 0.0382 | 0.0385 | 0.0347 | -0.0314 | -0.0286 | 1.57 / 1.60 | 0 / 0 | 12 of 12 |
| season N=4 | 0.0258 | 0.0258 | 0.0227 | -0.0147 | -0.0111 | 3.30 / 3.49 | 4 / 0 | 12 of 12 |

Every corrected variant's error falls and its bias shrinks; the uncorrected
error rises from 0.128 to 0.131, because the test year's capacity is now the
register's 5,080 MW rather than ENTSO-E's 5,130. The reported fit stays
degenerate: cluster 0 carries a scalar of 3.18. The reported configuration
stays fixed N=4: it was chosen on the 2023 test year before this change, and
season N=4, also degenerate, now has the lowest error.

*Still open, not corrected here.*
- The 1.05 test finds a farm's unregistered turbines only in hours windy
  enough to exceed its registered capacity, so a calm build-out month can
  still count against too small a capacity.
- The last hour of 2021 is missing: NVE's file ends at 22:00 UTC.
- The zonal files (`no_1` to `no_4`) keep ENTSO-E's data, and the fetcher
  still sums an hour missing a zone as a smaller fleet and adds both October
  clock-change hours; neither is read by this row any more.
- NVE allows its data to be used with NVE credited and the content unchanged.

**Correction notice, 2026-10-25: Sweden's capacity-factor denominator was
built from its own generation in 2015-19, and the row is rerun on
Energimyndigheten's fleet.** Training years 2015-21, test year 2023, national
scope, the scorecard config `configs/regions/scorecard/se_country.toml`,
licensed curve library (`power_curves.csv` sha256 `689cfee7…`, one curve,
Vestas.V90.3000, origin `external`, 0% substituted, 0% extrapolated), roughness
filled within each calendar year (#140).

*What was wrong.* Sweden's generation is sound: each year's energy is 0.976 to
1.007 of Energimyndigheten's official wind production. Its denominator is not.
The training file, summed from the four bidding zones, carries 8,354 MW in
every year from 2015 to 2019: the sum of four zonal capacities an old fetcher
derived as peak generation over 0.9 when ENTSO-E returned none
(`method-country-level.md`). The 2020-21 values are ENTSO-E's register, and
the 2023 value its four zones' sum. Both count onshore wind at the start of the
year, so they leave out the year's growth and about 200 MW offshore that the
generation includes.

| Year | Stored capacity, MW | ENTSO-E register, MW | Energimyndigheten, previous to this 31 Dec, MW | Mean CF stored | Mean CF rebuilt |
|---|---|---|---|---|---|
| 2015 | 8,354 | 3,500 | 5,088 to 5,818 | 0.227 | 0.347 |
| 2016 | 8,354 | 5,493 | 5,818 to 6,434 | 0.213 | 0.289 |
| 2017 | 8,354 | 5,960 | 6,434 to 6,611 | 0.236 | 0.303 |
| 2018 | 8,354 | 6,247 | 6,611 to 7,300 | 0.223 | 0.267 |
| 2019 | 8,354 | 7,506 | 7,300 to 8,681 | 0.266 | 0.280 |
| 2020 | 9,648 | 9,648 | 8,681 to 10,082 | 0.317 | 0.328 |
| 2021 | 10,016 | 10,017 | 10,082 to 12,116 | 0.304 | 0.274 |
| 2023 (test) | 14,765 | 14,700 | 14,278 to 16,224 | 0.264 | 0.257 |

*What changed.* `scripts/region_tools/build_se_capacity_from_energimyndigheten.py`
(`768a278`) keeps every national Swedish file's generation and hours and
divides by Energimyndigheten's 31 December fleet (table EN0105_1, onshore plus
offshore), interpolated linearly through the year as France's builder does
with RTE's. The files were built first in a separate input root,
`input/combined-se-em/`, and then promoted into `input/combined/` with the
builder's `--replace`, which keeps each old file as
`.entsoe-capacity.bak.csv`. Every file passes the physical gates, and the
training file no longer fails the frozen-capacity one. Data:
`output/se_register_2026-10-25/` (comparison, build log, checksums, and the
runs below).

Every variant: the published run of 2026-10-16 (`6b3e6d9`, record-wide fill),
a control at `768a278` on the old files, and the run at `768a278` on the
rebuilt files, both from a clean tree. The control differs from the published
run by at most 6e-6 in RMSE and MBE, the year-local fill's effect, and its
scalars are identical, so the change from control to rebuilt is the
denominator's. Rerun on `input/combined/` after the promotion, every output file
except the manifest is byte-identical to the rebuilt run, `metrics.csv`
included.

| Variant | RMSE published | RMSE control | RMSE rebuilt | MBE control | MBE rebuilt | Max scalar control / rebuilt | Implausible / failed rebuilt | Months scored |
|---|---|---|---|---|---|---|---|---|
| uncorrected | 0.0980 | 0.0980 | 0.0922 | -0.0963 | -0.0891 |  |  | 12 of 12 |
| fixed N=1 | 0.0358 | 0.0358 | 0.0261 | -0.0330 | +0.0124 | 1.37 / 1.60 | 0 / 0 | 12 of 12 |
| **fixed N=4** (reported) | 0.0347 | 0.0347 | 0.0272 | -0.0310 | +0.0148 | 1.64 / 1.92 | 0 / 0 | 12 of 12 |
| season N=1 | 0.0360 | 0.0360 | 0.0199 | -0.0336 | +0.0114 | 1.47 / 1.70 | 0 / 0 | 12 of 12 |
| season N=4 | 0.0335 | 0.0335 | 0.0234 | -0.0304 | +0.0146 | 1.84 / 2.14 | 0 / 0 | 12 of 12 |

Every variant's error falls, and every corrected bias changes sign, from about
-0.03 to between +0.011 and +0.015. The rebuilt scalars lie between 0.96 and
2.14, so no fit is degenerate. The reported configuration stays fixed N=4: it
was chosen on the 2023 test year before this change, and season N=1 now has
the lowest error.

*Still open, not corrected here.*
- The four zonal files (`se_1` to `se_4`), which only `se_bz.toml` reads, keep
  their derived zonal capacities; Energimyndigheten publishes the same fleet
  per bidding zone (table EN0105_2).
- Energimyndigheten's register keeps a decommissioned turbine for up to three
  years, so the fleet is slightly overstated.
- The interpolation spreads each year's additions evenly through it.

**Correction notice, 2026-10-23: France's observations left out offshore
generation in the test year and understated the fleet in 2018 and 2019, and
the row is rerun on RTE's complete series.** Training years 2015-21, test year
2023, national scope, the scorecard config
`configs/regions/scorecard/fr_country.toml`, licensed curve library
(`power_curves.csv` sha256 `689cfee7…`, one curve, Vestas.V80.2000, origin
`external`, 0% substituted, 0% extrapolated).

*What was wrong.* Matched hour by hour against ENTSO-E's per-type series
(`output/entsoe_audit_2026-10-23/REPORT.md`), the 2023 test file equals onshore
generation alone (B19, ratio 1.000); built again with the generator's own call
and `psr_type="onshore"` it is reproduced byte for byte, and the generator's
command line has defaulted to `--psr-type onshore` since `a95e134`. ENTSO-E
holds no French offshore before June 2023, while RTE's definitive eCO2mix
series reports offshore output in every month, 1.89 TWh in the year; peaks of
667 to 704 MW from September show Saint-Brieuc feeding in beside
Saint-Nazaire's 480 MW. The training files' capacity is ENTSO-E's register,
which matches RTE's year-end fleet in 2015-17 and then falls 1.0 to 2.5 GW
below it through 2018 and 1.5 to 2.9 GW through 2019, overstating those years'
mean capacity factors by 8% and 12%.

| Year | Old file, TWh | Old capacity, MW | RTE, TWh | RTE capacity, MW (1 Jan to 31 Dec) | Eurostat gross, TWh | Old / Eurostat | RTE / Eurostat | Mean CF old | Mean CF RTE |
|---|---|---|---|---|---|---|---|---|---|
| 2015 | 19.59 | 10,322 | 21.09 | 9,303 to 10,313 | 21.86 | 0.896 | 0.965 | 0.217 | 0.246 |
| 2016 | 19.56 | 11,757 | 20.91 | 10,313 to 11,794 | 21.82 | 0.897 | 0.958 | 0.190 | 0.218 |
| 2017 | 22.84 | 13,564 | 24.10 | 11,794 to 13,508 | 25.11 | 0.910 | 0.960 | 0.192 | 0.218 |
| 2018 | 26.79 | 12,521 | 28.12 | 13,508 to 15,067 | 29.18 | 0.918 | 0.964 | 0.244 | 0.226 |
| 2019 | 32.70 | 13,607 | 33.81 | 15,067 to 16,458 | 35.46 | 0.922 | 0.954 | 0.274 | 0.245 |
| 2020 | 38.36 | 16,592 | 39.68 | 16,458 to 17,548 | 40.71 | 0.942 | 0.975 | 0.263 | 0.266 |
| 2021 | 35.18 | 17,225 | 36.86 | 17,548 to 18,831 | 37.69 | 0.933 | 0.978 | 0.233 | 0.232 |
| 2023 (test) | 46.95 | 20,842 | 50.84 | 21,054 to 22,687 | 51.46 | 0.912 | 0.988 | 0.257 | 0.265 |

*What changed.* `scripts/region_tools/build_fr_obs_from_rte.py` (`65efa06`)
rebuilds every French split from RTE's annual definitive eCO2mix files
(national `Eolien`, onshore and offshore, half-hourly), hourly means, divided
by RTE's 31 December wind fleet (`parc-prod-par-filiere`) interpolated linearly
through the year; two hours a year are dropped at the October clock change.
The files were built first in a separate input root, `input/combined-fr-rte/`,
and then promoted into `input/combined/` with the builder's `--replace`, which
keeps each old file as `.entsoe.bak.csv`. Checksums:
`output/fr_rte_2026-10-23/rte_obs.sha256`, and for the downloads
`rte_downloads.sha256` beside it. Every split passes the physical gates.
*[2026-10-25: the builder now reads the same definitive series and fleet from
ODRE (`eco2mix-national-cons-def`, `parc-prod-par-filiere`, Licence Ouverte
v2.0) instead of RTE's eCO2mix files, whose terms forbid redistribution
(`09df0ae`). Hour by hour the promoted files equal the ones they replace,
except that each year's March clock-change hour is now empty as well, three
hours a year in all, and 2021-12-31 23:00 UTC reads 3,187.5 MW, not 3,182.0.
Training years 2015-21, test year 2023: rerun at `9a5682c`, every variant's
RMSE and MBE is within 4e-5 of a control on the old files at the same commit,
and the control within 7e-5 of the after column below, the year-local
roughness fill's effect (#140). No figure of France's row moves at its
displayed precision. Data: `output/fr_odre_2026-10-25/`.]*

Every variant, before (`6b3e6d9`, the notice of 2026-10-16) and after
(`65efa06`, clean tree, `PYVWF_INPUT=input/combined-fr-rte`). Data:
`output/offshore_regen_2026-10-16/after/FR/` and
`output/fr_rte_2026-10-23/evaluate-2023-rte-2026-10-23/metrics.csv`. The
current code on the old files (`evaluate-2023-control-2026-10-23`) reproduces
the before `metrics.csv` byte for byte, so the movement is the observations'.
Rerun on `input/combined/` after the promotion, every output file except the
manifest is byte-identical to the after run, `metrics.csv` included. Data:
`output/fr_rte_2026-10-23/promote/`.

| Variant | RMSE before | RMSE after | MBE before | MBE after | Max scalar before / after | Implausible / failed after | Months scored after |
|---|---|---|---|---|---|---|---|
| uncorrected | 0.0204 | 0.0190 | -0.0035 | -0.0116 |  |  | 12 of 12 |
| fixed N=1 | 0.0291 | 0.0248 | +0.0137 | +0.0118 | 1.07 / 1.09 | 0 / 0 | 12 of 12 |
| **fixed N=10** (reported) | 0.0260 | 0.0216 | +0.0106 | +0.0087 | 4.61 / 4.71 | 1 / 0 | 12 of 12 |
| season N=1 | 0.0270 | 0.0227 | +0.0148 | +0.0126 | 1.23 / 1.21 | 0 / 0 | 12 of 12 |
| season N=10 | 0.0240 | 0.0197 | +0.0121 | +0.0098 | 9.94 / 9.62 | 4 / 0 | 12 of 12 |

Every corrected variant's RMSE falls, and every one stays above the
uncorrected 0.0190. The reported fit stays degenerate: cluster 7 carries a
scalar of 4.71. The reported configuration stays fixed N=10: it was chosen on
the 2023 test year before this change.

*Still open, not corrected here.*
- The 2023 grid holds Saint-Nazaire's 480 MW offshore and not Saint-Brieuc,
  which fed in from September 2023.
- The capacity is interpolated evenly between year ends, while offshore farms
  arrive in steps.
- RTE's energy is 3 to 8% above ENTSO-E's in every year, so the training years
  move as well as the test year.
- Italy's and Portugal's 2023 test files also leave out offshore output (0.2%
  and 0.6%); rebuilt files exist and are not yet staged. *[2026-10-24: staged
  and both rows rerun; see the note under the suspended rows.]*
- Findings documents that score France's country row read the old files and
  are not yet reviewed.

**Correction notice, 2026-10-22: Belgium's
observations measured onshore generation in five of seven training years and
in the test year, and the row is rerun on Elia's complete series.** Training
years 2015-21, test year 2023, national scope, the scorecard config
`configs/regions/scorecard/be_country.toml`, licensed curve library
(`power_curves.csv` sha256 `689cfee7…`, one curve, Vestas.V90.3000, origin
`external`, 0% substituted, 0% extrapolated).

*What was wrong.* The Belgian files under
`observations/country/observations/be/` do not measure one quantity. Matched
hour by hour against ENTSO-E's per-type series (B18 offshore, B19 onshore) and
Elia's Open Data series `ods031`, the 2015-19 rows and the 2023 test file equal
onshore generation alone (r 0.998 to 1.000, ratio of means 1.00 to 1.01), with
onshore capacity as the denominator; the 2020-21 rows equal onshore plus
offshore. ENTSO-E holds offshore generation for every year, so the omission is
in how the files were built, not in the source. The 2015-19 rows carry no
`country` value and the 2020-21 rows carry `BE`, so the training file was
joined from two fetches. Every physical gate passed, because generation and
capacity were onshore together. The model simulates the whole fleet, offshore
on its own points since `fc8c7f5`, so the correction was fitted to turn
whole-fleet output into onshore output: the reported fit's cluster 0, which
holds the offshore and coastal points, carried scalars of 0.48 to 0.52.

| Year | Old file, TWh | ENTSO-E onshore | ENTSO-E offshore | Elia onshore + offshore | Eurostat gross |
|---|---|---|---|---|---|
| 2015 | 2.45 | 2.45 | 2.54 | 5.02 | 5.57 |
| 2016 | 2.24 | 2.24 | 2.32 | 4.58 | 5.42 |
| 2017 | 2.64 | 2.64 | 2.79 | 5.43 | 6.52 |
| 2018 | 2.96 | 2.96 | 3.32 | 6.28 | 7.57 |
| 2019 | 3.36 | 3.36 | 4.68 | 8.01 | 9.76 |
| 2020 | 10.97 | 4.09 | 6.87 | 10.82 | 12.82 |
| 2021 | 10.78 | 3.98 | 6.78 | 10.76 | 12.00 |
| 2023 (test) | 6.27 | 6.27 | 7.88 | 14.35 | 15.44 |

*What changed.* `scripts/region_tools/build_be_obs_from_elia.py` (`2e53985`,
`6c5272d`) rebuilds every Belgian split from `ods031`: offshore and onshore
summed, hourly means, divided by the summed monitored capacity, dropping the 108
quarter-hours in 2015-24 where Elia reports no monitored capacity. The new files
were built first in a separate input root, `input/combined-be-elia/`, which
linked every other input to `input/combined/`, and were then promoted into
`input/combined/` with the builder's `--replace`, which keeps each old file
beside its replacement as `.entsoe-mixed.bak.csv`. Checksums:
`output/be_elia_2026-10-22/elia_obs.sha256` for the new files and
`output/be_elia_2026-10-22/promote/old_files.sha256` for the old. The audit
now compares each year's energy with Eurostat (`dbb5352`): the old BE train and
test series fail it (0.34 to 0.44 of the statistic), the new ones pass.

Every variant, before (`6b3e6d9`, the notice of 2026-10-16) and after
(`6c5272d`, clean tree, `PYVWF_INPUT=input/combined-be-elia`). Data:
`output/offshore_regen_2026-10-16/after/BE/` and
`output/be_elia_2026-10-22/evaluate-2023-elia-2026-10-22/metrics.csv`. Rerun on
`input/combined/` after the promotion (`dc2ce83`, clean tree), every output
file except the manifest is byte-identical to that run, `metrics.csv`
included. Data: `output/be_elia_2026-10-22/promote/`.

| Variant | RMSE before | RMSE after | MBE before | MBE after | Scalar range after | Implausible / failed after | Months scored after |
|---|---|---|---|---|---|---|---|
| uncorrected | 0.1610 | 0.0812 | +0.1560 | +0.0788 |  |  | 12 of 12 |
| fixed N=1 | 0.0199 | 0.0212 | -0.0061 | -0.0147 | 0.754 | 0 / 0 | 12 of 12 |
| fixed N=3 | 0.0147 | 0.0111 | +0.0027 | +0.0003 | 0.657 to 1.239 | 0 / 0 | 12 of 12 |
| season N=1 | 0.0207 | 0.0247 | -0.0065 | -0.0148 | 0.737 to 0.773 | 0 / 0 | 12 of 12 |
| **season N=3** (reported) | 0.0191 | 0.0183 | +0.0034 | +0.0010 | 0.640 to 1.651 | 0 / 0 | 12 of 12 |

The reported configuration stays season N=3: it was chosen on the 2023 test
year before this change, and choosing again on the corrected series would be a
second selection on the same year. Cluster 0's scalars move from 0.48 to 0.52
before to 0.64 to 0.67 after.

*Still open, not corrected here.*
- The 2023 grid holds 1,608 MW onshore against Elia's 2,991 MW monitored, and
  2,256 MW offshore against 2,255 MW, so the national weighting puts 58% on
  offshore against a real 43%. Not measured: how much of cluster 0's scalar
  this accounts for.
- France's 2023 test file is onshore generation alone (46.95 TWh, equal to
  ENTSO-E B19, r 1.000), so Saint-Nazaire's 1.11 TWh, 2.3% of the national
  total, is missing while the 2023 grid simulates its 480 MW. France's
  training years predate any French offshore farm. *[2026-10-23: corrected by
  the notice of that date. The 1.11 TWh is ENTSO-E's June to December only;
  RTE gives 1.89 TWh for the year.]*
- The fetcher (`fetch_entsoe_capacity_factors.py`) summed ENTSO-E's
  consumption columns into wind generation, in all four of its generation
  paths; for Belgium that is `("Wind Offshore", "Actual Consumption")`, 0.27%
  of offshore output in 2021 and 0.22% in 2023, fetched on 2026-10-22. That
  replaces this notice's first estimate of 1 to 3%. Data:
  `output/be_elia_2026-10-22/entsoe_consumption/compare.txt`. Fixed in
  `6311446`. Whether any other country's stored files carry consumption is not
  checked.
- Findings documents that score Belgium's country row read the same files and
  are not yet reviewed, among them `method-country-level.md`,
  `method-country-weighting.md`, `method-cluster-selection.md`,
  `method-curve-library.md`, `method-eu-rerun.md` and
  `method-physics-informed-loco.md`.
- The country-level table below carries the rerun figures. Every Belgian
  figure in the earlier notices and in the superseded rows read the old files;
  they stay as the record of what was published and are not rerun.

**Notice, 2026-10-22: the derived roughness fill changed, and the fifteen rows
on it lag the code.** From 2026-10-22 the roughness derived at load is filled
within each calendar year instead of along the whole loaded record
(`docs/findings/method-roughness-fill-prereg.md`). Every row below except US
and BR, which read stored daily roughness, was computed with the old fill. Five
were re-run under both fills on the test year (training and test years as in
their tables): DK k100, DE k100 and CL k10 moved by at most 1.1e-7 in RMSE and
MBE; ES country by at most 2.8e-5 in RMSE and 6.0e-5 in MBE; FR country N=10
season by 2.3e-5 in RMSE and 1.61e-4 in MBE, a degenerate fit, and its
registered gate was breached and the change adopted by the maintainer's
decision (outcome tables and Data paths in that document). AR, AU-NEM, NZ, UK,
BE, IE, IT, NO, PT and SE were not re-run. The figures below stand as computed
until the next scorecard refresh re-runs them. *[2026-10-24: the Belgian,
French, Italian and Portuguese rows were since re-run for their observations
(the notices of 2026-10-22 and 2026-10-23 above, and the note under the
suspended rows) at commits that predate this fill, so their figures are on the
old fill too.]*

**Correction notice, 2026-10-16: offshore wind farms get their own grid
points, and Belgium's row moves.** The country lattices cover land only, so
until now every offshore project was summed onto the nearest land point and
simulated with coastal winds. Belgium's whole offshore fleet, 2,256 MW and 62%
of its 2021 grid, sat on two coastal points at 51.25N; its farms stand at
51.53-51.72N. From `fc8c7f5` each offshore site has its own point at its own
position, with the grid's turbine, hub height and the cluster of the nearest
land point, and from `6b3e6d9` Hywind Tampen (88 MW, 2022), which supplies oil
and gas platforms rather than the grid Norway's observations measure, is out of
Norway's fleet. The per-year grids were regenerated and all eight rows trained
and evaluated again at `6b3e6d9` from a clean tree, with the scorecard configs,
on `input/combined` (`power_curves.csv` sha256 `689cfee7…`); every run records
0% of capacity substituted and none extrapolated. Training years 2015-21 (IE
2017-21), test year 2023, national scope.

Offshore capacity on its own points, in the 2021 and 2023 grids a scorecard
run reads:

| Row | Sites | 2021 | 2023 |
|---|---|---|---|
| BE | 11 | 2,256 MW of 3,617 | 2,256 MW of 3,864 |
| FR | 5 | 0 of 20,662 | 480 MW of 23,409 (Saint-Nazaire) |
| SE | 5 | 188 MW of 10,312 | 188 MW of 14,923 |
| NO | 5 | 50 MW of 4,427 | 50 MW of 5,033 (Hywind Tampen out: 5,121 before) |
| IE | 1 | 25 MW of 3,722 | 25 MW of 4,118 |
| ES | 1 | 20 MW of 27,701 | 20 MW of 29,906 |
| PT | 1 | 25 MW of 4,883 | 25 MW of 5,071 |
| IT | 1 | 0 of 9,460 | 30 MW of 10,061 |

Grid totals are unchanged except Norway's 2023; only where capacity sits moved.

Every variant of every row, before (`6ab07ef`, the notice of 2026-09-29) and
after (`6b3e6d9`):

| Row | Variant | RMSE before | RMSE after | MBE before | MBE after | Max scalar before / after | Implausible / failed after | Months scored after |
|---|---|---|---|---|---|---|---|---|
| FR | uncorrected | 0.0204 | 0.0204 | -0.0016 | -0.0035 |  |  | 12 of 12 |
| FR | fixed N=1 | 0.0302 | 0.0291 | +0.0156 | +0.0137 | 1.07 / 1.07 | 0 / 0 | 12 of 12 |
| FR | **fixed N=10** (reported) | 0.0271 | 0.0260 | +0.0123 | +0.0106 | 4.61 / 4.61 | 1 / 0 | 12 of 12 |
| FR | season N=1 | 0.0281 | 0.0270 | +0.0167 | +0.0148 | 1.23 / 1.23 | 0 / 0 | 12 of 12 |
| FR | season N=10 | 0.0251 | 0.0240 | +0.0138 | +0.0121 | 9.94 / 9.94 | 4 / 0 | 12 of 12 |
| BE | uncorrected | 0.1226 | 0.1610 | +0.1164 | +0.1560 |  |  | 12 of 12 |
| BE | fixed N=1 | 0.0162 | 0.0199 | -0.0001 | -0.0061 | 0.67 / 0.58 | 0 / 0 | 12 of 12 |
| BE | fixed N=3 | 0.0160 | 0.0147 | +0.0062 | +0.0027 | 0.95 / 0.95 | 0 / 0 | 12 of 12 |
| BE | season N=1 | 0.0178 | 0.0207 | -0.0000 | -0.0065 | 0.69 / 0.59 | 0 / 0 | 12 of 12 |
| BE | **season N=3** (reported) | 0.0205 | 0.0191 | +0.0073 | +0.0034 | 1.23 / 1.23 | 0 / 0 | 12 of 12 |
| IE | uncorrected | 0.0453 | 0.0454 | +0.0361 | +0.0363 |  |  | 12 of 12 |
| IE | fixed N=1 | 0.0239 | 0.0238 | +0.0062 | +0.0062 | 0.91 / 0.91 | 0 / 0 | 12 of 12 |
| IE | fixed N=3 | 0.0236 | 0.0235 | +0.0052 | +0.0051 | 0.96 / 0.95 | 0 / 0 | 12 of 12 |
| IE | **season N=1** (reported) | 0.0213 | 0.0213 | +0.0061 | +0.0061 | 0.94 / 0.94 | 0 / 0 | 12 of 12 |
| IE | season N=3 | 0.0211 | 0.0211 | +0.0053 | +0.0053 | 1.00 / 0.99 | 0 / 0 | 12 of 12 |
| SE | uncorrected | 0.0984 | 0.0980 | -0.0967 | -0.0963 |  |  | 12 of 12 |
| SE | fixed N=1 | 0.0358 | 0.0358 | -0.0329 | -0.0330 | 1.37 / 1.37 | 0 / 0 | 12 of 12 |
| SE | **fixed N=4** (reported) | 0.0348 | 0.0347 | -0.0313 | -0.0310 | 1.64 / 1.64 | 0 / 0 | 12 of 12 |
| SE | season N=1 | 0.0359 | 0.0360 | -0.0336 | -0.0336 | 1.47 / 1.47 | 0 / 0 | 12 of 12 |
| SE | season N=4 | 0.0336 | 0.0335 | -0.0303 | -0.0304 | 1.84 / 1.84 | 0 / 0 | 12 of 12 |
| NO | uncorrected | 0.1286 | 0.1277 | -0.1245 | -0.1236 |  |  | 12 of 12 |
| NO | fixed N=1 | 0.0392 | 0.0376 | -0.0326 | -0.0313 | 1.47 / 1.47 | 0 / 0 | 12 of 12 |
| NO | **fixed N=4** (reported) | 0.0279 | 0.0260 | -0.0160 | -0.0138 | 3.05 / 3.05 | 1 / 0 | 12 of 12 |
| NO | season N=1 | 0.0397 | 0.0382 | -0.0324 | -0.0311 | 1.57 / 1.57 | 0 / 0 | 12 of 12 |
| NO | season N=4 | 0.0330 | 0.0258 | -0.0219 | -0.0147 | 3.30 / 3.30 | 3 / 0 | 12 of 12 |
| ES | uncorrected | 0.0709 | 0.0709 | -0.0697 | -0.0696 |  |  | 12 of 12 |
| ES | fixed N=1 | 0.0339 | 0.0339 | +0.0097 | +0.0098 | 1.50 / 1.50 | 0 / 0 | 12 of 12 |
| ES | **fixed N=4** (reported) | 0.0302 | 0.0302 | +0.0099 | +0.0098 | 1.90 / 1.90 | 0 / 0 | 12 of 12 |
| ES | season N=1 | 0.0313 | 0.0313 | +0.0101 | +0.0099 | 1.86 / 1.85 | 0 / 0 | 12 of 12 |
| ES | season N=4 | 0.0259 | 0.0271 | +0.0109 | +0.0111 | 2.45 / 2.45 | 0 / 0 | 12 of 12 |
| IT | uncorrected | n/a | n/a | n/a | n/a |  |  | 0 of 12 |
| IT | fixed N=1 | n/a | n/a | n/a | n/a | n/a / n/a | 0 / 1 | 0 of 12 |
| IT | fixed N=3 | n/a | n/a | n/a | n/a | n/a / n/a | 0 / 3 | 0 of 12 |
| IT | season N=1 | n/a | n/a | n/a | n/a | 2.26 / 2.26 | 0 / 3 | 0 of 12 |
| IT | **season N=3** (reported) | n/a | n/a | n/a | n/a | 3.74 / 3.74 | 1 / 9 | 0 of 12 |
| PT | uncorrected | 0.2205 | 0.2205 | -0.2201 | -0.2201 |  |  | 3 of 12 |
| PT | fixed N=1 | 0.0463 | 0.0490 | -0.0343 | -0.0373 | 3.03 / 3.03 | 1 / 0 | 3 of 12 |
| PT | fixed N=2 | 0.0133 | 0.0133 | -0.0101 | -0.0094 | 3.96 / 3.92 | 1 / 0 | 3 of 12 |
| PT | **season N=1** (reported) | 0.0266 | 0.0300 | +0.0027 | -0.0073 | 3.10 / 3.12 | 1 / 1 | 3 of 12 |
| PT | season N=2 | 0.1493 | 0.1021 | +0.1463 | +0.1007 | 3.66 / 3.60 | 1 / 6 | 3 of 12 |

**Belgium's uncorrected error rises from 0.123 to 0.161, and its reported
corrected RMSE falls from 0.0205 to 0.0191.** Simulated at sea, its offshore
fleet produces more, so the uncorrected mean bias grows from +0.116 to +0.156;
the correction absorbs it, and the reported season N=3 variant ends closer to
the observations (MBE +0.0073 to +0.0034). Not every variant improves: both
N=1 variants get worse (fixed 0.0162 to 0.0199, season 0.0178 to 0.0207),
their scalar falling from 0.67 to 0.58, while both N=3 variants get better.
Belgium's offshore sites all join cluster 0, with the coastal land points; only
an N=3 fit gives that cluster factors of its own. The reported configuration was
chosen on this test year, before this change.

France's reported row moves from 0.0271 to 0.0260 and stays worse than
uncorrected (0.0204), on the same degenerate fit (scalar 4.61); every training
scalar is unchanged, because France's 2021 grid carries no offshore capacity,
and the move comes from Saint-Nazaire in the 2023 grid. Norway's reported row
moves from 0.0279 to 0.0260 on the same degenerate fit (3.05); its season N=4
variant now has three implausible scalars against two. Spain's reported row is
unchanged, but its season N=4 variant gets worse (0.0259 to 0.0271). Ireland
and Sweden move in the fourth decimal at most. Italy stays suspended with no
month scored, and Portugal with 3 of 12.

Data: `output/offshore_regen_2026-10-16/` (`after/`, `before_after_metrics.txt`,
`grid_diff.txt`, `regen_grids.log`), with every replaced grid under `before/`,
its checksums in `before.sha256`, and the new grids' in `after.sha256`.

**Correction notice, 2026-09-29: Ireland's row moves with the corrected
wind-fleet register, and the other country rows stand.** Since `acfbbfa`
(#73), a year's fleet from the Global Wind Power Tracker includes projects
retired since, placed by their start and retirement years, and takes start
years from national registers for records the tracker leaves undated
(`configs/curation/gwpt_start_years.csv`). Until then a retired project was
missing from every year it ran, and an undated one was counted in every year.
The per-year country grids, and Ireland's capacity-factor denominator, which is
built from the same register (`scripts/region_tools/repair_country_capacity.py`),
were regenerated, and all eight rows were trained and evaluated again at
`6ab07ef` from a clean tree, with the scorecard configs, on `input/combined`
(`power_curves.csv` sha256 `689cfee7…`); every run records 0% of capacity
substituted. Training years 2015-21 (IE 2017-21), test year 2023, national
scope.

A scorecard run reads two grids per row: the last training year's (2021) and
the test year's (2023). Those moved in three rows:

- **Ireland:** +60 MW in 2021, the Derrybrien wind farm (2006 to 2022), restored
  at one grid point. The training denominator rises by the same 60 MW in every
  training year, which lowers the observed mean capacity factor over 2017-21
  from 0.322 to 0.316. The 2023 grid and observations are unchanged.
- **France:** -159 MW in 2021 across ten grid points, and -41 MW in 2023, from
  records the registers date after 2021 that had been counted in every year.
- **Italy:** +20 MW in 2021, Camporeale phase 1 (2005 to 2023).

Belgium, Sweden, Norway, Spain and Portugal read byte-identical grids, and their
runs reproduce the previous `metrics.csv` byte for byte, which also shows that
nothing else changed between the two sets of runs. Norway's correction (296 MW
of 2018-21 projects that had been counted from 2015, and a retired phase
restored) falls in 2015-20 only, which no scorecard run reads.

Every variant of the three rows that moved, before (`bf97f4a`, the third
notice of 2026-09-25) and after (`6ab07ef`):

| Row | Variant | RMSE before | RMSE after | MBE before | MBE after | Max scalar before / after | Implausible / failed after | Months scored after |
|---|---|---|---|---|---|---|---|---|
| FR | uncorrected | 0.0204 | 0.0204 | -0.0015 | -0.0016 |  |  | 12 of 12 |
| FR | fixed N=1 | 0.0301 | 0.0302 | +0.0155 | +0.0156 | 1.07 / 1.07 | 0 / 0 | 12 of 12 |
| FR | **fixed N=10** (reported) | 0.0271 | 0.0271 | +0.0123 | +0.0123 | 4.61 / 4.61 | 1 / 0 | 12 of 12 |
| FR | season N=1 | 0.0280 | 0.0281 | +0.0166 | +0.0167 | 1.23 / 1.23 | 0 / 0 | 12 of 12 |
| FR | season N=10 | 0.0251 | 0.0251 | +0.0139 | +0.0138 | 9.94 / 9.94 | 4 / 0 | 12 of 12 |
| IE | uncorrected | 0.0453 | 0.0453 | +0.0361 | +0.0361 |  |  | 12 of 12 |
| IE | fixed N=1 | 0.0264 | 0.0239 | +0.0115 | +0.0062 | 0.93 / 0.91 | 0 / 0 | 12 of 12 |
| IE | fixed N=3 | 0.0260 | 0.0236 | +0.0107 | +0.0052 | 0.97 / 0.96 | 0 / 0 | 12 of 12 |
| IE | **season N=1** (reported) | 0.0239 | 0.0213 | +0.0114 | +0.0061 | 0.96 / 0.94 | 0 / 0 | 12 of 12 |
| IE | season N=3 | 0.0237 | 0.0211 | +0.0108 | +0.0053 | 1.02 / 1.00 | 0 / 0 | 12 of 12 |
| IT | uncorrected | n/a | n/a | n/a | n/a |  |  | 0 of 12 |
| IT | fixed N=1 | n/a | n/a | n/a | n/a | n/a / n/a | 0 / 1 | 0 of 12 |
| IT | fixed N=3 | n/a | n/a | n/a | n/a | n/a / n/a | 0 / 3 | 0 of 12 |
| IT | season N=1 | n/a | n/a | n/a | n/a | 2.67 / 2.26 | 0 / 3 | 0 of 12 |
| IT | **season N=3** (reported) | n/a | n/a | n/a | n/a | 3.49 / 3.74 | 1 / 9 | 0 of 12 |

**Ireland's corrected RMSE falls from 0.024 to 0.021, and its corrected mean
bias from +0.011 to +0.006**, by 0.002 to 0.003 in RMSE in every variant, with
no implausible scalar and no refused factor. France changes in the fourth
decimal and its reported row not at all: it stays worse than uncorrected, on
the same degenerate fit. Italy stays suspended and scores no month; its season
N=1 fit refuses one more factor than before (3 against 2). Ireland's reported
configuration is still not its lowest-RMSE variant (season N=3, 0.0211,
against 0.0213), and was chosen on the same test year.

Data: `output/grid_regen_2026-09-29/` (`after/`, `before_after_metrics.txt`,
`grid_diff.txt`, `repair_ie.log`), with every replaced grid and observation
file, and their checksums, under `before/`.

**Correction notice, 2026-09-25 (third): the eight country-level rows now
simulate each grid's own turbine, and Italy and Portugal are suspended.** Every
country grid point names one Vestas key: `Vestas.V80.2000` (FR, IT, PT),
`Vestas.V90.2000` (ES, IE) or `Vestas.V90.3000` (BE, NO, SE). Only the
licensed library holds those curves, and the rows ran on the default input root,
so every unit was simulated on the open library's first column,
`2019COE_DW100_100kW_27.6`, a 100 kW distributed-wind curve at 167 W/m2. The
Substituted column has said 100% since 2026-09-11. From `a4f6f9b` a country-level
run refuses a curve the library lacks, and the rows run on `input/combined`
(`power_curves.csv` sha256 `689cfee7…`), where each key resolves to its own
curve. This is condition C1 of `method-curve-library.md`, now with the joint
national fit, in preference to the nearest open curve, which sits
19.4 W/m2 from the V80-2.0 and 70.6 W/m2 from the V90-3.0. **The eight rows
are therefore licensed-lineage and not reproducible by a third party**, like DE,
DK, UK and US.

Every variant of the eight rows, trained and evaluated before the change
(`26e0b67`, default input root, the "after" runs of the second notice below)
and after it (`bf97f4a`, `input/combined`), from a clean tree, with the
scorecard configs. Training years 2015-21 (IE 2017-21), test year 2023,
national scope. Every after manifest records 0% of capacity substituted. Fit
quality uses the dagger's bounds: a scalar outside 0.2 to 3.0 is implausible,
and a refused factor counts as a failed offset.

| Row | Variant | RMSE before | RMSE after | MBE before | MBE after | Max scalar before / after | Implausible / failed after | Months scored after |
|---|---|---|---|---|---|---|---|---|
| FR | uncorrected | 0.1711 | 0.0204 | +0.1648 | -0.0015 |  |  | 12 of 12 |
| FR | fixed N=1 | 0.0125 | 0.0301 | +0.0074 | +0.0155 | 0.62 / 1.07 | 0 / 0 | 12 of 12 |
| FR | **fixed N=10** (reported) | 0.0102 | 0.0271 | +0.0030 | +0.0123 | 2.03 / 4.61 | 1 / 0 | 12 of 12 |
| FR | season N=1 | 0.0126 | 0.0280 | +0.0069 | +0.0166 | 0.62 / 1.23 | 0 / 0 | 12 of 12 |
| FR | season N=10 | 0.0113 | 0.0251 | +0.0030 | +0.0139 | 3.06 / 9.94 | 4 / 0 | 12 of 12 |
| BE | uncorrected | 0.3399 | 0.1226 | +0.3367 | +0.1164 |  |  | 12 of 12 |
| BE | fixed N=1 | 0.0243 | 0.0162 | -0.0055 | -0.0001 | 0.39 / 0.67 | 0 / 0 | 12 of 12 |
| BE | fixed N=3 | 0.0266 | 0.0160 | -0.0092 | +0.0062 | 0.50 / 0.95 | 0 / 0 | 12 of 12 |
| BE | season N=1 | 0.0225 | 0.0178 | -0.0065 | -0.0000 | 0.43 / 0.69 | 0 / 0 | 12 of 12 |
| BE | **season N=3** (reported) | 0.0247 | 0.0205 | -0.0102 | +0.0073 | 0.53 / 1.23 | 0 / 0 | 12 of 12 |
| IE | uncorrected | 0.1721 | 0.0453 | +0.1680 | +0.0361 |  |  | 12 of 12 |
| IE | fixed N=1 | 0.0230 | 0.0264 | +0.0105 | +0.0115 | 0.68 / 0.93 | 0 / 0 | 12 of 12 |
| IE | fixed N=3 | 0.0208 | 0.0260 | +0.0075 | +0.0107 | 0.70 / 0.97 | 0 / 0 | 12 of 12 |
| IE | **season N=1** (reported) | 0.0212 | 0.0239 | +0.0092 | +0.0114 | 0.70 / 0.96 | 0 / 0 | 12 of 12 |
| IE | season N=3 | 0.0197 | 0.0237 | +0.0063 | +0.0108 | 0.73 / 1.02 | 0 / 0 | 12 of 12 |
| SE | uncorrected | 0.0876 | 0.0984 | +0.0844 | -0.0967 |  |  | 12 of 12 |
| SE | fixed N=1 | 0.0319 | 0.0358 | -0.0289 | -0.0329 | 0.68 / 1.37 | 0 / 0 | 12 of 12 |
| SE | **fixed N=4** (reported) | 0.0284 | 0.0348 | -0.0247 | -0.0313 | 0.77 / 1.64 | 0 / 0 | 12 of 12 |
| SE | season N=1 | 0.0325 | 0.0359 | -0.0292 | -0.0336 | 0.70 / 1.47 | 0 / 0 | 12 of 12 |
| SE | season N=4 | 0.0287 | 0.0336 | -0.0246 | -0.0303 | 0.82 / 1.84 | 0 / 0 | 12 of 12 |
| NO | uncorrected | 0.0350 | 0.1286 | +0.0271 | -0.1245 |  |  | 12 of 12 |
| NO | fixed N=1 | 0.0390 | 0.0392 | -0.0314 | -0.0326 | 0.82 / 1.47 | 0 / 0 | 12 of 12 |
| NO | **fixed N=4** (reported) | 0.0357 | 0.0279 | -0.0267 | -0.0160 | 1.32 / 3.05 | 1 / 0 | 12 of 12 |
| NO | season N=1 | 0.0386 | 0.0397 | -0.0304 | -0.0324 | 0.88 / 1.57 | 0 / 0 | 12 of 12 |
| NO | season N=4 | 0.0368 | 0.0330 | -0.0255 | -0.0219 | 1.48 / 3.30 | 2 / 0 | 12 of 12 |
| ES | uncorrected | 0.0281 | 0.0709 | +0.0128 | -0.0697 |  |  | 12 of 12 |
| ES | fixed N=1 | 0.0269 | 0.0339 | +0.0109 | +0.0097 | 0.99 / 1.50 | 0 / 0 | 12 of 12 |
| ES | **fixed N=4** (reported) | 0.0257 | 0.0302 | +0.0114 | +0.0099 | 1.20 / 1.90 | 0 / 0 | 12 of 12 |
| ES | season N=1 | 0.0265 | 0.0313 | +0.0110 | +0.0101 | 1.10 / 1.86 | 0 / 0 | 12 of 12 |
| ES | season N=4 | 0.0265 | 0.0259 | +0.0125 | +0.0109 | 1.36 / 2.45 | 0 / 0 | 12 of 12 |
| IT | uncorrected | 0.0703 | n/a | -0.0692 | n/a |  |  | 0 of 12 |
| IT | fixed N=1 | 0.0169 | n/a | -0.0042 | n/a | 1.38 / n/a | 0 / 1 | 0 of 12 |
| IT | fixed N=3 | 0.0166 | n/a | -0.0043 | n/a | 1.92 / n/a | 0 / 3 | 0 of 12 |
| IT | season N=1 | 0.0164 | n/a | -0.0040 | n/a | 1.73 / 2.67 | 0 / 2 | 0 of 12 |
| IT | **season N=3** (reported) | 0.0155 | n/a | -0.0041 | n/a | 2.65 / 3.49 | 1 / 9 | 0 of 12 |
| PT | uncorrected | 0.0893 | 0.2205 | -0.0847 | -0.2201 |  |  | 3 of 12 |
| PT | fixed N=1 | 0.0293 | 0.0463 | +0.0177 | -0.0343 | 1.55 / 3.03 | 1 / 0 | 3 of 12 |
| PT | fixed N=2 | 0.0310 | 0.0133 | +0.0193 | -0.0101 | 1.94 / 3.96 | 1 / 0 | 3 of 12 |
| PT | **season N=1** (reported) | 0.0274 | 0.0266 | +0.0176 | +0.0027 | 1.65 / 3.10 | 1 / 1 | 3 of 12 |
| PT | season N=2 | 0.0278 | 0.1493 | +0.0160 | +0.1463 | 2.35 / 3.66 | 1 / 6 | 3 of 12 |

**Italy and Portugal are suspended.** On their own curves the joint fit refuses
54 of Italy's 70 training periods and 33 of Portugal's (offsets on the ±10 m/s
bound, or an abnormal end to the line search); Norway refuses 2 and Spain 1,
every other row none. Italy's N=1 fixed fit keeps 1 of 7 training years and its
N=3 fixed fit none, so each is refused whole, and because every variant is
scored on the rows all of them can score, Italy scores no month in any variant.
Portugal scores 3 of 12 months. Neither row is a result. They move to the
suspended rows under the country-level table, with the figures they carried
until today. Joint-fit record: `output/country_curves_2026-09-25/diag/`.

The country-level table below now carries the after figures of the other six
rows, in each row's reported configuration, which is kept as reported. Two of
them are degenerate and take a dagger: France's N=10 fixed fit applies a scalar
of 4.61 (cluster 7) and Norway's N=4 fixed fit one of 3.05. Until today the
table carried: FR 0.171 / 0.010, BE 0.340 / 0.025, IE 0.172 / 0.021, SE 0.088 /
0.028, NO 0.035 / 0.036, ES 0.028 / 0.026, IT 0.070 / 0.015, PT 0.089 / 0.027
(uncorrected / corrected RMSE).

The reported configurations are kept, and are no longer the lowest-RMSE
variant in five of the six rows: FR (season N=10, 0.0251, with four
implausible scalars, against 0.0271), BE (fixed N=3, 0.0160, against 0.0205),
IE (season N=3, 0.0237, against 0.0239), SE (season N=4, 0.0336, against
0.0348) and ES (season N=4, 0.0259, against 0.0302). Norway's reported
configuration is its lowest. Every configuration was chosen on the same test
year it is scored on.

Not re-measured: every other figure in this repository from a country-level run,
which all ran on the fallback curve. Data: `output/country_curves_2026-09-25/`
(`after/`, `diag/`, `pins/`, `before_after_metrics.csv`).

**Correction notice, 2026-09-25 (second): the joint national fit stopped
at its starting point in 47 of 540 training periods.** The fit minimises the
squared national capacity-factor error, about 1e-4 at its start, and L-BFGS-B's
`ftol` of 1e-6 acts as an absolute threshold on a value that small, so a fit
whose first step lowered the error by less stopped there and reported success.
The record of every joint fit of the eight rows at `dc14687` (training years
2015-21, IE 2017-21) has 45 fits stopped after one iteration and 2 after two
(ES 25, NO 21, BE 1). Every fit that left an error above 1e-3 (ES 22, NO 16,
largest 1.1e-2) is one of them, with its offsets within 0.054 m/s of zero.
Fixed in `26e0b67` (`ftol` 1e-12, `gtol` 1e-8). Re-recorded after the fix: 540
fits, 3 to 11 iterations each, none refused, at the iteration limit or on a
bound, largest error 1.6e-7. This corrects the figures of the notice below,
which were made with the fit as it was.

Every variant of the eight rows, trained and evaluated before the fix (`ac26f6a`,
the "after" runs of the notice below) and after it (`26e0b67`), from a clean
tree, on the default input root, with the scorecard configs. Training years
2015-21 (IE 2017-21), test year 2023, national scope, 12 months each. Every
factors file changed, N=1 included, since national N=1 fits also take the
joint fit. No fit failed an offset either way. All eight rows simulate 100% of
capacity on the substituted `2019COE_DW100_100kW_27.6` curve, as before.

| Row | Variant | RMSE before | RMSE after | MBE before | MBE after |
|---|---|---|---|---|---|
| FR | uncorrected | 0.17111 | 0.17111 | +0.16478 | +0.16478 |
| FR | fixed N=1 | 0.01249 | 0.01249 | +0.00738 | +0.00739 |
| FR | **fixed N=10** (reported) | 0.01023 | 0.01024 | +0.00295 | +0.00297 |
| FR | season N=1 | 0.01258 | 0.01258 | +0.00689 | +0.00689 |
| FR | season N=10 | 0.01125 | 0.01126 | +0.00294 | +0.00296 |
| BE | uncorrected | 0.33991 | 0.33991 | +0.33672 | +0.33672 |
| BE | fixed N=1 | 0.02429 | 0.02429 | -0.00553 | -0.00552 |
| BE | fixed N=3 | 0.02657 | 0.02657 | -0.00920 | -0.00920 |
| BE | season N=1 | 0.02252 | 0.02252 | -0.00654 | -0.00653 |
| BE | **season N=3** (reported) | 0.02468 | 0.02468 | -0.01016 | -0.01016 |
| IE | uncorrected | 0.17208 | 0.17208 | +0.16796 | +0.16796 |
| IE | fixed N=1 | 0.02301 | 0.02301 | +0.01047 | +0.01045 |
| IE | fixed N=3 | 0.02079 | 0.02079 | +0.00749 | +0.00750 |
| IE | **season N=1** (reported) | 0.02123 | 0.02123 | +0.00917 | +0.00916 |
| IE | season N=3 | 0.01971 | 0.01971 | +0.00627 | +0.00628 |
| SE | uncorrected | 0.08760 | 0.08760 | +0.08441 | +0.08441 |
| SE | fixed N=1 | 0.03191 | 0.03192 | -0.02887 | -0.02888 |
| SE | **fixed N=4** (reported) | 0.02844 | 0.02844 | -0.02470 | -0.02469 |
| SE | season N=1 | 0.03254 | 0.03253 | -0.02922 | -0.02921 |
| SE | season N=4 | 0.02868 | 0.02868 | -0.02459 | -0.02458 |
| NO | uncorrected | 0.03495 | 0.03495 | +0.02708 | +0.02708 |
| NO | fixed N=1 | 0.03954 | 0.03901 | -0.03206 | -0.03141 |
| NO | **fixed N=4** (reported) | 0.03515 | 0.03570 | -0.02594 | -0.02667 |
| NO | season N=1 | 0.03847 | 0.03863 | -0.03026 | -0.03037 |
| NO | season N=4 | 0.03700 | 0.03684 | -0.02506 | -0.02555 |
| ES | uncorrected | 0.02813 | 0.02813 | +0.01278 | +0.01278 |
| ES | fixed N=1 | 0.02672 | 0.02688 | +0.01054 | +0.01089 |
| ES | **fixed N=4** (reported) | 0.02533 | 0.02569 | +0.01066 | +0.01138 |
| ES | season N=1 | 0.02654 | 0.02655 | +0.01076 | +0.01102 |
| ES | season N=4 | 0.02729 | 0.02653 | +0.01331 | +0.01250 |
| IT | uncorrected | 0.07029 | 0.07029 | -0.06920 | -0.06920 |
| IT | fixed N=1 | 0.01692 | 0.01693 | -0.00416 | -0.00417 |
| IT | fixed N=3 | 0.01657 | 0.01657 | -0.00426 | -0.00426 |
| IT | season N=1 | 0.01636 | 0.01636 | -0.00394 | -0.00397 |
| IT | **season N=3** (reported) | 0.01549 | 0.01550 | -0.00405 | -0.00414 |
| PT | uncorrected | 0.08927 | 0.08927 | -0.08469 | -0.08469 |
| PT | fixed N=1 | 0.02930 | 0.02930 | +0.01767 | +0.01767 |
| PT | fixed N=2 | 0.03097 | 0.03098 | +0.01925 | +0.01926 |
| PT | **season N=1** (reported) | 0.02741 | 0.02740 | +0.01756 | +0.01755 |
| PT | season N=2 | 0.02780 | 0.02779 | +0.01600 | +0.01598 |

The country table below now carries the after figures. At its three decimals
two rows change: Spain's corrected RMSE, 0.025 to 0.026, and Norway's, 0.035 to
0.036, with its corrected MBE -0.026 to -0.027. Norway's corrected RMSE, 0.0357,
is above its uncorrected 0.0350. The reported configuration remains the
lowest-RMSE variant in ES and NO; BE and IE are as the notice below states.
The figures the notice below lists as not re-measured are not re-measured here
either.

Data: `output/joint_fit_tolerance_2026-09-25/` (`after/`, `diag/`,
`before_after_metrics.csv`); before-fix fit record
`output/c2_country_offsets_2026-09-25_joint/after/`.

**Correction notice, 2026-09-25: every country-level fit with more than one
cluster used the per-cluster solver, not the joint national fit.** The router
`country_obs_is_per_cluster` counted distinct observation values per period.
Each cluster's observation is a capacity-weighted mean of one national number,
and those means differ in their last bits (1.1e-16 for FR at N=10, 8.3e-17 for
ES at N=4), so every national fit with N greater than 1 was treated as zonal and
each cluster's offset was fitted alone against the national series. The
documented method, and every statement here about under-determined joint
offsets, describes the joint fit. Fixed in `ac26f6a`. The scalars are computed
before the routing and do not change; neither does any N=1 fit, nor any
uncorrected figure. The defect entered with `dd87232` (2026-08-12), and the
July runs behind the earliest country figures already show the same factors.

Every variant of the eight rows, trained and evaluated before the fix (`eb6b351`)
and after it (`ac26f6a`), from a clean tree, on the default input root, with the
scorecard configs. Training years 2015-21 (IE 2017-21), test year 2023,
national scope, 12 months each. Before the fix, every row reproduced its
`refresh_2026-09-20` `metrics.csv` to within 3e-9. As predicted before the runs,
all 16 N=1 factors files are byte-identical and all 16 N greater than 1 files
changed. No fit failed an offset either way.

| Row | Variant | RMSE before | RMSE after | MBE before | MBE after |
|---|---|---|---|---|---|
| FR | uncorrected | 0.1711 | 0.1711 | +0.1648 | +0.1648 |
| FR | fixed N=1 | 0.0125 | 0.0125 | +0.0074 | +0.0074 |
| FR | **fixed N=10** (reported) | 0.0122 | 0.0102 | +0.0062 | +0.0029 |
| FR | season N=1 | 0.0126 | 0.0126 | +0.0069 | +0.0069 |
| FR | season N=10 | 0.0130 | 0.0113 | +0.0064 | +0.0029 |
| BE | uncorrected | 0.3399 | 0.3399 | +0.3367 | +0.3367 |
| BE | fixed N=1 | 0.0243 | 0.0243 | -0.0055 | -0.0055 |
| BE | fixed N=3 | 0.0214 | 0.0266 | -0.0016 | -0.0092 |
| BE | season N=1 | 0.0225 | 0.0225 | -0.0065 | -0.0065 |
| BE | **season N=3** (reported) | 0.0201 | 0.0247 | -0.0024 | -0.0102 |
| IE | uncorrected | 0.1721 | 0.1721 | +0.1680 | +0.1680 |
| IE | fixed N=1 | 0.0230 | 0.0230 | +0.0105 | +0.0105 |
| IE | fixed N=3 | 0.0230 | 0.0208 | +0.0103 | +0.0075 |
| IE | **season N=1** (reported) | 0.0212 | 0.0212 | +0.0092 | +0.0092 |
| IE | season N=3 | 0.0213 | 0.0197 | +0.0092 | +0.0063 |
| SE | uncorrected | 0.0876 | 0.0876 | +0.0844 | +0.0844 |
| SE | fixed N=1 | 0.0319 | 0.0319 | -0.0289 | -0.0289 |
| SE | **fixed N=4** (reported) | 0.0298 | 0.0284 | -0.0272 | -0.0247 |
| SE | season N=1 | 0.0325 | 0.0325 | -0.0292 | -0.0292 |
| SE | season N=4 | 0.0301 | 0.0287 | -0.0268 | -0.0246 |
| NO | uncorrected | 0.0350 | 0.0350 | +0.0271 | +0.0271 |
| NO | fixed N=1 | 0.0395 | 0.0395 | -0.0321 | -0.0321 |
| NO | **fixed N=4** (reported) | 0.0363 | 0.0351 | -0.0279 | -0.0259 |
| NO | season N=1 | 0.0385 | 0.0385 | -0.0303 | -0.0303 |
| NO | season N=4 | 0.0366 | 0.0370 | -0.0264 | -0.0251 |
| ES | uncorrected | 0.0281 | 0.0281 | +0.0128 | +0.0128 |
| ES | fixed N=1 | 0.0267 | 0.0267 | +0.0105 | +0.0105 |
| ES | **fixed N=4** (reported) | 0.0262 | 0.0253 | +0.0112 | +0.0107 |
| ES | season N=1 | 0.0265 | 0.0265 | +0.0108 | +0.0108 |
| ES | season N=4 | 0.0272 | 0.0273 | +0.0124 | +0.0133 |
| IT | uncorrected | 0.0703 | 0.0703 | -0.0692 | -0.0692 |
| IT | fixed N=1 | 0.0169 | 0.0169 | -0.0042 | -0.0042 |
| IT | fixed N=3 | 0.0171 | 0.0166 | -0.0040 | -0.0043 |
| IT | season N=1 | 0.0164 | 0.0164 | -0.0039 | -0.0039 |
| IT | **season N=3** (reported) | 0.0169 | 0.0155 | -0.0031 | -0.0041 |
| PT | uncorrected | 0.0893 | 0.0893 | -0.0847 | -0.0847 |
| PT | fixed N=1 | 0.0293 | 0.0293 | +0.0177 | +0.0177 |
| PT | fixed N=2 | 0.0300 | 0.0310 | +0.0180 | +0.0192 |
| PT | **season N=1** (reported) | 0.0274 | 0.0274 | +0.0176 | +0.0176 |
| PT | season N=2 | 0.0277 | 0.0278 | +0.0162 | +0.0160 |

The country table below now carries the after figures of each row's reported
variant. *[2026-09-25, later: for ES and NO, superseded by the notice above.]*
Until today it carried: FR 0.012 / +0.006, BE 0.020 / -0.002, SE
0.030 / -0.027, NO 0.036 / -0.028, ES 0.026 / +0.011, IT 0.017 / -0.003
(corrected RMSE / MBE). IE and PT report N=1 and do not change.

The "Best cfg" column is left as reported. After the fix it is no longer the
lowest-RMSE variant in two rows: BE (N=1 season, 0.0225, against the reported
N=3 season, 0.0247) and IE (N=3 season, 0.0197, against the reported N=1
season, 0.0212). Re-selecting the configuration is not done here.

Not re-measured: the figures from multi-cluster country fits in the notices
below (marked where they stand in this document) and in the other documents
that quote country-level multi-cluster offsets, corrected metrics,
gains or failed-offset counts (`method-country-level.md`, `method-eu-rerun.md`,
`method-curve-library.md` and its preregistration,
`method-roughness-treatment.md` and its preregistration, and the cost figures in
`method-cluster-selection-prereg.md` and
`method-national-single-cluster-prereg.md`). Data:
`output/country_routing_2026-09-25/` (`before/`, `after/`,
`before_after_metrics.csv`).

**Correction notice, 2026-09-24: the corrected figures of the AU-NEM and NZ
rows were simulated on other units' power curves and capacities.**
`correct_wind_speed` rebuilt the turbine axis in sorted ID order and then
attached each unit's model key and capacity by position, in the fleet's own
order. A fleet whose IDs were not already sorted therefore ran its corrected
simulation with curves and capacities belonging to other units. Fixed in
`a94670c`. The factors, the fit quality and every uncorrected figure are
unaffected: training and the uncorrected simulation build the turbine axis in
fleet order. Of the seventeen rows, only AU-NEM and NZ have test fleets that
are both unsorted and on more than one model key, so only their corrected
figures moved. CL, whose fleet is sorted and on ten model keys, was re-run as
a control and reproduced its `metrics.csv` byte for byte.

Each row's full `metrics.csv`, before and after the fix. Both are evaluations
of the unchanged `refresh_2026-09-20` training runs, from a clean tree, on the
licensed library as input root (`power_curves.csv` sha256 `689cfee7…`), with
every unit resolved to a curve of `open` origin. Before the fix, at `b6bdfe4`,
each reproduced its published `metrics.csv` byte for byte.

AU-NEM, k45 season, training years 2020-22, test year 2023, 75 farms scored:

| Variant | RMSE before | RMSE after | MBE before | MBE after | MAE before | MAE after | r before | r after |
|---|---|---|---|---|---|---|---|---|
| uncorrected | 0.1160 | 0.1160 | +0.0086 | +0.0086 | 0.0953 | 0.0953 | 0.460 | 0.460 |
| affine-wind k45 season | 0.0933 | **0.0787** | -0.0051 | **+0.0011** | 0.0742 | **0.0592** | 0.615 | **0.694** |

NZ, k7 fixed, training years 2019-23, test year 2024, 12 farms scored:

| Variant | RMSE before | RMSE after | MBE before | MBE after | MAE before | MAE after | r before | r after |
|---|---|---|---|---|---|---|---|---|
| uncorrected | 0.1568 | 0.1568 | -0.0617 | -0.0617 | 0.1426 | 0.1426 | 0.639 | 0.639 |
| affine-wind k7 fixed | 0.1063 | **0.1046** | +0.0206 | **+0.0253** | 0.0777 | **0.0761** | 0.663 | **0.742** |

The turbine-level table below now carries the after figures. The figures it
carried until today were: AU-NEM corrected RMSE 0.093, MBE -0.005, r 0.62; NZ
corrected RMSE 0.106, MBE +0.021, r 0.66. Fit quality is unchanged: AU-NEM
keeps its dagger (maximum scalar 2.64, four failed offsets), and NZ stays clean
(maximum scalar 1.81).

- **NZ keeps ‡.** Re-resampled on the frames after the fix with the unchanged
  driver (1,000 paired draws, seed 20260911), its RMSE gain is 0.052 with a 95%
  interval of -0.032 to 0.115; on the frames before the fix it was 0.051,
  -0.032 to 0.114. The interval includes zero either way.
- **AU-NEM's resampled gain is not known.** The interval quoted in the notice
  below (0.001 to 0.040) was computed on frames with this defect, so it is
  withdrawn. It cannot be recomputed yet: `baseline_bootstrap.py` stops on the
  `refresh_2026-09-20` AU-NEM run, before and after the fix alike, because its
  rebuilt uncorrected MBE (0.00886) does not reproduce the run's `metrics.csv`
  (0.00856). *[Correction to this bullet, 2026-09-24: the mismatch was the
  driver scoring the uncorrected frame on 24 unit-months that the run's
  common-row scoring had dropped (`scoring_exclusions.csv`); `95d0ec1` applies
  those exclusions, and the rebuild then reproduces `metrics.csv` to 1e-16.
  Re-resampled with the unchanged draws (1,000, seed 20260911), 75 farms: the
  RMSE gain is **0.037, 95% interval 0.019 to 0.058**, after the fix, and was
  0.023, 0.002 to 0.043, before it. The interval excludes zero, so the row
  takes no ‡. Data: `output/c1_turbine_order_2026-09-24/bootstrap_before/` and
  `bootstrap_after/`.]*
- **Other documents.** `region-nz.md` carries its own notice of the same date,
  with its run re-evaluated before and after the fix. `region-au-nem.md`
  carries a notice withdrawing every corrected figure, because its runs were
  not retained and cannot be re-evaluated. `method-generalisation.md` carries a
  notice withdrawing its Australia/NEM corrected figures and its DK → AU
  transfer, with both transfers re-run before and after the fix from this
  scorecard's training runs.

Data: `output/c1_turbine_order_2026-09-24/` (`before/` and `after/` evaluate
runs, and `bootstrap_before/` and `bootstrap_after/`).

**Correction notice, 2026-09-11: the UK and NZ rows do not show that the
correction improves those regions.** Both rows report a lower corrected RMSE,
and the reading under the turbine-level table counts both among the clean rows
where the correction lowers RMSE and drives MBE toward zero. Every figure in
both rows stands as measured. The claim does not: when the test year's units
are resampled, neither row's gain can be distinguished from zero, and in both a
few units that the correction makes worse decide the result.

| Row | Units | Capacity-effective units | RMSE gain [95% interval] | MAE gain [95% interval] | Corrected squared error: largest unit, largest five |
|---|---|---|---|---|---|
| UK k50 fixed | 348 farms | 104 | 0.031 [-0.001, 0.069] | 0.034 [0.014, 0.057] | 37%, 60% |
| NZ k7 fixed | 12 farms | 9.3 | 0.051 [-0.032, 0.114] | 0.065 [-0.018, 0.119] | 61%, 87% |

*[Note, 2026-09-24: the NZ row above was computed on corrected frames that
simulated units on other units' curves. On the `refresh_2026-09-20` frames
after the fix, the RMSE gain is 0.052 [-0.032, 0.115], so the ‡ marker stands; the MAE interval and the
concentration shares were not recomputed. See the notice of that date.]*

The gain is uncorrected minus corrected error, capacity-weighted as in
`metrics.csv`. Each interval comes from 1,000 paired draws of the test year's
units with replacement. Capacity-effective units is (sum of weights) squared
over the sum of squared weights. Every row's rebuilt metrics reproduced its
`metrics.csv` before resampling. Scripts: `scripts/analysis/baseline_bootstrap.py`
and `scripts/analysis/unit_concentration.py`. Data:
`output/curve_library_study_2026-09-11/baseline_bootstrap/`
(`all_concentration.csv`, `UK_top5_units.csv`, `NZ_top5_units.csv`), from the
`curve_resolution_backfill_2026-09-11` evaluate frames.

- **UK.** Four of the five farms that carry the most corrected error are worse
  corrected than uncorrected. The largest, 2.2% of capacity, goes from a farm
  RMSE of 0.31 to 0.47. MBE moves from +0.037 to -0.038, not toward zero. The
  MAE gain is resolved; the RMSE gain is not. The choice of `k=50` over `k=100`
  (0.115 against 0.123) cannot be resampled without a re-run, because the
  cluster sweep's corrected frames were not saved. It remains open.
- **NZ.** One farm, 12% of capacity, goes from a farm RMSE of 0.05 to 0.24 and
  carries 61% of the corrected squared error. Neither gain is resolved.
- **The NZ limit is structural.** With 12 farms, 9.3 of them
  capacity-effective, any one farm can decide the fleet result. Such a fleet
  resolves a gain only if nearly every farm improves by a similar amount, in
  any test year. The marker records a limit of the region, not an unlucky year.
- **The UK's 348 units are farms, not turbines.** They are ROC stations whose
  generation is spread equally over turbine-shaped rows (`docs/design/harness.md`).
  Until this notice, the table's Fleet column said 348 turbines.
- **The MBE reading is also false for the US**, whose MBE moves from +0.022
  to +0.024.

These intervals understate the uncertainty, for three reasons. They are
conditional on each row's single test year. They treat units as independent,
although neighbouring units share weather. And both rows' cluster counts were
chosen as the best of a sweep on that same test year (`method-cluster-count.md`),
and the resampling leaves that choice out.

The same check finds no such pattern in DE, DK or US. Their RMSE gain intervals
are 0.027 to 0.031, 0.059 to 0.063 and 0.008 to 0.017, and no single unit
carries more than 2.6% of corrected squared error in any of them. BR's gain is
also resolved, at 0.016 to 0.050.

Two rows are resolved only narrowly, and their claims are not withdrawn.
AU-NEM's RMSE gain is 0.021, with an interval of 0.001 to 0.040 *[withdrawn
2026-09-24: computed on corrected frames that simulated units on other units'
curves; on the `refresh_2026-09-20` run after the fix it is 0.037, 0.019 to
0.058; see the notice of that date]*. AR's is 0.018,
with an interval of 0.002 to 0.031, and its MAE gain interval, -0.001 to 0.024,
includes zero. The interval is itself a lower bound on the uncertainty, so an
interval that excludes zero by 0.001 or 0.002 is not a clean result. CL is not
assessed in this notice.

**Correction notice, 2026-09-11: the daggered rows' worst damage never reached
their scores.** The reading under the turbine-level table says that, in the four
daggered rows, the fleet average absorbed the damage of a degenerate fit. It did
not absorb it. The corrected score never contained it.

A degenerate cluster loses corrected values in two ways. A failed offset leaves
the cluster with no factors to apply. An implausible scalar pushes corrected
speeds above 40 m/s, where the power curves end, and a large negative offset
pushes them below 0 m/s. Outside the curve, the interpolator returns no value
rather than zero output. Until 2026-09-11 a missing value dropped out of the
score, and a monthly mean still skips missing days. So the corrected score
left out exactly the units and days that a degenerate fit damaged most.

| Row | Degenerate clusters with missing values | Units (capacity) | Missing unit-days: failed offset / above 40 m/s / below 0 m/s | Unit-months wholly / partly missing | Corrected RMSE, as scored / with off-curve days as zero |
|---|---|---|---|---|---|
| CL k10 fixed | 6, 8 | 6 plants (26%) | 1,464 / 695 / 0 | 65 / 7 | 0.104 / 0.109 |
| AR k10 fixed | 7 | 2 plants (5.1%) | 0 / 308 / 0 | 0 / 24 | 0.133 / 0.132 |
| US k250 fixed | 15, 38, 102, 156, 183 | 28 plants (1.4%) | 0 / 727 / 2,569 | 21 / 244 | 0.097 / 0.096 |
| BR k60 fixed | 29 | 1 complex (0.3%) | 0 / 0 / 128 | 0 / 12 | 0.105 / 0.107 |

The last column scores the saved frames on common rows twice, the second time
with every off-curve corrected day counted as zero output. Zero is the physical
value below cut-in and above cut-out. This is a measurement, and the simulation
is unchanged. Scripts: `scripts/analysis/missing_value_audit.py`,
`off_curve_sensitivity.py` and `common_row_rescore.py`. Data:
`output/curve_library_study_2026-09-11/` (`missing_value_audit/`,
`off_curve_sensitivity/`, `common_row_rescore/`).

- **CL compared two different sets of plants.** Its published uncorrected
  score covered 59 plants and 677 plant-months, and its corrected score 55
  plants and 635. The rows missing from the corrected side are the worst
  uncorrected. So the published gain of 0.018 was mainly a comparison with a
  different denominator on each side, not an overstated improvement. On the
  same rows, the uncorrected RMSE itself falls from 0.123 to 0.110, which is
  more than the gain under either scoring. The harness now scores every
  variant of a run on the rows all of them can score (commit `513f57c`). For
  CL this excludes 49 plant-months, 16.0% of capacity, and six plants wholly:
  the four in cluster 6, and two that `fixed_10` cannot score in most months
  and the unreported `season_10` variant cannot score in the rest. The table
  now shows the re-run on the fixed harness
  (`output/validation/common_row_rerun_2026-09-11/CL/evaluate-2024-rerun/`):
  RMSE 0.110 uncorrected and 0.104 corrected, where 0.123 and 0.105 were
  published; MBE -0.015 and +0.001, where -0.026 and -0.002 were published;
  53 of 59 plants scored. On those 53 plants (35.8 capacity-effective), the
  gain of 0.006 has a 95% interval of -0.007 to 0.021 when plants are
  resampled, and the MAE gain of 0.004 has one of -0.009 to 0.017. The gain
  cannot be distinguished from zero, so the row carries the ‡ marker.
- **An unreported configuration moved AR's reported figures.** The common rows
  are shared by all variants of a run. AR's reported `fixed_10` row moved
  because its unreported `season_10` variant lacks two rows. The table now
  shows the re-run (`output/validation/common_row_rerun_2026-09-11/AR/evaluate-2024-rerun/`).
  Uncorrected RMSE went from 0.151 to 0.150, uncorrected MBE from +0.010 to
  +0.011, and correlation from 0.44 to 0.43. Corrected RMSE and MBE are
  unchanged at displayed precision. AR's resampled gain interval on the common
  rows is 0.002 to 0.030, against 0.002 to 0.031 in the notice above. The
  re-run does not reach AR's 24 partly missing plant-months, which are still
  scored on the days that remain.
- **The US** loses 11 of 6,078 plant-months to common-row scoring, and nothing
  moves at displayed precision. Every other row, all eight country-level rows
  included, reproduces its `metrics.csv` exactly.
- **The `min_cluster_size` comparisons under the degenerate-fit table** used
  different rows too. The pre-registered gates in `method-scalar-bounds.md`
  still pass on common rows, with smaller margins; that document carries the
  figures.
- **Below-zero days also occur in clusters that `fit_quality` calls clean**, in
  the US, BR, DE, UK, NZ, AU-NEM and FR. `fit_quality` bounds the scalar and
  checks that each offset converged. It does not check whether a scalar and
  offset pair goes negative over the observed speed range. Counting those days
  as zero moves no corrected RMSE in those rows by more than 0.002. ES and IT
  are a different case, covered by the suspension notice below.

The CL interval has the limits stated in the UK and NZ notice above, and the
choice of `k=10` on the test year also stays outside the resampling.

**Suspension notice, 2026-09-11: the IT, PT and ES rows are suspended.** They
are not simulations of those fleets' winds. The chain, in order:

1. **The ERA5 download never covered them.** The European ERA5 files cover
   12°W to 22°E and 42°N to 72°N. The IT, PT and ES configurations' boxes
   extend south of 42°N, and the download was never extended to match.
2. **The harness extrapolated silently.** `interpolate_wind` interpolates with
   `fill_value=None`, which extrapolates linearly outside the grid without a
   warning. Grid points up to 5° outside the data receive extrapolated winds,
   and uncorrected hub-height speeds reach -57.7 m/s.
3. **The fit fitted factors to that input.** In ES, clusters 0 and 3 lie
   wholly outside the data. They carry offsets of -5.64 and -4.46 m/s, at
   scalars near 0.5.
4. **Those factors push most days off the curve.** In the ES training years,
   clusters 0 and 3 put 54% and 53% of their capacity-weighted days below
   0 m/s. Those days drop out of the fit's objective and out of the score.
   This is the mechanism in the daggered-rows notice above, set off by
   fabricated input rather than by a degenerate fit.
5. **The corrected CF reads too high.** Counting the dropped days as zero
   lowers the mean corrected CF by 0.036 in ES and by 0.036 in IT. ES's
   corrected MBE goes from +0.016 to -0.020, and IT's corrected RMSE from
   0.034 to 0.064. In the ES training years, the fitted national CF matches
   the observed one only with the dropped days removed. Counted as zero, it is
   0.03 to 0.05 below observed in every year. *[Note, 2026-09-25: the offsets
   and corrected figures in items 3 to 5 come from N=4 (ES) and N=3 (IT) fits
   routed to the per-cluster solver, and were not re-measured with the joint
   fit; see the notice of that date. The suspension they explain stands on the
   extent defect in items 1 and 2.]*

| Row | Capacity outside the ERA5 data | Grid points outside | Furthest outside |
|---|---|---|---|
| IT | 94.5% | 19 of 28 | 4.4° |
| PT | 89.9% | 23 of 26 | 4.5° |
| ES | 50.3% | 34 of 52 | 5.0° |

**PT shows why a stable result is not evidence.** 90% of PT's capacity is
extrapolated from a single row of data at 42°N, its northern border. Yet its
figures do not move when off-curve days count as zero, because few of its days
fall off the curve. Nothing in PT's metrics shows that its winds were never in
the input, and a reader comparing it with a sound row cannot tell the two
apart. The check that exposed ES and IT cannot see PT.

The three rows move to a table of suspended rows under the country-level
table, with their published figures kept for the record. They return once ERA5
is downloaded to cover their boxes and they are re-run. NO (4.4% of capacity,
3 of 26 grid points, up to 7.5° east of the data) and SE (0.8%, 1 of 40, up to
1.0°) stay in the table with those shares stated, and get the same download.
Script for the training-year figures: `scripts/analysis/training_objective_check.py`.
Data: `output/curve_library_study_2026-09-11/training_objective_check/`.

**Resolved, 2026-09-13.** The download was made and the three rows were
re-run: `era5/EU_2026-09` covers 12 W to 31.5 E and 36 to 72 N, and every unit
of all three fleets is now inside the extent its configuration loads. Their
extrapolated shares are zero. They are back in the country-level table above,
and none of them carries a dagger: no implausible scalar and no failed offset
between them. Their figures improved, Italy's corrected RMSE halving and
Portugal's falling by two thirds, because the winds are real rather than
extrapolated; the evidence that it is the winds and not something else is in
`method-eu-rerun.md`, which splits each fleet by whether a grid point lay
outside the old extent. *[Note, 2026-09-25, third: Italy and Portugal are
suspended again, because on their grids' own curves most of their fits are
refused; see the third notice of that date.]* *[Note, 2026-09-25: Italy's halving compares N=3
season fits made with the per-cluster solver on both sides; with the joint fit
its reported row is 0.0155. Portugal's row is N=1 and stands. See the notice
of that date.]* This resolves the input defect only. Italy's fit still
sends 1.0% of its capacity-weighted steps off the curve on calm days, which is
an ordinary property of the affine correction rather than a symptom of bad
input, and is a finding in its own right in that document. Sweden and Norway
were re-run in the same pass and are no longer marked either: the sentence
above, that they stay in the table with their shares stated, describes what was
true until this download.

*[Correction, 2026-09-12: NO and SE are not the only rows left with
extrapolated winds. DK is one too, 0.6% of capacity and 47 of 5,446 units, and
now carries § in the turbine-level table above. This paragraph listed the
country-level rows only, and the check of 2026-09-11 behind it compared the
European rows against the extent of the ERA5 files rather than against each
row's own bbox-sliced extent, which is what a run loads. DK's box stops inside
the files. The commit message of c480f46, which added the extent guard, says
"the other twelve rows are inside their grids" on the same mistaken basis; it
is eleven, and DK is not one of them. A commit message is dated history, so it
is corrected here and not rewritten.

`scripts/analysis/extent_audit.py` now asks the second question for all
seventeen rows, so the count has a script behind it. Six rows carry
extrapolated winds: ES 50.3%, IT 94.5%, PT 89.9%, NO 4.4%, SE 0.8% and DK 0.6%.
The five published shares reproduce exactly, including the distances out, and
DK is the addition. The other eleven rows have no unit outside the extent their
own configuration loads. Data:
`output/extent_audit_2026-09-12/`.]*

**Correction notice, 2026-09-12: the rows do not share one roughness
treatment.** The difference between regions is not the formula, it is whether
the result of the formula varies in time. Every region derives the surface
roughness z0 by inverting the log wind profile from the 10 m and 100 m winds,
the equation the method describes. The European files carry a single annual
mean of that quantity, computed once per year in
`src/pyvwf/datasets/combine_era5_files.py`. Every other region derives z0 hour by
hour and averages it to daily along with the winds.

| Roughness applied | Route | Rows |
|---|---|---|
| Per timestep, averaged to daily | derived at load from the file's hourly winds | DE, DK, UK, the eight country-level rows, AU-NEM, NZ, CL and AR: 15 of 17 |
| Per timestep, averaged to daily | derived and stored as a daily field by `scripts/era5/combine.py` | US, BR: 2 of 17 |

*[Updated 2026-09-13: the eleven European rows were re-run on the per-timestep
treatment (`method-eu-rerun.md`), so no scorecard row applies an annual-mean
roughness any more and the split this notice recorded is closed. The published
rows that did are in the Superseded section below. The rows still differ by
route, and a manifest still cannot tell the second route from a stored annual
mean.]*

The Roughness column of the tables below carries these two values per row.
They are the same treatment computed at different stages. A run's manifest
cannot tell the second from a stored annual mean: it reports `stored` whenever
the file carries a roughness field, whatever that field is. The second route
also drops the 10 m winds, so the US and Brazilian rows cannot derive a
roughness at all (`docs/design/roughness-temporal-treatment.md`).

**Cross-region comparison is confounded.** This table invites reading rows
against each other, and the two halves differ in an input, not only in fleet,
observations and climate. Every statement in this repository that ranks or
contrasts regions inherits that, including the transfer and physics-informed
work. A reader cannot work this out from the rows.

**The per-timestep derivation was adopted as the method on 2026-09-12**
(`method-roughness-treatment.md`), on method fidelity and comparability rather
than on accuracy: the measured effect on Denmark is 0.0002 in corrected RMSE,
resolved by the pre-registered gate and far too small to carry a method change
on its own. The eleven European rows were re-run on it on 2026-09-13, and the
figures in the tables below are those re-runs. The treatment moved every one of
them by less than 0.0002 in corrected RMSE, and only Denmark's difference
excludes zero *[Note, 2026-09-25: for the country rows with N greater than 1
that comparison used the per-cluster solver and was not re-measured with the
joint fit; Denmark is turbine-level and stands]*; what moved the returning rows was the wider box, not the
treatment (`method-eu-rerun.md`).

**What the dating evidence supports.** No PyVWF run output surviving in this
repository predates the combined European files of 11 February 2026; the
earliest surviving output is 13 February 2026. Run directories are pruned, so
this does not establish that no earlier run existed, only that none survives to
be checked.

**The thesis is silent, not wrong.** Its method section gives the equation and
states that the ERA5 input is hourly. It does not state how z0 is treated in
time, and a reader would naturally take the equation as applying per timestep.
Nothing in the thesis changes. Any paper drawn from those chapters has to state
the temporal treatment that actually ran.

**Two documentation errors, corrected on 2026-09-12.** They are not the
finding. [`src/pyvwf/datasets/COMBINED_ERA5_USAGE.md`](https://github.com/ellyess/PyVWF/blob/ecf0cc3172a875b27ef4a2fe2d9c2e2a25a42621/src/pyvwf/datasets/COMBINED_ERA5_USAGE.md) said the European z0 is
derived from terrain data; it is not, and the code has no terrain option. The
usage line in `combine_era5_files.py` offered `--roughness-source terrain`,
which does not exist.

**Every row comes from one refresh.** All seventeen were re-run, training and
evaluation, on 2026-09-20 at commit `967b485` from a clean tree, one step per
process, into `output/validation/refresh_2026-09-20/<CODE>/`, outside the
repository. Every manifest records that commit with `git_dirty: false`. Before
this the rows came from three different runs at three commits: the 2026-08-24
refresh at `41462e9` (AU-NEM and NZ), the European re-run at `b5d47d0` (DE, DK,
UK and the eight country rows) and the accepted-years re-runs at `0fd6574`
(US, CL, AR and BR). *[Note, 2026-09-24: the corrected figures of AU-NEM and NZ
now come from a re-evaluation of the same training runs at `a94670c`, in
`output/c1_turbine_order_2026-09-24/after/`; their uncorrected figures are
unchanged. See the notice of that date.]*

**The scored set changed where refused clusters dropped plants.** A cluster
whose accepted years are not a majority of its training years is refused, its
units get no corrected values, and the common-row rule then drops those
units from every variant, the uncorrected one included. So the uncorrected
figures of DE, UK and AU-NEM move as well as the corrected ones, and the
count each row scores is now given beside its fleet. The figures the dated
notices above quote are superseded rather than re-measured, the resampled
gain intervals among them. Each row
was run from the single-configuration file committed under
`configs/regions/scorecard/` (`<code>_k<N>.toml` or `<code>_country.toml`), which
fixes the cluster count and time slice the row reports. For the eight
country-level regions that file is identical to the maintained
`configs/regions/<code>.toml`. For the nine turbine-level regions it is not: the
maintained configs carry a cluster sweep, and for seven of the nine (all but DK
and UK) that sweep does not contain the reported cluster count, so re-running the
maintained config does not reproduce the row.

Each evaluate manifest's `trained_from` records a temporary path that no longer
exists. The link to `train-refresh/` was verified on 2026-09-11: re-running every
evaluation at the same commit, against the surviving training directories,
reproduced all seventeen `metrics.csv` files byte for byte.

Seven rows (DE, DK, UK, US, BR, AU-NEM, NZ) were run with a licensed curve
library that is not redistributable, identified by sha256 in each manifest.
Four of them (DE, DK, UK, US) simulate 74% to 90% of their capacity on curves
only that library contains, and are not reproducible by a third party. The
other three (BR, AU-NEM, NZ) simulate only on curves the open library also
contains: re-run on the open library on 2026-09-11, each reproduced its
`metrics.csv` byte for byte (`output/validation/open_library_check_2026-09-11/`).
That check was made against the runs those rows reported then; the curves each
unit resolves have not changed since, and each row's `curve_resolution.csv`
records them for the 2026-09-20 runs, but the byte comparison has not been
repeated.
The remaining ten (CL, AR and the eight country-level regions) were run on the
bundled open library. The country-level grid points name Vestas models that the
open library does not contain, so every unit in those eight rows fell back to a
single default curve, the open library's first column:
`2019COE_DW100_100kW_27.6`, a 100 kW distributed-wind turbine. *[Note,
2026-09-25: the country-level rows now run on the licensed library, where each
grid's own Vestas curve resolves, and are not third-party reproducible; CL and
AR remain the only rows on the open library. See the third notice of that
date.]*

**Other brand** is the share of each row's fitted training fleet, by capacity,
simulated on another manufacturer's curve. **Reference curve** is the share on a
research reference design or generic composite curve, which is never the unit's
own machine; on the open library it is often the only kind available, so read
it as a fact about the library rather than about the matching. **Unverifiable**
is the share where either side cannot be identified, so the match cannot be
checked in either direction. Unit counts and the largest mismatched pairs are in
`output/validation/curve_resolution_backfill_2026-09-11/cross_manufacturer_audit.csv`,
produced by `scripts/analysis/curve_match_audit.py`.

Every turbine-level row except BR is matched on specific power, by one of four
routes:

- **DK and UK:** `add_models` at load time, a fuzzy manufacturer match then
  nearest specific power. Both registers also record a model designation, which
  `add_models` does not read.
- **DE:** the same `add_models` route, but its register carries no model
  designation, so the fuzzy manufacturer tier is all it can use. Grouping DE
  with DK and UK, as this list did until 2026-09-13, hid that difference.
- **US, NZ, CL and AR:** `assign_curves_from_library` at processing time,
  nearest specific power within a rating band, with no manufacturer step.
- **AU-NEM:** a specific-power class.

BR assigns one uniform curve to every complex, and nothing records its
manufacturers, so its match shares are n/a rather than zero.

The two mismatch columns therefore measure how far each row rests on the
assumption that specific power fixes a curve's shape. Whether held-out skill
survives that assumption is not assessed here. Largest case: 13.4% of DE
capacity is Vestas turbines on Gamesa curves.

*[Note, 2026-09-13: for Germany the columns measure something else. The German
register records a manufacturer, a rating, a rotor diameter and a date, and no
model designation at all, so no assignment better than the nearest specific
power is available from it: a brand-and-spec matcher places **0.0% of DE
capacity, 0 of 11,433 units** against the licensed library
(`scripts/analysis/curve_library_match.py`, coverage in
`method-curve-library-prereg.md`). The 13.4% above is therefore a limit of the
register rather than of the matching, and the two argue for different things:
one for a better register, the other for changing the method. The figure
stands; what it is evidence of does not.]*

The rule is strict and names brands, not lineages. A Bonus turbine on a
Siemens curve counts as other brand. A GE plant on the DOE reference curve of a
GE 1.5 MW machine counts as a reference curve; that is 6.6% of US capacity.

**Correction notice, 2026-09-13: Denmark's three audit shares were computed on
truncated manufacturers.** `load_turbine_metadata` truncates the Danish
manufacturer to its first word, so "NEG Micon" becomes "NEG", and the audit's
stop-word list contains "neg", correctly, as a word that never identifies a
brand on its own. Together they classify every NEG Micon unit as unverifiable:
994 units in the register, 1,034 in the fitted fleet, 19.9% of its capacity.
Read with the full manufacturer string the row reads:

| | Published | Corrected |
|---|---|---|
| Other brand | 11.6% | **15.0%** |
| Reference curve | 0.6% | **1.7%** |
| Unverifiable | 23.0% | **3.1%** |

The table above now carries the corrected figures. **The claim those numbers
supported was wrong in kind, not only in size:** Denmark's manufacturers are
recorded, and the pipeline discards part of each one. No other row is affected,
because no other loader branch truncates. The truncation itself is not changed
here: it is an input to a published row, and changing it is separate work.

**`add_models` reads a manufacturer, a capacity, a rotor diameter and a hub
height, and no model designation, for any region.** So the designations the
Danish and British registers do record are unused by curve assignment as it
stands. Whether reading them would assign better curves is the T1 condition of
the curve library study (`method-curve-library-prereg.md`), and it is untested
here.

Every model key in every turbine-level row has a curve in its
`power_curves.csv`, so nothing there was substituted. That is a different table
from `models.csv`, which the audit reads for each model's manufacturer: 11 AU-NEM
farms (14.3% of capacity) carry keys with a curve in `power_curves.csv` and no
row in `models.csv`, because the open library's normalized composites are
curve-only by design; the audit identifies them from the open library's
provenance file as reference curves. The country-level table's **Substituted**
column is the share of the evaluated fleet whose model key has no curve in
`power_curves.csv` at all. Per-row records are in each row's
`curve_resolution.csv` under the same folder.

## Turbine / plant-level (observed capacity factor per farm)

Matched real turbine curves and hub heights; k-swept affine fit; best held-out
`affine-wind` row, fleet scope.

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

**‡ Gain not distinguishable from zero when the test year's units are resampled;
see the correction notices above.**

**§ Part of the fleet lies outside the loaded ERA5 extent, and its winds were
extrapolated; the share of capacity follows the marker** (the rule is in
`docs/README.md`). DK's box stops at 13.5°E and Bornholm lies near 14.9°E, so
47 of the 5,446 units in its test fleet, 0.6% of capacity, sit up to 1.64°
beyond the data, as do 15 of the 3,707 in the fleet the row was trained on,
0.5% of that capacity and up to 1.55°. The row's own run records the share in
its `metrics.csv`, as the rule requires; the figure was first measured by the
extent audit of 2026-09-12 (`scripts/analysis/extent_audit.py`, data in
`output/extent_audit_2026-09-12/`), because the run published then predated
`extrapolated_capacity_share`. The marker says the figures were produced partly
from winds that were extrapolated rather than interpolated, and it survived the
re-run because the extended download does not reach Bornholm: DK's own box
stops first. The data covering Bornholm is already in the European files, so
this is a bounding-box error and not a missing download; widening the box is
logged as separate work, and produces a different DK row with its own
configuration.

**† The fit behind this row is degenerate.** `fit_quality` run against the exact
factors file each row reports, with the calibrated bounds (scalar in 0.2 to 3.0,
offsets required to converge):

| Region | Config | Max scalar | Implausible scalars | Failed offsets |
|---|---|---|---|---|
| Chile (CL) | k10 fixed | **3.25** | 1 | **2** |
| Brazil (BR) | k60 fixed | 2.87 | 0 | **2** |
| United States (US) | k250 fixed | 2.72 | 0 | **5** |
| Australia (AU-NEM) | k45 season | 2.64 | 0 | **4** |
| United Kingdom (UK) | k50 fixed | 1.86 | 0 | **2** |
| Germany (DE) | k100 fixed | 1.35 | 0 | **1** |
| Argentina (AR) | k10 fixed | 1.29 | 0 | **1** |

The other two are clean: NZ 1.81 and DK 1.14, inside the ceiling with no
failed offsets, as is every country-level fit below. *[Note, 2026-09-25: on
their grids' own curves two country-level fits are degenerate, France's and
Norway's; they are daggered, with their fit quality, under the country-level
table.]* DE, UK and AU-NEM joined
the list in the 2026-09-20 refresh: each refuses a cluster whose accepted
years are not a majority of its training years, and a refused factor counts as
a failed offset. Every maximum scalar in the table is now inside the
plausible range except Chile's.

Since 2026-09-19 (#28) a factor averages its scalar and offset over its
accepted years, and a factor whose accepted years are not a majority of its
training years is refused and carries no scalar, so every maximum scalar in
the table covers applied factors only. The scalars four of these rows showed
before belonged to clusters now refused: the US's 46.39 (cluster 38), CL's
80.23 (cluster 6), AR's 15.53 (cluster 7) and BR's 4.82 (cluster 27), and
DE's 2.79 (cluster 70). Six rows carry their daggers through failed offsets
alone, with maximum scalars inside the bounds; CL keeps its through two
refused clusters and an applied scalar of 3.25 (cluster 2).
These figures travel in `metrics.csv` automatically, so a future run cannot hide
them.

Chile is the worst and is documented in `method-scalar-bounds.md`, which
shows no `min_cluster_size` setting satisfies all three of its gates: raising it
to 3 clears the failed offset and still beats uncorrected (0.1069 against
0.1226) but leaves a scalar of 39.3, and raising it to 5 collapses the
correction entirely (0.2366 against an uncorrected 0.1226). The same document
reports the exported Chile field flagging 717 of 2,697 grid cells, 26.6%, as
degenerate. Two of its ten clusters are now refused (6 and 8, whose per-year
scalars reached 80 and 29), so their plants carry no corrected values, and a
third applies a scalar of 3.25. A corrected RMSE of 0.104 from that fit is not
a result to quote without this context.

Read this as: across four continents, on fleets the model never saw, the
correction lowers capacity-factor RMSE and drives the systematic bias (MBE)
toward zero in every case. It is strongest where the raw ERA5 bias is a
clusterable wind-speed offset (DE, DK, BR), and weakest where the residual is
not a wind-speed error at all (CL, AR; see caveats).

That reading holds for the five clean rows. For the four daggered ones the
aggregate improved while individual clusters did not: an RMSE that improves
while the fit contains a scalar of 80 is telling you the fleet average absorbed
the damage, not that the correction is sound everywhere it was applied. The
per-cluster factors from those four rows should not be used, and the gridded
exports built from them carry a `degenerate` layer for exactly this reason.

## Country-level (ENTSO-E national aggregate, 2023)

Capacity-weighted national monthly CF, held-out 2023. Training years 2015-21
(IE 2017-21). Each grid's own Vestas curve from the licensed library (sha256
`689cfee7…`) since 2026-09-25; until then the bundled open library, on its
100 kW fallback curve. *[Note, 2026-09-25, third: every figure below is from
the licensed-curve runs, Italy and Portugal are suspended, and the
Substituted column is 0%; see the third notice of that date.]* *[Note, 2026-09-25: the rows with N greater than 1 now carry the
joint-fit figures; see the notice of that date. Norway's corrected RMSE, 0.0351,
is still above its uncorrected 0.0350.]* *[Note, 2026-09-25, later: every row now
carries the figures after the joint fit's tolerance fix (`26e0b67`); see the
second notice of that date. Norway's corrected RMSE is 0.0357.]* *[Note, 2026-09-29: Ireland's row carries the
figures after the fleet-register correction, 0.024 to 0.021 corrected; France
and Italy move below this table's precision. See the notice of that date.]*
*[Note, 2026-10-16: every row carries the figures with offshore farms on their
own grid points. Belgium moves, 0.123 to 0.161 uncorrected and 0.021 to 0.019
corrected; France 0.027 to 0.026 corrected, still worse than uncorrected, and
its uncorrected MBE from -0.002 to -0.004; Norway 0.028 to 0.026 on the same
degenerate fit. See the notice of that date.]*
*[Note, 2026-10-22: Belgium's row is rerun at `6c5272d` on Elia's complete
observations (`ods031`, onshore and offshore), now in `input/combined/`; the
other rows still read ENTSO-E. Belgium moves 0.161 to 0.081 uncorrected and
0.019 to 0.018 corrected, its corrected MBE +0.003 to +0.001. Both N=1
variants get worse (fixed 0.020 to 0.021, season 0.021 to 0.025). Its 2023
grid puts 58% of its weight on offshore against Elia's 43%. See the notice of
that date.]*
*[Note, 2026-10-23: France's row is rerun at `65efa06` on RTE's complete
observations (total wind over RTE's interpolated fleet), now in
`input/combined/`. France moves 0.020 to 0.019 uncorrected and 0.026 to 0.022
corrected, still worse than uncorrected, its uncorrected MBE -0.004 to -0.012,
on a fit as degenerate as before (scalar 4.71). See the notice of that date.]*
*[Note, 2026-10-25: France's row is rerun at `9a5682c` on ODRE's openly
licensed copy of the same RTE series, with the year-local roughness fill; no
figure moves at the precision shown. See the bracket in the notice of
2026-10-23.]*

*[Note, 2026-10-25: Sweden's row is rerun at `768a278` with its generation
divided by Energimyndigheten's fleet instead of a denominator built from its
own generation, on the year-local roughness fill. Sweden moves 0.098 to 0.092
uncorrected and 0.035 to 0.027 corrected, its corrected MBE -0.031 to +0.015.
See the notice of that date.]*
*[Note, 2026-10-25: Norway's row is rerun at `b9e812b` on NVE's metered farm
output over its turbine register for 2015-21, and ENTSO-E's 2023 generation
over that register, on the year-local roughness fill. Norway moves 0.128 to
0.131 uncorrected and stays at 0.026 corrected, its corrected MBE -0.014 to
-0.012, on a fit still degenerate (scalar 3.18). See the second notice of that
date.]*

| Region | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Best cfg | Roughness | Substituted |
|---|---|---|---|---|---|---|---|
| France (FR) | 0.019 | 0.022 | -0.012 | +0.009 | N=10 fixed † | per timestep | 0% |
| Belgium (BE) | 0.081 | **0.018** | +0.079 | +0.001 | N=3 season | per timestep | 0% |
| Ireland (IE) | 0.045 | **0.021** | +0.036 | +0.006 | N=1 season | per timestep | 0% |
| Sweden (SE) | 0.092 | **0.027** | -0.089 | +0.015 | N=4 fixed | per timestep | 0% |
| Norway (NO) | 0.131 | **0.026** | -0.127 | -0.012 | N=4 fixed † | per timestep | 0% |
| Spain (ES) | 0.071 | **0.030** | -0.070 | +0.010 | N=4 fixed | per timestep | 0% |

France's corrected RMSE, 0.022, is above its uncorrected 0.019: the
correction adds error. *[Note, 2026-10-23: until then this read "nearly
unbiased (MBE -0.004)"; on RTE's series the uncorrected MBE is -0.012.]*

**† The fit behind this row is degenerate** (bounds as in the turbine-level
table):

| Region | Config | Max scalar | Implausible scalars | Failed offsets |
|---|---|---|---|---|
| France (FR) | N=10 fixed | **4.71** | 1 | 0 |
| Norway (NO) | N=4 fixed | **3.18** | 1 | 0 |

**Suspended rows, 2026-09-25.** On their grids' own curves most of their
joint fits are refused, so Italy scores no month and Portugal 3 of 12 (third
notice of that date). The figures they carried until then, on the 100 kW
fallback curve:

| Region | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Best cfg | Was |
|---|---|---|---|---|---|---|
| Italy (IT) | 0.070 | 0.015 | -0.069 | -0.004 | N=3 season | in the table, 100% substituted |
| Portugal (PT) | 0.089 | 0.027 | -0.085 | +0.018 | N=1 season | in the table, 100% substituted |

*[Note, 2026-10-24: Italy's and Portugal's 2023 test files left out offshore
output, the Taranto farm's 0.053 TWh (0.2%) and WindFloat Atlantic's 0.079 TWh (0.6%),
because they were built with the generator's `--psr-type onshore` default; the
same call reproduces them byte for byte. They are rebuilt with onshore and
offshore (`psr_type="all"`), equal to ENTSO-E's total in every hour, 0.986 and
0.983 of Eurostat against 0.984 and 0.977 before; the old files are kept as
`.entsoe-onshore.bak.csv`. Their training files were already complete. Both rows
were rerun at `7fc0e16` from a clean tree (training years 2015-21, test year
2023, licensed library, 0% substituted); the current code on the old files
reproduces the 2026-10-16 `metrics.csv` byte for byte, and the trained factors
are unchanged. Italy still scores no month. Portugal, on the same 3 of 12
months:*

| Variant | RMSE before | RMSE after | MBE before | MBE after | Max scalar after | Implausible / failed after |
|---|---|---|---|---|---|---|
| uncorrected | 0.2205 | 0.2206 | -0.2201 | -0.2202 |  |  |
| fixed N=1 | 0.0490 | 0.0493 | -0.0373 | -0.0373 | 3.03 | 1 / 0 |
| fixed N=2 | 0.0133 | 0.0135 | -0.0094 | -0.0095 | 3.92 | 1 / 0 |
| **season N=1** (reported) | 0.0300 | 0.0304 | -0.0073 | -0.0073 | 3.12 | 1 / 1 |
| season N=2 | 0.1021 | 0.1021 | +0.1007 | +0.1007 | 3.60 | 1 / 6 |

*Both rows stay suspended. Data: `output/it_pt_2023_2026-10-24/`.]*

**No country-level row carries § any more.** Sweden and Norway did, at 0.8%
and 4.4% of capacity, and the wider download of 2026-09-12 covers both fleets;
their re-runs record an extrapolated share of zero. Denmark is the only
scorecard row still marked, because its own bounding box, not the data, stops
short of Bornholm. The marker rule is in `docs/README.md`.

**Superseded rows, 2026-09-13.** The published figures of every European row,
kept for the record and replaced in the tables above. All eleven ran on
`era5/EU`, the annual-mean roughness, and the narrower ERA5 box; their
configurations are in `configs/regions/scorecard/superseded/2026-09-13/`, under
their own names and unchanged. The canonical name in
`configs/regions/scorecard/` points at the row standing today.
What moved, and why, is in `method-eu-rerun.md`.

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

A superseded row is not withdrawn. It states what was published, on what
input, and the date it was replaced. The three that were suspended were never
results at all, and their figures are kept only so the correction can be
checked. *[Note, 2026-10-22: Belgium's superseded row, like every Belgian
figure before that date, read observations that left out offshore generation
in 2015-19 and 2023; see the notice of that date.]*

The country-level fit removes very large mean biases (FR, BE and IE all from
0.17-0.34 down to ~0.01-0.02). Two honest notes: NO is already close to
unbiased uncorrected (RMSE 0.035) and the correction does not help (0.036, with
the interval on the difference including zero); and the country method
fits under-determined offsets against one national series per month, so the
offsets largely repair the scalar's cube-law overshoot rather than a genuine
additive spatial bias (`method-country-level.md`). *[Note, 2026-09-25: with the
joint fit, NO is 0.0351 against 0.0350 uncorrected, still no help; its interval
was not recomputed. The "largely repair the scalar" reading rests on offsets
from the per-cluster solver, and may partly be its artefact: matching every
cluster to one national value ties each offset to its own scalar. It is not
re-established for the joint fit. See the notice of that date.]* *[Note,
2026-09-25, later: after the joint fit's tolerance fix NO is 0.0357 against
0.0350; see the second notice of that date.]* *[Note, 2026-09-25, third: on the
grids' own curves the mean biases this paragraph names were largely the
fallback curve's. Uncorrected RMSE is 0.020 for FR, 0.123 for BE and 0.045 for
IE, and NO's is 0.129, which the correction lowers to 0.028 with a degenerate
fit. See the third notice of that date.]* *[Note, 2026-10-22: Belgium's
uncorrected RMSE is 0.081 on Elia's complete observations. On the same grids,
replacing the old files halves its uncorrected MBE, from +0.156 to +0.079, so
part of every Belgian mean bias this paragraph names came from observations
that left out offshore generation. See the notice of that date.]*

## What must NOT be overclaimed

- **Seven of the nine turbine-level rows rest on degenerate fits** (CL, US,
  AR, BR, and since the 2026-09-20 refresh DE, UK and AU-NEM). The aggregate
  metric is real; the underlying per-cluster factors are not all usable. Chile
  refuses two of its ten clusters and applies a scalar of 3.25 in a third; the
  United States refuses five of 250, Brazil two of 60, Australia one of 45,
  the United Kingdom two of 50, Germany one of 100 and Argentina one of 10.
  None of that is visible in the skill metric, which is the point. Only DK and
  NZ are clean.
- **An aggregate that barely moves does not mean the fit did not move.** A
  change of fitting code moved the United States' headline RMSE by 0.0005 while
  more than doubling its worst fitted scalar, from 20.54 to 46.39. Check
  `fit_quality`, not the metric, when judging whether a correction is sound.
- **One test year per region.** Orderings of two close configs are not
  meaningful; the headline is the uncorrected-to-corrected drop, not the exact k.
- **CL and AR are caveated, not headline wins.** ERA5 exaggerates the
  north-south wind gradient in both (Atacama too calm, Patagonia too windy). The
  correction removes the mean bias but adds limited skill; for CL it only helps
  once real turbine curves are used, and CL is unstable at low k. The k=10 row
  reported above is itself degenerate, so "use k=1 or k>=8" is not sufficient
  guidance: check `fit_quality` rather than the cluster count. AR is usable
  only after its capacity denominators were rebuilt; a northern cluster fitted
  an extreme scalar that a higher-resolution wind product, not more data, would
  fix, and since 2026-09-19 its factor is refused (`region-south-america.md`).
- **NO gets worse; NL is excluded** (an ENTSO-E coverage defect makes its CF
  series unusable). Reporting either as a corrected region would be false.
  *[Note, 2026-09-25: on the grids' own curves Norway's correction lowers RMSE
  from 0.129 to 0.028 on a degenerate fit (scalar 3.05), and France's raises it
  from 0.020 to 0.027. Italy and Portugal are suspended. See the third notice
  of that date.]*
- **Sweden's training capacity for 2015-19 is built from its own generation.**
  *[Added 2026-10-24. 2026-10-25: corrected; the row is rerun on
  Energimyndigheten's fleet, see the notice of that date.]* The training file
  carries 8,354 MW in every one of those five years: the sum of four zonal
  capacities that the fetcher of the time derived as peak generation over 0.9
  when ENTSO-E returned none
  (`method-country-level.md`). ENTSO-E's national register gives 3,500 to
  7,506 MW over the same years and the GWPT grid 4,048 to 6,082 MW
  (`output/entsoe_audit_2026-10-23/classification.csv`). 8,354 MW is above
  both in every one of those years, by 11% (2019) to 139% (2015) against
  ENTSO-E's, so those years' capacity factors are understated if either
  register is right; neither is established as the fleet. The row's generation
  matches ENTSO-E hour by hour, and the capacity of 2020-21 and the 2023 test
  year matches ENTSO-E's register within 0.5%. Its figures are not corrected
  here.
- **The country-level rows are licensed-lineage.** Since 2026-09-25 they
  simulate each grid's own Vestas curve, which only the licensed library holds,
  so a third party cannot reproduce them. Each grid carries one representative
  turbine per country, not the fleet's machines.
- **US carries an unscreened curtailment confound** (ERCOT/SPP); its near-zero
  fleet MBE is partly an aggregation artefact.
- **AU-NEM's "does correction help?" is config-dependent** (it improves the
  fleet seasonal cycle but can worsen absolute farm RMSE in curtailed South
  Australia); the table row is the matched-curve k-swept result. *[Note,
  2026-09-24: the figures behind this caveat, in `region-au-nem.md`, are
  withdrawn by that document's notice of this date.]*
- Screening-level throughout: not MEASNET/DNV-accredited, not investment advice.

## Data provenance

Turbine/plant observations by region: DK Danish Energy Agency; DE public
turbine register; UK REPD/Ofgem; US EIA-923; BR ONS; AU-NEM AEMO; NZ EMI; CL
Coordinador (CEN); AR CAMMESA (capacities rebuilt from turbine specs).
Country-level: ENTSO-E Transparency. All correction operates on ERA5 at 0.25deg.
Confidential inputs (WindStats DE/ES, Ofgem certificate warehouse, licensed
curve library) are not redistributed and are not required to reproduce the open
rows above, which run on the bundled open curve library. *[Note, 2026-09-25: the
country-level rows now need the licensed curve library; CL and AR are the open
rows.]* *[Note, 2026-10-22: Belgium's country-level observations come from Elia
Open Data `ods031`, not ENTSO-E.]* *[Note, 2026-10-23: France's come from
RTE's eCO2mix definitive series and its annual wind fleet, not ENTSO-E.]*
*[Note, 2026-10-25: France's are read from ODRE, RTE's open-data portal, under
Licence Ouverte v2.0.]*
*[Note, 2026-10-25: Sweden's generation is ENTSO-E's and its capacity-factor
denominator Energimyndigheten's wind fleet.]* *[Note, 2026-10-25: Norway's
training years come from NVE's metered farm output and its capacity from NVE's
turbine register; its 2023 generation is ENTSO-E's.]*
