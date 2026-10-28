# Country-level regions (ENTSO-E and national sources)

**Data source:** ENTSO-E Transparency Platform national wind generation and installed capacity, per bidding zone for Norway and Sweden; national sources for BE (Elia), FR (RTE, via ODRE), SE (Energimyndigheten), NO (NVE) and PT (REN). **Adapter:** `entsoe-country`, and `entsoe-zonal` for SE-BZ · country-level · unit = country · open, except PT, whose capacity is REN's (mixed; see [Source and licence](#source-and-licence)).

A country-level region fits one national series per period, with grid points
standing in for its fleet. The rows (BE, ES, FR, IE, NO, SE; IT and PT
suspended, NL excluded) are in the [scorecard](../scorecard.md), the estimator
in [The country-level estimator](../results.md#the-country-level-estimator).
Each row reruns from `configs/regions/scorecard/<cc>_country.toml`; the
maintained configs are `configs/regions/<cc>.toml` and `configs/regions/se_bz.toml`
(SE-BZ, Sweden per bidding zone, has no row). New countries:
[A country-level region](../guides/adding-a-region.md#a-country-level-region).

## Source and licence

ENTSO-E, Elia, ODRE, Energimyndigheten, NVE, REN and Eurostat each publish
under their own terms, listed in the
[source table](../guides/data-sources.md#source-urls). REN publishes no reuse
licence for its Data Hub. Nothing from any of them is committed except the
Eurostat annual totals in `configs/curation/eurostat_wind_gwh.csv`; `input/`
is git-ignored. The capacity weights come from the Global Wind Power Tracker
(Global Energy Monitor, CC-BY-4.0), which requires attribution. The
bidding-zone polygons committed under `configs/curation/zones/` carry their
own sources and licences in that folder's README. Results computed from these
data may be shared.

## 1. Observations and grid points

Run the steps in this order. The scorecard rows train on 2015-21 (IE 2017-21)
and test on 2023, reading `<cc>_train_2015_2021.csv` (IE
`ie_train_2017_2021.csv`) and `<cc>_test_2023.csv`, with an `_aggregated`
suffix for NO and SE. The generator writes to `input/observations/country/`
whatever `PYVWF_INPUT` says (`--output-dir` changes it), the other scripts to
`<input root>/observations/country/`; both work because `input/combined`
shares `observations/` with `input/`. What each builder does, and the layout,
are in [`data-sources.md`](../guides/data-sources.md#4-country-level-entso-e-layout).

| Step | What | Reads |
|---|---|---|
| 1 | Generator: grid points of all eight, ENTSO-E observations of ES, IE, IT, NO, PT and SE | ENTSO-E |
| 2 | Ireland's capacity repair | step 1's IE files |
| 3 | Belgium's and France's builders | Elia, ODRE |
| 4 | Sweden's, Norway's and Portugal's builders | step 1's files, and the national source |
| 5 | Capacity weighting of the grid points | step 1's grids, GWPT |
| 6 | Observation audit | everything above |

1. **Generator.** Needs the `data` extra; pass the key in the environment.
   The third command fetches SE-BZ's per-zone 2015-19 files.

   ```bash
   ENTSOE_API_KEY=<key> python -m pyvwf.datasets.generate_country_level_training_data \
       --countries ES IE IT NO PT SE \
       --train-years 2015 2016 2017 2018 2019 2020 2021 --test-year 2023
   python -m pyvwf.datasets.generate_country_level_training_data \
       --countries BE FR --skip-observations
   ENTSOE_API_KEY=<key> python -m pyvwf.datasets.generate_country_level_training_data \
       --countries SE --train-years 2015 2016 2017 2018 2019 --test-year 2023 --skip-grids
   ```

   Without `--train-years` and `--test-year` it writes 2015-18 and 2019,
   which no row reads. **Never run it for BE or FR without
   `--skip-observations`:** it would overwrite their builders' files.

2. **Ireland.** Rewrite ENTSO-E's frozen capacity from the GWPT (`--dry-run` writes nothing):

   ```bash
   python scripts/region_tools/repair_country_capacity.py IE --from-year 2017
   ```

3. **Belgium and France** download their sources, with no key, into
   `<input root>/raw/elia/` and `<input root>/raw/rte/` (`--cache` moves them;
   `--out <dir>` writes elsewhere). Without `--replace` an existing file stops it.

   ```bash
   python scripts/region_tools/build_be_obs_from_elia.py --replace
   python scripts/region_tools/build_fr_obs_from_rte.py --replace
   ```

4. **Sweden, Norway and Portugal.** SE and PT keep step 1's ENTSO-E generation
   and replace its capacity; NO replaces its 2015-21 generation with NVE's
   metered output and keeps ENTSO-E's for 2023. SE and
   PT download into `<input root>/raw/energimyndigheten/` and
   `<input root>/raw/ren/`. NO needs two NVE files downloaded by hand into
   `<input root>/raw/nve/`: the metered output per farm,
   `https://www.nve.no/media/13399/vindprod2002-2021_kraftverk_utcplus1.xlsx`,
   and the turbine register, `https://api.nve.no/web/WindPowerplant/GetWindPowerPlants`,
   looked for as `api_GetWindPowerPlants_2026-09-18.json` (`--register <file>`
   names another). Never use NVE's `timeseries_utc_kwh_2022_all.zip` as
   observations: its output is modelled.

   ```bash
   python scripts/region_tools/build_se_capacity_from_energimyndigheten.py --replace
   python scripts/region_tools/build_no_obs_from_nve.py --replace
   python scripts/region_tools/build_pt_capacity_from_ren.py --replace
   ```

   Steps 2 to 4 keep each old file as `.entsoe-capacity.bak.csv` (IE, SE, PT),
   `.entsoe-mixed.bak.csv` (BE) or `.entsoe.bak.csv` (FR, NO). SE, NO and PT
   then read their ENTSO-E generation from the backup, so move the backups
   aside after a new step 1 fetch.

5. **Capacity weighting.** GWPT capacity, one grid per year, replaces the
   uniform synthetic capacity; `--zone-aware` keeps NO and SE within each zone.
   After regenerating the grids, move aside any `<cc>_grid_points.uniform.bak.csv`
   from an earlier run: it is read in place of the grid.

   ```bash
   python scripts/region_tools/weight_country_grid_points.py BE ES FR IE IT PT --per-year 2015 2024
   python scripts/region_tools/weight_country_grid_points.py NO SE --per-year 2015 2024 --zone-aware
   ```

6. **Audit** the series against Eurostat before any fit, because the affine
   correction absorbs a constant observation error:
   `python scripts/analysis/audit_country_observations.py`.

## 2. ERA5

Every country config reads one European box, `era5/EU_2026-09`, fetched once
([general fetch](../guides/data-sources.md#3-shared-inputs-all-regions)):

```bash
python scripts/fetch/era5.py --code eu --file-tag EU_2026-09 \
    --bbox -12 31.5 36 72 --years 2015 2016 2017 2018 2019 2020 2021 2022 2023
```

## 3. Train and evaluate

Every country-level row runs on `input/combined`, because its grid names a
licensed Vestas curve ([why](../guides/training.md#choose-the-input-root)).
With `<cc>` the lower-case code and `<CC>` upper-case, the scorecard row is:

```bash
PYVWF_INPUT=input/combined pyvwf-validate train \
    --region configs/regions/scorecard/<cc>_country.toml
PYVWF_INPUT=input/combined pyvwf-validate evaluate \
    --region configs/regions/scorecard/<cc>_country.toml \
    --train-run output/validation/<CC>/train-<stamp>
```

Run it through `scripts/dev/run_locked.py`
([Reproducing a row](../scorecard.md#reproducing-a-row)); the maintained
configs run the same way ([training guide](../guides/training.md)). Each
`cluster_list` is `1` and the grid's own cluster count (BE 3, ES 4, FR 10,
IE 3, IT 3, NO 4, PT 2, SE 4, SE-BZ 4); any other value raises.

## Caveats

- **The joint fit is under-determined** (N offsets, one national series a
  month): check `fit_quality` before using a per-cluster factor.
- **Norway's years mix sources:** NVE's metered output for 2015-21, ENTSO-E's
  generation over NVE's register for 2023.
- **Sweden's fleet is slightly overstated:** decommissioned turbines stay
  in Energimyndigheten's register for up to three years.
- **France has three empty hours a year,** on the clock-change days.
- **SE-BZ trains on 2015-19,** not 2015-21; for a like-for-like comparison,
  run `se.toml` with `train_years = [2015, 2019]`.
- **France's correction adds error,** and the France and Norway fits are
  degenerate (†); see the [scorecard](../scorecard.md).
- Every result rests on the single test year 2023.
