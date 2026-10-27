# Country-level regions (ENTSO-E)

**Data sources:** the ENTSO-E Transparency Platform: national wind generation
and installed capacity, and per bidding zone for Norway and Sweden. Belgium's
and France's observations come from their transmission system operators
instead: Elia for BE and RTE for FR.
**Adapters:** `entsoe-country`, and `entsoe-zonal` for Sweden's per-zone
region (SE-BZ) · country-level · unit = country · open, except FR
(see [Licence](#licence)).
**Regions:** BE, ES, FR, IE, IT, NO, PT and SE in the scorecard. NL is
excluded; see [`scorecard.md`](../findings/scorecard.md).
**Configs:** `configs/regions/<cc>.toml`; each scorecard row is
`configs/regions/scorecard/<cc>_country.toml`.

A country-level region fits against one national series per period, and grid
points stand in for its fleet. No raw data or processed input is committed,
because `input/` is git-ignored. This runbook builds them. How to add a
country is in
[A country-level region](../guides/adding-a-region.md#a-country-level-region).

## 1. Observations and grid points

1. Generate the grid points of every region, and the ENTSO-E observations of
   every region but Belgium and France. The fetch needs the `data` extra.
   Pass your key in the environment for this one command:

   ```bash
   ENTSOE_API_KEY=<key> python -m pyvwf.datasets.generate_country_level_training_data \
       --countries ES IE IT NO PT SE
   python -m pyvwf.datasets.generate_country_level_training_data \
       --countries BE FR --skip-observations
   ```

   **Never run the generator for BE or FR without `--skip-observations`.** It
   would overwrite their observations with ENTSO-E's series, which do not
   measure their whole fleet in every year.

   The files go to `<input root>/observations/country/`, in the layout
   [`data-sources.md`](../guides/data-sources.md#4-country-level-entso-e-layout)
   describes, with what the fetch keeps. It fetches every wind type unless
   `--psr-type` says otherwise. `--help` lists the year options.
2. Build Belgium's observations from Elia and France's from RTE. Neither needs
   a key:

   ```bash
   python scripts/region_tools/build_be_obs_from_elia.py --replace
   python scripts/region_tools/build_fr_obs_from_rte.py --replace
   ```

   Each writes the splits the configs read, in the same layout. Without
   `--replace` an existing file stops the build; with it, the old file is kept
   beside the new one as `.entsoe-mixed.bak.csv` (BE) or `.entsoe.bak.csv`
   (FR). `--out <dir>` writes somewhere else.

   Then rebuild Sweden's, Norway's and Portugal's from their national
   sources. Each reads the files step 1 wrote, so run them after it:

   ```bash
   python scripts/region_tools/build_se_capacity_from_energimyndigheten.py --replace
   python scripts/region_tools/build_no_obs_from_nve.py --replace
   python scripts/region_tools/build_pt_capacity_from_ren.py --replace
   ```

   The old files are kept as `.entsoe-capacity.bak.csv` (SE, PT) or
   `.entsoe.bak.csv` (NO). Once a backup exists, a builder reads its
   ENTSO-E generation from the backup, so a second run reproduces the first;
   after a new fetch in step 1, move the backups aside before running them.
3. Replace the synthetic capacity with GWPT capacity, one grid per year:

   ```bash
   python scripts/region_tools/weight_country_grid_points.py --all --per-year 2015 2024
   ```

   Add `--zone-aware` for Norway and Sweden. Capacity is then never summed
   across a bidding-zone boundary.
4. Audit the observed series before any fit:

   ```bash
   python scripts/analysis/audit_country_observations.py
   ```

   The affine correction absorbs a constant observation error, so a series that
   is wrong everywhere still fits well in sample. The audit also compares each
   year's observed energy with Eurostat's national wind production, which
   catches a series that measures only part of the fleet.

## 2. ERA5

The scorecard rows read one European box, fetched once for all of them, into
`<input root>/era5/EU_2026-09/`:

```bash
python scripts/fetch/era5.py --code eu --file-tag EU_2026-09 \
    --bbox -12 31.5 36 72 --years 2015 2016 2017 2018 2019 2020 2021 2022 2023
```

## 3. Train and evaluate

The country-level rows run on the licensed input root, `input/combined`. The
grid points name `Vestas.V80.2000`, `Vestas.V90.2000` or `Vestas.V90.3000`,
which only the licensed library holds, and a country-level run refuses a curve
the loaded library lacks (`CurveSubstitutionError`). For each country, with
`<cc>` its lower-case code:

```bash
PYVWF_INPUT=input/combined python scripts/analysis/validate_region.py train \
    --region configs/regions/scorecard/<cc>_country.toml
PYVWF_INPUT=input/combined python scripts/analysis/validate_region.py evaluate \
    --region configs/regions/scorecard/<cc>_country.toml \
    --train-run output/validation/<CC>/train-<stamp>
```

Set `cluster_list` to `1` or to the grid's own cluster count. Any other value
raises.

**The rows are not reproducible without the licensed library.** Until
2026-09-25 they ran on the default root, where every unit was simulated on the
open library's 100 kW fallback curve; `curve_resolution.csv` recorded it, and
the run went on. Italy and Portugal are suspended on the licensed curves,
because most of their joint fits are refused (`docs/findings/scorecard.md`).

## Licence

ENTSO-E, Elia, ODRE, Energimyndigheten, NVE, REN and Eurostat each publish
under their own terms, listed in the
[source table](../guides/data-sources.md#source-urls). REN publishes no reuse
licence for its Data Hub. Nothing from any of them is committed except the
Eurostat annual totals in `configs/curation/eurostat_wind_gwh.csv`. The
capacity weights come from the
Global Wind Power Tracker (Global Energy Monitor, CC-BY-4.0), which requires
attribution. The bidding-zone polygons committed under `configs/curation/zones/`
carry their own sources and licences in that folder's README. Results computed
from these data may be shared.
