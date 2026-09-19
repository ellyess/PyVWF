# Country-level regions (ENTSO-E)

**Data source:** the ENTSO-E Transparency Platform: national wind generation
and installed capacity, and per bidding zone for Norway and Sweden.
**Adapters:** `entsoe-country`, and `entsoe-zonal` for Sweden's per-zone
region (SE-BZ) · country-level · unit = country · open.
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

1. Generate the grid points and the ENTSO-E observations. The fetch needs the
   `data` extra. Pass your key in the environment for this one command:

   ```bash
   ENTSOE_API_KEY=<key> python -m vwf.datasets.generate_country_level_training_data \
       --countries BE ES FR IE IT NO PT SE
   ```

   The files go to `<input root>/observations/country/`, in the layout
   [`data-sources.md`](../guides/data-sources.md#4-country-level-entso-e-layout)
   describes. `--help` lists the year options.
2. Replace the synthetic capacity with GWPT capacity, one grid per year:

   ```bash
   python scripts/region_tools/weight_country_grid_points.py --all --per-year 2015 2024
   ```

   Add `--zone-aware` for Norway and Sweden. Capacity is then never summed
   across a bidding-zone boundary.
3. Audit the observed series before any fit:

   ```bash
   python scripts/analysis/audit_country_observations.py
   ```

   The affine correction absorbs a constant observation error, so a series that
   is wrong everywhere still fits well in sample.

## 2. ERA5

The scorecard rows read one European box, fetched once for all of them, into
`<input root>/era5/EU_2026-09/`:

```bash
python scripts/fetch/era5.py --code eu --file-tag EU_2026-09 \
    --bbox -12 31.5 36 72 --years 2015 2016 2017 2018 2019 2020 2021 2022 2023
```

## 3. Train and evaluate

The country-level rows run on the default input root, `input/`. For each
country, with `<cc>` its lower-case code:

```bash
python scripts/analysis/validate_region.py train \
    --region configs/regions/scorecard/<cc>_country.toml
python scripts/analysis/validate_region.py evaluate \
    --region configs/regions/scorecard/<cc>_country.toml \
    --train-run output/validation/<CC>/train-<stamp>
```

Set `cluster_list` to `1` or to the grid's own cluster count. Any other value
raises.

**Read the substituted share before any result.** The grid points name
Vestas models that the open library does not contain, so every unit is
simulated on the fallback curve. `curve_resolution.csv` records it, and the
scorecard's Substituted column reports it.

## Licence

ENTSO-E data are public by regulation (the
[source table](../guides/data-sources.md#source-urls)). Nothing from them is
committed. The capacity weights come from the Global Wind Power Tracker (Global
Energy Monitor, CC-BY-4.0), which requires attribution. The bidding-zone
polygons committed under `configs/curation/zones/` carry their own sources and
licences in that folder's README. Results computed from these data may be
shared.
