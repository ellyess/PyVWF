# PyVWF Data Pipeline Reference

## 1. Overview

PyVWF operates at two distinct data levels:

- **Country-level:** Aggregated wind generation from ENTSO-E Transparency Platform,
  paired with grid-sampled reanalysis data. Used for national-scale capacity factor
  correction. Countries: NL, FR, BE, NO, SE, ES, IT, PT, IE.

- **Turbine-level:** Individual turbine metadata and monthly generation records.
  Used for site-specific correction factor training. Countries: DE, DK, UK.

Data processing scripts live in `src/vwf/datasets/`, loader functions in `src/vwf/data.py`,
and workflow orchestration scripts in `scripts/`.

## 2. Directory Structure

### input/country_level_data/

ENTSO-E grid points and observed capacity factors for country-level workflows.

```
input/country_level_data/
├── grid_points/
│   ├── nl/
│   │   ├── nl_grid_points.csv
│   │   └── nl_correction_regions.geojson
│   ├── no/
│   │   ├── no_grid_points_zones.csv           # Bidding-zone variant
│   │   └── no_bidding_zones.geojson
│   └── {country}/...                           # FR, BE, SE, ES, IT, PT, IE
├── observations/
│   ├── nl/
│   │   ├── nl_train_2015_2018.csv
│   │   └── nl_test_2019.csv
│   ├── no/
│   │   ├── no_1_train_2015_2018.csv            # Per-zone files
│   │   └── no_train_2015_2018_aggregated.csv   # Country aggregate
│   └── {country}/...
└── pyvwf_config.py                             # Generated configuration
```

Grid point CSVs contain: `ID`, `lat`, `lon`, `height`, `model`, `capacity`, `cluster`.
Bidding-zone countries (NO, SE) add a `zone` column. Observation CSVs are
datetime-indexed with: `capacity_factor`, `generation_mw`, `capacity_mw`.

### input/turbine_level_data/

Individual turbine metadata and generation observations.

```
input/turbine_level_data/
├── DE/
│   ├── DE_md.csv                  # 10,889 turbines
│   ├── DE_data.csv                # Long-format monthly generation
│   └── geolocate.germany.csv     # Postcode-to-coordinate mapping
├── DK/
│   ├── dk_md.csv                  # 6,296 turbines (pre-calculated lat/lon)
│   └── dk_obs_2002_2020.csv       # Long-format monthly generation, 2002-2020
└── UK/
    ├── uk_md.csv                  # 6,618 turbines
    └── ukobs.csv                  # Wide-format monthly generation
```

## 3. Data Loaders

All loader functions are in `src/vwf/data.py`. Three path constants are used internally:

```python
COUNTRY_DIR       = Path("input/country-data")        # Legacy country aggregates
TURBINE_DIR       = Path("input/turbine_level_data")   # Turbine-level data
COUNTRY_LEVEL_DIR = Path("input/country_level_data")   # Country-level grid workflows
```

### Country-level loaders

**`load_country_level_data(country, train_years, test_year)`** -- Convenience
function returning a dict with keys `grid_points`, `train_obs`, `test_obs`,
`country`, `train_years`, and `test_year`. Calls the two helpers below.

```python
from vwf.data import load_country_level_data
data = load_country_level_data('NL', [2015, 2016, 2017, 2018], 2019)
```

**`load_country_level_grid_points(country)`** -- Loads grid sampling points.
Automatically detects zone-based files for NO and SE.

**`load_country_level_observations(country, year_start, year_end, mode)`** --
Loads ENTSO-E capacity factors. `mode` is `'train'` or `'test'`. Raises
`FileNotFoundError` with generation instructions if files are missing.

### Turbine-level loaders

**`load_turbine_metadata(country)`** -- Returns a DataFrame of turbine metadata
(ID, capacity, coordinates, hub height, manufacturer, model). DK coordinates are
pre-converted from UTM; DE uses postcode geolocation via `geolocate.germany.csv`.

```python
from vwf.data import load_turbine_metadata
dk_md = load_turbine_metadata('DK')   # 5,618 turbines after filtering
de_md = load_turbine_metadata('DE')   # 10,889 turbines
uk_md = load_turbine_metadata('UK')   # 6,618 turbines
```

**`load_turbine_observations(country, year_start, year_end)`** -- Returns monthly
generation pivoted to wide format (columns = months 1-12). DK and DE read
long-format CSVs and pivot; UK data is already wide.

```python
from vwf.data import load_turbine_observations
dk_obs = load_turbine_observations('DK', 2002, 2020)
de_obs = load_turbine_observations('DE', 2015, 2018)
uk_obs = load_turbine_observations('UK', 2015, 2019)
```

## 4. Dataset Generation Scripts

All scripts are in the `src/vwf/datasets/` subpackage.

**generate_country_level_training_data.py** -- Creates grid points and fetches
ENTSO-E observations for country-level workflows.

```bash
python src/vwf/datasets/generate_country_level_training_data.py \
    --countries NL FR ES SE \
    --train-years 2015 2016 2017 2018 \
    --test-year 2019
```

**fetch_entsoe_capacity_factors.py** -- Standalone ENTSO-E capacity factor download.

```bash
python src/vwf/datasets/fetch_entsoe_capacity_factors.py \
    --countries NL FR BE NO --year-start 2015 --year-end 2020
```

**process_dk_raw_data.py** -- Converts raw Danish Energy Agency Excel files
(`anlaeg.xlsx`, `maanedsdata_2002_2020.xlsx`) into processed CSVs with automatic
UTM-to-lat/lon conversion via pyproj.

```bash
python src/vwf/datasets/process_dk_raw_data.py                # Full processing
python src/vwf/datasets/process_dk_raw_data.py --metadata-only
python src/vwf/datasets/process_dk_raw_data.py --observations-only
```

## 5. Workflow Scripts

Orchestration scripts in `scripts/` drive end-to-end runs.

### scripts/run_all_country_level.py

Trains and evaluates PyVWF across country-level configurations. Reads data from
`input/country_level_data/` via `PyVWF.from_config()`. Requires the generation
script to have been run first.

```bash
python src/vwf/datasets/generate_country_level_training_data.py \
    --countries NL FR BE NO --train-years 2015 2016 2017 2018 --test-year 2019

python scripts/run_all_country_level.py --countries NL FR BE NO
python scripts/run_all_country_level.py --countries NL --dry-run
```

### scripts/run_all_turbine_level.py

Trains PyVWF across turbine-level configurations (DE, DK, UK; onshore/offshore).
Uses `PyVWF()` with `obs_level="turbine"`. No generation step is needed -- the
processed CSVs in `input/turbine_level_data/` are used directly.

```bash
python scripts/run_all_turbine_level.py --configs DK-onshore UK-onshore DE-onshore
python scripts/run_all_turbine_level.py --configs DK-onshore --dry-run
```

## 6. Country Data Summary

### Turbine-level countries

| Country | Code | Turbines | Obs. period | Data source | Format |
|---------|------|----------|-------------|-------------|--------|
| Denmark | DK | 5,618 | 2002-2020 | Danish Energy Agency | Long CSV |
| Germany | DE | 10,889 | Multiple years | ANEMOS/ISET 250 MW | Long CSV |
| United Kingdom | UK | 6,618 | 2015-2019 | REF/RenewableUK | Wide CSV |

### Country-level countries

| Country | Code | Bidding zones | Train | Test | Source |
|---------|------|---------------|-------|------|--------|
| Netherlands | NL | -- | 2015-2018 | 2019 | ENTSO-E |
| France | FR | -- | 2015-2018 | 2019 | ENTSO-E |
| Belgium | BE | -- | 2015-2018 | 2019 | ENTSO-E |
| Norway | NO | NO_1 .. NO_5 | 2015-2018 | 2019 | ENTSO-E |
| Sweden | SE | SE_1 .. SE_4 | 2015-2018 | 2019 | ENTSO-E |
| Spain | ES | -- | 2015-2018 | 2019 | ENTSO-E |
| Italy | IT | -- | 2015-2018 | 2019 | ENTSO-E |
| Portugal | PT | -- | 2015-2018 | 2019 | ENTSO-E |
| Ireland | IE | -- | 2015-2018 | 2019 | ENTSO-E |

Norway and Sweden use bidding-zone-aware grid points and zone-aggregated
observation files. The loaders detect this automatically.
