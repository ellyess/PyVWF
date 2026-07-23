# PyVWF Bias Correction Training Guide

## 1. Overview

The `train_all_bias_corrections.py` script is a unified training orchestrator for PyVWF bias
correction factors. It supports both turbine-level corrections (using individual turbine
observations) and country-level corrections (using ENTSO-E aggregate data), with flexible
per-country configuration and onshore/offshore variant support.

### Pre-Configured Training Sets

| Set Name | Obs Level | Countries | Train Period | Test Year | Key Variable |
|----------|-----------|-----------|--------------|-----------|--------------|
| `turbine_simple_2015_2019` | turbine | DK, DE, UK | 2015-2019 | 2020 | Clusters: 5, 10, 15 |
| `turbine_temporal_2015_2019` | turbine | DK, DE, UK | 2015-2019 | 2020 | Time res: fixed, season, bimonth |
| `turbine_extended_2015_2021` | turbine | DK | 2015-2021 | 2023 | Extended training period |
| `country_simple_2015_2019` | country | NL, FR, BE, NO, SE, ES, IT, PT, IE | 2015-2019 | 2023 | Per-country clusters |
| `country_extended_2015_2021` | country | NL, FR, BE, NO, SE, ES, IT, PT, IE | 2015-2021 | 2023 | Extended training period |

---

## 2. Quick Start

### List available training sets

```bash
python scripts/analysis/train_all_bias_corrections.py --list
```

### Run all training sets

```bash
python scripts/analysis/train_all_bias_corrections.py
```

### Run specific sets

```bash
python scripts/analysis/train_all_bias_corrections.py --sets turbine_simple_2015_2019 country_simple_2015_2019
```

### Custom output directory

```bash
python scripts/analysis/train_all_bias_corrections.py --outdir custom_output/runs
```

### Log output for later review

```bash
python scripts/analysis/train_all_bias_corrections.py 2>&1 | tee training_log_$(date +%Y%m%d_%H%M).txt
```

---

## 3. Training Set Configuration

Training sets are defined in the `get_training_sets()` function inside
`train_all_bias_corrections.py`. Each set is a dictionary keyed by a unique prefix name.

### Minimal configuration

```python
def get_training_sets():
    configs = {}

    configs['my_custom_set'] = {
        'name': 'My Custom Training Set',
        'obs_level': 'turbine',        # 'turbine' or 'country'
        'train_years': (2015, 2019),   # (start, end) inclusive
        'test_year': 2020,             # year to simulate after training
        'countries': ['DK', 'DE'],
        'cluster_mode': 'all',         # 'all', 'onshore', or 'offshore'
        'cluster_list': [5, 10],       # spatial cluster counts to test
        'time_res_list': ['fixed', 'season'],  # temporal resolutions
        'calc_z0': True,               # calculate roughness length
    }

    return configs
```

Then run it:

```bash
python scripts/analysis/train_all_bias_corrections.py --sets my_custom_set
```

All parameters shown above are required in the base configuration. The script iterates
over every combination of country, cluster count, and temporal resolution within a set.

---

## 4. Per-Country Configuration

Add a `per_country_config` dictionary to override any base parameter for specific countries.
Countries not listed in the overrides use the base defaults.

### Override priority

1. Base configuration -- defines defaults for all countries.
2. `per_country_config` entries -- override specific settings per country.

### Example: different training periods and cluster counts

```python
configs['turbine_mixed'] = {
    'name': 'Mixed Per-Country Config',
    'obs_level': 'turbine',

    # Base defaults
    'train_years': (2015, 2019),
    'test_year': 2020,
    'countries': ['DK', 'DE', 'UK'],
    'cluster_mode': 'all',
    'cluster_list': [10],
    'time_res_list': ['fixed'],
    'calc_z0': True,

    # Per-country overrides
    'per_country_config': {
        'DK': {
            'train_years': (2015, 2021),   # DK has data to 2023
            'test_year': 2023,
            'cluster_list': [10, 15],
            'time_res_list': ['fixed', 'season'],
        },
        'DE': {
            'cluster_list': [10, 15, 20],  # larger country, more clusters
        },
        # UK: not listed, uses all base defaults
    }
}
```

Result:

- DK trains on 2015-2021, tests on 2023, clusters [10, 15], time res [fixed, season].
- DE trains on 2015-2019, tests on 2020, clusters [10, 15, 20], time res [fixed].
- UK trains on 2015-2019, tests on 2020, cluster [10], time res [fixed].

The `get_country_config(country, base_config)` function in the training script handles
merging base and per-country settings at runtime.

---

## 5. Onshore/Offshore Variants

To train the same country separately for onshore and offshore turbines, append `_onshore`
or `_offshore` to the country code in the `countries` list.

### Variant syntax

| Entry | Resulting `cluster_mode` |
|-------|--------------------------|
| `'DK'` | Uses base config `cluster_mode` |
| `'DK_onshore'` | Forced to `'onshore'` |
| `'DK_offshore'` | Forced to `'offshore'` |

The script automatically extracts the base country code (`DK`), loads its turbine data,
and sets the cluster mode from the suffix. Each variant produces a separate output
directory.

### Example: onshore vs offshore comparison

```python
configs['dk_onshore_vs_offshore'] = {
    'name': 'Denmark Onshore vs Offshore',
    'obs_level': 'turbine',
    'train_years': (2015, 2019),
    'test_year': 2020,
    'countries': ['DK_onshore', 'DK_offshore'],
    'cluster_list': [10],
    'time_res_list': ['fixed'],
    'calc_z0': True,

    # Optionally customise each variant
    'per_country_config': {
        'DK_onshore': {
            'cluster_list': [8, 10, 12],
            'time_res_list': ['fixed', 'season'],
        },
        'DK_offshore': {
            'cluster_list': [5, 7],
        },
    }
}
```

Important: the `_onshore`/`_offshore` suffix automatically sets the cluster mode.
Manually setting `cluster_mode` in `per_country_config` for a variant is redundant
(but harmless). Use lowercase with an underscore separator -- `DK_onshore` not
`DK-onshore` or `DK_Onshore`.

---

## 6. Output Structure

### Directory layout

```
out/runs/<set_prefix>/output/run/<run_name>/
    training/
        correction-factors/
            cluster_<N>_<time_res>_bias_factors.csv
        simulated-turbines/
            cluster_<N>_<time_res>_sim.csv
    results/
        capacity-factor/
            <year>_cluster_<N>_<time_res>_cf.csv
        wind-speed/
            <year>_cluster_<N>_<time_res>_ws.csv
    plots/
```

### Run name format

The `<run_name>` directory encodes the country, cluster mode, observation level, and
roughness setting:

```
<COUNTRY>-<CLUSTER_MODE>-obs_<OBS_LEVEL>-corrected-calc_z0
```

Examples:

```
DK-all-obs_turbine-corrected-calc_z0
UK-offshore-obs_turbine-corrected-calc_z0
FR-all-obs_country-corrected-calc_z0
```

### Concrete example

```
out/runs/turbine_simple_2015_2019/
    output/run/
        DK-all-obs_turbine-corrected-calc_z0/
            training/correction-factors/
                cluster_5_fixed_bias_factors.csv
                cluster_10_fixed_bias_factors.csv
                cluster_15_fixed_bias_factors.csv
            results/capacity-factor/
                2020_cluster_5_fixed_cf.csv
                2020_cluster_10_fixed_cf.csv
                2020_cluster_15_fixed_cf.csv
        DE-all-obs_turbine-corrected-calc_z0/
        UK-all-obs_turbine-corrected-calc_z0/
```

### Inspecting results

```bash
# List all generated correction factor files
find out/runs -name "*bias_factors.csv"
```

---

## 7. Available Parameters

All parameters below can be set at the base level and overridden per-country via
`per_country_config`.

| Parameter | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| `name` | str | Human-readable set description | `'My Training Set'` |
| `obs_level` | str | Observation data level | `'turbine'`, `'country'` |
| `train_years` | tuple | Training period (start, end) inclusive | `(2015, 2019)`, `(2015, 2021)` |
| `test_year` | int | Year to simulate after training | `2020`, `2023` |
| `countries` | list | Country codes (with optional variant suffix) | `['DK', 'DE_onshore']` |
| `cluster_mode` | str | How to cluster turbines/grid points | `'all'`, `'onshore'`, `'offshore'` |
| `cluster_list` | list | Number of spatial clusters to test | `[5, 10, 15]` |
| `time_res_list` | list | Temporal resolutions for bias factors | `['fixed', 'season', 'bimonth', 'month']` |
| `calc_z0` | bool | Calculate roughness length from land cover | `True`, `False` |
| `per_country_config` | dict | Per-country overrides (optional, base-level only) | See section 4 |

### Temporal resolution options

| Value | Description |
|-------|-------------|
| `'fixed'` | Single correction factor for all times |
| `'season'` | Separate factors per meteorological season |
| `'bimonth'` | Separate factors per two-month period |
| `'month'` | Separate factors per calendar month |

### Cluster mode options

| Value | Description |
|-------|-------------|
| `'all'` | Cluster all turbines/grid points together |
| `'onshore'` | Cluster only onshore turbines/grid points |
| `'offshore'` | Cluster only offshore turbines/grid points |

---

## Troubleshooting

**"No country configs available"** -- `pyvwf_config.py` not found in
`input/observations/country/`. Run `python scripts/generate_country_level_training_data.py`
first.

**"Training failed: FileNotFoundError"** -- Missing input data. Check
`input/observations/turbine/<country>/` for turbine-level runs, or
`input/observations/country/grid_points/` and `observations/` for country-level runs.
Verify ERA5 data exists in `input/era5/EU/`.

**Per-country config not being used** -- Ensure the country code in `per_country_config`
exactly matches an entry in the `countries` list (e.g. `'DK_onshore'` not `'DK'` if the
countries list uses variants). Also check parameter spelling -- `cluster_list` not
`clusters`.
