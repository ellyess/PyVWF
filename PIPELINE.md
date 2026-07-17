# PyVWF Pipeline: Script Execution Order

Core workflow: **bias-correction training** followed by **evaluation** of
corrected against uncorrected capacity factors.

Once PyVWF is installed (conda env, or `pip install -e .` from a checkout), the
common one-shot case also has a console entry point:

| Console command | Wraps                                                |
|-----------------|------------------------------------------------------|
| `pyvwf-train`   | `PyVWF.train` + `simulate_cf` (single country/year) |

The scripts below remain the right entry point for batch and multi-country
workflows.

> The experimental grid-interpolation and machine-learning workflows are not
> part of the core pipeline. They live on the `development` branch.

---

## Bias Correction Training

Train bias correction factors from observed vs reanalysis capacity factors.

### Prerequisites: Generate Country-Level Training Data

```bash
python src/vwf/datasets/generate_country_level_training_data.py
```

- Fetches ENTSO-E capacity factor observations via API
- Creates grid sample points and Voronoi regions per country
- Splits into training/test years
- **Generates** `input/country_level_data/pyvwf_config.py` (used by all subsequent scripts)
- **Output:** `input/country_level_data/observations/`, `input/country_level_data/grid_points/`
- Required for country-level workflows (NL, FR, BE, NO, SE, ES, IT, PT, IE)
- Not needed if only running turbine-level (DK, DE, UK)

### Train All Corrections (Recommended)

```bash
# List available configuration sets
python scripts/train_all_bias_corrections.py --list

# Run turbine + country workflows
python scripts/train_all_bias_corrections.py --sets turbine_grid country_grid_2015_2021_2023
```

Master orchestrator supporting all training configurations:

| Config Set | Level | Countries | Training Years |
|---|---|---|---|
| `turbine_fixed_2015_2019` | Turbine | DK, DE, UK | 2015-2019 |
| `turbine_dk_research` | Turbine | DK (onshore+offshore) | 2015-2019 |
| `turbine_grid` | Turbine | DK, DE, UK | 2015-2019 |
| `country_grid_2015_2021_2023` | Country | NL, FR, BE, NO, SE, ES, IT, PT, IE | 2015-2021 |

**Output:** `output/runs/<prefix>/<country-run>/training/correction-factors/`

### Evaluate Corrections

```bash
python scripts/evaluate_all_pyvwf_runs.py --prefix turbine_grid
```

- Calculates MAE, RMSE, R² for corrected vs uncorrected capacity factors
- **Output:** `output/runs/turbine_grid/pyvwf_evaluation_metrics.csv`

---

## Quick Reference: Minimal Pipeline

```bash
# 1. (Country-level only) Generate training data
python src/vwf/datasets/generate_country_level_training_data.py

# 2. Train corrections
python scripts/train_all_bias_corrections.py --sets turbine_grid

# 3. Evaluate
python scripts/evaluate_all_pyvwf_runs.py --prefix turbine_grid
```

## Dependency Graph

```
generate_country_level_training_data.py
          │
          v
train_all_bias_corrections.py
          │
          v
evaluate_all_pyvwf_runs.py
```
