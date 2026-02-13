# Scripts Directory

This directory contains utility scripts, tests, and experimental code.

## Country-Level Training

**`generate_country_level_training_data.py`**
Generate grid points and cluster geometries for country-level corrections (NL, FR, BE, NO).

## Testing & Debugging

**Test Scripts:**
- `test_fr_be_normalization.py` - Test France/Belgium normalization
- `test_nl_normalization.py` - Test Netherlands normalization
- `test_offset_optimization.py` - Test offset optimization
- `test_turbine_level_workflow.py` - Test turbine-level workflow

**Debug Scripts:**
- `debug_country_training.py` - Debug country-level training issues
- `evaluate_country_corrections.py` - Evaluate country correction performance

## Utilities

- `fix_capacity_data.py` - Fix capacity data issues
- `visualize_normalization_results.py` - Visualize normalization results
- `thesis_plots.py` - Generate plots for thesis

## Production Scripts

For production scripts used in the main workflow, see the root directory:
- `compare_unified_corrections_to_grid.py`
- `create_unified_correction_dataframe.py`
- `generate_turbine_cluster_geometries.py`
- `visualize_corrections_and_interpolations.py`
