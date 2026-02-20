# Scripts Directory

Analysis and processing scripts organised by pipeline stage. See [PIPELINE.md](../PIPELINE.md) for execution order and dependencies.

## Top-Level Scripts

| Script | Purpose |
|--------|---------|
| `evaluate_all_pyvwf_runs.py` | Calculate MAE, RMSE, R² for corrected vs uncorrected capacity factors |
| `analyse_grid_and_ml_results.py` | Combined analysis of grid interpolation and ML results |
| `plot_denmark_from_vwf_output.py` | Denmark-specific diagnostic plotting |
| `regenerate_grid_points_with_gwpt.py` | Regenerate grid points using Global Wind Power Tracker |

## Grid Interpolation (`pyvwf_to_grid/`)

Scripts for interpolating point-based corrections onto a continuous European ERA5 grid.

| Script | Purpose |
|--------|---------|
| `create_unified_correction_dataframe.py` | Combine all Stage 1 corrections into one dataset |
| `generate_turbine_cluster_geometries.py` | Create Voronoi cluster geometries clipped to country boundaries |
| `compare_unified_corrections_to_grid.py` | Interpolate corrections onto 0.25 deg ERA5 grid (NN, IDW, RBF, Kriging) |
| `visualize_corrections_and_interpolations.py` | Generate diagnostic maps of corrections and interpolations |
| `evaluate_grid_corrections.py` | Validate gridded corrections against observations |
| `generate_best_correction_grids.py` | Generate production-ready correction NetCDF files for Atlite/PyPSA-Eur |
| `generate_ch4_grid_plots.py` | Publication-quality figures for thesis Chapter 4 (grid interpolation) |
| `test_kriging_improvements.py` | Compare Kriging configurations via spatial cross-validation |

## Machine Learning (`pyvwf_ml/`)

Scripts for training ML models to predict correction factors from terrain and environmental features.

| Script | Purpose |
|--------|---------|
| `download_terrain_data.py` | Download and process terrain data (elevation, slope, aspect, roughness) |
| `download_corine_data.py` | Download CORINE Land Cover 2018 data (requires Copernicus account) |
| `quick_terrain_setup.py` | Quick terrain setup utility |
| `enhance_terrain_features.py` | Add derived features (distance to coast, sub-grid variance, terrain complexity) |
| `prepare_turbine_fleet_features.py` | Link turbine metadata to correction clusters |
| `train_unified_ml_corrections.py` | Train ML correction models (RF, GBM, XGBoost, LightGBM, Ridge, etc.) |
| `generate_ch2_ml_plots.py` | Publication-quality figures for thesis Chapter 2 (ML corrections) |
