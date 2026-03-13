# Scripts Directory

Analysis and processing scripts organised by pipeline stage. See [PIPELINE.md](../PIPELINE.md) for execution order and dependencies.

## Top-Level Scripts

| Script | Purpose |
|--------|---------|
| `evaluate_all_pyvwf_runs.py` | Calculate MAE, RMSE, R² for corrected vs uncorrected capacity factors |
| `generate_ch3_plots.py` | Publication-quality figures for thesis Chapter 3 (PyVWF framework) |
| `regenerate_grid_points_with_gwpt.py` | Regenerate grid points using Global Wind Power Tracker |

## Grid Interpolation (`pyvwf_to_grid/`)

Scripts for interpolating point-based corrections onto a continuous European ERA5 grid.

| Script | Purpose |
|--------|---------|
| `create_unified_correction_dataframe.py` | Combine all Stage 1 corrections into one dataset |
| `generate_turbine_cluster_geometries.py` | Create Voronoi cluster geometries clipped to country boundaries |
| `compare_unified_corrections_to_grid.py` | Interpolate corrections onto 0.25 deg ERA5 grid (NN, IDW, RBF, Kriging) |
| `evaluate_grid_corrections.py` | Validate gridded corrections against observations |
| `generate_best_correction_grids.py` | Generate production-ready correction NetCDF files for Atlite/PyPSA-Eur |
| `generate_ch4_grid_plots.py` | Publication-quality figures for thesis Chapter 4 (grid interpolation) |

## Machine Learning (`pyvwf_ml/`)

Scripts for training ML models to predict correction factors from terrain and environmental features.

| Script | Purpose |
|--------|---------|
| `download_terrain_data.py` | Download and process terrain data (elevation, slope, aspect, roughness) |
| `download_corine_data.py` | Download CORINE Land Cover 2018 data (requires Copernicus account) |
| `enhance_terrain_features.py` | Add derived features (distance to coast, sub-grid variance, terrain complexity) |
| `prepare_turbine_fleet_features.py` | Link turbine metadata to correction clusters |
| `train_unified_ml_corrections.py` | Train ML correction models (RF, GBM, XGBoost, LightGBM, Ridge, etc.) |
| `run_turbine_model_comparisons.py` | Run turbine-level model comparison experiments (35-feat default/tuned, 7-feat tuned) |
| `generate_ch5_ml_plots.py` | Publication-quality figures for thesis Chapter 5 (ML corrections) |
