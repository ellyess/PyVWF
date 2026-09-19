# Study drivers

One directory per study, named by the stem of its findings document in
`docs/findings/`. Each holds the drivers that produce that study's numbers.
Reusable tools stay in `scripts/analysis/`; logic more than one study needs
belongs in `src/vwf/`.

[`docs/guides/adding-a-study.md`](../../docs/guides/adding-a-study.md) says how
to add a study: where each piece goes, the order of commits, and the rules a
driver follows. This file keeps what is specific to this folder: the deferred
move and the path map.

## Deferred

`scripts/pinn/` moves here as one unit, to `scripts/studies/physics-informed/`,
once the turbine-only study has run on its registered paths (issue #12).
`scripts/analysis/ml_transfer_retest.py` and `ml_transfer_expanded.py` move
with it, because the `scripts/pinn/` drivers import the first by that path.

## Path map

Every driver that has moved here, with its old path and the last commit that
touched it there. A command in a dated findings document that names an old path
refers to this file; `git show <commit>:<old path>` recovers it as it stood,
and the document's header names the commit its numbers came from.

| old path | new path | last commit at old path |
|---|---|---|
| `scripts/analysis/curve_library_assign.py` | `scripts/studies/method-curve-library/curve_library_assign.py` | `34d9c3a` |
| `scripts/analysis/curve_library_match.py` | `scripts/studies/method-curve-library/curve_library_match.py` | `b7ed5fd` |
| `scripts/analysis/curve_library_study.py` | `scripts/studies/method-curve-library/curve_library_study.py` | `6dd7af2` |
| `scripts/analysis/curve_library_tables.py` | `scripts/studies/method-curve-library/curve_library_tables.py` | `9aaa9d5` |
| `scripts/analysis/roughness_treatment_study.py` | `scripts/studies/method-roughness-treatment/roughness_treatment_study.py` | `1ff1d54` |
| `scripts/analysis/missing_value_audit.py` | `scripts/studies/scorecard/missing_value_audit.py` | `bbaf5b3` |
| `scripts/analysis/off_curve_sensitivity.py` | `scripts/studies/scorecard/off_curve_sensitivity.py` | `56a78d2` |
| `scripts/analysis/training_objective_check.py` | `scripts/studies/scorecard/training_objective_check.py` | `b7826d3` |
| `scripts/analysis/unit_concentration.py` | `scripts/studies/scorecard/unit_concentration.py` | `4b6d143` |
| `scripts/analysis/cluster_selection_study.py` | `scripts/studies/method-cluster-selection/cluster_selection_study.py` | `cc9fc14` |
| `scripts/analysis/cluster_selection_gaps.py` | `scripts/studies/method-cluster-selection/cluster_selection_gaps.py` | `1ae99a3` |
| `scripts/analysis/cluster_sweep_cost.py` | `scripts/studies/method-cluster-selection/cluster_sweep_cost.py` | `4af1ba4` |
| `scripts/analysis/national_single_cluster_study.py` | `scripts/studies/method-national-single-cluster/national_single_cluster_study.py` | `8144934` |
| `scripts/analysis/loco_interpolation.py` | `scripts/studies/method-loco-interpolation/loco_interpolation.py` | `4a1fd79` |
| `scripts/analysis/correction_identifiability.py` | `scripts/studies/method-correction-identifiability/correction_identifiability.py` | `074f648` |
| `scripts/analysis/loco_reference_wind.py` | `scripts/studies/method-correction-identifiability/loco_reference_wind.py` | `fdb7101` |
| `scripts/analysis/pivot_probe.py` | `scripts/studies/method-correction-identifiability/pivot_probe.py` | `1051a6c` |
| `scripts/analysis/pool_as_training_set.py` | `scripts/studies/method-why-corrections-do-not-transfer/pool_as_training_set.py` | `6b0afc3` |
| `scripts/analysis/regime_coverage.py` | `scripts/studies/method-why-corrections-do-not-transfer/regime_coverage.py` | `8a7bcb8` |
| `scripts/analysis/era5_overlap_check.py` | `scripts/studies/method-eu-rerun/era5_overlap_check.py` | `3898bdc` |
| `scripts/analysis/hourly_resolution_test.py` | `scripts/studies/method-hourly-resolution/hourly_resolution_test.py` | `d2017f3` |
| `scripts/analysis/min_cluster_size_tradeoff.py` | `scripts/studies/method-scalar-bounds/min_cluster_size_tradeoff.py` | `6effe0e` |
| `scripts/analysis/offshore_pool_study.py` | `scripts/studies/method-offshore-pool/offshore_pool_study.py` | `bd09937` |
| `scripts/analysis/domain_split_study.py` | `scripts/studies/method-domain-split/domain_split_study.py` | `f305b28` |
| `scripts/analysis/unmasked_surface_bands.py` | `scripts/studies/method-distance-mask/unmasked_surface_bands.py` | `171f530` |
| `scripts/analysis/chapter_capacity_weights.py` | `scripts/studies/method-country-level/chapter_capacity_weights.py` | `bff4467` |
| `scripts/analysis/refit_control_points.py` | `scripts/studies/manuscript-chapters-45/refit_control_points.py` | `3930f26` |
| `scripts/region_tools/export_au_grid_netcdf.py` | `scripts/studies/method-generalisation/export_au_grid_netcdf.py` | `6effe0e` |
| `examples/notebooks/au_nem_validation.ipynb` | `scripts/studies/method-generalisation/au_nem_validation.ipynb` | `6effe0e` |
