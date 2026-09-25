# API reference

Everything below is importable from the `pyvwf` package. The data-acquisition
modules under `pyvwf.datasets` are command-line scripts that depend on the
optional `data` extra (and, for ENTSO-E, an API key); they are not part of the
supported programmatic API and are documented in
{doc}`guides/data-sources` instead.

## The validation harness

The path for new work: train, evaluate and transfer one region from its
config. The commands are in {doc}`guides/training`, and the design in
{doc}`design/harness`.

```{eval-rst}
.. automodule:: pyvwf.harness.driver
   :members: run_train, run_evaluate, run_transfer, resolve_source,
             load_obs_and_fleet, era5_dir, tidy_eval_frame, country_pairs,
             country_skill, error_metrics, score_on_common_rows, SCOPE_KEYS

.. automodule:: pyvwf.harness.regions
   :members: RegionSpec, load_region, load_region_by_code, region_stem,
             season_of_month

.. automodule:: pyvwf.harness.corrections
   :members:

.. automodule:: pyvwf.harness.skill
   :members:

.. automodule:: pyvwf.provenance
   :members:

.. automodule:: pyvwf.harness.bootstrap
   :members:

.. automodule:: pyvwf.harness.export
   :members:

.. automodule:: pyvwf.harness.hindcast
   :members:
```

## Adapters

Observed generation and unit metadata come from adapters, so supporting a new
data source means writing an adapter rather than editing the core pipeline.
See {doc}`guides/adding-an-adapter`.

```{eval-rst}
.. automodule:: pyvwf.sources.base
   :members:

.. automodule:: pyvwf.sources.registry
   :members: register, resolve, get_source, available_sources

.. automodule:: pyvwf.sources.european
   :members:

.. automodule:: pyvwf.sources.aemo
   :members:

.. automodule:: pyvwf.sources.eia_us
   :members:

.. automodule:: pyvwf.sources.ons_br
   :members:

.. automodule:: pyvwf.sources.emi_nz
   :members:

.. automodule:: pyvwf.sources.cen_cl
   :members:

.. automodule:: pyvwf.sources.cammesa_ar
   :members:

.. automodule:: pyvwf.sources.windstats
   :members:

.. automodule:: pyvwf.sources.client_csv
   :members:

.. automodule:: pyvwf.sources.entsoe_files
   :members:

.. automodule:: pyvwf.sources.entsoe_zonal
   :members:

.. automodule:: pyvwf.sources.in_memory
   :members:
```

## Data preparation

Assembles the training and validation sets: observations, turbine metadata,
reanalysis, and power curves.

```{eval-rst}
.. automodule:: pyvwf.data
   :members: train_set, val_set, val_obs_and_fleet, cluster_train_set, prep_country,
             clean_obs_data, interp_nans

.. automodule:: pyvwf.curves
   :members: load_power_curves, add_models
```

## Wind simulation

Hub-height extrapolation, power-curve conversion, and application of the
learned wind-speed correction.

```{eval-rst}
.. automodule:: pyvwf.wind
   :members: interpolate_wind, simulate_wind, correct_wind_speed,
             train_simulate_wind, train_simulate_wind_from_ws, fast_simulate_cf,
             prepare_offset_arrays, aggregate_turbines_to_grid
```

## Bias correction

Fits the linear correction: the scalar is the capacity-weighted ratio of
observed to simulated capacity factor; the offset is fitted numerically so the
corrected simulation matches the observations.

```{eval-rst}
.. automodule:: pyvwf.correction
   :members: calculate_scalar, find_offset, find_offsets_country_level
```

## Clustering

Groups turbines spatially, so a correction can be learned per cluster rather
than once for a whole country.

```{eval-rst}
.. automodule:: pyvwf.clustering
   :members: cluster_turbines, get_country_shape, load_region_shapes

.. automodule:: pyvwf.sampling
   :members: cluster_with_geometries, create_sampling_points, add_turbine_metadata
```

## Metrics

Error metrics between simulated and observed capacity factors. All aggregations
are capacity-weighted.

```{eval-rst}
.. automodule:: pyvwf.metrics
   :members: calculate_error, overall_error, prepare_monthly_data,
             weighted_average_vectorized
```

## Visualisation

Diagnostics for a run: how well the corrected simulation reproduces the observed
distribution, what the correction learned, and how error responds to the two
hyperparameters.

```{eval-rst}
.. automodule:: pyvwf.viz.distribution
   :members: Results, load_results, plot_cf_distribution, plot_qq

.. automodule:: pyvwf.viz.factors
   :members: plot_correction_factor_map, plot_factor_joint

.. automodule:: pyvwf.viz.evaluation
   :members: plot_error_vs_clusters, plot_sim_vs_obs
```

## Datasets and loaders

```{eval-rst}
.. automodule:: pyvwf.era5
   :members: prep_era5, log_roughness_from_shear, Z0_BOUNDS

.. automodule:: pyvwf.loaders.turbine_loaders
   :members: load_turbine_metadata, load_turbine_observations

```

## Configuration and utilities

```{eval-rst}
.. automodule:: pyvwf.config
   :members:

.. automodule:: pyvwf.time_utils
   :members:

.. automodule:: pyvwf.utils
   :members:
```
