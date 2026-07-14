# API reference

Everything below is importable from the `vwf` package. The data-acquisition
modules under `vwf.datasets` are command-line scripts that depend on the
optional `data` extra (and, for ENTSO-E, an API key); they are not part of the
supported programmatic API and are documented in the
{doc}`ENTSOE_API_GUIDE` instead.

## The model

The entry point. `PyVWF` owns the run directory, trains the correction factors,
and simulates capacity factors for a test year.

```{eval-rst}
.. automodule:: vwf.vwf
   :members:
   :member-order: bysource
```

## Observation sources

Observed generation and site metadata come from pluggable adapters, so
supporting a new region means writing a source rather than editing the core
pipeline. See {doc}`ADDING_AN_OBSERVATION_SOURCE`.

```{eval-rst}
.. automodule:: vwf.sources.base
   :members:

.. automodule:: vwf.sources.registry
   :members: register, resolve, get_source, available_sources

.. automodule:: vwf.sources.european
   :members:

.. automodule:: vwf.sources.in_memory
   :members:
```

## Data preparation

Assembles the training and validation sets: observations, turbine metadata,
reanalysis, and power curves.

```{eval-rst}
.. automodule:: vwf.data
   :members: train_set, val_set, cluster_train_set, prep_country, clean_obs_data,
             add_models, interp_nans, load_power_curves, sim_turbines_to_country_cf
```

## Wind simulation

Hub-height extrapolation, power-curve conversion, and application of the
learned wind-speed correction.

```{eval-rst}
.. automodule:: vwf.wind
   :members: interpolate_wind, simulate_wind, correct_wind_speed,
             train_simulate_wind, train_simulate_wind_from_ws, fast_simulate_cf,
             prepare_offset_arrays, aggregate_turbines_to_grid, simulate_country_cf
```

## Bias correction

Fits the linear correction: the scalar is the capacity-weighted ratio of
observed to simulated capacity factor; the offset is fitted numerically so the
corrected simulation matches the observations.

```{eval-rst}
.. automodule:: vwf.correction
   :members: calculate_scalar, find_offset, find_offsets_country_level
```

## Clustering

Groups turbines spatially, so a correction can be learned per cluster rather
than once for a whole country.

```{eval-rst}
.. automodule:: vwf.clustering
   :members: cluster_turbines, cluster_with_geometries, create_sampling_points,
             add_turbine_metadata, get_country_shape, load_region_shapes
```

## Metrics

Error metrics between simulated and observed capacity factors. All aggregations
are capacity-weighted.

```{eval-rst}
.. automodule:: vwf.metrics
   :members: calculate_error, overall_error, prepare_monthly_data,
             weighted_average_vectorized
```

## Visualisation

Diagnostics for a run: how well the corrected simulation reproduces the observed
distribution, what the correction learned, and how error responds to the two
hyperparameters.

```{eval-rst}
.. automodule:: vwf.viz.distribution
   :members: Results, load_results, plot_cf_distribution, plot_qq

.. automodule:: vwf.viz.factors
   :members: plot_correction_factor_map, plot_factor_joint

.. automodule:: vwf.viz.evaluation
   :members: plot_error_vs_clusters, plot_sim_vs_obs
```

## Datasets and loaders

```{eval-rst}
.. automodule:: vwf.datasets.era5
   :members: prep_era5

.. automodule:: vwf.loaders.turbine_loaders
   :members: load_turbine_metadata, load_turbine_observations

.. automodule:: vwf.loaders.country_level_loaders
   :members: load_year_specific_grid_points, country_gen_to_cf
```

## Configuration and utilities

```{eval-rst}
.. automodule:: vwf.config
   :members:

.. automodule:: vwf.time_utils
   :members:

.. automodule:: vwf.utils
   :members:
```
