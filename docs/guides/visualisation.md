# Visualisation

`pyvwf.viz` turns a run's outputs into diagnostic figures: how well the corrected
simulation reproduces the observed capacity-factor distribution, what the
correction learned spatially, and how error responds to cluster count and
time slice.

`load_results()` reads a harness evaluate run back into a `Results` object, so
the plot functions are self-contained. It pairs each variant with the
observations as the run's `metrics.csv` did, on the same unit-months, and
reduces them to monthly series. The observations are read again through the
region's adapter, so it needs the region's input data. A data-free
reproduction of every figure below is in
[`examples/viz_demo.py`](../../examples/viz_demo.py).

## Distribution and QQ

```python
from pyvwf.viz import load_results, plot_cf_distribution, plot_qq

res = load_results(
    "configs/regions/dk.toml",
    "output/validation/DK/evaluate-2020-<run>",  # the training run is read from its manifest
)
sims = {"uncorrected": res.uncorrected, "corrected": res.corrected[(100, "season")]}

plot_cf_distribution(res.obs, sims).savefig("cf_distribution.png", dpi=150)
plot_qq(res.obs, sims).savefig("cf_qq.png", dpi=150)
```

![CF distribution diagnostic](../img/viz_distribution.png)

The legend annotates each series with its mean and KS distance to observed. The
tail inset zooms into `CF >= 0.7`, so differences in the upper tail stay visible.

## What the correction learned

`plot_correction_factor_map()` colours each cluster's Voronoi cell by its learned
scalar and offset, on a diverging scale centred at the neutral value (scalar 1,
offset 0), so over- and under-correction read at a glance. Pass the *training*
fleet, so the deterministic clustering reproduces the cluster IDs the factors
were fitted on.

```python
from shapely.geometry import box
from pyvwf.viz import plot_correction_factor_map

fig = plot_correction_factor_map(
    res.factors[(100, "season")],         # one (n_clu, time_res) configuration
    res.train_turb_info,                  # the fleet the factors were fitted on
    boundary=box(8.0, 54.5, 13.0, 57.8),  # optional clip: any shapely geometry,
)                                         # GeoDataFrame, or path to a GeoJSON
```

![Correction factor map](../img/viz_factor_map.png)

`plot_factor_joint()` shows the same factors in factor space: scalar against
offset with marginal histograms and guides at the neutral values. Tight
clustering around (1, 0) means the reanalysis needed little correction.

```python
from pyvwf.viz import plot_factor_joint

plot_factor_joint(res.factors[(100, "season")]).savefig("factor_joint.png", dpi=150)
```

![Factor joint distribution](../img/viz_factor_joint.png)

## Per-turbine bias

`plot_sim_vs_obs()` scatters each turbine's mean simulated capacity factor
against its mean observed one, so distance from the diagonal is that turbine's
bias. The panel is annotated with fleet-level MBE and RMSE. It takes the wide
simulated frame an evaluate run writes and the observations its adapter reads.

```python
import pandas as pd
from pyvwf.harness.driver import load_obs_and_fleet
from pyvwf.harness.regions import load_region
from pyvwf.viz import plot_sim_vs_obs

spec = load_region("configs/regions/dk.toml")
obs, fleet = load_obs_and_fleet(spec, 2020)
fig = plot_sim_vs_obs(
    pd.read_csv("output/validation/DK/evaluate-2020-<run>/unc_cf.csv"),
    obs,
    turb_info=fleet,   # optional: colour onshore/offshore
)
```

![Per-turbine sim vs obs](../img/viz_sim_vs_obs.png)

## Choosing `n_clu` and `time_res`

`plot_error_vs_clusters()` takes the `metrics.csv` of an evaluate run whose
training run swept several cluster counts, and plots error against cluster
count, one line per time slice, with the uncorrected error as a reference. A
country-level run with a zonal source writes a `national` and a `per-zone`
scope; pick one.

```python
import pandas as pd
from pyvwf.viz import plot_error_vs_clusters

metrics = pd.read_csv("output/validation/DK/evaluate-2020-<run>/metrics.csv")
plot_error_vs_clusters(metrics).savefig("error_vs_clusters.png", dpi=150)
```

![Error vs clusters](../img/viz_error_vs_clusters.png)

Bear in mind that with one test year per region, the shape of this curve is
more informative than its exact minimum.

## Animated cluster maps in TouchDesigner

`scripts/analysis/export_voronoi_frames.py` exports a cluster sweep for
[TouchDesigner](https://derivative.ca/). The output is one `.npz` file with a
frame for every cluster count. Each frame holds the Voronoi cells of the
cluster centroids, clipped to the region and coloured by the fitted scalar.

These cells reproduce the clusters exactly. Training clusters the units by
k-means in longitude and latitude, so each unit belongs to its nearest
centroid.

TouchDesigner's Python has numpy but not scipy or shapely. The script
therefore does all the geometry, and writes flat float32 triangles.

1. Install the extra:

   ```bash
   pip install -e ".[touchdesigner]"
   ```

2. Train a cluster sweep with the harness. The export reads the
   `train-k<N>/` directories under `<sweep>/<CODE>/`.
3. Run the export:

   ```bash
   PYTHONPATH=src python scripts/analysis/export_voronoi_frames.py \
       --sweep output/validation/<sweep> --region DK \
       --out output/viz/dk_voronoi_frames.npz
   ```

4. Point the TouchDesigner component's file parameter at the `.npz`.

The script also writes a `.json` file beside the `.npz`, with the colour
limits and the cluster counts. The region outline comes from
`input/reference/shapes/country_shapes.geojson`. Some islands are missing
from that outline. A coastline file recovers them, at
`input/reference/terrain/coastlines.geojson` by default. To skip the repair,
pass a `--coastline` path that does not exist.
