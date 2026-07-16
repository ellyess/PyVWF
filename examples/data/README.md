# Example data (fully synthetic)

This folder holds the small, **synthetic** dataset bundled so that
[`examples/run_minimal.py`](../run_minimal.py) runs end-to-end in under a minute
with **no ERA5 download and no private turbine data**.

Nothing here is real reanalysis or real generation data:

| File | What it is |
|---|---|
| `era5/era5_example.nc` | An ERA5-*shaped* NetCDF (`u100`, `v100`, `u10`, `v10` on a small lat/lon/time grid). The wind field is fabricated, not real ERA5. |
| `turbines_example.csv` | Six synthetic turbine control points (ID, lat, lon, hub height, model, capacity, cluster). |
| `observations_example.csv` | Per-cluster "observed" capacity factors, produced by applying a **known** bias (`scalar`, `offset`) to the simulated capacity factor. |

The observations are generated from the model itself with a known bias, so
`run_minimal.py` can demonstrably recover a correction that reproduces them. It
is an illustration of the workflow, not a validation result. The power curves
come from the bundled open library (`input/power_curves.csv`, real
redistributable curves); see [`input/README.md`](../../input/README.md).

Regenerate everything with:

```bash
python examples/data/generate_example_data.py
```
