"""Generate the small, fully synthetic dataset bundled for ``run_minimal.py``.

This writes an ERA5-shaped NetCDF (``u100``/``v100``/``u10``/``v10`` on a small
lat/lon/time grid) plus a handful of synthetic turbine control points and their
"observed" capacity factors. Nothing here is real reanalysis or real generation
data: the wind field is fabricated, and the observations are produced by
applying a KNOWN bias correction to the simulated capacity factor so that
``run_minimal.py`` can demonstrably recover it.

Re-run to regenerate the committed files:

    python examples/data/generate_example_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ERA5_DIR = HERE / "era5"

# Known "true" bias baked into the synthetic observations (recovered by the
# example): corrected wind = TRUE_SCALAR * w + TRUE_OFFSET (offset in m/s).
TRUE_SCALAR = 1.12
TRUE_OFFSET = 0.4

# Small grid over a Denmark-ish box, and a short hourly period (kept tiny).
LATS = np.array([55.0, 55.4, 55.8, 56.2], dtype="float32")
LONS = np.array([8.0, 8.5, 9.0, 9.5, 10.0], dtype="float32")
TIMES = pd.date_range("2020-01-01", periods=24 * 90, freq="h")  # 90 days hourly

# Wind-shear ratio wnd10/wnd100 ~ 0.72 implies a plausible roughness z0 ~ 0.03 m.
SHEAR_10_OVER_100 = 0.72


def _make_era5() -> xr.Dataset:
    rng = np.random.default_rng(42)
    nt, ny, nx = len(TIMES), len(LATS), len(LONS)

    # A smooth diurnal + synoptic 100 m wind magnitude with mild spatial drift.
    hours = np.arange(nt)
    diurnal = 1.0 * np.sin(2 * np.pi * hours / 24.0)
    synoptic = 2.5 * np.sin(2 * np.pi * hours / (24 * 6.0))
    base = 8.5 + diurnal + synoptic
    spatial = np.linspace(-0.6, 0.6, ny)[:, None] + np.linspace(-0.4, 0.4, nx)[None, :]
    mag100 = base[:, None, None] + spatial[None, :, :] + rng.normal(0, 0.8, (nt, ny, nx))
    mag100 = np.clip(mag100, 0.2, None).astype("float32")

    # Split into u/v with a slowly rotating direction.
    direction = np.deg2rad(240 + 20 * np.sin(2 * np.pi * hours / (24 * 10.0)))
    u100 = (mag100 * np.cos(direction)[:, None, None]).astype("float32")
    v100 = (mag100 * np.sin(direction)[:, None, None]).astype("float32")
    u10 = (u100 * SHEAR_10_OVER_100).astype("float32")
    v10 = (v100 * SHEAR_10_OVER_100).astype("float32")

    return xr.Dataset(
        {
            "u100": (("time", "lat", "lon"), u100),
            "v100": (("time", "lat", "lon"), v100),
            "u10": (("time", "lat", "lon"), u10),
            "v10": (("time", "lat", "lon"), v10),
        },
        coords={"time": TIMES, "lat": LATS, "lon": LONS},
    )


def _make_turbines() -> pd.DataFrame:
    # Six control points inside the grid, split into two clusters.
    return pd.DataFrame(
        {
            "ID": [f"t{i}" for i in range(1, 7)],
            "lat": [55.2, 55.3, 55.5, 55.9, 56.0, 56.1],
            "lon": [8.3, 8.7, 9.1, 9.2, 9.6, 8.9],
            "height": [100.0] * 6,
            "model": ["Synthetic.Onshore2000"] * 6,
            "capacity": [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
            "cluster": [0, 0, 0, 1, 1, 1],
        }
    )


def _make_observations(turbines: pd.DataFrame) -> pd.DataFrame:
    """Synthesize per-cluster 'observed' capacity factors by applying the known
    bias (TRUE_SCALAR, TRUE_OFFSET) to the simulated capacity factor. This is
    what ``run_minimal.py`` then recovers blind."""
    import pandas as pd  # local, so the file top stays import-light
    from vwf.config import PyVWFPaths
    from vwf.datasets.era5 import prep_era5
    from vwf.wind import train_simulate_wind

    PyVWFPaths.ERA5_DATA = ERA5_DIR
    ds = prep_era5("example", train=False, calc_z0=True, bbox=None)
    power_curves = pd.read_csv("input/power_curves.csv")

    rows = []
    for cluster, cl in turbines.groupby("cluster"):
        obs_cf = train_simulate_wind(ds, cl, power_curves, TRUE_SCALAR, TRUE_OFFSET)
        rows.append({"cluster": int(cluster), "obs_cf": float(obs_cf), "year": 2020})
    return pd.DataFrame(rows)


def main() -> None:
    ERA5_DIR.mkdir(parents=True, exist_ok=True)

    era5 = _make_era5()
    enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4} for v in era5.data_vars}
    era5.to_netcdf(ERA5_DIR / "era5_example.nc", encoding=enc)

    turbines = _make_turbines()
    turbines.to_csv(HERE / "turbines_example.csv", index=False)

    observations = _make_observations(turbines)
    observations.to_csv(HERE / "observations_example.csv", index=False)

    print(
        f"Wrote era5_example.nc ({dict(era5.sizes)}), "
        f"turbines_example.csv ({len(turbines)} turbines), "
        f"observations_example.csv ({len(observations)} clusters). "
        f"True bias baked in: scalar={TRUE_SCALAR}, offset={TRUE_OFFSET} m/s."
    )


if __name__ == "__main__":
    main()
