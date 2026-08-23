"""Multi-scale terrain descriptors for the physics-informed correction.

ERA5 resolves orography at roughly 30 km. Everything below that -- the ridge a
turbine stands on, the pass it sits in, the escarpment that accelerates the flow
over it -- is invisible to the reanalysis, and D1/D3 found that the elevation
RANGE inside an ERA5 cell tracks the fitted correction consistently across
regions where the point elevation, and the 90 m derivatives the earlier ML
experiment used, do not.

The descriptors here are therefore computed at four scales, from micro-siting up
to the orographic blocking scale, and are deliberately all functions of terrain
alone: no longitude, no latitude, nothing that could encode region identity and
so score well in-sample while failing to transfer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter

# Window widths in kilometres, and what each is meant to capture:
#   1  km  micro-siting (the scale the published ML features used)
#   5  km  an individual hill or ridge
#   28 km  the ERA5 grid cell -- the terrain the reanalysis cannot see
#   84 km  the orographic drag / blocking scale
SCALES_KM: dict[str, float] = {"1km": 1.0, "5km": 5.0, "28km": 28.0, "84km": 84.0}

FEATURES: tuple[str, ...] = (
    ("z_site", "land_frac_28km")
    + tuple(f"{p}_{s}" for s in SCALES_KM for p in ("tpi", "std", "relief"))
)


def _odd(n: int) -> int:
    n = max(int(n), 3)
    return n if n % 2 else n + 1


def terrain_descriptors(
    lon: np.ndarray,
    lat: np.ndarray,
    etopo_path: str | Path,
    pad_deg: float = 1.0,
) -> pd.DataFrame:
    """Terrain position, spread and relief at four scales, at each point.

    Args:
        lon: Longitudes of the points to describe, in [-180, 180].
        lat: Latitudes of the points to describe.
        etopo_path: Path to the global ETOPO NetCDF (30 arc-second ``z``).
        pad_deg: Degrees of margin read around the point bounding box, so that
            the largest filter window is not truncated at the edge.

    Returns:
        DataFrame with one row per input point and the columns in ``FEATURES``:
        ``z_site`` (elevation), ``land_frac_28km`` (fraction of the ERA5 cell
        above sea level), and per scale ``tpi_*`` (height above the local mean),
        ``std_*`` (elevation standard deviation) and ``relief_*`` (elevation
        range).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    ds = xr.open_dataset(etopo_path)
    try:
        # A pad of one degree is not enough for the 84 km window near the edge;
        # widen it to at least the largest half-window.
        pad = max(pad_deg, (max(SCALES_KM.values()) / 111.32) * 0.75)
        z = ds.z.sel(
            lon=slice(lon.min() - pad, lon.max() + pad),
            lat=slice(lat.min() - pad, lat.max() + pad),
        ).load()
    finally:
        ds.close()

    elev = z.values.astype("float32")
    zlat, zlon = z.lat.values, z.lon.values
    km_per_px = float(abs(zlat[1] - zlat[0])) * 111.32

    ii = np.abs(zlat[None, :] - lat[:, None]).argmin(axis=1)
    jj = np.abs(zlon[None, :] - lon[:, None]).argmin(axis=1)
    z_site = elev[ii, jj].astype("float64")

    out: dict[str, np.ndarray] = {"z_site": z_site}
    n28 = _odd(round(28.0 / km_per_px))
    out["land_frac_28km"] = uniform_filter(
        (elev > 0).astype("float32"), size=n28, mode="nearest"
    )[ii, jj].astype("float64")

    for name, km in SCALES_KM.items():
        n = _odd(round(km / km_per_px))
        mean = uniform_filter(elev, size=n, mode="nearest")
        msq = uniform_filter(elev.astype("float64") ** 2, size=n, mode="nearest")
        std = np.sqrt(np.clip(msq - mean.astype("float64") ** 2, 0.0, None))
        relief = (maximum_filter(elev, size=n, mode="nearest")
                  - minimum_filter(elev, size=n, mode="nearest"))
        out[f"tpi_{name}"] = z_site - mean[ii, jj].astype("float64")
        out[f"std_{name}"] = std[ii, jj]
        out[f"relief_{name}"] = relief[ii, jj].astype("float64")
        del mean, msq, std, relief

    return pd.DataFrame(out)
